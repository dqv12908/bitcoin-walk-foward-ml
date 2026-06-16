from __future__ import annotations

import dataclasses
import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class DataConfig:
    symbol: str = "BTCUSDT"
    market: str = "futures"
    base_interval: str = "5m"
    raw_dir: Path = Path("data/raw")
    model_dir: Path = Path("models")
    report_dir: Path = Path("reports")
    paper_db: Path = Path("data/paper_predictions.sqlite3")
    experiment_db: Path = Path("data/experiments.sqlite3")
    include_spot: bool = True
    include_open_interest: bool = True
    include_funding: bool = True
    include_hyperliquid: bool = True
    include_coinbase: bool = True
    include_bybit: bool = True
    include_okx: bool = True
    include_agg_trades: bool = True
    include_live_flow: bool = True
    include_order_book: bool = True
    include_derivative_ratios: bool = True
    include_basis_klines: bool = True
    include_context_timeframes: bool = True
    context_timeframes: list[str] = field(default_factory=lambda: ["15m", "1h", "4h"])


@dataclass(slots=True)
class PredictionConfig:
    timeframes: list[str] = field(default_factory=lambda: ["1m", "3m", "5m", "15m", "30m", "1h", "4h"])
    horizons: list[int] = field(default_factory=lambda: [1, 2, 3])
    label_mode: str = "close_to_close"
    min_move_bps: float = 0.0
    feature_groups: list[str] = field(default_factory=lambda: ["all"])
    high_confidence_threshold: float = 0.60
    edge_accuracy_threshold: float = 0.60
    high_confidence_edge_threshold: float = 0.60
    min_high_confidence_coverage: float = 0.10
    min_margin_over_best_baseline: float = 0.01
    max_rows_per_task: int | None = 180_000


@dataclass(slots=True)
class WalkForwardConfig:
    splits: int = 5
    min_train_size: int = 8_000
    test_size: int = 4_000
    max_train_size: int | None = None
    calibration_fraction: float = 0.20
    embargo_multiplier: int = 1


@dataclass(slots=True)
class ModelConfig:
    candidates: list[str] = field(
        default_factory=lambda: [
            "hist_gradient_boosting",
            "extra_trees",
            "lightgbm",
            "xgboost",
            "logistic",
        ]
    )
    use_gpu: bool = True
    random_seed: int = 42
    n_jobs: int = -1


@dataclass(slots=True)
class RuntimeConfig:
    log_level: str = "INFO"
    fetch_sleep_seconds: float = 0.035


@dataclass(slots=True)
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return ""
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        if any(ch in value for ch in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _minimal_yaml_load(text: str) -> dict[str, Any]:
    """Small fallback parser for the simple section/key YAML configs used here."""
    root: dict[str, Any] = {}
    current: dict[str, Any] | None = None
    current_key: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        stripped = line.strip()
        if indent == 0 and stripped.endswith(":"):
            section = stripped[:-1].strip()
            root[section] = {}
            current = root[section]
            current_key = None
            continue
        if current is None:
            raise ValueError(f"unsupported YAML line outside section: {raw_line}")
        if stripped.startswith("- "):
            if current_key is None:
                raise ValueError(f"unsupported YAML list without key: {raw_line}")
            current.setdefault(current_key, []).append(_parse_scalar(stripped[2:]))
            continue
        if ":" not in stripped:
            raise ValueError(f"unsupported YAML line: {raw_line}")
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value:
            current[key] = _parse_scalar(value)
            current_key = None
        else:
            current[key] = []
            current_key = key
    return root


def _load_raw_config(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".toml":
        with path.open("rb") as fh:
            return tomllib.load(fh)
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if suffix in {".yaml", ".yml"}:
        text = path.read_text(encoding="utf-8")
        try:
            import yaml  # type: ignore
        except Exception:  # noqa: BLE001 - optional dependency fallback
            return _minimal_yaml_load(text)
        loaded = yaml.safe_load(text)
        return loaded or {}
    raise ValueError(f"unsupported config file extension: {path.suffix}")


def _coerce_dataclass(cls: type, values: dict[str, Any]):
    field_map = {f.name: f for f in dataclasses.fields(cls)}
    kwargs: dict[str, Any] = {}
    for key, value in values.items():
        if key not in field_map:
            raise ValueError(f"unknown config key for {cls.__name__}: {key}")
        default_type = field_map[key].type
        if default_type is Path or isinstance(getattr(cls(), key), Path):
            kwargs[key] = Path(value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def load_config(path: Path | str | None = None) -> AppConfig:
    cfg = AppConfig()
    if path is None:
        default_path = Path("configs/btc_direction.toml")
        path = default_path if default_path.exists() else None
    if path is None:
        return cfg

    raw = _load_raw_config(Path(path))
    if "data" in raw:
        cfg.data = _coerce_dataclass(DataConfig, raw["data"])
    if "prediction" in raw:
        cfg.prediction = _coerce_dataclass(PredictionConfig, raw["prediction"])
    if "walk_forward" in raw:
        cfg.walk_forward = _coerce_dataclass(WalkForwardConfig, raw["walk_forward"])
    if "models" in raw:
        cfg.models = _coerce_dataclass(ModelConfig, raw["models"])
    if "runtime" in raw:
        cfg.runtime = _coerce_dataclass(RuntimeConfig, raw["runtime"])
    return cfg


def apply_overrides(
    cfg: AppConfig,
    *,
    timeframes: str | None = None,
    horizons: str | None = None,
    candidates: str | None = None,
    max_rows: int | None = None,
) -> AppConfig:
    if timeframes:
        cfg.prediction.timeframes = [x.strip() for x in timeframes.split(",") if x.strip()]
    if horizons:
        cfg.prediction.horizons = [int(x.strip()) for x in horizons.split(",") if x.strip()]
    if candidates:
        cfg.models.candidates = [x.strip() for x in candidates.split(",") if x.strip()]
    if max_rows is not None:
        cfg.prediction.max_rows_per_task = max_rows
    return cfg
