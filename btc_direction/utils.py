from __future__ import annotations

import json
import math
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

TIMEFRAME_RE = re.compile(r"^(?P<value>\d+)(?P<unit>[mhd])$")


def utc_now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def parse_datetime(value: str | None, default: datetime | None = None) -> datetime:
    if value is None:
        if default is None:
            raise ValueError("datetime value is required")
        return default
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    if len(text) == 10:
        return datetime.fromisoformat(text).replace(tzinfo=UTC)
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def to_ms(dt: datetime) -> int:
    return int(dt.astimezone(UTC).timestamp() * 1000)


def timeframe_to_minutes(timeframe: str) -> int:
    match = TIMEFRAME_RE.match(timeframe)
    if not match:
        raise ValueError(f"unsupported timeframe: {timeframe}")
    value = int(match.group("value"))
    unit = match.group("unit")
    if unit == "m":
        return value
    if unit == "h":
        return value * 60
    if unit == "d":
        return value * 1440
    raise ValueError(f"unsupported timeframe: {timeframe}")


def timeframe_to_timedelta(timeframe: str) -> timedelta:
    return timedelta(minutes=timeframe_to_minutes(timeframe))


def timeframe_to_pandas_rule(timeframe: str) -> str:
    minutes = timeframe_to_minutes(timeframe)
    return f"{minutes}min"


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def finite_or_none(value: float | int | None) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"{type(value)!r} is not JSON serializable")


def stable_json_dumps(value: Any) -> str:
    return json.dumps(value, default=json_default, indent=2, sort_keys=True)


def hardware_info() -> dict[str, Any]:
    info: dict[str, Any] = {}
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda_device_count"] = int(torch.cuda.device_count())
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            info["cuda_capability"] = list(torch.cuda.get_device_capability(0))
    except Exception as exc:  # noqa: BLE001 - diagnostic metadata only
        info["torch_error"] = f"{type(exc).__name__}: {exc}"
    return info
