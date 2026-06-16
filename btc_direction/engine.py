from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .config import AppConfig
from .data import load_timeframe_dataset
from .features import build_supervised_frame
from .reporting import load_artifact
from .utils import timeframe_to_timedelta
from .validation import TaskResult, evaluate_task


def run_walk_forward(cfg: AppConfig, on_task: Callable[[list[TaskResult]], None] | None = None) -> list[TaskResult]:
    tasks: list[TaskResult] = []
    for timeframe in cfg.prediction.timeframes:
        candles = load_timeframe_dataset(cfg, timeframe)
        supervised = build_supervised_frame(
            candles,
            timeframe,
            cfg.prediction.horizons,
            label_mode=cfg.prediction.label_mode,
            min_move_bps=cfg.prediction.min_move_bps,
            feature_groups=cfg.prediction.feature_groups,
        )
        for horizon in cfg.prediction.horizons:
            tasks.append(evaluate_task(supervised, timeframe, horizon, cfg))
            if on_task is not None:
                on_task(tasks)
    return tasks


def latest_predictions(
    cfg: AppConfig,
    artifact_path: Path,
    *,
    timeframes: list[str] | None = None,
    horizons: list[int] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    artifact = load_artifact(artifact_path)
    models: dict[str, Any] = artifact.get("models", {})
    requested_timeframes = timeframes or cfg.prediction.timeframes
    requested_horizons = horizons or cfg.prediction.horizons
    rows: list[dict[str, Any]] = []
    for timeframe in requested_timeframes:
        candles = load_timeframe_dataset(cfg, timeframe)
        supervised = build_supervised_frame(
            candles,
            timeframe,
            requested_horizons,
            label_mode=cfg.prediction.label_mode,
            min_move_bps=cfg.prediction.min_move_bps,
            feature_groups=cfg.prediction.feature_groups,
        )
        frame = supervised.frame
        usable = frame.dropna(subset=["close"]).copy()
        if usable.empty:
            continue
        latest = usable.tail(1)
        candle_time = pd.Timestamp(latest.iloc[0]["open_time"]).to_pydatetime()
        base_close = float(latest.iloc[0]["close"])
        for horizon in requested_horizons:
            key = f"{timeframe}|{horizon}"
            model = models.get(key)
            if model is None:
                continue
            prob_up = float(model.predict_proba_up(latest)[0])
            prob_up = min(1.0, max(0.0, prob_up))
            confidence = max(prob_up, 1.0 - prob_up)
            target_time = candle_time.astimezone(UTC) + horizon * timeframe_to_timedelta(timeframe)
            rows.append(
                {
                    "created_at": datetime.now(UTC).isoformat(),
                    "candle_time": candle_time.astimezone(UTC).isoformat(),
                    "target_time": target_time.isoformat(),
                    "symbol": cfg.data.symbol,
                    "timeframe": timeframe,
                    "horizon": int(horizon),
                    "model_name": model.model_name,
                    "accepted": bool(model.accepted),
                    "status": "VALIDATED" if model.accepted else "WEAK",
                    "acceptance_reason": model.acceptance_reason,
                    "base_close": base_close,
                    "prob_up": prob_up,
                    "prob_down": 1.0 - prob_up,
                    "confidence": confidence,
                    "prediction": "UP" if prob_up >= 0.5 else "DOWN",
                    "validated_accuracy": model.metrics.get("accuracy"),
                    "validated_high_confidence_accuracy": model.metrics.get("high_confidence_accuracy"),
                }
            )
    rows.sort(key=lambda r: (r["timeframe"], r["horizon"]))
    return artifact, rows


def compact_prediction_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No predictions available for the requested tasks."
    headers = ["TF", "H", "Dir", "P(UP)", "Conf", "Model", "Status", "ValAcc", "HCAcc", "Target"]
    widths = [max(len(str(row.get(key.lower(), ""))) for row in []) if False else 0]
    table_rows = []
    for row in rows:
        table_rows.append(
            [
                row["timeframe"],
                str(row["horizon"]),
                row["prediction"],
                f"{row['prob_up']:.3f}",
                f"{row['confidence']:.3f}",
                row["model_name"],
                row["status"],
                "n/a" if row["validated_accuracy"] is None else f"{row['validated_accuracy']:.3f}",
                "n/a"
                if row["validated_high_confidence_accuracy"] is None
                else f"{row['validated_high_confidence_accuracy']:.3f}",
                row["target_time"],
            ]
        )
    col_widths = [
        max(len(headers[i]), *(len(r[i]) for r in table_rows))
        for i in range(len(headers))
    ]
    lines = ["  ".join(headers[i].ljust(col_widths[i]) for i in range(len(headers)))]
    lines.append("  ".join("-" * col_widths[i] for i in range(len(headers))))
    for row in table_rows:
        lines.append("  ".join(row[i].ljust(col_widths[i]) for i in range(len(headers))))
    return "\n".join(lines)
