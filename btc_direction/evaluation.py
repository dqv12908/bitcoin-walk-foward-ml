from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(slots=True)
class MetricBundle:
    metrics: dict[str, Any]
    calibration: list[dict[str, Any]]
    high_confusion: dict[str, int]
    regime: list[dict[str, Any]]


def _safe_div(num: float, den: float) -> float | None:
    if den == 0:
        return None
    return float(num / den)


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    return {
        "down_down": int(((y_true == 0) & (y_pred == 0)).sum()),
        "down_up": int(((y_true == 0) & (y_pred == 1)).sum()),
        "up_down": int(((y_true == 1) & (y_pred == 0)).sum()),
        "up_up": int(((y_true == 1) & (y_pred == 1)).sum()),
    }


def metric_dict(
    y_true: np.ndarray,
    prob_up: np.ndarray,
    *,
    high_confidence_threshold: float,
) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    prob_up = np.asarray(prob_up, dtype=float)
    y_pred = (prob_up >= 0.5).astype(int)
    counts = confusion_counts(y_true, y_pred)
    n = int(len(y_true))
    correct = int((y_true == y_pred).sum())
    tp = counts["up_up"]
    tn = counts["down_down"]
    fp = counts["down_up"]
    fn = counts["up_down"]
    confidence = np.maximum(prob_up, 1.0 - prob_up)
    high_mask = confidence >= high_confidence_threshold
    sweep: list[dict[str, Any]] = []
    for threshold in (0.50, 0.525, 0.55, 0.575, 0.60, 0.625, 0.65, 0.675, 0.70, 0.75):
        mask = confidence >= threshold
        threshold_n = int(mask.sum())
        if threshold_n:
            threshold_pred = y_pred[mask]
            threshold_true = y_true[mask]
            threshold_counts = confusion_counts(threshold_true, threshold_pred)
            threshold_accuracy = float((threshold_true == threshold_pred).mean())
        else:
            threshold_counts = {"down_down": 0, "down_up": 0, "up_down": 0, "up_up": 0}
            threshold_accuracy = None
        sweep.append(
            {
                "threshold": threshold,
                "n": threshold_n,
                "coverage": _safe_div(float(threshold_n), n),
                "accuracy": threshold_accuracy,
                "confusion": threshold_counts,
            }
        )
    populated_sweep = [row for row in sweep if row.get("accuracy") is not None]
    best_selective = max(
        populated_sweep,
        key=lambda row: (row.get("accuracy") or 0.0, row.get("coverage") or 0.0),
        default=None,
    )
    best_selective_10 = max(
        [row for row in populated_sweep if (row.get("coverage") or 0.0) >= 0.10],
        key=lambda row: (row.get("accuracy") or 0.0, row.get("coverage") or 0.0),
        default=None,
    )
    out: dict[str, Any] = {
        "n": n,
        "accuracy": _safe_div(correct, n),
        "up_rate": _safe_div(float(y_true.sum()), n),
        "predicted_up_rate": _safe_div(float(y_pred.sum()), n),
        "precision_up": _safe_div(tp, tp + fp),
        "recall_up": _safe_div(tp, tp + fn),
        "precision_down": _safe_div(tn, tn + fn),
        "recall_down": _safe_div(tn, tn + fp),
        "brier": float(np.mean((prob_up - y_true) ** 2)) if n else None,
        "mean_confidence": float(np.mean(confidence)) if n else None,
        "high_confidence_threshold": high_confidence_threshold,
        "high_confidence_n": int(high_mask.sum()),
        "high_confidence_coverage": _safe_div(float(high_mask.sum()), n),
        "confidence_threshold_sweep": sweep,
        "best_selective_threshold": best_selective.get("threshold") if best_selective else None,
        "best_selective_accuracy": best_selective.get("accuracy") if best_selective else None,
        "best_selective_coverage": best_selective.get("coverage") if best_selective else None,
        "best_selective_threshold_10pct": best_selective_10.get("threshold") if best_selective_10 else None,
        "best_selective_accuracy_10pct": best_selective_10.get("accuracy") if best_selective_10 else None,
        "best_selective_coverage_10pct": best_selective_10.get("coverage") if best_selective_10 else None,
        "confusion": counts,
    }
    if high_mask.any():
        high_true = y_true[high_mask]
        high_prob = prob_up[high_mask]
        high_pred = (high_prob >= 0.5).astype(int)
        high_counts = confusion_counts(high_true, high_pred)
        out.update(
            {
                "high_confidence_accuracy": float((high_true == high_pred).mean()),
                "high_confidence_precision_up": _safe_div(high_counts["up_up"], high_counts["up_up"] + high_counts["down_up"]),
                "high_confidence_recall_up": _safe_div(high_counts["up_up"], high_counts["up_up"] + high_counts["up_down"]),
                "high_confidence_precision_down": _safe_div(
                    high_counts["down_down"], high_counts["down_down"] + high_counts["up_down"]
                ),
                "high_confidence_recall_down": _safe_div(
                    high_counts["down_down"], high_counts["down_down"] + high_counts["down_up"]
                ),
                "high_confidence_confusion": high_counts,
            }
        )
    else:
        out.update(
            {
                "high_confidence_accuracy": None,
                "high_confidence_precision_up": None,
                "high_confidence_recall_up": None,
                "high_confidence_precision_down": None,
                "high_confidence_recall_down": None,
                "high_confidence_confusion": {"down_down": 0, "down_up": 0, "up_down": 0, "up_up": 0},
            }
        )
    return out


def calibration_bins(y_true: np.ndarray, prob_up: np.ndarray, *, bins: int = 10) -> tuple[list[dict[str, Any]], float | None]:
    y_true = np.asarray(y_true, dtype=int)
    prob_up = np.asarray(prob_up, dtype=float)
    if len(y_true) == 0:
        return [], None
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: list[dict[str, Any]] = []
    ece = 0.0
    for i in range(bins):
        lo = edges[i]
        hi = edges[i + 1]
        if i == bins - 1:
            mask = (prob_up >= lo) & (prob_up <= hi)
        else:
            mask = (prob_up >= lo) & (prob_up < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        mean_prob = float(prob_up[mask].mean())
        observed = float(y_true[mask].mean())
        abs_error = abs(mean_prob - observed)
        ece += (n / len(y_true)) * abs_error
        rows.append(
            {
                "bin": i,
                "prob_low": float(lo),
                "prob_high": float(hi),
                "n": n,
                "mean_prob_up": mean_prob,
                "observed_up_rate": observed,
                "abs_error": abs_error,
            }
        )
    return rows, float(ece)


def regime_metrics(meta: pd.DataFrame, y_true: np.ndarray, prob_up: np.ndarray) -> list[dict[str, Any]]:
    if meta.empty:
        return []
    candidates = [c for c in ("return_std_55", "return_std_34", "return_std_21", "range_mean_55") if c in meta.columns]
    if not candidates:
        return []
    reg_col = candidates[0]
    values = pd.to_numeric(meta[reg_col], errors="coerce")
    valid = values.notna().to_numpy()
    if valid.sum() < 30:
        return []
    labels = pd.Series(index=meta.index, dtype="object")
    try:
        labels.loc[valid] = pd.qcut(values.loc[valid], q=3, labels=["low_vol", "mid_vol", "high_vol"], duplicates="drop")
    except ValueError:
        return []
    rows: list[dict[str, Any]] = []
    y_true = np.asarray(y_true, dtype=int)
    prob_up = np.asarray(prob_up, dtype=float)
    for name in ("low_vol", "mid_vol", "high_vol"):
        mask = (labels == name).to_numpy()
        if mask.sum() == 0:
            continue
        basic = metric_dict(y_true[mask], prob_up[mask], high_confidence_threshold=0.60)
        rows.append(
            {
                "regime": name,
                "n": basic["n"],
                "accuracy": basic["accuracy"],
                "up_rate": basic["up_rate"],
                "precision_up": basic["precision_up"],
                "precision_down": basic["precision_down"],
            }
        )
    return rows


def evaluate_predictions(
    y_true: np.ndarray,
    prob_up: np.ndarray,
    *,
    high_confidence_threshold: float,
    meta: pd.DataFrame | None = None,
) -> MetricBundle:
    metrics = metric_dict(y_true, prob_up, high_confidence_threshold=high_confidence_threshold)
    cal, ece = calibration_bins(y_true, prob_up)
    metrics["expected_calibration_error"] = ece
    high_conf = metrics.get("high_confidence_confusion") or {}
    regime = regime_metrics(meta if meta is not None else pd.DataFrame(), y_true, prob_up)
    return MetricBundle(metrics=metrics, calibration=cal, high_confusion=high_conf, regime=regime)
