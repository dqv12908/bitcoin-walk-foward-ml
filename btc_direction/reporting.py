from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .config import AppConfig
from .utils import ensure_dirs, hardware_info, json_default, stable_json_dumps
from .validation import CandidateResult, TaskResult


def _metric(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _candidate_summary(candidate: CandidateResult) -> dict[str, Any]:
    if candidate.metrics is None:
        return {"name": candidate.name, "error": candidate.error}
    return {
        "name": candidate.name,
        "metrics": candidate.metrics,
        "calibration": candidate.calibration,
        "regime": candidate.regime,
        "fold_metrics": candidate.fold_metrics,
        "error": candidate.error,
    }


def task_to_dict(task: TaskResult) -> dict[str, Any]:
    return {
        "timeframe": task.timeframe,
        "horizon": task.horizon,
        "n_samples": task.n_samples,
        "feature_count": task.feature_count,
        "splits": task.splits,
        "accepted": task.accepted,
        "acceptance_reason": task.acceptance_reason,
        "best_candidate": task.best_candidate.name if task.best_candidate else None,
        "best_metrics": task.best_candidate.metrics if task.best_candidate else None,
        "baselines": task.baselines,
        "candidates": [_candidate_summary(c) for c in task.candidates],
    }


def save_artifact(
    cfg: AppConfig,
    tasks: list[TaskResult],
    *,
    path: Path | None = None,
) -> Path:
    ensure_dirs(cfg.data.model_dir)
    if path is None:
        path = cfg.data.model_dir / "btc_direction_latest.joblib"
    models = {
        f"{task.timeframe}|{task.horizon}": task.fitted_model
        for task in tasks
        if task.fitted_model is not None
    }
    artifact = {
        "created_at": datetime.now(UTC).isoformat(),
        "config": cfg.to_dict(),
        "hardware": hardware_info(),
        "models": models,
        "summary": [task_to_dict(task) for task in tasks],
    }
    joblib.dump(artifact, path)
    return path


def load_artifact(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing model artifact: {path}")
    return joblib.load(path)


def write_reports(
    cfg: AppConfig,
    tasks: list[TaskResult],
    *,
    stem: str | None = None,
) -> tuple[Path, Path, Path]:
    ensure_dirs(cfg.data.report_dir)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    stem = stem or f"btc_direction_walkforward_{stamp}"
    json_path = cfg.data.report_dir / f"{stem}.json"
    md_path = cfg.data.report_dir / f"{stem}.md"
    pred_path = cfg.data.report_dir / f"{stem}_predictions.csv"

    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "config": cfg.to_dict(),
        "hardware": hardware_info(),
        "tasks": [task_to_dict(task) for task in tasks],
    }
    json_path.write_text(json.dumps(payload, default=json_default, indent=2), encoding="utf-8")

    lines: list[str] = []
    lines.append("# BTC Candle Direction Walk-Forward Report")
    lines.append("")
    lines.append("This report is generated only from embargoed walk-forward validation. In-sample fit metrics are not reported as proof of edge.")
    lines.append("")
    hw = hardware_info()
    if hw:
        lines.append("## Hardware")
        lines.append("")
        for key, value in hw.items():
            lines.append(f"- `{key}`: `{value}`")
        lines.append("")
    lines.append("## Task Summary")
    lines.append("")
    lines.append("| Timeframe | Horizon | Samples | Features | Best | Accepted | Accuracy | High-conf acc | High-conf cov | Selective acc >=10% cov | Best baseline | Reason |")
    lines.append("|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---|")
    for task in tasks:
        best = task.best_candidate
        metrics = best.metrics if best and best.metrics else {}
        baseline_acc = max((m.get("accuracy") or 0.0 for m in task.baselines.values()), default=0.0)
        lines.append(
            "| "
            f"{task.timeframe} | {task.horizon} | {task.n_samples} | {task.feature_count} | "
            f"{best.name if best else 'none'} | {task.accepted} | "
            f"{_metric(metrics.get('accuracy'))} | {_metric(metrics.get('high_confidence_accuracy'))} | "
            f"{_metric(metrics.get('high_confidence_coverage'))} | {_metric(metrics.get('best_selective_accuracy_10pct'))} | "
            f"{_metric(baseline_acc)} | "
            f"{task.acceptance_reason} |"
        )
    lines.append("")

    for task in tasks:
        lines.append(f"## {task.timeframe} Horizon {task.horizon}")
        lines.append("")
        lines.append(f"- Samples: `{task.n_samples}`")
        lines.append(f"- Features: `{task.feature_count}`")
        lines.append(f"- Splits: `{len(task.splits)}`")
        lines.append(f"- Acceptance: `{task.accepted}` - {task.acceptance_reason}")
        lines.append("")
        lines.append("### Candidates")
        lines.append("")
        lines.append("| Candidate | Accuracy | Precision UP | Recall UP | Precision DOWN | Recall DOWN | High-conf acc | High-conf cov | Sel acc >=10% cov | Brier | ECE | Error |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for candidate in task.candidates:
            metrics = candidate.metrics or {}
            lines.append(
                "| "
                f"{candidate.name} | {_metric(metrics.get('accuracy'))} | "
                f"{_metric(metrics.get('precision_up'))} | {_metric(metrics.get('recall_up'))} | "
                f"{_metric(metrics.get('precision_down'))} | {_metric(metrics.get('recall_down'))} | "
                f"{_metric(metrics.get('high_confidence_accuracy'))} | {_metric(metrics.get('high_confidence_coverage'))} | "
                f"{_metric(metrics.get('best_selective_accuracy_10pct'))} | "
                f"{_metric(metrics.get('brier'))} | {_metric(metrics.get('expected_calibration_error'))} | "
                f"{candidate.error or ''} |"
            )
        lines.append("")
        lines.append("### Baselines")
        lines.append("")
        lines.append("| Baseline | Accuracy | Precision UP | Recall UP | Precision DOWN | Recall DOWN | Brier |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for name, metrics in task.baselines.items():
            lines.append(
                "| "
                f"{name} | {_metric(metrics.get('accuracy'))} | {_metric(metrics.get('precision_up'))} | "
                f"{_metric(metrics.get('recall_up'))} | {_metric(metrics.get('precision_down'))} | "
                f"{_metric(metrics.get('recall_down'))} | {_metric(metrics.get('brier'))} |"
            )
        if task.best_candidate and task.best_candidate.metrics:
            lines.append("")
            lines.append("### Confusion Matrix")
            conf = task.best_candidate.metrics.get("confusion", {})
            lines.append("")
            lines.append("| Actual \\ Predicted | DOWN | UP |")
            lines.append("|---|---:|---:|")
            lines.append(f"| DOWN | {conf.get('down_down', 0)} | {conf.get('down_up', 0)} |")
            lines.append(f"| UP | {conf.get('up_down', 0)} | {conf.get('up_up', 0)} |")
            lines.append("")
            lines.append("### Calibration")
            lines.append("")
            lines.append("| Bin | Mean P(UP) | Observed UP | N | Abs error |")
            lines.append("|---:|---:|---:|---:|---:|")
            for row in task.best_candidate.calibration:
                lines.append(
                    f"| {row['bin']} | {_metric(row['mean_prob_up'])} | {_metric(row['observed_up_rate'])} | "
                    f"{row['n']} | {_metric(row['abs_error'])} |"
                )
            if task.best_candidate.regime:
                lines.append("")
                lines.append("### Market Regimes")
                lines.append("")
                lines.append("| Regime | N | Accuracy | UP rate | Precision UP | Precision DOWN |")
                lines.append("|---|---:|---:|---:|---:|---:|")
                for row in task.best_candidate.regime:
                    lines.append(
                        f"| {row['regime']} | {row['n']} | {_metric(row['accuracy'])} | "
                        f"{_metric(row['up_rate'])} | {_metric(row['precision_up'])} | {_metric(row['precision_down'])} |"
                    )
            sweep = task.best_candidate.metrics.get("confidence_threshold_sweep") or []
            if sweep:
                lines.append("")
                lines.append("### Confidence Threshold Sweep")
                lines.append("")
                lines.append("| Threshold | N | Coverage | Accuracy |")
                lines.append("|---:|---:|---:|---:|")
                for row in sweep:
                    lines.append(
                        f"| {_metric(row.get('threshold'))} | {row.get('n', 0)} | "
                        f"{_metric(row.get('coverage'))} | {_metric(row.get('accuracy'))} |"
                    )
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")

    pred_frames = [task.best_candidate.predictions for task in tasks if task.best_candidate and not task.best_candidate.predictions.empty]
    if pred_frames:
        pd.concat(pred_frames, ignore_index=True).to_csv(pred_path, index=False)
    else:
        pd.DataFrame().to_csv(pred_path, index=False)
    return json_path, md_path, pred_path
