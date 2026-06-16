from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from contextlib import closing
from pathlib import Path
from typing import Any

from .config import AppConfig
from .utils import ensure_dirs, json_default, stable_json_dumps, utc_now
from .validation import TaskResult


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    command TEXT NOT NULL,
    profile TEXT,
    status TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    config_json TEXT NOT NULL,
    report_json TEXT,
    report_md TEXT,
    predictions_csv TEXT,
    artifact_path TEXT,
    accepted_count INTEGER,
    task_count INTEGER,
    mean_accuracy REAL,
    best_accuracy REAL,
    best_high_confidence_accuracy REAL,
    notes TEXT,
    next_step TEXT
);

CREATE INDEX IF NOT EXISTS idx_runs_config ON runs(config_hash, command, profile, status);
CREATE INDEX IF NOT EXISTS idx_runs_score ON runs(status, accepted_count, best_accuracy);

CREATE TABLE IF NOT EXISTS run_tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    timeframe TEXT NOT NULL,
    horizon INTEGER NOT NULL,
    best_candidate TEXT,
    accepted INTEGER NOT NULL,
    accuracy REAL,
    high_confidence_accuracy REAL,
    high_confidence_coverage REAL,
    best_baseline_accuracy REAL,
    margin_over_best_baseline REAL,
    reason TEXT,
    metrics_json TEXT,
    baselines_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_run_tasks_score ON run_tasks(accepted, accuracy, high_confidence_accuracy);
CREATE INDEX IF NOT EXISTS idx_run_tasks_task ON run_tasks(timeframe, horizon);
"""


def _connect(cfg: AppConfig) -> sqlite3.Connection:
    ensure_dirs(cfg.data.experiment_db.parent)
    conn = sqlite3.connect(cfg.data.experiment_db)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


def _compute_source_digest() -> str:
    package_dir = Path(__file__).resolve().parent
    source_files = [
        "config.py",
        "data.py",
        "engine.py",
        "evaluation.py",
        "features.py",
        "models.py",
        "validation.py",
    ]
    digest = hashlib.sha256()
    for name in source_files:
        path = package_dir / name
        digest.update(name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


_SOURCE_DIGEST = _compute_source_digest()


def source_digest() -> str:
    return _SOURCE_DIGEST


def config_digest(cfg: AppConfig, *, extra: dict[str, Any] | None = None) -> str:
    payload = {"config": cfg.to_dict(), "extra": extra or {}, "source_digest": source_digest()}
    raw = stable_json_dumps(payload).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def make_run_id(command: str, profile: str | None = None) -> str:
    stamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid.uuid4().hex[:8]
    bits = [stamp, command]
    if profile:
        bits.append(profile)
    bits.append(suffix)
    return "_".join(part.replace(" ", "_") for part in bits)


def start_run(
    cfg: AppConfig,
    *,
    command: str,
    profile: str | None,
    config_hash: str,
    run_id: str | None = None,
) -> str:
    run_id = run_id or make_run_id(command, profile)
    now = utc_now().isoformat()
    with closing(_connect(cfg)) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO runs (
                run_id, created_at, updated_at, command, profile, status, config_hash, config_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                now,
                now,
                command,
                profile,
                "running",
                config_hash,
                json.dumps(cfg.to_dict(), default=json_default, sort_keys=True),
            ),
        )
        conn.commit()
    return run_id


def _best_baseline(task: TaskResult) -> float | None:
    values = [m.get("accuracy") for m in task.baselines.values() if m.get("accuracy") is not None]
    if not values:
        return None
    return float(max(values))


def _metrics(task: TaskResult) -> dict[str, Any]:
    if not task.best_candidate or not task.best_candidate.metrics:
        return {}
    return task.best_candidate.metrics


def summarize_tasks(tasks: list[TaskResult]) -> dict[str, Any]:
    accuracies = [
        float(_metrics(task).get("accuracy"))
        for task in tasks
        if _metrics(task).get("accuracy") is not None
    ]
    high_conf = [
        float(_metrics(task).get("high_confidence_accuracy"))
        for task in tasks
        if _metrics(task).get("high_confidence_accuracy") is not None
    ]
    return {
        "accepted_count": sum(1 for task in tasks if task.accepted),
        "task_count": len(tasks),
        "mean_accuracy": float(sum(accuracies) / len(accuracies)) if accuracies else None,
        "best_accuracy": float(max(accuracies)) if accuracies else None,
        "best_high_confidence_accuracy": float(max(high_conf)) if high_conf else None,
    }


def next_escalation_step(tasks: list[TaskResult]) -> str:
    if any(task.accepted for task in tasks):
        return "Validated task found; evaluate paper/live performance and compare against newer candidates."
    best_accuracy = None
    best_task: TaskResult | None = None
    for task in tasks:
        accuracy = _metrics(task).get("accuracy")
        if accuracy is not None and (best_accuracy is None or float(accuracy) > best_accuracy):
            best_accuracy = float(accuracy)
            best_task = task
    if best_task is None:
        return "All candidates failed; broaden model families and verify data coverage."
    best_baseline = _best_baseline(best_task)
    if best_baseline is not None and best_accuracy is not None and best_accuracy <= best_baseline:
        return "Candidate failed to beat baseline; try alternative labels, feature groups, and regime-specific models."
    if best_accuracy is not None and best_accuracy < 0.60:
        return "Accuracy below acceptance gate; continue with wider data, confidence filters, ensembles, and higher-compute profiles."
    return "Accuracy near gate but not accepted; inspect baseline margin, calibration, and high-confidence coverage."


def finish_run(
    cfg: AppConfig,
    *,
    run_id: str,
    tasks: list[TaskResult],
    report_paths: tuple[Path, Path, Path] | None = None,
    artifact_path: Path | None = None,
    status: str = "completed",
    notes: str | None = None,
) -> None:
    summary = summarize_tasks(tasks)
    report_json = report_md = predictions_csv = None
    if report_paths:
        report_json, report_md, predictions_csv = (str(path) for path in report_paths)
    now = utc_now().isoformat()
    with closing(_connect(cfg)) as conn:
        conn.execute("DELETE FROM run_tasks WHERE run_id = ?", (run_id,))
        conn.execute(
            """
            UPDATE runs
            SET updated_at = ?, status = ?, report_json = ?, report_md = ?, predictions_csv = ?,
                artifact_path = ?, accepted_count = ?, task_count = ?, mean_accuracy = ?,
                best_accuracy = ?, best_high_confidence_accuracy = ?, notes = ?, next_step = ?
            WHERE run_id = ?
            """,
            (
                now,
                status,
                report_json,
                report_md,
                predictions_csv,
                str(artifact_path) if artifact_path else None,
                summary["accepted_count"],
                summary["task_count"],
                summary["mean_accuracy"],
                summary["best_accuracy"],
                summary["best_high_confidence_accuracy"],
                notes,
                next_escalation_step(tasks),
                run_id,
            ),
        )
        for task in tasks:
            metrics = _metrics(task)
            baseline = _best_baseline(task)
            accuracy = metrics.get("accuracy")
            margin = None if accuracy is None or baseline is None else float(accuracy) - baseline
            conn.execute(
                """
                INSERT INTO run_tasks (
                    run_id, timeframe, horizon, best_candidate, accepted, accuracy,
                    high_confidence_accuracy, high_confidence_coverage, best_baseline_accuracy,
                    margin_over_best_baseline, reason, metrics_json, baselines_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    task.timeframe,
                    int(task.horizon),
                    task.best_candidate.name if task.best_candidate else None,
                    1 if task.accepted else 0,
                    metrics.get("accuracy"),
                    metrics.get("high_confidence_accuracy"),
                    metrics.get("high_confidence_coverage"),
                    baseline,
                    margin,
                    task.acceptance_reason,
                    json.dumps(metrics, default=json_default, sort_keys=True),
                    json.dumps(task.baselines, default=json_default, sort_keys=True),
                ),
            )
        conn.commit()


def mark_run_failed(cfg: AppConfig, *, run_id: str, error: str) -> None:
    now = utc_now().isoformat()
    with closing(_connect(cfg)) as conn:
        conn.execute(
            "UPDATE runs SET updated_at = ?, status = ?, notes = ?, next_step = ? WHERE run_id = ?",
            (now, "failed", error, "Resolve failed run and resume search.", run_id),
        )
        conn.commit()


def completed_run_exists(
    cfg: AppConfig,
    *,
    command: str,
    profile: str | None,
    config_hash: str,
) -> str | None:
    with closing(_connect(cfg)) as conn:
        row = conn.execute(
            """
            SELECT run_id FROM runs
            WHERE command = ? AND COALESCE(profile, '') = COALESCE(?, '')
              AND config_hash = ? AND status = 'completed'
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (command, profile, config_hash),
        ).fetchone()
    return str(row["run_id"]) if row else None


def best_run_id(cfg: AppConfig) -> str | None:
    with closing(_connect(cfg)) as conn:
        row = conn.execute(
            """
            SELECT run_id FROM runs
            WHERE status = 'completed'
            ORDER BY accepted_count DESC, best_accuracy DESC, best_high_confidence_accuracy DESC, updated_at DESC
            LIMIT 1
            """
        ).fetchone()
    return str(row["run_id"]) if row else None


def accepted_run_exists(cfg: AppConfig) -> str | None:
    with closing(_connect(cfg)) as conn:
        row = conn.execute(
            """
            SELECT run_id FROM runs
            WHERE status = 'completed' AND COALESCE(accepted_count, 0) > 0
            ORDER BY accepted_count DESC, best_accuracy DESC, updated_at DESC
            LIMIT 1
            """
        ).fetchone()
    return str(row["run_id"]) if row else None


def get_run(cfg: AppConfig, run_id: str) -> dict[str, Any] | None:
    if run_id == "best":
        resolved = best_run_id(cfg)
        if resolved is None:
            return None
        run_id = resolved
    with closing(_connect(cfg)) as conn:
        run = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if run is None:
            return None
        tasks = conn.execute(
            "SELECT * FROM run_tasks WHERE run_id = ? ORDER BY accepted DESC, accuracy DESC, timeframe, horizon",
            (run_id,),
        ).fetchall()
    return {
        "run": dict(run),
        "tasks": [dict(row) for row in tasks],
    }


def leaderboard_rows(
    cfg: AppConfig,
    *,
    limit: int = 20,
    accepted_only: bool = False,
    min_high_confidence_coverage: float | None = None,
    timeframe: str | None = None,
    horizon: int | None = None,
) -> list[dict[str, Any]]:
    where = ["r.status = 'completed'"]
    params: list[Any] = []
    if accepted_only:
        where.append("t.accepted = 1")
    if min_high_confidence_coverage is not None:
        where.append("COALESCE(t.high_confidence_coverage, 0) >= ?")
        params.append(float(min_high_confidence_coverage))
    if timeframe:
        where.append("t.timeframe = ?")
        params.append(timeframe)
    if horizon is not None:
        where.append("t.horizon = ?")
        params.append(int(horizon))
    params.append(int(limit))
    sql = f"""
        SELECT
            r.run_id, r.created_at, r.command, r.profile, r.report_json, r.report_md, r.artifact_path,
            t.timeframe, t.horizon, t.best_candidate, t.accepted, t.accuracy,
            t.high_confidence_accuracy, t.high_confidence_coverage, t.best_baseline_accuracy,
            t.margin_over_best_baseline, t.reason, t.metrics_json, r.next_step
        FROM run_tasks t
        JOIN runs r ON r.run_id = t.run_id
        WHERE {' AND '.join(where)}
        ORDER BY t.accepted DESC, t.accuracy DESC, t.high_confidence_accuracy DESC, r.created_at DESC
        LIMIT ?
    """
    with closing(_connect(cfg)) as conn:
        rows = conn.execute(sql, params).fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        try:
            metrics = json.loads(item.get("metrics_json") or "{}")
        except json.JSONDecodeError:
            metrics = {}
        item["best_selective_accuracy_10pct"] = metrics.get("best_selective_accuracy_10pct")
        item["best_selective_coverage_10pct"] = metrics.get("best_selective_coverage_10pct")
        item["best_selective_threshold_10pct"] = metrics.get("best_selective_threshold_10pct")
        out.append(item)
    return out
