from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import AppConfig
from .data import load_timeframe_dataset
from .evaluation import metric_dict
from .utils import ensure_dirs


SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    candle_time TEXT NOT NULL,
    target_time TEXT NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    horizon INTEGER NOT NULL,
    model_name TEXT NOT NULL,
    accepted INTEGER NOT NULL,
    base_close REAL NOT NULL,
    prob_up REAL NOT NULL,
    confidence REAL NOT NULL,
    prediction TEXT NOT NULL,
    actual TEXT,
    target_close REAL,
    correct INTEGER,
    settled_at TEXT,
    artifact_created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_predictions_unsettled ON predictions(actual, timeframe, target_time);
"""


def _ensure_unique_signal_index(conn: sqlite3.Connection) -> None:
    try:
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_predictions_unique_signal
            ON predictions(symbol, candle_time, target_time, timeframe, horizon, model_name, artifact_created_at)
            """
        )
        conn.commit()
        return
    except sqlite3.IntegrityError:
        conn.execute(
            """
            DELETE FROM predictions
            WHERE id NOT IN (
                SELECT MIN(id)
                FROM predictions
                GROUP BY symbol, candle_time, target_time, timeframe, horizon, model_name, artifact_created_at
            )
            """
        )
        conn.commit()
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_predictions_unique_signal
        ON predictions(symbol, candle_time, target_time, timeframe, horizon, model_name, artifact_created_at)
        """
    )
    conn.commit()


def _delete_non_forward_predictions(conn: sqlite3.Connection) -> None:
    rows = conn.execute("SELECT id, created_at, target_time FROM predictions").fetchall()
    stale_ids: list[int] = []
    for row_id, created_at_raw, target_time_raw in rows:
        try:
            created_at = pd.Timestamp(created_at_raw)
            target_time = pd.Timestamp(target_time_raw)
        except Exception:  # noqa: BLE001 - malformed rows cannot be scored safely
            stale_ids.append(int(row_id))
            continue
        if target_time <= created_at:
            stale_ids.append(int(row_id))
    if stale_ids:
        conn.executemany("DELETE FROM predictions WHERE id = ?", [(row_id,) for row_id in stale_ids])
        conn.commit()


def connect(db_path: Path) -> sqlite3.Connection:
    ensure_dirs(db_path.parent)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    _delete_non_forward_predictions(conn)
    _ensure_unique_signal_index(conn)
    return conn


def record_predictions(cfg: AppConfig, rows: list[dict[str, Any]], *, artifact_created_at: str | None = None) -> int:
    if not rows:
        return 0
    forward_rows: list[dict[str, Any]] = []
    for row in rows:
        try:
            created_at = pd.Timestamp(row["created_at"])
            target_time = pd.Timestamp(row["target_time"])
        except Exception:  # noqa: BLE001 - malformed prediction rows should not be paper-scored
            continue
        if target_time > created_at:
            forward_rows.append(row)
    if not forward_rows:
        return 0
    conn = connect(cfg.data.paper_db)
    try:
        conn.executemany(
            """
            INSERT OR IGNORE INTO predictions (
                created_at, candle_time, target_time, symbol, timeframe, horizon,
                model_name, accepted, base_close, prob_up, confidence, prediction,
                artifact_created_at
            ) VALUES (
                :created_at, :candle_time, :target_time, :symbol, :timeframe, :horizon,
                :model_name, :accepted, :base_close, :prob_up, :confidence, :prediction,
                :artifact_created_at
            )
            """,
            [{**row, "artifact_created_at": artifact_created_at} for row in forward_rows],
        )
        conn.commit()
        return conn.total_changes
    finally:
        conn.close()


def settle_predictions(cfg: AppConfig) -> dict[str, Any]:
    conn = connect(cfg.data.paper_db)
    try:
        unsettled = pd.read_sql_query(
            "SELECT * FROM predictions WHERE actual IS NULL ORDER BY target_time",
            conn,
        )
        if unsettled.empty:
            return {"settled": 0, "remaining": 0}
        settled_rows: list[tuple[Any, ...]] = []
        for timeframe, group in unsettled.groupby("timeframe"):
            candles = load_timeframe_dataset(cfg, timeframe)
            candles["open_time_key"] = pd.to_datetime(candles["open_time"], utc=True, format="mixed").map(lambda x: x.isoformat())
            by_time = candles.set_index("open_time_key")["close"].to_dict()
            for row in group.itertuples(index=False):
                target_close = by_time.get(row.target_time)
                if target_close is None:
                    continue
                actual = "UP" if float(target_close) > float(row.base_close) else "DOWN"
                correct = int(actual == row.prediction)
                settled_rows.append((actual, float(target_close), correct, pd.Timestamp.utcnow().isoformat(), int(row.id)))
        if settled_rows:
            conn.executemany(
                "UPDATE predictions SET actual=?, target_close=?, correct=?, settled_at=? WHERE id=?",
                settled_rows,
            )
            conn.commit()
        remaining = conn.execute("SELECT COUNT(*) FROM predictions WHERE actual IS NULL").fetchone()[0]
        return {"settled": len(settled_rows), "remaining": int(remaining)}
    finally:
        conn.close()


def paper_summary(cfg: AppConfig) -> dict[str, Any]:
    conn = connect(cfg.data.paper_db)
    try:
        df = pd.read_sql_query("SELECT * FROM predictions WHERE actual IS NOT NULL", conn)
    finally:
        conn.close()
    if df.empty:
        return {"n": 0, "by_task": [], "overall": None}
    y = (df["actual"] == "UP").astype(int).to_numpy()
    prob = pd.to_numeric(df["prob_up"], errors="coerce").to_numpy(dtype=float)
    overall = metric_dict(y, prob, high_confidence_threshold=cfg.prediction.high_confidence_threshold)
    rows = []
    for (timeframe, horizon), group in df.groupby(["timeframe", "horizon"]):
        y_g = (group["actual"] == "UP").astype(int).to_numpy()
        p_g = pd.to_numeric(group["prob_up"], errors="coerce").to_numpy(dtype=float)
        metrics = metric_dict(y_g, p_g, high_confidence_threshold=cfg.prediction.high_confidence_threshold)
        rows.append({"timeframe": timeframe, "horizon": int(horizon), **metrics})
    return {"n": int(len(df)), "overall": overall, "by_task": rows}
