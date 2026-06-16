from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from datetime import timedelta
from pathlib import Path

import pandas as pd

from .config import AppConfig, apply_overrides, load_config
from .data import (
    CROSS_ASSET_SYMBOLS,
    FUTURES_RATIO_SOURCES,
    FUTURES_BASIS_SOURCES,
    bybit_funding_path,
    bybit_kline_path,
    bybit_open_interest_path,
    collect_binance_live_flow,
    fetch_binance_agg_trades,
    fetch_binance_vision_agg_trades,
    fetch_binance_vision_book_depth,
    fetch_binance_vision_metrics,
    fetch_binance_klines,
    fetch_binance_order_book_snapshot,
    fetch_bybit_funding_history,
    fetch_bybit_klines,
    fetch_bybit_open_interest,
    fetch_funding_rates,
    fetch_coinbase_candles,
    fetch_futures_basis_klines,
    fetch_futures_ratio,
    fetch_hyperliquid_candles,
    fetch_okx_klines,
    fetch_okx_long_short_ratio_history,
    fetch_okx_open_interest_history,
    fetch_open_interest,
    raw_agg_trade_path,
    raw_agg_trade_feature_path,
    raw_symbol_kline_path,
    coinbase_path,
    raw_live_flow_path,
    raw_order_book_path,
    raw_book_depth_feature_path,
    raw_metrics_feature_path,
    okx_kline_path,
    okx_long_short_ratio_path,
    okx_open_interest_path,
    raw_futures_basis_path,
    raw_futures_ratio_path,
    raw_kline_path,
)
from .engine import compact_prediction_table, latest_predictions, run_walk_forward
from .experiments import (
    accepted_run_exists,
    completed_run_exists,
    config_digest,
    finish_run,
    get_run,
    leaderboard_rows,
    mark_run_failed,
    next_escalation_step,
    start_run,
)
from .paper import paper_summary, record_predictions, settle_predictions
from .reporting import load_artifact, save_artifact, write_reports
from .utils import parse_datetime, stable_json_dumps, utc_now
from .utils import hardware_info


def _load(args: argparse.Namespace) -> AppConfig:
    cfg = load_config(args.config)
    return apply_overrides(
        cfg,
        timeframes=getattr(args, "timeframes", None),
        horizons=getattr(args, "horizons", None),
        candidates=getattr(args, "candidates", None),
        max_rows=getattr(args, "max_rows", None),
    )


def _parse_csv(text: str | None, cast=str):
    if text is None:
        return None
    return [cast(x.strip()) for x in text.split(",") if x.strip()]


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def cmd_fetch(args: argparse.Namespace) -> int:
    cfg = _load(args)
    end = parse_datetime(args.end, utc_now())
    start = parse_datetime(args.start, end - timedelta(days=args.days))
    intervals = _parse_csv(args.intervals, str) or [cfg.data.base_interval]
    markets = _parse_csv(args.markets, str) or [cfg.data.market]
    for market in markets:
        for interval in intervals:
            path = fetch_binance_klines(cfg, market=market, interval=interval, start=start, end=end, force=args.force)
            print(f"candles {market} {interval}: {path}")
    if args.cross_assets:
        for symbol in _parse_csv(args.cross_asset_symbols, str) or list(CROSS_ASSET_SYMBOLS):
            for interval in intervals:
                path = fetch_binance_klines(cfg, market=cfg.data.market, interval=interval, start=start, end=end, force=args.force, symbol=symbol)
                print(f"cross-asset candles {cfg.data.market} {symbol} {interval}: {path}")
    if args.funding:
        print(f"funding: {fetch_funding_rates(cfg, start=start, end=end, force=args.force)}")
    if args.open_interest:
        print(f"open interest: {fetch_open_interest(cfg, start=start, end=end, force=args.force)}")
    if args.derivative_ratios:
        ratio_periods = _parse_csv(args.ratio_periods, str) or ["5m"]
        ratio_sources = _parse_csv(args.ratio_sources, str) or list(FUTURES_RATIO_SOURCES)
        for period in ratio_periods:
            for source in ratio_sources:
                print(f"derivative ratio {source} {period}: {fetch_futures_ratio(cfg, source=source, period=period, start=start, end=end, force=args.force)}")
    if args.basis_klines:
        basis_intervals = _parse_csv(args.basis_intervals, str) or intervals
        basis_sources = _parse_csv(args.basis_sources, str) or list(FUTURES_BASIS_SOURCES)
        for interval in basis_intervals:
            for source in basis_sources:
                print(f"basis kline {source} {interval}: {fetch_futures_basis_klines(cfg, source=source, interval=interval, start=start, end=end, force=args.force)}")
    if args.hyperliquid:
        for interval in _parse_csv(args.hyperliquid_intervals, str) or ["5m"]:
            print(f"hyperliquid {interval}: {fetch_hyperliquid_candles(cfg, interval=interval, start=start, end=end, force=args.force)}")
    if args.coinbase:
        for interval in _parse_csv(args.coinbase_intervals, str) or ["5m"]:
            print(f"coinbase {interval}: {fetch_coinbase_candles(cfg, interval=interval, start=start, end=end, force=args.force)}")
    if args.bybit:
        bybit_intervals = _parse_csv(args.bybit_intervals, str) or ["5m"]
        for interval in bybit_intervals:
            print(f"bybit {interval}: {fetch_bybit_klines(cfg, interval=interval, start=start, end=end, force=args.force)}")
        for interval in _parse_csv(args.bybit_oi_intervals, str) or ["5m"]:
            print(f"bybit open_interest {interval}: {fetch_bybit_open_interest(cfg, interval=interval, start=start, end=end, force=args.force)}")
        print(f"bybit funding: {fetch_bybit_funding_history(cfg, start=start, end=end, force=args.force)}")
    if args.okx:
        for interval in _parse_csv(args.okx_intervals, str) or ["5m"]:
            print(f"okx {interval}: {fetch_okx_klines(cfg, interval=interval, start=start, end=end, force=args.force)}")
        if args.okx_derivatives:
            for period in _parse_csv(args.okx_periods, str) or ["5m"]:
                print(f"okx open_interest {period}: {fetch_okx_open_interest_history(cfg, period=period, start=start, end=end, force=args.force)}")
                print(f"okx long_short_ratio {period}: {fetch_okx_long_short_ratio_history(cfg, period=period, start=start, end=end, force=args.force)}")
    if args.agg_trades:
        trade_start = parse_datetime(args.agg_trade_start, end - timedelta(days=args.agg_trade_days))
        for market in markets:
            print(
                f"agg trades {market}: "
                f"{fetch_binance_agg_trades(cfg, market=market, start=trade_start, end=end, force=args.force, max_batches=args.agg_trade_max_batches, checkpoint_rows=args.agg_trade_checkpoint_rows, max_seconds=args.agg_trade_max_seconds)}"
            )
    if args.agg_trade_archive:
        trade_start = parse_datetime(args.agg_trade_start, end - timedelta(days=args.agg_trade_days))
        feature_timeframes = _parse_csv(args.agg_trade_feature_timeframes, str) or intervals
        for market in markets:
            print(
                f"agg trade archive {market}: "
                f"{fetch_binance_vision_agg_trades(cfg, market=market, start=trade_start, end=end, force=args.force, max_files=args.agg_trade_archive_max_files, checkpoint_files=args.agg_trade_archive_checkpoint_files, feature_timeframes=feature_timeframes, cache_raw=args.agg_trade_archive_cache_raw)}"
            )
    if args.book_depth_archive:
        depth_start = parse_datetime(args.book_depth_start, end - timedelta(days=args.book_depth_days))
        for market in markets:
            print(
                f"book depth archive {market}: "
                f"{fetch_binance_vision_book_depth(cfg, market=market, start=depth_start, end=end, force=args.force, max_files=args.book_depth_archive_max_files, checkpoint_files=args.book_depth_archive_checkpoint_files, feature_timeframe=args.book_depth_feature_timeframe)}"
            )
    if args.metrics_archive:
        metrics_start = parse_datetime(args.metrics_start, end - timedelta(days=args.metrics_days))
        for market in markets:
            print(
                f"metrics archive {market}: "
                f"{fetch_binance_vision_metrics(cfg, market=market, start=metrics_start, end=end, force=args.force, max_files=args.metrics_archive_max_files, checkpoint_files=args.metrics_archive_checkpoint_files, timeframe=args.metrics_feature_timeframe)}"
            )
    if args.live_flow:
        print(f"live flow futures: {collect_binance_live_flow(cfg, market='futures', seconds=args.live_flow_seconds)}")
    if args.order_book:
        print(f"order book futures: {fetch_binance_order_book_snapshot(cfg, market='futures', limit=args.order_book_limit)}")
    return 0


def cmd_capture_flow(args: argparse.Namespace) -> int:
    cfg = _load(args)
    chunk = max(1.0, float(args.chunk_seconds))
    sleep_seconds = max(0.0, float(args.sleep_seconds))
    max_chunks = args.max_chunks
    chunk_idx = 0
    try:
        while max_chunks is None or chunk_idx < max_chunks:
            chunk_idx += 1
            started = utc_now()
            path = collect_binance_live_flow(cfg, market="futures", seconds=chunk)
            line = f"chunk={chunk_idx} started={started.isoformat()} live_flow={path}"
            if args.order_book:
                book_path = fetch_binance_order_book_snapshot(cfg, market="futures", limit=args.order_book_limit)
                line += f" order_book={book_path}"
            print(line)
            if max_chunks is not None and chunk_idx >= max_chunks:
                break
            if sleep_seconds:
                time.sleep(sleep_seconds)
    except KeyboardInterrupt:
        print("capture_flow_stopped")
    return 0


def _csv_status(path: Path, time_columns: tuple[str, ...]) -> dict[str, object]:
    row: dict[str, object] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return row
    try:
        df = pd.read_csv(path, usecols=lambda col: col in time_columns)
        row["rows"] = len(df)
        for col in time_columns:
            if col in df.columns and not df.empty:
                row["start"] = str(df[col].iloc[0])
                row["end"] = str(df[col].iloc[-1])
                break
        row["bytes"] = path.stat().st_size
    except Exception as exc:  # noqa: BLE001 - diagnostic command
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def cmd_data_status(args: argparse.Namespace) -> int:
    cfg = _load(args)
    rows: list[dict[str, object]] = []
    intervals = _parse_csv(args.intervals, str) or ["1m", "5m"]
    markets = _parse_csv(args.markets, str) or ["futures", "spot"]
    for market in markets:
        for interval in intervals:
            item = _csv_status(raw_kline_path(cfg, market, interval), ("open_time",))
            item.update({"kind": "candles", "market": market, "interval": interval})
            rows.append(item)
        agg = _csv_status(raw_agg_trade_path(cfg, market), ("trade_time",))
        agg.update({"kind": "agg_trades", "market": market, "interval": None})
        rows.append(agg)
        for interval in intervals:
            agg_features = _csv_status(raw_agg_trade_feature_path(cfg, market, interval), ("open_time",))
            agg_features.update({"kind": "agg_trade_features", "market": market, "interval": interval})
            rows.append(agg_features)
            depth_features = _csv_status(raw_book_depth_feature_path(cfg, market, interval), ("open_time",))
            depth_features.update({"kind": "book_depth_features", "market": market, "interval": interval})
            rows.append(depth_features)
            metrics_features = _csv_status(raw_metrics_feature_path(cfg, market, interval), ("open_time",))
            metrics_features.update({"kind": "metrics_features", "market": market, "interval": interval})
            rows.append(metrics_features)
        live_flow = _csv_status(raw_live_flow_path(cfg, market), ("trade_time",))
        live_flow.update({"kind": "live_flow", "market": market, "interval": None})
        rows.append(live_flow)
        order_book = _csv_status(raw_order_book_path(cfg, market), ("open_time",))
        order_book.update({"kind": "order_book", "market": market, "interval": None})
        rows.append(order_book)
    extras = [
        ("funding", cfg.data.raw_dir / f"binance_futures_{cfg.data.symbol}_funding.csv", ("funding_time",)),
        ("open_interest", cfg.data.raw_dir / f"binance_futures_{cfg.data.symbol}_open_interest_5m.csv", ("open_time",)),
        ("hyperliquid", cfg.data.raw_dir / "hyperliquid_BTC_5m.csv", ("open_time",)),
        ("experiments", cfg.data.experiment_db, ()),
        ("paper", cfg.data.paper_db, ()),
    ]
    for interval in intervals:
        extras.append((f"coinbase_{interval}", coinbase_path(cfg, interval), ("open_time",)))
        extras.append((f"bybit_{interval}", bybit_kline_path(cfg, interval), ("open_time",)))
        extras.append((f"okx_{interval}", okx_kline_path(cfg, interval), ("open_time",)))
        for symbol in CROSS_ASSET_SYMBOLS:
            extras.append((f"cross_asset_{symbol}_{interval}", raw_symbol_kline_path(cfg, cfg.data.market, symbol, interval), ("open_time",)))
    for source in FUTURES_RATIO_SOURCES:
        extras.append((source, raw_futures_ratio_path(cfg, source, "5m"), ("open_time",)))
    for source in FUTURES_BASIS_SOURCES:
        extras.append((source, raw_futures_basis_path(cfg, source, "5m"), ("open_time",)))
    extras.append(("bybit_open_interest", bybit_open_interest_path(cfg, "5m"), ("open_time",)))
    extras.append(("bybit_funding", bybit_funding_path(cfg), ("funding_time",)))
    extras.append(("okx_open_interest", okx_open_interest_path(cfg, "5m"), ("open_time",)))
    extras.append(("okx_long_short_ratio", okx_long_short_ratio_path(cfg, "5m"), ("open_time",)))
    for kind, path, cols in extras:
        item = _csv_status(path, cols) if cols else {"path": str(path), "exists": path.exists(), "bytes": path.stat().st_size if path.exists() else None}
        item.update({"kind": kind, "market": None, "interval": None})
        rows.append(item)
    if args.json:
        print(json.dumps(rows, indent=2, default=str))
    else:
        headers = ["Kind", "Market", "Interval", "Rows", "Start", "End", "Bytes", "Path"]
        table = []
        for row in rows:
            table.append(
                [
                    str(row.get("kind", "")),
                    "" if row.get("market") is None else str(row.get("market")),
                    "" if row.get("interval") is None else str(row.get("interval")),
                    "n/a" if row.get("rows") is None else str(row.get("rows")),
                    str(row.get("start", "")),
                    str(row.get("end", "")),
                    "n/a" if row.get("bytes") is None else str(row.get("bytes")),
                    str(row.get("path", "")),
                ]
            )
        widths = [max(len(headers[i]), *(len(r[i]) for r in table)) for i in range(len(headers))]
        print("  ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
        print("  ".join("-" * widths[i] for i in range(len(headers))))
        for row in table:
            print("  ".join(row[i].ljust(widths[i]) for i in range(len(headers))))
    return 0


def _print_task_summary(tasks) -> None:
    for task in tasks:
        best = task.best_candidate
        metrics = best.metrics if best and best.metrics else {}
        acc = metrics.get("accuracy")
        hc = metrics.get("high_confidence_accuracy")
        cov = metrics.get("high_confidence_coverage")
        best_name = best.name if best else "none"
        print(
            f"{task.timeframe:>4} h{task.horizon:<2} best={best_name:<24} "
            f"accepted={str(task.accepted):<5} acc={acc if acc is not None else 'n/a'} "
            f"hc_acc={hc if hc is not None else 'n/a'} hc_cov={cov if cov is not None else 'n/a'}"
        )
        print(f"     {task.acceptance_reason}")


def cmd_train(args: argparse.Namespace) -> int:
    cfg = _load(args)
    command = getattr(args, "registry_command", args.command)
    run_id = None
    partial_tasks = []
    if not getattr(args, "no_registry", False):
        run_hash = config_digest(cfg, extra={"command": command, "profile": None})
        run_id = start_run(cfg, command=command, profile=None, config_hash=run_hash)
        print(f"run_id:      {run_id}")
    def checkpoint(tasks) -> None:
        nonlocal partial_tasks
        partial_tasks = list(tasks)
        if run_id is not None:
            finish_run(
                cfg,
                run_id=run_id,
                tasks=partial_tasks,
                status="running",
                notes="partial task checkpoint",
            )
    try:
        tasks = run_walk_forward(cfg, on_task=checkpoint if run_id is not None else None)
    except Exception as exc:  # noqa: BLE001 - preserve failed run state
        if run_id is not None:
            mark_run_failed(cfg, run_id=run_id, error=f"{type(exc).__name__}: {exc}")
        raise
    _print_task_summary(tasks)
    json_path, md_path, pred_path = write_reports(cfg, tasks, stem=args.report_stem)
    print(f"report json: {json_path}")
    print(f"report md:   {md_path}")
    print(f"predictions: {pred_path}")
    artifact_path = None
    if not args.no_save:
        artifact_path = Path(args.artifact) if args.artifact else None
        saved = save_artifact(cfg, tasks, path=artifact_path)
        artifact_path = saved
        print(f"artifact:    {saved}")
    if run_id is not None:
        finish_run(
            cfg,
            run_id=run_id,
            tasks=tasks,
            report_paths=(json_path, md_path, pred_path),
            artifact_path=artifact_path,
        )
        print(f"registry:    {cfg.data.experiment_db}")
    weak = [task for task in tasks if not task.accepted]
    if weak:
        print(f"weak_or_failed_tasks: {len(weak)} of {len(tasks)}")
        print(f"next_step: {next_escalation_step(tasks)}")
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    args.no_save = True
    args.registry_command = "backtest"
    return cmd_train(args)


SEARCH_PROFILES = {
    "default": {},
    "linear": {"candidates": ["logistic"]},
    "linear_variants": {"candidates": ["logistic", "logistic_l1", "logistic_elastic"]},
    "tree": {"candidates": ["hist_gradient_boosting", "extra_trees"]},
    "tree_variants": {
        "candidates": [
            "hist_gradient_boosting",
            "hist_gradient_boosting_fast",
            "hist_gradient_boosting_deep",
            "extra_trees",
            "extra_trees_deep",
            "random_forest",
        ]
    },
    "boosting": {"candidates": ["lightgbm", "xgboost"]},
    "boosting_variants": {
        "candidates": [
            "lightgbm",
            "lightgbm_fast",
            "lightgbm_wide",
            "xgboost",
            "xgboost_fast",
            "xgboost_deep",
        ]
    },
    "neural": {"candidates": ["torch_mlp", "torch_gru"]},
    "neural_variants": {"candidates": ["torch_mlp", "torch_mlp_wide", "torch_gru", "torch_gru_wide"]},
    "gpu_heavy": {"candidates": ["xgboost", "xgboost_fast", "xgboost_deep", "torch_mlp", "torch_mlp_wide", "torch_gru"]},
    "ensembles": {"candidates": ["ensemble_fast", "ensemble_linear_tree", "ensemble_boosting"]},
    "diverse_ensemble": {"candidates": ["ensemble_diverse"]},
    "stacked": {"candidates": ["stacked_fast", "stacked_boosting", "stacked_gpu"]},
    "meta_confidence": {"candidates": ["meta_logistic", "meta_xgboost", "meta_lightgbm", "meta_ensemble"]},
    "meta_confidence_gpu": {"candidates": ["meta_xgboost", "meta_xgboost_deep", "meta_lightgbm"]},
    "recency_weighted": {
        "candidates": ["recency_logistic", "recency_hgb", "recency_lightgbm", "recency_xgboost", "recency_ensemble"],
    },
    "recency_weighted_gpu": {
        "candidates": ["recency_xgboost", "recency_xgboost_deep", "recency_lightgbm"],
    },
    "feature_selection": {
        "candidates": ["select_logistic", "select_hgb", "select_lightgbm", "select_xgboost"],
    },
    "feature_selection_gpu": {
        "candidates": ["select_xgboost", "select_lightgbm", "select_hgb"],
    },
    "balanced_gpu": {
        "candidates": ["balanced_xgboost_fast", "balanced_xgboost", "balanced_xgboost_deep", "balanced_lightgbm", "balanced_select_xgboost"],
    },
    "live_triggers": {
        "feature_groups": ["price_action", "technical", "flow", "derivatives", "cross_market", "calendar"],
        "candidates": ["trigger_confluence", "trigger_flow", "trigger_momentum", "trigger_reversal"],
    },
    "live_feed_triggers": {
        "feature_groups": ["price_action", "technical", "flow", "derivatives", "cross_market", "calendar"],
        "candidates": [
            "event_trigger_confluence",
            "event_trigger_flow",
            "event_trigger_liquidation",
            "event_trigger_orderbook",
            "event_trigger_cross_venue",
            "event_trigger_impulse",
            "event_trigger_reversal",
        ],
    },
    "live_feed_gated": {
        "feature_groups": ["price_action", "technical", "flow", "derivatives", "cross_market", "calendar"],
        "candidates": [
            "gated_event_flow",
            "gated_event_confluence",
            "gated_event_liquidation",
            "gated_event_ensemble",
        ],
    },
    "five_min_ltf": {
        "timeframes": ["5m"],
        "horizons": [1, 2, 3],
        "feature_groups": ["price_action", "technical", "flow", "derivatives", "cross_market", "context", "calendar"],
        "candidates": [
            "gated_event_flow",
            "gated_event_liquidation",
            "gated_event_confluence",
            "xgboost_search_fast",
            "xgboost_search_gpu",
            "select_xgboost",
            "balanced_xgboost_fast",
            "adaptive_ensemble_gpu",
        ],
    },
    "five_min_ltf_volatility": {
        "timeframes": ["5m"],
        "horizons": [1, 2, 3],
        "label_mode": "volatility_adjusted_050",
        "feature_groups": ["price_action", "technical", "flow", "derivatives", "cross_market", "context", "calendar"],
        "candidates": [
            "xgboost_fast",
            "xgboost_deep",
            "balanced_xgboost_fast",
            "balanced_xgboost",
            "select_xgboost",
            "gated_event_flow",
            "gated_event_liquidation",
        ],
    },
    "five_min_ltf_signals": {
        "timeframes": ["5m"],
        "horizons": [1, 2, 3],
        "feature_groups": ["price_action", "technical", "flow", "derivatives", "cross_market", "context", "calendar"],
        "candidates": [
            "univariate_signal",
            "univariate_signal_flow",
            "univariate_signal_conservative",
            "state_flow",
            "state_event",
            "state_vol_flow",
            "time_window_selector",
            "temporal_selector_gpu",
        ],
    },
    "five_min_body_gpu": {
        "timeframes": ["5m"],
        "horizons": [1, 2, 3],
        "label_mode": "target_body",
        "feature_groups": ["price_action", "technical", "flow", "derivatives", "cross_market", "context", "calendar"],
        "candidates": ["xgboost_search_fast", "xgboost_fast", "xgboost_deep", "balanced_xgboost_fast", "select_xgboost", "gated_event_flow"],
    },
    "five_min_next_open_gpu": {
        "timeframes": ["5m"],
        "horizons": [1, 2, 3],
        "label_mode": "next_open_to_future_close",
        "feature_groups": ["price_action", "technical", "flow", "derivatives", "cross_market", "context", "calendar"],
        "candidates": ["xgboost_search_fast", "xgboost_fast", "xgboost_deep", "balanced_xgboost_fast", "select_xgboost", "gated_event_flow"],
    },
    "five_min_long_horizons": {
        "timeframes": ["5m"],
        "horizons": [2, 3, 6, 12],
        "feature_groups": ["price_action", "technical", "flow", "derivatives", "cross_market", "context", "calendar"],
        "candidates": ["xgboost_search_fast", "xgboost_search_gpu", "xgboost_deep", "balanced_xgboost_fast", "temporal_selector_gpu", "adaptive_ensemble_gpu"],
    },
    "five_min_recent_ltf": {
        "timeframes": ["5m"],
        "horizons": [1, 2, 3],
        "feature_groups": ["price_action", "technical", "flow", "derivatives", "cross_market", "context", "calendar"],
        "candidates": [
            "gated_event_flow",
            "gated_event_liquidation",
            "xgboost_search_fast",
            "xgboost_fast",
            "balanced_xgboost_fast",
            "recency_xgboost",
            "temporal_selector_gpu",
        ],
        "max_rows_per_task": 12000,
        "splits": 6,
        "test_size": 1200,
        "min_train_size": 3600,
    },
    "five_min_recent_volatility": {
        "timeframes": ["5m"],
        "horizons": [1, 2, 3],
        "label_mode": "volatility_adjusted_025",
        "feature_groups": ["price_action", "technical", "flow", "derivatives", "cross_market", "context", "calendar"],
        "candidates": [
            "xgboost_search_fast",
            "xgboost_fast",
            "xgboost_deep",
            "balanced_xgboost_fast",
            "recency_xgboost",
            "gated_event_flow",
        ],
        "max_rows_per_task": 12000,
        "splits": 6,
        "test_size": 1200,
        "min_train_size": 3600,
    },
    "five_min_recent_volatility_rolling": {
        "timeframes": ["5m"],
        "horizons": [2, 3, 4, 6],
        "label_mode": "volatility_adjusted_025",
        "feature_groups": ["price_action", "technical", "flow", "derivatives", "cross_market", "context", "calendar"],
        "candidates": [
            "xgboost_search_gpu",
            "xgboost_search_fast",
            "xgboost_fast",
            "xgboost_deep",
            "balanced_xgboost_fast",
            "balanced_xgboost_deep",
            "temporal_selector_gpu",
            "adaptive_ensemble_gpu",
            "meta_xgboost",
        ],
        "max_rows_per_task": 18000,
        "splits": 8,
        "test_size": 1200,
        "min_train_size": 4800,
        "max_train_size": 9000,
    },
    "five_min_recent_volatility_short": {
        "timeframes": ["5m"],
        "horizons": [2, 3],
        "label_mode": "volatility_adjusted_025",
        "feature_groups": ["price_action", "technical", "flow", "derivatives", "cross_market", "context", "calendar"],
        "candidates": [
            "xgboost_search_gpu",
            "xgboost_search_fast",
            "xgboost_fast",
            "xgboost_deep",
            "balanced_xgboost_fast",
            "temporal_selector_gpu",
            "adaptive_ensemble_gpu",
            "meta_xgboost",
        ],
        "max_rows_per_task": 8000,
        "splits": 8,
        "test_size": 500,
        "min_train_size": 2500,
        "max_train_size": 3500,
    },
    "five_min_ltf_sequence": {
        "timeframes": ["5m"],
        "horizons": [2, 3, 4],
        "feature_groups": ["price_action", "technical", "flow", "derivatives", "cross_market", "context", "calendar"],
        "candidates": [
            "xgboost_search_gpu",
            "xgboost_search_fast",
            "xgboost_fast",
            "balanced_xgboost_fast",
            "select_xgboost",
            "temporal_selector_gpu",
            "adaptive_ensemble_gpu",
            "gated_event_flow",
        ],
        "max_rows_per_task": 16000,
        "splits": 7,
        "test_size": 1000,
        "min_train_size": 4500,
        "max_train_size": 9000,
    },
    "five_min_time_corrected": {
        "timeframes": ["5m"],
        "horizons": [2, 3, 4, 6],
        "label_mode": "volatility_adjusted_025",
        "feature_groups": ["price_action", "technical", "flow", "derivatives", "cross_market", "context", "calendar"],
        "candidates": [
            "time_corrected_xgboost",
            "time_corrected_xgboost_search",
            "time_corrected_xgboost_weekday",
            "time_corrected_ensemble",
            "xgboost_search_gpu",
            "xgboost_fast",
            "adaptive_ensemble_gpu",
        ],
        "max_rows_per_task": 12000,
        "splits": 6,
        "test_size": 1200,
        "min_train_size": 3600,
    },
    "five_min_time_corrected_short": {
        "timeframes": ["5m"],
        "horizons": [2, 3],
        "label_mode": "volatility_adjusted_025",
        "feature_groups": ["price_action", "technical", "flow", "derivatives", "cross_market", "context", "calendar"],
        "candidates": [
            "time_corrected_xgboost",
            "time_corrected_xgboost_search",
            "time_corrected_xgboost_weekday",
            "xgboost_search_gpu",
            "xgboost_fast",
        ],
        "max_rows_per_task": 8000,
        "splits": 8,
        "test_size": 500,
        "min_train_size": 2500,
        "max_train_size": 3500,
    },
    "five_min_status_context": {
        "timeframes": ["5m"],
        "horizons": [2, 3, 4],
        "label_mode": "volatility_adjusted_025",
        "feature_groups": ["price_action", "technical", "flow", "derivatives", "cross_market", "context", "calendar"],
        "candidates": [
            "xgboost_search_gpu",
            "xgboost_search_fast",
            "xgboost_fast",
            "xgboost_deep",
            "balanced_xgboost_fast",
            "time_corrected_xgboost",
            "time_corrected_xgboost_search",
            "temporal_selector_gpu",
            "adaptive_ensemble_gpu",
        ],
        "max_rows_per_task": 12000,
        "splits": 6,
        "test_size": 1200,
        "min_train_size": 3600,
    },
    "five_min_status_context_short": {
        "timeframes": ["5m"],
        "horizons": [2, 3],
        "label_mode": "volatility_adjusted_025",
        "feature_groups": ["price_action", "technical", "flow", "derivatives", "cross_market", "context", "calendar"],
        "candidates": [
            "xgboost_search_gpu",
            "xgboost_search_fast",
            "xgboost_fast",
            "time_corrected_xgboost",
            "time_corrected_xgboost_search",
            "adaptive_ensemble_gpu",
        ],
        "max_rows_per_task": 8000,
        "splits": 8,
        "test_size": 500,
        "min_train_size": 2500,
        "max_train_size": 3500,
    },
    "five_min_status_close": {
        "timeframes": ["5m"],
        "horizons": [1, 2, 3],
        "label_mode": "close_to_close",
        "feature_groups": ["price_action", "technical", "flow", "derivatives", "cross_market", "context", "calendar"],
        "candidates": [
            "xgboost_search_gpu",
            "xgboost_search_fast",
            "xgboost_fast",
            "balanced_xgboost_fast",
            "time_corrected_xgboost",
            "time_corrected_xgboost_search",
            "adaptive_ensemble_gpu",
        ],
        "max_rows_per_task": 12000,
        "splits": 6,
        "test_size": 1200,
        "min_train_size": 3600,
    },
    "five_min_threshold_3bps_recent": {
        "timeframes": ["5m"],
        "horizons": [1, 2, 3, 4],
        "label_mode": "close_to_close",
        "min_move_bps": 3.0,
        "feature_groups": ["price_action", "technical", "flow", "derivatives", "cross_market", "context", "calendar"],
        "candidates": [
            "xgboost_search_gpu",
            "xgboost_search_fast",
            "xgboost_fast",
            "xgboost_deep",
            "balanced_xgboost_fast",
            "time_corrected_xgboost",
            "time_corrected_xgboost_search",
            "adaptive_ensemble_gpu",
        ],
        "max_rows_per_task": 12000,
        "splits": 6,
        "test_size": 1200,
        "min_train_size": 3600,
    },
    "five_min_threshold_5bps_recent": {
        "timeframes": ["5m"],
        "horizons": [1, 2, 3, 4],
        "label_mode": "close_to_close",
        "min_move_bps": 5.0,
        "feature_groups": ["price_action", "technical", "flow", "derivatives", "cross_market", "context", "calendar"],
        "candidates": [
            "xgboost_search_gpu",
            "xgboost_search_fast",
            "xgboost_fast",
            "balanced_xgboost_fast",
            "time_corrected_xgboost",
            "adaptive_ensemble_gpu",
        ],
        "max_rows_per_task": 12000,
        "splits": 6,
        "test_size": 1000,
        "min_train_size": 3000,
    },
    "five_min_volatility_strict_recent": {
        "timeframes": ["5m"],
        "horizons": [2, 3, 4, 6],
        "label_mode": "volatility_adjusted_075",
        "feature_groups": ["price_action", "technical", "flow", "derivatives", "cross_market", "context", "calendar"],
        "candidates": [
            "xgboost_search_gpu",
            "xgboost_search_fast",
            "xgboost_fast",
            "xgboost_deep",
            "balanced_xgboost_fast",
            "time_corrected_xgboost",
            "time_corrected_xgboost_search",
        ],
        "max_rows_per_task": 12000,
        "splits": 6,
        "test_size": 900,
        "min_train_size": 3000,
    },
    "five_min_archive_flow_close": {
        "timeframes": ["5m"],
        "horizons": [1, 2, 3, 4],
        "label_mode": "close_to_close",
        "feature_groups": ["price_action", "technical", "flow", "derivatives", "cross_market", "context", "calendar"],
        "candidates": [
            "xgboost_search_gpu",
            "xgboost_search_fast",
            "xgboost_deep",
            "balanced_xgboost_deep",
            "recency_xgboost_deep",
            "time_corrected_xgboost_search",
            "adaptive_ensemble_gpu",
        ],
        "max_rows_per_task": 18000,
        "splits": 5,
        "test_size": 1800,
        "min_train_size": 6000,
        "max_train_size": 12000,
    },
    "five_min_archive_flow_3bps": {
        "timeframes": ["5m"],
        "horizons": [1, 2, 3, 4],
        "label_mode": "close_to_close",
        "min_move_bps": 3.0,
        "feature_groups": ["price_action", "technical", "flow", "derivatives", "cross_market", "context", "calendar"],
        "candidates": [
            "xgboost_search_gpu",
            "xgboost_search_fast",
            "xgboost_deep",
            "balanced_xgboost_deep",
            "recency_xgboost_deep",
            "time_corrected_xgboost_search",
            "adaptive_ensemble_gpu",
        ],
        "max_rows_per_task": 18000,
        "splits": 5,
        "test_size": 1800,
        "min_train_size": 6000,
        "max_train_size": 12000,
    },
    "five_min_archive_flow_volatility": {
        "timeframes": ["5m"],
        "horizons": [2, 3, 4, 6],
        "label_mode": "volatility_adjusted_025",
        "feature_groups": ["price_action", "technical", "flow", "derivatives", "cross_market", "context", "calendar"],
        "candidates": [
            "xgboost_search_gpu",
            "xgboost_search_fast",
            "xgboost_deep",
            "balanced_xgboost_deep",
            "recency_xgboost_deep",
            "time_corrected_xgboost_search",
            "adaptive_ensemble_gpu",
        ],
        "max_rows_per_task": 18000,
        "splits": 5,
        "test_size": 1800,
        "min_train_size": 6000,
        "max_train_size": 12000,
    },
    "five_min_cross_asset_flow": {
        "timeframes": ["5m"],
        "horizons": [2, 3, 4, 6],
        "label_mode": "close_to_close",
        "feature_groups": ["price_action", "technical", "flow", "derivatives", "cross_market", "context", "calendar"],
        "candidates": [
            "xgboost_search_gpu",
            "xgboost_search_fast",
            "xgboost_deep",
            "balanced_xgboost_deep",
            "recency_xgboost_deep",
            "time_corrected_xgboost_search",
            "adaptive_ensemble_gpu",
        ],
        "max_rows_per_task": 18000,
        "splits": 5,
        "test_size": 1800,
        "min_train_size": 6000,
        "max_train_size": 12000,
    },
    "five_min_cross_asset_flow_volatility": {
        "timeframes": ["5m"],
        "horizons": [2, 3, 4, 6],
        "label_mode": "volatility_adjusted_025",
        "feature_groups": ["price_action", "technical", "flow", "derivatives", "cross_market", "context", "calendar"],
        "candidates": [
            "xgboost_search_gpu",
            "xgboost_search_fast",
            "xgboost_deep",
            "balanced_xgboost_deep",
            "recency_xgboost_deep",
            "time_corrected_xgboost_search",
            "adaptive_ensemble_gpu",
        ],
        "max_rows_per_task": 18000,
        "splits": 5,
        "test_size": 1800,
        "min_train_size": 6000,
        "max_train_size": 12000,
    },
    "five_min_metrics_flow": {
        "timeframes": ["5m"],
        "horizons": [2, 3, 4, 6],
        "label_mode": "close_to_close",
        "feature_groups": ["price_action", "technical", "flow", "derivatives", "cross_market", "context", "calendar"],
        "candidates": [
            "xgboost_search_gpu",
            "xgboost_search_fast",
            "xgboost_deep",
            "balanced_xgboost_deep",
            "recency_xgboost_deep",
            "time_corrected_xgboost_search",
            "adaptive_trigger_ensemble",
            "univariate_signal_derivatives",
            "state_derivatives",
            "state_vol_flow",
        ],
        "max_rows_per_task": 18000,
        "splits": 5,
        "test_size": 1800,
        "min_train_size": 6000,
        "max_train_size": 12000,
    },
    "five_min_metrics_flow_short": {
        "timeframes": ["5m"],
        "horizons": [2, 3, 4],
        "label_mode": "close_to_close",
        "feature_groups": ["price_action", "technical", "flow", "derivatives", "cross_market", "context", "calendar"],
        "candidates": [
            "xgboost_search_gpu",
            "xgboost_search_fast",
            "balanced_xgboost_deep",
            "recency_xgboost_deep",
            "time_corrected_xgboost_search",
            "adaptive_trigger_ensemble",
            "state_derivatives",
        ],
        "max_rows_per_task": 9000,
        "splits": 7,
        "test_size": 800,
        "min_train_size": 2800,
        "max_train_size": 4200,
    },
    "five_min_threshold_optimized": {
        "timeframes": ["5m"],
        "horizons": [1, 2, 3, 4, 6],
        "label_mode": "close_to_close",
        "feature_groups": ["price_action", "technical", "flow", "derivatives", "cross_market", "context", "calendar"],
        "candidates": [
            "threshold_xgboost_fast",
            "threshold_xgboost_search",
            "threshold_xgboost_deep",
            "threshold_adaptive_ensemble",
            "xgboost_search_fast",
            "balanced_xgboost_deep",
        ],
        "max_rows_per_task": 18000,
        "splits": 5,
        "test_size": 1800,
        "min_train_size": 6000,
        "max_train_size": 12000,
    },
    "five_min_depth_signal_models": {
        "timeframes": ["5m"],
        "horizons": [2, 3, 4, 6],
        "label_mode": "close_to_close",
        "feature_groups": ["price_action", "technical", "flow", "derivatives", "cross_market", "context", "calendar"],
        "candidates": [
            "univariate_signal",
            "univariate_signal_flow",
            "univariate_signal_cross_venue",
            "state_flow",
            "state_cross_venue",
            "state_vol_flow",
            "event_trigger_flow",
            "event_trigger_cross_venue",
            "gated_event_flow",
            "gated_event_ensemble",
        ],
        "max_rows_per_task": 18000,
        "splits": 5,
        "test_size": 1800,
        "min_train_size": 6000,
        "max_train_size": 12000,
    },
    "five_min_full_neural": {
        "timeframes": ["5m"],
        "horizons": [2, 3, 4, 6],
        "label_mode": "close_to_close",
        "feature_groups": ["price_action", "technical", "flow", "derivatives", "cross_market", "context", "calendar"],
        "candidates": [
            "torch_mlp",
            "torch_mlp_wide",
            "torch_gru",
            "torch_gru_wide",
            "xgboost_search_gpu",
            "adaptive_ensemble_gpu",
        ],
        "max_rows_per_task": 18000,
        "splits": 5,
        "test_size": 1800,
        "min_train_size": 6000,
        "max_train_size": 12000,
    },
    "calendar_patterns": {
        "feature_groups": ["calendar"],
        "candidates": ["calendar_hourly", "calendar_quarter_hour", "calendar_daypart"],
    },
    "time_window_selectors": {
        "feature_groups": ["calendar"],
        "candidates": ["time_window_selector", "time_window_selector_15m", "time_window_selector_strict"],
    },
    "xgboost_search_gpu": {
        "candidates": ["xgboost_search_fast", "xgboost_search_gpu", "xgboost_search_conservative"],
    },
    "temporal_selector": {
        "candidates": ["temporal_selector_fast", "temporal_selector_diverse"],
    },
    "temporal_selector_gpu": {
        "candidates": ["temporal_selector_gpu", "temporal_selector_fast"],
    },
    "adaptive_ensembles": {
        "feature_groups": ["price_action", "technical", "flow", "derivatives", "cross_market", "calendar"],
        "candidates": ["adaptive_ensemble_fast", "adaptive_ensemble_gpu", "adaptive_trigger_ensemble"],
    },
    "event_adaptive": {
        "feature_groups": ["price_action", "technical", "flow", "derivatives", "cross_market", "calendar"],
        "candidates": [
            "adaptive_trigger_ensemble",
            "event_trigger_confluence",
            "event_trigger_flow",
            "event_trigger_liquidation",
            "event_trigger_orderbook",
            "trigger_confluence",
            "trigger_flow",
            "trigger_reversal",
        ],
    },
    "feature_state_models": {
        "feature_groups": ["price_action", "technical", "flow", "derivatives", "cross_market", "calendar"],
        "candidates": ["state_flow", "state_derivatives", "state_cross_venue", "state_vol_flow", "state_event"],
    },
    "univariate_signals": {
        "feature_groups": ["price_action", "technical", "flow", "derivatives", "cross_market", "calendar"],
        "candidates": [
            "univariate_signal",
            "univariate_signal_flow",
            "univariate_signal_derivatives",
            "univariate_signal_cross_venue",
            "univariate_signal_conservative",
        ],
    },
    "regime_models": {"candidates": ["regime_logistic", "regime_hgb", "regime_lightgbm", "regime_ensemble"]},
    "price_only": {
        "include_spot": False,
        "include_open_interest": False,
        "include_funding": False,
        "include_derivative_ratios": False,
        "include_basis_klines": False,
        "include_coinbase": False,
        "include_okx": False,
    },
    "price_action_features": {"feature_groups": ["price_action", "calendar"]},
    "technical_features": {"feature_groups": ["price_action", "technical", "calendar"]},
    "flow_features": {"feature_groups": ["price_action", "flow", "derivatives", "cross_market", "calendar"]},
    "derivative_ratio_features": {"feature_groups": ["price_action", "technical", "derivatives", "calendar"]},
    "recent_derivatives": {
        "timeframes": ["5m", "15m", "30m"],
        "horizons": [1, 2, 3],
        "feature_groups": ["price_action", "technical", "flow", "derivatives", "calendar"],
        "candidates": ["xgboost_fast", "xgboost", "lightgbm_fast", "hist_gradient_boosting_fast", "logistic"],
        "max_rows_per_task": 9000,
    },
    "recent_derivatives_gpu": {
        "timeframes": ["5m", "15m"],
        "horizons": [1, 2, 3],
        "feature_groups": ["price_action", "technical", "flow", "derivatives", "calendar"],
        "candidates": ["xgboost_fast", "xgboost", "xgboost_deep", "torch_mlp", "torch_gru"],
        "max_rows_per_task": 9000,
    },
    "context_features": {"feature_groups": ["price_action", "technical", "context", "calendar"]},
    "context_flow_features": {"feature_groups": ["price_action", "technical", "context", "flow", "derivatives", "cross_market", "calendar"]},
    "cross_venue_features": {
        "feature_groups": ["price_action", "technical", "cross_market", "calendar"],
        "candidates": ["logistic", "hist_gradient_boosting_fast", "extra_trees", "lightgbm_fast", "xgboost_fast"],
    },
    "cross_venue_gpu": {
        "feature_groups": ["price_action", "technical", "cross_market", "calendar"],
        "candidates": ["xgboost_fast", "xgboost", "xgboost_deep", "torch_mlp"],
    },
    "bybit_features": {
        "feature_groups": ["price_action", "technical", "cross_market", "derivatives", "calendar"],
        "candidates": ["logistic", "hist_gradient_boosting_fast", "extra_trees", "lightgbm_fast", "xgboost_fast"],
    },
    "bybit_gpu": {
        "feature_groups": ["price_action", "technical", "cross_market", "derivatives", "calendar"],
        "candidates": ["xgboost_fast", "xgboost", "xgboost_deep", "balanced_xgboost_fast", "xgboost_search_fast", "torch_mlp"],
    },
    "bybit_divergence_gpu": {
        "feature_groups": ["price_action", "technical", "cross_market", "derivatives", "calendar"],
        "candidates": ["xgboost_search_fast", "xgboost_fast", "xgboost", "balanced_xgboost_fast", "xgboost_deep", "torch_mlp"],
    },
    "okx_cross_venue_gpu": {
        "feature_groups": ["price_action", "technical", "cross_market", "derivatives", "calendar"],
        "candidates": ["xgboost_search_fast", "xgboost_fast", "xgboost", "balanced_xgboost_fast", "lightgbm_fast", "torch_mlp"],
    },
    "all": {},
    "wide_timeframes": {
        "timeframes": ["1m", "3m", "5m", "15m", "30m", "1h", "4h"],
        "horizons": [1, 2, 3],
    },
    "body_label": {"label_mode": "target_body"},
    "next_open_label": {"label_mode": "next_open_to_future_close"},
    "volatility_label": {"label_mode": "volatility_adjusted"},
    "volatility_label_gpu": {
        "label_mode": "volatility_adjusted",
        "feature_groups": ["price_action", "technical", "flow", "derivatives", "cross_market", "calendar"],
        "candidates": ["xgboost_fast", "xgboost", "xgboost_deep", "balanced_xgboost_fast", "balanced_xgboost", "torch_mlp", "torch_gru"],
        "max_rows_per_task": 30000,
    },
    "volatility_label_cross_venue": {
        "label_mode": "volatility_adjusted_025",
        "feature_groups": ["price_action", "technical", "cross_market", "derivatives", "calendar"],
        "candidates": ["xgboost_fast", "xgboost", "xgboost_deep", "lightgbm_fast", "torch_mlp"],
        "max_rows_per_task": 30000,
    },
    "volatility_label_strict_gpu": {
        "label_mode": "volatility_adjusted_050",
        "feature_groups": ["price_action", "technical", "flow", "derivatives", "cross_market", "calendar"],
        "candidates": ["xgboost_fast", "xgboost", "xgboost_deep", "balanced_xgboost_deep", "balanced_select_xgboost", "torch_mlp_wide", "torch_gru"],
        "max_rows_per_task": 30000,
    },
    "threshold_1bps": {"min_move_bps": 1.0},
    "threshold_2bps": {"min_move_bps": 2.0},
    "confidence_65": {"high_confidence_threshold": 0.65},
    "confidence_70": {"high_confidence_threshold": 0.70},
    "long_horizons": {"horizons": [2, 3, 6, 12]},
    "larger_walkforward": {"splits": 7, "test_size": 6000, "min_train_size": 12000},
    "full_high_compute": {
        "timeframes": ["1m", "3m", "5m", "15m", "30m", "1h", "4h"],
        "horizons": [1, 2, 3, 6],
        "candidates": [
            "hist_gradient_boosting",
            "hist_gradient_boosting_deep",
            "extra_trees",
            "extra_trees_deep",
            "lightgbm",
            "lightgbm_wide",
            "xgboost",
            "xgboost_deep",
            "torch_mlp",
            "logistic",
            "ensemble_fast",
            "ensemble_boosting",
            "meta_xgboost",
            "meta_ensemble",
            "recency_xgboost",
            "recency_lightgbm",
            "select_xgboost",
            "select_lightgbm",
            "temporal_selector_gpu",
            "temporal_selector_diverse",
        ],
        "max_rows_per_task": None,
    },
}

AUTO_PROFILES = [
    "boosting",
    "gpu_heavy",
    "volatility_label_gpu",
    "volatility_label_cross_venue",
    "flow_features",
    "derivative_ratio_features",
    "cross_venue_gpu",
    "bybit_gpu",
    "bybit_divergence_gpu",
    "okx_cross_venue_gpu",
    "recent_derivatives_gpu",
    "recency_weighted_gpu",
    "feature_selection_gpu",
    "balanced_gpu",
    "live_feed_triggers",
    "live_feed_gated",
    "five_min_ltf",
    "five_min_ltf_volatility",
    "five_min_ltf_signals",
    "five_min_body_gpu",
    "five_min_next_open_gpu",
    "five_min_long_horizons",
    "five_min_recent_ltf",
    "five_min_recent_volatility",
    "five_min_recent_volatility_rolling",
    "five_min_recent_volatility_short",
    "five_min_ltf_sequence",
    "five_min_time_corrected",
    "five_min_time_corrected_short",
    "five_min_status_context",
    "five_min_status_context_short",
    "five_min_status_close",
    "five_min_threshold_3bps_recent",
    "five_min_threshold_5bps_recent",
    "five_min_volatility_strict_recent",
    "five_min_archive_flow_close",
    "five_min_archive_flow_3bps",
    "five_min_archive_flow_volatility",
    "five_min_cross_asset_flow",
    "five_min_cross_asset_flow_volatility",
    "five_min_metrics_flow",
    "five_min_metrics_flow_short",
    "five_min_threshold_optimized",
    "five_min_depth_signal_models",
    "five_min_full_neural",
    "live_triggers",
    "calendar_patterns",
    "time_window_selectors",
    "xgboost_search_gpu",
    "temporal_selector_gpu",
    "adaptive_ensembles",
    "event_adaptive",
    "feature_state_models",
    "univariate_signals",
    "meta_confidence_gpu",
    "ensembles",
    "neural",
]
ESCALATION_PROFILES = [
    "linear_variants",
    "tree_variants",
    "boosting_variants",
    "regime_models",
    "meta_confidence",
    "meta_confidence_gpu",
    "recency_weighted",
    "recency_weighted_gpu",
    "feature_selection",
    "feature_selection_gpu",
    "balanced_gpu",
    "live_feed_triggers",
    "live_feed_gated",
    "five_min_ltf",
    "five_min_ltf_volatility",
    "five_min_ltf_signals",
    "five_min_body_gpu",
    "five_min_next_open_gpu",
    "five_min_long_horizons",
    "five_min_recent_ltf",
    "five_min_recent_volatility",
    "five_min_recent_volatility_rolling",
    "five_min_recent_volatility_short",
    "five_min_ltf_sequence",
    "five_min_time_corrected",
    "five_min_time_corrected_short",
    "five_min_status_context",
    "five_min_status_context_short",
    "five_min_status_close",
    "five_min_threshold_3bps_recent",
    "five_min_threshold_5bps_recent",
    "five_min_volatility_strict_recent",
    "five_min_archive_flow_close",
    "five_min_archive_flow_3bps",
    "five_min_archive_flow_volatility",
    "five_min_cross_asset_flow",
    "five_min_cross_asset_flow_volatility",
    "five_min_metrics_flow",
    "five_min_metrics_flow_short",
    "five_min_threshold_optimized",
    "five_min_depth_signal_models",
    "five_min_full_neural",
    "live_triggers",
    "calendar_patterns",
    "time_window_selectors",
    "xgboost_search_gpu",
    "temporal_selector",
    "temporal_selector_gpu",
    "adaptive_ensembles",
    "event_adaptive",
    "feature_state_models",
    "univariate_signals",
    "cross_venue_features",
    "cross_venue_gpu",
    "bybit_features",
    "bybit_gpu",
    "bybit_divergence_gpu",
    "okx_cross_venue_gpu",
    "technical_features",
    "flow_features",
    "derivative_ratio_features",
    "recent_derivatives",
    "recent_derivatives_gpu",
    "context_features",
    "context_flow_features",
    "wide_timeframes",
    "body_label",
    "next_open_label",
    "volatility_label",
    "volatility_label_gpu",
    "volatility_label_cross_venue",
    "volatility_label_strict_gpu",
    "threshold_1bps",
    "threshold_2bps",
    "confidence_65",
    "confidence_70",
    "long_horizons",
    "larger_walkforward",
    "neural_variants",
    "gpu_heavy",
    "diverse_ensemble",
    "stacked",
    "full_high_compute",
]


def _apply_profile(cfg: AppConfig, profile: str) -> AppConfig:
    overrides = SEARCH_PROFILES[profile]
    if "candidates" in overrides:
        cfg.models.candidates = list(overrides["candidates"])
    if "timeframes" in overrides:
        cfg.prediction.timeframes = list(overrides["timeframes"])
    if "horizons" in overrides:
        cfg.prediction.horizons = [int(x) for x in overrides["horizons"]]
    for key in (
        "label_mode",
        "min_move_bps",
        "high_confidence_threshold",
        "max_rows_per_task",
        "feature_groups",
    ):
        if key in overrides:
            setattr(cfg.prediction, key, overrides[key])
    for key in ("splits", "test_size", "min_train_size", "max_train_size", "calibration_fraction"):
        if key in overrides:
            setattr(cfg.walk_forward, key, overrides[key])
    for key in (
        "include_spot",
        "include_open_interest",
        "include_funding",
        "include_hyperliquid",
        "include_coinbase",
        "include_bybit",
        "include_okx",
        "include_agg_trades",
        "include_live_flow",
        "include_order_book",
        "include_derivative_ratios",
        "include_basis_klines",
    ):
        if key in overrides:
            setattr(cfg.data, key, bool(overrides[key]))
    return cfg


def cmd_search(args: argparse.Namespace) -> int:
    base_cfg = _load(args)
    profile_names = _parse_csv(args.profiles, str) or ["boosting", "gpu_heavy", "flow_features", "ensembles"]
    if "auto" in profile_names:
        profile_names = [name for name in AUTO_PROFILES if name not in profile_names] + [
            name for name in profile_names if name != "auto"
        ]
    if not args.no_escalate:
        profile_names.extend(name for name in ESCALATION_PROFILES if name not in profile_names)
    unknown = [name for name in profile_names if name not in SEARCH_PROFILES]
    if unknown:
        raise SystemExit(f"unknown search profiles: {', '.join(unknown)}")
    summary_rows = []
    best_profile = None
    best_score = float("-inf")
    best_tasks = None
    best_cfg = None
    best_run_id = None
    best_report_paths = None
    stem_base = args.report_stem or "btc_direction_search"
    executed = 0
    accepted_seen = False
    for profile in profile_names:
        if args.max_runs is not None and executed >= args.max_runs:
            print(f"\nmax_runs reached: {args.max_runs}")
            break
        cfg = _apply_profile(copy.deepcopy(base_cfg), profile)
        run_hash = config_digest(cfg, extra={"command": "search", "profile": profile})
        if args.resume:
            existing = completed_run_exists(cfg, command="search", profile=profile, config_hash=run_hash)
            if existing:
                print(f"\n=== profile: {profile} skipped existing run {existing} ===")
                continue
        print(f"\n=== profile: {profile} candidates={','.join(cfg.models.candidates)} ===")
        run_id = start_run(cfg, command="search", profile=profile, config_hash=run_hash)
        print(f"run_id: {run_id}")
        executed += 1
        partial_tasks = []
        def checkpoint(tasks) -> None:
            nonlocal partial_tasks
            partial_tasks = list(tasks)
            finish_run(
                cfg,
                run_id=run_id,
                tasks=partial_tasks,
                status="running",
                notes="partial task checkpoint",
            )
        try:
            tasks = run_walk_forward(cfg, on_task=checkpoint)
        except Exception as exc:  # noqa: BLE001 - unattended search should continue
            error = f"{type(exc).__name__}: {exc}"
            mark_run_failed(cfg, run_id=run_id, error=error)
            print(f"profile_failed: {profile}: {error}", file=sys.stderr)
            continue
        _print_task_summary(tasks)
        report_paths = write_reports(cfg, tasks, stem=f"{stem_base}_{profile}")
        accepted_count = sum(1 for task in tasks if task.accepted)
        accepted_seen = accepted_seen or accepted_count > 0
        accuracies = [
            task.best_candidate.metrics.get("accuracy")
            for task in tasks
            if task.best_candidate and task.best_candidate.metrics and task.best_candidate.metrics.get("accuracy") is not None
        ]
        score = float(sum(accuracies) / len(accuracies)) if accuracies else float("-inf")
        if accepted_count:
            score += accepted_count
        if score > best_score:
            best_score = score
            best_profile = profile
            best_tasks = tasks
            best_cfg = cfg
            best_run_id = run_id
            best_report_paths = report_paths
        finish_run(cfg, run_id=run_id, tasks=tasks, report_paths=report_paths)
        print(f"registry: {cfg.data.experiment_db}")
        if accepted_count == 0:
            print(f"next_step: {next_escalation_step(tasks)}")
        for task in tasks:
            metrics = task.best_candidate.metrics if task.best_candidate and task.best_candidate.metrics else {}
            summary_rows.append(
                {
                    "profile": profile,
                    "timeframe": task.timeframe,
                    "horizon": task.horizon,
                    "best_candidate": task.best_candidate.name if task.best_candidate else None,
                    "accepted": task.accepted,
                    "accuracy": metrics.get("accuracy"),
                    "high_confidence_accuracy": metrics.get("high_confidence_accuracy"),
                    "high_confidence_coverage": metrics.get("high_confidence_coverage"),
                    "run_id": run_id,
                    "reason": task.acceptance_reason,
                }
            )
        if accepted_count and args.stop_on_accepted:
            print("accepted candidate found; stopping because --stop-on-accepted was set")
            break
    summary_path = base_cfg.data.report_dir / f"{stem_base}_summary.csv"
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        print(f"\nsearch_summary: {summary_path}")
    else:
        print("\nsearch_summary: no new completed runs")
    print(f"best_profile: {best_profile}")
    print(f"best_run_id: {best_run_id}")
    if args.save_best and best_tasks is not None and best_cfg is not None and best_run_id is not None:
        run_artifact = best_cfg.data.model_dir / f"{best_run_id}.joblib"
        saved_run = save_artifact(best_cfg, best_tasks, path=run_artifact)
        best_has_accepted = any(task.accepted for task in best_tasks)
        existing_accepted = accepted_run_exists(best_cfg)
        saved_latest = None
        if best_has_accepted or existing_accepted is None or existing_accepted == best_run_id:
            saved_latest = save_artifact(best_cfg, best_tasks)
        finish_run(
            best_cfg,
            run_id=best_run_id,
            tasks=best_tasks,
            report_paths=best_report_paths,
            artifact_path=saved_run,
        )
        if saved_latest is not None:
            print(f"artifact: {saved_latest}")
        else:
            print(f"artifact: preserved existing accepted latest from run {existing_accepted}")
        print(f"run_artifact: {saved_run}")
    if not accepted_seen:
        print("search_status: no accepted candidate found; this is an intermediate failed-search state, not completion")
    return 0


def _print_leaderboard(rows: list[dict[str, object]]) -> None:
    if not rows:
        print("No completed experiment tasks found.")
        return
    headers = ["Rank", "TF", "H", "Model", "Accepted", "Acc", "HCAcc", "HCCov", "Sel10", "SelCov", "Base", "Margin", "Profile", "Run"]
    table = []
    for idx, row in enumerate(rows, start=1):
        table.append(
            [
                str(idx),
                str(row["timeframe"]),
                str(row["horizon"]),
                str(row["best_candidate"]),
                str(bool(row["accepted"])),
                _fmt(row["accuracy"]),
                _fmt(row["high_confidence_accuracy"]),
                _fmt(row["high_confidence_coverage"]),
                _fmt(row.get("best_selective_accuracy_10pct")),
                _fmt(row.get("best_selective_coverage_10pct")),
                _fmt(row["best_baseline_accuracy"]),
                _fmt(row["margin_over_best_baseline"]),
                str(row.get("profile", "")),
                str(row.get("run_id", "")),
            ]
        )
    widths = [max(len(headers[i]), *(len(r[i]) for r in table)) for i in range(len(headers))]
    print("  ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("  ".join("-" * widths[i] for i in range(len(headers))))
    for row in table:
        print("  ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


def _refresh_prediction_data(cfg: AppConfig, *, days: int, live_flow_seconds: float = 0.0) -> None:
    end = utc_now()
    start = end - timedelta(days=days)
    intervals = {cfg.data.base_interval, "5m"}
    for interval in intervals:
        fetch_binance_klines(cfg, market=cfg.data.market, interval=interval, start=start, end=end)
        if cfg.data.include_spot:
            fetch_binance_klines(cfg, market="spot", interval=interval, start=start, end=end)
    if cfg.data.include_funding:
        fetch_funding_rates(cfg, start=start, end=end)
    if cfg.data.include_open_interest:
        fetch_open_interest(cfg, start=start, end=end)
    if cfg.data.include_derivative_ratios:
        for source in FUTURES_RATIO_SOURCES:
            fetch_futures_ratio(cfg, source=source, period="5m", start=start, end=end)
    if cfg.data.include_basis_klines:
        for interval in intervals:
            for source in FUTURES_BASIS_SOURCES:
                fetch_futures_basis_klines(cfg, source=source, interval=interval, start=start, end=end)
    if cfg.data.include_hyperliquid:
        fetch_hyperliquid_candles(cfg, interval="5m", start=start, end=end)
    if cfg.data.include_coinbase:
        for interval in intervals:
            try:
                fetch_coinbase_candles(cfg, interval=interval, start=start, end=end)
            except ValueError:
                continue
    if cfg.data.include_bybit:
        for interval in intervals:
            try:
                fetch_bybit_klines(cfg, interval=interval, start=start, end=end)
            except ValueError:
                continue
        fetch_bybit_open_interest(cfg, interval="5m", start=start, end=end)
        fetch_bybit_funding_history(cfg, start=start, end=end)
    if cfg.data.include_okx:
        for interval in intervals:
            try:
                fetch_okx_klines(cfg, interval=interval, start=start, end=end)
            except ValueError:
                continue
        fetch_okx_open_interest_history(cfg, period="5m", start=start, end=end)
        fetch_okx_long_short_ratio_history(cfg, period="5m", start=start, end=end)
    if cfg.data.include_agg_trades:
        fetch_binance_agg_trades(
            cfg,
            market=cfg.data.market,
            start=end - timedelta(days=min(days, 1)),
            end=end,
            max_batches=30,
            checkpoint_rows=25_000,
            max_seconds=20.0,
        )
    if cfg.data.include_live_flow and live_flow_seconds > 0:
        collect_binance_live_flow(cfg, market="futures", seconds=live_flow_seconds)
    if cfg.data.include_order_book:
        fetch_binance_order_book_snapshot(cfg, market="futures", limit=100)


def cmd_leaderboard(args: argparse.Namespace) -> int:
    cfg = _load(args)
    rows = leaderboard_rows(
        cfg,
        limit=args.limit,
        accepted_only=args.accepted_only,
        min_high_confidence_coverage=args.min_high_confidence_coverage,
        timeframe=args.timeframe,
        horizon=args.horizon,
    )
    if args.json:
        print(json.dumps(rows, indent=2, default=str))
    else:
        _print_leaderboard(rows)
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    cfg = _load(args)
    if args.artifact:
        artifact = load_artifact(Path(args.artifact))
        rows = artifact.get("summary", [])
        if args.json:
            print(json.dumps(rows, indent=2, default=str))
        else:
            print(f"artifact_created_at: {artifact.get('created_at', 'unknown')}")
            for task in rows:
                metrics = task.get("best_metrics") or {}
                print(
                    f"{task.get('timeframe'):>4} h{task.get('horizon')} "
                    f"best={task.get('best_candidate')} accepted={task.get('accepted')} "
                    f"acc={_fmt(metrics.get('accuracy'))} "
                    f"hc_acc={_fmt(metrics.get('high_confidence_accuracy'))} "
                    f"hc_cov={_fmt(metrics.get('high_confidence_coverage'))}"
                )
        return 0

    run_id = args.run_id or "best"
    payload = get_run(cfg, run_id)
    if payload is None:
        raise SystemExit(f"run not found: {run_id}")
    if args.json:
        print(json.dumps(payload, indent=2, default=str))
        return 0
    run = payload["run"]
    print(f"run_id:      {run['run_id']}")
    print(f"status:      {run['status']}")
    print(f"command:     {run['command']}")
    print(f"profile:     {run['profile']}")
    print(f"created_at:  {run['created_at']}")
    print(f"accepted:    {run['accepted_count']} / {run['task_count']}")
    print(f"best_acc:    {_fmt(run['best_accuracy'])}")
    print(f"best_hc_acc: {_fmt(run['best_high_confidence_accuracy'])}")
    print(f"report:      {run['report_md'] or run['report_json']}")
    print(f"next_step:   {run['next_step']}")
    _print_leaderboard(payload["tasks"])
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    cfg = _load(args)
    artifact_path = Path(args.artifact or cfg.data.model_dir / "btc_direction_latest.joblib")
    if args.fetch:
        _refresh_prediction_data(cfg, days=args.fetch_days, live_flow_seconds=args.live_flow_seconds)
    artifact, rows = latest_predictions(
        cfg,
        artifact_path,
        timeframes=_parse_csv(args.timeframes, str),
        horizons=_parse_csv(args.horizons, int),
    )
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        created = artifact.get("created_at", "unknown")
        print(f"artifact_created_at: {created}")
        print(compact_prediction_table(rows))
    if args.paper:
        count = record_predictions(cfg, rows, artifact_created_at=artifact.get("created_at"))
        print(f"paper_recorded: {count}")
    return 0


def cmd_live(args: argparse.Namespace) -> int:
    cfg = _load(args)
    artifact_path = Path(args.artifact or cfg.data.model_dir / "btc_direction_latest.joblib")
    while True:
        try:
            if args.fetch:
                _refresh_prediction_data(cfg, days=args.fetch_days, live_flow_seconds=args.live_flow_seconds)
            artifact, rows = latest_predictions(
                cfg,
                artifact_path,
                timeframes=_parse_csv(args.timeframes, str),
                horizons=_parse_csv(args.horizons, int),
            )
            print("\n" + utc_now().isoformat())
            print(compact_prediction_table(rows))
            if args.paper:
                count = record_predictions(cfg, rows, artifact_created_at=artifact.get("created_at"))
                print(f"paper_recorded: {count}")
        except KeyboardInterrupt:
            return 0
        except Exception as exc:  # noqa: BLE001 - live loop must keep reporting
            print(f"live_error: {type(exc).__name__}: {exc}", file=sys.stderr)
        if args.once:
            return 0
        time.sleep(args.poll_seconds)


def cmd_paper_settle(args: argparse.Namespace) -> int:
    cfg = _load(args)
    print(stable_json_dumps(settle_predictions(cfg)))
    return 0


def cmd_paper_summary(args: argparse.Namespace) -> int:
    cfg = _load(args)
    print(stable_json_dumps(paper_summary(cfg)))
    return 0


def cmd_paper_default(args: argparse.Namespace) -> int:
    args.paper = True
    args.json = getattr(args, "json", False)
    return cmd_predict(args)


def cmd_gpu_status(args: argparse.Namespace) -> int:
    status = hardware_info()
    try:
        import torch

        if torch.cuda.is_available():
            status["cuda_memory_allocated_mb"] = round(torch.cuda.memory_allocated(0) / 1024**2, 2)
            status["cuda_memory_reserved_mb"] = round(torch.cuda.memory_reserved(0) / 1024**2, 2)
    except Exception:
        pass
    if args.json:
        print(json.dumps(status, indent=2))
    else:
        for key, value in status.items():
            print(f"{key}: {value}")
    return 0


def cmd_research_loop(args: argparse.Namespace) -> int:
    cfg = _load(args)
    cycle = 0
    while args.cycles is None or cycle < args.cycles:
        cycle += 1
        print(f"\nresearch_cycle: {cycle} started_at={utc_now().isoformat()}")
        if args.fetch:
            end = utc_now()
            start = end - timedelta(days=args.fetch_days)
            for market in _parse_csv(args.markets, str) or [cfg.data.market]:
                for interval in _parse_csv(args.intervals, str) or [cfg.data.base_interval]:
                    try:
                        print(f"fetch candles {market} {interval}: {fetch_binance_klines(cfg, market=market, interval=interval, start=start, end=end)}")
                    except Exception as exc:  # noqa: BLE001 - unattended loop should continue
                        print(f"fetch_error candles {market} {interval}: {type(exc).__name__}: {exc}", file=sys.stderr)
            if args.funding:
                try:
                    print(f"fetch funding: {fetch_funding_rates(cfg, start=start, end=end)}")
                except Exception as exc:  # noqa: BLE001
                    print(f"fetch_error funding: {type(exc).__name__}: {exc}", file=sys.stderr)
            if args.open_interest:
                try:
                    print(f"fetch open_interest: {fetch_open_interest(cfg, start=start, end=end)}")
                except Exception as exc:  # noqa: BLE001
                    print(f"fetch_error open_interest: {type(exc).__name__}: {exc}", file=sys.stderr)
            if args.derivative_ratios:
                for period in _parse_csv(args.ratio_periods, str) or ["5m"]:
                    for source in _parse_csv(args.ratio_sources, str) or list(FUTURES_RATIO_SOURCES):
                        try:
                            path = fetch_futures_ratio(cfg, source=source, period=period, start=start, end=end)
                            print(f"fetch derivative_ratio {source} {period}: {path}")
                        except Exception as exc:  # noqa: BLE001
                            print(f"fetch_error derivative_ratio {source} {period}: {type(exc).__name__}: {exc}", file=sys.stderr)
            if args.metrics_archive:
                metrics_start = end - timedelta(days=args.metrics_days)
                for market in _parse_csv(args.markets, str) or [cfg.data.market]:
                    try:
                        path = fetch_binance_vision_metrics(
                            cfg,
                            market=market,
                            start=metrics_start,
                            end=end,
                            max_files=args.metrics_archive_max_files,
                            checkpoint_files=args.metrics_archive_checkpoint_files,
                            timeframe=args.metrics_feature_timeframe,
                        )
                        print(f"fetch metrics_archive {market}: {path}")
                    except Exception as exc:  # noqa: BLE001
                        print(f"fetch_error metrics_archive {market}: {type(exc).__name__}: {exc}", file=sys.stderr)
            if args.basis_klines:
                for interval in _parse_csv(args.basis_intervals, str) or _parse_csv(args.intervals, str) or [cfg.data.base_interval]:
                    for source in _parse_csv(args.basis_sources, str) or list(FUTURES_BASIS_SOURCES):
                        try:
                            path = fetch_futures_basis_klines(cfg, source=source, interval=interval, start=start, end=end)
                            print(f"fetch basis_kline {source} {interval}: {path}")
                        except Exception as exc:  # noqa: BLE001
                            print(f"fetch_error basis_kline {source} {interval}: {type(exc).__name__}: {exc}", file=sys.stderr)
            if args.hyperliquid:
                for interval in _parse_csv(args.hyperliquid_intervals, str) or ["5m"]:
                    try:
                        path = fetch_hyperliquid_candles(cfg, interval=interval, start=start, end=end)
                        print(f"fetch hyperliquid {interval}: {path}")
                    except Exception as exc:  # noqa: BLE001
                        print(f"fetch_error hyperliquid {interval}: {type(exc).__name__}: {exc}", file=sys.stderr)
            if args.coinbase:
                for interval in _parse_csv(args.coinbase_intervals, str) or ["5m"]:
                    try:
                        path = fetch_coinbase_candles(cfg, interval=interval, start=start, end=end)
                        print(f"fetch coinbase {interval}: {path}")
                    except Exception as exc:  # noqa: BLE001
                        print(f"fetch_error coinbase {interval}: {type(exc).__name__}: {exc}", file=sys.stderr)
            if args.bybit:
                for interval in _parse_csv(args.bybit_intervals, str) or ["5m"]:
                    try:
                        path = fetch_bybit_klines(cfg, interval=interval, start=start, end=end)
                        print(f"fetch bybit {interval}: {path}")
                    except Exception as exc:  # noqa: BLE001
                        print(f"fetch_error bybit {interval}: {type(exc).__name__}: {exc}", file=sys.stderr)
                for interval in _parse_csv(args.bybit_oi_intervals, str) or ["5m"]:
                    try:
                        path = fetch_bybit_open_interest(cfg, interval=interval, start=start, end=end)
                        print(f"fetch bybit_open_interest {interval}: {path}")
                    except Exception as exc:  # noqa: BLE001
                        print(f"fetch_error bybit_open_interest {interval}: {type(exc).__name__}: {exc}", file=sys.stderr)
                try:
                    print(f"fetch bybit_funding: {fetch_bybit_funding_history(cfg, start=start, end=end)}")
                except Exception as exc:  # noqa: BLE001
                    print(f"fetch_error bybit_funding: {type(exc).__name__}: {exc}", file=sys.stderr)
            if args.okx:
                for interval in _parse_csv(args.okx_intervals, str) or ["5m"]:
                    try:
                        path = fetch_okx_klines(cfg, interval=interval, start=start, end=end)
                        print(f"fetch okx {interval}: {path}")
                    except Exception as exc:  # noqa: BLE001
                        print(f"fetch_error okx {interval}: {type(exc).__name__}: {exc}", file=sys.stderr)
                if args.okx_derivatives:
                    for period in _parse_csv(args.okx_periods, str) or ["5m"]:
                        try:
                            path = fetch_okx_open_interest_history(cfg, period=period, start=start, end=end)
                            print(f"fetch okx_open_interest {period}: {path}")
                        except Exception as exc:  # noqa: BLE001
                            print(f"fetch_error okx_open_interest {period}: {type(exc).__name__}: {exc}", file=sys.stderr)
                        try:
                            path = fetch_okx_long_short_ratio_history(cfg, period=period, start=start, end=end)
                            print(f"fetch okx_long_short_ratio {period}: {path}")
                        except Exception as exc:  # noqa: BLE001
                            print(f"fetch_error okx_long_short_ratio {period}: {type(exc).__name__}: {exc}", file=sys.stderr)
            if args.agg_trades:
                trade_start = end - timedelta(days=args.agg_trade_days)
                for market in _parse_csv(args.markets, str) or [cfg.data.market]:
                    try:
                        path = fetch_binance_agg_trades(
                            cfg,
                            market=market,
                            start=trade_start,
                            end=end,
                            max_batches=args.agg_trade_max_batches,
                            checkpoint_rows=args.agg_trade_checkpoint_rows,
                            max_seconds=args.agg_trade_max_seconds,
                        )
                        print(f"fetch agg_trades {market}: {path}")
                    except Exception as exc:  # noqa: BLE001
                        print(f"fetch_error agg_trades {market}: {type(exc).__name__}: {exc}", file=sys.stderr)
            if args.live_flow:
                try:
                    path = collect_binance_live_flow(cfg, market="futures", seconds=args.live_flow_seconds)
                    print(f"fetch live_flow futures: {path}")
                except Exception as exc:  # noqa: BLE001
                    print(f"fetch_error live_flow futures: {type(exc).__name__}: {exc}", file=sys.stderr)
            if args.order_book:
                try:
                    path = fetch_binance_order_book_snapshot(cfg, market="futures", limit=args.order_book_limit)
                    print(f"fetch order_book futures: {path}")
                except Exception as exc:  # noqa: BLE001
                    print(f"fetch_error order_book futures: {type(exc).__name__}: {exc}", file=sys.stderr)

        search_args = argparse.Namespace(
            config=args.config,
            command="search",
            profiles=args.profiles,
            timeframes=args.timeframes,
            horizons=args.horizons,
            candidates=args.candidates,
            max_rows=args.max_rows,
            max_runs=args.runs_per_cycle,
            report_stem=f"{args.report_stem}_cycle{cycle:04d}" if args.report_stem else None,
            resume=True,
            no_escalate=args.no_escalate,
            stop_on_accepted=args.stop_on_accepted,
            save_best=args.save_best,
        )
        cmd_search(search_args)
        accepted = accepted_run_exists(cfg)
        if accepted:
            print(f"accepted_run: {accepted}")
            if args.stop_on_accepted:
                return 0
        if args.cycles is not None and cycle >= args.cycles:
            break
        print(f"research_sleep_seconds: {args.sleep_seconds}")
        time.sleep(args.sleep_seconds)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="btc_predict",
        description="CLI Bitcoin candle direction prediction with walk-forward validation and paper tracking.",
    )
    parser.add_argument("--config", type=Path, default=None, help="TOML, YAML, or JSON config path.")
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="Fetch or update Binance public market data.")
    fetch.add_argument("--start", default=None)
    fetch.add_argument("--end", default=None)
    fetch.add_argument("--days", type=int, default=365)
    fetch.add_argument("--intervals", default="1m,5m")
    fetch.add_argument("--markets", default="futures,spot")
    fetch.add_argument("--cross-assets", action="store_true")
    fetch.add_argument("--cross-asset-symbols", default=",".join(CROSS_ASSET_SYMBOLS))
    fetch.add_argument("--funding", action="store_true")
    fetch.add_argument("--open-interest", action="store_true")
    fetch.add_argument("--derivative-ratios", action="store_true")
    fetch.add_argument("--ratio-periods", default="5m")
    fetch.add_argument("--ratio-sources", default=None)
    fetch.add_argument("--basis-klines", action="store_true")
    fetch.add_argument("--basis-intervals", default=None)
    fetch.add_argument("--basis-sources", default=None)
    fetch.add_argument("--hyperliquid", action="store_true")
    fetch.add_argument("--hyperliquid-intervals", default="5m")
    fetch.add_argument("--coinbase", action="store_true")
    fetch.add_argument("--coinbase-intervals", default="5m")
    fetch.add_argument("--bybit", action="store_true")
    fetch.add_argument("--bybit-intervals", default="5m")
    fetch.add_argument("--bybit-oi-intervals", default="5m")
    fetch.add_argument("--okx", action="store_true")
    fetch.add_argument("--okx-intervals", default="5m")
    fetch.add_argument("--okx-derivatives", action="store_true")
    fetch.add_argument("--okx-periods", default="5m")
    fetch.add_argument("--agg-trades", action="store_true")
    fetch.add_argument("--agg-trade-archive", action="store_true", help="Fetch Binance Vision daily aggTrade ZIP archives.")
    fetch.add_argument("--agg-trade-days", type=int, default=1)
    fetch.add_argument("--agg-trade-start", default=None)
    fetch.add_argument("--agg-trade-max-batches", type=int, default=None)
    fetch.add_argument("--agg-trade-checkpoint-rows", type=int, default=25_000)
    fetch.add_argument("--agg-trade-max-seconds", type=float, default=None)
    fetch.add_argument("--agg-trade-archive-max-files", type=int, default=None)
    fetch.add_argument("--agg-trade-archive-checkpoint-files", type=int, default=3)
    fetch.add_argument("--agg-trade-feature-timeframes", default="1m,5m")
    fetch.add_argument("--agg-trade-archive-cache-raw", action="store_true")
    fetch.add_argument("--book-depth-archive", action="store_true", help="Fetch Binance Vision daily bookDepth ZIP archives.")
    fetch.add_argument("--book-depth-days", type=int, default=30)
    fetch.add_argument("--book-depth-start", default=None)
    fetch.add_argument("--book-depth-archive-max-files", type=int, default=None)
    fetch.add_argument("--book-depth-archive-checkpoint-files", type=int, default=5)
    fetch.add_argument("--book-depth-feature-timeframe", default="5m")
    fetch.add_argument("--metrics-archive", action="store_true", help="Fetch Binance Vision daily futures metrics ZIP archives.")
    fetch.add_argument("--metrics-days", type=int, default=60)
    fetch.add_argument("--metrics-start", default=None)
    fetch.add_argument("--metrics-archive-max-files", type=int, default=None)
    fetch.add_argument("--metrics-archive-checkpoint-files", type=int, default=10)
    fetch.add_argument("--metrics-feature-timeframe", default="5m")
    fetch.add_argument("--live-flow", action="store_true")
    fetch.add_argument("--live-flow-seconds", type=float, default=30.0)
    fetch.add_argument("--order-book", action="store_true")
    fetch.add_argument("--order-book-limit", type=int, default=100)
    fetch.add_argument("--force", action="store_true")
    fetch.set_defaults(func=cmd_fetch)

    capture = sub.add_parser("capture-flow", help="Continuously collect live Binance futures trade/liquidation flow.")
    capture.add_argument("--chunk-seconds", type=float, default=60.0)
    capture.add_argument("--sleep-seconds", type=float, default=2.0)
    capture.add_argument("--max-chunks", type=int, default=None)
    capture.add_argument("--order-book", action="store_true")
    capture.add_argument("--order-book-limit", type=int, default=100)
    capture.set_defaults(func=cmd_capture_flow)

    data_status = sub.add_parser("data-status", help="Show local market-data cache coverage.")
    data_status.add_argument("--intervals", default="1m,5m")
    data_status.add_argument("--markets", default="futures,spot")
    data_status.add_argument("--json", action="store_true")
    data_status.set_defaults(func=cmd_data_status)

    train = sub.add_parser("train", help="Run walk-forward model search and save a prediction artifact.")
    train.add_argument("--timeframes", default=None, help="Comma-separated timeframes, e.g. 5m,15m,1h.")
    train.add_argument("--horizons", default=None, help="Comma-separated horizons, e.g. 1,2,3.")
    train.add_argument("--candidates", default=None, help="Comma-separated model candidates.")
    train.add_argument("--max-rows", type=int, default=None)
    train.add_argument("--artifact", default=None)
    train.add_argument("--report-stem", default=None)
    train.add_argument("--no-registry", action="store_true")
    train.add_argument("--no-save", action="store_true")
    train.set_defaults(func=cmd_train)

    backtest = sub.add_parser("backtest", help="Run historical walk-forward validation without saving a model artifact.")
    backtest.add_argument("--timeframes", default=None)
    backtest.add_argument("--horizons", default=None)
    backtest.add_argument("--candidates", default=None)
    backtest.add_argument("--max-rows", type=int, default=None)
    backtest.add_argument("--artifact", default=None)
    backtest.add_argument("--report-stem", default=None)
    backtest.add_argument("--no-registry", action="store_true")
    backtest.set_defaults(func=cmd_backtest)

    search = sub.add_parser("search", help="Run multiple reproducible model/data search profiles.")
    search.add_argument("--profiles", default="auto")
    search.add_argument("--timeframes", default=None)
    search.add_argument("--horizons", default=None)
    search.add_argument("--candidates", default=None)
    search.add_argument("--max-rows", type=int, default=None)
    search.add_argument("--max-runs", type=int, default=None, help="Maximum new profile runs for this invocation.")
    search.add_argument("--report-stem", default=None)
    search.add_argument("--resume", action="store_true", help="Skip completed profile/config hashes.")
    search.add_argument("--no-escalate", action="store_true", help="Disable automatic failed-candidate escalation profiles.")
    search.add_argument("--stop-on-accepted", action="store_true")
    search.add_argument("--save-best", action="store_true")
    search.set_defaults(func=cmd_search)

    evaluate = sub.add_parser("evaluate", help="Evaluate a saved model artifact or registry run.")
    evaluate.add_argument("--run-id", default="best")
    evaluate.add_argument("--artifact", default=None)
    evaluate.add_argument("--json", action="store_true")
    evaluate.set_defaults(func=cmd_evaluate)

    leaderboard = sub.add_parser("leaderboard", help="Show the experiment registry leaderboard.")
    leaderboard.add_argument("--limit", type=int, default=20)
    leaderboard.add_argument("--accepted-only", action="store_true")
    leaderboard.add_argument("--min-high-confidence-coverage", type=float, default=None)
    leaderboard.add_argument("--timeframe", default=None)
    leaderboard.add_argument("--horizon", type=int, default=None)
    leaderboard.add_argument("--json", action="store_true")
    leaderboard.set_defaults(func=cmd_leaderboard)

    gpu = sub.add_parser("gpu-status", help="Show Torch/CUDA training availability.")
    gpu.add_argument("--json", action="store_true")
    gpu.set_defaults(func=cmd_gpu_status)

    loop = sub.add_parser("research-loop", help="Run unattended fetch/search cycles until stopped or accepted.")
    loop.add_argument("--profiles", default="gpu_heavy,volatility_label_gpu,volatility_label_cross_venue,bybit_gpu,derivative_ratio_features,flow_features,live_feed_triggers,live_feed_gated,adaptive_ensembles,event_adaptive,feature_state_models,univariate_signals,regime_models,ensembles,neural_variants")
    loop.add_argument("--timeframes", default="5m,15m,1h")
    loop.add_argument("--horizons", default="1,2")
    loop.add_argument("--candidates", default=None)
    loop.add_argument("--max-rows", type=int, default=24000)
    loop.add_argument("--runs-per-cycle", type=int, default=2)
    loop.add_argument("--cycles", type=int, default=None)
    loop.add_argument("--sleep-seconds", type=float, default=300.0)
    loop.add_argument("--report-stem", default="research_loop")
    loop.add_argument("--no-escalate", action="store_true")
    loop.add_argument("--stop-on-accepted", action="store_true")
    loop.add_argument("--save-best", action="store_true")
    loop.add_argument("--fetch", action="store_true")
    loop.add_argument("--fetch-days", type=int, default=1)
    loop.add_argument("--intervals", default="1m,5m")
    loop.add_argument("--markets", default="futures")
    loop.add_argument("--funding", action="store_true")
    loop.add_argument("--open-interest", action="store_true")
    loop.add_argument("--derivative-ratios", action="store_true")
    loop.add_argument("--ratio-periods", default="5m")
    loop.add_argument("--ratio-sources", default=None)
    loop.add_argument("--metrics-archive", action="store_true")
    loop.add_argument("--metrics-days", type=int, default=60)
    loop.add_argument("--metrics-archive-max-files", type=int, default=None)
    loop.add_argument("--metrics-archive-checkpoint-files", type=int, default=10)
    loop.add_argument("--metrics-feature-timeframe", default="5m")
    loop.add_argument("--basis-klines", action="store_true")
    loop.add_argument("--basis-intervals", default=None)
    loop.add_argument("--basis-sources", default=None)
    loop.add_argument("--hyperliquid", action="store_true")
    loop.add_argument("--hyperliquid-intervals", default="5m")
    loop.add_argument("--coinbase", action="store_true")
    loop.add_argument("--coinbase-intervals", default="5m")
    loop.add_argument("--bybit", action="store_true")
    loop.add_argument("--bybit-intervals", default="5m")
    loop.add_argument("--bybit-oi-intervals", default="5m")
    loop.add_argument("--okx", action="store_true")
    loop.add_argument("--okx-intervals", default="5m")
    loop.add_argument("--okx-derivatives", action="store_true")
    loop.add_argument("--okx-periods", default="5m")
    loop.add_argument("--agg-trades", action="store_true")
    loop.add_argument("--agg-trade-days", type=int, default=1)
    loop.add_argument("--agg-trade-max-batches", type=int, default=100)
    loop.add_argument("--agg-trade-checkpoint-rows", type=int, default=25_000)
    loop.add_argument("--agg-trade-max-seconds", type=float, default=90.0)
    loop.add_argument("--live-flow", action="store_true")
    loop.add_argument("--live-flow-seconds", type=float, default=30.0)
    loop.add_argument("--order-book", action="store_true")
    loop.add_argument("--order-book-limit", type=int, default=100)
    loop.set_defaults(func=cmd_research_loop)

    predict = sub.add_parser("predict", help="Predict latest candle directions from a trained artifact.")
    predict.add_argument("--artifact", default=None)
    predict.add_argument("--timeframes", default=None)
    predict.add_argument("--horizons", default=None)
    predict.add_argument("--json", action="store_true")
    predict.add_argument("--paper", action="store_true")
    predict.add_argument("--fetch", action="store_true")
    predict.add_argument("--fetch-days", type=int, default=3)
    predict.add_argument("--live-flow-seconds", type=float, default=0.0)
    predict.set_defaults(func=cmd_predict)

    live = sub.add_parser("live", help="Near-live compact prediction monitor.")
    live.add_argument("--artifact", default=None)
    live.add_argument("--timeframes", default=None)
    live.add_argument("--horizons", default=None)
    live.add_argument("--poll-seconds", type=float, default=60.0)
    live.add_argument("--paper", action="store_true")
    live.add_argument("--fetch", action="store_true")
    live.add_argument("--fetch-days", type=int, default=3)
    live.add_argument("--live-flow-seconds", type=float, default=0.0)
    live.add_argument("--once", action="store_true")
    live.set_defaults(func=cmd_live)

    paper = sub.add_parser("paper", help="Record, settle, or summarize paper-mode predictions.")
    paper.add_argument("--artifact", default=None)
    paper.add_argument("--timeframes", default=None)
    paper.add_argument("--horizons", default=None)
    paper.add_argument("--json", action="store_true")
    paper.add_argument("--fetch", action="store_true")
    paper.add_argument("--fetch-days", type=int, default=3)
    paper.add_argument("--live-flow-seconds", type=float, default=0.0)
    paper.set_defaults(func=cmd_paper_default)
    paper_sub = paper.add_subparsers(dest="paper_command")
    settle = paper_sub.add_parser("settle", help="Settle predictions whose target candles are now known.")
    settle.set_defaults(func=cmd_paper_settle)
    summary = paper_sub.add_parser("summary", help="Summarize settled paper-mode prediction quality.")
    summary.set_defaults(func=cmd_paper_summary)
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(line_buffering=True)
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
