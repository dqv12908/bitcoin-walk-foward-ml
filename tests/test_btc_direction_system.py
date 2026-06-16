from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from btc_direction.config import AppConfig, load_config
from btc_direction.data import (
    aggregate_live_flow_to_timeframe,
    aggregate_trades_to_timeframe,
    context_features,
    lower_timeframe_rollup_features,
    normalize_agg_trades,
    normalize_futures_ratio,
    normalize_live_flow,
    normalize_okx_klines,
    normalize_okx_long_short_ratio,
    normalize_okx_open_interest,
    normalize_order_book_features,
    resample_ohlcv,
)
from btc_direction.evaluation import metric_dict
from btc_direction.experiments import config_digest, finish_run, get_run, leaderboard_rows, start_run
from btc_direction.features import build_supervised_frame
from btc_direction.models import make_estimator, predict_proba_up
from btc_direction.paper import record_predictions, paper_summary
from btc_direction.validation import CandidateResult, TaskResult, _acceptance, make_walk_forward_splits


class DirectionSystemTests(unittest.TestCase):
    def synthetic_candles(self, n: int = 500) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        times = pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC")
        pulse = np.where(np.arange(n) % 12 < 6, 0.001, -0.001)
        noise = rng.normal(0, 0.0003, size=n)
        close = 100_000 * np.exp(np.cumsum(pulse + noise))
        open_ = np.r_[close[0], close[:-1]]
        high = np.maximum(open_, close) * 1.0005
        low = np.minimum(open_, close) * 0.9995
        volume = rng.uniform(10, 100, size=n)
        taker = volume * rng.uniform(0.35, 0.65, size=n)
        return pd.DataFrame(
            {
                "open_time": times,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "quote_volume": volume * close,
                "trade_count": rng.integers(100, 1000, size=n),
                "taker_buy_base": taker,
                "taker_buy_quote": taker * close,
            }
        )

    def test_feature_builder_outputs_targets_and_features(self) -> None:
        supervised = build_supervised_frame(self.synthetic_candles(), "5m", [1, 2])
        self.assertIn("target_up_h1", supervised.frame.columns)
        self.assertIn("target_up_h2", supervised.frame.columns)
        self.assertGreater(len(supervised.feature_columns), 40)
        self.assertIn("inwin_15m_return", supervised.feature_columns)
        self.assertIn("inwin_1h_close_position", supervised.feature_columns)
        self.assertIn("signal_composite_ema5", supervised.feature_columns)
        self.assertIn("signal_position_015_005", supervised.feature_columns)
        self.assertNotIn("future_close_h1", supervised.feature_columns)
        self.assertNotIn("target_up_h1", supervised.feature_columns)

    def test_feature_builder_supports_label_variants(self) -> None:
        supervised = build_supervised_frame(
            self.synthetic_candles(),
            "5m",
            [1],
            label_mode="target_body",
            min_move_bps=0.5,
        )
        self.assertIn("target_up_h1", supervised.frame.columns)
        self.assertGreater(supervised.frame["target_up_h1"].notna().sum(), 100)

    def test_feature_builder_supports_volatility_adjusted_labels(self) -> None:
        candles = self.synthetic_candles(260)
        basic = build_supervised_frame(candles, "5m", [1, 2])
        volatility_adjusted = build_supervised_frame(
            candles,
            "5m",
            [1, 2],
            label_mode="volatility_adjusted_050",
        )
        self.assertIn("target_up_h2", volatility_adjusted.frame.columns)
        basic_n = int(basic.frame["target_up_h1"].notna().sum())
        adjusted_n = int(volatility_adjusted.frame["target_up_h1"].notna().sum())
        self.assertGreater(adjusted_n, 20)
        self.assertLess(adjusted_n, basic_n)

    def test_feature_builder_supports_feature_groups(self) -> None:
        all_features = build_supervised_frame(self.synthetic_candles(), "5m", [1])
        price_features = build_supervised_frame(
            self.synthetic_candles(),
            "5m",
            [1],
            feature_groups=["price_action", "calendar"],
        )
        self.assertLess(len(price_features.feature_columns), len(all_features.feature_columns))
        self.assertIn("body_return", price_features.feature_columns)
        self.assertNotIn("rsi_14", price_features.feature_columns)

    def test_lower_timeframe_rollups_feed_5m_features(self) -> None:
        rng = np.random.default_rng(7)
        times = pd.date_range("2025-01-01", periods=40, freq="1min", tz="UTC")
        close = 100_000 * np.exp(np.cumsum(rng.normal(0, 0.0004, len(times))))
        open_ = np.r_[close[0], close[:-1]]
        volume = rng.uniform(1, 4, len(times))
        one_min = pd.DataFrame(
            {
                "open_time": times,
                "open": open_,
                "high": np.maximum(open_, close) * 1.0002,
                "low": np.minimum(open_, close) * 0.9998,
                "close": close,
                "volume": volume,
                "quote_volume": volume * close,
                "trade_count": rng.integers(20, 80, len(times)),
                "taker_buy_base": volume * rng.uniform(0.35, 0.65, len(times)),
                "taker_buy_quote": volume * close * rng.uniform(0.35, 0.65, len(times)),
            }
        )
        rollup = lower_timeframe_rollup_features(one_min, "5m")
        self.assertIn("ltf_return_last", rollup.columns)
        self.assertIn("ltf_return_pos1", rollup.columns)
        self.assertIn("ltf_taker_quote_imbalance_pos5", rollup.columns)
        self.assertIn("ltf_taker_quote_imbalance_change", rollup.columns)
        self.assertEqual(float(rollup.iloc[0]["ltf_child_count"]), 5.0)
        five_min = resample_ohlcv(one_min, "5m").merge(rollup, on="open_time", how="left")
        supervised = build_supervised_frame(five_min, "5m", [1], feature_groups=["technical", "flow"])
        self.assertIn("ltf_return_last", supervised.feature_columns)
        self.assertIn("ltf_return_pos1", supervised.feature_columns)
        self.assertIn("ltf_taker_quote_imbalance_pos5", supervised.feature_columns)
        self.assertIn("ltf_taker_quote_imbalance_change", supervised.feature_columns)

    def test_aggregate_trades_to_timeframe(self) -> None:
        trades = pd.DataFrame(
            {
                "trade_time": pd.to_datetime(
                    [
                        "2025-01-01T00:00:05Z",
                        "2025-01-01T00:00:30Z",
                        "2025-01-01T00:01:05Z",
                    ]
                ),
                "agg_id": [1, 2, 3],
                "price": [100.0, 101.0, 102.0],
                "qty": [1.0, 2.0, 3.0],
                "quote_qty": [100.0, 202.0, 306.0],
                "buyer_is_maker": [False, True, False],
            }
        )
        out = aggregate_trades_to_timeframe(trades, "1m")
        first = out.iloc[0]
        self.assertEqual(first["agg_trade_count"], 2)
        self.assertAlmostEqual(first["agg_trade_quote_qty"], 302.0)
        self.assertAlmostEqual(first["agg_taker_quote_imbalance"], (100.0 - 202.0) / 302.0)

    def test_binance_vision_agg_trade_columns_normalize(self) -> None:
        trades = normalize_agg_trades(
            pd.DataFrame(
                {
                    "agg_trade_id": [1, "agg_trade_id", 2],
                    "price": [100.0, "price", 101.0],
                    "quantity": [1.0, "quantity", 2.0],
                    "first_trade_id": [10, "first_trade_id", 11],
                    "last_trade_id": [10, "last_trade_id", 12],
                    "transact_time": [1735689600000, "transact_time", 1735689630000],
                    "is_buyer_maker": [False, "is_buyer_maker", True],
                }
            )
        )
        self.assertEqual(len(trades), 2)
        self.assertEqual(trades.iloc[0]["agg_id"], 1)
        self.assertAlmostEqual(trades.iloc[1]["quote_qty"], 202.0)
        self.assertTrue(bool(trades.iloc[1]["buyer_is_maker"]))

    def test_live_flow_aggregation_and_features(self) -> None:
        flow = normalize_live_flow(
            pd.DataFrame(
                {
                    "trade_time": pd.to_datetime(
                        [
                            "2025-01-01T00:00:05Z",
                            "2025-01-01T00:00:30Z",
                            "2025-01-01T00:01:05Z",
                        ]
                    ),
                    "event_time": pd.to_datetime(
                        [
                            "2025-01-01T00:00:05Z",
                            "2025-01-01T00:00:30Z",
                            "2025-01-01T00:01:05Z",
                        ]
                    ),
                    "event_type": ["agg_trade", "agg_trade", "force_order"],
                    "symbol": ["BTCUSDT", "BTCUSDT", "BTCUSDT"],
                    "side": ["buy", "sell", "sell"],
                    "price": [100.0, 101.0, 102.0],
                    "qty": [1.0, 2.0, 3.0],
                    "quote_qty": [100.0, 202.0, 306.0],
                    "trade_id": [1, 2, 3],
                    "order_status": ["", "", "FILLED"],
                }
            )
        )
        out = aggregate_live_flow_to_timeframe(flow, "5m")
        self.assertIn("live_taker_quote_imbalance", out.columns)
        self.assertIn("liq_buy_sell_imbalance", out.columns)
        first = out.iloc[0]
        self.assertEqual(first["live_trade_count"], 2)
        self.assertAlmostEqual(first["live_taker_quote_imbalance"], (100.0 - 202.0) / 302.0)
        candles = self.synthetic_candles(140)
        merged = candles.merge(out, on="open_time", how="left")
        supervised = build_supervised_frame(merged, "5m", [1], feature_groups=["flow"])
        self.assertIn("live_taker_quote_imbalance_mean_5", supervised.feature_columns)
        self.assertIn("liq_pressure", supervised.feature_columns)

    def test_order_book_features_are_flow_inputs(self) -> None:
        candles = self.synthetic_candles(140)
        book = normalize_order_book_features(
            pd.DataFrame(
                {
                    "open_time": candles["open_time"],
                    "book_bid_price_1": candles["close"] - 1.0,
                    "book_ask_price_1": candles["close"] + 1.0,
                    "book_mid": candles["close"],
                    "book_spread_bps": 0.2,
                    "book_depth_quote_top20": np.linspace(1_000_000, 2_000_000, len(candles)),
                    "book_imbalance_top20": np.linspace(-0.2, 0.2, len(candles)),
                    "book_depth_quote_top100": np.linspace(4_000_000, 5_000_000, len(candles)),
                    "book_imbalance_top100": np.linspace(-0.1, 0.1, len(candles)),
                }
            )
        )
        frame = candles.merge(book, on="open_time", how="left")
        supervised = build_supervised_frame(frame, "5m", [1], feature_groups=["flow"])
        self.assertIn("book_imbalance_top20", supervised.feature_columns)
        self.assertIn("book_imbalance_top20_mean_3", supervised.feature_columns)
        self.assertIn("book_depth_quote_top20_z_21", supervised.feature_columns)

    def test_futures_ratio_features_are_derivatives(self) -> None:
        candles = self.synthetic_candles(120)
        ratios = normalize_futures_ratio(
            pd.DataFrame(
                {
                    "timestamp": [int(ts.timestamp() * 1000) for ts in candles["open_time"].iloc[:120]],
                    "longAccount": np.linspace(0.45, 0.60, 120),
                    "shortAccount": np.linspace(0.55, 0.40, 120),
                    "longShortRatio": np.linspace(0.8, 1.5, 120),
                }
            ),
            "global_long_short_account",
        )
        frame = candles.merge(ratios, on="open_time", how="left")
        supervised = build_supervised_frame(frame, "5m", [1], feature_groups=["derivatives"])
        self.assertIn("global_lsr_long_short_ratio", supervised.feature_columns)
        self.assertIn("global_lsr_long_short_ratio_z_21", supervised.feature_columns)
        self.assertIn("global_lsr_account_spread", supervised.feature_columns)

    def test_coinbase_features_are_cross_market(self) -> None:
        candles = self.synthetic_candles(140)
        candles["coinbase_close"] = candles["close"] * 0.9998
        candles["coinbase_return_1"] = np.log(candles["coinbase_close"] / candles["coinbase_close"].shift(1))
        candles["coinbase_quote_volume"] = candles["quote_volume"] * 0.7
        candles["futures_coinbase_spread_bps"] = (candles["close"] / candles["coinbase_close"] - 1.0) * 10_000.0
        supervised = build_supervised_frame(candles, "5m", [1], feature_groups=["cross_market"])
        self.assertIn("coinbase_close", supervised.feature_columns)
        self.assertIn("coinbase_return_sum_12", supervised.feature_columns)
        self.assertIn("futures_minus_coinbase_return", supervised.feature_columns)
        self.assertIn("futures_coinbase_spread_bps_z_55", supervised.feature_columns)

    def test_bybit_features_cover_cross_market_and_derivatives(self) -> None:
        candles = self.synthetic_candles(140)
        candles["bybit_close"] = candles["close"] * 1.0002
        candles["bybit_return_1"] = np.log(candles["bybit_close"] / candles["bybit_close"].shift(1))
        candles["bybit_quote_volume"] = candles["quote_volume"] * 0.8
        candles["futures_bybit_spread_bps"] = (candles["close"] / candles["bybit_close"] - 1.0) * 10_000.0
        candles["coinbase_close"] = candles["close"] * 0.9998
        candles["coinbase_return_1"] = np.log(candles["coinbase_close"] / candles["coinbase_close"].shift(1))
        candles["bybit_open_interest"] = np.linspace(10_000, 12_000, len(candles))
        candles["sum_open_interest"] = np.linspace(20_000, 22_000, len(candles))
        candles["sum_open_interest_value"] = candles["sum_open_interest"] * candles["close"]
        candles["bybit_funding_rate"] = np.linspace(-0.0001, 0.0002, len(candles))
        cross_market = build_supervised_frame(candles, "5m", [1], feature_groups=["cross_market"])
        derivatives = build_supervised_frame(candles, "5m", [1], feature_groups=["derivatives"])
        self.assertIn("bybit_close", cross_market.feature_columns)
        self.assertIn("futures_bybit_spread_bps_z_55", cross_market.feature_columns)
        self.assertIn("futures_minus_bybit_return", cross_market.feature_columns)
        self.assertIn("bybit_minus_coinbase_return", cross_market.feature_columns)
        self.assertIn("bybit_coinbase_spread_bps_z_55", cross_market.feature_columns)
        self.assertIn("bybit_open_interest", derivatives.feature_columns)
        self.assertIn("bybit_open_interest_z_55", derivatives.feature_columns)
        self.assertIn("binance_bybit_open_interest_ratio_log", derivatives.feature_columns)
        self.assertIn("bybit_funding_rate_z_21", derivatives.feature_columns)

    def test_okx_features_are_cross_market(self) -> None:
        candles = self.synthetic_candles(140)
        raw = pd.DataFrame(
            {
                "ts": [int(ts.timestamp() * 1000) for ts in candles["open_time"]],
                "o": candles["open"] * 1.0001,
                "h": candles["high"] * 1.0001,
                "l": candles["low"] * 1.0001,
                "c": candles["close"] * 1.0001,
                "vol": candles["volume"] * 0.9,
                "volCcy": candles["volume"] * 0.9,
                "volCcyQuote": candles["quote_volume"] * 0.9,
                "confirm": ["1"] * len(candles),
            }
        )
        okx = normalize_okx_klines(raw).rename(
            columns={
                "open": "okx_open",
                "high": "okx_high",
                "low": "okx_low",
                "close": "okx_close",
                "volume": "okx_volume",
                "quote_volume": "okx_quote_volume",
            }
        )
        frame = candles.merge(okx, on="open_time", how="left")
        frame["okx_return_1"] = np.log(frame["okx_close"] / frame["okx_close"].shift(1))
        frame["futures_okx_spread_bps"] = (frame["close"] / frame["okx_close"] - 1.0) * 10_000.0
        frame["coinbase_close"] = frame["close"] * 0.9998
        frame["coinbase_return_1"] = np.log(frame["coinbase_close"] / frame["coinbase_close"].shift(1))
        supervised = build_supervised_frame(frame, "5m", [1], feature_groups=["cross_market"])
        self.assertIn("okx_close", supervised.feature_columns)
        self.assertIn("okx_return_sum_12", supervised.feature_columns)
        self.assertIn("futures_minus_okx_return", supervised.feature_columns)
        self.assertIn("futures_okx_spread_bps_z_55", supervised.feature_columns)
        self.assertIn("okx_minus_coinbase_return", supervised.feature_columns)

    def test_okx_derivative_features(self) -> None:
        candles = self.synthetic_candles(140)
        oi = normalize_okx_open_interest(
            pd.DataFrame(
                {
                    "ts": [int(ts.timestamp() * 1000) for ts in candles["open_time"]],
                    "oi": np.linspace(1_000_000, 1_200_000, len(candles)),
                    "oiCcy": np.linspace(10_000, 12_000, len(candles)),
                    "oiUsd": np.linspace(800_000_000, 900_000_000, len(candles)),
                }
            )
        )
        lsr = normalize_okx_long_short_ratio(
            pd.DataFrame(
                {
                    "ts": [int(ts.timestamp() * 1000) for ts in candles["open_time"]],
                    "ratio": np.linspace(0.9, 1.3, len(candles)),
                }
            )
        )
        frame = candles.merge(oi, on="open_time", how="left").merge(lsr, on="open_time", how="left")
        frame["sum_open_interest_value"] = np.linspace(900_000_000, 1_000_000_000, len(candles))
        supervised = build_supervised_frame(frame, "5m", [1], feature_groups=["derivatives"])
        self.assertIn("okx_open_interest_usd", supervised.feature_columns)
        self.assertIn("okx_open_interest_usd_return_1", supervised.feature_columns)
        self.assertIn("binance_okx_open_interest_value_ratio_log", supervised.feature_columns)
        self.assertIn("okx_long_short_ratio_z_21", supervised.feature_columns)

    def test_basis_features_are_derivatives(self) -> None:
        candles = self.synthetic_candles(120)
        candles["mark_price_close"] = candles["close"] * 0.9999
        candles["index_price_close"] = candles["close"] * 1.0001
        candles["premium_index_close"] = -0.0002
        candles["futures_mark_spread_bps"] = (candles["close"] / candles["mark_price_close"] - 1.0) * 10_000.0
        candles["mark_index_basis_bps"] = (candles["mark_price_close"] / candles["index_price_close"] - 1.0) * 10_000.0
        candles["premium_index_bps"] = candles["premium_index_close"] * 10_000.0
        supervised = build_supervised_frame(candles, "5m", [1], feature_groups=["derivatives"])
        self.assertIn("mark_index_basis_bps", supervised.feature_columns)
        self.assertIn("mark_index_basis_bps_z_55", supervised.feature_columns)
        self.assertIn("futures_minus_index_return", supervised.feature_columns)

    def test_recency_weighted_candidate_fits(self) -> None:
        supervised = build_supervised_frame(self.synthetic_candles(260), "5m", [1])
        frame = supervised.frame.dropna(subset=["target_up_h1"]).tail(180)
        model = make_estimator("recency_logistic", AppConfig())
        model.fit(frame[supervised.feature_columns], frame["target_up_h1"].to_numpy(dtype=int))
        prob = predict_proba_up(model, frame[supervised.feature_columns].tail(5))
        self.assertEqual(len(prob), 5)
        self.assertTrue(np.all((prob > 0.0) & (prob < 1.0)))

    def test_feature_selection_candidate_fits(self) -> None:
        supervised = build_supervised_frame(self.synthetic_candles(260), "5m", [1])
        frame = supervised.frame.dropna(subset=["target_up_h1"]).tail(180)
        model = make_estimator("select_logistic", AppConfig())
        model.fit(frame[supervised.feature_columns], frame["target_up_h1"].to_numpy(dtype=int))
        prob = predict_proba_up(model, frame[supervised.feature_columns].tail(5))
        self.assertEqual(len(prob), 5)
        self.assertTrue(np.all((prob > 0.0) & (prob < 1.0)))

    def test_balanced_gpu_candidate_fits(self) -> None:
        supervised = build_supervised_frame(self.synthetic_candles(220), "5m", [1])
        frame = supervised.frame.dropna(subset=["target_up_h1"]).tail(160)
        model = make_estimator("balanced_xgboost_fast", AppConfig())
        model.fit(frame[supervised.feature_columns], frame["target_up_h1"].to_numpy(dtype=int))
        prob = predict_proba_up(model, frame[supervised.feature_columns].tail(5))
        self.assertEqual(len(prob), 5)
        self.assertTrue(np.all((prob > 0.0) & (prob < 1.0)))

    def test_xgboost_search_candidate_fits(self) -> None:
        supervised = build_supervised_frame(self.synthetic_candles(420), "5m", [1])
        frame = supervised.frame.dropna(subset=["target_up_h1"]).tail(320)
        cfg = AppConfig()
        model = make_estimator("xgboost_search_fast", cfg)
        model.fit(frame[supervised.feature_columns], frame["target_up_h1"].to_numpy(dtype=int))
        prob = predict_proba_up(model, frame[supervised.feature_columns].tail(5))
        self.assertEqual(len(prob), 5)
        self.assertTrue(np.all((prob > 0.0) & (prob < 1.0)))

    def test_rule_trigger_candidate_fits(self) -> None:
        candles = self.synthetic_candles(260)
        candles["bybit_close"] = candles["close"] * 1.0002
        candles["bybit_return_1"] = np.log(candles["bybit_close"] / candles["bybit_close"].shift(1))
        candles["futures_bybit_spread_bps"] = (candles["close"] / candles["bybit_close"] - 1.0) * 10_000.0
        candles["bybit_open_interest"] = np.linspace(10_000, 12_000, len(candles))
        supervised = build_supervised_frame(candles, "5m", [1])
        frame = supervised.frame.dropna(subset=["target_up_h1"]).tail(180)
        model = make_estimator("trigger_confluence", AppConfig())
        model.fit(frame[supervised.feature_columns], frame["target_up_h1"].to_numpy(dtype=int))
        prob = predict_proba_up(model, frame[supervised.feature_columns].tail(5))
        self.assertEqual(len(prob), 5)
        self.assertTrue(np.all((prob > 0.0) & (prob < 1.0)))

    def test_learned_event_trigger_candidate_fits(self) -> None:
        candles = self.synthetic_candles(620)
        signal = np.sin(np.arange(len(candles)) / 7.0)
        candles["live_trade_quote_qty"] = 1_000_000.0 * (1.0 + 0.8 * np.abs(signal))
        candles["live_trade_count"] = 200.0 + 80.0 * np.abs(signal)
        candles["live_taker_quote_imbalance"] = signal
        candles["agg_trade_quote_qty"] = 800_000.0 * (1.0 + 0.6 * np.abs(signal))
        candles["agg_trade_count"] = 160.0 + 60.0 * np.abs(signal)
        candles["agg_taker_quote_imbalance"] = signal * 0.8
        supervised = build_supervised_frame(candles, "5m", [1])
        frame = supervised.frame.dropna(subset=["live_taker_quote_imbalance_mean_5"]).tail(500).copy()
        y = (pd.to_numeric(frame["live_taker_quote_imbalance_mean_5"], errors="coerce").fillna(0.0) > 0.0).to_numpy(dtype=int)
        model = make_estimator("event_trigger_flow", AppConfig())
        model.fit(frame[supervised.feature_columns], y)
        ranked = frame.assign(_signal=pd.to_numeric(frame["live_taker_quote_imbalance_mean_5"], errors="coerce"))
        low = ranked.nsmallest(8, "_signal")
        high = ranked.nlargest(8, "_signal")
        low_prob = predict_proba_up(model, low[supervised.feature_columns])
        high_prob = predict_proba_up(model, high[supervised.feature_columns])
        self.assertEqual(len(high_prob), 8)
        self.assertGreater(float(high_prob.mean()), float(low_prob.mean()))
        self.assertTrue(np.all((high_prob > 0.0) & (high_prob < 1.0)))

    def test_confidence_gated_event_candidate_fits(self) -> None:
        candles = self.synthetic_candles(520)
        signal = np.sin(np.arange(len(candles)) / 6.0)
        candles["live_trade_quote_qty"] = 1_000_000.0 * (1.0 + np.abs(signal))
        candles["live_trade_count"] = 150.0 + 70.0 * np.abs(signal)
        candles["live_taker_quote_imbalance"] = signal
        candles["agg_trade_quote_qty"] = 900_000.0 * (1.0 + 0.5 * np.abs(signal))
        candles["agg_trade_count"] = 120.0 + 50.0 * np.abs(signal)
        candles["agg_taker_quote_imbalance"] = signal * 0.7
        supervised = build_supervised_frame(candles, "5m", [1])
        frame = supervised.frame.dropna(subset=["live_taker_quote_imbalance_mean_5"]).tail(420).copy()
        y = (pd.to_numeric(frame["live_taker_quote_imbalance_mean_5"], errors="coerce").fillna(0.0) > 0.0).to_numpy(dtype=int)
        cfg = AppConfig()
        cfg.models.use_gpu = False
        model = make_estimator("gated_event_flow", cfg)
        model.fit(frame[supervised.feature_columns], y)
        prob = predict_proba_up(model, frame[supervised.feature_columns].tail(8))
        self.assertEqual(len(prob), 8)
        self.assertTrue(np.all((prob > 0.0) & (prob < 1.0)))

    def test_calendar_pattern_candidate_fits(self) -> None:
        supervised = build_supervised_frame(self.synthetic_candles(360), "5m", [1], feature_groups=["calendar"])
        frame = supervised.frame.dropna(subset=["target_up_h1"]).tail(260)
        model = make_estimator("calendar_hourly", AppConfig())
        model.fit(frame[supervised.feature_columns], frame["target_up_h1"].to_numpy(dtype=int))
        prob = predict_proba_up(model, frame[supervised.feature_columns].tail(5))
        self.assertEqual(len(prob), 5)
        self.assertTrue(np.all((prob > 0.0) & (prob < 1.0)))

    def test_time_window_selector_candidate_fits(self) -> None:
        supervised = build_supervised_frame(self.synthetic_candles(720), "5m", [1], feature_groups=["calendar"])
        frame = supervised.frame.dropna(subset=["target_up_h1"]).tail(600).copy()
        y = frame["target_up_h1"].to_numpy(dtype=int)
        # Create one repeatable training-only intraday edge so the selector has
        # a deterministic bucket to discover in this unit test.
        edge_mask = (pd.to_numeric(frame["minute_of_day"], errors="coerce") < 60).to_numpy()
        y[edge_mask] = 1
        model = make_estimator("time_window_selector", AppConfig())
        model.fit(frame[supervised.feature_columns], y)
        prob = predict_proba_up(model, frame[supervised.feature_columns].tail(12))
        self.assertEqual(len(prob), 12)
        self.assertTrue(np.all((prob > 0.0) & (prob < 1.0)))

    def test_feature_state_candidate_fits(self) -> None:
        candles = self.synthetic_candles(520)
        phase = np.sin(np.arange(len(candles)) / 9.0)
        candles["bybit_close"] = candles["close"] * (1.0 + 0.0002 * phase)
        candles["coinbase_close"] = candles["close"] * (1.0 - 0.0001 * phase)
        candles["bybit_return_1"] = np.log(candles["bybit_close"] / candles["bybit_close"].shift(1))
        candles["coinbase_return_1"] = np.log(candles["coinbase_close"] / candles["coinbase_close"].shift(1))
        candles["futures_bybit_spread_bps"] = (candles["close"] / candles["bybit_close"] - 1.0) * 10_000.0
        candles["futures_coinbase_spread_bps"] = (candles["close"] / candles["coinbase_close"] - 1.0) * 10_000.0
        supervised = build_supervised_frame(candles, "5m", [1])
        frame = supervised.frame.dropna(subset=["target_up_h1"]).tail(400)
        model = make_estimator("state_cross_venue", AppConfig())
        model.fit(frame[supervised.feature_columns], frame["target_up_h1"].to_numpy(dtype=int))
        prob = predict_proba_up(model, frame[supervised.feature_columns].tail(8))
        self.assertEqual(len(prob), 8)
        self.assertTrue(np.all((prob > 0.0) & (prob < 1.0)))

    def test_univariate_signal_candidate_fits(self) -> None:
        candles = self.synthetic_candles(520)
        candles["sum_open_interest"] = np.linspace(10_000, 15_000, len(candles)) * (
            1.0 + 0.01 * np.sin(np.arange(len(candles)) / 11.0)
        )
        candles["sum_open_interest_value"] = candles["sum_open_interest"] * candles["close"]
        supervised = build_supervised_frame(candles, "5m", [1])
        frame = supervised.frame.dropna(subset=["target_up_h1"]).tail(420)
        model = make_estimator("univariate_signal", AppConfig())
        model.fit(frame[supervised.feature_columns], frame["target_up_h1"].to_numpy(dtype=int))
        prob = predict_proba_up(model, frame[supervised.feature_columns].tail(8))
        self.assertEqual(len(prob), 8)
        self.assertTrue(np.all((prob > 0.0) & (prob < 1.0)))

    def test_temporal_selector_candidate_fits(self) -> None:
        supervised = build_supervised_frame(self.synthetic_candles(360), "5m", [1])
        frame = supervised.frame.dropna(subset=["target_up_h1"]).tail(260)
        model = make_estimator("temporal_selector_fast", AppConfig())
        model.fit(frame[supervised.feature_columns], frame["target_up_h1"].to_numpy(dtype=int))
        prob = predict_proba_up(model, frame[supervised.feature_columns].tail(5))
        self.assertEqual(len(prob), 5)
        self.assertTrue(np.all((prob > 0.0) & (prob < 1.0)))

    def test_adaptive_ensemble_candidate_fits(self) -> None:
        candles = self.synthetic_candles(460)
        candles["bybit_close"] = candles["close"] * (1.0 + 0.0001 * np.sin(np.arange(len(candles)) / 5.0))
        candles["bybit_return_1"] = np.log(candles["bybit_close"] / candles["bybit_close"].shift(1))
        candles["futures_bybit_spread_bps"] = (candles["close"] / candles["bybit_close"] - 1.0) * 10_000.0
        supervised = build_supervised_frame(candles, "5m", [1])
        frame = supervised.frame.dropna(subset=["target_up_h1"]).tail(360)
        cfg = AppConfig()
        cfg.models.use_gpu = False
        model = make_estimator("adaptive_ensemble_fast", cfg)
        model.fit(frame[supervised.feature_columns], frame["target_up_h1"].to_numpy(dtype=int))
        prob = predict_proba_up(model, frame[supervised.feature_columns].tail(8))
        self.assertEqual(len(prob), 8)
        self.assertTrue(np.all((prob > 0.0) & (prob < 1.0)))

    def test_context_features_are_shifted_to_completed_candle(self) -> None:
        candles = self.synthetic_candles(100)
        ctx = context_features(candles, "15m")
        self.assertIn("ctx_15m_return_1", ctx.columns)
        first_source_open = pd.Timestamp(candles["open_time"].iloc[0])
        self.assertGreaterEqual(pd.Timestamp(ctx["open_time"].iloc[0]), first_source_open + pd.Timedelta(minutes=15))

    def test_walk_forward_splits_embargo(self) -> None:
        cfg = AppConfig()
        cfg.walk_forward.splits = 3
        cfg.walk_forward.min_train_size = 100
        cfg.walk_forward.test_size = 50
        splits = make_walk_forward_splits(400, cfg, embargo=2)
        self.assertEqual(len(splits), 3)
        for split in splits:
            self.assertGreaterEqual(split.test_start - split.train_end, 2)
            self.assertGreater(split.test_end, split.test_start)

    def test_metrics_confusion(self) -> None:
        metrics = metric_dict(
            np.array([1, 1, 0, 0]),
            np.array([0.9, 0.4, 0.3, 0.8]),
            high_confidence_threshold=0.7,
        )
        self.assertEqual(metrics["n"], 4)
        self.assertEqual(metrics["confusion"]["up_up"], 1)
        self.assertEqual(metrics["confusion"]["up_down"], 1)
        self.assertEqual(metrics["confusion"]["down_down"], 1)
        self.assertEqual(metrics["confusion"]["down_up"], 1)
        self.assertAlmostEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["high_confidence_n"], 3)

    def test_acceptance_requires_overall_accuracy_gate(self) -> None:
        cfg = AppConfig()
        candidate = CandidateResult(
            name="selective_only",
            metrics={
                "accuracy": 0.55,
                "high_confidence_accuracy": 0.72,
                "high_confidence_coverage": 0.25,
                "confidence_threshold_sweep": [
                    {"threshold": 0.65, "accuracy": 0.72, "coverage": 0.25},
                ],
            },
        )
        accepted, reason = _acceptance(candidate, {"majority": {"accuracy": 0.51}}, cfg)
        self.assertFalse(accepted)
        self.assertIn("all-prediction gate", reason)

    def test_paper_recording_deduplicates_same_signal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = AppConfig()
            cfg.data.paper_db = Path(tmp) / "paper.sqlite3"
            row = {
                "created_at": "2026-05-20T00:00:00+00:00",
                "candle_time": "2026-05-20T00:00:00+00:00",
                "target_time": "2026-05-20T00:05:00+00:00",
                "symbol": "BTCUSDT",
                "timeframe": "5m",
                "horizon": 1,
                "model_name": "unit_model",
                "accepted": False,
                "base_close": 100_000.0,
                "prob_up": 0.55,
                "confidence": 0.55,
                "prediction": "UP",
            }
            inserted = record_predictions(cfg, [row], artifact_created_at="artifact-a")
            duplicate = record_predictions(cfg, [row], artifact_created_at="artifact-a")
            self.assertEqual(inserted, 1)
            self.assertEqual(duplicate, 0)
            self.assertEqual(paper_summary(cfg)["n"], 0)

    def test_paper_recording_skips_stale_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = AppConfig()
            cfg.data.paper_db = Path(tmp) / "paper.sqlite3"
            row = {
                "created_at": "2026-05-20T00:10:00+00:00",
                "candle_time": "2026-05-20T00:00:00+00:00",
                "target_time": "2026-05-20T00:05:00+00:00",
                "symbol": "BTCUSDT",
                "timeframe": "5m",
                "horizon": 1,
                "model_name": "unit_model",
                "accepted": False,
                "base_close": 100_000.0,
                "prob_up": 0.55,
                "confidence": 0.55,
                "prediction": "UP",
            }
            self.assertEqual(record_predictions(cfg, [row], artifact_created_at="artifact-a"), 0)

    def test_yaml_config_and_registry(self) -> None:
        cfg = load_config(Path("configs/backtest.yaml"))
        self.assertEqual(cfg.data.base_interval, "1m")
        self.assertIn("5m", cfg.prediction.timeframes)
        with tempfile.TemporaryDirectory() as tmp:
            cfg.data.experiment_db = Path(tmp) / "experiments.sqlite3"
            candidate = CandidateResult(
                name="logistic",
                metrics={
                    "accuracy": 0.55,
                    "high_confidence_accuracy": 0.58,
                    "high_confidence_coverage": 0.20,
                },
            )
            task = TaskResult(
                timeframe="5m",
                horizon=1,
                n_samples=1000,
                feature_count=10,
                splits=[],
                candidates=[candidate],
                baselines={"majority": {"accuracy": 0.52}},
                best_candidate=candidate,
                accepted=False,
                acceptance_reason="weak candidate",
                fitted_model=None,
            )
            digest = config_digest(cfg, extra={"command": "test", "profile": "unit"})
            run_id = start_run(cfg, command="test", profile="unit", config_hash=digest)
            finish_run(cfg, run_id=run_id, tasks=[task])
            rows = leaderboard_rows(cfg, limit=5)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["run_id"], run_id)
            payload = get_run(cfg, run_id)
            self.assertIsNotNone(payload)
            self.assertEqual(payload["run"]["accepted_count"], 0)


if __name__ == "__main__":
    unittest.main()
