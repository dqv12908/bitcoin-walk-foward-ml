from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .utils import timeframe_to_minutes, timeframe_to_timedelta

BASE_EXCLUDE = {
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "trade_count",
    "taker_buy_base",
    "taker_buy_quote",
}


@dataclass(slots=True)
class SupervisedFrame:
    frame: pd.DataFrame
    feature_columns: list[str]
    target_columns: dict[int, str]


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    avg_up = up.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_down = down.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    rs = avg_up / avg_down.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    min_periods = min(window, max(2, window // 3))
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std(ddof=0)
    return (series - mean) / std.replace(0.0, np.nan)


def _safe_log_ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    return np.log(a.clip(lower=1e-12) / b.clip(lower=1e-12))


def _threshold_position(signal: pd.Series, entry_threshold: float = 0.15, exit_threshold: float = 0.05) -> pd.Series:
    values = pd.to_numeric(signal, errors="coerce").to_numpy(dtype=float)
    out = np.zeros(len(values), dtype=float)
    pos = 0.0
    for idx, value in enumerate(values):
        if not np.isfinite(value):
            out[idx] = 0.0
            continue
        if pos == 0.0:
            if value > entry_threshold:
                pos = 1.0
            elif value < -entry_threshold:
                pos = -1.0
        elif pos == 1.0:
            if value < -entry_threshold:
                pos = -1.0
            elif value < exit_threshold:
                pos = 0.0
        elif pos == -1.0:
            if value > entry_threshold:
                pos = 1.0
            elif value > -exit_threshold:
                pos = 0.0
        out[idx] = pos
    return pd.Series(out, index=signal.index)


def _volatility_label_threshold(df: pd.DataFrame, mode: str, horizon: int, base_threshold: float) -> pd.Series:
    factor = 0.35
    if any(token in mode for token in ("025", "25pct", "quarter")):
        factor = 0.25
    elif any(token in mode for token in ("050", "50pct", "half")):
        factor = 0.50
    elif any(token in mode for token in ("075", "75pct")):
        factor = 0.75
    if "range_return" in df.columns:
        range_return = pd.to_numeric(df["range_return"], errors="coerce")
    elif {"high", "low"}.issubset(df.columns):
        range_return = _safe_log_ratio(pd.to_numeric(df["high"], errors="coerce"), pd.to_numeric(df["low"], errors="coerce"))
    else:
        range_return = pd.Series(np.nan, index=df.index)
    rolling_range = range_return.rolling(55, min_periods=10).median()
    scaled = rolling_range * factor * math.sqrt(max(1, int(horizon)))
    return scaled.clip(lower=float(base_threshold))


def _binary_target(
    df: pd.DataFrame,
    *,
    horizon: int,
    label_mode: str,
    min_move_bps: float,
) -> tuple[pd.Series, pd.Series]:
    mode = label_mode.strip().lower()
    threshold = float(min_move_bps) / 10_000.0
    if mode in {"close_to_close", "forward_close", "return"}:
        anchor = df["close"]
        future = df["close"].shift(-horizon)
        dynamic_threshold = None
    elif mode in {
        "volatility_adjusted",
        "volatility_adjusted_close",
        "vol_adjusted",
        "volatility_adjusted_025",
        "volatility_adjusted_050",
        "volatility_adjusted_075",
        "volatility_adjusted_25pct",
        "volatility_adjusted_50pct",
        "volatility_adjusted_75pct",
    }:
        anchor = df["close"]
        future = df["close"].shift(-horizon)
        dynamic_threshold = _volatility_label_threshold(df, mode, horizon, threshold)
    elif mode in {"target_body", "body", "open_to_close"}:
        anchor = df["open"].shift(-horizon)
        future = df["close"].shift(-horizon)
        dynamic_threshold = None
    elif mode in {"next_open_to_future_close", "future_from_next_open"}:
        anchor = df["open"].shift(-1)
        future = df["close"].shift(-horizon)
        dynamic_threshold = None
    else:
        raise ValueError(f"unknown label_mode: {label_mode}")

    move = future / anchor.replace(0.0, np.nan) - 1.0
    target = pd.Series(np.where(move >= threshold, 1.0, 0.0), index=df.index)
    if dynamic_threshold is not None:
        threshold_series = dynamic_threshold.fillna(threshold)
        target.loc[move.abs() < threshold_series] = np.nan
    elif threshold > 0:
        target.loc[move.abs() < threshold] = np.nan
    target.loc[future.isna() | anchor.isna()] = np.nan
    return target, future


def _feature_group(column: str) -> str:
    if column in {
        "minute_of_day_sin",
        "minute_of_day_cos",
        "minute_of_week_sin",
        "minute_of_week_cos",
        "weekday",
        "hour",
        "minute_of_day",
        "minute_of_week",
    }:
        return "calendar"
    if column.startswith("ctx_"):
        return "context"
    if (
        column.startswith("open_interest")
        or column.startswith("funding")
        or column.startswith("binance_metrics_")
        or column.startswith("bybit_open_interest")
        or column.startswith("bybit_funding")
        or column.startswith("okx_open_interest")
        or column.startswith("okx_long_short_ratio")
        or column.startswith("binance_bybit_open_interest")
        or column.startswith("binance_minus_bybit_open_interest")
        or column.startswith("binance_minus_bybit_funding")
        or column.startswith("binance_okx_open_interest")
        or column.startswith("binance_minus_okx_open_interest")
        or column.startswith("global_lsr_")
        or column.startswith("top_account_lsr_")
        or column.startswith("top_position_lsr_")
        or column.startswith("taker_lsr_")
        or column.startswith("mark_price_")
        or column.startswith("index_price_")
        or column.startswith("premium_index_")
        or column.startswith("futures_mark_spread")
        or column.startswith("futures_minus_index_return")
        or column.startswith("mark_index_basis")
    ):
        return "derivatives"
    if (
        column.startswith("spot_")
        or "spot" in column
        or "hyperliquid" in column
        or column.startswith("coinbase_")
        or "coinbase" in column
        or column.startswith("bybit_")
        or "bybit" in column
        or column.startswith("okx_")
        or "okx" in column
        or column.startswith(("eth_", "sol_", "bnb_", "xrp_", "doge_", "ada_"))
        or any(token in column for token in ("eth", "sol", "bnb", "xrp", "doge", "ada"))
        or column.startswith("cross_asset_")
        or column.startswith(("futures_eth", "futures_sol", "futures_bnb", "futures_xrp", "futures_doge", "futures_ada"))
    ):
        return "cross_market"
    if (
        column.startswith("volume")
        or column.startswith("log_volume")
        or column.startswith("log_quote_volume")
        or column.startswith("log_trade_count")
        or column.startswith("ltf_quote_volume")
        or column.startswith("ltf_log_quote_volume")
        or column.startswith("ltf_trade_count")
        or column.startswith("ltf_taker")
        or column.startswith("buy_imbalance")
        or column.startswith("taker")
        or column.startswith("agg_")
        or column.startswith("live_")
        or column.startswith("liq_")
        or column.startswith("book_")
    ):
        return "flow"
    if (
        column.startswith("return_sum")
        or column.startswith("return_mean")
        or column.startswith("return_std")
        or column.startswith("range_mean")
        or column.startswith("close_z")
        or column.startswith("ema_gap")
        or column.startswith("macd")
        or column.startswith("rsi")
        or column.startswith("breakout")
        or column.startswith("close_vs_vwap")
        or column.startswith("ltf_return")
        or column.startswith("ltf_body")
        or column.startswith("ltf_range")
        or column.startswith("ltf_up")
        or column.startswith("ltf_down")
        or column.startswith("ltf_close")
        or column.startswith("ltf_late")
        or column.startswith("inwin_")
        or column.startswith("signal_")
    ):
        return "technical"
    return "price_action"


def build_supervised_frame(
    candles: pd.DataFrame,
    timeframe: str,
    horizons: list[int],
    *,
    label_mode: str = "close_to_close",
    min_move_bps: float = 0.0,
    feature_groups: list[str] | None = None,
) -> SupervisedFrame:
    df = candles.copy()
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True, format="mixed")
    df = df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    for col in ("open", "high", "low", "close", "volume", "quote_volume", "trade_count", "taker_buy_base", "taker_buy_quote"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    eps = 1e-12
    df["return_1"] = _safe_log_ratio(df["close"], df["close"].shift(1))
    df["body_return"] = _safe_log_ratio(df["close"], df["open"])
    df["range_return"] = _safe_log_ratio(df["high"], df["low"])
    df["upper_wick_bps"] = (df["high"] / np.maximum(df["open"], df["close"]) - 1.0) * 10_000.0
    df["lower_wick_bps"] = (np.minimum(df["open"], df["close"]) / df["low"].replace(0.0, np.nan) - 1.0) * 10_000.0
    df["close_position"] = (df["close"] - df["low"]) / (df["high"] - df["low"]).replace(0.0, np.nan)
    df["log_volume"] = np.log1p(df.get("volume", pd.Series(0.0, index=df.index)).clip(lower=0.0))
    df["log_quote_volume"] = np.log1p(df.get("quote_volume", pd.Series(0.0, index=df.index)).clip(lower=0.0))
    df["log_trade_count"] = np.log1p(df.get("trade_count", pd.Series(0.0, index=df.index)).clip(lower=0.0))
    if "taker_buy_base" in df.columns and "volume" in df.columns:
        df["taker_buy_ratio"] = df["taker_buy_base"] / df["volume"].replace(0.0, np.nan)
        df["taker_buy_imbalance"] = 2.0 * df["taker_buy_ratio"] - 1.0
    else:
        df["taker_buy_ratio"] = np.nan
        df["taker_buy_imbalance"] = np.nan

    extra: dict[str, pd.Series | np.ndarray] = {}
    windows = [2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
    for window in windows:
        min_periods = max(2, min(window, window // 3))
        extra[f"return_sum_{window}"] = df["return_1"].rolling(window, min_periods=min_periods).sum()
        extra[f"return_mean_{window}"] = df["return_1"].rolling(window, min_periods=min_periods).mean()
        extra[f"return_std_{window}"] = df["return_1"].rolling(window, min_periods=min_periods).std(ddof=0)
        extra[f"range_mean_{window}"] = df["range_return"].rolling(window, min_periods=min_periods).mean()
        extra[f"close_z_{window}"] = _rolling_zscore(df["close"], window)
        extra[f"volume_z_{window}"] = _rolling_zscore(df["log_quote_volume"], window)
        extra[f"buy_imbalance_mean_{window}"] = df["taker_buy_imbalance"].rolling(window, min_periods=min_periods).mean()
        rolling_high = df["high"].rolling(window, min_periods=min_periods).max()
        rolling_low = df["low"].rolling(window, min_periods=min_periods).min()
        extra[f"breakout_high_{window}"] = df["close"] / rolling_high.replace(0.0, np.nan) - 1.0
        extra[f"breakout_low_{window}"] = df["close"] / rolling_low.replace(0.0, np.nan) - 1.0
        if "quote_volume" in df.columns and "volume" in df.columns:
            rolling_vwap = df["quote_volume"].rolling(window, min_periods=min_periods).sum() / df["volume"].rolling(
                window, min_periods=min_periods
            ).sum().replace(0.0, np.nan)
            extra[f"close_vs_vwap_{window}"] = df["close"] / rolling_vwap - 1.0

    for span in (5, 12, 21, 55):
        ema = df["close"].ewm(span=span, adjust=False, min_periods=span).mean()
        extra[f"ema_gap_{span}"] = df["close"] / ema.replace(0.0, np.nan) - 1.0
    ema_fast = df["close"].ewm(span=12, adjust=False, min_periods=12).mean()
    ema_slow = df["close"].ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema_fast - ema_slow
    extra["macd_bps"] = macd / df["close"].replace(0.0, np.nan) * 10_000.0
    extra["macd_signal_bps"] = macd.ewm(span=9, adjust=False, min_periods=9).mean() / df["close"].replace(0.0, np.nan) * 10_000.0

    for window in (7, 14, 28):
        extra[f"rsi_{window}"] = _rsi(df["close"], window)

    close_mean_20 = df["close"].rolling(20, min_periods=7).mean()
    close_std_20 = df["close"].rolling(20, min_periods=7).std(ddof=0)
    signal_zscore_mr = -((df["close"] - close_mean_20) / close_std_20.replace(0.0, np.nan))
    extra["signal_zscore_mr_20"] = signal_zscore_mr
    extra["signal_bollinger_mr_20"] = signal_zscore_mr.clip(-1.0, 1.0) / 2.0
    rolling_vwap_15 = df["quote_volume"].rolling(15, min_periods=5).sum() / df["volume"].rolling(
        15, min_periods=5
    ).sum().replace(0.0, np.nan)
    vwap_deviation = df["close"] / rolling_vwap_15.replace(0.0, np.nan) - 1.0
    extra["signal_vwap_mr_15"] = -(vwap_deviation / vwap_deviation.rolling(15, min_periods=5).std(ddof=0).replace(0.0, np.nan)).clip(-3.0, 3.0)
    if "agg_taker_buy_qty" in df.columns and "agg_taker_sell_qty" in df.columns:
        buy_volume = pd.to_numeric(df["agg_taker_buy_qty"], errors="coerce")
        sell_volume = pd.to_numeric(df["agg_taker_sell_qty"], errors="coerce")
    else:
        buy_volume = pd.to_numeric(df.get("taker_buy_base", pd.Series(np.nan, index=df.index)), errors="coerce")
        sell_volume = pd.to_numeric(df.get("volume", pd.Series(np.nan, index=df.index)), errors="coerce") - buy_volume
    net_flow = (buy_volume - sell_volume) / (buy_volume + sell_volume).replace(0.0, np.nan)
    ofi_spread = net_flow.rolling(5, min_periods=2).mean() - net_flow.rolling(20, min_periods=7).mean()
    extra["signal_ofi_contrarian_5_20"] = -(ofi_spread / ofi_spread.rolling(20, min_periods=7).std(ddof=0).replace(0.0, np.nan)).clip(-3.0, 3.0)
    intensity_count = pd.to_numeric(df.get("agg_trade_count", df.get("trade_count", pd.Series(np.nan, index=df.index))), errors="coerce")
    intensity_z = (intensity_count - intensity_count.rolling(20, min_periods=7).mean()) / intensity_count.rolling(20, min_periods=7).std(ddof=0).replace(0.0, np.nan)
    extra["signal_trade_intensity_20"] = (intensity_z * np.sign(df["return_1"])).clip(-3.0, 3.0)
    rsi_14 = _rsi(df["close"], 14)
    extra["signal_rsi_mr_14"] = (-(rsi_14 - 50.0) / 20.0).clip(-1.5, 1.5)
    ema_fast_signal = df["close"].ewm(span=5, adjust=False, min_periods=5).mean()
    ema_slow_signal = df["close"].ewm(span=15, adjust=False, min_periods=15).mean()
    ma_spread = ema_fast_signal - ema_slow_signal
    extra["signal_ma_cross_mr_5_15"] = -(ma_spread / df["close"].rolling(15, min_periods=5).std(ddof=0).replace(0.0, np.nan)).clip(-3.0, 3.0)
    rel_volume = df["volume"] / df["volume"].rolling(10, min_periods=4).mean().replace(0.0, np.nan)
    vol_weighted_return = df["return_1"] * rel_volume
    vol_momentum = vol_weighted_return.rolling(10, min_periods=4).sum()
    extra["signal_vol_momentum_mr_10"] = -(vol_momentum / vol_momentum.rolling(20, min_periods=7).std(ddof=0).replace(0.0, np.nan)).clip(-3.0, 3.0)
    signal_weights = {
        "signal_zscore_mr_20": 0.15,
        "signal_bollinger_mr_20": 0.15,
        "signal_vwap_mr_15": 0.20,
        "signal_ofi_contrarian_5_20": 0.15,
        "signal_trade_intensity_20": 0.05,
        "signal_rsi_mr_14": 0.10,
        "signal_ma_cross_mr_5_15": 0.10,
        "signal_vol_momentum_mr_10": 0.10,
    }
    signal_frame = pd.DataFrame({name: pd.to_numeric(extra[name], errors="coerce") for name in signal_weights}, index=df.index)
    signal_composite = sum(signal_frame[name] * weight for name, weight in signal_weights.items())
    extra["signal_composite_raw"] = signal_composite
    extra["signal_composite_ema5"] = signal_composite.ewm(span=5, adjust=False, min_periods=3).mean()
    extra["signal_position_015_005"] = _threshold_position(extra["signal_composite_ema5"], 0.15, 0.05)
    extra["signal_abs_composite"] = extra["signal_composite_ema5"].abs()

    open_time = pd.to_datetime(df["open_time"], utc=True)
    minute_of_day = open_time.dt.hour * 60 + open_time.dt.minute
    minute_of_week = open_time.dt.weekday * 1440 + minute_of_day
    extra["minute_of_day_sin"] = np.sin(2.0 * math.pi * minute_of_day / 1440.0)
    extra["minute_of_day_cos"] = np.cos(2.0 * math.pi * minute_of_day / 1440.0)
    extra["minute_of_week_sin"] = np.sin(2.0 * math.pi * minute_of_week / (7.0 * 1440.0))
    extra["minute_of_week_cos"] = np.cos(2.0 * math.pi * minute_of_week / (7.0 * 1440.0))
    extra["weekday"] = open_time.dt.weekday.astype(float)
    extra["hour"] = open_time.dt.hour.astype(float)
    extra["minute_of_day"] = minute_of_day.astype(float)
    extra["minute_of_week"] = minute_of_week.astype(float)

    current_minutes = timeframe_to_minutes(timeframe)
    for anchor_name, anchor_minutes in (("15m", 15), ("30m", 30), ("1h", 60), ("4h", 240)):
        if anchor_minutes <= current_minutes or anchor_minutes % current_minutes != 0:
            continue
        anchor_key = open_time.dt.floor(f"{anchor_minutes}min")
        steps = max(1, anchor_minutes // current_minutes)
        bar_index = ((minute_of_day % anchor_minutes) // current_minutes).astype(float)
        anchor_open = df.groupby(anchor_key, sort=False)["open"].transform("first")
        anchor_high_so_far = df.groupby(anchor_key, sort=False)["high"].cummax()
        anchor_low_so_far = df.groupby(anchor_key, sort=False)["low"].cummin()
        inwin_return = df["close"] / anchor_open.replace(0.0, np.nan) - 1.0
        inwin_range = (anchor_high_so_far - anchor_low_so_far).replace(0.0, np.nan)
        elapsed = (bar_index + 1.0) / float(steps)
        prefix = f"inwin_{anchor_name}_"
        extra[f"{prefix}bar_index"] = bar_index
        extra[f"{prefix}elapsed"] = elapsed
        extra[f"{prefix}return"] = inwin_return
        extra[f"{prefix}return_bps"] = inwin_return * 10_000.0
        extra[f"{prefix}direction"] = np.sign(inwin_return)
        extra[f"{prefix}close_position"] = (df["close"] - anchor_low_so_far) / inwin_range
        extra[f"{prefix}range_bps"] = inwin_range / anchor_open.replace(0.0, np.nan) * 10_000.0
        extra[f"{prefix}impulse_bps"] = (inwin_return * 10_000.0) / np.sqrt(np.maximum(elapsed, 1e-6))

    if "spot_close" in df.columns:
        df["spot_return_1"] = pd.to_numeric(df["spot_return_1"], errors="coerce")
        extra["spot_return_sum_12"] = df["spot_return_1"].rolling(12, min_periods=4).sum()
        extra["futures_minus_spot_return"] = df["return_1"] - df["spot_return_1"]
    for spread_col in (
        "futures_spot_spread_bps",
        "futures_hyperliquid_spread_bps",
        "futures_coinbase_spread_bps",
        "futures_bybit_spread_bps",
        "futures_okx_spread_bps",
    ):
        if spread_col in df.columns:
            spread = pd.to_numeric(df[spread_col], errors="coerce")
            extra[f"{spread_col}_change_1"] = spread.diff()
            extra[f"{spread_col}_mean_21"] = spread.rolling(21, min_periods=7).mean()
            extra[f"{spread_col}_z_55"] = _rolling_zscore(spread, 55)
    if "hyperliquid_close" in df.columns:
        df["hyperliquid_return_1"] = pd.to_numeric(df["hyperliquid_return_1"], errors="coerce")
        extra["hyperliquid_return_sum_12"] = df["hyperliquid_return_1"].rolling(12, min_periods=4).sum()
        extra["futures_minus_hyperliquid_return"] = df["return_1"] - df["hyperliquid_return_1"]
        if "hyperliquid_quote_volume" in df.columns:
            extra["hyperliquid_quote_volume_z_21"] = _rolling_zscore(
                np.log1p(pd.to_numeric(df["hyperliquid_quote_volume"], errors="coerce").clip(lower=0.0)),
                21,
            )
    if "coinbase_close" in df.columns:
        df["coinbase_return_1"] = pd.to_numeric(df["coinbase_return_1"], errors="coerce")
        extra["coinbase_return_sum_12"] = df["coinbase_return_1"].rolling(12, min_periods=4).sum()
        extra["futures_minus_coinbase_return"] = df["return_1"] - df["coinbase_return_1"]
        if "coinbase_quote_volume" in df.columns:
            extra["coinbase_quote_volume_z_21"] = _rolling_zscore(
                np.log1p(pd.to_numeric(df["coinbase_quote_volume"], errors="coerce").clip(lower=0.0)),
                21,
            )
    if "bybit_close" in df.columns:
        df["bybit_return_1"] = pd.to_numeric(df["bybit_return_1"], errors="coerce")
        extra["bybit_return_sum_12"] = df["bybit_return_1"].rolling(12, min_periods=4).sum()
        extra["futures_minus_bybit_return"] = df["return_1"] - df["bybit_return_1"]
        if "bybit_quote_volume" in df.columns:
            extra["bybit_quote_volume_z_21"] = _rolling_zscore(
                np.log1p(pd.to_numeric(df["bybit_quote_volume"], errors="coerce").clip(lower=0.0)),
                21,
            )
        if "coinbase_close" in df.columns:
            extra["bybit_minus_coinbase_return"] = df["bybit_return_1"] - pd.to_numeric(df["coinbase_return_1"], errors="coerce")
            bybit_coinbase_spread = (df["bybit_close"] / pd.to_numeric(df["coinbase_close"], errors="coerce").replace(0.0, np.nan) - 1.0) * 10_000.0
            extra["bybit_coinbase_spread_bps"] = bybit_coinbase_spread
            extra["bybit_coinbase_spread_bps_z_55"] = _rolling_zscore(bybit_coinbase_spread, 55)
        if "hyperliquid_close" in df.columns:
            extra["bybit_minus_hyperliquid_return"] = df["bybit_return_1"] - pd.to_numeric(df["hyperliquid_return_1"], errors="coerce")
            bybit_hyperliquid_spread = (df["bybit_close"] / pd.to_numeric(df["hyperliquid_close"], errors="coerce").replace(0.0, np.nan) - 1.0) * 10_000.0
            extra["bybit_hyperliquid_spread_bps"] = bybit_hyperliquid_spread
            extra["bybit_hyperliquid_spread_bps_z_55"] = _rolling_zscore(bybit_hyperliquid_spread, 55)
        if "spot_close" in df.columns:
            bybit_spot_spread = (df["bybit_close"] / pd.to_numeric(df["spot_close"], errors="coerce").replace(0.0, np.nan) - 1.0) * 10_000.0
            extra["bybit_spot_spread_bps"] = bybit_spot_spread
            extra["bybit_spot_spread_bps_z_55"] = _rolling_zscore(bybit_spot_spread, 55)
    if "okx_close" in df.columns:
        df["okx_return_1"] = pd.to_numeric(df["okx_return_1"], errors="coerce")
        extra["okx_return_sum_12"] = df["okx_return_1"].rolling(12, min_periods=4).sum()
        extra["futures_minus_okx_return"] = df["return_1"] - df["okx_return_1"]
        if "okx_quote_volume" in df.columns:
            extra["okx_quote_volume_z_21"] = _rolling_zscore(
                np.log1p(pd.to_numeric(df["okx_quote_volume"], errors="coerce").clip(lower=0.0)),
                21,
            )
        if "coinbase_close" in df.columns:
            extra["okx_minus_coinbase_return"] = df["okx_return_1"] - pd.to_numeric(df["coinbase_return_1"], errors="coerce")
            okx_coinbase_spread = (df["okx_close"] / pd.to_numeric(df["coinbase_close"], errors="coerce").replace(0.0, np.nan) - 1.0) * 10_000.0
            extra["okx_coinbase_spread_bps"] = okx_coinbase_spread
            extra["okx_coinbase_spread_bps_z_55"] = _rolling_zscore(okx_coinbase_spread, 55)
        if "bybit_close" in df.columns:
            extra["okx_minus_bybit_return"] = df["okx_return_1"] - pd.to_numeric(df["bybit_return_1"], errors="coerce")
            okx_bybit_spread = (df["okx_close"] / pd.to_numeric(df["bybit_close"], errors="coerce").replace(0.0, np.nan) - 1.0) * 10_000.0
            extra["okx_bybit_spread_bps"] = okx_bybit_spread
            extra["okx_bybit_spread_bps_z_55"] = _rolling_zscore(okx_bybit_spread, 55)
        if "hyperliquid_close" in df.columns:
            extra["okx_minus_hyperliquid_return"] = df["okx_return_1"] - pd.to_numeric(df["hyperliquid_return_1"], errors="coerce")
            okx_hyperliquid_spread = (df["okx_close"] / pd.to_numeric(df["hyperliquid_close"], errors="coerce").replace(0.0, np.nan) - 1.0) * 10_000.0
            extra["okx_hyperliquid_spread_bps"] = okx_hyperliquid_spread
            extra["okx_hyperliquid_spread_bps_z_55"] = _rolling_zscore(okx_hyperliquid_spread, 55)
        if "spot_close" in df.columns:
            okx_spot_spread = (df["okx_close"] / pd.to_numeric(df["spot_close"], errors="coerce").replace(0.0, np.nan) - 1.0) * 10_000.0
            extra["okx_spot_spread_bps"] = okx_spot_spread
            extra["okx_spot_spread_bps_z_55"] = _rolling_zscore(okx_spot_spread, 55)
    alt_returns: list[pd.Series] = []
    for prefix in ("eth", "sol", "bnb", "xrp", "doge", "ada"):
        close_col = f"{prefix}_close"
        return_col = f"{prefix}_return_1"
        if close_col not in df.columns or return_col not in df.columns:
            continue
        df[return_col] = pd.to_numeric(df[return_col], errors="coerce")
        alt_returns.append(df[return_col])
        extra[f"{prefix}_return_sum_12"] = df[return_col].rolling(12, min_periods=4).sum()
        extra[f"futures_minus_{prefix}_return"] = df["return_1"] - df[return_col]
        if f"{prefix}_quote_volume" in df.columns:
            extra[f"{prefix}_quote_volume_z_21"] = _rolling_zscore(
                np.log1p(pd.to_numeric(df[f"{prefix}_quote_volume"], errors="coerce").clip(lower=0.0)),
                21,
            )
        spread_col = f"futures_{prefix}_spread_bps"
        if spread_col in df.columns:
            spread = pd.to_numeric(df[spread_col], errors="coerce")
            extra[f"{spread_col}_change_1"] = spread.diff()
            extra[f"{spread_col}_mean_21"] = spread.rolling(21, min_periods=7).mean()
            extra[f"{spread_col}_z_55"] = _rolling_zscore(spread, 55)
    if len(alt_returns) >= 2:
        alt_market_return = pd.concat(alt_returns, axis=1).mean(axis=1)
        beta = df["return_1"].rolling(55, min_periods=20).cov(alt_market_return) / alt_market_return.rolling(55, min_periods=20).var(ddof=0).replace(0.0, np.nan)
        residual = df["return_1"] - beta * alt_market_return
        residual_path = residual.fillna(0.0).rolling(55, min_periods=20).sum()
        extra["cross_asset_market_return"] = alt_market_return
        extra["cross_asset_beta_55"] = beta
        extra["cross_asset_residual_return"] = residual
        extra["cross_asset_residual_sum_55"] = residual_path
        extra["cross_asset_residual_z_55"] = _rolling_zscore(residual_path, 55)
    if "sum_open_interest" in df.columns:
        open_interest_log = np.log(pd.to_numeric(df["sum_open_interest"], errors="coerce").clip(lower=eps))
        extra["open_interest_log"] = open_interest_log
        extra["open_interest_return_1"] = open_interest_log.diff()
        extra["open_interest_z_55"] = _rolling_zscore(open_interest_log, 55)
        extra["open_interest_value_z_55"] = _rolling_zscore(
            np.log(pd.to_numeric(df["sum_open_interest_value"], errors="coerce").clip(lower=eps)),
            55,
        )
    if "funding_rate" in df.columns:
        df["funding_rate"] = pd.to_numeric(df["funding_rate"], errors="coerce")
        extra["funding_rate_z_21"] = _rolling_zscore(df["funding_rate"], 21)
    if "bybit_open_interest" in df.columns:
        bybit_oi_log = np.log(pd.to_numeric(df["bybit_open_interest"], errors="coerce").clip(lower=eps))
        extra["bybit_open_interest_log"] = bybit_oi_log
        extra["bybit_open_interest_return_1"] = bybit_oi_log.diff()
        extra["bybit_open_interest_z_55"] = _rolling_zscore(bybit_oi_log, 55)
        if "sum_open_interest" in df.columns:
            binance_oi_log = np.log(pd.to_numeric(df["sum_open_interest"], errors="coerce").clip(lower=eps))
            extra["binance_minus_bybit_open_interest_return"] = binance_oi_log.diff() - bybit_oi_log.diff()
            extra["binance_bybit_open_interest_ratio_log"] = binance_oi_log - bybit_oi_log
    if "bybit_funding_rate" in df.columns:
        df["bybit_funding_rate"] = pd.to_numeric(df["bybit_funding_rate"], errors="coerce")
        extra["bybit_funding_rate_z_21"] = _rolling_zscore(df["bybit_funding_rate"], 21)
        if "funding_rate" in df.columns:
            extra["binance_minus_bybit_funding_rate"] = df["funding_rate"] - df["bybit_funding_rate"]
    if "okx_open_interest_usd" in df.columns:
        okx_oi_log = np.log(pd.to_numeric(df["okx_open_interest_usd"], errors="coerce").clip(lower=eps))
        extra["okx_open_interest_usd_log"] = okx_oi_log
        extra["okx_open_interest_usd_return_1"] = okx_oi_log.diff()
        extra["okx_open_interest_usd_z_55"] = _rolling_zscore(okx_oi_log, 55)
        if "sum_open_interest_value" in df.columns:
            binance_oi_value_log = np.log(pd.to_numeric(df["sum_open_interest_value"], errors="coerce").clip(lower=eps))
            extra["binance_minus_okx_open_interest_value_return"] = binance_oi_value_log.diff() - okx_oi_log.diff()
            extra["binance_okx_open_interest_value_ratio_log"] = binance_oi_value_log - okx_oi_log
    if "okx_long_short_ratio" in df.columns:
        ratio_log = np.log(pd.to_numeric(df["okx_long_short_ratio"], errors="coerce").clip(lower=eps))
        extra["okx_long_short_ratio_log"] = ratio_log
        extra["okx_long_short_ratio_change_1"] = ratio_log.diff()
        extra["okx_long_short_ratio_z_21"] = _rolling_zscore(ratio_log, 21)
        extra["okx_long_short_ratio_z_55"] = _rolling_zscore(ratio_log, 55)
    for col in ("futures_mark_spread_bps", "mark_index_basis_bps", "premium_index_bps"):
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce")
            extra[f"{col}_change_1"] = series.diff()
            extra[f"{col}_mean_21"] = series.rolling(21, min_periods=7).mean()
            extra[f"{col}_z_55"] = _rolling_zscore(series, 55)
    if "mark_price_close" in df.columns:
        mark_close = pd.to_numeric(df["mark_price_close"], errors="coerce")
        extra["mark_price_return_1"] = _safe_log_ratio(mark_close, mark_close.shift(1))
    if "index_price_close" in df.columns:
        index_close = pd.to_numeric(df["index_price_close"], errors="coerce")
        index_return = _safe_log_ratio(index_close, index_close.shift(1))
        extra["index_price_return_1"] = index_return
        extra["futures_minus_index_return"] = df["return_1"] - index_return
    ratio_columns = [
        col
        for col in df.columns
        if col.endswith("_long_short_ratio") or col.endswith("_buy_sell_ratio")
    ]
    for col in ratio_columns:
        ratio = pd.to_numeric(df[col], errors="coerce")
        ratio_log = np.log(ratio.clip(lower=eps))
        extra[f"{col}_log"] = ratio_log
        extra[f"{col}_change_1"] = ratio_log.diff()
        extra[f"{col}_z_21"] = _rolling_zscore(ratio_log, 21)
        extra[f"{col}_z_55"] = _rolling_zscore(ratio_log, 55)
        extra[f"{col}_mean_21"] = ratio_log.rolling(21, min_periods=7).mean()
    for prefix in ("global_lsr", "top_account_lsr", "top_position_lsr"):
        long_col = f"{prefix}_long_account"
        short_col = f"{prefix}_short_account"
        if long_col in df.columns and short_col in df.columns:
            long_account = pd.to_numeric(df[long_col], errors="coerce")
            short_account = pd.to_numeric(df[short_col], errors="coerce")
            spread = long_account - short_account
            extra[f"{prefix}_account_spread"] = spread
            extra[f"{prefix}_account_spread_change_1"] = spread.diff()
            extra[f"{prefix}_account_spread_z_55"] = _rolling_zscore(spread, 55)
    if "taker_lsr_buy_vol" in df.columns and "taker_lsr_sell_vol" in df.columns:
        buy_vol = pd.to_numeric(df["taker_lsr_buy_vol"], errors="coerce")
        sell_vol = pd.to_numeric(df["taker_lsr_sell_vol"], errors="coerce")
        total_vol = buy_vol + sell_vol
        net_share = (buy_vol - sell_vol) / total_vol.replace(0.0, np.nan)
        extra["taker_lsr_net_buy_share"] = net_share
        extra["taker_lsr_net_buy_share_mean_8"] = net_share.rolling(8, min_periods=3).mean()
        extra["taker_lsr_net_buy_share_mean_21"] = net_share.rolling(21, min_periods=7).mean()
        extra["taker_lsr_total_vol_z_55"] = _rolling_zscore(np.log1p(total_vol.clip(lower=0.0)), 55)
    if "agg_trade_quote_qty" in df.columns:
        quote = pd.to_numeric(df["agg_trade_quote_qty"], errors="coerce")
        count = pd.to_numeric(df.get("agg_trade_count", pd.Series(np.nan, index=df.index)), errors="coerce")
        imbalance = pd.to_numeric(df.get("agg_taker_quote_imbalance", pd.Series(np.nan, index=df.index)), errors="coerce")
        extra["agg_trade_quote_z_21"] = _rolling_zscore(np.log1p(quote.clip(lower=0.0)), 21)
        extra["agg_trade_count_z_21"] = _rolling_zscore(np.log1p(count.clip(lower=0.0)), 21)
        extra["agg_taker_quote_imbalance_mean_8"] = imbalance.rolling(8, min_periods=3).mean()
        extra["agg_taker_quote_imbalance_mean_21"] = imbalance.rolling(21, min_periods=7).mean()
        if "agg_trade_vwap" in df.columns:
            vwap = pd.to_numeric(df["agg_trade_vwap"], errors="coerce")
            extra["close_vs_agg_trade_vwap"] = df["close"] / vwap.replace(0.0, np.nan) - 1.0
        if "agg_max_trade_quote" in df.columns:
            extra["agg_max_trade_quote_z_21"] = _rolling_zscore(
                np.log1p(pd.to_numeric(df["agg_max_trade_quote"], errors="coerce").clip(lower=0.0)),
                21,
            )
    if "live_trade_quote_qty" in df.columns:
        quote = pd.to_numeric(df["live_trade_quote_qty"], errors="coerce")
        count = pd.to_numeric(df.get("live_trade_count", pd.Series(np.nan, index=df.index)), errors="coerce")
        imbalance = pd.to_numeric(df.get("live_taker_quote_imbalance", pd.Series(np.nan, index=df.index)), errors="coerce")
        extra["live_trade_quote_z_21"] = _rolling_zscore(np.log1p(quote.clip(lower=0.0)), 21)
        extra["live_trade_count_z_21"] = _rolling_zscore(np.log1p(count.clip(lower=0.0)), 21)
        extra["live_taker_quote_imbalance_mean_5"] = imbalance.rolling(5, min_periods=2).mean()
        extra["live_taker_quote_imbalance_mean_13"] = imbalance.rolling(13, min_periods=4).mean()
        if "live_trade_vwap" in df.columns:
            vwap = pd.to_numeric(df["live_trade_vwap"], errors="coerce")
            extra["close_vs_live_trade_vwap"] = df["close"] / vwap.replace(0.0, np.nan) - 1.0
        if "live_max_trade_quote" in df.columns:
            extra["live_max_trade_quote_z_21"] = _rolling_zscore(
                np.log1p(pd.to_numeric(df["live_max_trade_quote"], errors="coerce").clip(lower=0.0)),
                21,
            )
    if "liq_quote_qty" in df.columns:
        quote = pd.to_numeric(df["liq_quote_qty"], errors="coerce")
        count = pd.to_numeric(df.get("liq_count", pd.Series(np.nan, index=df.index)), errors="coerce")
        imbalance = pd.to_numeric(df.get("liq_buy_sell_imbalance", pd.Series(np.nan, index=df.index)), errors="coerce")
        extra["liq_quote_z_21"] = _rolling_zscore(np.log1p(quote.clip(lower=0.0)), 21)
        extra["liq_count_z_21"] = _rolling_zscore(np.log1p(count.clip(lower=0.0)), 21)
        extra["liq_buy_sell_imbalance_mean_5"] = imbalance.rolling(5, min_periods=2).mean()
        extra["liq_buy_sell_imbalance_mean_13"] = imbalance.rolling(13, min_periods=4).mean()
        extra["liq_pressure"] = imbalance.fillna(0.0) * np.log1p(quote.clip(lower=0.0))
        if "liq_max_quote" in df.columns:
            extra["liq_max_quote_z_21"] = _rolling_zscore(
                np.log1p(pd.to_numeric(df["liq_max_quote"], errors="coerce").clip(lower=0.0)),
                21,
            )
    if "book_imbalance_top20" in df.columns:
        for depth in (5, 20, 100):
            imbalance_col = f"book_imbalance_top{depth}"
            depth_col = f"book_depth_quote_top{depth}"
            if imbalance_col in df.columns:
                imbalance = pd.to_numeric(df[imbalance_col], errors="coerce")
                extra[f"{imbalance_col}_mean_3"] = imbalance.rolling(3, min_periods=1).mean()
                extra[f"{imbalance_col}_change_1"] = imbalance.diff()
            if depth_col in df.columns:
                extra[f"{depth_col}_z_21"] = _rolling_zscore(
                    np.log1p(pd.to_numeric(df[depth_col], errors="coerce").clip(lower=0.0)),
                    21,
                )
        if "book_spread_bps" in df.columns:
            spread = pd.to_numeric(df["book_spread_bps"], errors="coerce")
            extra["book_spread_bps_z_21"] = _rolling_zscore(spread, 21)

    target_columns: dict[int, str] = {}
    delta = timeframe_to_timedelta(timeframe)
    for horizon in horizons:
        target = f"target_up_h{horizon}"
        target_columns[horizon] = target
        target_series, future_close = _binary_target(
            df,
            horizon=horizon,
            label_mode=label_mode,
            min_move_bps=min_move_bps,
        )
        extra[f"future_close_h{horizon}"] = future_close
        extra[f"target_time_h{horizon}"] = df["open_time"] + horizon * delta
        extra[target] = target_series

    if extra:
        df = pd.concat([df, pd.DataFrame(extra, index=df.index)], axis=1).copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    excluded = set(BASE_EXCLUDE)
    for horizon in horizons:
        excluded.update({f"future_close_h{horizon}", f"target_time_h{horizon}", f"target_up_h{horizon}"})
    feature_columns = [
        col
        for col in df.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(df[col])
    ]
    requested_groups = {group.strip().lower() for group in (feature_groups or ["all"])}
    if "all" not in requested_groups:
        feature_columns = [col for col in feature_columns if _feature_group(col) in requested_groups]
    return SupervisedFrame(frame=df, feature_columns=feature_columns, target_columns=target_columns)
