from __future__ import annotations

import asyncio
import json
import os
import re
import time
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

from .config import AppConfig
from .utils import (
    ensure_dirs,
    timeframe_to_timedelta,
    timeframe_to_minutes,
    timeframe_to_pandas_rule,
    to_ms,
    utc_now,
)

BINANCE_SPOT_URL = "https://api.binance.com"
BINANCE_FUTURES_URL = "https://fapi.binance.com"
BINANCE_VISION_URL = "https://data.binance.vision"
HYPERLIQUID_URL = "https://api.hyperliquid.xyz/info"
COINBASE_EXCHANGE_URL = "https://api.exchange.coinbase.com"
BYBIT_URL = "https://api.bybit.com"
OKX_URL = "https://www.okx.com"

KLINE_COLUMNS = [
    "open_time_ms",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time_ms",
    "quote_volume",
    "trade_count",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
]

NUMERIC_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "trade_count",
    "taker_buy_base",
    "taker_buy_quote",
]

FUTURES_RATIO_SOURCES: dict[str, dict[str, object]] = {
    "global_long_short_account": {
        "endpoint": "/futures/data/globalLongShortAccountRatio",
        "columns": {
            "longAccount": "global_lsr_long_account",
            "shortAccount": "global_lsr_short_account",
            "longShortRatio": "global_lsr_long_short_ratio",
        },
    },
    "top_long_short_account": {
        "endpoint": "/futures/data/topLongShortAccountRatio",
        "columns": {
            "longAccount": "top_account_lsr_long_account",
            "shortAccount": "top_account_lsr_short_account",
            "longShortRatio": "top_account_lsr_long_short_ratio",
        },
    },
    "top_long_short_position": {
        "endpoint": "/futures/data/topLongShortPositionRatio",
        "columns": {
            "longAccount": "top_position_lsr_long_account",
            "shortAccount": "top_position_lsr_short_account",
            "longShortRatio": "top_position_lsr_long_short_ratio",
        },
    },
    "taker_long_short": {
        "endpoint": "/futures/data/takerlongshortRatio",
        "columns": {
            "buySellRatio": "taker_lsr_buy_sell_ratio",
            "buyVol": "taker_lsr_buy_vol",
            "sellVol": "taker_lsr_sell_vol",
        },
    },
}
FUTURES_RATIO_MAX_LOOKBACK = timedelta(days=29)

FUTURES_BASIS_SOURCES: dict[str, dict[str, str]] = {
    "mark_price": {"endpoint": "/fapi/v1/markPriceKlines", "param": "symbol", "prefix": "mark_price"},
    "index_price": {"endpoint": "/fapi/v1/indexPriceKlines", "param": "pair", "prefix": "index_price"},
    "premium_index": {"endpoint": "/fapi/v1/premiumIndexKlines", "param": "symbol", "prefix": "premium_index"},
}

CROSS_ASSET_SYMBOLS = ("ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT")


def raw_kline_path(cfg: AppConfig, market: str, interval: str) -> Path:
    return cfg.data.raw_dir / f"binance_{market}_{cfg.data.symbol}_{interval}.csv"


def raw_symbol_kline_path(cfg: AppConfig, market: str, symbol: str, interval: str) -> Path:
    return cfg.data.raw_dir / f"binance_{market}_{symbol}_{interval}.csv"


def raw_agg_trade_path(cfg: AppConfig, market: str) -> Path:
    return cfg.data.raw_dir / f"binance_{market}_{cfg.data.symbol}_aggTrades.csv"


def raw_agg_trade_feature_path(cfg: AppConfig, market: str, timeframe: str) -> Path:
    return cfg.data.raw_dir / f"binance_{market}_{cfg.data.symbol}_aggTrades_{timeframe}_features.csv"


def raw_live_flow_path(cfg: AppConfig, market: str = "futures") -> Path:
    return cfg.data.raw_dir / f"binance_{market}_{cfg.data.symbol}_live_flow.csv"


def raw_order_book_path(cfg: AppConfig, market: str = "futures") -> Path:
    return cfg.data.raw_dir / f"binance_{market}_{cfg.data.symbol}_order_book_features.csv"


def raw_book_depth_feature_path(cfg: AppConfig, market: str, timeframe: str) -> Path:
    return cfg.data.raw_dir / f"binance_{market}_{cfg.data.symbol}_bookDepth_{timeframe}_features.csv"


def raw_metrics_feature_path(cfg: AppConfig, market: str, timeframe: str = "5m") -> Path:
    return cfg.data.raw_dir / f"binance_{market}_{cfg.data.symbol}_metrics_{timeframe}.csv"


def raw_futures_ratio_path(cfg: AppConfig, source: str, period: str = "5m") -> Path:
    return cfg.data.raw_dir / f"binance_futures_{cfg.data.symbol}_{source}_{period}.csv"


def raw_futures_basis_path(cfg: AppConfig, source: str, interval: str = "5m") -> Path:
    return cfg.data.raw_dir / f"binance_futures_{cfg.data.symbol}_{source}_{interval}.csv"


def hyperliquid_path(cfg: AppConfig, interval: str = "5m") -> Path:
    return cfg.data.raw_dir / f"hyperliquid_BTC_{interval}.csv"


def coinbase_path(cfg: AppConfig, interval: str = "5m") -> Path:
    return cfg.data.raw_dir / f"coinbase_BTC-USD_{interval}.csv"


def bybit_kline_path(cfg: AppConfig, interval: str = "5m") -> Path:
    return cfg.data.raw_dir / f"bybit_linear_{cfg.data.symbol}_{interval}.csv"


def bybit_open_interest_path(cfg: AppConfig, interval: str = "5m") -> Path:
    return cfg.data.raw_dir / f"bybit_linear_{cfg.data.symbol}_open_interest_{interval}.csv"


def bybit_funding_path(cfg: AppConfig) -> Path:
    return cfg.data.raw_dir / f"bybit_linear_{cfg.data.symbol}_funding.csv"


def okx_kline_path(cfg: AppConfig, interval: str = "5m") -> Path:
    return cfg.data.raw_dir / f"okx_BTC-USDT-SWAP_{interval}.csv"


def okx_open_interest_path(cfg: AppConfig, period: str = "5m") -> Path:
    return cfg.data.raw_dir / f"okx_BTC-USDT-SWAP_open_interest_{period}.csv"


def okx_long_short_ratio_path(cfg: AppConfig, period: str = "5m") -> Path:
    return cfg.data.raw_dir / f"okx_BTC-USDT-SWAP_long_short_ratio_{period}.csv"


def request_json(method: str, url: str, *, max_retries: int = 6, sleep_base: float = 1.5, **kwargs):
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = requests.request(method, url, timeout=30, **kwargs)
            if response.status_code in (418, 429):
                time.sleep(min(60.0, 2.0**attempt))
                continue
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # noqa: BLE001 - network/API retry boundary
            last_error = exc
            time.sleep(min(30.0, sleep_base**attempt))
    raise RuntimeError(f"request failed after {max_retries} attempts: {url}") from last_error


def _atomic_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dirs(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "open_time" in df.columns:
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True, format="mixed")
    if "funding_time" in df.columns:
        df["funding_time"] = pd.to_datetime(df["funding_time"], utc=True, format="mixed")
    return df


def normalize_klines(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True, format="mixed")
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    keep = ["open_time"] + [c for c in NUMERIC_COLUMNS if c in df.columns]
    return (
        df[keep]
        .dropna(subset=["open_time", "open", "high", "low", "close"])
        .drop_duplicates("open_time")
        .sort_values("open_time")
        .reset_index(drop=True)
    )


def normalize_agg_trades(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    frame = df.copy()
    rename = {
        "a": "agg_id",
        "agg_trade_id": "agg_id",
        "p": "price",
        "q": "qty",
        "quantity": "qty",
        "T": "trade_time",
        "transact_time": "trade_time",
        "timestamp": "trade_time",
        "m": "buyer_is_maker",
        "is_buyer_maker": "buyer_is_maker",
    }
    frame = frame.rename(columns=rename)
    if "trade_time" not in frame.columns and "T" in frame.columns:
        frame["trade_time"] = frame["T"]
    if pd.api.types.is_datetime64_any_dtype(frame["trade_time"]):
        frame["trade_time"] = pd.to_datetime(frame["trade_time"], utc=True, format="mixed", errors="coerce")
    else:
        numeric_trade_time = pd.to_numeric(frame["trade_time"], errors="coerce")
        if pd.api.types.is_numeric_dtype(frame["trade_time"]) or numeric_trade_time.notna().sum() >= max(1, int(len(frame) * 0.5)):
            frame["trade_time"] = pd.to_datetime(numeric_trade_time, unit="ms", utc=True, errors="coerce")
        else:
            frame["trade_time"] = pd.to_datetime(frame["trade_time"], utc=True, format="mixed", errors="coerce")
    if "agg_id" in frame.columns:
        frame["agg_id"] = pd.to_numeric(frame["agg_id"], errors="coerce").astype("Int64")
    for col in ("price", "qty", "quote_qty"):
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if "quote_qty" not in frame.columns:
        frame["quote_qty"] = frame["price"] * frame["qty"]
    if "buyer_is_maker" in frame.columns:
        if frame["buyer_is_maker"].dtype == object:
            frame["buyer_is_maker"] = frame["buyer_is_maker"].astype(str).str.lower().isin({"true", "1"})
        else:
            frame["buyer_is_maker"] = frame["buyer_is_maker"].astype(bool)
    elif "side" in frame.columns:
        frame["buyer_is_maker"] = frame["side"].astype(str).str.lower().eq("sell")
    else:
        frame["buyer_is_maker"] = False
    keep = ["trade_time", "agg_id", "price", "qty", "quote_qty", "buyer_is_maker"]
    return (
        frame[keep]
        .dropna(subset=["trade_time", "price", "qty", "quote_qty"])
        .drop_duplicates(["trade_time", "agg_id"])
        .sort_values(["trade_time", "agg_id"])
        .reset_index(drop=True)
    )


def normalize_live_flow(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    frame = df.copy()
    if "trade_time" not in frame.columns and "trade_time_utc" in frame.columns:
        frame["trade_time"] = frame["trade_time_utc"]
    if "event_time" not in frame.columns and "event_time_utc" in frame.columns:
        frame["event_time"] = frame["event_time_utc"]
    if "received_time" not in frame.columns and "received_utc" in frame.columns:
        frame["received_time"] = frame["received_utc"]
    if "event_type" not in frame.columns:
        stream = frame.get("stream", pd.Series("", index=frame.index)).astype(str)
        status = frame.get("order_status", pd.Series("", index=frame.index)).astype(str)
        frame["event_type"] = np.where(
            stream.str.contains("forceOrder", case=False, na=False) | status.ne(""),
            "force_order",
            "agg_trade",
        )
    for col in ("trade_time", "event_time", "received_time"):
        if col in frame.columns:
            frame[col] = pd.to_datetime(frame[col], utc=True, format="mixed", errors="coerce")
    if "event_time" not in frame.columns:
        frame["event_time"] = frame["trade_time"]
    if "received_time" not in frame.columns:
        frame["received_time"] = frame["event_time"]
    if "side" in frame.columns:
        frame["side"] = frame["side"].astype(str).str.lower()
    else:
        frame["side"] = ""
    for col in ("price", "qty", "quote_qty"):
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        else:
            frame[col] = np.nan
    if "quote_qty" not in frame.columns or frame["quote_qty"].isna().all():
        frame["quote_qty"] = frame["price"] * frame["qty"]
    if "trade_id" not in frame.columns:
        frame["trade_id"] = ""
    if "order_status" not in frame.columns:
        frame["order_status"] = ""
    if "symbol" not in frame.columns:
        frame["symbol"] = ""
    keep = [
        "received_time",
        "event_time",
        "trade_time",
        "event_type",
        "symbol",
        "side",
        "price",
        "qty",
        "quote_qty",
        "trade_id",
        "order_status",
    ]
    return (
        frame[keep]
        .dropna(subset=["trade_time", "price", "qty", "quote_qty"])
        .drop_duplicates(["trade_time", "event_type", "trade_id", "side", "price", "qty"])
        .sort_values(["trade_time", "event_type", "trade_id"])
        .reset_index(drop=True)
    )


def normalize_order_book_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    frame = df.copy()
    frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True, format="mixed")
    for col in frame.columns:
        if col != "open_time":
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return (
        frame.dropna(subset=["open_time"])
        .drop_duplicates("open_time")
        .sort_values("open_time")
        .reset_index(drop=True)
    )


def normalize_futures_ratio(df: pd.DataFrame, source: str) -> pd.DataFrame:
    if df.empty:
        return df
    if source not in FUTURES_RATIO_SOURCES:
        raise ValueError(f"unknown futures ratio source: {source}")
    columns = FUTURES_RATIO_SOURCES[source]["columns"]
    if not isinstance(columns, dict):
        raise ValueError(f"invalid futures ratio source mapping: {source}")
    frame = df.copy().rename(columns=columns)
    if "open_time" not in frame.columns:
        if "timestamp" not in frame.columns:
            raise ValueError(f"futures ratio source {source} has no timestamp column")
        frame["open_time"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    else:
        frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True, format="mixed")
    keep = ["open_time"] + [col for col in columns.values() if col in frame.columns]
    for col in keep:
        if col != "open_time":
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return (
        frame[keep]
        .dropna(subset=["open_time"])
        .drop_duplicates("open_time")
        .sort_values("open_time")
        .reset_index(drop=True)
    )


def bybit_interval(interval: str) -> str:
    minutes = timeframe_to_minutes(interval)
    mapping = {
        1: "1",
        3: "3",
        5: "5",
        15: "15",
        30: "30",
        60: "60",
        120: "120",
        240: "240",
        360: "360",
        720: "720",
        1440: "D",
    }
    if minutes not in mapping:
        raise ValueError(f"bybit klines do not support interval {interval}")
    return mapping[minutes]


def bybit_open_interest_interval(interval: str) -> str:
    minutes = timeframe_to_minutes(interval)
    mapping = {5: "5min", 15: "15min", 30: "30min", 60: "1h", 240: "4h", 1440: "1d"}
    if minutes not in mapping:
        raise ValueError(f"bybit open interest does not support interval {interval}; use 5m,15m,30m,1h,4h,1d")
    return mapping[minutes]


def okx_interval(interval: str) -> str:
    minutes = timeframe_to_minutes(interval)
    mapping = {
        1: "1m",
        3: "3m",
        5: "5m",
        15: "15m",
        30: "30m",
        60: "1H",
        120: "2H",
        240: "4H",
    }
    if minutes not in mapping:
        raise ValueError(f"okx candles do not support interval {interval}; use 1m,3m,5m,15m,30m,1h,2h,4h")
    return mapping[minutes]


def normalize_bybit_open_interest(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    frame = df.copy()
    if "open_time" not in frame.columns:
        if "timestamp" not in frame.columns:
            raise ValueError("bybit open interest data has no timestamp column")
        frame["open_time"] = pd.to_datetime(pd.to_numeric(frame["timestamp"], errors="coerce"), unit="ms", utc=True)
    else:
        frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True, format="mixed")
    if "bybit_open_interest" not in frame.columns:
        frame["bybit_open_interest"] = pd.to_numeric(frame.get("openInterest"), errors="coerce")
    else:
        frame["bybit_open_interest"] = pd.to_numeric(frame["bybit_open_interest"], errors="coerce")
    return (
        frame[["open_time", "bybit_open_interest"]]
        .dropna(subset=["open_time"])
        .drop_duplicates("open_time")
        .sort_values("open_time")
        .reset_index(drop=True)
    )


def normalize_bybit_funding(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    frame = df.copy()
    if "funding_time" not in frame.columns:
        if "fundingRateTimestamp" not in frame.columns:
            raise ValueError("bybit funding data has no funding timestamp column")
        frame["funding_time"] = pd.to_datetime(pd.to_numeric(frame["fundingRateTimestamp"], errors="coerce"), unit="ms", utc=True)
    else:
        frame["funding_time"] = pd.to_datetime(frame["funding_time"], utc=True, format="mixed")
    if "bybit_funding_rate" not in frame.columns:
        frame["bybit_funding_rate"] = pd.to_numeric(frame.get("fundingRate"), errors="coerce")
    else:
        frame["bybit_funding_rate"] = pd.to_numeric(frame["bybit_funding_rate"], errors="coerce")
    return (
        frame[["funding_time", "bybit_funding_rate"]]
        .dropna(subset=["funding_time"])
        .drop_duplicates("funding_time")
        .sort_values("funding_time")
        .reset_index(drop=True)
    )


def normalize_okx_klines(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    frame = df.copy()
    if "open_time" not in frame.columns:
        if "ts" not in frame.columns:
            raise ValueError("okx kline data has no timestamp column")
        frame["open_time"] = pd.to_datetime(pd.to_numeric(frame["ts"], errors="coerce"), unit="ms", utc=True)
    else:
        frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True, format="mixed")
    rename = {
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "vol": "volume",
        "volCcyQuote": "quote_volume",
    }
    frame = frame.rename(columns=rename)
    if "quote_volume" not in frame.columns and "volCcy" in frame.columns:
        frame["quote_volume"] = frame["volCcy"]
    for col in ("open", "high", "low", "close", "volume", "quote_volume"):
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if "trade_count" not in frame.columns:
        frame["trade_count"] = np.nan
    if "taker_buy_base" not in frame.columns:
        frame["taker_buy_base"] = np.nan
    if "taker_buy_quote" not in frame.columns:
        frame["taker_buy_quote"] = np.nan
    if "confirm" in frame.columns:
        frame = frame[frame["confirm"].astype(str).ne("0")]
    keep = ["open_time"] + [col for col in NUMERIC_COLUMNS if col in frame.columns]
    return (
        frame[keep]
        .dropna(subset=["open_time", "open", "high", "low", "close"])
        .drop_duplicates("open_time")
        .sort_values("open_time")
        .reset_index(drop=True)
    )


def normalize_okx_open_interest(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    frame = df.copy()
    if "open_time" not in frame.columns:
        if "ts" not in frame.columns:
            raise ValueError("okx open-interest data has no timestamp column")
        frame["open_time"] = pd.to_datetime(pd.to_numeric(frame["ts"], errors="coerce"), unit="ms", utc=True)
    else:
        frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True, format="mixed")
    rename = {
        "oi": "okx_open_interest",
        "oiCcy": "okx_open_interest_ccy",
        "oiUsd": "okx_open_interest_usd",
    }
    frame = frame.rename(columns=rename)
    for col in ("okx_open_interest", "okx_open_interest_ccy", "okx_open_interest_usd"):
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    keep = ["open_time"] + [col for col in ("okx_open_interest", "okx_open_interest_ccy", "okx_open_interest_usd") if col in frame.columns]
    return (
        frame[keep]
        .dropna(subset=["open_time"])
        .drop_duplicates("open_time")
        .sort_values("open_time")
        .reset_index(drop=True)
    )


def normalize_okx_long_short_ratio(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    frame = df.copy()
    if "open_time" not in frame.columns:
        if "ts" not in frame.columns:
            raise ValueError("okx long-short ratio data has no timestamp column")
        frame["open_time"] = pd.to_datetime(pd.to_numeric(frame["ts"], errors="coerce"), unit="ms", utc=True)
    else:
        frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True, format="mixed")
    if "okx_long_short_ratio" not in frame.columns:
        if "ratio" in frame.columns:
            frame["okx_long_short_ratio"] = frame["ratio"]
        elif 1 in frame.columns:
            frame["okx_long_short_ratio"] = frame[1]
    frame["okx_long_short_ratio"] = pd.to_numeric(frame["okx_long_short_ratio"], errors="coerce")
    return (
        frame[["open_time", "okx_long_short_ratio"]]
        .dropna(subset=["open_time"])
        .drop_duplicates("open_time")
        .sort_values("open_time")
        .reset_index(drop=True)
    )


def fetch_binance_klines(
    cfg: AppConfig,
    *,
    market: str,
    interval: str,
    start: datetime,
    end: datetime | None = None,
    force: bool = False,
    symbol: str | None = None,
) -> Path:
    ensure_dirs(cfg.data.raw_dir)
    end = end or utc_now()
    symbol = symbol or cfg.data.symbol
    path = raw_symbol_kline_path(cfg, market, symbol, interval)
    existing = pd.DataFrame() if force else load_csv(path)
    existing = normalize_klines(existing) if not existing.empty else existing
    start_ms = to_ms(start)
    interval_ms = timeframe_to_minutes(interval) * 60_000
    if not existing.empty:
        last_ms = int(existing["open_time"].max().timestamp() * 1000)
        start_ms = max(start_ms, last_ms + interval_ms)
    end_ms = to_ms(end)
    if start_ms >= end_ms and path.exists():
        return path

    base_url = BINANCE_FUTURES_URL if market == "futures" else BINANCE_SPOT_URL
    endpoint = "/fapi/v1/klines" if market == "futures" else "/api/v3/klines"
    rows: list[list] = []
    cursor = start_ms
    while cursor < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cursor,
            "endTime": end_ms - 1,
            "limit": 1500,
        }
        batch = request_json("GET", base_url + endpoint, params=params)
        if not batch:
            break
        rows.extend(batch)
        next_cursor = int(batch[-1][0]) + interval_ms
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        time.sleep(cfg.runtime.fetch_sleep_seconds)

    if rows:
        fetched = pd.DataFrame(rows, columns=KLINE_COLUMNS)
        fetched["open_time"] = pd.to_datetime(fetched["open_time_ms"], unit="ms", utc=True)
        for col in NUMERIC_COLUMNS:
            fetched[col] = pd.to_numeric(fetched[col], errors="coerce")
        fetched = fetched[["open_time", *NUMERIC_COLUMNS]]
        combined = pd.concat([existing, fetched], ignore_index=True)
    else:
        combined = existing
    combined = normalize_klines(combined)
    combined = combined[combined["open_time"] < end]
    _atomic_csv(combined, path)
    return path


def fetch_futures_basis_klines(
    cfg: AppConfig,
    *,
    source: str,
    interval: str,
    start: datetime,
    end: datetime | None = None,
    force: bool = False,
) -> Path:
    if source not in FUTURES_BASIS_SOURCES:
        raise ValueError(f"unknown futures basis source: {source}")
    ensure_dirs(cfg.data.raw_dir)
    end = end or utc_now()
    path = raw_futures_basis_path(cfg, source, interval)
    existing = pd.DataFrame() if force else load_csv(path)
    existing = normalize_klines(existing) if not existing.empty else existing
    start_ms = to_ms(start)
    interval_ms = timeframe_to_minutes(interval) * 60_000
    if not existing.empty:
        last_ms = int(existing["open_time"].max().timestamp() * 1000)
        start_ms = max(start_ms, last_ms + interval_ms)
    end_ms = to_ms(end)
    if start_ms >= end_ms and path.exists():
        return path

    spec = FUTURES_BASIS_SOURCES[source]
    rows: list[list] = []
    cursor = start_ms
    while cursor < end_ms:
        params = {
            spec["param"]: cfg.data.symbol,
            "interval": interval,
            "startTime": cursor,
            "endTime": end_ms - 1,
            "limit": 1500,
        }
        batch = request_json("GET", BINANCE_FUTURES_URL + spec["endpoint"], params=params)
        if not batch:
            break
        rows.extend(batch)
        next_cursor = int(batch[-1][0]) + interval_ms
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        time.sleep(cfg.runtime.fetch_sleep_seconds)

    if rows:
        fetched = pd.DataFrame(rows, columns=KLINE_COLUMNS)
        fetched["open_time"] = pd.to_datetime(fetched["open_time_ms"], unit="ms", utc=True)
        for col in NUMERIC_COLUMNS:
            fetched[col] = pd.to_numeric(fetched[col], errors="coerce")
        fetched = fetched[["open_time", *NUMERIC_COLUMNS]]
        combined = pd.concat([existing, fetched], ignore_index=True)
    else:
        combined = existing
    combined = normalize_klines(combined)
    combined = combined[combined["open_time"] < end]
    _atomic_csv(combined, path)
    return path


def fetch_hyperliquid_candles(
    cfg: AppConfig,
    *,
    interval: str = "5m",
    start: datetime,
    end: datetime | None = None,
    force: bool = False,
) -> Path:
    ensure_dirs(cfg.data.raw_dir)
    end = end or utc_now()
    path = hyperliquid_path(cfg, interval)
    existing = pd.DataFrame() if force else load_csv(path)
    existing = normalize_klines(existing) if not existing.empty else existing
    interval_ms = timeframe_to_minutes(interval) * 60_000
    start_ms = to_ms(start)
    if not existing.empty:
        last_ms = int(existing["open_time"].max().timestamp() * 1000)
        start_ms = max(start_ms, last_ms + interval_ms)
    end_ms = to_ms(end)
    if start_ms >= end_ms and path.exists():
        return path

    rows: list[dict] = []
    cursor = start_ms
    max_span_ms = interval_ms * 5000
    while cursor < end_ms:
        chunk_end = min(end_ms - 1, cursor + max_span_ms - 1)
        batch = request_json(
            "POST",
            HYPERLIQUID_URL,
            json={
                "type": "candleSnapshot",
                "req": {
                    "coin": "BTC",
                    "interval": interval,
                    "startTime": cursor,
                    "endTime": chunk_end,
                },
            },
        )
        if not batch:
            break
        rows.extend(batch)
        next_cursor = int(batch[-1]["t"]) + interval_ms
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        time.sleep(cfg.runtime.fetch_sleep_seconds)

    if rows:
        fetched = pd.DataFrame(rows)
        fetched["open_time"] = pd.to_datetime(fetched["t"], unit="ms", utc=True)
        fetched["open"] = pd.to_numeric(fetched["o"], errors="coerce")
        fetched["high"] = pd.to_numeric(fetched["h"], errors="coerce")
        fetched["low"] = pd.to_numeric(fetched["l"], errors="coerce")
        fetched["close"] = pd.to_numeric(fetched["c"], errors="coerce")
        fetched["volume"] = pd.to_numeric(fetched["v"], errors="coerce")
        fetched["quote_volume"] = fetched["volume"] * fetched["close"]
        fetched["trade_count"] = pd.to_numeric(fetched.get("n", 0), errors="coerce")
        fetched["taker_buy_base"] = np.nan
        fetched["taker_buy_quote"] = np.nan
        fetched = fetched[["open_time", *NUMERIC_COLUMNS]]
        combined = pd.concat([existing, fetched], ignore_index=True)
    else:
        combined = existing
    combined = normalize_klines(combined)
    combined = combined[combined["open_time"] < end]
    _atomic_csv(combined, path)
    return path


def coinbase_granularity_seconds(interval: str) -> int:
    seconds = timeframe_to_minutes(interval) * 60
    allowed = {60, 300, 900, 3600, 21600, 86400}
    if seconds not in allowed:
        raise ValueError(f"coinbase candles do not support interval {interval}; use 1m,5m,15m,1h,6h,1d")
    return seconds


def fetch_coinbase_candles(
    cfg: AppConfig,
    *,
    interval: str = "5m",
    start: datetime,
    end: datetime | None = None,
    force: bool = False,
) -> Path:
    ensure_dirs(cfg.data.raw_dir)
    end = end or utc_now()
    path = coinbase_path(cfg, interval)
    existing = pd.DataFrame() if force else load_csv(path)
    existing = normalize_klines(existing) if not existing.empty else existing
    granularity = coinbase_granularity_seconds(interval)
    interval_ms = granularity * 1000
    start_ms = to_ms(start)
    if not existing.empty:
        last_ms = int(existing["open_time"].max().timestamp() * 1000)
        start_ms = max(start_ms, last_ms + interval_ms)
    end_ms = to_ms(end)
    if start_ms >= end_ms and path.exists():
        return path

    rows: list[list] = []
    cursor = start_ms
    max_span_ms = interval_ms * 300
    while cursor < end_ms:
        chunk_end = min(end_ms - 1, cursor + max_span_ms - 1)
        batch = request_json(
            "GET",
            COINBASE_EXCHANGE_URL + "/products/BTC-USD/candles",
            params={
                "granularity": granularity,
                "start": datetime.fromtimestamp(cursor / 1000, tz=UTC).isoformat(),
                "end": datetime.fromtimestamp(chunk_end / 1000, tz=UTC).isoformat(),
            },
            max_retries=4,
        )
        if not batch:
            cursor = chunk_end + interval_ms
            continue
        rows.extend(batch)
        cursor = chunk_end + interval_ms
        time.sleep(max(cfg.runtime.fetch_sleep_seconds, 0.10))

    if rows:
        fetched = pd.DataFrame(rows, columns=["time", "low", "high", "open", "close", "volume"])
        fetched["open_time"] = pd.to_datetime(fetched["time"], unit="s", utc=True)
        for col in ("open", "high", "low", "close", "volume"):
            fetched[col] = pd.to_numeric(fetched[col], errors="coerce")
        fetched["quote_volume"] = fetched["volume"] * fetched["close"]
        fetched["trade_count"] = np.nan
        fetched["taker_buy_base"] = np.nan
        fetched["taker_buy_quote"] = np.nan
        fetched = fetched[["open_time", *NUMERIC_COLUMNS]]
        combined = pd.concat([existing, fetched], ignore_index=True)
    else:
        combined = existing
    combined = normalize_klines(combined)
    combined = combined[combined["open_time"] < end]
    _atomic_csv(combined, path)
    return path


def fetch_bybit_klines(
    cfg: AppConfig,
    *,
    interval: str = "5m",
    start: datetime,
    end: datetime | None = None,
    force: bool = False,
) -> Path:
    ensure_dirs(cfg.data.raw_dir)
    end = end or utc_now()
    path = bybit_kline_path(cfg, interval)
    existing = pd.DataFrame() if force else load_csv(path)
    existing = normalize_klines(existing) if not existing.empty else existing
    interval_ms = timeframe_to_minutes(interval) * 60_000
    start_ms = to_ms(start)
    if not existing.empty:
        last_ms = int(existing["open_time"].max().timestamp() * 1000)
        start_ms = max(start_ms, last_ms + interval_ms)
    end_ms = to_ms(end)
    if start_ms >= end_ms and path.exists():
        return path

    rows: list[list] = []
    cursor = start_ms
    bybit_int = bybit_interval(interval)
    while cursor < end_ms:
        chunk_end = min(end_ms - 1, cursor + interval_ms * 1000 - 1)
        payload = request_json(
            "GET",
            BYBIT_URL + "/v5/market/kline",
            params={
                "category": "linear",
                "symbol": cfg.data.symbol,
                "interval": bybit_int,
                "start": cursor,
                "end": chunk_end,
                "limit": 1000,
            },
            max_retries=4,
        )
        if int(payload.get("retCode", -1)) != 0:
            raise RuntimeError(f"bybit kline error: {payload.get('retMsg', payload)}")
        batch = payload.get("result", {}).get("list", [])
        if not batch:
            cursor = chunk_end + interval_ms
            continue
        rows.extend(batch)
        max_time = max(int(item[0]) for item in batch)
        next_cursor = max_time + interval_ms
        if next_cursor <= cursor:
            cursor = chunk_end + interval_ms
        else:
            cursor = next_cursor
        time.sleep(max(cfg.runtime.fetch_sleep_seconds, 0.06))

    if rows:
        fetched = pd.DataFrame(rows, columns=["open_time_ms", "open", "high", "low", "close", "volume", "quote_volume"])
        fetched["open_time"] = pd.to_datetime(pd.to_numeric(fetched["open_time_ms"], errors="coerce"), unit="ms", utc=True)
        for col in ("open", "high", "low", "close", "volume", "quote_volume"):
            fetched[col] = pd.to_numeric(fetched[col], errors="coerce")
        fetched["trade_count"] = np.nan
        fetched["taker_buy_base"] = np.nan
        fetched["taker_buy_quote"] = np.nan
        fetched = fetched[["open_time", *NUMERIC_COLUMNS]]
        combined = pd.concat([existing, fetched], ignore_index=True)
    else:
        combined = existing
    combined = normalize_klines(combined)
    combined = combined[combined["open_time"] < end]
    _atomic_csv(combined, path)
    return path


def fetch_bybit_open_interest(
    cfg: AppConfig,
    *,
    interval: str = "5m",
    start: datetime,
    end: datetime | None = None,
    force: bool = False,
) -> Path:
    ensure_dirs(cfg.data.raw_dir)
    end = end or utc_now()
    path = bybit_open_interest_path(cfg, interval)
    existing = pd.DataFrame() if force else load_csv(path)
    existing = normalize_bybit_open_interest(existing) if not existing.empty else existing
    period_ms = timeframe_to_minutes(interval) * 60_000
    start_ms = to_ms(start)
    if not existing.empty:
        last_ms = int(existing["open_time"].max().timestamp() * 1000)
        start_ms = max(start_ms, last_ms + period_ms)
    end_ms = to_ms(end)
    if start_ms >= end_ms and path.exists():
        return path

    rows: list[dict] = []
    cursor = start_ms
    interval_time = bybit_open_interest_interval(interval)
    while cursor < end_ms:
        chunk_end = min(end_ms - 1, cursor + period_ms * 200 - 1)
        payload = request_json(
            "GET",
            BYBIT_URL + "/v5/market/open-interest",
            params={
                "category": "linear",
                "symbol": cfg.data.symbol,
                "intervalTime": interval_time,
                "startTime": cursor,
                "endTime": chunk_end,
                "limit": 200,
            },
            max_retries=4,
        )
        if int(payload.get("retCode", -1)) != 0:
            raise RuntimeError(f"bybit open interest error: {payload.get('retMsg', payload)}")
        batch = payload.get("result", {}).get("list", [])
        if not batch:
            cursor = chunk_end + period_ms
            continue
        rows.extend(batch)
        max_time = max(int(item["timestamp"]) for item in batch)
        next_cursor = max_time + period_ms
        cursor = next_cursor if next_cursor > cursor else chunk_end + period_ms
        time.sleep(max(cfg.runtime.fetch_sleep_seconds, 0.06))

    if rows:
        fetched = normalize_bybit_open_interest(pd.DataFrame(rows))
        combined = pd.concat([existing, fetched], ignore_index=True)
    else:
        combined = existing
    if combined.empty:
        _atomic_csv(pd.DataFrame(columns=["open_time", "bybit_open_interest"]), path)
        return path
    combined = normalize_bybit_open_interest(combined)
    combined = combined[combined["open_time"] < end]
    _atomic_csv(combined, path)
    return path


def fetch_bybit_funding_history(
    cfg: AppConfig,
    *,
    start: datetime,
    end: datetime | None = None,
    force: bool = False,
) -> Path:
    ensure_dirs(cfg.data.raw_dir)
    end = end or utc_now()
    path = bybit_funding_path(cfg)
    existing = pd.DataFrame() if force else load_csv(path)
    existing = normalize_bybit_funding(existing) if not existing.empty else existing
    funding_ms = 8 * 60 * 60 * 1000
    start_ms = to_ms(start)
    if not existing.empty:
        last_ms = int(existing["funding_time"].max().timestamp() * 1000)
        start_ms = max(start_ms, last_ms + 1)
    end_ms = to_ms(end)
    if start_ms >= end_ms and path.exists():
        return path

    rows: list[dict] = []
    cursor = start_ms
    while cursor < end_ms:
        chunk_end = min(end_ms - 1, cursor + funding_ms * 200 - 1)
        payload = request_json(
            "GET",
            BYBIT_URL + "/v5/market/funding/history",
            params={
                "category": "linear",
                "symbol": cfg.data.symbol,
                "startTime": cursor,
                "endTime": chunk_end,
                "limit": 200,
            },
            max_retries=4,
        )
        if int(payload.get("retCode", -1)) != 0:
            raise RuntimeError(f"bybit funding error: {payload.get('retMsg', payload)}")
        batch = payload.get("result", {}).get("list", [])
        if not batch:
            cursor = chunk_end + 1
            continue
        rows.extend(batch)
        max_time = max(int(item["fundingRateTimestamp"]) for item in batch)
        next_cursor = max_time + 1
        cursor = next_cursor if next_cursor > cursor else chunk_end + 1
        time.sleep(max(cfg.runtime.fetch_sleep_seconds, 0.06))

    if rows:
        fetched = normalize_bybit_funding(pd.DataFrame(rows))
        combined = pd.concat([existing, fetched], ignore_index=True)
    else:
        combined = existing
    if combined.empty:
        _atomic_csv(pd.DataFrame(columns=["funding_time", "bybit_funding_rate"]), path)
        return path
    combined = normalize_bybit_funding(combined)
    combined = combined[combined["funding_time"] < end]
    _atomic_csv(combined, path)
    return path


def fetch_okx_klines(
    cfg: AppConfig,
    *,
    interval: str = "5m",
    start: datetime,
    end: datetime | None = None,
    force: bool = False,
) -> Path:
    ensure_dirs(cfg.data.raw_dir)
    end = end or utc_now()
    path = okx_kline_path(cfg, interval)
    existing = pd.DataFrame() if force else load_csv(path)
    existing = normalize_okx_klines(existing) if not existing.empty else existing
    start_ms = to_ms(start)
    end_ms = to_ms(end)
    if not existing.empty:
        last_ms = int(existing["open_time"].max().timestamp() * 1000)
        start_ms = max(start_ms, last_ms + timeframe_to_minutes(interval) * 60_000)
    if start_ms >= end_ms and path.exists():
        return path

    rows: list[list] = []
    cursor_after: int | None = None
    bar = okx_interval(interval)
    while True:
        params: dict[str, object] = {"instId": "BTC-USDT-SWAP", "bar": bar, "limit": "300"}
        if cursor_after is not None:
            params["after"] = str(cursor_after)
        payload = request_json("GET", OKX_URL + "/api/v5/market/history-candles", params=params, max_retries=4)
        if str(payload.get("code", "")) != "0":
            raise RuntimeError(f"okx kline error: {payload.get('msg', payload)}")
        batch = payload.get("data", [])
        if not batch:
            break
        batch_times = [int(item[0]) for item in batch]
        for item in batch:
            ts = int(item[0])
            if start_ms <= ts < end_ms:
                rows.append(item)
        oldest = min(batch_times)
        newest = max(batch_times)
        if oldest <= start_ms:
            break
        next_after = oldest
        if cursor_after is not None and next_after >= cursor_after:
            break
        cursor_after = next_after
        if newest < start_ms:
            break
        time.sleep(max(cfg.runtime.fetch_sleep_seconds, 0.10))

    if rows:
        fetched = pd.DataFrame(rows, columns=["ts", "o", "h", "l", "c", "vol", "volCcy", "volCcyQuote", "confirm"])
        fetched = normalize_okx_klines(fetched)
        combined = pd.concat([existing, fetched], ignore_index=True)
    else:
        combined = existing
    if combined.empty:
        _atomic_csv(pd.DataFrame(columns=["open_time", *NUMERIC_COLUMNS]), path)
        return path
    combined = normalize_okx_klines(combined)
    combined = combined[combined["open_time"] < end]
    _atomic_csv(combined, path)
    return path


def fetch_okx_open_interest_history(
    cfg: AppConfig,
    *,
    period: str = "5m",
    start: datetime,
    end: datetime | None = None,
    force: bool = False,
) -> Path:
    ensure_dirs(cfg.data.raw_dir)
    end = end or utc_now()
    path = okx_open_interest_path(cfg, period)
    existing = pd.DataFrame() if force else load_csv(path)
    existing = normalize_okx_open_interest(existing) if not existing.empty else existing
    start_ms = to_ms(start)
    end_ms = to_ms(end)
    if not existing.empty:
        last_ms = int(existing["open_time"].max().timestamp() * 1000)
        start_ms = max(start_ms, last_ms + timeframe_to_minutes(period) * 60_000)
    if start_ms >= end_ms and path.exists():
        return path

    rows: list[list] = []
    cursor_end: int | None = end_ms
    while True:
        params: dict[str, object] = {
            "instId": "BTC-USDT-SWAP",
            "period": okx_interval(period),
            "limit": "100",
        }
        if cursor_end is not None:
            params["end"] = str(cursor_end)
        payload = request_json(
            "GET",
            OKX_URL + "/api/v5/rubik/stat/contracts/open-interest-history",
            params=params,
            max_retries=4,
        )
        if str(payload.get("code", "")) != "0":
            raise RuntimeError(f"okx open interest error: {payload.get('msg', payload)}")
        batch = payload.get("data", [])
        if not batch:
            break
        batch_times = [int(item[0]) for item in batch]
        for item in batch:
            ts = int(item[0])
            if start_ms <= ts < end_ms:
                rows.append(item)
        oldest = min(batch_times)
        if oldest <= start_ms:
            break
        if cursor_end is not None and oldest >= cursor_end:
            break
        cursor_end = oldest
        time.sleep(max(cfg.runtime.fetch_sleep_seconds, 0.10))

    if rows:
        fetched = pd.DataFrame(rows, columns=["ts", "oi", "oiCcy", "oiUsd"])
        fetched = normalize_okx_open_interest(fetched)
        combined = pd.concat([existing, fetched], ignore_index=True)
    else:
        combined = existing
    if combined.empty:
        _atomic_csv(pd.DataFrame(columns=["open_time", "okx_open_interest", "okx_open_interest_ccy", "okx_open_interest_usd"]), path)
        return path
    combined = normalize_okx_open_interest(combined)
    combined = combined[combined["open_time"] < end]
    _atomic_csv(combined, path)
    return path


def fetch_okx_long_short_ratio_history(
    cfg: AppConfig,
    *,
    period: str = "5m",
    start: datetime,
    end: datetime | None = None,
    force: bool = False,
) -> Path:
    ensure_dirs(cfg.data.raw_dir)
    end = end or utc_now()
    path = okx_long_short_ratio_path(cfg, period)
    existing = pd.DataFrame() if force else load_csv(path)
    existing = normalize_okx_long_short_ratio(existing) if not existing.empty else existing
    start_ms = to_ms(start)
    end_ms = to_ms(end)
    if not existing.empty:
        last_ms = int(existing["open_time"].max().timestamp() * 1000)
        start_ms = max(start_ms, last_ms + timeframe_to_minutes(period) * 60_000)
    if start_ms >= end_ms and path.exists():
        return path

    rows: list[list] = []
    cursor_end: int | None = end_ms
    while True:
        params: dict[str, object] = {
            "instId": "BTC-USDT-SWAP",
            "period": okx_interval(period),
            "limit": "100",
        }
        if cursor_end is not None:
            params["end"] = str(cursor_end)
        payload = request_json(
            "GET",
            OKX_URL + "/api/v5/rubik/stat/contracts/long-short-account-ratio-contract",
            params=params,
            max_retries=4,
        )
        if str(payload.get("code", "")) != "0":
            raise RuntimeError(f"okx long-short ratio error: {payload.get('msg', payload)}")
        batch = payload.get("data", [])
        if not batch:
            break
        batch_times = [int(item[0]) for item in batch]
        for item in batch:
            ts = int(item[0])
            if start_ms <= ts < end_ms:
                rows.append(item)
        oldest = min(batch_times)
        if oldest <= start_ms:
            break
        if cursor_end is not None and oldest >= cursor_end:
            break
        cursor_end = oldest
        time.sleep(max(cfg.runtime.fetch_sleep_seconds, 0.10))

    if rows:
        fetched = pd.DataFrame(rows, columns=["ts", "ratio"])
        fetched = normalize_okx_long_short_ratio(fetched)
        combined = pd.concat([existing, fetched], ignore_index=True)
    else:
        combined = existing
    if combined.empty:
        _atomic_csv(pd.DataFrame(columns=["open_time", "okx_long_short_ratio"]), path)
        return path
    combined = normalize_okx_long_short_ratio(combined)
    combined = combined[combined["open_time"] < end]
    _atomic_csv(combined, path)
    return path


def fetch_binance_agg_trades(
    cfg: AppConfig,
    *,
    market: str,
    start: datetime,
    end: datetime | None = None,
    force: bool = False,
    max_batches: int | None = None,
    checkpoint_rows: int = 100_000,
    max_seconds: float | None = None,
) -> Path:
    ensure_dirs(cfg.data.raw_dir)
    end = end or utc_now()
    path = raw_agg_trade_path(cfg, market)
    existing = pd.DataFrame() if force else load_csv(path)
    existing = normalize_agg_trades(existing) if not existing.empty else existing
    start_ms = to_ms(start)
    if not existing.empty:
        last_ms = int(existing["trade_time"].max().timestamp() * 1000)
        start_ms = max(start_ms, last_ms + 1)
    end_ms = to_ms(end)
    if start_ms >= end_ms and path.exists():
        return path

    base_url = BINANCE_FUTURES_URL if market == "futures" else BINANCE_SPOT_URL
    endpoint = "/fapi/v1/aggTrades" if market == "futures" else "/api/v3/aggTrades"
    rows: list[dict] = []
    batches = 0
    started = time.monotonic()

    def flush_rows() -> None:
        nonlocal existing, rows
        if not rows:
            return
        fetched = normalize_agg_trades(pd.DataFrame(rows))
        combined = pd.concat([existing, fetched], ignore_index=True)
        existing = normalize_agg_trades(combined)
        existing = existing[existing["trade_time"] < end]
        _atomic_csv(existing, path)
        rows = []

    cursor = start_ms
    while cursor < end_ms:
        if max_seconds is not None and time.monotonic() - started >= float(max_seconds):
            break
        batch = request_json(
            "GET",
            base_url + endpoint,
            max_retries=3,
            params={"symbol": cfg.data.symbol, "startTime": cursor, "endTime": end_ms - 1, "limit": 1000},
        )
        if not batch:
            break
        rows.extend(batch)
        batches += 1
        next_cursor = int(batch[-1]["T"]) + 1
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        if len(rows) >= checkpoint_rows:
            flush_rows()
        if max_batches is not None and batches >= max_batches:
            break
        if max_seconds is not None and time.monotonic() - started >= float(max_seconds):
            break
        time.sleep(cfg.runtime.fetch_sleep_seconds)

    flush_rows()
    if not path.exists():
        _atomic_csv(normalize_agg_trades(existing), path)
    return path


def _binance_vision_market_prefix(market: str) -> str:
    if market == "futures":
        return "data/futures/um"
    if market == "spot":
        return "data/spot"
    raise ValueError(f"Binance Vision aggTrade archives only support futures or spot, got {market!r}")


def _binance_vision_agg_trade_url(cfg: AppConfig, market: str, day: datetime) -> str:
    day_text = day.strftime("%Y-%m-%d")
    prefix = _binance_vision_market_prefix(market)
    symbol = cfg.data.symbol
    return f"{BINANCE_VISION_URL}/{prefix}/daily/aggTrades/{symbol}/{symbol}-aggTrades-{day_text}.zip"


def _binance_vision_agg_trade_zip_path(cfg: AppConfig, market: str, day: datetime) -> Path:
    day_text = day.strftime("%Y-%m-%d")
    return (
        cfg.data.raw_dir
        / "binance_vision"
        / market
        / cfg.data.symbol
        / "aggTrades"
        / f"{cfg.data.symbol}-aggTrades-{day_text}.zip"
    )


def _binance_vision_book_depth_url(cfg: AppConfig, market: str, day: datetime) -> str:
    day_text = day.strftime("%Y-%m-%d")
    prefix = _binance_vision_market_prefix(market)
    symbol = cfg.data.symbol
    return f"{BINANCE_VISION_URL}/{prefix}/daily/bookDepth/{symbol}/{symbol}-bookDepth-{day_text}.zip"


def _binance_vision_book_depth_zip_path(cfg: AppConfig, market: str, day: datetime) -> Path:
    day_text = day.strftime("%Y-%m-%d")
    return (
        cfg.data.raw_dir
        / "binance_vision"
        / market
        / cfg.data.symbol
        / "bookDepth"
        / f"{cfg.data.symbol}-bookDepth-{day_text}.zip"
    )


def _binance_vision_metrics_url(cfg: AppConfig, market: str, day: datetime) -> str:
    if market != "futures":
        raise ValueError("Binance Vision metrics archives are available for futures only")
    day_text = day.strftime("%Y-%m-%d")
    symbol = cfg.data.symbol
    return f"{BINANCE_VISION_URL}/data/futures/um/daily/metrics/{symbol}/{symbol}-metrics-{day_text}.zip"


def _binance_vision_metrics_zip_path(cfg: AppConfig, market: str, day: datetime) -> Path:
    day_text = day.strftime("%Y-%m-%d")
    return (
        cfg.data.raw_dir
        / "binance_vision"
        / market
        / cfg.data.symbol
        / "metrics"
        / f"{cfg.data.symbol}-metrics-{day_text}.zip"
    )


def _download_binance_vision_file(url: str, path: Path, *, force: bool = False) -> bool:
    if path.exists() and path.stat().st_size > 0 and not force:
        return True
    ensure_dirs(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    response = requests.get(url, timeout=60, stream=True)
    if response.status_code == 404:
        if tmp.exists():
            tmp.unlink()
        return False
    response.raise_for_status()
    with tmp.open("wb") as fh:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                fh.write(chunk)
    os.replace(tmp, path)
    return True


def _read_binance_vision_agg_trade_zip(path: Path) -> pd.DataFrame:
    names = [
        "agg_id",
        "price",
        "qty",
        "first_trade_id",
        "last_trade_id",
        "trade_time",
        "buyer_is_maker",
    ]
    with zipfile.ZipFile(path) as archive:
        csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not csv_names:
            return pd.DataFrame()
        with archive.open(csv_names[0]) as fh:
            frame = pd.read_csv(fh, header=None, names=names, usecols=list(range(len(names))), low_memory=False)
    return normalize_agg_trades(frame)


def update_agg_trade_feature_cache(
    cfg: AppConfig,
    *,
    market: str,
    timeframes: Iterable[str],
) -> list[Path]:
    trades = load_csv(raw_agg_trade_path(cfg, market))
    trades = normalize_agg_trades(trades) if not trades.empty else trades
    if trades.empty:
        return []
    paths: list[Path] = []
    for timeframe in dict.fromkeys(timeframes):
        features = aggregate_trades_to_timeframe(trades, timeframe)
        if features.empty:
            continue
        path = raw_agg_trade_feature_path(cfg, market, timeframe)
        _atomic_csv(features, path)
        paths.append(path)
    return paths


def fetch_binance_vision_agg_trades(
    cfg: AppConfig,
    *,
    market: str,
    start: datetime,
    end: datetime | None = None,
    force: bool = False,
    max_files: int | None = None,
    checkpoint_files: int = 3,
    feature_timeframes: Iterable[str] = ("1m", "5m"),
    cache_raw: bool = False,
) -> Path:
    """Fetch public daily Binance aggTrade archives and cache trade-flow features.

    The REST aggTrade endpoint is too slow for broad 5m research. Binance Vision
    daily ZIPs make historical trade-flow features practical and keep the job
    resumable because downloaded archives are cached separately. Raw trade CSV
    caching is optional; by default the large archives are aggregated directly
    into timeframe feature files.
    """

    ensure_dirs(cfg.data.raw_dir)
    end = end or utc_now()
    if start.tzinfo is None:
        start = start.replace(tzinfo=UTC)
    if end.tzinfo is None:
        end = end.replace(tzinfo=UTC)

    raw_path = raw_agg_trade_path(cfg, market)
    raw_existing = pd.DataFrame() if (force and cache_raw) else (load_csv(raw_path) if cache_raw else pd.DataFrame())
    raw_existing = normalize_agg_trades(raw_existing) if not raw_existing.empty else raw_existing
    raw_frames: list[pd.DataFrame] = []
    timeframes = list(dict.fromkeys(feature_timeframes))
    feature_existing: dict[str, pd.DataFrame] = {}
    feature_frames: dict[str, list[pd.DataFrame]] = {timeframe: [] for timeframe in timeframes}
    for timeframe in timeframes:
        feature_path = raw_agg_trade_feature_path(cfg, market, timeframe)
        existing_features = pd.DataFrame() if force else load_csv(feature_path)
        if not existing_features.empty:
            existing_features["open_time"] = pd.to_datetime(existing_features["open_time"], utc=True, format="mixed", errors="coerce")
            existing_features = existing_features.dropna(subset=["open_time"])
        feature_existing[timeframe] = existing_features
    files_since_flush = 0
    fetched_files = 0

    def flush_raw_frames() -> None:
        nonlocal raw_existing, raw_frames
        if not cache_raw or not raw_frames:
            return
        combined = pd.concat([raw_existing, *raw_frames], ignore_index=True)
        raw_existing = normalize_agg_trades(combined)
        _atomic_csv(raw_existing, raw_path)
        raw_frames = []

    def flush_feature_frames() -> None:
        nonlocal files_since_flush
        for timeframe in timeframes:
            frames = feature_frames[timeframe]
            if not frames:
                continue
            existing_features = feature_existing[timeframe]
            combined = pd.concat([existing_features, *frames], ignore_index=True)
            combined["open_time"] = pd.to_datetime(combined["open_time"], utc=True, format="mixed", errors="coerce")
            combined = (
                combined.dropna(subset=["open_time"])
                .drop_duplicates("open_time", keep="last")
                .sort_values("open_time")
                .reset_index(drop=True)
            )
            feature_existing[timeframe] = combined
            feature_frames[timeframe] = []
            _atomic_csv(combined, raw_agg_trade_feature_path(cfg, market, timeframe))
        files_since_flush = 0

    current_day = datetime(start.year, start.month, start.day, tzinfo=UTC)
    final_day = datetime(end.year, end.month, end.day, tzinfo=UTC)
    while current_day < final_day:
        if max_files is not None and fetched_files >= max_files:
            break
        zip_path = _binance_vision_agg_trade_zip_path(cfg, market, current_day)
        url = _binance_vision_agg_trade_url(cfg, market, current_day)
        downloaded = _download_binance_vision_file(url, zip_path, force=force)
        if downloaded:
            day_frame = _read_binance_vision_agg_trade_zip(zip_path)
            if not day_frame.empty:
                day_frame = day_frame[(day_frame["trade_time"] >= start) & (day_frame["trade_time"] < end)]
                if not day_frame.empty:
                    if cache_raw:
                        raw_frames.append(day_frame)
                    for timeframe in timeframes:
                        features = aggregate_trades_to_timeframe(day_frame, timeframe)
                        if not features.empty:
                            feature_frames[timeframe].append(features)
            fetched_files += 1
            files_since_flush += 1
            if files_since_flush >= max(1, checkpoint_files):
                flush_raw_frames()
                flush_feature_frames()
        current_day += timedelta(days=1)

    flush_raw_frames()
    flush_feature_frames()
    return raw_path


def _read_binance_vision_book_depth_zip(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not csv_names:
            return pd.DataFrame()
        with archive.open(csv_names[0]) as fh:
            frame = pd.read_csv(fh, low_memory=False)
    if frame.empty:
        return frame
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, format="mixed", errors="coerce")
    for col in ("percentage", "depth", "notional"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame.dropna(subset=["timestamp", "percentage", "depth", "notional"]).sort_values("timestamp").reset_index(drop=True)


def aggregate_book_depth_to_timeframe(depth: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if depth.empty:
        return pd.DataFrame()
    frame = depth.copy()
    rows: dict[str, pd.Series] = {}
    grouped = frame.groupby("timestamp", sort=True)
    for radius in (1, 2, 3, 5):
        bid_mask = frame["percentage"].lt(0.0) & frame["percentage"].abs().le(float(radius))
        ask_mask = frame["percentage"].gt(0.0) & frame["percentage"].abs().le(float(radius))
        bid_notional = frame.loc[bid_mask].groupby("timestamp")["notional"].sum()
        ask_notional = frame.loc[ask_mask].groupby("timestamp")["notional"].sum()
        bid_depth = frame.loc[bid_mask].groupby("timestamp")["depth"].sum()
        ask_depth = frame.loc[ask_mask].groupby("timestamp")["depth"].sum()
        total_notional = bid_notional.add(ask_notional, fill_value=0.0)
        total_depth = bid_depth.add(ask_depth, fill_value=0.0)
        rows[f"book_depth_bid_notional_{radius}pct"] = bid_notional
        rows[f"book_depth_ask_notional_{radius}pct"] = ask_notional
        rows[f"book_depth_notional_imbalance_{radius}pct"] = (bid_notional - ask_notional) / total_notional.replace(0.0, np.nan)
        rows[f"book_depth_total_notional_{radius}pct"] = total_notional
        rows[f"book_depth_bid_qty_{radius}pct"] = bid_depth
        rows[f"book_depth_ask_qty_{radius}pct"] = ask_depth
        rows[f"book_depth_qty_imbalance_{radius}pct"] = (bid_depth - ask_depth) / total_depth.replace(0.0, np.nan)
    snapshots = pd.DataFrame(rows).sort_index()
    if snapshots.empty:
        return pd.DataFrame()
    rule = timeframe_to_pandas_rule(timeframe)
    aggregated = snapshots.resample(rule, label="left", closed="left", origin="epoch").agg(["last", "mean"])
    aggregated.columns = [f"{name}_{agg}" for name, agg in aggregated.columns.to_flat_index()]
    return aggregated.reset_index(names="open_time")


def fetch_binance_vision_book_depth(
    cfg: AppConfig,
    *,
    market: str,
    start: datetime,
    end: datetime | None = None,
    force: bool = False,
    max_files: int | None = None,
    checkpoint_files: int = 5,
    feature_timeframe: str = "5m",
) -> Path:
    ensure_dirs(cfg.data.raw_dir)
    end = end or utc_now()
    if start.tzinfo is None:
        start = start.replace(tzinfo=UTC)
    if end.tzinfo is None:
        end = end.replace(tzinfo=UTC)
    feature_path = raw_book_depth_feature_path(cfg, market, feature_timeframe)
    existing = pd.DataFrame() if force else load_csv(feature_path)
    if not existing.empty:
        existing["open_time"] = pd.to_datetime(existing["open_time"], utc=True, format="mixed", errors="coerce")
        existing = existing.dropna(subset=["open_time"])
    frames: list[pd.DataFrame] = []
    files_since_flush = 0
    fetched_files = 0

    def flush_frames() -> None:
        nonlocal existing, frames, files_since_flush
        if not frames:
            return
        combined = pd.concat([existing, *frames], ignore_index=True)
        combined["open_time"] = pd.to_datetime(combined["open_time"], utc=True, format="mixed", errors="coerce")
        existing = (
            combined.dropna(subset=["open_time"])
            .drop_duplicates("open_time", keep="last")
            .sort_values("open_time")
            .reset_index(drop=True)
        )
        frames = []
        files_since_flush = 0
        _atomic_csv(existing, feature_path)

    current_day = datetime(start.year, start.month, start.day, tzinfo=UTC)
    final_day = datetime(end.year, end.month, end.day, tzinfo=UTC)
    while current_day < final_day:
        if max_files is not None and fetched_files >= max_files:
            break
        zip_path = _binance_vision_book_depth_zip_path(cfg, market, current_day)
        url = _binance_vision_book_depth_url(cfg, market, current_day)
        downloaded = _download_binance_vision_file(url, zip_path, force=force)
        if downloaded:
            day_frame = _read_binance_vision_book_depth_zip(zip_path)
            if not day_frame.empty:
                day_frame = day_frame[(day_frame["timestamp"] >= start) & (day_frame["timestamp"] < end)]
                features = aggregate_book_depth_to_timeframe(day_frame, feature_timeframe)
                if not features.empty:
                    frames.append(features)
            fetched_files += 1
            files_since_flush += 1
            if files_since_flush >= max(1, checkpoint_files):
                flush_frames()
        current_day += timedelta(days=1)
    flush_frames()
    return feature_path


def _read_binance_vision_metrics_zip(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not csv_names:
            return pd.DataFrame()
        with archive.open(csv_names[0]) as fh:
            frame = pd.read_csv(fh, low_memory=False)
    if frame.empty:
        return frame
    if "create_time" not in frame.columns:
        return pd.DataFrame()
    frame["open_time"] = pd.to_datetime(frame["create_time"], utc=True, format="mixed", errors="coerce")
    rename = {
        "sum_open_interest": "binance_metrics_sum_open_interest",
        "sum_open_interest_value": "binance_metrics_sum_open_interest_value",
        "count_toptrader_long_short_ratio": "binance_metrics_top_count_long_short_ratio",
        "sum_toptrader_long_short_ratio": "binance_metrics_top_sum_long_short_ratio",
        "count_long_short_ratio": "binance_metrics_global_long_short_ratio",
        "sum_taker_long_short_vol_ratio": "binance_metrics_taker_vol_long_short_ratio",
    }
    keep = ["open_time"]
    for source_col, dest_col in rename.items():
        if source_col in frame.columns:
            frame[dest_col] = pd.to_numeric(frame[source_col], errors="coerce")
            keep.append(dest_col)
    return (
        frame[keep]
        .dropna(subset=["open_time"])
        .drop_duplicates("open_time")
        .sort_values("open_time")
        .reset_index(drop=True)
    )


def fetch_binance_vision_metrics(
    cfg: AppConfig,
    *,
    market: str,
    start: datetime,
    end: datetime | None = None,
    force: bool = False,
    max_files: int | None = None,
    checkpoint_files: int = 10,
    timeframe: str = "5m",
) -> Path:
    ensure_dirs(cfg.data.raw_dir)
    end = end or utc_now()
    if start.tzinfo is None:
        start = start.replace(tzinfo=UTC)
    if end.tzinfo is None:
        end = end.replace(tzinfo=UTC)
    feature_path = raw_metrics_feature_path(cfg, market, timeframe)
    existing = pd.DataFrame() if force else load_csv(feature_path)
    if not existing.empty:
        existing["open_time"] = pd.to_datetime(existing["open_time"], utc=True, format="mixed", errors="coerce")
        existing = existing.dropna(subset=["open_time"])
    frames: list[pd.DataFrame] = []
    files_since_flush = 0
    fetched_files = 0

    def flush_frames() -> None:
        nonlocal existing, frames, files_since_flush
        if not frames:
            return
        combined = pd.concat([existing, *frames], ignore_index=True)
        combined["open_time"] = pd.to_datetime(combined["open_time"], utc=True, format="mixed", errors="coerce")
        existing = (
            combined.dropna(subset=["open_time"])
            .drop_duplicates("open_time", keep="last")
            .sort_values("open_time")
            .reset_index(drop=True)
        )
        frames = []
        files_since_flush = 0
        _atomic_csv(existing, feature_path)

    current_day = datetime(start.year, start.month, start.day, tzinfo=UTC)
    final_day = datetime(end.year, end.month, end.day, tzinfo=UTC)
    while current_day < final_day:
        if max_files is not None and fetched_files >= max_files:
            break
        zip_path = _binance_vision_metrics_zip_path(cfg, market, current_day)
        url = _binance_vision_metrics_url(cfg, market, current_day)
        downloaded = _download_binance_vision_file(url, zip_path, force=force)
        if downloaded:
            day_frame = _read_binance_vision_metrics_zip(zip_path)
            if not day_frame.empty:
                day_frame = day_frame[(day_frame["open_time"] >= start) & (day_frame["open_time"] < end)]
                if not day_frame.empty:
                    frames.append(day_frame)
            fetched_files += 1
            files_since_flush += 1
            if files_since_flush >= max(1, checkpoint_files):
                flush_frames()
        current_day += timedelta(days=1)
    flush_frames()
    return feature_path


async def _collect_binance_live_flow_async(
    cfg: AppConfig,
    *,
    market: str,
    seconds: float,
) -> pd.DataFrame:
    if market != "futures":
        raise ValueError("live forceOrder flow is supported for Binance futures only")
    try:
        import websockets
    except Exception as exc:  # noqa: BLE001 - optional dependency boundary
        raise RuntimeError("websockets is required for live-flow collection") from exc

    stream_symbol = cfg.data.symbol.lower()
    url = f"wss://fstream.binance.com/market/stream?streams={stream_symbol}@aggTrade/{stream_symbol}@forceOrder"
    deadline = time.monotonic() + max(0.0, float(seconds))
    rows: list[dict[str, object]] = []

    def iso_ms(ms: int | float | str | None) -> str:
        value = int(float(ms or 0))
        return datetime.fromtimestamp(value / 1000.0, UTC).isoformat()

    async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
        while time.monotonic() < deadline:
            remaining = max(0.1, deadline - time.monotonic())
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
            except asyncio.TimeoutError:
                break
            received = datetime.now(UTC).isoformat()
            msg = json.loads(raw)
            data = msg.get("data", {})
            event_type = data.get("e")
            if event_type == "aggTrade":
                price = float(data.get("p", 0.0) or 0.0)
                qty = float(data.get("q", 0.0) or 0.0)
                rows.append(
                    {
                        "received_time": received,
                        "event_time": iso_ms(data.get("E")),
                        "trade_time": iso_ms(data.get("T")),
                        "event_type": "agg_trade",
                        "symbol": data.get("s", cfg.data.symbol),
                        "side": "sell" if bool(data.get("m")) else "buy",
                        "price": price,
                        "qty": qty,
                        "quote_qty": price * qty,
                        "trade_id": data.get("a", ""),
                        "order_status": "",
                    }
                )
            elif event_type == "forceOrder":
                order = data.get("o", {})
                avg_price = float(order.get("ap", 0.0) or order.get("p", 0.0) or 0.0)
                qty = float(order.get("z", 0.0) or order.get("l", 0.0) or order.get("q", 0.0) or 0.0)
                rows.append(
                    {
                        "received_time": received,
                        "event_time": iso_ms(data.get("E")),
                        "trade_time": iso_ms(order.get("T")),
                        "event_type": "force_order",
                        "symbol": order.get("s", cfg.data.symbol),
                        "side": str(order.get("S", "")).lower(),
                        "price": avg_price,
                        "qty": qty,
                        "quote_qty": avg_price * qty,
                        "trade_id": order.get("i", ""),
                        "order_status": order.get("X", ""),
                    }
                )
    return normalize_live_flow(pd.DataFrame(rows))


def collect_binance_live_flow(
    cfg: AppConfig,
    *,
    seconds: float = 30.0,
    market: str = "futures",
) -> Path:
    ensure_dirs(cfg.data.raw_dir)
    path = raw_live_flow_path(cfg, market)
    fetched = asyncio.run(_collect_binance_live_flow_async(cfg, market=market, seconds=seconds))
    existing = load_csv(path) if path.exists() else pd.DataFrame()
    existing = normalize_live_flow(existing) if not existing.empty else existing
    if fetched.empty:
        combined = existing
    else:
        combined = pd.concat([existing, fetched], ignore_index=True)
    if combined.empty:
        _atomic_csv(pd.DataFrame(columns=[
            "received_time",
            "event_time",
            "trade_time",
            "event_type",
            "symbol",
            "side",
            "price",
            "qty",
            "quote_qty",
            "trade_id",
            "order_status",
        ]), path)
        return path
    _atomic_csv(normalize_live_flow(combined), path)
    return path


def fetch_binance_order_book_snapshot(
    cfg: AppConfig,
    *,
    market: str = "futures",
    limit: int = 100,
) -> Path:
    if market != "futures":
        raise ValueError("order-book snapshot features are currently implemented for Binance futures")
    ensure_dirs(cfg.data.raw_dir)
    valid_limits = {5, 10, 20, 50, 100, 500, 1000}
    if int(limit) not in valid_limits:
        raise ValueError(f"invalid depth limit {limit}; use one of {sorted(valid_limits)}")
    payload = request_json(
        "GET",
        BINANCE_FUTURES_URL + "/fapi/v1/depth",
        params={"symbol": cfg.data.symbol, "limit": int(limit)},
        max_retries=3,
    )
    bids = [(float(price), float(qty)) for price, qty in payload.get("bids", [])]
    asks = [(float(price), float(qty)) for price, qty in payload.get("asks", [])]
    event_ms = payload.get("E") or payload.get("T")
    snapshot_time = datetime.fromtimestamp(float(event_ms) / 1000.0, UTC) if event_ms else datetime.now(UTC)
    row: dict[str, float | str] = {"open_time": snapshot_time.isoformat()}
    if bids and asks:
        best_bid, best_bid_qty = bids[0]
        best_ask, best_ask_qty = asks[0]
        mid = (best_bid + best_ask) / 2.0
        row.update(
            {
                "book_bid_price_1": best_bid,
                "book_ask_price_1": best_ask,
                "book_bid_qty_1": best_bid_qty,
                "book_ask_qty_1": best_ask_qty,
                "book_mid": mid,
                "book_spread_bps": (best_ask / best_bid - 1.0) * 10_000.0 if best_bid else np.nan,
            }
        )
        for depth in (5, 20, 100):
            bid_slice = bids[: min(depth, len(bids))]
            ask_slice = asks[: min(depth, len(asks))]
            bid_quote = sum(price * qty for price, qty in bid_slice)
            ask_quote = sum(price * qty for price, qty in ask_slice)
            total_quote = bid_quote + ask_quote
            row[f"book_bid_quote_top{depth}"] = bid_quote
            row[f"book_ask_quote_top{depth}"] = ask_quote
            row[f"book_depth_quote_top{depth}"] = total_quote
            row[f"book_imbalance_top{depth}"] = (bid_quote - ask_quote) / total_quote if total_quote else np.nan
    path = raw_order_book_path(cfg, market)
    existing = load_csv(path) if path.exists() else pd.DataFrame()
    existing = normalize_order_book_features(existing) if not existing.empty else existing
    combined = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    _atomic_csv(normalize_order_book_features(combined), path)
    return path


def fetch_funding_rates(
    cfg: AppConfig,
    *,
    start: datetime,
    end: datetime | None = None,
    force: bool = False,
) -> Path:
    ensure_dirs(cfg.data.raw_dir)
    end = end or utc_now()
    path = cfg.data.raw_dir / f"binance_futures_{cfg.data.symbol}_funding.csv"
    existing = pd.DataFrame() if force else load_csv(path)
    start_ms = to_ms(start)
    if not existing.empty:
        existing["funding_time"] = pd.to_datetime(existing["funding_time"], utc=True, format="mixed")
        last_ms = int(existing["funding_time"].max().timestamp() * 1000)
        start_ms = max(start_ms, last_ms + 1)
    end_ms = to_ms(end)
    rows: list[dict] = []
    cursor = start_ms
    while cursor < end_ms:
        batch = request_json(
            "GET",
            BINANCE_FUTURES_URL + "/fapi/v1/fundingRate",
            params={"symbol": cfg.data.symbol, "startTime": cursor, "endTime": end_ms - 1, "limit": 1000},
        )
        if not batch:
            break
        rows.extend(batch)
        next_cursor = int(batch[-1]["fundingTime"]) + 1
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        time.sleep(cfg.runtime.fetch_sleep_seconds)

    if rows:
        fetched = pd.DataFrame(rows)
        fetched["funding_time"] = pd.to_datetime(fetched["fundingTime"], unit="ms", utc=True)
        fetched["funding_rate"] = pd.to_numeric(fetched["fundingRate"], errors="coerce")
        fetched = fetched[["funding_time", "funding_rate"]]
        combined = pd.concat([existing, fetched], ignore_index=True)
    else:
        combined = existing
    if combined.empty:
        _atomic_csv(pd.DataFrame(columns=["funding_time", "funding_rate"]), path)
        return path
    combined["funding_time"] = pd.to_datetime(combined["funding_time"], utc=True, format="mixed")
    combined["funding_rate"] = pd.to_numeric(combined["funding_rate"], errors="coerce")
    combined = combined.dropna().drop_duplicates("funding_time").sort_values("funding_time")
    _atomic_csv(combined, path)
    return path


def fetch_open_interest(
    cfg: AppConfig,
    *,
    start: datetime,
    end: datetime | None = None,
    period: str = "5m",
    force: bool = False,
) -> Path:
    ensure_dirs(cfg.data.raw_dir)
    end = end or utc_now()
    path = cfg.data.raw_dir / f"binance_futures_{cfg.data.symbol}_open_interest_{period}.csv"
    existing = pd.DataFrame() if force else load_csv(path)
    start_ms = to_ms(start)
    if not existing.empty:
        existing["open_time"] = pd.to_datetime(existing["open_time"], utc=True, format="mixed")
        last_ms = int(existing["open_time"].max().timestamp() * 1000)
        start_ms = max(start_ms, last_ms + timeframe_to_minutes(period) * 60_000)
    end_ms = to_ms(end)
    rows: list[dict] = []
    cursor = start_ms
    while cursor < end_ms:
        batch = request_json(
            "GET",
            BINANCE_FUTURES_URL + "/futures/data/openInterestHist",
            params={
                "symbol": cfg.data.symbol,
                "period": period,
                "startTime": cursor,
                "endTime": end_ms - 1,
                "limit": 500,
            },
        )
        if not batch:
            break
        rows.extend(batch)
        next_cursor = int(batch[-1]["timestamp"]) + timeframe_to_minutes(period) * 60_000
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        time.sleep(cfg.runtime.fetch_sleep_seconds)

    if rows:
        fetched = pd.DataFrame(rows)
        fetched["open_time"] = pd.to_datetime(fetched["timestamp"], unit="ms", utc=True)
        fetched["sum_open_interest"] = pd.to_numeric(fetched["sumOpenInterest"], errors="coerce")
        fetched["sum_open_interest_value"] = pd.to_numeric(fetched["sumOpenInterestValue"], errors="coerce")
        fetched = fetched[["open_time", "sum_open_interest", "sum_open_interest_value"]]
        combined = pd.concat([existing, fetched], ignore_index=True)
    else:
        combined = existing
    if combined.empty:
        _atomic_csv(pd.DataFrame(columns=["open_time", "sum_open_interest", "sum_open_interest_value"]), path)
        return path
    combined["open_time"] = pd.to_datetime(combined["open_time"], utc=True, format="mixed")
    for col in ("sum_open_interest", "sum_open_interest_value"):
        combined[col] = pd.to_numeric(combined[col], errors="coerce")
    combined = combined.dropna(subset=["open_time"]).drop_duplicates("open_time").sort_values("open_time")
    _atomic_csv(combined, path)
    return path


def fetch_futures_ratio(
    cfg: AppConfig,
    *,
    source: str,
    start: datetime,
    end: datetime | None = None,
    period: str = "5m",
    force: bool = False,
) -> Path:
    if source not in FUTURES_RATIO_SOURCES:
        raise ValueError(f"unknown futures ratio source: {source}")
    ensure_dirs(cfg.data.raw_dir)
    end = end or utc_now()
    path = raw_futures_ratio_path(cfg, source, period)
    existing = pd.DataFrame() if force else load_csv(path)
    existing = normalize_futures_ratio(existing, source) if not existing.empty else existing
    start_ms = to_ms(start)
    period_ms = timeframe_to_minutes(period) * 60_000
    if not existing.empty:
        last_ms = int(existing["open_time"].max().timestamp() * 1000)
        start_ms = max(start_ms, last_ms + period_ms)
    end_ms = to_ms(end)
    start_ms = max(start_ms, to_ms(end - FUTURES_RATIO_MAX_LOOKBACK))

    rows: list[dict] = []
    cursor = start_ms
    endpoint = FUTURES_RATIO_SOURCES[source]["endpoint"]
    if not isinstance(endpoint, str):
        raise ValueError(f"invalid futures ratio source endpoint: {source}")
    while cursor < end_ms:
        request_end = min(end_ms - 1, cursor + period_ms * 500 - 1)
        batch = request_json(
            "GET",
            BINANCE_FUTURES_URL + endpoint,
            params={
                "symbol": cfg.data.symbol,
                "period": period,
                "startTime": cursor,
                "endTime": request_end,
                "limit": 500,
            },
        )
        if not batch:
            break
        rows.extend(batch)
        next_cursor = int(batch[-1]["timestamp"]) + period_ms
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        time.sleep(cfg.runtime.fetch_sleep_seconds)

    if rows:
        fetched = normalize_futures_ratio(pd.DataFrame(rows), source)
        combined = pd.concat([existing, fetched], ignore_index=True)
    else:
        combined = existing
    columns = ["open_time"] + list(FUTURES_RATIO_SOURCES[source]["columns"].values())  # type: ignore[index, union-attr]
    if combined.empty:
        _atomic_csv(pd.DataFrame(columns=columns), path)
        return path
    combined = normalize_futures_ratio(combined, source)
    combined = combined[combined["open_time"] < end]
    _atomic_csv(combined, path)
    return path


def load_klines(cfg: AppConfig, *, market: str | None = None, interval: str | None = None) -> pd.DataFrame:
    market = market or cfg.data.market
    interval = interval or cfg.data.base_interval
    path = raw_kline_path(cfg, market, interval)
    if not path.exists():
        raise FileNotFoundError(f"missing candle data: {path}. Run `python btc_predict.py fetch` first.")
    return normalize_klines(load_csv(path))


def available_kline_intervals(cfg: AppConfig, *, market: str) -> list[str]:
    prefix = f"binance_{market}_{cfg.data.symbol}_"
    intervals: list[str] = []
    for path in cfg.data.raw_dir.glob(f"{prefix}*.csv"):
        match = re.match(rf"{re.escape(prefix)}(.+)\.csv$", path.name)
        if match:
            interval = match.group(1)
            try:
                timeframe_to_minutes(interval)
            except ValueError:
                continue
            intervals.append(interval)
    return sorted(set(intervals), key=timeframe_to_minutes)


def select_source_interval(cfg: AppConfig, *, market: str, timeframe: str) -> str:
    requested_minutes = timeframe_to_minutes(timeframe)
    configured = cfg.data.base_interval
    candidates = available_kline_intervals(cfg, market=market)
    if configured not in candidates and raw_kline_path(cfg, market, configured).exists():
        candidates.append(configured)
    if timeframe in candidates:
        return timeframe
    usable = [interval for interval in candidates if timeframe_to_minutes(interval) <= requested_minutes]
    if usable:
        return max(usable, key=timeframe_to_minutes)
    if raw_kline_path(cfg, market, configured).exists() and timeframe_to_minutes(configured) <= requested_minutes:
        return configured
    raise FileNotFoundError(
        f"missing source candles for {market} {timeframe}. "
        f"Run `python btc_predict.py fetch --intervals {timeframe},1m,5m --markets {market}` first."
    )


def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if df.empty:
        return df
    frame = normalize_klines(df).set_index("open_time").sort_index()
    agg: dict[str, str] = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "quote_volume": "sum",
        "trade_count": "sum",
        "taker_buy_base": "sum",
        "taker_buy_quote": "sum",
    }
    present = {k: v for k, v in agg.items() if k in frame.columns}
    resampled = frame.resample(timeframe_to_pandas_rule(timeframe), label="left", closed="left", origin="epoch").agg(present)
    resampled = resampled.dropna(subset=["open", "high", "low", "close"])
    return resampled.reset_index()


def lower_timeframe_rollup_features(candles: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    source = normalize_klines(candles)
    if source.empty or len(source) < 3:
        return pd.DataFrame()
    target_minutes = timeframe_to_minutes(timeframe)
    times = pd.to_datetime(source["open_time"], utc=True, format="mixed").sort_values()
    median_step = times.diff().dropna().median()
    if pd.isna(median_step):
        return pd.DataFrame()
    source_minutes = max(1, int(round(pd.Timedelta(median_step).total_seconds() / 60.0)))
    if source_minutes >= target_minutes or target_minutes % source_minutes != 0:
        return pd.DataFrame()

    frame = source.set_index("open_time").sort_index()
    close = frame["close"].replace(0.0, np.nan)
    volume = frame.get("volume", pd.Series(np.nan, index=frame.index)).replace(0.0, np.nan)
    quote_volume = frame.get("quote_volume", pd.Series(np.nan, index=frame.index)).replace(0.0, np.nan)
    trade_count = frame.get("trade_count", pd.Series(np.nan, index=frame.index)).replace(0.0, np.nan)
    taker_buy_base = frame.get("taker_buy_base", pd.Series(np.nan, index=frame.index))
    taker_buy_quote = frame.get("taker_buy_quote", pd.Series(np.nan, index=frame.index))
    taker_base_imbalance = 2.0 * taker_buy_base / volume - 1.0
    taker_quote_imbalance = 2.0 * taker_buy_quote / quote_volume - 1.0
    ret = np.log(close / close.shift(1))
    body = np.log(frame["close"].replace(0.0, np.nan) / frame["open"].replace(0.0, np.nan))
    range_return = np.log(frame["high"].replace(0.0, np.nan) / frame["low"].replace(0.0, np.nan))
    work = pd.DataFrame(
        {
            "ltf_child_count": 1.0,
            "ltf_return": ret,
            "ltf_body_return": body,
            "ltf_range_return": range_return,
            "ltf_up": (ret > 0.0).astype(float),
            "ltf_down": (ret < 0.0).astype(float),
            "ltf_quote_volume": quote_volume,
            "ltf_log_quote_volume": np.log1p(quote_volume.clip(lower=0.0)),
            "ltf_trade_count": trade_count,
            "ltf_taker_base_imbalance": taker_base_imbalance,
            "ltf_taker_quote_imbalance": taker_quote_imbalance,
            "ltf_open": frame["open"],
            "ltf_high": frame["high"],
            "ltf_low": frame["low"],
            "ltf_close": frame["close"],
        },
        index=frame.index,
    )
    rule = timeframe_to_pandas_rule(timeframe)
    grouped = work.resample(rule, label="left", closed="left", origin="epoch")
    out = grouped.agg(
        {
            "ltf_child_count": "sum",
            "ltf_return": ["sum", "mean", "std", "min", "max", "first", "last"],
            "ltf_body_return": ["sum", "mean", "last"],
            "ltf_range_return": ["mean", "max", "last"],
            "ltf_up": "sum",
            "ltf_down": "sum",
            "ltf_quote_volume": ["sum", "max", "last"],
            "ltf_log_quote_volume": ["mean", "last"],
            "ltf_trade_count": ["sum", "max", "last"],
            "ltf_taker_base_imbalance": ["mean", "std", "first", "last"],
            "ltf_taker_quote_imbalance": ["mean", "std", "first", "last"],
            "ltf_open": "first",
            "ltf_high": "max",
            "ltf_low": "min",
            "ltf_close": "last",
        }
    )
    out.columns = ["_".join(part for part in col if part).rstrip("_") for col in out.columns.to_flat_index()]
    out = out.rename(
        columns={
            "ltf_child_count_sum": "ltf_child_count",
            "ltf_return_sum": "ltf_return_sum",
            "ltf_return_mean": "ltf_return_mean",
            "ltf_return_std": "ltf_return_std",
            "ltf_return_min": "ltf_return_min",
            "ltf_return_max": "ltf_return_max",
            "ltf_return_first": "ltf_return_first",
            "ltf_return_last": "ltf_return_last",
        }
    )
    out["ltf_up_share"] = out["ltf_up_sum"] / out["ltf_child_count"].replace(0.0, np.nan)
    out["ltf_down_share"] = out["ltf_down_sum"] / out["ltf_child_count"].replace(0.0, np.nan)
    out["ltf_quote_volume_last_share"] = out["ltf_quote_volume_last"] / out["ltf_quote_volume_sum"].replace(0.0, np.nan)
    out["ltf_quote_volume_max_share"] = out["ltf_quote_volume_max"] / out["ltf_quote_volume_sum"].replace(0.0, np.nan)
    out["ltf_trade_count_last_share"] = out["ltf_trade_count_last"] / out["ltf_trade_count_sum"].replace(0.0, np.nan)
    out["ltf_trade_count_max_share"] = out["ltf_trade_count_max"] / out["ltf_trade_count_sum"].replace(0.0, np.nan)
    out["ltf_taker_quote_imbalance_change"] = out["ltf_taker_quote_imbalance_last"] - out["ltf_taker_quote_imbalance_first"]
    out["ltf_taker_base_imbalance_change"] = out["ltf_taker_base_imbalance_last"] - out["ltf_taker_base_imbalance_first"]
    out["ltf_close_position"] = (
        (out["ltf_close_last"] - out["ltf_low_min"]) / (out["ltf_high_max"] - out["ltf_low_min"]).replace(0.0, np.nan)
    )
    out["ltf_close_vs_open"] = out["ltf_close_last"] / out["ltf_open_first"].replace(0.0, np.nan) - 1.0
    out["ltf_late_impulse"] = out["ltf_return_last"] - out["ltf_return_mean"]
    out["ltf_range_expansion"] = out["ltf_range_return_last"] / out["ltf_range_return_mean"].replace(0.0, np.nan)
    child_count = target_minutes // source_minutes
    if 1 < child_count <= 12:
        bucket = work.index.floor(rule)
        position = ((work.index - bucket) / pd.Timedelta(minutes=source_minutes)).astype(int)
        sequence = work.assign(__bucket=bucket, __position=position)
        sequence_columns = (
            "ltf_return",
            "ltf_body_return",
            "ltf_range_return",
            "ltf_log_quote_volume",
            "ltf_taker_base_imbalance",
            "ltf_taker_quote_imbalance",
        )
        for column in sequence_columns:
            pivot = sequence.pivot_table(index="__bucket", columns="__position", values=column, aggfunc="last")
            for pos in range(child_count):
                if pos in pivot.columns:
                    out[f"{column}_pos{pos + 1}"] = pivot[pos].reindex(out.index)
                else:
                    out[f"{column}_pos{pos + 1}"] = np.nan
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.drop(columns=[col for col in ("ltf_open_first", "ltf_high_max", "ltf_low_min", "ltf_close_last") if col in out.columns])
    return out.reset_index(names="open_time").dropna(subset=["ltf_child_count"])


def aggregate_trades_to_timeframe(trades: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    trades = normalize_agg_trades(trades)
    if trades.empty:
        return pd.DataFrame()
    frame = trades.set_index("trade_time").sort_index()
    taker_buy = ~frame["buyer_is_maker"].astype(bool)
    signed_qty = frame["qty"].where(taker_buy, -frame["qty"])
    signed_quote = frame["quote_qty"].where(taker_buy, -frame["quote_qty"])
    work = pd.DataFrame(
        {
            "agg_trade_count": 1.0,
            "agg_trade_qty": frame["qty"],
            "agg_trade_quote_qty": frame["quote_qty"],
            "agg_taker_buy_qty": frame["qty"].where(taker_buy, 0.0),
            "agg_taker_sell_qty": frame["qty"].where(~taker_buy, 0.0),
            "agg_taker_buy_quote": frame["quote_qty"].where(taker_buy, 0.0),
            "agg_taker_sell_quote": frame["quote_qty"].where(~taker_buy, 0.0),
            "agg_signed_qty": signed_qty,
            "agg_signed_quote": signed_quote,
            "agg_price_qty": frame["price"] * frame["qty"],
            "agg_max_trade_quote": frame["quote_qty"],
        },
        index=frame.index,
    )
    rule = timeframe_to_pandas_rule(timeframe)
    aggregated = work.resample(rule, label="left", closed="left", origin="epoch").agg(
        {
            "agg_trade_count": "sum",
            "agg_trade_qty": "sum",
            "agg_trade_quote_qty": "sum",
            "agg_taker_buy_qty": "sum",
            "agg_taker_sell_qty": "sum",
            "agg_taker_buy_quote": "sum",
            "agg_taker_sell_quote": "sum",
            "agg_signed_qty": "sum",
            "agg_signed_quote": "sum",
            "agg_price_qty": "sum",
            "agg_max_trade_quote": "max",
        }
    )
    aggregated["agg_trade_vwap"] = aggregated["agg_price_qty"] / aggregated["agg_trade_qty"].replace(0.0, np.nan)
    aggregated["agg_taker_buy_ratio"] = aggregated["agg_taker_buy_qty"] / aggregated["agg_trade_qty"].replace(0.0, np.nan)
    aggregated["agg_taker_quote_imbalance"] = aggregated["agg_signed_quote"] / aggregated["agg_trade_quote_qty"].replace(0.0, np.nan)
    aggregated["agg_avg_trade_quote"] = aggregated["agg_trade_quote_qty"] / aggregated["agg_trade_count"].replace(0.0, np.nan)
    aggregated = aggregated.drop(columns=["agg_price_qty"])
    return aggregated.reset_index(names="open_time")


def aggregate_live_flow_to_timeframe(flow: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    flow = normalize_live_flow(flow)
    if flow.empty:
        return pd.DataFrame()
    rule = timeframe_to_pandas_rule(timeframe)
    pieces: list[pd.DataFrame] = []

    trades = flow[flow["event_type"].eq("agg_trade")].copy()
    if not trades.empty:
        frame = trades.set_index("trade_time").sort_index()
        taker_buy = frame["side"].astype(str).str.lower().eq("buy")
        signed_quote = frame["quote_qty"].where(taker_buy, -frame["quote_qty"])
        signed_qty = frame["qty"].where(taker_buy, -frame["qty"])
        work = pd.DataFrame(
            {
                "live_trade_count": 1.0,
                "live_trade_qty": frame["qty"],
                "live_trade_quote_qty": frame["quote_qty"],
                "live_taker_buy_quote": frame["quote_qty"].where(taker_buy, 0.0),
                "live_taker_sell_quote": frame["quote_qty"].where(~taker_buy, 0.0),
                "live_signed_quote": signed_quote,
                "live_signed_qty": signed_qty,
                "live_price_qty": frame["price"] * frame["qty"],
                "live_max_trade_quote": frame["quote_qty"],
            },
            index=frame.index,
        )
        trade_agg = work.resample(rule, label="left", closed="left", origin="epoch").agg(
            {
                "live_trade_count": "sum",
                "live_trade_qty": "sum",
                "live_trade_quote_qty": "sum",
                "live_taker_buy_quote": "sum",
                "live_taker_sell_quote": "sum",
                "live_signed_quote": "sum",
                "live_signed_qty": "sum",
                "live_price_qty": "sum",
                "live_max_trade_quote": "max",
            }
        )
        trade_agg["live_trade_vwap"] = trade_agg["live_price_qty"] / trade_agg["live_trade_qty"].replace(0.0, np.nan)
        trade_agg["live_taker_quote_imbalance"] = trade_agg["live_signed_quote"] / trade_agg[
            "live_trade_quote_qty"
        ].replace(0.0, np.nan)
        trade_agg["live_avg_trade_quote"] = trade_agg["live_trade_quote_qty"] / trade_agg[
            "live_trade_count"
        ].replace(0.0, np.nan)
        pieces.append(trade_agg.drop(columns=["live_price_qty"]))

    liquidations = flow[flow["event_type"].eq("force_order")].copy()
    if not liquidations.empty:
        frame = liquidations.set_index("trade_time").sort_index()
        buy_side = frame["side"].astype(str).str.lower().eq("buy")
        signed_quote = frame["quote_qty"].where(buy_side, -frame["quote_qty"])
        work = pd.DataFrame(
            {
                "liq_count": 1.0,
                "liq_qty": frame["qty"],
                "liq_quote_qty": frame["quote_qty"],
                "liq_buy_quote": frame["quote_qty"].where(buy_side, 0.0),
                "liq_sell_quote": frame["quote_qty"].where(~buy_side, 0.0),
                "liq_signed_quote": signed_quote,
                "liq_max_quote": frame["quote_qty"],
            },
            index=frame.index,
        )
        liq_agg = work.resample(rule, label="left", closed="left", origin="epoch").agg(
            {
                "liq_count": "sum",
                "liq_qty": "sum",
                "liq_quote_qty": "sum",
                "liq_buy_quote": "sum",
                "liq_sell_quote": "sum",
                "liq_signed_quote": "sum",
                "liq_max_quote": "max",
            }
        )
        liq_agg["liq_buy_sell_imbalance"] = liq_agg["liq_signed_quote"] / liq_agg["liq_quote_qty"].replace(0.0, np.nan)
        liq_agg["liq_avg_quote"] = liq_agg["liq_quote_qty"] / liq_agg["liq_count"].replace(0.0, np.nan)
        pieces.append(liq_agg)

    if not pieces:
        return pd.DataFrame()
    out = pd.concat(pieces, axis=1).reset_index(names="open_time")
    return out.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)


def context_features(candles: pd.DataFrame, context_timeframe: str) -> pd.DataFrame:
    frame = normalize_klines(candles).copy()
    if frame.empty:
        return pd.DataFrame()
    if timeframe_to_minutes(context_timeframe) != timeframe_to_minutes("1m"):
        frame = resample_ohlcv(frame, context_timeframe)
    close = frame["close"].replace(0.0, np.nan)
    out = pd.DataFrame({"open_time": pd.to_datetime(frame["open_time"], utc=True)})
    prefix = f"ctx_{context_timeframe}_"
    ret = np.log(close / close.shift(1))
    out[prefix + "return_1"] = ret
    for window in (2, 3, 6, 12):
        out[prefix + f"return_sum_{window}"] = ret.rolling(window, min_periods=max(2, window // 2)).sum()
        out[prefix + f"return_std_{window}"] = ret.rolling(window, min_periods=max(2, window // 2)).std(ddof=0)
    out[prefix + "range_return"] = np.log(frame["high"].replace(0.0, np.nan) / frame["low"].replace(0.0, np.nan))
    for span in (5, 12, 21):
        ema = close.ewm(span=span, adjust=False, min_periods=span).mean()
        out[prefix + f"ema_gap_{span}"] = close / ema.replace(0.0, np.nan) - 1.0
    out["open_time"] = out["open_time"] + timeframe_to_timedelta(context_timeframe)
    return out.drop_duplicates("open_time").sort_values("open_time")


def _asof_join(left: pd.DataFrame, right: pd.DataFrame, columns: Iterable[str], *, tolerance: timedelta | None = None) -> pd.DataFrame:
    if right.empty:
        return left
    out = left.sort_index().copy()
    series = right.sort_index()[list(columns)]
    if tolerance is None:
        joined = series.reindex(out.index, method="ffill")
    else:
        joined = pd.merge_asof(
            out.reset_index()[["open_time"]],
            series.reset_index(),
            on="open_time",
            direction="backward",
            tolerance=pd.Timedelta(tolerance),
        ).set_index("open_time")[list(columns)]
    for col in columns:
        out[col] = joined[col].to_numpy()
    return out


def _load_basis_source(cfg: AppConfig, source: str, timeframe: str) -> pd.DataFrame:
    source_minutes = timeframe_to_minutes(timeframe)
    candidates = [timeframe, "5m", "1m"]
    seen: set[str] = set()
    for interval in candidates:
        if interval in seen:
            continue
        seen.add(interval)
        path = raw_futures_basis_path(cfg, source, interval)
        if not path.exists():
            continue
        frame = normalize_klines(load_csv(path))
        if frame.empty:
            continue
        interval_minutes = timeframe_to_minutes(interval)
        if source_minutes >= interval_minutes and source_minutes != interval_minutes:
            frame = resample_ohlcv(frame, timeframe)
        prefix = FUTURES_BASIS_SOURCES[source]["prefix"]
        columns = {
            "open": f"{prefix}_open",
            "high": f"{prefix}_high",
            "low": f"{prefix}_low",
            "close": f"{prefix}_close",
        }
        return frame[["open_time", *columns.keys()]].rename(columns=columns)
    return pd.DataFrame()


def _load_binance_cross_symbol_source(cfg: AppConfig, symbol: str, timeframe: str) -> pd.DataFrame:
    source_minutes = timeframe_to_minutes(timeframe)
    candidates = [timeframe, "5m", "1m"]
    seen: set[str] = set()
    for interval in candidates:
        if interval in seen:
            continue
        seen.add(interval)
        path = raw_symbol_kline_path(cfg, cfg.data.market, symbol, interval)
        if not path.exists():
            continue
        try:
            interval_minutes = timeframe_to_minutes(interval)
        except ValueError:
            continue
        if interval_minutes > source_minutes:
            continue
        frame = normalize_klines(load_csv(path))
        if frame.empty:
            continue
        if source_minutes != interval_minutes:
            frame = resample_ohlcv(frame, timeframe)
        prefix = symbol.replace("USDT", "").lower()
        columns = {
            "open": f"{prefix}_open",
            "high": f"{prefix}_high",
            "low": f"{prefix}_low",
            "close": f"{prefix}_close",
            "volume": f"{prefix}_volume",
            "quote_volume": f"{prefix}_quote_volume",
            "trade_count": f"{prefix}_trade_count",
            "taker_buy_base": f"{prefix}_taker_buy_base",
            "taker_buy_quote": f"{prefix}_taker_buy_quote",
        }
        keep = ["open_time"] + [col for col in columns if col in frame.columns]
        return frame[keep].rename(columns=columns)
    return pd.DataFrame()


def _load_coinbase_source(cfg: AppConfig, timeframe: str) -> pd.DataFrame:
    source_minutes = timeframe_to_minutes(timeframe)
    candidates = [timeframe, "5m", "1m"]
    seen: set[str] = set()
    for interval in candidates:
        if interval in seen:
            continue
        seen.add(interval)
        path = coinbase_path(cfg, interval)
        if not path.exists():
            continue
        try:
            interval_minutes = timeframe_to_minutes(interval)
        except ValueError:
            continue
        if interval_minutes > source_minutes:
            continue
        frame = normalize_klines(load_csv(path))
        if frame.empty:
            continue
        if source_minutes != interval_minutes:
            frame = resample_ohlcv(frame, timeframe)
        columns = {
            "open": "coinbase_open",
            "high": "coinbase_high",
            "low": "coinbase_low",
            "close": "coinbase_close",
            "volume": "coinbase_volume",
            "quote_volume": "coinbase_quote_volume",
        }
        keep = ["open_time"] + [col for col in columns if col in frame.columns]
        return frame[keep].rename(columns=columns)
    return pd.DataFrame()


def _load_bybit_source(cfg: AppConfig, timeframe: str) -> pd.DataFrame:
    source_minutes = timeframe_to_minutes(timeframe)
    candidates = [timeframe, "5m", "1m"]
    seen: set[str] = set()
    for interval in candidates:
        if interval in seen:
            continue
        seen.add(interval)
        path = bybit_kline_path(cfg, interval)
        if not path.exists():
            continue
        try:
            interval_minutes = timeframe_to_minutes(interval)
        except ValueError:
            continue
        if interval_minutes > source_minutes:
            continue
        frame = normalize_klines(load_csv(path))
        if frame.empty:
            continue
        if source_minutes != interval_minutes:
            frame = resample_ohlcv(frame, timeframe)
        columns = {
            "open": "bybit_open",
            "high": "bybit_high",
            "low": "bybit_low",
            "close": "bybit_close",
            "volume": "bybit_volume",
            "quote_volume": "bybit_quote_volume",
        }
        keep = ["open_time"] + [col for col in columns if col in frame.columns]
        return frame[keep].rename(columns=columns)
    return pd.DataFrame()


def _load_okx_source(cfg: AppConfig, timeframe: str) -> pd.DataFrame:
    source_minutes = timeframe_to_minutes(timeframe)
    candidates = [timeframe, "5m", "1m"]
    seen: set[str] = set()
    for interval in candidates:
        if interval in seen:
            continue
        seen.add(interval)
        path = okx_kline_path(cfg, interval)
        if not path.exists():
            continue
        try:
            interval_minutes = timeframe_to_minutes(interval)
        except ValueError:
            continue
        if interval_minutes > source_minutes:
            continue
        frame = normalize_okx_klines(load_csv(path))
        if frame.empty:
            continue
        if source_minutes != interval_minutes:
            frame = resample_ohlcv(frame, timeframe)
        columns = {
            "open": "okx_open",
            "high": "okx_high",
            "low": "okx_low",
            "close": "okx_close",
            "volume": "okx_volume",
            "quote_volume": "okx_quote_volume",
        }
        keep = ["open_time"] + [col for col in columns if col in frame.columns]
        return frame[keep].rename(columns=columns)
    return pd.DataFrame()


def load_timeframe_dataset(cfg: AppConfig, timeframe: str) -> pd.DataFrame:
    source_interval = select_source_interval(cfg, market=cfg.data.market, timeframe=timeframe)
    source_base = load_klines(cfg, market=cfg.data.market, interval=source_interval)
    base = source_base.copy()
    if timeframe_to_minutes(timeframe) != timeframe_to_minutes(source_interval):
        base = resample_ohlcv(base, timeframe)
    df = base.set_index("open_time").sort_index()

    lower_source = source_base
    if timeframe_to_minutes(timeframe) > 1 and raw_kline_path(cfg, cfg.data.market, "1m").exists():
        lower_source = load_klines(cfg, market=cfg.data.market, interval="1m")
    lower_features = lower_timeframe_rollup_features(lower_source, timeframe)
    if not lower_features.empty:
        lower_features = lower_features.set_index("open_time").sort_index()
        df = pd.concat([df, lower_features.reindex(df.index)], axis=1).copy()

    if cfg.data.include_basis_klines:
        for source in FUTURES_BASIS_SOURCES:
            basis = _load_basis_source(cfg, source, timeframe)
            if basis.empty:
                continue
            basis = basis.set_index("open_time").sort_index()
            for col in basis.columns:
                df[col] = basis[col].reindex(df.index, method="ffill")
        if "mark_price_close" in df.columns:
            df["futures_mark_spread_bps"] = (df["close"] / df["mark_price_close"].replace(0.0, np.nan) - 1.0) * 10_000.0
        if "mark_price_close" in df.columns and "index_price_close" in df.columns:
            df["mark_index_basis_bps"] = (
                df["mark_price_close"] / df["index_price_close"].replace(0.0, np.nan) - 1.0
            ) * 10_000.0
        if "premium_index_close" in df.columns:
            df["premium_index_bps"] = pd.to_numeric(df["premium_index_close"], errors="coerce") * 10_000.0

    if cfg.data.include_context_timeframes:
        for symbol in CROSS_ASSET_SYMBOLS:
            cross_symbol = _load_binance_cross_symbol_source(cfg, symbol, timeframe)
            if cross_symbol.empty:
                continue
            cross_symbol = cross_symbol.set_index("open_time").sort_index()
            prefix = symbol.replace("USDT", "").lower()
            close_col = f"{prefix}_close"
            df[close_col] = cross_symbol[close_col].reindex(df.index, method="ffill")
            df[f"{prefix}_return_1"] = np.log(df[close_col] / df[close_col].shift(1))
            for col in (f"{prefix}_volume", f"{prefix}_quote_volume", f"{prefix}_trade_count", f"{prefix}_taker_buy_base", f"{prefix}_taker_buy_quote"):
                if col in cross_symbol.columns:
                    df[col] = cross_symbol[col].reindex(df.index, method="ffill")
            df[f"futures_{prefix}_spread_bps"] = (df["close"] / df[close_col].replace(0.0, np.nan) - 1.0) * 10_000.0

    if cfg.data.include_coinbase:
        coinbase = _load_coinbase_source(cfg, timeframe)
        if not coinbase.empty:
            coinbase = coinbase.set_index("open_time").sort_index()
            df["coinbase_close"] = coinbase["coinbase_close"].reindex(df.index, method="ffill")
            df["coinbase_return_1"] = np.log(df["coinbase_close"] / df["coinbase_close"].shift(1))
            if "coinbase_volume" in coinbase.columns:
                df["coinbase_volume"] = coinbase["coinbase_volume"].reindex(df.index, method="ffill")
            if "coinbase_quote_volume" in coinbase.columns:
                df["coinbase_quote_volume"] = coinbase["coinbase_quote_volume"].reindex(df.index, method="ffill")
            df["futures_coinbase_spread_bps"] = (df["close"] / df["coinbase_close"].replace(0.0, np.nan) - 1.0) * 10_000.0

    if cfg.data.include_bybit:
        bybit = _load_bybit_source(cfg, timeframe)
        if not bybit.empty:
            bybit = bybit.set_index("open_time").sort_index()
            df["bybit_close"] = bybit["bybit_close"].reindex(df.index, method="ffill")
            df["bybit_return_1"] = np.log(df["bybit_close"] / df["bybit_close"].shift(1))
            if "bybit_volume" in bybit.columns:
                df["bybit_volume"] = bybit["bybit_volume"].reindex(df.index, method="ffill")
            if "bybit_quote_volume" in bybit.columns:
                df["bybit_quote_volume"] = bybit["bybit_quote_volume"].reindex(df.index, method="ffill")
            df["futures_bybit_spread_bps"] = (df["close"] / df["bybit_close"].replace(0.0, np.nan) - 1.0) * 10_000.0

    if cfg.data.include_okx:
        okx = _load_okx_source(cfg, timeframe)
        if not okx.empty:
            okx = okx.set_index("open_time").sort_index()
            df["okx_close"] = okx["okx_close"].reindex(df.index, method="ffill")
            df["okx_return_1"] = np.log(df["okx_close"] / df["okx_close"].shift(1))
            if "okx_volume" in okx.columns:
                df["okx_volume"] = okx["okx_volume"].reindex(df.index, method="ffill")
            if "okx_quote_volume" in okx.columns:
                df["okx_quote_volume"] = okx["okx_quote_volume"].reindex(df.index, method="ffill")
            df["futures_okx_spread_bps"] = (df["close"] / df["okx_close"].replace(0.0, np.nan) - 1.0) * 10_000.0

    if cfg.data.include_spot:
        try:
            spot_interval = select_source_interval(cfg, market="spot", timeframe=timeframe)
        except FileNotFoundError:
            spot_interval = cfg.data.base_interval
        spot_path = raw_kline_path(cfg, "spot", spot_interval)
        if spot_path.exists():
            spot = normalize_klines(load_csv(spot_path))
            if timeframe_to_minutes(timeframe) != timeframe_to_minutes(spot_interval):
                spot = resample_ohlcv(spot, timeframe)
            spot = spot.set_index("open_time").sort_index()
            df["spot_close"] = spot["close"].reindex(df.index, method="ffill")
            df["spot_return_1"] = np.log(df["spot_close"] / df["spot_close"].shift(1))
            df["futures_spot_spread_bps"] = (df["close"] / df["spot_close"] - 1.0) * 10_000.0

    if cfg.data.include_hyperliquid:
        hl_path = hyperliquid_path(cfg, "5m")
        if hl_path.exists() and timeframe_to_minutes(timeframe) >= 5:
            hyper = normalize_klines(load_csv(hl_path))
            if not hyper.empty:
                if timeframe_to_minutes(timeframe) != 5:
                    hyper = resample_ohlcv(hyper, timeframe)
                hyper = hyper.set_index("open_time").sort_index()
                df["hyperliquid_close"] = hyper["close"].reindex(df.index, method="ffill")
                df["hyperliquid_return_1"] = np.log(df["hyperliquid_close"] / df["hyperliquid_close"].shift(1))
                if "volume" in hyper.columns:
                    df["hyperliquid_volume"] = hyper["volume"].reindex(df.index, method="ffill")
                if "quote_volume" in hyper.columns:
                    df["hyperliquid_quote_volume"] = hyper["quote_volume"].reindex(df.index, method="ffill")
                df["futures_hyperliquid_spread_bps"] = (df["close"] / df["hyperliquid_close"] - 1.0) * 10_000.0

    if cfg.data.include_agg_trades:
        feature_path = raw_agg_trade_feature_path(cfg, cfg.data.market, timeframe)
        if feature_path.exists():
            trade_features = load_csv(feature_path)
            if not trade_features.empty:
                trade_features = trade_features.set_index("open_time").sort_index()
                df = pd.concat([df, trade_features.reindex(df.index)], axis=1).copy()
        else:
            paths = [raw_agg_trade_path(cfg, cfg.data.market), cfg.data.raw_dir / f"binance_{cfg.data.market}_{cfg.data.symbol}_recent_aggTrades.csv"]
            trade_frames = [load_csv(path) for path in paths if path.exists()]
            if trade_frames:
                trades = normalize_agg_trades(pd.concat(trade_frames, ignore_index=True))
                if not trades.empty:
                    trade_features = aggregate_trades_to_timeframe(trades, timeframe).set_index("open_time").sort_index()
                    df = pd.concat([df, trade_features.reindex(df.index)], axis=1).copy()

    if cfg.data.include_live_flow:
        live_flow_file = raw_live_flow_path(cfg, cfg.data.market)
        if live_flow_file.exists():
            live_flow = normalize_live_flow(load_csv(live_flow_file))
            if not live_flow.empty:
                live_features = aggregate_live_flow_to_timeframe(live_flow, timeframe).set_index("open_time").sort_index()
                df = pd.concat([df, live_features.reindex(df.index)], axis=1).copy()

    if cfg.data.include_order_book:
        depth_feature_file = raw_book_depth_feature_path(cfg, cfg.data.market, timeframe)
        if depth_feature_file.exists():
            depth_features = load_csv(depth_feature_file)
            if not depth_features.empty:
                depth_features = depth_features.set_index("open_time").sort_index()
                df = pd.concat([df, depth_features.reindex(df.index)], axis=1).copy()
        book_file = raw_order_book_path(cfg, cfg.data.market)
        if book_file.exists():
            book = normalize_order_book_features(load_csv(book_file))
            if not book.empty:
                book = book.set_index("open_time").sort_index()
                columns = [col for col in book.columns if col.startswith("book_")]
                df = _asof_join(
                    df,
                    book,
                    columns,
                    tolerance=timedelta(minutes=max(5, timeframe_to_minutes(timeframe) * 2)),
                )

    if cfg.data.include_context_timeframes:
        for context_timeframe in cfg.data.context_timeframes:
            if timeframe_to_minutes(context_timeframe) <= timeframe_to_minutes(timeframe):
                continue
            context = context_features(base, context_timeframe)
            if context.empty:
                continue
            context = context.set_index("open_time").sort_index()
            columns = [col for col in context.columns if col.startswith(f"ctx_{context_timeframe}_")]
            df = _asof_join(
                df,
                context,
                columns,
                tolerance=timedelta(days=14),
            )

    if cfg.data.include_open_interest:
        oi_path = cfg.data.raw_dir / f"binance_futures_{cfg.data.symbol}_open_interest_5m.csv"
        if oi_path.exists():
            oi = load_csv(oi_path)
            if not oi.empty:
                oi["open_time"] = pd.to_datetime(oi["open_time"], utc=True, format="mixed")
                for col in ("sum_open_interest", "sum_open_interest_value"):
                    oi[col] = pd.to_numeric(oi[col], errors="coerce")
                oi = oi.set_index("open_time").sort_index()
                df = _asof_join(
                    df,
                    oi,
                    ["sum_open_interest", "sum_open_interest_value"],
                    tolerance=timedelta(days=7),
                )

        metrics_path = raw_metrics_feature_path(cfg, cfg.data.market, "5m")
        if metrics_path.exists():
            metrics = load_csv(metrics_path)
            if not metrics.empty:
                metrics["open_time"] = pd.to_datetime(metrics["open_time"], utc=True, format="mixed")
                metric_cols = [
                    col
                    for col in metrics.columns
                    if col.startswith("binance_metrics_") and col not in {"binance_metrics_sum_open_interest", "binance_metrics_sum_open_interest_value"}
                ]
                for col in metric_cols:
                    metrics[col] = pd.to_numeric(metrics[col], errors="coerce")
                if metric_cols:
                    metrics = metrics.set_index("open_time").sort_index()
                    df = _asof_join(df, metrics, metric_cols, tolerance=timedelta(days=7))

    if cfg.data.include_derivative_ratios:
        for source, spec in FUTURES_RATIO_SOURCES.items():
            ratio_path = raw_futures_ratio_path(cfg, source, "5m")
            if not ratio_path.exists():
                continue
            ratios = load_csv(ratio_path)
            if ratios.empty:
                continue
            ratios = normalize_futures_ratio(ratios, source).set_index("open_time").sort_index()
            columns = list(spec["columns"].values()) if isinstance(spec["columns"], dict) else []
            if columns:
                df = _asof_join(df, ratios, columns, tolerance=timedelta(days=7))

    if cfg.data.include_bybit:
        bybit_oi_path = bybit_open_interest_path(cfg, "5m")
        if bybit_oi_path.exists():
            bybit_oi = load_csv(bybit_oi_path)
            if not bybit_oi.empty:
                bybit_oi = normalize_bybit_open_interest(bybit_oi).set_index("open_time").sort_index()
                df = _asof_join(df, bybit_oi, ["bybit_open_interest"], tolerance=timedelta(days=7))

    if cfg.data.include_okx:
        okx_oi_path = okx_open_interest_path(cfg, "5m")
        if okx_oi_path.exists():
            okx_oi = load_csv(okx_oi_path)
            if not okx_oi.empty:
                okx_oi = normalize_okx_open_interest(okx_oi).set_index("open_time").sort_index()
                cols = [col for col in ("okx_open_interest", "okx_open_interest_ccy", "okx_open_interest_usd") if col in okx_oi.columns]
                if cols:
                    df = _asof_join(df, okx_oi, cols, tolerance=timedelta(days=7))
        okx_lsr_path = okx_long_short_ratio_path(cfg, "5m")
        if okx_lsr_path.exists():
            okx_lsr = load_csv(okx_lsr_path)
            if not okx_lsr.empty:
                okx_lsr = normalize_okx_long_short_ratio(okx_lsr).set_index("open_time").sort_index()
                df = _asof_join(df, okx_lsr, ["okx_long_short_ratio"], tolerance=timedelta(days=7))

    if cfg.data.include_funding:
        funding_path = cfg.data.raw_dir / f"binance_futures_{cfg.data.symbol}_funding.csv"
        if funding_path.exists():
            funding = load_csv(funding_path)
            if not funding.empty:
                funding = funding.rename(columns={"funding_time": "open_time"})
                funding["open_time"] = pd.to_datetime(funding["open_time"], utc=True, format="mixed")
                funding["funding_rate"] = pd.to_numeric(funding["funding_rate"], errors="coerce")
                funding = funding.set_index("open_time").sort_index()
                df = _asof_join(df, funding, ["funding_rate"], tolerance=timedelta(days=3))

    if cfg.data.include_bybit:
        bybit_funding_file = bybit_funding_path(cfg)
        if bybit_funding_file.exists():
            bybit_funding = load_csv(bybit_funding_file)
            if not bybit_funding.empty:
                bybit_funding = normalize_bybit_funding(bybit_funding).rename(columns={"funding_time": "open_time"})
                bybit_funding = bybit_funding.set_index("open_time").sort_index()
                df = _asof_join(df, bybit_funding, ["bybit_funding_rate"], tolerance=timedelta(days=3))

    return df.reset_index().drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
