# Bitcoin Walk-Forward ML

A command-line research engine for **BTC candle-direction prediction**. It fetches public crypto-market data, engineers a large multi-source feature set, and searches across dozens of model families to predict whether the next candle(s) close **UP** or **DOWN** — with every candidate scored under **leakage-resistant walk-forward validation** against naive baselines.

The system is deliberately skeptical: a model is only marked `VALIDATED` when its out-of-sample, all-prediction accuracy clears a configured edge threshold *and* beats the best naive baseline by a margin. Everything else is recorded as a `WEAK` failed candidate so the search can continue from where it left off.

> **Disclaimer.** This is a research and educational tool, not financial advice and not a trading system. Predicting short-horizon crypto direction is extremely hard; weak results are the expected default. Do not trade real money based on its output.

---

## Table of contents

- [What it does](#what-it-does)
- [How it works](#how-it-works)
- [Architecture](#architecture)
- [Data sources](#data-sources)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Command reference](#command-reference)
- [Validation methodology](#validation-methodology)
- [Configuration](#configuration)
- [Project layout](#project-layout)
- [Testing](#testing)

---

## What it does

- **Multi-timeframe, multi-horizon** UP/DOWN prediction (`1m`–`4h` timeframes; horizons of 1, 2, 3 candles ahead). Each `timeframe + horizon` pair is treated as an independent **task**.
- **Walk-forward validation** with an embargo between train and test slices, probability calibration on the train tail, and out-of-sample scoring.
- **Automated model search** across 40+ classifier families and 80+ named search profiles, with a resumable experiment registry and an escalation queue.
- **Live and paper-mode prediction** — predict from the latest candle, record forward-looking predictions, then settle and score them once the target candles are known.
- **Rich evaluation** — per-task metrics, confusion matrices, precision/recall, calibration bins / ECE, high-confidence coverage, and volatility-regime breakdowns.

## How it works

```
fetch ──► data cache ──► feature engineering ──► walk-forward search ──► experiment registry
                                                          │                      │
                                                          ▼                      ▼
                                                   model artifact          leaderboard
                                                          │
                                          predict / live / paper-mode scoring
```

1. **Fetch** public market data into a local cache (`data/`).
2. **Build features** — 150+ columns across price action, technical indicators, order/trade flow, derivatives, cross-venue spreads, calendar effects, and higher-timeframe context.
3. **Search** — for each task, fit candidate models on expanding train slices, calibrate, and evaluate out-of-sample. Compare against majority / persistence / anti-persistence / time-of-week baselines.
4. **Record** every run (metrics, baselines, acceptance status, next escalation step) in a SQLite experiment registry. `search --resume` skips completed work.
5. **Predict / track** — load the best artifact, predict the latest candle, and optionally record predictions for paper-mode settlement and scoring.

## Architecture

The package lives in `btc_direction/`. Each module has a single, focused responsibility:

| Module | Responsibility |
| --- | --- |
| `cli.py` | Argparse command-line interface; dispatches all subcommands. |
| `config.py` | Config dataclasses (`AppConfig` and sub-configs); loads TOML/YAML/JSON and applies CLI overrides. |
| `data.py` | Market-data fetching & normalization (Binance spot/futures, Hyperliquid, Coinbase, Bybit, OKX, funding, open interest, agg trades, live flow, order book). |
| `features.py` | Feature engineering. `build_supervised_frame()` produces labeled feature/target frames for each task. |
| `models.py` | 40+ candidate estimators and custom classifiers (boosting, neural, ensembles, rule/event triggers, regime/temporal/state selectors). `make_estimator()` is the factory. |
| `validation.py` | Walk-forward harness: embargoed splits, candidate search per task, baseline comparison, and the acceptance gate. |
| `evaluation.py` | Metrics: accuracy, precision/recall, Brier, calibration/ECE, confusion matrices, regime metrics. |
| `engine.py` | Orchestration: `run_walk_forward()` over all tasks and `latest_predictions()` for live inference. |
| `experiments.py` | SQLite experiment registry — run tracking, config hashing, resume support, leaderboard queries. |
| `paper.py` | Paper-mode prediction recording, settlement against realized closes, and summary scoring. |
| `reporting.py` | Artifact (joblib) save/load and JSON / Markdown / CSV report generation. |
| `utils.py` | Datetime/timeframe helpers, directory setup, and GPU/CUDA hardware detection. |

## Data sources

All data is **public** and fetched over REST/websocket APIs — no keys required:

- **Binance** futures & spot klines, funding rates, open interest, long/short ratios, mark/index/premium basis, aggregate trades, live trade/liquidation flow, and order-book snapshots.
- **Cross-venue** candles and spreads from **Hyperliquid**, **Coinbase**, **Bybit**, and **OKX** (incl. Bybit/OKX open interest, funding, and long/short ratios).

Which sources are pulled and used as features is controlled per-config (`include_*` flags) and per-fetch (CLI flags).

## Installation

Requires **Python ≥ 3.11**.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

On macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Or install dependencies without the editable package:

```bash
pip install -r requirements.txt
```

Installing the package exposes the `btc-predict` console script. You can also run the CLI directly via `python btc_predict.py`. GPU training (XGBoost/PyTorch) uses CUDA when available — check with `btc-predict gpu-status`.

## Quickstart

```powershell
# 1. Fetch ~180 days of data across a few intervals
btc-predict fetch --days 180 --intervals 1m,5m,15m,1h --markets futures,spot --funding --open-interest

# 2. Run walk-forward search and save a model artifact
btc-predict train --timeframes 5m,15m,1h --horizons 1,2,3

# 3. Inspect the experiment leaderboard
btc-predict leaderboard

# 4. Predict the latest candle directions
btc-predict predict --timeframes 5m,15m,1h --horizons 1,2

# 5. Track predictions in paper mode, then settle and score them
btc-predict live --fetch --paper --timeframes 5m,15m,1h --horizons 1,2
btc-predict paper settle
btc-predict paper summary
```

For the full operational playbook — background search runs, the unattended research loop, live-flow capture, and cross-venue fetching — see **[BTC_DIRECTION_SYSTEM.md](BTC_DIRECTION_SYSTEM.md)**.

## Command reference

Run `btc-predict --help` or `btc-predict <command> --help` for full options.

| Command | Description |
| --- | --- |
| `fetch` | Download/update public market data (klines, funding, OI, agg trades, vision archives, order book, cross-venue). |
| `capture-flow` | Continuously collect live Binance futures trade/liquidation flow (+ optional order book). |
| `data-status` | Show local cache coverage (rows, date ranges, sizes). |
| `train` | Run walk-forward search, save the best artifact + reports, log to the registry. |
| `backtest` | Run walk-forward validation **without** saving an artifact. |
| `search` | Run one or many reproducible search profiles; resumable via the registry. |
| `evaluate` | Evaluate a saved artifact (`--artifact`) or a registry run (`--run-id`, default `best`). |
| `leaderboard` | Show the experiment-registry leaderboard. |
| `gpu-status` | Show Torch/CUDA training availability. |
| `research-loop` | Run unattended fetch/search cycles until stopped or an accepted model is found. |
| `predict` | Predict latest candle directions from a trained artifact. |
| `live` | Near-live compact prediction monitor with polling and optional paper recording. |
| `paper settle` | Settle paper predictions whose target candles are now known. |
| `paper summary` | Summarize settled paper-mode prediction quality. |

## Validation methodology

The project's non-completion rule is strict: **a weak model is an intermediate failed experiment, not a success.**

- **Expanding walk-forward splits** with an embargo between train and test slices. Models fit only on historical train data, calibrate on the train tail, and score on the following test slice. Reported accuracy is out-of-sample.
- **Baselines** every candidate must beat: majority class, persistence, anti-persistence, and time-of-week frequency.
- **Acceptance gate** — a task is `VALIDATED` only if all-prediction walk-forward accuracy clears `edge_accuracy_threshold` **and** exceeds the best baseline by `min_margin_over_best_baseline`. High-confidence / selective accuracy is reported for triage but **cannot** rescue a weak overall model.
- **Reported per task**: out-of-sample accuracy, per-timeframe/per-horizon metrics, confusion matrices, precision/recall for UP & DOWN, calibration bins + expected calibration error, high-confidence coverage & accuracy, and volatility-regime metrics where data allows.

A 50–59% all-prediction result is a **failed** candidate and should trigger continued search (broader features, alternate labels, more data, ensembles, regime analysis, calibration changes, paper/live scoring).

## Configuration

Configs live in `configs/` and can be TOML or YAML:

- `configs/btc_direction.toml` — default config.
- `configs/backtest.yaml` — a focused backtest profile.
- `configs/search.yaml` — broad long-search profile.

Pass one with the global `--config` flag (e.g. `btc-predict --config configs/search.yaml search --resume`). Key sections: `[data]` (symbol, market, source toggles, cache/DB paths), `[prediction]` (timeframes, horizons, label mode, thresholds), `[walk_forward]` (splits, train/test sizes, embargo, calibration), `[models]` (candidate list, GPU, seed), and `[runtime]`. Most fields can be overridden on the command line.

## Project layout

```
.
├── btc_direction/         # Python package (engine, data, features, models, validation, …)
├── configs/               # TOML/YAML configs (default, backtest, search)
├── tests/                 # unittest suite (synthetic-data unit tests)
├── btc_predict.py         # CLI entry point (python btc_predict.py …)
├── pyproject.toml         # packaging + console script (btc-predict)
├── requirements.txt       # dependency pins
├── README.md              # this file
└── BTC_DIRECTION_SYSTEM.md# detailed operational workflow guide
```

Generated artifacts — `data/`, `models/`, `reports/`, `logs/`, SQLite registries, and model files — are intentionally **git-ignored**. A fresh checkout starts clean and recreates them through the CLI.

## Testing

The suite uses synthetic data, so it runs without any network access or downloaded data:

```bash
python -m pytest
# or
python -m unittest discover -s tests
```

It covers feature construction, label variants, data normalization for each venue, model fitting across families, walk-forward splitting/embargo, metric computation, the acceptance gate, and paper-mode recording.
