# BTC Direction Prediction System

CLI entry point:

```powershell
python btc_predict.py --help
```

Installed console script after `pip install -e .`:

```powershell
btc-predict --help
```

Default config:

```text
configs/btc_direction.toml
```

Long-search examples are also provided as:

```text
configs/search.yaml
configs/backtest.yaml
```

## Workflow

Fetch or update public Binance data:

```powershell
python btc_predict.py fetch --days 900 --intervals 1m,5m --markets futures,spot --funding --open-interest
python btc_predict.py fetch --days 14 --intervals 1m,5m --markets futures --agg-trades --agg-trade-days 14 --agg-trade-max-seconds 300
python btc_predict.py fetch --days 30 --intervals 1m,5m --markets futures --derivative-ratios --ratio-periods 5m --basis-klines --hyperliquid --coinbase --bybit --okx --okx-derivatives
python btc_predict.py fetch --live-flow --live-flow-seconds 120 --order-book
python btc_predict.py capture-flow --chunk-seconds 60 --sleep-seconds 2 --order-book
```

Inspect local data coverage:

```powershell
python btc_predict.py data-status
```

Run walk-forward validation and save a model artifact:

```powershell
python btc_predict.py train --timeframes 5m,15m,1h --horizons 1,2,3
```

Run validation only, without saving a model artifact:

```powershell
python btc_predict.py backtest --timeframes 5m,15m,1h --horizons 1,2 --candidates hist_gradient_boosting,lightgbm,xgboost,logistic
```

Run broader failed-candidate search profiles:

```powershell
python btc_predict.py --config configs/search.yaml search --resume
```

The search queue includes boosting, CUDA XGBoost, Torch MLP/GRU, balanced
GPU booster variants, temporal XGBoost hyperparameter selectors,
deterministic live-trigger candidates,
learned live-feed event trigger candidates that mine train-only directional
edges from public trade imbalance, liquidation spikes, order-book pressure, and
cross-venue divergence,
confidence-gated trigger variants that choose trigger-versus-fallback behavior
on trailing validation data inside each walk-forward fold,
fold-local adaptive ensembles that weight or contrarian-flip ML/trigger members
using only trailing validation data,
feature-state frequency models that bucket flow/derivative/cross-venue/event
regimes with shrinkage,
univariate signal ensembles that mine individual feature-bin edges with
train-only shrinkage and combine the strongest signals,
calendar/time-of-week pattern candidates,
fold-local time-window selectors,
volatility-adjusted label profiles, averaged and stacked ensembles, recency-weighted variants,
fold-local feature selection,
temporal validation model selectors, cross-venue profiles, regime-specific, aggregate-trade microstructure, Binance
futures long/short and taker buy/sell ratios, mark/index/premium basis,
Hyperliquid, Coinbase, Bybit, and OKX cross-venue candles, Bybit open interest/funding,
OKX open interest and long/short ratio history,
live public Binance aggTrade/liquidation flow and order-book snapshots for trigger and paper-mode inputs,
completed higher-timeframe context,
technical, and flow/derivatives feature profiles, then
escalates into alternate labels, confidence thresholds, longer horizons, larger
walk-forward windows, and higher-compute model variants. Pure price-action-only
feature search remains available manually but is not part of the default
escalation queue. Failed candidates are written to the registry with their
metrics and next step.

Run only one new profile from the resumable queue:

```powershell
python btc_predict.py --config configs/search.yaml search --resume --max-runs 1
```

Run the non-ML live-feed trigger path directly:

```powershell
python btc_predict.py --config configs/search.yaml search --profiles live_feed_triggers --fetch --live-flow --live-flow-seconds 30 --order-book --resume
python btc_predict.py --config configs/search.yaml search --profiles live_feed_gated --fetch --live-flow --live-flow-seconds 30 --order-book --resume
```

Run a heavier ML-focused queue in the background from PowerShell:

```powershell
New-Item -ItemType Directory -Force logs | Out-Null
Start-Process -WindowStyle Hidden -FilePath python -WorkingDirectory (Get-Location) `
  -ArgumentList @(
    "btc_predict.py", "--config", "configs/search.yaml", "search", "--resume",
    "--profiles", "boosting_variants,ensembles,flow_features,neural_variants,diverse_ensemble,long_horizons,larger_walkforward",
    "--max-runs", "3", "--report-stem", "ml_heavy_resume"
  ) `
  -RedirectStandardOutput "logs/ml_search_stdout.log" `
  -RedirectStandardError "logs/ml_search_stderr.log"
```

Run the built-in unattended research loop:

```powershell
python btc_predict.py --config configs/search.yaml research-loop --fetch --funding --open-interest --agg-trades --live-flow --live-flow-seconds 30 --order-book `
  --derivative-ratios --basis-klines --hyperliquid --coinbase --bybit --okx --okx-derivatives --profiles gpu_heavy,volatility_label_gpu,volatility_label_cross_venue,bybit_gpu,okx_cross_venue_gpu,derivative_ratio_features,recent_derivatives_gpu,flow_features,regime_models,ensembles,neural_variants `
  --timeframes 5m,15m,1h --horizons 1,2 --runs-per-cycle 2 --sleep-seconds 300 --stop-on-accepted
```

Aggregate-trade refreshes are checkpointed and time-boxed in the loop so
rate-limit retries cannot prevent training/search from starting.

Inspect saved candidates:

```powershell
python btc_predict.py leaderboard
python btc_predict.py evaluate --run-id best
python btc_predict.py evaluate --artifact models/btc_direction_latest.joblib
```

Check CUDA training availability:

```powershell
python btc_predict.py gpu-status
```

Predict from the latest available candle:

```powershell
python btc_predict.py predict --timeframes 5m,15m,1h --horizons 1,2
```

Near-live monitoring:

```powershell
python btc_predict.py live --fetch --live-flow-seconds 20 --paper --poll-seconds 60
```

`predict --fetch` and `live --fetch` refresh the configured auxiliary inputs as
well as candles: spot, funding, open interest, futures ratios, mark/index/premium
basis, Hyperliquid, Coinbase, Bybit, OKX, and a short aggregate-trade update. Add
`--live-flow-seconds` to capture current public Binance trade/liquidation
websocket flow before each prediction cycle; order-book snapshots are refreshed
when `--fetch` is enabled. Those live-feed features are for live and paper
tracking unless enough captured history exists for walk-forward testing.

Settle and summarize paper-mode predictions:

```powershell
python btc_predict.py paper settle
python btc_predict.py paper summary
python btc_predict.py paper --timeframes 5m,15m,1h --horizons 1,2
```

## Experiment Registry

Every `train`, `backtest`, and `search` run is recorded in:

```text
data/experiments.sqlite3
```

The registry stores the run id, config hash, profile, reports, artifact path,
task-level metrics, baselines, acceptance status, and the next escalation step.
`search --resume` skips completed profile/config hashes and leaves failed or
interrupted work resumable. Weak candidates remain in the leaderboard so failed
experiments can be compared instead of hidden.
After an accepted run exists, `search --save-best` still saves each run-specific
artifact, but it does not replace `models/btc_direction_latest.joblib` with a
weak run.

## Validation Rules

The system uses expanding walk-forward validation with an embargo between train
and test slices. Candidate models are fitted only on historical train slices,
calibrated on the tail of each train slice, and evaluated on the following test
slice. Reported accuracy is out-of-sample walk-forward accuracy.

Each task is a `timeframe + horizon` pair. Horizon `1` predicts the close of the
next candle versus the current candle close; horizon `2` predicts two candles
ahead, and so on.

Alternate label profiles are available for research. `volatility_label_gpu`
and related profiles use the current/past median candle range to abstain from
training on tiny forward moves whose direction is likely noise, then test the
remaining UP/DOWN labels with the same walk-forward validation and baseline
gate. They are failed candidates unless they clear the same acceptance rules.

The report includes:

- out-of-sample walk-forward accuracy
- per-timeframe and per-horizon metrics
- confusion matrices
- precision and recall for UP and DOWN
- naive baselines: majority, persistence, anti-persistence, time-of-week frequency
- calibration bins and expected calibration error
- high-confidence coverage and accuracy
- volatility-regime metrics where enough data exists

Models are marked `VALIDATED` only if all-prediction walk-forward accuracy
clears the configured edge threshold and baseline-margin checks.
High-confidence and threshold-sweep results are reported for triage, but
selective accuracy alone does not turn a weak overall model into an accepted
model. Otherwise candidates are saved and displayed as `WEAK`, so live output
does not imply a proven edge.

The non-completion rule for this project is strict: a weak model is an
intermediate failed experiment. It is not a success condition and should trigger
continued search, broader features, alternate labels, more data, ensembles,
regime analysis, calibration changes, and paper/live scoring.

## Public Repository Artifact Policy

Generated market data, experiment registries, reports, logs, and model artifacts
are intentionally not committed. Public checkouts start clean and recreate those
paths through the CLI.

Run resumed searches with the registry until a model clears the acceptance gate
or the run is explicitly stopped:

```powershell
python btc_predict.py --config configs/search.yaml search --resume
python btc_predict.py leaderboard --limit 50
```
