from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .config import AppConfig
from .evaluation import evaluate_predictions, metric_dict
from .features import SupervisedFrame
from .models import FittedDirectionModel, candidate_score, fit_calibrated, make_estimator, predict_proba_up


@dataclass(slots=True)
class Split:
    fold: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int


@dataclass(slots=True)
class CandidateResult:
    name: str
    metrics: dict[str, Any] | None = None
    calibration: list[dict[str, Any]] = field(default_factory=list)
    regime: list[dict[str, Any]] = field(default_factory=list)
    fold_metrics: list[dict[str, Any]] = field(default_factory=list)
    predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    error: str | None = None


@dataclass(slots=True)
class TaskResult:
    timeframe: str
    horizon: int
    n_samples: int
    feature_count: int
    splits: list[dict[str, Any]]
    candidates: list[CandidateResult]
    baselines: dict[str, dict[str, Any]]
    best_candidate: CandidateResult | None
    accepted: bool
    acceptance_reason: str
    fitted_model: FittedDirectionModel | None


def make_walk_forward_splits(n: int, cfg: AppConfig, *, embargo: int) -> list[Split]:
    requested = max(1, cfg.walk_forward.splits)
    min_train = min(cfg.walk_forward.min_train_size, max(100, n // 3))
    if n <= min_train + embargo + 100:
        return []
    available_for_tests = n - min_train - embargo
    test_size = min(cfg.walk_forward.test_size, max(100, available_for_tests // requested))
    if test_size <= 0:
        return []
    split_count = min(requested, max(1, available_for_tests // test_size))
    first_test = n - split_count * test_size
    splits: list[Split] = []
    for fold in range(split_count):
        test_start = first_test + fold * test_size
        test_end = min(n, test_start + test_size)
        train_end = test_start - embargo
        if train_end < min_train or test_end <= test_start:
            continue
        train_start = 0
        if cfg.walk_forward.max_train_size is not None:
            train_start = max(0, train_end - cfg.walk_forward.max_train_size)
        splits.append(Split(fold=fold, train_start=train_start, train_end=train_end, test_start=test_start, test_end=test_end))
    return splits


def _slot_frequency_prob(train: pd.DataFrame, test: pd.DataFrame, target_col: str) -> np.ndarray:
    global_mean = float(train[target_col].mean())
    alpha = 12.0
    grouped = train.groupby("minute_of_week")[target_col].agg(["sum", "count"])
    grouped["prob"] = (grouped["sum"] + alpha * global_mean) / (grouped["count"] + alpha)
    mapping = grouped["prob"].to_dict()
    return test["minute_of_week"].map(mapping).fillna(global_mean).to_numpy(dtype=float)


def _baseline_predictions(train: pd.DataFrame, test: pd.DataFrame, target_col: str) -> dict[str, np.ndarray]:
    train_mean = float(train[target_col].mean())
    majority_prob = np.full(len(test), 0.55 if train_mean >= 0.5 else 0.45, dtype=float)
    persistence = np.where(pd.to_numeric(test["return_1"], errors="coerce").fillna(0.0).to_numpy() >= 0.0, 0.55, 0.45)
    anti_persistence = 1.0 - persistence
    slot = _slot_frequency_prob(train, test, target_col)
    return {
        "majority": majority_prob,
        "persistence": persistence,
        "anti_persistence": anti_persistence,
        "time_of_week_frequency": slot,
    }


def _acceptance(best: CandidateResult, baselines: dict[str, dict[str, Any]], cfg: AppConfig) -> tuple[bool, str]:
    if best.metrics is None:
        return False, "no successful candidate"
    accuracy = best.metrics.get("accuracy") or 0.0
    high_accuracy = best.metrics.get("high_confidence_accuracy")
    high_coverage = best.metrics.get("high_confidence_coverage") or 0.0
    sweep = best.metrics.get("confidence_threshold_sweep") or []
    eligible_selective = [
        row
        for row in sweep
        if row.get("accuracy") is not None
        and (row.get("coverage") or 0.0) >= cfg.prediction.min_high_confidence_coverage
        and row.get("accuracy") >= cfg.prediction.high_confidence_edge_threshold
    ]
    best_selective = max(eligible_selective, key=lambda row: (row.get("accuracy") or 0.0, row.get("coverage") or 0.0), default=None)
    best_baseline = max((m.get("accuracy") or 0.0 for m in baselines.values()), default=0.0)
    margin = accuracy - best_baseline
    clears_all = (
        accuracy >= cfg.prediction.edge_accuracy_threshold
        and margin >= cfg.prediction.min_margin_over_best_baseline
    )
    if clears_all:
        return True, (
            f"validated edge: all-prediction accuracy {accuracy:.3f}, "
            f"{margin:.3f} over best baseline"
        )
    selective_note = ""
    if (
        high_accuracy is not None
        and high_accuracy >= cfg.prediction.high_confidence_edge_threshold
        and high_coverage >= cfg.prediction.min_high_confidence_coverage
    ):
        selective_note = f", selective high-confidence accuracy {high_accuracy:.3f} at {high_coverage:.1%} coverage"
    elif best_selective is not None:
        selective_note = (
            f", best selective threshold {best_selective['threshold']:.3f} "
            f"accuracy {best_selective['accuracy']:.3f} at {best_selective['coverage']:.1%} coverage"
        )
    return False, (
        f"weak candidate: accuracy {accuracy:.3f}, best baseline {best_baseline:.3f}, "
        f"high-confidence accuracy {high_accuracy if high_accuracy is not None else 'n/a'}"
        f"{selective_note}; all-prediction gate is required"
    )


def evaluate_task(supervised: SupervisedFrame, timeframe: str, horizon: int, cfg: AppConfig) -> TaskResult:
    target_col = supervised.target_columns[horizon]
    df = supervised.frame.dropna(subset=[target_col]).copy()
    if cfg.prediction.max_rows_per_task is not None and len(df) > cfg.prediction.max_rows_per_task:
        df = df.tail(cfg.prediction.max_rows_per_task).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    feature_columns = supervised.feature_columns
    embargo = max(1, horizon * max(1, cfg.walk_forward.embargo_multiplier))
    splits = make_walk_forward_splits(len(df), cfg, embargo=embargo)
    split_records = [
        {
            "fold": s.fold,
            "train_start": str(df.loc[s.train_start, "open_time"]),
            "train_end": str(df.loc[s.train_end - 1, "open_time"]),
            "test_start": str(df.loc[s.test_start, "open_time"]),
            "test_end": str(df.loc[s.test_end - 1, "open_time"]),
            "train_n": s.train_end - s.train_start,
            "test_n": s.test_end - s.test_start,
            "embargo_rows": embargo,
        }
        for s in splits
    ]
    if not splits:
        return TaskResult(
            timeframe=timeframe,
            horizon=horizon,
            n_samples=len(df),
            feature_count=len(feature_columns),
            splits=[],
            candidates=[],
            baselines={},
            best_candidate=None,
            accepted=False,
            acceptance_reason="not enough samples for walk-forward validation",
            fitted_model=None,
        )

    baseline_probs: dict[str, list[np.ndarray]] = {
        "majority": [],
        "persistence": [],
        "anti_persistence": [],
        "time_of_week_frequency": [],
    }
    baseline_y: list[np.ndarray] = []
    baseline_meta: list[pd.DataFrame] = []
    for split in splits:
        train = df.iloc[split.train_start : split.train_end]
        test = df.iloc[split.test_start : split.test_end]
        preds = _baseline_predictions(train, test, target_col)
        for name, prob in preds.items():
            baseline_probs[name].append(prob)
        baseline_y.append(test[target_col].to_numpy(dtype=int))
        baseline_meta.append(test)

    y_all = np.concatenate(baseline_y)
    meta_all = pd.concat(baseline_meta, ignore_index=True)
    baselines: dict[str, dict[str, Any]] = {}
    for name, chunks in baseline_probs.items():
        prob = np.concatenate(chunks)
        bundle = evaluate_predictions(
            y_all,
            prob,
            high_confidence_threshold=cfg.prediction.high_confidence_threshold,
            meta=meta_all,
        )
        baselines[name] = bundle.metrics

    candidates: list[CandidateResult] = []
    for name in cfg.models.candidates:
        result = CandidateResult(name=name)
        probs: list[np.ndarray] = []
        truths: list[np.ndarray] = []
        pred_frames: list[pd.DataFrame] = []
        fold_metrics: list[dict[str, Any]] = []
        try:
            estimator = make_estimator(name, cfg)
            for split in splits:
                train = df.iloc[split.train_start : split.train_end]
                test = df.iloc[split.test_start : split.test_end]
                x_train = train[feature_columns]
                y_train = train[target_col].to_numpy(dtype=int)
                x_test = test[feature_columns]
                y_test = test[target_col].to_numpy(dtype=int)
                model, calibrator = fit_calibrated(
                    estimator,
                    x_train,
                    y_train,
                    calibration_fraction=cfg.walk_forward.calibration_fraction,
                )
                raw_prob = predict_proba_up(model, x_test)
                prob = calibrator.predict(raw_prob)
                probs.append(prob)
                truths.append(y_test)
                fold_bundle = evaluate_predictions(
                    y_test,
                    prob,
                    high_confidence_threshold=cfg.prediction.high_confidence_threshold,
                    meta=test,
                )
                row = {"fold": split.fold, **fold_bundle.metrics}
                fold_metrics.append(row)
                pred_frames.append(
                    pd.DataFrame(
                        {
                            "open_time": test["open_time"].to_numpy(),
                            "timeframe": timeframe,
                            "horizon": horizon,
                            "fold": split.fold,
                            "model": name,
                            "actual_up": y_test,
                            "prob_up": prob,
                            "predicted_up": (prob >= 0.5).astype(int),
                            "confidence": np.maximum(prob, 1.0 - prob),
                        }
                    )
                )
            y_model = np.concatenate(truths)
            p_model = np.concatenate(probs)
            pred_df = pd.concat(pred_frames, ignore_index=True)
            bundle = evaluate_predictions(
                y_model,
                p_model,
                high_confidence_threshold=cfg.prediction.high_confidence_threshold,
                meta=meta_all,
            )
            result.metrics = bundle.metrics
            result.calibration = bundle.calibration
            result.regime = bundle.regime
            result.fold_metrics = fold_metrics
            result.predictions = pred_df
        except Exception as exc:  # noqa: BLE001 - keep model search moving
            result.error = f"{type(exc).__name__}: {exc}"
        candidates.append(result)

    successful = [c for c in candidates if c.metrics is not None]
    best = max(successful, key=lambda c: candidate_score(c.metrics or {}), default=None)
    fitted_model = None
    if best is not None:
        accepted, reason = _acceptance(best, baselines, cfg)
        estimator = make_estimator(best.name, cfg)
        x_all = df[feature_columns]
        y_all_fit = df[target_col].to_numpy(dtype=int)
        model, calibrator = fit_calibrated(
            estimator,
            x_all,
            y_all_fit,
            calibration_fraction=cfg.walk_forward.calibration_fraction,
        )
        fitted_model = FittedDirectionModel(
            timeframe=timeframe,
            horizon=horizon,
            model_name=best.name,
            estimator=model,
            calibrator=calibrator,
            feature_columns=feature_columns,
            accepted=accepted,
            acceptance_reason=reason,
            metrics=best.metrics or {},
        )
    else:
        accepted = False
        reason = "all candidates failed"

    return TaskResult(
        timeframe=timeframe,
        horizon=horizon,
        n_samples=len(df),
        feature_count=len(feature_columns),
        splits=split_records,
        candidates=candidates,
        baselines=baselines,
        best_candidate=best,
        accepted=accepted,
        acceptance_reason=reason,
        fitted_model=fitted_model,
    )
