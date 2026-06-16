from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import AppConfig


class AdaptiveSelectKBest(SelectKBest):
    def fit(self, X, y=None):  # noqa: N803 - sklearn API
        if isinstance(self.k, int):
            n_features = X.shape[1]
            self.k = min(int(self.k), int(n_features))
        return super().fit(X, y)


class ProbabilityAveragingClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, estimators: Sequence[BaseEstimator]) -> None:
        self.estimators = estimators

    def fit(self, X, y):  # noqa: N803 - sklearn API
        y_arr = np.asarray(y, dtype=int)
        if len(np.unique(y_arr)) < 2:
            raise RuntimeError("training slice has one class only")
        self.classes_ = np.array([0, 1])
        self.estimators_ = []
        for estimator in self.estimators:
            fitted = clone(estimator)
            fitted.fit(X, y_arr)
            self.estimators_.append(fitted)
        return self

    def predict_proba(self, X):  # noqa: N803 - sklearn API
        if not getattr(self, "estimators_", None):
            raise RuntimeError("ensemble is not fitted")
        probs = [predict_proba_up(estimator, X) for estimator in self.estimators_]
        up = np.mean(np.vstack(probs), axis=0)
        return np.column_stack([1.0 - up, up])


class RegimeSwitchingClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        base_estimator: BaseEstimator,
        regime_columns: Sequence[str] = ("return_std_34", "return_std_21", "range_mean_21", "return_std_13"),
        min_regime_samples: int = 400,
    ) -> None:
        self.base_estimator = base_estimator
        self.regime_columns = regime_columns
        self.min_regime_samples = min_regime_samples

    def _regime_values(self, X) -> np.ndarray:  # noqa: N803 - sklearn API
        if isinstance(X, pd.DataFrame):
            for column in self.regime_columns:
                if column in X.columns:
                    self.regime_column_ = column
                    return pd.to_numeric(X[column], errors="coerce").to_numpy(dtype=float)
            self.regime_column_ = None
            return np.zeros(len(X), dtype=float)
        self.regime_column_ = None
        return np.zeros(len(X), dtype=float)

    def _bucketize(self, values: np.ndarray) -> np.ndarray:
        filled = np.asarray(values, dtype=float)
        median = np.nanmedian(filled) if np.isfinite(filled).any() else 0.0
        filled = np.nan_to_num(filled, nan=median, posinf=median, neginf=median)
        low, high = self.quantiles_
        return np.where(filled <= low, 0, np.where(filled >= high, 2, 1))

    def fit(self, X, y):  # noqa: N803 - sklearn API
        y_arr = np.asarray(y, dtype=int)
        if len(np.unique(y_arr)) < 2:
            raise RuntimeError("training slice has one class only")
        self.classes_ = np.array([0, 1])
        values = self._regime_values(X)
        finite = values[np.isfinite(values)]
        if len(finite) < max(10, int(self.min_regime_samples)):
            self.quantiles_ = (float("-inf"), float("inf"))
        else:
            self.quantiles_ = tuple(np.quantile(finite, [0.33, 0.67]).astype(float))
        buckets = self._bucketize(values)
        self.global_estimator_ = clone(self.base_estimator)
        self.global_estimator_.fit(X, y_arr)
        self.regime_estimators_: dict[int, BaseEstimator] = {}
        for bucket in (0, 1, 2):
            mask = buckets == bucket
            if mask.sum() < int(self.min_regime_samples) or len(np.unique(y_arr[mask])) < 2:
                continue
            model = clone(self.base_estimator)
            model.fit(X.iloc[mask] if isinstance(X, pd.DataFrame) else X[mask], y_arr[mask])
            self.regime_estimators_[bucket] = model
        return self

    def predict_proba(self, X):  # noqa: N803 - sklearn API
        if not hasattr(self, "global_estimator_"):
            raise RuntimeError("regime classifier is not fitted")
        values = self._regime_values(X)
        buckets = self._bucketize(values)
        up = predict_proba_up(self.global_estimator_, X)
        out = np.column_stack([1.0 - up, up])
        for bucket, estimator in self.regime_estimators_.items():
            mask = buckets == bucket
            if mask.any():
                subset = X.iloc[mask] if isinstance(X, pd.DataFrame) else X[mask]
                bucket_up = predict_proba_up(estimator, subset)
                out[mask] = np.column_stack([1.0 - bucket_up, bucket_up])
        return out


class StackedProbabilityClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        estimators: Sequence[BaseEstimator],
        meta_fraction: float = 0.25,
        random_state: int = 42,
    ) -> None:
        self.estimators = estimators
        self.meta_fraction = meta_fraction
        self.random_state = random_state

    def fit(self, X, y):  # noqa: N803 - sklearn API
        y_arr = np.asarray(y, dtype=int)
        if len(np.unique(y_arr)) < 2:
            raise RuntimeError("training slice has one class only")
        self.classes_ = np.array([0, 1])
        n = len(y_arr)
        meta_n = min(max(200, int(n * float(self.meta_fraction))), max(1, n // 2))
        split = max(1, n - meta_n)
        x_fit = X.iloc[:split] if isinstance(X, pd.DataFrame) else X[:split]
        y_fit = y_arr[:split]
        x_meta = X.iloc[split:] if isinstance(X, pd.DataFrame) else X[split:]
        y_meta = y_arr[split:]
        if len(np.unique(y_fit)) < 2 or len(np.unique(y_meta)) < 2:
            raise RuntimeError("stacking split has one class only")
        self.estimators_ = []
        meta_columns = []
        for estimator in self.estimators:
            fitted = clone(estimator)
            fitted.fit(x_fit, y_fit)
            self.estimators_.append(fitted)
            meta_columns.append(predict_proba_up(fitted, x_meta))
        meta_x = np.vstack(meta_columns).T
        self.meta_model_ = LogisticRegression(C=0.5, solver="lbfgs", max_iter=500, random_state=self.random_state)
        self.meta_model_.fit(meta_x, y_meta)
        self.full_estimators_ = []
        for estimator in self.estimators:
            fitted = clone(estimator)
            fitted.fit(X, y_arr)
            self.full_estimators_.append(fitted)
        return self

    def predict_proba(self, X):  # noqa: N803 - sklearn API
        if not hasattr(self, "meta_model_"):
            raise RuntimeError("stacked classifier is not fitted")
        meta_x = np.vstack([predict_proba_up(estimator, X) for estimator in self.full_estimators_]).T
        return self.meta_model_.predict_proba(meta_x)


class MetaConfidenceClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        base_estimator: BaseEstimator,
        meta_estimator: BaseEstimator,
        meta_fraction: float = 0.25,
        blend: float = 0.70,
    ) -> None:
        self.base_estimator = base_estimator
        self.meta_estimator = meta_estimator
        self.meta_fraction = meta_fraction
        self.blend = blend

    @staticmethod
    def _meta_features(X, base_prob: np.ndarray):  # noqa: N803 - sklearn API
        confidence = np.maximum(base_prob, 1.0 - base_prob)
        margin = np.abs(base_prob - 0.5)
        direction = (base_prob >= 0.5).astype(float)
        if isinstance(X, pd.DataFrame):
            frame = X.copy()
            frame["__base_prob_up"] = base_prob
            frame["__base_confidence"] = confidence
            frame["__base_margin"] = margin
            frame["__base_direction"] = direction
            return frame
        return np.column_stack([np.asarray(X), base_prob, confidence, margin, direction])

    def fit(self, X, y):  # noqa: N803 - sklearn API
        y_arr = np.asarray(y, dtype=int)
        if len(np.unique(y_arr)) < 2:
            raise RuntimeError("training slice has one class only")
        self.classes_ = np.array([0, 1])
        n = len(y_arr)
        meta_n = min(max(300, int(n * float(self.meta_fraction))), max(1, n // 2))
        split = max(1, n - meta_n)
        x_fit = X.iloc[:split] if isinstance(X, pd.DataFrame) else X[:split]
        y_fit = y_arr[:split]
        x_meta = X.iloc[split:] if isinstance(X, pd.DataFrame) else X[split:]
        y_meta = y_arr[split:]
        if len(np.unique(y_fit)) < 2:
            raise RuntimeError("meta-confidence base split has one class only")
        base_for_meta = clone(self.base_estimator)
        base_for_meta.fit(x_fit, y_fit)
        meta_base_prob = predict_proba_up(base_for_meta, x_meta)
        meta_target = ((meta_base_prob >= 0.5).astype(int) == y_meta).astype(int)
        self.meta_prior_ = float(np.clip(meta_target.mean() if len(meta_target) else 0.5, 1e-4, 1.0 - 1e-4))
        if len(np.unique(meta_target)) >= 2:
            self.meta_model_ = clone(self.meta_estimator)
            self.meta_model_.fit(self._meta_features(x_meta, meta_base_prob), meta_target)
        else:
            self.meta_model_ = None
        self.base_model_ = clone(self.base_estimator)
        self.base_model_.fit(X, y_arr)
        return self

    def predict_proba(self, X):  # noqa: N803 - sklearn API
        if not hasattr(self, "base_model_"):
            raise RuntimeError("meta-confidence classifier is not fitted")
        base_prob = predict_proba_up(self.base_model_, X)
        if self.meta_model_ is None:
            correct_prob = np.full(len(base_prob), self.meta_prior_, dtype=float)
        else:
            correct_prob = predict_proba_up(self.meta_model_, self._meta_features(X, base_prob))
        direction_prob = np.where(base_prob >= 0.5, correct_prob, 1.0 - correct_prob)
        prob = float(self.blend) * direction_prob + (1.0 - float(self.blend)) * base_prob
        prob = np.clip(prob, 1e-6, 1.0 - 1e-6)
        return np.column_stack([1.0 - prob, prob])


class RecencyWeightedClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, base_estimator: BaseEstimator, half_life_fraction: float = 0.35) -> None:
        self.base_estimator = base_estimator
        self.half_life_fraction = half_life_fraction

    def fit(self, X, y):  # noqa: N803 - sklearn API
        y_arr = np.asarray(y, dtype=int)
        if len(np.unique(y_arr)) < 2:
            raise RuntimeError("training slice has one class only")
        self.classes_ = np.array([0, 1])
        n = len(y_arr)
        half_life = max(1.0, n * float(self.half_life_fraction))
        age = np.arange(n - 1, -1, -1, dtype=float)
        weights = np.exp(-math.log(2.0) * age / half_life)
        weights = weights / np.mean(weights)
        self.estimator_ = clone(self.base_estimator)
        try:
            self.estimator_.fit(X, y_arr, sample_weight=weights)
        except (TypeError, ValueError):
            try:
                self.estimator_.fit(X, y_arr, model__sample_weight=weights)
            except (TypeError, ValueError):
                self.estimator_.fit(X, y_arr)
        return self

    def predict_proba(self, X):  # noqa: N803 - sklearn API
        if not hasattr(self, "estimator_"):
            raise RuntimeError("recency-weighted classifier is not fitted")
        up = predict_proba_up(self.estimator_, X)
        return np.column_stack([1.0 - up, up])


class BalancedSampleWeightClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, base_estimator: BaseEstimator) -> None:
        self.base_estimator = base_estimator

    def fit(self, X, y):  # noqa: N803 - sklearn API
        y_arr = np.asarray(y, dtype=int)
        if len(np.unique(y_arr)) < 2:
            raise RuntimeError("training slice has one class only")
        self.classes_ = np.array([0, 1])
        positives = max(1, int((y_arr == 1).sum()))
        negatives = max(1, int((y_arr == 0).sum()))
        n = positives + negatives
        up_weight = n / (2.0 * positives)
        down_weight = n / (2.0 * negatives)
        weights = np.where(y_arr == 1, up_weight, down_weight)
        self.estimator_ = clone(self.base_estimator)
        try:
            self.estimator_.fit(X, y_arr, sample_weight=weights)
        except (TypeError, ValueError):
            try:
                self.estimator_.fit(X, y_arr, model__sample_weight=weights)
            except (TypeError, ValueError):
                self.estimator_.fit(X, y_arr)
        return self

    def predict_proba(self, X):  # noqa: N803 - sklearn API
        if not hasattr(self, "estimator_"):
            raise RuntimeError("balanced sample-weight classifier is not fitted")
        up = predict_proba_up(self.estimator_, X)
        return np.column_stack([1.0 - up, up])


class ThresholdOptimizedClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        base_estimator: BaseEstimator,
        validation_fraction: float = 0.25,
        thresholds: Sequence[float] = (0.38, 0.40, 0.425, 0.45, 0.475, 0.50, 0.525, 0.55, 0.575, 0.60, 0.625),
        temperature: float = 0.08,
        min_class_share: float = 0.20,
    ) -> None:
        self.base_estimator = base_estimator
        self.validation_fraction = validation_fraction
        self.thresholds = tuple(thresholds)
        self.temperature = temperature
        self.min_class_share = min_class_share

    @staticmethod
    def _accuracy(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> float:
        pred = (prob >= float(threshold)).astype(int)
        return float((pred == y_true).mean())

    def _shift_prob(self, prob: np.ndarray) -> np.ndarray:
        threshold = float(getattr(self, "selected_threshold_", 0.5))
        temperature = max(1e-4, float(self.temperature))
        shifted = 1.0 / (1.0 + np.exp(-np.clip((prob - threshold) / temperature, -35.0, 35.0)))
        return np.clip(shifted, 1e-6, 1.0 - 1e-6)

    def fit(self, X, y):  # noqa: N803 - sklearn API
        y_arr = np.asarray(y, dtype=int)
        if len(np.unique(y_arr)) < 2:
            raise RuntimeError("training slice has one class only")
        self.classes_ = np.array([0, 1])
        n = len(y_arr)
        val_n = min(max(260, int(n * float(self.validation_fraction))), max(1, n // 2))
        split = max(1, n - val_n)
        x_fit = X.iloc[:split] if isinstance(X, pd.DataFrame) else X[:split]
        y_fit = y_arr[:split]
        x_val = X.iloc[split:] if isinstance(X, pd.DataFrame) else X[split:]
        y_val = y_arr[split:]
        if len(np.unique(y_fit)) < 2 or len(np.unique(y_val)) < 2:
            x_fit = X
            y_fit = y_arr
            x_val = None
            y_val = None
        threshold_model = clone(self.base_estimator)
        threshold_model.fit(x_fit, y_fit)
        self.threshold_scores_: list[dict[str, float]] = []
        best_threshold = 0.5
        best_score = float("-inf")
        if x_val is not None and y_val is not None:
            val_prob = predict_proba_up(threshold_model, x_val)
            for threshold in self.thresholds:
                pred = val_prob >= float(threshold)
                up_share = float(pred.mean())
                if up_share < float(self.min_class_share) or up_share > 1.0 - float(self.min_class_share):
                    continue
                accuracy = self._accuracy(y_val, val_prob, float(threshold))
                confidence = np.maximum(val_prob, 1.0 - val_prob)
                high_mask = confidence >= 0.575
                high_bonus = 0.0
                if high_mask.mean() >= 0.05:
                    high_bonus = 0.015 * float(((val_prob[high_mask] >= float(threshold)).astype(int) == y_val[high_mask]).mean() - accuracy)
                score = accuracy + high_bonus
                self.threshold_scores_.append(
                    {
                        "threshold": float(threshold),
                        "accuracy": accuracy,
                        "up_share": up_share,
                        "score": score,
                    }
                )
                if score > best_score:
                    best_score = score
                    best_threshold = float(threshold)
        self.selected_threshold_ = float(best_threshold)
        self.estimator_ = clone(self.base_estimator)
        self.estimator_.fit(X, y_arr)
        return self

    def predict_proba(self, X):  # noqa: N803 - sklearn API
        if not hasattr(self, "estimator_"):
            raise RuntimeError("threshold-optimized classifier is not fitted")
        up = self._shift_prob(predict_proba_up(self.estimator_, X))
        return np.column_stack([1.0 - up, up])


class TemporalXGBoostSearchClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        validation_fraction: float = 0.25,
        config_set: str = "standard",
        random_state: int = 42,
        n_jobs: int = -1,
        use_gpu: bool = True,
    ) -> None:
        self.validation_fraction = validation_fraction
        self.config_set = config_set
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.use_gpu = use_gpu

    def _params(self, y_fit: np.ndarray) -> list[dict[str, Any]]:
        positives = max(1, int((y_fit == 1).sum()))
        negatives = max(1, int((y_fit == 0).sum()))
        scale = negatives / positives
        base: dict[str, Any] = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "n_jobs": int(self.n_jobs),
            "random_state": int(self.random_state),
        }
        if self.use_gpu:
            base["device"] = "cuda"
        candidates = [
            {"n_estimators": 180, "max_depth": 2, "learning_rate": 0.055, "subsample": 0.90, "colsample_bytree": 0.90, "min_child_weight": 12, "reg_alpha": 0.02, "reg_lambda": 1.5},
            {"n_estimators": 260, "max_depth": 2, "learning_rate": 0.038, "subsample": 0.84, "colsample_bytree": 0.84, "min_child_weight": 18, "reg_alpha": 0.04, "reg_lambda": 2.5},
            {"n_estimators": 380, "max_depth": 3, "learning_rate": 0.026, "subsample": 0.78, "colsample_bytree": 0.76, "min_child_weight": 24, "reg_alpha": 0.06, "reg_lambda": 3.0},
            {"n_estimators": 520, "max_depth": 4, "learning_rate": 0.018, "subsample": 0.72, "colsample_bytree": 0.70, "min_child_weight": 36, "reg_alpha": 0.10, "reg_lambda": 4.0},
            {"n_estimators": 240, "max_depth": 2, "learning_rate": 0.042, "subsample": 0.86, "colsample_bytree": 0.82, "min_child_weight": 18, "reg_alpha": 0.05, "reg_lambda": 2.0, "scale_pos_weight": scale},
            {"n_estimators": 420, "max_depth": 3, "learning_rate": 0.024, "subsample": 0.78, "colsample_bytree": 0.74, "min_child_weight": 28, "reg_alpha": 0.08, "reg_lambda": 3.5, "scale_pos_weight": scale},
            {"n_estimators": 280, "max_depth": 1, "learning_rate": 0.050, "subsample": 0.92, "colsample_bytree": 0.88, "min_child_weight": 10, "reg_alpha": 0.02, "reg_lambda": 1.8},
            {"n_estimators": 620, "max_depth": 3, "learning_rate": 0.014, "subsample": 0.70, "colsample_bytree": 0.68, "min_child_weight": 42, "reg_alpha": 0.14, "reg_lambda": 5.0, "scale_pos_weight": scale},
        ]
        if self.config_set == "fast":
            candidates = candidates[:4]
        elif self.config_set == "conservative":
            candidates = candidates[2:]
        return [{**base, **params} for params in candidates]

    @staticmethod
    def _score(y_true: np.ndarray, prob: np.ndarray) -> float:
        pred = (prob >= 0.5).astype(int)
        accuracy = float((pred == y_true).mean())
        brier = float(np.mean((prob - y_true) ** 2))
        confidence = np.maximum(prob, 1.0 - prob)
        coverage_mask = confidence >= 0.55
        selective_bonus = 0.0
        if coverage_mask.mean() >= 0.10:
            selective_bonus = 0.15 * float((pred[coverage_mask] == y_true[coverage_mask]).mean() - accuracy)
        return accuracy - 0.05 * brier + selective_bonus

    def fit(self, X, y):  # noqa: N803 - sklearn API
        try:
            from xgboost import XGBClassifier
        except Exception as exc:  # noqa: BLE001 - optional dependency boundary
            raise RuntimeError("xgboost is not installed") from exc
        y_arr = np.asarray(y, dtype=int)
        if len(np.unique(y_arr)) < 2:
            raise RuntimeError("training slice has one class only")
        self.classes_ = np.array([0, 1])
        n = len(y_arr)
        val_n = min(max(300, int(n * float(self.validation_fraction))), max(1, n // 2))
        split = max(1, n - val_n)
        x_fit = X.iloc[:split] if isinstance(X, pd.DataFrame) else X[:split]
        y_fit = y_arr[:split]
        x_val = X.iloc[split:] if isinstance(X, pd.DataFrame) else X[split:]
        y_val = y_arr[split:]
        if len(np.unique(y_fit)) < 2 or len(np.unique(y_val)) < 2:
            x_fit = X
            y_fit = y_arr
            x_val = None
            y_val = None
        best_score = float("-inf")
        best_params: dict[str, Any] | None = None
        self.search_scores_: list[dict[str, Any]] = []
        for idx, params in enumerate(self._params(y_fit)):
            model = XGBClassifier(**params)
            model.fit(x_fit, y_fit)
            if x_val is None or y_val is None:
                score = 0.0
            else:
                prob = predict_proba_up(model, x_val)
                score = self._score(y_val, prob)
            row = {"idx": idx, "score": score, **{k: v for k, v in params.items() if k not in {"objective", "eval_metric"}}}
            self.search_scores_.append(row)
            if score > best_score:
                best_score = score
                best_params = params
        if best_params is None:
            raise RuntimeError("xgboost temporal search had no candidates")
        self.selected_params_ = dict(best_params)
        self.estimator_ = XGBClassifier(**best_params)
        self.estimator_.fit(X, y_arr)
        return self

    def predict_proba(self, X):  # noqa: N803 - sklearn API
        if not hasattr(self, "estimator_"):
            raise RuntimeError("xgboost temporal search classifier is not fitted")
        return self.estimator_.predict_proba(X)


class RuleTriggerClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, mode: str = "confluence", strength: float = 0.95) -> None:
        self.mode = mode
        self.strength = strength

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -35.0, 35.0)))

    @staticmethod
    def _col(X, name: str) -> np.ndarray:  # noqa: N803 - sklearn API
        if isinstance(X, pd.DataFrame) and name in X.columns:
            return pd.to_numeric(X[name], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        return np.zeros(len(X), dtype=float)

    @staticmethod
    def _squash(values: np.ndarray, scale: float) -> np.ndarray:
        return np.tanh(np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0) * float(scale))

    def fit(self, X, y):  # noqa: N803 - sklearn API
        y_arr = np.asarray(y, dtype=int)
        if len(np.unique(y_arr)) < 2:
            raise RuntimeError("training slice has one class only")
        self.classes_ = np.array([0, 1])
        up_rate = float(np.clip(y_arr.mean(), 1e-4, 1.0 - 1e-4))
        self.base_logit_ = math.log(up_rate / (1.0 - up_rate))
        return self

    def _score(self, X) -> np.ndarray:  # noqa: N803 - sklearn API
        momentum_short = (
            0.70 * self._squash(self._col(X, "return_sum_3"), 900.0)
            + 0.45 * self._squash(self._col(X, "return_sum_5"), 700.0)
            + 0.30 * self._squash(self._col(X, "ema_gap_5"), 500.0)
        )
        momentum_medium = (
            0.35 * self._squash(self._col(X, "return_sum_13"), 420.0)
            + 0.25 * self._squash(self._col(X, "ema_gap_21"), 350.0)
        )
        taker_flow = (
            0.65 * self._squash(self._col(X, "taker_buy_imbalance"), 2.2)
            + 0.60 * self._squash(self._col(X, "agg_taker_quote_imbalance_mean_8"), 3.0)
            + 0.40 * self._squash(self._col(X, "agg_taker_quote_imbalance_mean_21"), 2.5)
            + 0.50 * self._squash(self._col(X, "live_taker_quote_imbalance_mean_5"), 3.0)
            + 0.34 * self._squash(self._col(X, "live_taker_quote_imbalance_mean_13"), 2.4)
            + 0.45 * self._squash(self._col(X, "taker_lsr_net_buy_share_mean_8"), 2.5)
            + 0.30 * self._squash(self._col(X, "book_imbalance_top20"), 2.0)
            + 0.18 * self._squash(self._col(X, "book_imbalance_top100"), 1.8)
        )
        liquidation_flow = (
            0.35 * self._squash(self._col(X, "liq_buy_sell_imbalance_mean_5"), 2.6)
            + 0.25 * self._squash(self._col(X, "liq_buy_sell_imbalance_mean_13"), 2.0)
            + 0.10 * self._squash(self._col(X, "liq_pressure"), 0.08)
        )
        venue_lead = (
            0.38 * self._squash(self._col(X, "futures_minus_bybit_return"), 1800.0)
            + 0.30 * self._squash(self._col(X, "futures_minus_coinbase_return"), 1800.0)
            + 0.26 * self._squash(self._col(X, "futures_minus_okx_return"), 1800.0)
            + 0.22 * self._squash(self._col(X, "bybit_minus_coinbase_return"), 1800.0)
            + 0.18 * self._squash(self._col(X, "okx_minus_coinbase_return"), 1800.0)
            - 0.12 * self._squash(self._col(X, "futures_bybit_spread_bps_z_55"), 0.7)
            - 0.10 * self._squash(self._col(X, "futures_coinbase_spread_bps_z_55"), 0.7)
            - 0.10 * self._squash(self._col(X, "futures_okx_spread_bps_z_55"), 0.7)
        )
        oi_confirmation = (
            self._squash(self._col(X, "open_interest_return_1"), 650.0)
            + self._squash(self._col(X, "bybit_open_interest_return_1"), 650.0)
            + self._squash(self._col(X, "okx_open_interest_usd_return_1"), 650.0)
        )
        confirmation = 0.18 * np.sign(momentum_short + taker_flow) * oi_confirmation
        funding_drag = -0.10 * self._squash(self._col(X, "binance_minus_bybit_funding_rate"), 10_000.0)
        mode = self.mode.strip().lower()
        if mode in {"momentum", "trigger_momentum"}:
            return momentum_short + momentum_medium + 0.55 * taker_flow + 0.20 * liquidation_flow + 0.25 * venue_lead + confirmation
        if mode in {"flow", "trigger_flow"}:
            return 0.55 * momentum_short + taker_flow + 0.35 * liquidation_flow + 0.35 * venue_lead + confirmation
        if mode in {"reversal", "mean_reversion", "trigger_reversal"}:
            overextension = (
                self._squash(self._col(X, "close_z_21"), 0.8)
                + 0.45 * self._squash(self._col(X, "futures_bybit_spread_bps_z_55"), 0.9)
                + 0.35 * self._squash(self._col(X, "futures_coinbase_spread_bps_z_55"), 0.9)
            )
            return -0.65 * momentum_short - 0.55 * overextension + 0.25 * taker_flow + 0.20 * liquidation_flow
        return (
            0.65 * momentum_short
            + 0.35 * momentum_medium
            + 0.70 * taker_flow
            + 0.25 * liquidation_flow
            + 0.45 * venue_lead
            + confirmation
            + funding_drag
        )

    def predict_proba(self, X):  # noqa: N803 - sklearn API
        if not hasattr(self, "base_logit_"):
            raise RuntimeError("rule trigger classifier is not fitted")
        raw = self.base_logit_ + float(self.strength) * self._score(X)
        up = self._sigmoid(raw)
        return np.column_stack([1.0 - up, up])


class LearnedEventTriggerClassifier(ClassifierMixin, BaseEstimator):
    _SIGNAL_GROUPS: dict[str, tuple[str, ...]] = {
        "flow": (
            "taker_buy_imbalance",
            "buy_imbalance_mean_3",
            "buy_imbalance_mean_5",
            "agg_taker_quote_imbalance",
            "agg_taker_quote_imbalance_mean_8",
            "agg_taker_quote_imbalance_mean_21",
            "agg_trade_quote_z_21",
            "agg_trade_count_z_21",
            "agg_max_trade_quote_z_21",
            "live_taker_quote_imbalance",
            "live_taker_quote_imbalance_mean_5",
            "live_taker_quote_imbalance_mean_13",
            "live_trade_quote_z_21",
            "live_trade_count_z_21",
            "live_max_trade_quote_z_21",
            "close_vs_agg_trade_vwap",
            "close_vs_live_trade_vwap",
            "taker_lsr_net_buy_share",
            "taker_lsr_net_buy_share_mean_8",
            "taker_lsr_net_buy_share_mean_21",
            "taker_lsr_total_vol_z_55",
        ),
        "liquidation": (
            "liq_buy_sell_imbalance",
            "liq_buy_sell_imbalance_mean_5",
            "liq_buy_sell_imbalance_mean_13",
            "liq_quote_z_21",
            "liq_count_z_21",
            "liq_max_quote_z_21",
            "liq_pressure",
        ),
        "orderbook": (
            "book_imbalance_top5",
            "book_imbalance_top20",
            "book_imbalance_top100",
            "book_imbalance_top5_mean_3",
            "book_imbalance_top20_mean_3",
            "book_imbalance_top100_mean_3",
            "book_imbalance_top5_change_1",
            "book_imbalance_top20_change_1",
            "book_imbalance_top100_change_1",
            "book_depth_quote_top5_z_21",
            "book_depth_quote_top20_z_21",
            "book_depth_quote_top100_z_21",
            "book_spread_bps_z_21",
        ),
        "cross_venue": (
            "futures_minus_bybit_return",
            "futures_minus_coinbase_return",
            "futures_minus_okx_return",
            "futures_minus_hyperliquid_return",
            "bybit_minus_coinbase_return",
            "okx_minus_coinbase_return",
            "okx_minus_bybit_return",
            "futures_bybit_spread_bps_z_55",
            "futures_coinbase_spread_bps_z_55",
            "futures_okx_spread_bps_z_55",
            "bybit_coinbase_spread_bps_z_55",
            "okx_coinbase_spread_bps_z_55",
            "okx_bybit_spread_bps_z_55",
        ),
        "derivatives": (
            "open_interest_return_1",
            "open_interest_z_55",
            "open_interest_value_z_55",
            "bybit_open_interest_return_1",
            "bybit_open_interest_z_55",
            "okx_open_interest_usd_return_1",
            "okx_open_interest_usd_z_55",
            "funding_rate_z_21",
            "bybit_funding_rate_z_21",
            "binance_minus_bybit_funding_rate",
            "global_lsr_account_spread_change_1",
            "top_account_lsr_account_spread_change_1",
            "top_position_lsr_account_spread_change_1",
            "okx_long_short_ratio_change_1",
            "okx_long_short_ratio_z_21",
            "premium_index_bps_z_55",
            "mark_index_basis_bps_z_55",
        ),
        "impulse": (
            "return_sum_2",
            "return_sum_3",
            "return_sum_5",
            "return_sum_8",
            "return_std_13",
            "range_mean_13",
            "close_z_21",
            "ema_gap_5",
            "ema_gap_12",
            "ema_gap_21",
            "macd_bps",
            "macd_signal_bps",
            "breakout_high_21",
            "breakout_low_21",
            "volume_z_21",
            "volume_z_55",
            "close_vs_vwap_21",
        ),
    }

    def __init__(
        self,
        mode: str = "confluence",
        event_quantile: float = 0.72,
        min_event_samples: int = 80,
        shrinkage: float = 180.0,
        min_abs_edge: float = 0.006,
        strength: float = 0.85,
        max_events: int = 24,
    ) -> None:
        self.mode = mode
        self.event_quantile = event_quantile
        self.min_event_samples = min_event_samples
        self.shrinkage = shrinkage
        self.min_abs_edge = min_abs_edge
        self.strength = strength
        self.max_events = max_events

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -35.0, 35.0)))

    @staticmethod
    def _logit(p: float) -> float:
        clipped = float(np.clip(p, 1e-4, 1.0 - 1e-4))
        return math.log(clipped / (1.0 - clipped))

    @staticmethod
    def _values(X, column: str) -> np.ndarray:  # noqa: N803 - sklearn API
        if isinstance(X, pd.DataFrame) and column in X.columns:
            return pd.to_numeric(X[column], errors="coerce").to_numpy(dtype=float)
        return np.zeros(len(X), dtype=float)

    @staticmethod
    def _robust_stats(values: np.ndarray) -> tuple[float, float]:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return 0.0, 1.0
        center = float(np.nanmedian(finite))
        mad = float(np.nanmedian(np.abs(finite - center))) * 1.4826
        if not np.isfinite(mad) or mad < 1e-9:
            q10, q90 = np.nanpercentile(finite, [10, 90])
            mad = float((q90 - q10) / 2.563)
        if not np.isfinite(mad) or mad < 1e-9:
            mad = float(np.nanstd(finite))
        if not np.isfinite(mad) or mad < 1e-9:
            mad = 1.0
        return center, mad

    @classmethod
    def _all_signal_columns(cls) -> tuple[str, ...]:
        columns: list[str] = []
        for group_columns in cls._SIGNAL_GROUPS.values():
            columns.extend(group_columns)
        return tuple(dict.fromkeys(columns))

    def _columns_for_mode(self) -> tuple[str, ...]:
        mode = self.mode.strip().lower()
        if mode in {"flow", "trade_flow", "live_flow"}:
            groups = ("flow", "orderbook")
        elif mode in {"liquidation", "liquidations", "force_order"}:
            groups = ("liquidation", "flow", "impulse")
        elif mode in {"orderbook", "book"}:
            groups = ("orderbook", "flow")
        elif mode in {"cross_venue", "venue", "basis"}:
            groups = ("cross_venue", "derivatives")
        elif mode in {"derivatives", "positioning"}:
            groups = ("derivatives", "flow")
        elif mode in {"impulse", "breakout", "momentum"}:
            groups = ("impulse", "flow", "orderbook")
        elif mode in {"reversal", "exhaustion"}:
            groups = ("impulse", "liquidation", "flow", "cross_venue")
        else:
            return self._all_signal_columns()
        columns: list[str] = []
        for group in groups:
            columns.extend(self._SIGNAL_GROUPS[group])
        return tuple(dict.fromkeys(columns))

    def fit(self, X, y):  # noqa: N803 - sklearn API
        y_arr = np.asarray(y, dtype=int)
        if len(np.unique(y_arr)) < 2:
            raise RuntimeError("training slice has one class only")
        self.classes_ = np.array([0, 1])
        self.global_rate_ = float(np.clip(y_arr.mean(), 1e-4, 1.0 - 1e-4))
        self.base_logit_ = self._logit(self.global_rate_)
        self.events_: list[dict[str, Any]] = []
        if not isinstance(X, pd.DataFrame):
            return self

        for column in self._columns_for_mode():
            if column not in X.columns:
                continue
            raw = self._values(X, column)
            center, scale = self._robust_stats(raw)
            z = np.clip(np.nan_to_num((raw - center) / scale, nan=0.0, posinf=0.0, neginf=0.0), -8.0, 8.0)
            finite_abs = np.abs(z[np.isfinite(z)])
            if finite_abs.size < max(20, int(self.min_event_samples)):
                continue
            threshold = float(np.nanquantile(finite_abs, np.clip(float(self.event_quantile), 0.50, 0.98)))
            threshold = float(np.clip(threshold, 0.30, 4.50))
            for side in (1.0, -1.0):
                signed = side * z
                mask = signed >= threshold
                n = int(mask.sum())
                if n < int(self.min_event_samples):
                    continue
                raw_rate = float(y_arr[mask].mean())
                smoothed_rate = (raw_rate * n + self.global_rate_ * float(self.shrinkage)) / (n + float(self.shrinkage))
                edge = smoothed_rate - self.global_rate_
                if abs(edge) < float(self.min_abs_edge):
                    continue
                tail = signed[mask]
                spread = float(np.nanquantile(tail, 0.90) - threshold) if len(tail) > 1 else 1.0
                if not np.isfinite(spread) or spread < 0.25:
                    spread = 0.75
                reliability = math.sqrt(n / (n + float(self.shrinkage)))
                weight = (self._logit(smoothed_rate) - self.base_logit_) * reliability
                self.events_.append(
                    {
                        "column": column,
                        "center": center,
                        "scale": scale,
                        "side": side,
                        "threshold": threshold,
                        "spread": spread,
                        "weight": float(weight),
                        "n": n,
                        "rate": smoothed_rate,
                        "edge": edge,
                    }
                )

        self.events_.sort(key=lambda row: (abs(row["weight"]), row["n"]), reverse=True)
        self.events_ = self.events_[: max(1, int(self.max_events))]
        self.event_summary_ = [
            {
                "column": row["column"],
                "side": row["side"],
                "n": row["n"],
                "rate": row["rate"],
                "edge": row["edge"],
                "weight": row["weight"],
            }
            for row in self.events_
        ]
        return self

    def predict_proba(self, X):  # noqa: N803 - sklearn API
        if not hasattr(self, "base_logit_"):
            raise RuntimeError("learned event trigger classifier is not fitted")
        raw_score = np.full(len(X), self.base_logit_, dtype=float)
        for event in getattr(self, "events_", []):
            values = self._values(X, event["column"])
            z = np.clip(
                np.nan_to_num((values - event["center"]) / event["scale"], nan=0.0, posinf=0.0, neginf=0.0),
                -8.0,
                8.0,
            )
            activation = (float(event["side"]) * z - float(event["threshold"])) / float(event["spread"])
            activation = np.clip(activation, 0.0, 1.75)
            raw_score += float(self.strength) * float(event["weight"]) * activation
        up = self._sigmoid(raw_score)
        return np.column_stack([1.0 - up, up])


class ConfidenceGatedClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        primary_estimator: BaseEstimator,
        fallback_estimators: Sequence[tuple[str, BaseEstimator]],
        validation_fraction: float = 0.25,
        thresholds: Sequence[float] = (0.50, 0.525, 0.55, 0.575, 0.60, 0.625, 0.65, 0.675, 0.70),
        min_primary_coverage: float = 0.05,
    ) -> None:
        self.primary_estimator = primary_estimator
        self.fallback_estimators = fallback_estimators
        self.validation_fraction = validation_fraction
        self.thresholds = thresholds
        self.min_primary_coverage = min_primary_coverage

    @staticmethod
    def _accuracy(y: np.ndarray, prob: np.ndarray) -> float:
        return float(((prob >= 0.5).astype(int) == y).mean())

    def fit(self, X, y):  # noqa: N803 - sklearn API
        y_arr = np.asarray(y, dtype=int)
        if len(np.unique(y_arr)) < 2:
            raise RuntimeError("training slice has one class only")
        self.classes_ = np.array([0, 1])
        n = len(X)
        val_n = min(max(240, int(n * float(self.validation_fraction))), max(1, n // 2))
        split = max(1, n - val_n)
        if split <= 0 or split >= n:
            split = max(1, int(n * 0.75))
        x_fit = X.iloc[:split] if hasattr(X, "iloc") else X[:split]
        y_fit = y_arr[:split]
        x_val = X.iloc[split:] if hasattr(X, "iloc") else X[split:]
        y_val = y_arr[split:]
        if len(np.unique(y_fit)) < 2 or len(np.unique(y_val)) < 2:
            split = max(1, int(n * 0.80))
            x_fit = X.iloc[:split] if hasattr(X, "iloc") else X[:split]
            y_fit = y_arr[:split]
            x_val = X.iloc[split:] if hasattr(X, "iloc") else X[split:]
            y_val = y_arr[split:]
        if len(np.unique(y_fit)) < 2:
            raise RuntimeError("training slice has one class only")

        primary = clone(self.primary_estimator)
        primary.fit(x_fit, y_fit)
        primary_prob = predict_proba_up(primary, x_val)
        fallback_options: list[tuple[str, np.ndarray, BaseEstimator | None]] = [("__primary__", primary_prob, None)]
        for fallback_name, fallback_estimator in self.fallback_estimators:
            try:
                fallback = clone(fallback_estimator)
                fallback.fit(x_fit, y_fit)
                fallback_options.append((fallback_name, predict_proba_up(fallback, x_val), fallback_estimator))
            except Exception:  # noqa: BLE001 - failed fallback should not kill the gated candidate
                continue

        confidence = np.maximum(primary_prob, 1.0 - primary_prob)
        best_score = float("-inf")
        best_threshold = 0.50
        best_fallback_name = "__primary__"
        best_fallback_estimator: BaseEstimator | None = None
        self.gate_scores_: list[dict[str, Any]] = []
        for fallback_name, fallback_prob, fallback_estimator in fallback_options:
            for threshold in self.thresholds:
                threshold_f = float(threshold)
                use_primary = confidence >= threshold_f
                coverage = float(use_primary.mean())
                if threshold_f > 0.50 and coverage < float(self.min_primary_coverage):
                    continue
                combined = np.where(use_primary, primary_prob, fallback_prob)
                accuracy = self._accuracy(y_val, combined)
                score = accuracy + 0.001 * coverage
                self.gate_scores_.append(
                    {
                        "fallback": fallback_name,
                        "threshold": threshold_f,
                        "accuracy": accuracy,
                        "primary_coverage": coverage,
                    }
                )
                if score > best_score:
                    best_score = score
                    best_threshold = threshold_f
                    best_fallback_name = fallback_name
                    best_fallback_estimator = fallback_estimator

        self.selected_threshold_ = float(best_threshold)
        self.selected_fallback_name_ = best_fallback_name
        self.primary_model_ = clone(self.primary_estimator)
        self.primary_model_.fit(X, y_arr)
        if best_fallback_estimator is None:
            self.fallback_model_ = None
        else:
            self.fallback_model_ = clone(best_fallback_estimator)
            self.fallback_model_.fit(X, y_arr)
        return self

    def predict_proba(self, X):  # noqa: N803 - sklearn API
        if not hasattr(self, "primary_model_"):
            raise RuntimeError("confidence gated classifier is not fitted")
        primary_prob = predict_proba_up(self.primary_model_, X)
        if self.fallback_model_ is None:
            up = primary_prob
        else:
            fallback_prob = predict_proba_up(self.fallback_model_, X)
            confidence = np.maximum(primary_prob, 1.0 - primary_prob)
            up = np.where(confidence >= float(self.selected_threshold_), primary_prob, fallback_prob)
        return np.column_stack([1.0 - up, up])


class CalendarPatternClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, bucket_minutes: int = 60, min_bucket_samples: int = 60, shrinkage: float = 160.0) -> None:
        self.bucket_minutes = bucket_minutes
        self.min_bucket_samples = min_bucket_samples
        self.shrinkage = shrinkage

    @staticmethod
    def _series(X, name: str) -> pd.Series:  # noqa: N803 - sklearn API
        if isinstance(X, pd.DataFrame) and name in X.columns:
            return pd.to_numeric(X[name], errors="coerce")
        return pd.Series(np.nan, index=np.arange(len(X)))

    def _keys(self, X) -> np.ndarray:  # noqa: N803 - sklearn API
        minute_of_week = self._series(X, "minute_of_week")
        if minute_of_week.notna().any():
            bucket = int(max(1, self.bucket_minutes))
            return np.floor(minute_of_week.fillna(-1.0).to_numpy(dtype=float) / bucket).astype(int)
        minute_of_day = self._series(X, "minute_of_day")
        if minute_of_day.notna().any():
            bucket = int(max(1, self.bucket_minutes))
            return np.floor(minute_of_day.fillna(-1.0).to_numpy(dtype=float) / bucket).astype(int)
        hour = self._series(X, "hour")
        return hour.fillna(-1.0).to_numpy(dtype=float).astype(int)

    def fit(self, X, y):  # noqa: N803 - sklearn API
        y_arr = np.asarray(y, dtype=int)
        if len(np.unique(y_arr)) < 2:
            raise RuntimeError("training slice has one class only")
        self.classes_ = np.array([0, 1])
        self.global_rate_ = float(np.clip(y_arr.mean(), 1e-4, 1.0 - 1e-4))
        keys = self._keys(X)
        self.bucket_rates_: dict[int, float] = {}
        self.bucket_counts_: dict[int, int] = {}
        for key in np.unique(keys):
            if key < 0:
                continue
            mask = keys == key
            n = int(mask.sum())
            self.bucket_counts_[int(key)] = n
            if n < int(self.min_bucket_samples):
                continue
            raw_rate = float(y_arr[mask].mean())
            weight = n / (n + float(self.shrinkage))
            self.bucket_rates_[int(key)] = float(np.clip(weight * raw_rate + (1.0 - weight) * self.global_rate_, 1e-4, 1.0 - 1e-4))
        return self

    def predict_proba(self, X):  # noqa: N803 - sklearn API
        if not hasattr(self, "global_rate_"):
            raise RuntimeError("calendar pattern classifier is not fitted")
        keys = self._keys(X)
        rates = np.array([self.bucket_rates_.get(int(key), self.global_rate_) for key in keys], dtype=float)
        return np.column_stack([1.0 - rates, rates])


class TimeWindowSelectorClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        bucket_minutes: int = 5,
        min_bucket_samples: int = 60,
        min_edge: float = 0.055,
        max_windows: int = 24,
        neutral_strength: float = 0.15,
    ) -> None:
        self.bucket_minutes = bucket_minutes
        self.min_bucket_samples = min_bucket_samples
        self.min_edge = min_edge
        self.max_windows = max_windows
        self.neutral_strength = neutral_strength

    @staticmethod
    def _series(X, name: str) -> pd.Series:  # noqa: N803 - sklearn API
        if isinstance(X, pd.DataFrame) and name in X.columns:
            return pd.to_numeric(X[name], errors="coerce")
        return pd.Series(np.nan, index=np.arange(len(X)))

    def _keys(self, X) -> np.ndarray:  # noqa: N803 - sklearn API
        minute_of_week = self._series(X, "minute_of_week")
        bucket = int(max(1, self.bucket_minutes))
        if minute_of_week.notna().any():
            return np.floor(minute_of_week.fillna(-1.0).to_numpy(dtype=float) / bucket).astype(int)
        minute_of_day = self._series(X, "minute_of_day")
        if minute_of_day.notna().any():
            weekday = self._series(X, "weekday").fillna(0.0).to_numpy(dtype=float)
            day_bucket = np.floor(minute_of_day.fillna(-1.0).to_numpy(dtype=float) / bucket)
            return (weekday * (1440 // bucket) + day_bucket).astype(int)
        return np.full(len(X), -1, dtype=int)

    def fit(self, X, y):  # noqa: N803 - sklearn API
        y_arr = np.asarray(y, dtype=int)
        if len(np.unique(y_arr)) < 2:
            raise RuntimeError("training slice has one class only")
        self.classes_ = np.array([0, 1])
        self.global_rate_ = float(np.clip(y_arr.mean(), 1e-4, 1.0 - 1e-4))
        keys = self._keys(X)
        candidates: list[tuple[float, int, float, int]] = []
        for key in np.unique(keys):
            if key < 0:
                continue
            mask = keys == key
            n = int(mask.sum())
            if n < int(self.min_bucket_samples):
                continue
            rate = float(y_arr[mask].mean())
            edge = abs(rate - self.global_rate_)
            if edge < float(self.min_edge):
                continue
            candidates.append((edge, int(key), rate, n))
        candidates.sort(reverse=True)
        self.selected_windows_: dict[int, float] = {}
        self.selected_counts_: dict[int, int] = {}
        for edge, key, rate, n in candidates[: int(self.max_windows)]:
            direction = 1.0 if rate >= self.global_rate_ else -1.0
            clipped_rate = self.global_rate_ + direction * min(0.18, max(edge, float(self.min_edge)))
            self.selected_windows_[key] = float(np.clip(clipped_rate, 1e-4, 1.0 - 1e-4))
            self.selected_counts_[key] = n
        return self

    def predict_proba(self, X):  # noqa: N803 - sklearn API
        if not hasattr(self, "global_rate_"):
            raise RuntimeError("time-window selector classifier is not fitted")
        keys = self._keys(X)
        neutral = 0.5 + float(self.neutral_strength) * (self.global_rate_ - 0.5)
        rates = np.array([self.selected_windows_.get(int(key), neutral) for key in keys], dtype=float)
        return np.column_stack([1.0 - rates, rates])


class FeatureStateFrequencyClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        columns: Sequence[str],
        quantiles: Sequence[float] = (0.33, 0.67),
        min_bucket_samples: int = 80,
        shrinkage: float = 220.0,
        neutral_strength: float = 0.20,
    ) -> None:
        self.columns = tuple(columns)
        self.quantiles = tuple(quantiles)
        self.min_bucket_samples = min_bucket_samples
        self.shrinkage = shrinkage
        self.neutral_strength = neutral_strength

    @staticmethod
    def _series(X, name: str) -> pd.Series:  # noqa: N803 - sklearn API
        if isinstance(X, pd.DataFrame) and name in X.columns:
            return pd.to_numeric(X[name], errors="coerce")
        return pd.Series(np.nan, index=np.arange(len(X)))

    def _resolve_columns(self, X) -> list[str]:  # noqa: N803 - sklearn API
        if not isinstance(X, pd.DataFrame):
            return []
        return [col for col in self.columns if col in X.columns]

    def _bucket_frame(self, X) -> pd.DataFrame:  # noqa: N803 - sklearn API
        buckets: dict[str, np.ndarray] = {}
        for col in self.active_columns_:
            values = self._series(X, col)
            thresholds = self.thresholds_.get(col)
            if thresholds is None or not np.isfinite(thresholds).all():
                buckets[col] = np.full(len(values), -1, dtype=int)
                continue
            arr = values.to_numpy(dtype=float)
            bucket = np.digitize(arr, thresholds, right=False).astype(int)
            bucket[~np.isfinite(arr)] = -1
            buckets[col] = bucket
        if not buckets:
            return pd.DataFrame({"__state": np.full(len(X), -1, dtype=int)})
        return pd.DataFrame(buckets)

    @staticmethod
    def _keys_from_buckets(buckets: pd.DataFrame) -> np.ndarray:
        if buckets.empty:
            return np.array([], dtype=object)
        return buckets.astype(str).agg("|".join, axis=1).to_numpy(dtype=object)

    def fit(self, X, y):  # noqa: N803 - sklearn API
        y_arr = np.asarray(y, dtype=int)
        if len(np.unique(y_arr)) < 2:
            raise RuntimeError("training slice has one class only")
        self.classes_ = np.array([0, 1])
        self.global_rate_ = float(np.clip(y_arr.mean(), 1e-4, 1.0 - 1e-4))
        self.active_columns_ = self._resolve_columns(X)
        if not self.active_columns_:
            self.thresholds_ = {}
            self.bucket_rates_ = {}
            return self
        self.thresholds_: dict[str, np.ndarray] = {}
        for col in self.active_columns_:
            values = self._series(X, col).replace([np.inf, -np.inf], np.nan).dropna()
            if len(values) < max(20, int(self.min_bucket_samples)):
                continue
            thresholds = np.quantile(values.to_numpy(dtype=float), self.quantiles)
            thresholds = np.unique(thresholds[np.isfinite(thresholds)])
            if len(thresholds):
                self.thresholds_[col] = thresholds
        self.active_columns_ = [col for col in self.active_columns_ if col in self.thresholds_]
        if not self.active_columns_:
            self.bucket_rates_ = {}
            return self
        keys = self._keys_from_buckets(self._bucket_frame(X))
        self.bucket_rates_: dict[str, float] = {}
        self.bucket_counts_: dict[str, int] = {}
        for key in np.unique(keys):
            mask = keys == key
            n = int(mask.sum())
            self.bucket_counts_[str(key)] = n
            if n < int(self.min_bucket_samples):
                continue
            raw_rate = float(y_arr[mask].mean())
            weight = n / (n + float(self.shrinkage))
            rate = weight * raw_rate + (1.0 - weight) * self.global_rate_
            self.bucket_rates_[str(key)] = float(np.clip(rate, 1e-4, 1.0 - 1e-4))
        return self

    def predict_proba(self, X):  # noqa: N803 - sklearn API
        if not hasattr(self, "global_rate_"):
            raise RuntimeError("feature-state classifier is not fitted")
        neutral = 0.5 + float(self.neutral_strength) * (self.global_rate_ - 0.5)
        if not getattr(self, "active_columns_", None) or not getattr(self, "bucket_rates_", None):
            up = np.full(len(X), neutral, dtype=float)
            return np.column_stack([1.0 - up, up])
        keys = self._keys_from_buckets(self._bucket_frame(X))
        rates = np.array([self.bucket_rates_.get(str(key), neutral) for key in keys], dtype=float)
        return np.column_stack([1.0 - rates, rates])


class UnivariateSignalEnsembleClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        include_prefixes: Sequence[str] | None = None,
        exclude_prefixes: Sequence[str] = ("future_", "target_"),
        quantiles: Sequence[float] = (0.10, 0.25, 0.50, 0.75, 0.90),
        min_bin_samples: int = 160,
        shrinkage: float = 320.0,
        max_signals: int = 80,
        min_abs_edge: float = 0.012,
        signal_scale: float = 0.70,
        max_logit_abs: float = 2.2,
    ) -> None:
        self.include_prefixes = None if include_prefixes is None else tuple(include_prefixes)
        self.exclude_prefixes = tuple(exclude_prefixes)
        self.quantiles = tuple(quantiles)
        self.min_bin_samples = min_bin_samples
        self.shrinkage = shrinkage
        self.max_signals = max_signals
        self.min_abs_edge = min_abs_edge
        self.signal_scale = signal_scale
        self.max_logit_abs = max_logit_abs

    @staticmethod
    def _logit(p: float) -> float:
        clipped = float(np.clip(p, 1e-4, 1.0 - 1e-4))
        return math.log(clipped / (1.0 - clipped))

    def _candidate_columns(self, X) -> list[str]:  # noqa: N803 - sklearn API
        if not isinstance(X, pd.DataFrame):
            return []
        out: list[str] = []
        for col in X.columns:
            if any(col.startswith(prefix) for prefix in self.exclude_prefixes):
                continue
            if self.include_prefixes is not None and not any(col.startswith(prefix) or prefix in col for prefix in self.include_prefixes):
                continue
            if pd.api.types.is_numeric_dtype(X[col]):
                out.append(col)
        return out

    @staticmethod
    def _bin(values: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
        bucket = np.digitize(values, thresholds, right=False).astype(int)
        bucket[~np.isfinite(values)] = -1
        return bucket

    def fit(self, X, y):  # noqa: N803 - sklearn API
        y_arr = np.asarray(y, dtype=int)
        if len(np.unique(y_arr)) < 2:
            raise RuntimeError("training slice has one class only")
        self.classes_ = np.array([0, 1])
        self.global_rate_ = float(np.clip(y_arr.mean(), 1e-4, 1.0 - 1e-4))
        self.base_logit_ = self._logit(self.global_rate_)
        self.thresholds_: dict[str, np.ndarray] = {}
        signals: list[dict[str, Any]] = []
        if not isinstance(X, pd.DataFrame):
            self.signals_ = []
            return self
        for col in self._candidate_columns(X):
            series = pd.to_numeric(X[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            finite = series.dropna().to_numpy(dtype=float)
            if len(finite) < max(int(self.min_bin_samples) * 2, 400):
                continue
            thresholds = np.unique(np.quantile(finite, self.quantiles))
            thresholds = thresholds[np.isfinite(thresholds)]
            if len(thresholds) == 0:
                continue
            values = series.to_numpy(dtype=float)
            bins = self._bin(values, thresholds)
            self.thresholds_[col] = thresholds
            for bin_id in np.unique(bins):
                if bin_id < 0:
                    continue
                mask = bins == bin_id
                n = int(mask.sum())
                if n < int(self.min_bin_samples):
                    continue
                raw_rate = float(y_arr[mask].mean())
                edge = raw_rate - self.global_rate_
                if abs(edge) < float(self.min_abs_edge):
                    continue
                weight = n / (n + float(self.shrinkage))
                shrunk_rate = weight * raw_rate + (1.0 - weight) * self.global_rate_
                signal_logit = self._logit(shrunk_rate) - self.base_logit_
                score = abs(edge) * math.sqrt(n) * weight
                signals.append(
                    {
                        "column": col,
                        "bin": int(bin_id),
                        "n": n,
                        "raw_rate": raw_rate,
                        "shrunk_rate": shrunk_rate,
                        "edge": edge,
                        "score": score,
                        "logit": float(np.clip(signal_logit, -self.max_logit_abs, self.max_logit_abs)),
                    }
                )
        signals.sort(key=lambda row: (float(row["score"]), abs(float(row["edge"]))), reverse=True)
        self.signals_ = signals[: int(self.max_signals)]
        self.used_columns_ = sorted({str(row["column"]) for row in self.signals_})
        return self

    def predict_proba(self, X):  # noqa: N803 - sklearn API
        if not hasattr(self, "base_logit_"):
            raise RuntimeError("univariate signal ensemble is not fitted")
        raw = np.full(len(X), self.base_logit_, dtype=float)
        if isinstance(X, pd.DataFrame):
            for signal in getattr(self, "signals_", []):
                col = str(signal["column"])
                thresholds = self.thresholds_.get(col)
                if thresholds is None or col not in X.columns:
                    continue
                values = pd.to_numeric(X[col], errors="coerce").to_numpy(dtype=float)
                bins = self._bin(values, thresholds)
                mask = bins == int(signal["bin"])
                if mask.any():
                    raw[mask] += float(self.signal_scale) * float(signal["logit"])
        raw = np.clip(raw, -20.0, 20.0)
        up = 1.0 / (1.0 + np.exp(-raw))
        return np.column_stack([1.0 - up, up])


class TemporalValidationSelector(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        estimators: Sequence[tuple[str, BaseEstimator]],
        validation_fraction: float = 0.25,
    ) -> None:
        self.estimators = tuple(estimators)
        self.validation_fraction = validation_fraction

    def fit(self, X, y):  # noqa: N803 - sklearn API
        y_arr = np.asarray(y, dtype=int)
        if len(np.unique(y_arr)) < 2:
            raise RuntimeError("training slice has one class only")
        self.classes_ = np.array([0, 1])
        n = len(y_arr)
        val_n = min(max(300, int(n * float(self.validation_fraction))), max(1, n // 2))
        split = max(1, n - val_n)
        x_fit = X.iloc[:split] if isinstance(X, pd.DataFrame) else X[:split]
        y_fit = y_arr[:split]
        x_val = X.iloc[split:] if isinstance(X, pd.DataFrame) else X[split:]
        y_val = y_arr[split:]
        if len(np.unique(y_fit)) < 2 or len(np.unique(y_val)) < 2:
            raise RuntimeError("temporal selector split has one class only")
        best_score = float("-inf")
        best_name = None
        best_estimator = None
        self.selector_scores_: list[dict[str, Any]] = []
        for name, estimator in self.estimators:
            try:
                model = clone(estimator)
                model.fit(x_fit, y_fit)
                prob = predict_proba_up(model, x_val)
                pred = (prob >= 0.5).astype(int)
                accuracy = float((pred == y_val).mean())
                brier = float(np.mean((prob - y_val) ** 2))
                score = accuracy - 0.05 * brier
                self.selector_scores_.append({"name": name, "accuracy": accuracy, "brier": brier, "score": score})
            except Exception as exc:  # noqa: BLE001 - selector should continue to viable alternatives
                self.selector_scores_.append({"name": name, "error": f"{type(exc).__name__}: {exc}"})
                continue
            if score > best_score:
                best_score = score
                best_name = name
                best_estimator = estimator
        if best_estimator is None:
            raise RuntimeError("temporal selector had no successful estimator")
        self.selected_name_ = str(best_name)
        self.estimator_ = clone(best_estimator)
        self.estimator_.fit(X, y_arr)
        return self

    def predict_proba(self, X):  # noqa: N803 - sklearn API
        if not hasattr(self, "estimator_"):
            raise RuntimeError("temporal selector is not fitted")
        up = predict_proba_up(self.estimator_, X)
        return np.column_stack([1.0 - up, up])


class ValidationWeightedEnsembleClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        estimators: Sequence[tuple[str, BaseEstimator]],
        validation_fraction: float = 0.25,
        min_edge: float = 0.0,
        allow_flip: bool = True,
        softmax_temperature: float = 0.030,
        prior_blend: float = 0.05,
    ) -> None:
        self.estimators = tuple(estimators)
        self.validation_fraction = validation_fraction
        self.min_edge = min_edge
        self.allow_flip = allow_flip
        self.softmax_temperature = softmax_temperature
        self.prior_blend = prior_blend

    @staticmethod
    def _slice(X, start: int | None = None, stop: int | None = None):  # noqa: N803 - sklearn API
        return X.iloc[start:stop] if isinstance(X, pd.DataFrame) else X[start:stop]

    @staticmethod
    def _weighted_recent_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if len(y_true) == 0:
            return 0.0
        age = np.arange(len(y_true) - 1, -1, -1, dtype=float)
        half_life = max(1.0, len(y_true) / 3.0)
        weights = np.exp(-math.log(2.0) * age / half_life)
        return float(np.average((y_true == y_pred).astype(float), weights=weights))

    def _candidate_score(self, y_val: np.ndarray, prob: np.ndarray, baseline: float) -> tuple[float, bool, dict[str, Any]]:
        prob = np.clip(np.asarray(prob, dtype=float), 1e-6, 1.0 - 1e-6)
        best_prob = prob
        best_flip = False
        best_payload: dict[str, Any] = {}
        for flip, candidate_prob in ((False, prob), (True, 1.0 - prob)):
            if flip and not self.allow_flip:
                continue
            pred = (candidate_prob >= 0.5).astype(int)
            accuracy = float((pred == y_val).mean())
            recent_accuracy = self._weighted_recent_accuracy(y_val, pred)
            brier = float(np.mean((candidate_prob - y_val) ** 2))
            confidence = np.maximum(candidate_prob, 1.0 - candidate_prob)
            high_mask = confidence >= 0.55
            high_accuracy = float((pred[high_mask] == y_val[high_mask]).mean()) if high_mask.mean() >= 0.10 else accuracy
            score = (
                0.55 * (accuracy - baseline)
                + 0.30 * (recent_accuracy - baseline)
                + 0.15 * (high_accuracy - baseline)
                - 0.04 * brier
            )
            if not best_payload or score > float(best_payload["score"]):
                best_prob = candidate_prob
                best_flip = flip
                best_payload = {
                    "score": score,
                    "accuracy": accuracy,
                    "recent_accuracy": recent_accuracy,
                    "high_accuracy": high_accuracy,
                    "high_coverage": float(high_mask.mean()),
                    "brier": brier,
                    "flip": flip,
                }
        return float(best_payload["score"]), best_flip, best_payload

    def fit(self, X, y):  # noqa: N803 - sklearn API
        y_arr = np.asarray(y, dtype=int)
        if len(np.unique(y_arr)) < 2:
            raise RuntimeError("training slice has one class only")
        self.classes_ = np.array([0, 1])
        self.prior_ = float(np.clip(y_arr.mean(), 1e-4, 1.0 - 1e-4))
        n = len(y_arr)
        val_n = min(max(300, int(n * float(self.validation_fraction))), max(1, n // 2))
        split = max(1, n - val_n)
        x_fit = self._slice(X, None, split)
        y_fit = y_arr[:split]
        x_val = self._slice(X, split, None)
        y_val = y_arr[split:]
        if len(np.unique(y_fit)) < 2 or len(np.unique(y_val)) < 2:
            raise RuntimeError("validation-weighted ensemble split has one class only")

        majority = max(float(y_val.mean()), 1.0 - float(y_val.mean()))
        scored: list[dict[str, Any]] = []
        for name, estimator in self.estimators:
            row: dict[str, Any] = {"name": name}
            try:
                model = clone(estimator)
                model.fit(x_fit, y_fit)
                prob = predict_proba_up(model, x_val)
                score, flip, payload = self._candidate_score(y_val, prob, majority)
                row.update(payload)
                row["score"] = score
                row["flip"] = flip
                row["estimator"] = estimator
            except Exception as exc:  # noqa: BLE001 - keep viable ensemble members
                row["error"] = f"{type(exc).__name__}: {exc}"
            scored.append(row)

        successful = [row for row in scored if "score" in row and "estimator" in row]
        if not successful:
            raise RuntimeError("validation-weighted ensemble had no successful base estimator")
        threshold = float(self.min_edge) - 0.04
        selected = [row for row in successful if float(row["score"]) >= threshold]
        if not selected:
            selected = [max(successful, key=lambda row: float(row["score"]))]
        scores = np.array([float(row["score"]) for row in selected], dtype=float)
        centered = scores - float(scores.max())
        temperature = max(1e-4, float(self.softmax_temperature))
        weights = np.exp(np.clip(centered / temperature, -60.0, 0.0))
        weights = weights / weights.sum()

        self.ensemble_scores_ = [
            {k: v for k, v in row.items() if k != "estimator"}
            for row in scored
        ]
        self.members_: list[tuple[str, BaseEstimator, float, bool]] = []
        for row, weight in zip(selected, weights):
            model = clone(row["estimator"])
            model.fit(X, y_arr)
            self.members_.append((str(row["name"]), model, float(weight), bool(row.get("flip", False))))
        return self

    def predict_proba(self, X):  # noqa: N803 - sklearn API
        if not hasattr(self, "members_"):
            raise RuntimeError("validation-weighted ensemble is not fitted")
        up = np.full(len(X), float(self.prior_), dtype=float) * float(self.prior_blend)
        weight_sum = float(self.prior_blend)
        for _name, model, weight, flip in self.members_:
            prob = predict_proba_up(model, X)
            if flip:
                prob = 1.0 - prob
            up += float(weight) * np.asarray(prob, dtype=float)
            weight_sum += float(weight)
        up = np.clip(up / max(weight_sum, 1e-12), 1e-6, 1.0 - 1e-6)
        return np.column_stack([1.0 - up, up])


class TemporalCorrectionClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        base_estimator: BaseEstimator,
        validation_fraction: float = 0.25,
        bucket_minutes: int = 60,
        use_weekday: bool = False,
        condition_on_direction: bool = True,
        min_bucket_samples: int = 70,
        min_flip_edge: float = 0.025,
        correction_strength: float = 0.85,
    ) -> None:
        self.base_estimator = base_estimator
        self.validation_fraction = validation_fraction
        self.bucket_minutes = bucket_minutes
        self.use_weekday = use_weekday
        self.condition_on_direction = condition_on_direction
        self.min_bucket_samples = min_bucket_samples
        self.min_flip_edge = min_flip_edge
        self.correction_strength = correction_strength

    @staticmethod
    def _series(X, name: str) -> pd.Series:  # noqa: N803 - sklearn API
        if isinstance(X, pd.DataFrame) and name in X.columns:
            return pd.to_numeric(X[name], errors="coerce")
        return pd.Series(np.nan, index=np.arange(len(X)))

    def _base_bucket(self, X) -> np.ndarray:  # noqa: N803 - sklearn API
        minute = self._series(X, "minute_of_day")
        if minute.notna().any():
            bucket_count = max(1, 1440 // max(1, int(self.bucket_minutes)))
            base = np.floor(minute.fillna(-1.0).to_numpy(dtype=float) / max(1, int(self.bucket_minutes))).astype(int)
            if self.use_weekday:
                weekday = self._series(X, "weekday").fillna(0.0).to_numpy(dtype=float).astype(int)
                base = weekday * bucket_count + base
            return base
        hour = self._series(X, "hour")
        if hour.notna().any():
            base = hour.fillna(-1.0).to_numpy(dtype=float).astype(int)
            if self.use_weekday:
                weekday = self._series(X, "weekday").fillna(0.0).to_numpy(dtype=float).astype(int)
                base = weekday * 24 + base
            return base
        return np.full(len(X), -1, dtype=int)

    def _keys(self, X, pred: np.ndarray) -> np.ndarray:  # noqa: N803 - sklearn API
        base = self._base_bucket(X)
        if self.condition_on_direction:
            return base * 2 + np.asarray(pred, dtype=int)
        return base

    def fit(self, X, y):  # noqa: N803 - sklearn API
        y_arr = np.asarray(y, dtype=int)
        if len(np.unique(y_arr)) < 2:
            raise RuntimeError("training slice has one class only")
        self.classes_ = np.array([0, 1])
        n = len(y_arr)
        val_n = min(max(300, int(n * float(self.validation_fraction))), max(1, n // 2))
        split = max(1, n - val_n)
        x_fit = X.iloc[:split] if isinstance(X, pd.DataFrame) else X[:split]
        y_fit = y_arr[:split]
        x_val = X.iloc[split:] if isinstance(X, pd.DataFrame) else X[split:]
        y_val = y_arr[split:]
        if len(np.unique(y_fit)) < 2 or len(np.unique(y_val)) < 2:
            raise RuntimeError("temporal correction split has one class only")

        probe = clone(self.base_estimator)
        probe.fit(x_fit, y_fit)
        val_prob = predict_proba_up(probe, x_val)
        val_pred = (val_prob >= 0.5).astype(int)
        keys = self._keys(x_val, val_pred)
        self.flip_keys_: set[int] = set()
        self.keep_keys_: set[int] = set()
        self.correction_scores_: list[dict[str, Any]] = []
        for key in np.unique(keys):
            if key < 0:
                continue
            mask = keys == key
            n_key = int(mask.sum())
            if n_key < int(self.min_bucket_samples):
                continue
            acc = float((val_pred[mask] == y_val[mask]).mean())
            flip_acc = 1.0 - acc
            row = {"key": int(key), "n": n_key, "accuracy": acc, "flip_accuracy": flip_acc}
            if flip_acc - acc >= float(self.min_flip_edge):
                self.flip_keys_.add(int(key))
                row["action"] = "flip"
            elif acc - 0.5 >= float(self.min_flip_edge):
                self.keep_keys_.add(int(key))
                row["action"] = "keep"
            else:
                row["action"] = "neutral"
            self.correction_scores_.append(row)

        self.estimator_ = clone(self.base_estimator)
        self.estimator_.fit(X, y_arr)
        return self

    def predict_proba(self, X):  # noqa: N803 - sklearn API
        if not hasattr(self, "estimator_"):
            raise RuntimeError("temporal correction classifier is not fitted")
        base_prob = predict_proba_up(self.estimator_, X)
        pred = (base_prob >= 0.5).astype(int)
        keys = self._keys(X, pred)
        flip_mask = np.array([int(key) in self.flip_keys_ for key in keys], dtype=bool)
        corrected = base_prob.copy()
        if flip_mask.any():
            flipped = 1.0 - corrected[flip_mask]
            strength = float(np.clip(self.correction_strength, 0.0, 1.0))
            corrected[flip_mask] = strength * flipped + (1.0 - strength) * corrected[flip_mask]
        corrected = np.clip(corrected, 1e-6, 1.0 - 1e-6)
        return np.column_stack([1.0 - corrected, corrected])


class TorchMLPClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        hidden_layers: Sequence[int] = (192, 96),
        epochs: int = 18,
        batch_size: int = 2048,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        dropout: float = 0.15,
        validation_fraction: float = 0.15,
        patience: int = 4,
        random_state: int = 42,
        device: str = "auto",
    ) -> None:
        self.hidden_layers = tuple(hidden_layers)
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.validation_fraction = validation_fraction
        self.patience = patience
        self.random_state = random_state
        self.device = device

    def _device(self):
        import torch

        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def _build_net(self, n_features: int):
        import torch

        layers: list[Any] = []
        prev = n_features
        for width in self.hidden_layers:
            layers.extend(
                [
                    torch.nn.Linear(prev, int(width)),
                    torch.nn.ReLU(),
                    torch.nn.LayerNorm(int(width)),
                    torch.nn.Dropout(float(self.dropout)),
                ]
            )
            prev = int(width)
        layers.append(torch.nn.Linear(prev, 1))
        return torch.nn.Sequential(*layers)

    def fit(self, X, y):  # noqa: N803 - sklearn API
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        rng = np.random.default_rng(self.random_state)
        torch.manual_seed(int(self.random_state))
        x_np = np.asarray(X, dtype=np.float32)
        y_np = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        if len(np.unique(y_np)) < 2:
            raise RuntimeError("training slice has one class only")
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = x_np.shape[1]
        n = len(x_np)
        val_n = int(n * float(self.validation_fraction))
        if val_n < 256 or val_n >= n // 2:
            val_n = max(0, min(n // 5, n - 1))
        train_end = n - val_n
        x_train, y_train = x_np[:train_end], y_np[:train_end]
        x_val, y_val = x_np[train_end:], y_np[train_end:]

        device = self._device()
        self.model_ = self._build_net(self.n_features_in_).to(device)
        positives = float(y_train.sum())
        negatives = float(len(y_train) - positives)
        pos_weight = torch.tensor([negatives / max(1.0, positives)], dtype=torch.float32, device=device)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=float(self.learning_rate),
            weight_decay=float(self.weight_decay),
        )

        order = rng.permutation(len(x_train))
        train_ds = TensorDataset(
            torch.as_tensor(x_train[order], dtype=torch.float32),
            torch.as_tensor(y_train[order], dtype=torch.float32),
        )
        loader = DataLoader(train_ds, batch_size=int(self.batch_size), shuffle=True, drop_last=False)
        x_val_t = torch.as_tensor(x_val, dtype=torch.float32, device=device) if len(x_val) else None
        y_val_t = torch.as_tensor(y_val, dtype=torch.float32, device=device) if len(y_val) else None
        best_state: dict[str, Any] | None = None
        best_loss = float("inf")
        bad_epochs = 0
        for _epoch in range(int(self.epochs)):
            self.model_.train()
            for xb, yb in loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(self.model_(xb), yb)
                loss.backward()
                optimizer.step()
            if x_val_t is None or y_val_t is None:
                continue
            self.model_.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(self.model_(x_val_t), y_val_t).detach().cpu().item())
            if val_loss + 1e-5 < best_loss:
                best_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model_.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= int(self.patience):
                    break
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.model_.eval()
        self.device_ = str(device)
        return self

    def predict_proba(self, X):  # noqa: N803 - sklearn API
        import torch

        x_np = np.asarray(X, dtype=np.float32)
        device = self._device()
        self.model_.to(device)
        self.model_.eval()
        probs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(x_np), int(self.batch_size)):
                xb = torch.as_tensor(x_np[start : start + int(self.batch_size)], dtype=torch.float32, device=device)
                prob = torch.sigmoid(self.model_(xb)).detach().cpu().numpy().reshape(-1)
                probs.append(prob)
        up = np.concatenate(probs) if probs else np.array([], dtype=float)
        return np.column_stack([1.0 - up, up])


class TorchGRUClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        sequence_length: int = 16,
        hidden_size: int = 96,
        epochs: int = 14,
        batch_size: int = 1024,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        dropout: float = 0.10,
        validation_fraction: float = 0.15,
        patience: int = 3,
        random_state: int = 42,
        device: str = "auto",
    ) -> None:
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.validation_fraction = validation_fraction
        self.patience = patience
        self.random_state = random_state
        self.device = device

    def _device(self):
        import torch

        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def _make_windows(self, x_np: np.ndarray) -> np.ndarray:
        seq = int(self.sequence_length)
        if len(x_np) == 0:
            return np.empty((0, seq, x_np.shape[1]), dtype=np.float32)
        pad = np.repeat(x_np[:1], seq - 1, axis=0)
        padded = np.vstack([pad, x_np]).astype(np.float32, copy=False)
        windows = np.lib.stride_tricks.sliding_window_view(padded, (seq, x_np.shape[1]))[:, 0]
        return np.ascontiguousarray(windows, dtype=np.float32)

    def _build_net(self, n_features: int):
        import torch

        class Net(torch.nn.Module):
            def __init__(self, input_size: int, hidden_size: int, dropout: float) -> None:
                super().__init__()
                self.gru = torch.nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
                self.norm = torch.nn.LayerNorm(hidden_size)
                self.dropout = torch.nn.Dropout(dropout)
                self.head = torch.nn.Linear(hidden_size, 1)

            def forward(self, x):
                out, _ = self.gru(x)
                last = self.dropout(self.norm(out[:, -1, :]))
                return self.head(last)

        return Net(n_features, int(self.hidden_size), float(self.dropout))

    def fit(self, X, y):  # noqa: N803 - sklearn API
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(int(self.random_state))
        x_np = np.asarray(X, dtype=np.float32)
        y_np = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        if len(np.unique(y_np)) < 2:
            raise RuntimeError("training slice has one class only")
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = x_np.shape[1]
        windows = self._make_windows(x_np)
        n = len(windows)
        val_n = int(n * float(self.validation_fraction))
        if val_n < 256 or val_n >= n // 2:
            val_n = max(0, min(n // 5, n - 1))
        train_end = n - val_n
        x_train, y_train = windows[:train_end], y_np[:train_end]
        x_val, y_val = windows[train_end:], y_np[train_end:]

        device = self._device()
        self.model_ = self._build_net(self.n_features_in_).to(device)
        positives = float(y_train.sum())
        negatives = float(len(y_train) - positives)
        pos_weight = torch.tensor([negatives / max(1.0, positives)], dtype=torch.float32, device=device)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=float(self.learning_rate), weight_decay=float(self.weight_decay))
        loader = DataLoader(
            TensorDataset(torch.as_tensor(x_train, dtype=torch.float32), torch.as_tensor(y_train, dtype=torch.float32)),
            batch_size=int(self.batch_size),
            shuffle=True,
            drop_last=False,
        )
        x_val_t = torch.as_tensor(x_val, dtype=torch.float32, device=device) if len(x_val) else None
        y_val_t = torch.as_tensor(y_val, dtype=torch.float32, device=device) if len(y_val) else None
        best_state = None
        best_loss = float("inf")
        bad_epochs = 0
        for _epoch in range(int(self.epochs)):
            self.model_.train()
            for xb, yb in loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(self.model_(xb), yb)
                loss.backward()
                optimizer.step()
            if x_val_t is None or y_val_t is None:
                continue
            self.model_.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(self.model_(x_val_t), y_val_t).detach().cpu().item())
            if val_loss + 1e-5 < best_loss:
                best_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model_.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= int(self.patience):
                    break
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.model_.eval()
        self.device_ = str(device)
        return self

    def predict_proba(self, X):  # noqa: N803 - sklearn API
        import torch

        x_np = np.asarray(X, dtype=np.float32)
        windows = self._make_windows(x_np)
        device = self._device()
        self.model_.to(device)
        self.model_.eval()
        probs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(windows), int(self.batch_size)):
                xb = torch.as_tensor(windows[start : start + int(self.batch_size)], dtype=torch.float32, device=device)
                prob = torch.sigmoid(self.model_(xb)).detach().cpu().numpy().reshape(-1)
                probs.append(prob)
        up = np.concatenate(probs) if probs else np.array([], dtype=float)
        return np.column_stack([1.0 - up, up])


class ProbabilityCalibrator:
    def __init__(self) -> None:
        self.model: LogisticRegression | None = None

    @staticmethod
    def _logit(prob: np.ndarray) -> np.ndarray:
        clipped = np.clip(prob, 1e-5, 1.0 - 1e-5)
        return np.log(clipped / (1.0 - clipped)).reshape(-1, 1)

    def fit(self, raw_prob: np.ndarray, y_true: np.ndarray) -> "ProbabilityCalibrator":
        y_true = np.asarray(y_true, dtype=int)
        if len(np.unique(y_true)) < 2:
            self.model = None
            return self
        self.model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500)
        self.model.fit(self._logit(np.asarray(raw_prob, dtype=float)), y_true)
        return self

    def predict(self, raw_prob: np.ndarray) -> np.ndarray:
        raw_prob = np.asarray(raw_prob, dtype=float)
        if self.model is None:
            return np.clip(raw_prob, 0.0, 1.0)
        return self.model.predict_proba(self._logit(raw_prob))[:, 1]


@dataclass
class FittedDirectionModel:
    timeframe: str
    horizon: int
    model_name: str
    estimator: BaseEstimator
    calibrator: ProbabilityCalibrator
    feature_columns: list[str]
    accepted: bool
    acceptance_reason: str
    metrics: dict[str, Any] = field(default_factory=dict)

    def predict_proba_up(self, frame: pd.DataFrame) -> np.ndarray:
        x = frame.reindex(columns=self.feature_columns)
        raw = predict_proba_up(self.estimator, x)
        return self.calibrator.predict(raw)


def predict_proba_up(estimator: BaseEstimator, x: pd.DataFrame) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=".*Falling back to prediction using DMatrix due to mismatched devices.*",
            category=UserWarning,
        )
        return estimator.predict_proba(x)[:, 1]


def _pipeline(model: BaseEstimator, *, scale: bool = False) -> Pipeline:
    steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def _select_pipeline(model: BaseEstimator, *, k: int = 80, scale: bool = False) -> Pipeline:
    steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("variance", VarianceThreshold(threshold=0.0)),
        ("select", AdaptiveSelectKBest(score_func=f_classif, k=k)),
    ]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def make_estimator(name: str, cfg: AppConfig) -> BaseEstimator:
    seed = cfg.models.random_seed
    n_jobs = cfg.models.n_jobs
    key = name.strip().lower().replace("-", "_")
    if key in {"hist_gradient_boosting", "hgb", "histgb"}:
        return _pipeline(
            HistGradientBoostingClassifier(
                max_iter=260,
                learning_rate=0.035,
                max_leaf_nodes=31,
                l2_regularization=0.02,
                early_stopping=True,
                random_state=seed,
            )
        )
    if key in {"hist_gradient_boosting_deep", "hgb_deep", "histgb_deep"}:
        return _pipeline(
            HistGradientBoostingClassifier(
                max_iter=520,
                learning_rate=0.020,
                max_leaf_nodes=63,
                l2_regularization=0.05,
                early_stopping=True,
                random_state=seed,
            )
        )
    if key in {"hist_gradient_boosting_fast", "hgb_fast", "histgb_fast"}:
        return _pipeline(
            HistGradientBoostingClassifier(
                max_iter=160,
                learning_rate=0.055,
                max_leaf_nodes=23,
                l2_regularization=0.03,
                early_stopping=True,
                random_state=seed,
            )
        )
    if key in {"extra_trees", "et"}:
        return _pipeline(
            ExtraTreesClassifier(
                n_estimators=360,
                max_depth=10,
                min_samples_leaf=80,
                max_features=0.65,
                bootstrap=False,
                n_jobs=n_jobs,
                random_state=seed,
                class_weight="balanced_subsample",
            )
        )
    if key in {"extra_trees_deep", "et_deep"}:
        return _pipeline(
            ExtraTreesClassifier(
                n_estimators=700,
                max_depth=None,
                min_samples_leaf=45,
                max_features="sqrt",
                bootstrap=False,
                n_jobs=n_jobs,
                random_state=seed,
                class_weight="balanced_subsample",
            )
        )
    if key in {"random_forest", "rf"}:
        return _pipeline(
            RandomForestClassifier(
                n_estimators=300,
                max_depth=9,
                min_samples_leaf=100,
                max_features=0.65,
                n_jobs=n_jobs,
                random_state=seed,
                class_weight="balanced_subsample",
            )
        )
    if key in {"logistic", "logreg"}:
        return _pipeline(
            LogisticRegression(
                C=0.2,
                solver="lbfgs",
                max_iter=1200,
                class_weight="balanced",
            ),
            scale=True,
        )
    if key in {"logistic_l1", "logreg_l1"}:
        return _pipeline(
            LogisticRegression(
                C=0.08,
                solver="liblinear",
                l1_ratio=1.0,
                max_iter=1200,
                class_weight="balanced",
                random_state=seed,
            ),
            scale=True,
        )
    if key in {"logistic_elastic", "logreg_elastic"}:
        return _pipeline(
            LogisticRegression(
                C=0.08,
                solver="saga",
                l1_ratio=0.35,
                max_iter=1600,
                class_weight="balanced",
                random_state=seed,
            ),
            scale=True,
        )
    if key in {"xgboost", "xgb"}:
        try:
            from xgboost import XGBClassifier
        except Exception as exc:  # noqa: BLE001 - optional dependency boundary
            raise RuntimeError("xgboost is not installed") from exc
        params: dict[str, Any] = {
            "n_estimators": 420,
            "max_depth": 3,
            "learning_rate": 0.028,
            "subsample": 0.80,
            "colsample_bytree": 0.75,
            "min_child_weight": 20,
            "reg_alpha": 0.05,
            "reg_lambda": 2.0,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "n_jobs": n_jobs,
            "random_state": seed,
        }
        if cfg.models.use_gpu:
            params["device"] = "cuda"
        return _pipeline(XGBClassifier(**params))
    if key in {"xgboost_deep", "xgb_deep"}:
        try:
            from xgboost import XGBClassifier
        except Exception as exc:  # noqa: BLE001 - optional dependency boundary
            raise RuntimeError("xgboost is not installed") from exc
        params = {
            "n_estimators": 620,
            "max_depth": 4,
            "learning_rate": 0.018,
            "subsample": 0.72,
            "colsample_bytree": 0.70,
            "min_child_weight": 35,
            "reg_alpha": 0.10,
            "reg_lambda": 4.0,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "n_jobs": n_jobs,
            "random_state": seed,
        }
        if cfg.models.use_gpu:
            params["device"] = "cuda"
        return _pipeline(XGBClassifier(**params))
    if key in {"xgboost_fast", "xgb_fast"}:
        try:
            from xgboost import XGBClassifier
        except Exception as exc:  # noqa: BLE001 - optional dependency boundary
            raise RuntimeError("xgboost is not installed") from exc
        params = {
            "n_estimators": 220,
            "max_depth": 2,
            "learning_rate": 0.045,
            "subsample": 0.85,
            "colsample_bytree": 0.80,
            "min_child_weight": 15,
            "reg_alpha": 0.03,
            "reg_lambda": 2.0,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "n_jobs": n_jobs,
            "random_state": seed,
        }
        if cfg.models.use_gpu:
            params["device"] = "cuda"
        return _pipeline(XGBClassifier(**params))
    if key in {"lightgbm", "lgbm"}:
        try:
            from lightgbm import LGBMClassifier
        except Exception as exc:  # noqa: BLE001 - optional dependency boundary
            raise RuntimeError("lightgbm is not installed") from exc
        return _pipeline(
            LGBMClassifier(
                n_estimators=480,
                learning_rate=0.025,
                num_leaves=31,
                max_depth=7,
                min_child_samples=120,
                subsample=0.80,
                colsample_bytree=0.75,
                reg_alpha=0.05,
                reg_lambda=2.0,
                objective="binary",
                random_state=seed,
                n_jobs=n_jobs,
                verbose=-1,
            )
        )
    if key in {"lightgbm_wide", "lgbm_wide"}:
        try:
            from lightgbm import LGBMClassifier
        except Exception as exc:  # noqa: BLE001 - optional dependency boundary
            raise RuntimeError("lightgbm is not installed") from exc
        return _pipeline(
            LGBMClassifier(
                n_estimators=760,
                learning_rate=0.016,
                num_leaves=63,
                max_depth=-1,
                min_child_samples=180,
                subsample=0.72,
                colsample_bytree=0.70,
                reg_alpha=0.12,
                reg_lambda=4.0,
                objective="binary",
                random_state=seed,
                n_jobs=n_jobs,
                verbose=-1,
            )
        )
    if key in {"lightgbm_fast", "lgbm_fast"}:
        try:
            from lightgbm import LGBMClassifier
        except Exception as exc:  # noqa: BLE001 - optional dependency boundary
            raise RuntimeError("lightgbm is not installed") from exc
        return _pipeline(
            LGBMClassifier(
                n_estimators=220,
                learning_rate=0.050,
                num_leaves=23,
                max_depth=5,
                min_child_samples=160,
                subsample=0.88,
                colsample_bytree=0.85,
                reg_alpha=0.04,
                reg_lambda=2.0,
                objective="binary",
                random_state=seed,
                n_jobs=n_jobs,
                verbose=-1,
            )
        )
    if key in {"select_logistic", "selected_logistic"}:
        return _select_pipeline(
            LogisticRegression(C=0.18, solver="lbfgs", max_iter=1200, class_weight="balanced"),
            k=70,
            scale=True,
        )
    if key in {"select_hgb", "selected_hgb"}:
        return _select_pipeline(
            HistGradientBoostingClassifier(
                max_iter=220,
                learning_rate=0.040,
                max_leaf_nodes=23,
                l2_regularization=0.03,
                early_stopping=True,
                random_state=seed,
            ),
            k=90,
        )
    if key in {"select_lightgbm", "selected_lightgbm"}:
        try:
            from lightgbm import LGBMClassifier
        except Exception as exc:  # noqa: BLE001 - optional dependency boundary
            raise RuntimeError("lightgbm is not installed") from exc
        return _select_pipeline(
            LGBMClassifier(
                n_estimators=260,
                learning_rate=0.040,
                num_leaves=23,
                max_depth=5,
                min_child_samples=140,
                subsample=0.86,
                colsample_bytree=0.86,
                reg_alpha=0.05,
                reg_lambda=2.5,
                objective="binary",
                random_state=seed,
                n_jobs=n_jobs,
                verbose=-1,
            ),
            k=90,
        )
    if key in {"select_xgboost", "selected_xgboost"}:
        try:
            from xgboost import XGBClassifier
        except Exception as exc:  # noqa: BLE001 - optional dependency boundary
            raise RuntimeError("xgboost is not installed") from exc
        params = {
            "n_estimators": 240,
            "max_depth": 2,
            "learning_rate": 0.040,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "min_child_weight": 18,
            "reg_alpha": 0.04,
            "reg_lambda": 2.5,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "n_jobs": n_jobs,
            "random_state": seed,
        }
        if cfg.models.use_gpu:
            params["device"] = "cuda"
        return _select_pipeline(XGBClassifier(**params), k=90)
    if key in {"torch_mlp", "mlp", "neural_mlp"}:
        return _pipeline(
            TorchMLPClassifier(
                epochs=24,
                batch_size=4096,
                random_state=seed,
                device="auto" if cfg.models.use_gpu else "cpu",
            ),
            scale=True,
        )
    if key in {"torch_mlp_wide", "mlp_wide", "neural_mlp_wide"}:
        return _pipeline(
            TorchMLPClassifier(
                hidden_layers=(384, 192, 96),
                epochs=34,
                batch_size=4096,
                learning_rate=0.0008,
                weight_decay=0.0002,
                dropout=0.18,
                random_state=seed,
                device="auto" if cfg.models.use_gpu else "cpu",
            ),
            scale=True,
        )
    if key in {"torch_gru", "gru", "sequence_gru"}:
        return _pipeline(
            TorchGRUClassifier(
                sequence_length=16,
                hidden_size=96,
                epochs=16,
                batch_size=1536,
                random_state=seed,
                device="auto" if cfg.models.use_gpu else "cpu",
            ),
            scale=True,
        )
    if key in {"torch_gru_wide", "gru_wide", "sequence_gru_wide"}:
        return _pipeline(
            TorchGRUClassifier(
                sequence_length=32,
                hidden_size=160,
                epochs=22,
                batch_size=1024,
                learning_rate=0.0008,
                weight_decay=0.0002,
                dropout=0.15,
                random_state=seed,
                device="auto" if cfg.models.use_gpu else "cpu",
            ),
            scale=True,
        )
    if key in {"ensemble_fast", "ensemble_linear_boost"}:
        return ProbabilityAveragingClassifier(
            (
                make_estimator("logistic", cfg),
                make_estimator("hist_gradient_boosting_fast", cfg),
                make_estimator("lightgbm_fast", cfg),
            )
        )
    if key in {"ensemble_linear_tree", "ensemble_tree"}:
        return ProbabilityAveragingClassifier(
            (
                make_estimator("logistic_l1", cfg),
                make_estimator("hist_gradient_boosting", cfg),
                make_estimator("extra_trees", cfg),
            )
        )
    if key in {"ensemble_boosting", "ensemble_boost"}:
        return ProbabilityAveragingClassifier(
            (
                make_estimator("hist_gradient_boosting_deep", cfg),
                make_estimator("lightgbm", cfg),
                make_estimator("xgboost_fast", cfg),
            )
        )
    if key in {"ensemble_diverse", "ensemble_full"}:
        return ProbabilityAveragingClassifier(
            (
                make_estimator("logistic_elastic", cfg),
                make_estimator("extra_trees_deep", cfg),
                make_estimator("lightgbm_wide", cfg),
                make_estimator("xgboost", cfg),
            )
        )
    if key in {"stacked_fast", "stack_fast"}:
        return StackedProbabilityClassifier(
            (
                make_estimator("logistic", cfg),
                make_estimator("hist_gradient_boosting_fast", cfg),
                make_estimator("lightgbm_fast", cfg),
                make_estimator("xgboost_fast", cfg),
            ),
            random_state=seed,
        )
    if key in {"stacked_boosting", "stack_boosting"}:
        return StackedProbabilityClassifier(
            (
                make_estimator("hist_gradient_boosting_deep", cfg),
                make_estimator("lightgbm_wide", cfg),
                make_estimator("xgboost_deep", cfg),
            ),
            random_state=seed,
        )
    if key in {"stacked_gpu", "stack_gpu"}:
        return StackedProbabilityClassifier(
            (
                make_estimator("xgboost_fast", cfg),
                make_estimator("xgboost_deep", cfg),
                make_estimator("torch_mlp", cfg),
            ),
            random_state=seed,
        )
    if key in {"meta_logistic", "confidence_logistic"}:
        return MetaConfidenceClassifier(
            make_estimator("logistic", cfg),
            _pipeline(
                LogisticRegression(C=0.25, solver="lbfgs", max_iter=800, class_weight="balanced", random_state=seed),
                scale=True,
            ),
            blend=0.70,
        )
    if key in {"meta_xgboost", "confidence_xgboost"}:
        return MetaConfidenceClassifier(
            make_estimator("xgboost_fast", cfg),
            _pipeline(
                LogisticRegression(C=0.20, solver="lbfgs", max_iter=800, class_weight="balanced", random_state=seed),
                scale=True,
            ),
            blend=0.75,
        )
    if key in {"meta_xgboost_deep", "confidence_xgboost_deep"}:
        return MetaConfidenceClassifier(
            make_estimator("xgboost_deep", cfg),
            make_estimator("hist_gradient_boosting_fast", cfg),
            blend=0.70,
        )
    if key in {"meta_lightgbm", "confidence_lightgbm"}:
        return MetaConfidenceClassifier(
            make_estimator("lightgbm_fast", cfg),
            _pipeline(
                LogisticRegression(C=0.25, solver="lbfgs", max_iter=800, class_weight="balanced", random_state=seed),
                scale=True,
            ),
            blend=0.75,
        )
    if key in {"meta_ensemble", "confidence_ensemble"}:
        return MetaConfidenceClassifier(
            make_estimator("ensemble_fast", cfg),
            make_estimator("hist_gradient_boosting_fast", cfg),
            blend=0.70,
        )
    if key in {"recency_logistic", "recent_logistic"}:
        return RecencyWeightedClassifier(make_estimator("logistic", cfg), half_life_fraction=0.35)
    if key in {"recency_hgb", "recent_hgb"}:
        return RecencyWeightedClassifier(make_estimator("hist_gradient_boosting_fast", cfg), half_life_fraction=0.35)
    if key in {"recency_lightgbm", "recent_lightgbm"}:
        return RecencyWeightedClassifier(make_estimator("lightgbm_fast", cfg), half_life_fraction=0.30)
    if key in {"recency_xgboost", "recent_xgboost"}:
        return RecencyWeightedClassifier(make_estimator("xgboost_fast", cfg), half_life_fraction=0.30)
    if key in {"recency_xgboost_deep", "recent_xgboost_deep"}:
        return RecencyWeightedClassifier(make_estimator("xgboost_deep", cfg), half_life_fraction=0.30)
    if key in {"recency_ensemble", "recent_ensemble"}:
        return RecencyWeightedClassifier(make_estimator("ensemble_fast", cfg), half_life_fraction=0.30)
    if key in {"balanced_xgboost_fast", "xgboost_fast_balanced", "balanced_xgb_fast"}:
        return BalancedSampleWeightClassifier(make_estimator("xgboost_fast", cfg))
    if key in {"balanced_xgboost", "xgboost_balanced", "balanced_xgb"}:
        return BalancedSampleWeightClassifier(make_estimator("xgboost", cfg))
    if key in {"balanced_xgboost_deep", "xgboost_deep_balanced", "balanced_xgb_deep"}:
        return BalancedSampleWeightClassifier(make_estimator("xgboost_deep", cfg))
    if key in {"balanced_lightgbm", "lightgbm_balanced", "balanced_lgbm"}:
        return BalancedSampleWeightClassifier(make_estimator("lightgbm_fast", cfg))
    if key in {"balanced_select_xgboost", "select_xgboost_balanced", "balanced_selected_xgboost"}:
        return BalancedSampleWeightClassifier(make_estimator("select_xgboost", cfg))
    if key in {"threshold_xgboost_fast", "threshold_tuned_xgboost_fast"}:
        return ThresholdOptimizedClassifier(make_estimator("xgboost_fast", cfg), validation_fraction=0.28)
    if key in {"threshold_xgboost_search", "threshold_tuned_xgboost_search"}:
        return ThresholdOptimizedClassifier(make_estimator("xgboost_search_fast", cfg), validation_fraction=0.28)
    if key in {"threshold_xgboost_deep", "threshold_tuned_xgboost_deep"}:
        return ThresholdOptimizedClassifier(make_estimator("xgboost_deep", cfg), validation_fraction=0.28)
    if key in {"threshold_adaptive_ensemble", "threshold_tuned_adaptive_ensemble"}:
        return ThresholdOptimizedClassifier(make_estimator("adaptive_ensemble_gpu", cfg), validation_fraction=0.30)
    if key in {"xgboost_search_gpu", "xgb_search_gpu", "temporal_xgboost_search"}:
        return _pipeline(
            TemporalXGBoostSearchClassifier(
                config_set="standard",
                random_state=seed,
                n_jobs=n_jobs,
                use_gpu=cfg.models.use_gpu,
            )
        )
    if key in {"xgboost_search_fast", "xgb_search_fast"}:
        return _pipeline(
            TemporalXGBoostSearchClassifier(
                config_set="fast",
                random_state=seed,
                n_jobs=n_jobs,
                use_gpu=cfg.models.use_gpu,
            )
        )
    if key in {"xgboost_search_conservative", "xgb_search_conservative"}:
        return _pipeline(
            TemporalXGBoostSearchClassifier(
                config_set="conservative",
                random_state=seed,
                n_jobs=n_jobs,
                use_gpu=cfg.models.use_gpu,
            )
        )
    if key in {"trigger_confluence", "rule_trigger", "live_trigger"}:
        return RuleTriggerClassifier(mode="confluence", strength=0.95)
    if key in {"trigger_flow", "flow_trigger"}:
        return RuleTriggerClassifier(mode="flow", strength=1.00)
    if key in {"trigger_momentum", "momentum_trigger"}:
        return RuleTriggerClassifier(mode="momentum", strength=0.90)
    if key in {"trigger_reversal", "reversal_trigger"}:
        return RuleTriggerClassifier(mode="reversal", strength=0.90)
    if key in {"event_trigger_confluence", "learned_event_trigger", "learned_live_trigger"}:
        return LearnedEventTriggerClassifier(mode="confluence", event_quantile=0.72, min_event_samples=80, shrinkage=180.0)
    if key in {"event_trigger_flow", "learned_flow_trigger"}:
        return LearnedEventTriggerClassifier(mode="flow", event_quantile=0.70, min_event_samples=70, shrinkage=160.0, max_events=20)
    if key in {"event_trigger_liquidation", "learned_liquidation_trigger"}:
        return LearnedEventTriggerClassifier(mode="liquidation", event_quantile=0.76, min_event_samples=55, shrinkage=150.0, max_events=18)
    if key in {"event_trigger_orderbook", "learned_orderbook_trigger"}:
        return LearnedEventTriggerClassifier(mode="orderbook", event_quantile=0.70, min_event_samples=65, shrinkage=160.0, max_events=18)
    if key in {"event_trigger_cross_venue", "learned_cross_venue_trigger"}:
        return LearnedEventTriggerClassifier(mode="cross_venue", event_quantile=0.72, min_event_samples=70, shrinkage=170.0, max_events=20)
    if key in {"event_trigger_impulse", "learned_impulse_trigger"}:
        return LearnedEventTriggerClassifier(mode="impulse", event_quantile=0.74, min_event_samples=75, shrinkage=180.0, max_events=20)
    if key in {"event_trigger_reversal", "learned_reversal_trigger"}:
        return LearnedEventTriggerClassifier(mode="reversal", event_quantile=0.78, min_event_samples=55, shrinkage=150.0, strength=0.80, max_events=18)
    if key in {"gated_event_flow", "event_gate_flow"}:
        return ConfidenceGatedClassifier(
            make_estimator("event_trigger_flow", cfg),
            (
                ("trigger_flow", make_estimator("trigger_flow", cfg)),
                ("logistic", make_estimator("logistic", cfg)),
                ("xgboost_fast", make_estimator("xgboost_fast", cfg)),
            ),
            validation_fraction=0.25,
            min_primary_coverage=0.08,
        )
    if key in {"gated_event_confluence", "event_gate_confluence"}:
        return ConfidenceGatedClassifier(
            make_estimator("event_trigger_confluence", cfg),
            (
                ("trigger_confluence", make_estimator("trigger_confluence", cfg)),
                ("hist_gradient_boosting_fast", make_estimator("hist_gradient_boosting_fast", cfg)),
                ("xgboost_fast", make_estimator("xgboost_fast", cfg)),
            ),
            validation_fraction=0.25,
            min_primary_coverage=0.08,
        )
    if key in {"gated_event_liquidation", "event_gate_liquidation"}:
        return ConfidenceGatedClassifier(
            make_estimator("event_trigger_liquidation", cfg),
            (
                ("event_trigger_flow", make_estimator("event_trigger_flow", cfg)),
                ("trigger_reversal", make_estimator("trigger_reversal", cfg)),
                ("xgboost_fast", make_estimator("xgboost_fast", cfg)),
            ),
            validation_fraction=0.30,
            min_primary_coverage=0.05,
        )
    if key in {"gated_event_ensemble", "event_gate_ensemble"}:
        return ConfidenceGatedClassifier(
            make_estimator("event_trigger_flow", cfg),
            (
                ("event_trigger_confluence", make_estimator("event_trigger_confluence", cfg)),
                ("adaptive_trigger_ensemble", make_estimator("adaptive_trigger_ensemble", cfg)),
                ("recency_xgboost", make_estimator("recency_xgboost", cfg)),
            ),
            validation_fraction=0.30,
            min_primary_coverage=0.08,
        )
    if key in {"calendar_hourly", "time_of_week_hourly", "seasonal_hourly"}:
        return CalendarPatternClassifier(bucket_minutes=60, min_bucket_samples=40, shrinkage=120.0)
    if key in {"calendar_quarter_hour", "time_of_week_15m", "seasonal_15m"}:
        return CalendarPatternClassifier(bucket_minutes=15, min_bucket_samples=25, shrinkage=90.0)
    if key in {"calendar_daypart", "time_daypart", "seasonal_daypart"}:
        return CalendarPatternClassifier(bucket_minutes=240, min_bucket_samples=90, shrinkage=180.0)
    if key in {"time_window_selector", "calendar_window_selector"}:
        return TimeWindowSelectorClassifier(bucket_minutes=5, min_bucket_samples=45, min_edge=0.055, max_windows=32)
    if key in {"time_window_selector_15m", "calendar_window_selector_15m"}:
        return TimeWindowSelectorClassifier(bucket_minutes=15, min_bucket_samples=35, min_edge=0.050, max_windows=24)
    if key in {"time_window_selector_strict", "calendar_window_selector_strict"}:
        return TimeWindowSelectorClassifier(bucket_minutes=5, min_bucket_samples=60, min_edge=0.075, max_windows=16)
    if key in {"state_flow", "feature_state_flow"}:
        return FeatureStateFrequencyClassifier(
            columns=(
                "agg_taker_quote_imbalance_mean_8",
                "taker_lsr_net_buy_share_mean_8",
                "live_taker_quote_imbalance_mean_5",
                "book_imbalance_top20",
            ),
            min_bucket_samples=90,
            shrinkage=260.0,
        )
    if key in {"state_derivatives", "feature_state_derivatives"}:
        return FeatureStateFrequencyClassifier(
            columns=(
                "open_interest_return_1",
                "binance_metrics_taker_vol_long_short_ratio_change_1",
                "binance_metrics_global_long_short_ratio_z_21",
                "binance_metrics_top_sum_long_short_ratio_z_21",
                "bybit_open_interest_return_1",
                "okx_open_interest_usd_return_1",
                "funding_rate_z_21",
            ),
            min_bucket_samples=100,
            shrinkage=280.0,
        )
    if key in {"state_cross_venue", "feature_state_cross_venue"}:
        return FeatureStateFrequencyClassifier(
            columns=(
                "futures_minus_bybit_return",
                "futures_minus_coinbase_return",
                "futures_minus_okx_return",
                "futures_bybit_spread_bps_z_55",
                "cross_asset_residual_z_55",
                "futures_minus_eth_return",
                "futures_minus_sol_return",
                "futures_minus_bnb_return",
                "futures_minus_xrp_return",
                "futures_minus_doge_return",
            ),
            min_bucket_samples=90,
            shrinkage=260.0,
        )
    if key in {"state_vol_flow", "feature_state_vol_flow"}:
        return FeatureStateFrequencyClassifier(
            columns=(
                "return_std_21",
                "range_mean_21",
                "agg_trade_quote_z_21",
                "taker_lsr_total_vol_z_55",
            ),
            min_bucket_samples=100,
            shrinkage=300.0,
        )
    if key in {"state_event", "feature_state_event"}:
        return FeatureStateFrequencyClassifier(
            columns=(
                "live_trade_quote_z_21",
                "liq_pressure",
                "book_imbalance_top20",
                "agg_max_trade_quote_z_21",
            ),
            min_bucket_samples=60,
            shrinkage=180.0,
        )
    if key in {"univariate_signal", "signal_ensemble"}:
        return UnivariateSignalEnsembleClassifier(max_signals=90, min_bin_samples=180, shrinkage=360.0)
    if key in {"univariate_signal_flow", "signal_flow"}:
        return UnivariateSignalEnsembleClassifier(
            include_prefixes=("agg_", "live_", "liq_", "book_", "taker", "volume", "buy_imbalance"),
            max_signals=70,
            min_bin_samples=140,
            shrinkage=300.0,
        )
    if key in {"univariate_signal_derivatives", "signal_derivatives"}:
        return UnivariateSignalEnsembleClassifier(
            include_prefixes=("open_interest", "funding", "binance_metrics", "bybit_open_interest", "bybit_funding", "okx_open_interest", "okx_long_short", "global_lsr", "top_account_lsr", "top_position_lsr", "taker_lsr", "mark_", "premium_", "futures_minus_index"),
            max_signals=70,
            min_bin_samples=140,
            shrinkage=300.0,
        )
    if key in {"univariate_signal_cross_venue", "signal_cross_venue"}:
        return UnivariateSignalEnsembleClassifier(
            include_prefixes=("futures_minus", "bybit_", "coinbase_", "okx_", "hyperliquid_", "spot_", "spread", "eth_", "sol_", "bnb_", "xrp_", "doge_", "ada_", "cross_asset_"),
            max_signals=70,
            min_bin_samples=140,
            shrinkage=300.0,
        )
    if key in {"univariate_signal_conservative", "signal_conservative"}:
        return UnivariateSignalEnsembleClassifier(
            max_signals=45,
            min_bin_samples=260,
            shrinkage=520.0,
            min_abs_edge=0.018,
            signal_scale=0.45,
        )
    if key in {"temporal_selector_fast", "selector_fast"}:
        return TemporalValidationSelector(
            (
                ("logistic", make_estimator("logistic", cfg)),
                ("hgb_fast", make_estimator("hist_gradient_boosting_fast", cfg)),
                ("lightgbm_fast", make_estimator("lightgbm_fast", cfg)),
                ("xgboost_fast", make_estimator("xgboost_fast", cfg)),
            )
        )
    if key in {"temporal_selector_gpu", "selector_gpu"}:
        return TemporalValidationSelector(
            (
                ("xgboost_fast", make_estimator("xgboost_fast", cfg)),
                ("xgboost", make_estimator("xgboost", cfg)),
                ("xgboost_deep", make_estimator("xgboost_deep", cfg)),
                ("lightgbm_fast", make_estimator("lightgbm_fast", cfg)),
            )
        )
    if key in {"temporal_selector_diverse", "selector_diverse"}:
        return TemporalValidationSelector(
            (
                ("logistic_elastic", make_estimator("logistic_elastic", cfg)),
                ("select_xgboost", make_estimator("select_xgboost", cfg)),
                ("recency_xgboost", make_estimator("recency_xgboost", cfg)),
                ("ensemble_fast", make_estimator("ensemble_fast", cfg)),
            )
        )
    if key in {"adaptive_ensemble_fast", "validation_weighted_fast"}:
        return ValidationWeightedEnsembleClassifier(
            (
                ("logistic", make_estimator("logistic", cfg)),
                ("hgb_fast", make_estimator("hist_gradient_boosting_fast", cfg)),
                ("lightgbm_fast", make_estimator("lightgbm_fast", cfg)),
                ("xgboost_fast", make_estimator("xgboost_fast", cfg)),
                ("trigger_confluence", make_estimator("trigger_confluence", cfg)),
                ("trigger_flow", make_estimator("trigger_flow", cfg)),
            ),
            validation_fraction=0.25,
            min_edge=0.0,
            allow_flip=True,
            softmax_temperature=0.025,
        )
    if key in {"adaptive_ensemble_gpu", "validation_weighted_gpu"}:
        return ValidationWeightedEnsembleClassifier(
            (
                ("xgboost_fast", make_estimator("xgboost_fast", cfg)),
                ("xgboost", make_estimator("xgboost", cfg)),
                ("xgboost_deep", make_estimator("xgboost_deep", cfg)),
                ("torch_mlp", make_estimator("torch_mlp", cfg)),
                ("trigger_confluence", make_estimator("trigger_confluence", cfg)),
                ("trigger_flow", make_estimator("trigger_flow", cfg)),
            ),
            validation_fraction=0.25,
            min_edge=0.0,
            allow_flip=True,
            softmax_temperature=0.025,
        )
    if key in {"adaptive_trigger_ensemble", "event_adaptive_ensemble"}:
        return ValidationWeightedEnsembleClassifier(
            (
                ("trigger_confluence", make_estimator("trigger_confluence", cfg)),
                ("trigger_flow", make_estimator("trigger_flow", cfg)),
                ("event_trigger_confluence", make_estimator("event_trigger_confluence", cfg)),
                ("event_trigger_flow", make_estimator("event_trigger_flow", cfg)),
                ("trigger_momentum", make_estimator("trigger_momentum", cfg)),
                ("trigger_reversal", make_estimator("trigger_reversal", cfg)),
                ("recency_xgboost", make_estimator("recency_xgboost", cfg)),
                ("meta_xgboost", make_estimator("meta_xgboost", cfg)),
            ),
            validation_fraction=0.30,
            min_edge=-0.005,
            allow_flip=True,
            softmax_temperature=0.020,
            prior_blend=0.03,
        )
    if key in {"time_corrected_xgboost", "temporal_corrected_xgboost"}:
        return TemporalCorrectionClassifier(make_estimator("xgboost_fast", cfg), bucket_minutes=60, min_bucket_samples=70)
    if key in {"time_corrected_xgboost_search", "temporal_corrected_xgboost_search"}:
        return TemporalCorrectionClassifier(make_estimator("xgboost_search_fast", cfg), bucket_minutes=60, min_bucket_samples=70)
    if key in {"time_corrected_xgboost_weekday", "temporal_corrected_xgboost_weekday"}:
        return TemporalCorrectionClassifier(
            make_estimator("xgboost_fast", cfg),
            bucket_minutes=120,
            use_weekday=True,
            min_bucket_samples=36,
            min_flip_edge=0.035,
        )
    if key in {"time_corrected_ensemble", "temporal_corrected_ensemble"}:
        return TemporalCorrectionClassifier(make_estimator("adaptive_ensemble_gpu", cfg), bucket_minutes=60, min_bucket_samples=70)
    if key in {"regime_logistic", "regime_linear"}:
        return RegimeSwitchingClassifier(make_estimator("logistic", cfg), min_regime_samples=250)
    if key in {"regime_hgb", "regime_hist_gradient_boosting"}:
        return RegimeSwitchingClassifier(make_estimator("hist_gradient_boosting_fast", cfg), min_regime_samples=300)
    if key in {"regime_lightgbm", "regime_lgbm"}:
        return RegimeSwitchingClassifier(make_estimator("lightgbm_fast", cfg), min_regime_samples=300)
    if key in {"regime_ensemble"}:
        return RegimeSwitchingClassifier(make_estimator("ensemble_fast", cfg), min_regime_samples=350)
    raise ValueError(f"unknown model candidate: {name}")


def fit_calibrated(
    estimator: BaseEstimator,
    x: pd.DataFrame,
    y: np.ndarray,
    *,
    calibration_fraction: float,
) -> tuple[BaseEstimator, ProbabilityCalibrator]:
    n = len(x)
    min_calibration = min(max(200, int(n * calibration_fraction)), max(1, n // 2))
    split = max(1, n - min_calibration)
    if split <= 0 or split >= n:
        split = max(1, int(n * 0.8))
    x_fit = x.iloc[:split]
    y_fit = y[:split]
    x_cal = x.iloc[split:]
    y_cal = y[split:]
    if len(np.unique(y_fit)) < 2:
        raise RuntimeError("training slice has one class only")
    model = clone(estimator)
    model.fit(x_fit, y_fit)
    raw_cal = predict_proba_up(model, x_cal) if len(x_cal) else predict_proba_up(model, x_fit)
    calibrator = ProbabilityCalibrator().fit(raw_cal, y_cal if len(x_cal) else y_fit)
    return model, calibrator


def candidate_score(metrics: dict[str, Any]) -> float:
    accuracy = metrics.get("accuracy")
    if accuracy is None or not math.isfinite(float(accuracy)):
        return float("-inf")
    high_accuracy = metrics.get("high_confidence_accuracy")
    coverage = metrics.get("high_confidence_coverage") or 0.0
    brier = metrics.get("brier") or 0.25
    score = float(accuracy) - 0.02 * float(brier)
    if high_accuracy is not None and coverage >= 0.10:
        selective_edge = max(0.0, float(high_accuracy) - float(accuracy))
        score += 0.02 * selective_edge * min(1.0, float(coverage) / 0.25)
    return score
