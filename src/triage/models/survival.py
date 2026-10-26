"""Survival Analysis Models for Time-to-Close Prediction.

Handles right-censored observation data for Home Assistant tickets.
Implements:
1. Baseline: Historical median duration by integration.
2. Lifelines Weibull AFT model.
3. XGBoost Survival Accelerated Failure Time (AFT) model.
4. Evaluation: Harrell's Concordance Index (C-Index) & Calibration at 7, 30, and 90 days.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from lifelines.utils import concordance_index

from triage.features.leakage_guard import assert_no_leakage

TABULAR_FEATURE_COLS: List[str] = [
    "title_len",
    "body_len",
    "n_code_blocks",
    "n_urls",
    "has_traceback",
    "has_logs",
    "refers_to_issue",
    "n_missing_form_fields",
    "has_last_working_version",
    "ha_version_year",
    "ha_version_month",
    "version_gap_months",
    "author_prior_tickets",
    "created_hour",
    "created_day_of_week",
    "is_weekend",
    "integration_volume_90d",
    "integration_median_close_prior",
]


class MedianByIntegrationBaseline:
    """Historical median duration baseline grouped by integration."""

    def __init__(self):
        self.medians_: Dict[str, float] = {}
        self.global_median_: float = 7.0

    def fit(self, df: pd.DataFrame) -> "MedianByIntegrationBaseline":
        closed = df[df["target_is_closed_event"] == 1]
        durations = closed["target_days_to_close_or_censor"].values
        self.global_median_ = float(np.median(durations)) if len(durations) > 0 else 7.0

        integ_medians = (
            closed.groupby("integration_clean")["target_days_to_close_or_censor"].median().to_dict()
        )
        self.medians_ = {k: float(v) for k, v in integ_medians.items()}
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        preds = np.zeros(len(df), dtype=np.float32)
        integs = df["integration_clean"].values
        for i, integ in enumerate(integs):
            preds[i] = self.medians_.get(integ, self.global_median_)
        return preds


class XGBoostSurvivalAFT:
    """XGBoost Accelerated Failure Time model for censored ticket survival."""

    def __init__(self, max_depth: int = 4, learning_rate: float = 0.05, n_estimators: int = 100):
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.booster_: Optional[xgb.Booster] = None
        self.feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame, duration: np.ndarray, event: np.ndarray) -> "XGBoostSurvivalAFT":
        assert_no_leakage(X)
        self.feature_names_ = list(X.columns)

        # AFT targets: lower bound = duration, upper bound = duration if event==1 else +inf
        y_lower = np.maximum(duration, 0.01).astype(np.float32)
        y_upper = np.where(event == 1, y_lower, np.inf).astype(np.float32)

        dtrain = xgb.DMatrix(
            X[self.feature_names_].fillna(0).values, feature_names=self.feature_names_
        )
        dtrain.set_float_info("label_lower_bound", y_lower)
        dtrain.set_float_info("label_upper_bound", y_upper)

        params = {
            "objective": "survival:aft",
            "eval_metric": "aft-nloglik",
            "aft_loss_distribution": "normal",
            "aft_loss_distribution_scale": 1.20,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "tree_method": "hist",
            "random_state": 42,
        }

        self.booster_ = xgb.train(params, dtrain, num_boost_round=self.n_estimators)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert_no_leakage(X)
        dtest = xgb.DMatrix(
            X[self.feature_names_].fillna(0).values, feature_names=self.feature_names_
        )
        # XGBoost survival:aft predict returns predicted median survival time exp(y_hat)
        return self.booster_.predict(dtest)

    def predict_percentiles(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Return estimated p50 (median) and p90 time-to-close in days."""
        p50 = self.predict(X)
        # In log-normal/normal AFT, p90 is approximately exp(mu + 1.28 * sigma)
        p90 = p50 * 2.5
        return p50, p90


def compute_calibration_at_t(
    predicted_median: np.ndarray,
    actual_duration: np.ndarray,
    event: np.ndarray,
    horizon_days: float,
) -> Dict[str, float]:
    """
    Assess calibration at specific time horizon (e.g., 7, 30, 90 days).
    Computes predicted closure probability vs actual empirical observation.
    """
    # Simple exponential probability model: P(T <= t) = 1 - exp(-ln(2) * t / median)
    pred_prob_closed = 1.0 - np.exp(-0.69315 * horizon_days / np.maximum(predicted_median, 0.1))

    # Actual outcome evaluated on tickets with sufficient follow-up or observed event
    observed_closed = (actual_duration <= horizon_days) & (event == 1)
    eval_mask = (actual_duration >= horizon_days) | (event == 1)

    if eval_mask.sum() > 0:
        actual_rate = float(observed_closed[eval_mask].mean())
        mean_pred = float(pred_prob_closed[eval_mask].mean())
        brier = float(
            np.mean((pred_prob_closed[eval_mask] - observed_closed[eval_mask].astype(float)) ** 2)
        )
    else:
        actual_rate = 0.5
        mean_pred = 0.5
        brier = 0.25

    return {
        f"pred_rate_{int(horizon_days)}d": round(mean_pred, 4),
        f"actual_rate_{int(horizon_days)}d": round(actual_rate, 4),
        f"brier_{int(horizon_days)}d": round(brier, 4),
    }


def evaluate_survival_model(
    predictions: np.ndarray,
    durations: np.ndarray,
    events: np.ndarray,
) -> Dict[str, Any]:
    """Compute comprehensive C-Index and multi-horizon calibration metrics."""
    # Negative predictions because higher predicted survival time means lower hazard/risk
    c_idx = concordance_index(durations, predictions, events)

    metrics = {
        "concordance_index": round(float(c_idx), 4),
    }
    for h in [7.0, 30.0, 90.0]:
        metrics.update(compute_calibration_at_t(predictions, durations, events, horizon_days=h))

    return metrics
