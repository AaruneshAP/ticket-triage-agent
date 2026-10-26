"""Fixed-Horizon Resolution Classification Models (14-day and 30-day) [DIAGNOSTIC-ONLY].

NEGATIVE RESULT DOCUMENTATION:
Empirical evaluation shows out-of-sample ROC-AUC of 0.4621 (14-day) and 0.5329 (30-day)
with 95% bootstrap confidence intervals containing 0.50 (chance). Constant-prevalence baselines
(Brier 0.2952 on 14d, 0.2488 on 30d) match or beat Platt-scaled models (0.3032 on 14d, 0.2484 on 30d).
Ticket text features provide zero predictive skill for resolution latency in open-source queues.
These models are retained solely for offline diagnostic analysis and negative result documentation.
They are flagged as diagnostic-only and suppressed from actionable display in the API and UI.
"""

import logging
import os
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from triage.models.splits import get_time_based_splits
from triage.models.survival import TABULAR_FEATURE_COLS

logger = logging.getLogger(__name__)

MODELS_DIR = "models"
MODEL_14D_PATH = os.path.join(MODELS_DIR, "fixed_horizon_14d.joblib")
MODEL_30D_PATH = os.path.join(MODELS_DIR, "fixed_horizon_30d.joblib")


class FixedHorizonClassifier:
    """Predicts binary closure within fixed horizons (14 and 30 days) [DIAGNOSTIC-ONLY]."""

    is_diagnostic_only: bool = True
    is_actionable: bool = False
    negative_result: bool = True

    def __init__(self, horizon_days: int = 14):
        self.horizon_days = horizon_days
        self.clf = HistGradientBoostingClassifier(
            random_state=42,
            max_iter=80,
            min_samples_leaf=20,
        )
        self.prevalence_train = 0.0
        self.prevalence_val = 0.0
        self.roc_auc = 0.50
        self.pr_auc = 0.0

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Dict[str, float]:
        X_train = train_df[TABULAR_FEATURE_COLS].fillna(0)
        y_train = train_df["target_closed_in_h"]

        X_val = val_df[TABULAR_FEATURE_COLS].fillna(0)
        y_val = val_df["target_closed_in_h"]

        self.prevalence_train = float(y_train.mean())
        self.prevalence_val = float(y_val.mean())

        self.clf.fit(X_train, y_train)
        val_probs = self.clf.predict_proba(X_val)[:, 1]

        self.roc_auc = float(roc_auc_score(y_val, val_probs))
        self.pr_auc = float(average_precision_score(y_val, val_probs))

        return {
            "horizon_days": self.horizon_days,
            "train_prevalence": self.prevalence_train,
            "val_prevalence": self.prevalence_val,
            "roc_auc": round(self.roc_auc, 4),
            "pr_auc": round(self.pr_auc, 4),
        }

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_feats = X[TABULAR_FEATURE_COLS].fillna(0)
        return self.clf.predict_proba(X_feats)[:, 1]


def train_fixed_horizon_models(
    features_df: pd.DataFrame,
) -> Tuple[FixedHorizonClassifier, FixedHorizonClassifier, Dict[str, Any]]:
    """Train both 14-day and 30-day fixed horizon classifiers on eligible tickets."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    metrics = {}

    df_copy = features_df.copy()
    df_copy["created_at_dt"] = pd.to_datetime(df_copy["created_at"], errors="coerce")
    max_dt = df_copy["created_at_dt"].max()

    # 1. Horizon 14 days
    cutoff_14 = max_dt - pd.DateOffset(days=14)
    valid_14 = df_copy[df_copy["created_at_dt"] <= cutoff_14].copy()
    valid_14["target_closed_in_h"] = np.where(
        (valid_14["state"] == "closed") & (valid_14["duration_days"] <= 14), 1, 0
    )
    tr_14, vl_14, _ = get_time_based_splits(
        valid_14, timestamp_col="created_at", val_days=20, test_days=10
    )
    clf_14 = FixedHorizonClassifier(horizon_days=14)
    metrics["14d"] = clf_14.fit(tr_14, vl_14)
    joblib.dump(clf_14, MODEL_14D_PATH)

    # 2. Horizon 30 days
    cutoff_30 = max_dt - pd.DateOffset(days=30)
    valid_30 = df_copy[df_copy["created_at_dt"] <= cutoff_30].copy()
    valid_30["target_closed_in_h"] = np.where(
        (valid_30["state"] == "closed") & (valid_30["duration_days"] <= 30), 1, 0
    )
    tr_30, vl_30, _ = get_time_based_splits(
        valid_30, timestamp_col="created_at", val_days=20, test_days=10
    )
    clf_30 = FixedHorizonClassifier(horizon_days=30)
    metrics["30d"] = clf_30.fit(tr_30, vl_30)
    joblib.dump(clf_30, MODEL_30D_PATH)

    return clf_14, clf_30, metrics


def load_fixed_horizon_models() -> Tuple[
    Optional[FixedHorizonClassifier], Optional[FixedHorizonClassifier]
]:
    """Load pretrained fixed horizon models from disk if present."""
    clf_14 = joblib.load(MODEL_14D_PATH) if os.path.exists(MODEL_14D_PATH) else None
    clf_30 = joblib.load(MODEL_30D_PATH) if os.path.exists(MODEL_30D_PATH) else None
    return clf_14, clf_30
