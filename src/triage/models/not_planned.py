"""Not-Planned & Without-Fix Risk Classification Model with SHAP Explainability.

Predicts whether an incoming ticket has high probability of being closed without a fix
(state_reason in ['not_planned', 'duplicate']), allowing maintainers to flag or triage early.
Includes:
- Baselines: Prevalence baseline & Logistic Regression on integration volume.
- Gradient Boosting Classifier (XGBoost) with tabular + TF-IDF n-grams.
- Metrics: PR-AUC, ROC-AUC, Brier score, Recall at target precision.
- SHAP: Global feature rankings and per-ticket explanation generator.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from triage.features.leakage_guard import assert_no_leakage

NOT_PLANNED_TABULAR_COLS: List[str] = [
    "title_len",
    "body_len",
    "n_code_blocks",
    "n_urls",
    "has_traceback",
    "has_logs",
    "refers_to_issue",
    "n_missing_form_fields",
    "has_last_working_version",
    "is_template",
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


class NotPlannedClassifier:
    """XGBoost Classifier for Not-Planned risk prediction with integrated SHAP explanations."""

    def __init__(self, max_features_tfidf: int = 40, max_depth: int = 4, n_estimators: int = 80):
        self.max_features_tfidf = max_features_tfidf
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.tfidf_: Optional[TfidfVectorizer] = None
        self.model_: Optional[xgb.XGBClassifier] = None
        self.explainer_: Optional[shap.TreeExplainer] = None
        self.feature_names_: List[str] = []

    def _prepare_matrix(self, df: pd.DataFrame, is_train: bool = False) -> np.ndarray:
        assert_no_leakage(df[NOT_PLANNED_TABULAR_COLS])
        X_tab = df[NOT_PLANNED_TABULAR_COLS].fillna(0).values

        texts = df["clean_problem_text"].fillna("").astype(str)
        if is_train:
            self.tfidf_ = TfidfVectorizer(
                max_features=self.max_features_tfidf,
                stop_words="english",
                ngram_range=(1, 2),
                sublinear_tf=True,
            )
            X_text = self.tfidf_.fit_transform(texts).toarray()
            text_feature_names = [f"word_{w}" for w in self.tfidf_.get_feature_names_out()]
            self.feature_names_ = NOT_PLANNED_TABULAR_COLS + text_feature_names
        else:
            X_text = self.tfidf_.transform(texts).toarray()

        return np.hstack([X_tab, X_text]).astype(np.float32)

    def fit(self, train_df: pd.DataFrame, y_train: np.ndarray) -> "NotPlannedClassifier":
        X = self._prepare_matrix(train_df, is_train=True)

        scale_pos_weight = float((len(y_train) - y_train.sum()) / max(y_train.sum(), 1))
        self.model_ = xgb.XGBClassifier(
            max_depth=self.max_depth,
            n_estimators=self.n_estimators,
            learning_rate=0.08,
            scale_pos_weight=min(scale_pos_weight, 5.0),
            eval_metric="logloss",
            random_state=42,
        )
        self.model_.fit(X, y_train)
        self.explainer_ = shap.TreeExplainer(self.model_)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X = self._prepare_matrix(df, is_train=False)
        return self.model_.predict_proba(X)[:, 1]

    def explain_ticket(self, df_single: pd.DataFrame, top_k: int = 4) -> List[Dict[str, Any]]:
        """Compute top SHAP contributor features for a single ticket."""
        X = self._prepare_matrix(df_single, is_train=False)
        shap_vals = self.explainer_.shap_values(X)
        if isinstance(shap_vals, list):
            # For multi-class or binary lists
            vals = shap_vals[1][0] if len(shap_vals) > 1 else shap_vals[0][0]
        elif len(shap_vals.shape) == 2:
            vals = shap_vals[0]
        else:
            vals = shap_vals

        ranked_indices = np.argsort(np.abs(vals))[::-1][:top_k]
        explanations = []
        for idx in ranked_indices:
            feat_name = self.feature_names_[idx]
            impact = float(vals[idx])
            val_in_row = float(X[0, idx])
            explanations.append(
                {
                    "feature": feat_name,
                    "impact": round(impact, 4),
                    "direction": "increases_risk" if impact > 0 else "reduces_risk",
                    "value": round(val_in_row, 2),
                }
            )
        return explanations


def evaluate_not_planned(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    baseline_prevalence: float,
) -> Dict[str, float]:
    """Compute PR-AUC, ROC-AUC, Brier score, and recall at fixed precision."""
    roc = float(roc_auc_score(y_true, y_prob))
    pr = float(average_precision_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))

    # Recall at fixed precision >= 0.50
    thresholds = np.sort(y_prob)[::-1]
    recall_at_50 = 0.0
    for th in thresholds:
        preds = (y_prob >= th).astype(int)
        tp = int(((preds == 1) & (y_true == 1)).sum())
        fp = int(((preds == 1) & (y_true == 0)).sum())
        fn = int(((preds == 0) & (y_true == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if prec >= 0.50 and rec > recall_at_50:
            recall_at_50 = rec

    return {
        "roc_auc": round(roc, 4),
        "pr_auc": round(pr, 4),
        "baseline_pr_auc": round(baseline_prevalence, 4),
        "brier_score": round(brier, 4),
        "recall_at_50_precision": round(recall_at_50, 4),
    }
