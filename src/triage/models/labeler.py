"""Integration Label Recommendation Model for Unlabeled Tickets.

Provides fallback label suggestions for tickets where the issue author omitted the integration field.
CRITICAL LEAKAGE GUARD:
- Model is strictly trained on 'clean_text_for_label' where the 'Integration causing the issue'
  section has been stripped out.
- Predicts Top-N integrations + 'other'.
- Evaluates Macro-F1, Micro-F1, and Top-3 Accuracy.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


class IntegrationLabelRecommender:
    """TF-IDF + Multinomial Logistic Regression model for integration prediction."""

    def __init__(self, top_n_labels: int = 20, max_features: int = 1500):
        self.top_n_labels = top_n_labels
        self.max_features = max_features
        self.vectorizer_: Optional[TfidfVectorizer] = None
        self.classifier_: Optional[LogisticRegression] = None
        self.target_labels_: List[str] = []
        self.majority_label_: str = "other"

    def _prepare_targets(self, labels: pd.Series) -> pd.Series:
        valid_labels = labels[labels != "unlabeled"]
        top_labels = valid_labels.value_counts().head(self.top_n_labels).index.tolist()
        self.target_labels_ = top_labels + ["other"]
        self.majority_label_ = top_labels[0] if top_labels else "other"

        return labels.apply(lambda x: x if x in top_labels else "other")

    def fit(self, train_df: pd.DataFrame) -> "IntegrationLabelRecommender":
        # Filter out tickets that are truly unlabeled in ground truth
        labeled_mask = train_df["integration_clean"] != "unlabeled"
        train_labeled = train_df[labeled_mask]

        y_train = self._prepare_targets(train_labeled["integration_clean"])
        texts = train_labeled["clean_text_for_label"].fillna("").astype(str)

        self.vectorizer_ = TfidfVectorizer(
            max_features=self.max_features,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        X_train = self.vectorizer_.fit_transform(texts)

        self.classifier_ = LogisticRegression(
            max_iter=300,
            C=1.5,
            class_weight="balanced",
            random_state=42,
        )
        self.classifier_.fit(X_train, y_train)
        return self

    def predict_with_confidence(
        self, texts: pd.Series
    ) -> Tuple[List[str], List[float], List[List[str]]]:
        """Predict top suggested label, confidence probability, and top-3 candidates."""
        clean_texts = texts.fillna("").astype(str)
        X = self.vectorizer_.transform(clean_texts)
        probs = self.classifier_.predict_proba(X)
        classes = self.classifier_.classes_

        top_labels = []
        confidences = []
        top_3_candidates = []

        for p_row in probs:
            ranked_idx = np.argsort(p_row)[::-1]
            best_idx = ranked_idx[0]
            top_labels.append(str(classes[best_idx]))
            confidences.append(float(p_row[best_idx]))
            top_3 = [str(classes[i]) for i in ranked_idx[:3]]
            top_3_candidates.append(top_3)

        return top_labels, confidences, top_3_candidates

    def evaluate(self, val_df: pd.DataFrame) -> Dict[str, float]:
        """Evaluate Macro-F1, Micro-F1, Top-1 and Top-3 accuracy on validation set."""
        labeled_val = val_df[val_df["integration_clean"] != "unlabeled"]
        if labeled_val.empty:
            return {"macro_f1": 0.0, "micro_f1": 0.0, "top_1_acc": 0.0, "top_3_acc": 0.0}

        y_true = (
            labeled_val["integration_clean"]
            .apply(lambda x: x if x in self.target_labels_ else "other")
            .values
        )

        preds, confs, top_3_list = self.predict_with_confidence(labeled_val["clean_text_for_label"])

        macro_f1 = float(f1_score(y_true, preds, average="macro", zero_division=0))
        micro_f1 = float(f1_score(y_true, preds, average="micro", zero_division=0))
        top_1_acc = float(accuracy_score(y_true, preds))

        # Top-3 accuracy
        top_3_correct = sum(1 for yt, cands in zip(y_true, top_3_list) if yt in cands)
        top_3_acc = float(top_3_correct / len(y_true))

        # Majority baseline
        majority_preds = [self.majority_label_] * len(y_true)
        baseline_acc = float(accuracy_score(y_true, majority_preds))
        baseline_macro_f1 = float(
            f1_score(y_true, majority_preds, average="macro", zero_division=0)
        )

        return {
            "macro_f1": round(macro_f1, 4),
            "micro_f1": round(micro_f1, 4),
            "top_1_accuracy": round(top_1_acc, 4),
            "top_3_accuracy": round(top_3_acc, 4),
            "baseline_accuracy": round(baseline_acc, 4),
            "baseline_macro_f1": round(baseline_macro_f1, 4),
        }
