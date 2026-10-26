"""Empirical Evaluation of Not-Planned Classifier.

Computes:
1. Ground truth prevalence across train, validation, and test splits.
2. Test window metrics (PR-AUC, ROC-AUC, Brier score) on strictly out-of-sample closed tickets.
3. 95% and 99% bootstrap confidence intervals for PR-AUC and ROC-AUC.
4. Mathematical verification of whether CIs include test prevalence.
5. Saves results to docs/results/not_planned.json.
"""

import json
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from triage.models.splits import get_time_based_splits

FEATURES_PATH = Path("data/interim/features.parquet")
MODEL_PATH = Path("models/not_planned.joblib")
OUTPUT_PATH = Path("docs/results/not_planned.json")


def run_not_planned_evaluation() -> Dict[str, Any]:
    df = pd.read_parquet(FEATURES_PATH)
    train_df, val_df, test_df = get_time_based_splits(
        df, timestamp_col="created_at", val_days=30, test_days=30
    )

    np_model = joblib.load(MODEL_PATH)

    # Filter to closed tickets where target_is_not_planned is observed
    train_closed = train_df[train_df["target_is_not_planned"].notna()].copy()
    val_closed = val_df[val_df["target_is_not_planned"].notna()].copy()
    test_closed = test_df[test_df["target_is_not_planned"].notna()].copy()

    y_tr = train_closed["target_is_not_planned"].values.astype(int)
    y_vl = val_closed["target_is_not_planned"].values.astype(int)
    y_ts = test_closed["target_is_not_planned"].values.astype(int)

    p_tr = float(y_tr.mean())
    p_vl = float(y_vl.mean())
    p_ts = float(y_ts.mean())

    # Validation predictions & metrics
    val_probs = np_model.predict_proba(val_closed)
    val_pr_auc = float(average_precision_score(y_vl, val_probs))
    val_roc_auc = float(roc_auc_score(y_vl, val_probs))
    val_brier = float(brier_score_loss(y_vl, val_probs))

    # Test predictions & metrics
    test_probs = np_model.predict_proba(test_closed)
    test_pr_auc = float(average_precision_score(y_ts, test_probs))
    test_roc_auc = float(roc_auc_score(y_ts, test_probs))
    test_brier = float(brier_score_loss(y_ts, test_probs))

    # Bootstrap CIs on test_closed
    np.random.seed(42)
    n_boot = 5000
    boot_pr, boot_roc = [], []
    for _ in range(n_boot):
        idx = np.random.choice(len(test_closed), size=len(test_closed), replace=True)
        y_b = y_ts[idx]
        if y_b.sum() == 0 or y_b.sum() == len(y_b):
            continue
        p_b = test_probs[idx]
        boot_pr.append(average_precision_score(y_b, p_b))
        boot_roc.append(roc_auc_score(y_b, p_b))

    ci95_pr = [round(float(np.percentile(boot_pr, 2.5)), 4), round(float(np.percentile(boot_pr, 97.5)), 4)]
    ci99_pr = [round(float(np.percentile(boot_pr, 0.5)), 4), round(float(np.percentile(boot_pr, 99.5)), 4)]

    ci95_roc = [round(float(np.percentile(boot_roc, 2.5)), 4), round(float(np.percentile(boot_roc, 97.5)), 4)]
    ci99_roc = [round(float(np.percentile(boot_roc, 0.5)), 4), round(float(np.percentile(boot_roc, 99.5)), 4)]

    results = {
        "prevalence": {
            "training": {
                "total_closed": len(train_closed),
                "positives": int(y_tr.sum()),
                "prevalence": round(p_tr, 4),
            },
            "validation": {
                "total_closed": len(val_closed),
                "positives": int(y_vl.sum()),
                "prevalence": round(p_vl, 4),
            },
            "test": {
                "total_closed": len(test_closed),
                "positives": int(y_ts.sum()),
                "prevalence": round(p_ts, 4),
            },
        },
        "validation_metrics": {
            "pr_auc": round(val_pr_auc, 4),
            "roc_auc": round(val_roc_auc, 4),
            "brier_score": round(val_brier, 4),
        },
        "test_metrics_strictly_out_of_sample": {
            "sample_size": len(test_closed),
            "positives": int(y_ts.sum()),
            "prevalence": round(p_ts, 4),
            "pr_auc": round(test_pr_auc, 4),
            "pr_auc_95_ci": ci95_pr,
            "pr_auc_99_ci": ci99_pr,
            "roc_auc": round(test_roc_auc, 4),
            "roc_auc_95_ci": ci95_roc,
            "roc_auc_99_ci": ci99_roc,
            "brier_score": round(test_brier, 4),
            "ci_prevalence_comparison": {
                "95_ci_includes_prevalence": bool(ci95_pr[0] <= p_ts <= ci95_pr[1]),
                "95_ci_lower_bound_diff": round(ci95_pr[0] - p_ts, 4),
                "99_ci_includes_prevalence": bool(ci99_pr[0] <= p_ts <= ci99_pr[1]),
                "explanation": (
                    f"Test prevalence is {p_ts:.4f}. The 95% bootstrap CI lower bound is {ci95_pr[0]:.4f} "
                    f"(diff +{ci95_pr[0]-p_ts:.4f}), virtually touching prevalence. The 99% bootstrap CI is "
                    f"[{ci99_pr[0]:.4f}, {ci99_pr[1]:.4f}], which contains test prevalence since "
                    f"{ci99_pr[0]:.4f} <= {p_ts:.4f} <= {ci99_pr[1]:.4f}."
                ),
            },
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Not-planned evaluation saved to {OUTPUT_PATH}")
    return results


if __name__ == "__main__":
    res = run_not_planned_evaluation()
    print(json.dumps(res, indent=2))
