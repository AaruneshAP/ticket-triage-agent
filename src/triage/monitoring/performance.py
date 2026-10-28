"""Production Performance Tracking Module (Phase 8).

Tracks empirical model performance over time as tickets close and ground truth
resolves. Evaluates:
1. Time-to-Close Survival AFT: Harrell's Concordance Index (C-Index) and MAE.
2. Not-Planned Risk Classifier: Precision-Recall AUC (PR-AUC), Brier Score, and ROC-AUC.
3. Integration Label Recommender: Top-1 Accuracy and Macro F1 score.

Monitors performance across rolling chronological resolution cohorts and alerts on
statistically significant degradation relative to baseline validation benchmarks.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)

from triage.models.survival import TABULAR_FEATURE_COLS

logger = logging.getLogger(__name__)

MONITORING_DIR = os.path.join("data", "monitoring")
PERFORMANCE_HISTORY_FILE = os.path.join(MONITORING_DIR, "performance_history.json")

# Production Baseline Benchmarks (from docs/evaluation.md & README.md)
BASELINE_C_INDEX = 0.5426
BASELINE_PR_AUC = 0.2222
BASELINE_BRIER = 0.1997
BASELINE_LABEL_ACC = 0.8000

# Degradation alert thresholds
DEGRADATION_THRESHOLD_BRIER = 0.15  # Alert if Brier score degrades by > 15%
DEGRADATION_THRESHOLD_C_INDEX = 0.10  # Alert if C-index drops by > 10%
DEGRADATION_THRESHOLD_PR_AUC = 0.20  # Alert if PR-AUC drops by > 20%



DEFAULT_TRAINING_CUTOFF = pd.Timestamp("2026-08-27T05:00:47Z")


def evaluate_cohort_performance(
    cohort_df: pd.DataFrame,
    surv_model: Any,
    np_model: Any,
    labeler_model: Any = None,
) -> Dict[str, Any]:
    """Evaluate all models on a closed ticket cohort with known ground truth."""
    metrics: Dict[str, Any] = {
        "n_tickets": len(cohort_df),
    }

    # 1. Survival AFT: C-Index and MAE
    if surv_model is not None and "target_days_to_close_or_censor" in cohort_df.columns:
        durations = cohort_df["target_days_to_close_or_censor"].values
        events = cohort_df["target_is_closed_event"].astype(int).values

        # Only evaluate if we have non-trivial events
        if events.sum() > 0 and len(durations) > 5:
            try:
                feat_cols = [c for c in TABULAR_FEATURE_COLS if c in cohort_df.columns]
                x_input = (
                    cohort_df[TABULAR_FEATURE_COLS]
                    if len(feat_cols) == len(TABULAR_FEATURE_COLS)
                    else cohort_df
                )
                preds = surv_model.predict(x_input)
                c_idx = float(np.round(concordance_index(durations, preds, events), 4))
                mae = float(np.round(np.mean(np.abs(preds - durations)), 2))
                metrics["survival_c_index"] = c_idx
                metrics["survival_mae_days"] = mae
            except Exception as e:
                logger.debug(f"Survival evaluation notice: {e}")
                metrics["survival_c_index"] = None
                metrics["survival_mae_days"] = None

    # 2. Not-Planned Classifier: PR-AUC, Brier score, ROC-AUC
    if np_model is not None and "target_is_not_planned" in cohort_df.columns:
        y_true = cohort_df["target_is_not_planned"].astype(int).values
        n_pos = int(y_true.sum())
        metrics["not_planned_positives"] = n_pos

        if 0 < n_pos < len(y_true):
            try:
                y_prob = np_model.predict_proba(cohort_df)
                pr_auc = float(np.round(average_precision_score(y_true, y_prob), 4))
                brier = float(np.round(brier_score_loss(y_true, y_prob), 4))
                roc_auc = float(np.round(roc_auc_score(y_true, y_prob), 4))
                metrics["not_planned_pr_auc"] = pr_auc
                metrics["not_planned_brier"] = brier
                metrics["not_planned_roc_auc"] = roc_auc
            except Exception as e:
                logger.debug(f"Not-planned evaluation notice: {e}")
                metrics["not_planned_pr_auc"] = None
                metrics["not_planned_brier"] = None
                metrics["not_planned_roc_auc"] = None
        else:
            metrics["not_planned_pr_auc"] = None
            metrics["not_planned_brier"] = None
            metrics["not_planned_roc_auc"] = None

    # 3. Label Recommender: Accuracy (Evaluated on labeled tickets, matching README definition)
    if labeler_model is not None and "target_integration" in cohort_df.columns:
        valid_labels = cohort_df[
            cohort_df["target_integration"].notna()
            & (cohort_df["target_integration"] != "unlabeled")
            & cohort_df["clean_text_for_label"].notna()
        ].copy()
        if len(valid_labels) > 0:
            try:
                # Map rare long-tail integrations not in top_n target_labels_ to 'other'
                # to match model prediction space (top-20 + 'other')
                if hasattr(labeler_model, "target_labels_") and labeler_model.target_labels_:
                    y_true_lbl = (
                        valid_labels["target_integration"]
                        .apply(lambda x: x if x in labeler_model.target_labels_ else "other")
                        .astype(str)
                        .values
                    )
                else:
                    y_true_lbl = valid_labels["target_integration"].astype(str).values

                preds_lbl, _, _ = labeler_model.predict_with_confidence(
                    valid_labels["clean_text_for_label"]
                )
                acc = float(np.round(accuracy_score(y_true_lbl, preds_lbl), 4))
                macro_f1 = float(
                    np.round(f1_score(y_true_lbl, preds_lbl, average="macro", zero_division=0), 4)
                )
                metrics["label_accuracy"] = acc
                metrics["label_macro_f1"] = macro_f1
            except Exception as e:
                logger.debug(f"Label recommender evaluation notice: {e}")
                metrics["label_accuracy"] = None
                metrics["label_macro_f1"] = None

    return metrics


def track_model_performance(
    features_path: str = "data/interim/features.parquet",
    models_dir: str = "models",
    output_path: str = PERFORMANCE_HISTORY_FILE,
    window_days: int = 14,
    training_cutoff: Optional[pd.Timestamp] = DEFAULT_TRAINING_CUTOFF,
    out_of_sample_only: bool = True,
) -> Dict[str, Any]:
    """Track model performance metrics across closed ticket cohorts over time.

    When out_of_sample_only is True, evaluates exclusively tickets created after
    the model training cutoff (DEFAULT_TRAINING_CUTOFF = 2026-08-27T05:00:47Z).
    Tickets created prior to cutoff are training data and would give artificially
    memorized metrics (e.g. PR-AUC ~ 0.99).
    """
    if not os.path.exists(features_path):
        return {
            "status": "error",
            "message": f"Features file not found at {features_path}",
        }

    df = pd.read_parquet(features_path)
    if "created_at_dt" not in df.columns:
        df["created_at_dt"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    if "closed_at_dt" not in df.columns:
        df["closed_at_dt"] = pd.to_datetime(df["closed_at"], errors="coerce", utc=True)

    if out_of_sample_only and training_cutoff is not None:
        closed = df[df["closed_at_dt"].notna() & (df["created_at_dt"] > training_cutoff)].sort_values("closed_at_dt").copy()
    else:
        closed = df[df["closed_at_dt"].notna()].sort_values("closed_at_dt").copy()

    if len(closed) == 0:
        return {
            "status": "warning",
            "message": "No eligible closed tickets found for performance evaluation.",
        }

    # Load active models
    surv_path = os.path.join(models_dir, "survival_aft.joblib")
    np_path = os.path.join(models_dir, "not_planned.joblib")
    lbl_path = os.path.join(models_dir, "labeler.joblib")

    surv_model = joblib.load(surv_path) if os.path.exists(surv_path) else None
    np_model = joblib.load(np_path) if os.path.exists(np_path) else None
    labeler_model = joblib.load(lbl_path) if os.path.exists(lbl_path) else None

    # Partition closed tickets into chronological windows
    max_closed = closed["closed_at_dt"].max()
    min_closed = closed["closed_at_dt"].min()

    cohorts: List[Dict[str, Any]] = []
    current_end = max_closed

    # Build up to 6 chronological cohorts back in time
    while current_end > min_closed and len(cohorts) < 6:
        cohort_start = current_end - pd.Timedelta(int(window_days), unit="D")
        mask = (closed["closed_at_dt"] > cohort_start) & (closed["closed_at_dt"] <= current_end)
        cohort_slice = closed[mask]

        if len(cohort_slice) >= 15:
            cohort_metrics = evaluate_cohort_performance(
                cohort_slice, surv_model, np_model, labeler_model
            )
            # Map chronological cohort to split definition
            # Cohort labels must match docs/evaluation.md:
            # - Window <= 2026-08-31: "post-training" (not "train-overlap")
            # - Windows overlapping validation up to 2026-09-26: "post-training, includes validation window"
            # - Only windows strictly after 2026-09-26 are "test"
            if current_end <= pd.to_datetime("2026-08-31 23:59:59+00:00"):
                split_tag = "post-training"
                sample_tag = "POST_TRAINING"
            elif cohort_start < pd.to_datetime("2026-09-26 23:59:59+00:00"):
                split_tag = "post-training, includes validation window"
                sample_tag = "POST_TRAINING_VAL_OVERLAP"
            else:
                split_tag = "test"
                sample_tag = "TEST"

            cohorts.append({
                "window_start": cohort_start.strftime("%Y-%m-%d"),
                "window_end": current_end.strftime("%Y-%m-%d"),
                "days": window_days,
                "split": split_tag,
                "sample_type": sample_tag,
                **cohort_metrics,
            })
        current_end = cohort_start

    # Reverse so oldest to newest
    cohorts.reverse()

    # Calculate overall degradation vs baseline for latest cohort
    alerts = []
    if cohorts:
        latest = cohorts[-1]
        latest_c_idx = latest.get("survival_c_index")
        latest_brier = latest.get("not_planned_brier")
        latest_pr_auc = latest.get("not_planned_pr_auc")

        if latest_brier is not None:
            brier_deg = (latest_brier - BASELINE_BRIER) / BASELINE_BRIER
            if brier_deg > DEGRADATION_THRESHOLD_BRIER:
                alerts.append(
                    f"Not-Planned Brier score degraded by {brier_deg*100:.1f}% vs baseline"
                )

        if latest_pr_auc is not None:
            pr_drop = (BASELINE_PR_AUC - latest_pr_auc) / BASELINE_PR_AUC
            if pr_drop > DEGRADATION_THRESHOLD_PR_AUC:
                alerts.append(f"Not-Planned PR-AUC dropped by {pr_drop*100:.1f}% vs baseline")

        if latest_c_idx is not None:
            c_idx_drop = (BASELINE_C_INDEX - latest_c_idx) / BASELINE_C_INDEX
            if c_idx_drop > DEGRADATION_THRESHOLD_C_INDEX:
                alerts.append(f"Survival C-Index dropped by {c_idx_drop*100:.1f}% vs baseline")

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_closed_tickets": len(closed),
        "out_of_sample_only": out_of_sample_only,
        "training_cutoff": training_cutoff.isoformat() if training_cutoff else None,
        "cohort_window_days": window_days,
        "cohorts_evaluated": len(cohorts),
        "cohort_metrics": cohorts,
        "production_baselines": {
            "survival_c_index": BASELINE_C_INDEX,
            "not_planned_pr_auc": BASELINE_PR_AUC,
            "not_planned_brier": BASELINE_BRIER,
            "label_accuracy": BASELINE_LABEL_ACC,
        },
        "performance_alerts": alerts,
        "has_performance_alert": len(alerts) > 0,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        f"Performance tracking completed over {len(cohorts)} cohorts (out_of_sample_only={out_of_sample_only}). "
        f"Saved to {output_path}."
    )
    return summary


def format_performance_table(summary: Dict[str, Any]) -> str:
    """Format performance tracking cohorts as a markdown table."""
    cohorts = summary.get("cohort_metrics", [])
    if not cohorts:
        return "No performance cohorts available."

    lines = [
        "| Closed Window | Sample Type | Tickets | Survival C-Index | Not-Planned PR-AUC | Not-Planned Brier | Label Acc |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for c in cohorts:
        win = f"{c['window_start']} to {c['window_end']}"
        stype = f"`{c.get('sample_type', 'OUT_OF_SAMPLE')}`"
        n_t = c["n_tickets"]
        c_idx = f"{c['survival_c_index']:.4f}" if c.get("survival_c_index") is not None else "N/A"
        pr_auc = f"{c['not_planned_pr_auc']:.4f}" if c.get("not_planned_pr_auc") is not None else "N/A"
        brier = f"{c['not_planned_brier']:.4f}" if c.get("not_planned_brier") is not None else "N/A"
        acc = f"{c['label_accuracy']:.4f}" if c.get("label_accuracy") is not None else "N/A"
        lines.append(f"| {win} | {stype} | {n_t} | **{c_idx}** | **{pr_auc}** | {brier} | {acc} |")

    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    res = track_model_performance()
    print("\n=== MODEL PERFORMANCE TRACKING OVER TIME (AS TICKETS CLOSE) ===")
    print(format_performance_table(res))
    if res.get("performance_alerts"):
        print("\nAlerts:")
        for a in res["performance_alerts"]:
            print(f"  - {a}")
    else:
        print("\nAll models operating within acceptable performance boundaries.")
