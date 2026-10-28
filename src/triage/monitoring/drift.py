"""Feature and Prediction Drift Detection Module.

Calculates Population Stability Index (PSI) and two-sample Kolmogorov-Smirnov (KS)
tests comparing baseline training distributions against recent inference windows.
Features evaluated:
- body_len
- title_len
- has_traceback
- n_code_blocks
- n_urls
- n_missing_form_fields
- predicted_days_to_close (prediction drift)
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

try:
    from scipy import stats

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

logger = logging.getLogger(__name__)

MONITORING_DIR = os.path.join("data", "monitoring")
DRIFT_METRICS_FILE = os.path.join(MONITORING_DIR, "drift_metrics.json")
PSI_THRESHOLD_ALERT = 0.25
PSI_THRESHOLD_WARNING = 0.10


def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    num_bins: int = 10,
    eps: float = 1e-4,
) -> float:
    """Calculate Population Stability Index (PSI) between two numeric arrays.

    Standard operational thresholds applied consistently across monitoring:
    - PSI < 0.10: Insignificant change (NORMAL)
    - 0.10 <= PSI < 0.25: Moderate change (WARNING)
    - PSI >= 0.25: Significant change (ALERT / Retraining trigger)
    """
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    unique_exp = np.unique(expected)
    if len(unique_exp) <= 1:
        return 0.0

    if len(unique_exp) == 2:
        # Binary or 2-value discrete feature: split midway between the two discrete values
        split = float((unique_exp[0] + unique_exp[1]) / 2.0)
        bin_edges = np.array([-np.inf, split, np.inf])
    else:
        # Determine quantiles based on expected distribution
        percentiles = np.linspace(0, 100, num_bins + 1)
        bin_edges = np.unique(np.percentile(expected, percentiles))
        if len(bin_edges) <= 2:
            split = float((bin_edges[0] + bin_edges[-1]) / 2.0)
            bin_edges = np.array([-np.inf, split, np.inf])
        else:
            bin_edges[0] = -np.inf
            bin_edges[-1] = np.inf

    # Frequency binning
    exp_counts, _ = np.histogram(expected, bins=bin_edges)
    act_counts, _ = np.histogram(actual, bins=bin_edges)

    # Convert to fractions with smoothing epsilon
    exp_pct = (exp_counts / len(expected)) + eps
    act_pct = (act_counts / len(actual)) + eps

    # Normalize after epsilon
    exp_pct = exp_pct / exp_pct.sum()
    act_pct = act_pct / act_pct.sum()

    # PSI formula: sum((actual% - expected%) * ln(actual% / expected%))
    psi_value = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(np.round(psi_value, 4))


def calculate_ks_test(expected: np.ndarray, actual: np.ndarray) -> Tuple[float, float]:
    """Perform two-sample Kolmogorov-Smirnov test."""
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return 0.0, 1.0

    if HAS_SCIPY:
        res = stats.ks_2samp(expected, actual)
        return float(np.round(res.statistic, 4)), float(np.round(res.pvalue, 6))

    # Pure numpy KS statistic (max supremum distance between empirical CDFs)
    data_all = np.sort(np.concatenate([expected, actual]))
    cdf1 = np.searchsorted(np.sort(expected), data_all, side="right") / len(expected)
    cdf2 = np.searchsorted(np.sort(actual), data_all, side="right") / len(actual)
    ks_stat = float(np.max(np.abs(cdf1 - cdf2)))
    return float(np.round(ks_stat, 4)), 1.0


def format_drift_table(summary: Dict[str, Any]) -> str:
    """Format drift analysis results as a GitHub Markdown table."""
    features = summary.get("features_analyzed", {})
    if not features:
        return "No drift metrics available."

    lines = [
        "| Feature / Metric | Ref Mean | Current Mean | PSI Score | KS Stat | p-value | Alert Status |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for name, data in features.items():
        ref_m = data.get("ref_mean", 0.0)
        curr_m = data.get("curr_mean", 0.0)
        psi = data.get("psi", 0.0)
        ks = data.get("ks_statistic", 0.0)
        p_val = data.get("p_value", 1.0)
        alert = data.get("alert_level", "normal").upper()
        p_str = "< 0.0001" if p_val < 0.0001 else f"{p_val:.4f}"
        lines.append(
            f"| `{name}` | {ref_m:.2f} | {curr_m:.2f} | **{psi:.4f}** | {ks:.4f} | {p_str} | `{alert}` |"
        )

    return "\n".join(lines)


def run_drift_analysis(
    reference_path: str = "data/interim/features.parquet",
    current_path: Optional[str] = None,
    output_path: str = DRIFT_METRICS_FILE,
    days_current: int = 14,
) -> Dict[str, Any]:
    """Compute drift across key features and predictions comparing reference window vs recent window."""
    if not os.path.exists(reference_path):
        logger.warning(
            f"Reference features not found at {reference_path}. Generating default drift status."
        )
        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "skipped",
            "reason": f"File {reference_path} not found",
            "features_analyzed": {},
            "alert": False,
        }
        return summary

    df = pd.read_parquet(reference_path)
    if "created_at_dt" not in df.columns:
        df["created_at_dt"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)

    df = df.dropna(subset=["created_at_dt"]).sort_values("created_at_dt")
    n_rows = len(df)

    # Split: reference window vs last 14 days
    if current_path is None:
        if days_current is not None and "created_at_dt" in df.columns:
            max_dt = df["created_at_dt"].max()
            cutoff = max_dt - pd.Timedelta(days=days_current)
            ref_df = df[df["created_at_dt"] < cutoff].copy()
            curr_df = df[df["created_at_dt"] >= cutoff].copy()
            if len(ref_df) == 0 or len(curr_df) == 0:
                split_idx = int(n_rows * 0.8)
                ref_df = df.iloc[:split_idx].copy()
                curr_df = df.iloc[split_idx:].copy()
        else:
            split_idx = int(n_rows * 0.8)
            ref_df = df.iloc[:split_idx].copy()
            curr_df = df.iloc[split_idx:].copy()
    else:
        ref_df = df.copy()
        curr_df = pd.read_parquet(current_path)

    features_to_monitor = [
        "title_len",
        "body_len",
        "n_code_blocks",
        "n_urls",
        "has_traceback",
        "has_logs",
        "n_missing_form_fields",
    ]

    feature_results: Dict[str, Any] = {}
    has_alert = False

    for feat in features_to_monitor:
        if feat in ref_df.columns and feat in curr_df.columns:
            exp_vals = ref_df[feat].astype(float).values
            act_vals = curr_df[feat].astype(float).values

            psi = calculate_psi(exp_vals, act_vals)
            ks_stat, p_val = calculate_ks_test(exp_vals, act_vals)

            alert_level = "normal"
            if psi >= PSI_THRESHOLD_ALERT:
                alert_level = "alert"
                has_alert = True
            elif psi >= PSI_THRESHOLD_WARNING:
                alert_level = "warning"

            feature_results[feat] = {
                "psi": psi,
                "ks_statistic": ks_stat,
                "p_value": p_val,
                "alert_level": alert_level,
                "ref_mean": float(np.round(np.nanmean(exp_vals), 2)) if len(exp_vals) else 0.0,
                "curr_mean": float(np.round(np.nanmean(act_vals), 2)) if len(act_vals) else 0.0,
            }

    # Monitor model prediction drift if trained models are present
    models_dir = "models"
    surv_path = os.path.join(models_dir, "survival_aft.joblib")
    if os.path.exists(surv_path):
        try:
            from triage.models.survival import TABULAR_FEATURE_COLS

            surv_model = joblib.load(surv_path)
            ref_surv_preds = surv_model.predict(ref_df[TABULAR_FEATURE_COLS])
            curr_surv_preds = surv_model.predict(curr_df[TABULAR_FEATURE_COLS])

            psi_surv = calculate_psi(ref_surv_preds, curr_surv_preds)
            ks_surv, p_surv = calculate_ks_test(ref_surv_preds, curr_surv_preds)
            alert_surv = "normal"
            if psi_surv >= PSI_THRESHOLD_ALERT:
                alert_surv = "alert"
                has_alert = True
            elif psi_surv >= PSI_THRESHOLD_WARNING:
                alert_surv = "warning"

            feature_results["predicted_time_to_close"] = {
                "psi": psi_surv,
                "ks_statistic": ks_surv,
                "p_value": p_surv,
                "alert_level": alert_surv,
                "ref_mean": float(np.round(np.nanmean(ref_surv_preds), 2)),
                "curr_mean": float(np.round(np.nanmean(curr_surv_preds), 2)),
            }
        except Exception as e:
            logger.debug(f"Could not compute prediction drift for survival model: {e}")

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "reference_records": len(ref_df),
        "current_records": len(curr_df),
        "days_current": days_current,
        "features_analyzed": feature_results,
        "overall_drift_alert": has_alert,
        "psi_threshold_alert": PSI_THRESHOLD_ALERT,
        "psi_threshold_warning": PSI_THRESHOLD_WARNING,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Drift analysis saved to {output_path}. Overall alert: {has_alert}")
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    res = run_drift_analysis()
    print("\n=== DRIFT ANALYSIS RESULTS (REFERENCE WINDOW VS LAST 14 DAYS) ===")
    print(format_drift_table(res))
    print(f"\nOverall Drift Alert: {res['overall_drift_alert']}")

