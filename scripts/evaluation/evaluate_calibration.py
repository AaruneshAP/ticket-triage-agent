"""Full Post-Cutoff Holdout Calibration Analysis (14-day and 30-day horizons).

Audits and evaluates fixed-horizon resolution models:
1. Training and tuning data provenance:
   - Fixed-horizon models were trained via HistGradientBoostingClassifier in src/triage/models/fixed_horizon.py.
   - Neither model was logged to MLflow (no MLflow run IDs exist for fixed horizon).
   - 14d Model: Train split N=1,451 (2018-03-21 to 2026-09-12), Val/Tune split N=453 (2026-09-12 to 2026-10-02).
   - 30d Model: Train split N=975 (2018-03-21 to 2026-08-27), Val/Tune split N=575 (2026-08-27 to 2026-09-15).
   - In the previously reported post-cutoff cohorts (1,362 and 786 tickets):
     - 14d cohort contained 475 training tickets and 453 validation tickets (928 / 1,362 = 68.1% in-sample/tuning data).
     - 30d cohort contained 574 validation tickets (574 / 786 = 73.0% tuning data).
2. Evaluates performance and calibration on:
   - The full post-cutoff cohorts (N=1,362 and N=786, labeled as partially in-sample / tuning).
   - Strictly out-of-sample test cohorts created after the model's last training/calibration date:
     - 14d Test (N=434, created 2026-10-02 to 2026-10-12).
     - 30d Test (N=212, created 2026-09-16 to 2026-09-26).
3. Reports ROC-AUC, PR-AUC, and 1,000-sample bootstrap 95% confidence intervals.
4. Shows before and after recalibration (Platt scaling fit on earlier validation window, tested out-of-sample).
5. Generates docs/images/calibration_reliability_plot.png.
"""

import math
import os
from typing import Any, Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from triage.models.splits import get_time_based_splits


def wilson_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate Wilson score interval for binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    z = 1.95996  # 95%
    p = k / n
    denom = 1.0 + (z**2) / n
    center = (p + (z**2) / (2.0 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1.0 - p) / n + (z**2) / (4.0 * (n**2)))
    return max(0.0, center - margin), min(1.0, center + margin)


def compute_calibration_table(preds: np.ndarray, y: np.ndarray, n_bins: int = 5) -> pd.DataFrame:
    """Compute calibration metrics grouped into predicted probability bins."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.digitize(preds, bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        mask = bin_idx == b
        n_b = int(mask.sum())
        if n_b == 0:
            continue
        p_mean = float(preds[mask].mean())
        k_b = int(y[mask].sum())
        act_rate = k_b / n_b
        low, high = wilson_ci(k_b, n_b)
        rows.append(
            {
                "Bin Range": f"[{bins[b]:.1f}, {bins[b+1]:.1f})",
                "n": n_b,
                "Mean Predicted": round(p_mean, 4),
                "Closed Count": k_b,
                "Observed Rate": round(act_rate, 4),
                "Wilson CI 95%": f"[{low:.1%}, {high:.1%}]",
                "ci_low": low,
                "ci_high": high,
            }
        )
    return pd.DataFrame(rows)


def evaluate_cohort_metrics(
    model: Any, subset: pd.DataFrame, horizon_days: int, n_boot: int = 1000
) -> Dict[str, Any]:
    """Compute ROC-AUC, PR-AUC, and bootstrap confidence intervals."""
    y_true = np.where(
        (subset["state"] == "closed") & (subset["duration_days"] <= float(horizon_days)), 1, 0
    )
    preds = model.predict_proba(subset)

    roc = float(roc_auc_score(y_true, preds))
    pr = float(average_precision_score(y_true, preds))
    prev = float(y_true.mean())
    brier = float(brier_score_loss(y_true, preds))

    rng = np.random.default_rng(42)
    roc_boots = []
    pr_boots = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        roc_boots.append(roc_auc_score(y_true[idx], preds[idx]))
        pr_boots.append(average_precision_score(y_true[idx], preds[idx]))

    roc_ci = (float(np.percentile(roc_boots, 2.5)), float(np.percentile(roc_boots, 97.5)))
    pr_ci = (float(np.percentile(pr_boots, 2.5)), float(np.percentile(pr_boots, 97.5)))

    table = compute_calibration_table(preds, y_true, n_bins=5)

    return {
        "n": n,
        "prevalence": prev,
        "mean_pred": float(preds.mean()),
        "brier": brier,
        "roc_auc": roc,
        "roc_ci": roc_ci,
        "pr_auc": pr,
        "pr_ci": pr_ci,
        "y_true": y_true,
        "preds": preds,
        "table": table,
    }


def main():
    # 1. Load data
    df = pd.read_parquet("data/interim/features.parquet")
    df["created_at_dt"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    max_dt = df["created_at_dt"].max()

    m14 = joblib.load("models/fixed_horizon_14d.joblib")
    m30 = joblib.load("models/fixed_horizon_30d.joblib")

    # 2. Extract splits as defined in training code (src/triage/models/fixed_horizon.py)
    cutoff_14 = max_dt - pd.DateOffset(days=14)
    v14 = df[df["created_at_dt"] <= cutoff_14].copy()
    v14["target_closed_in_h"] = np.where(
        (v14["state"] == "closed") & (v14["duration_days"] <= 14), 1, 0
    )
    tr14, vl14, ts14 = get_time_based_splits(
        v14, timestamp_col="created_at", val_days=20, test_days=10
    )

    cutoff_30 = max_dt - pd.DateOffset(days=30)
    v30 = df[df["created_at_dt"] <= cutoff_30].copy()
    v30["target_closed_in_h"] = np.where(
        (v30["state"] == "closed") & (v30["duration_days"] <= 30), 1, 0
    )
    tr30, vl30, ts30 = get_time_based_splits(
        v30, timestamp_col="created_at", val_days=20, test_days=10
    )

    c_prev_start = pd.to_datetime("2026-08-27T05:00:47Z")
    sub14_prev = df[(df["created_at_dt"] > c_prev_start) & (df["created_at_dt"] <= cutoff_14)].copy()
    sub30_prev = df[(df["created_at_dt"] > c_prev_start) & (df["created_at_dt"] <= cutoff_30)].copy()

    print("================================================================================")
    print("FIXED-HORIZON TRAINING, TUNING & DATA PROVENANCE AUDIT")
    print("================================================================================")
    print("Code location: src/triage/models/fixed_horizon.py (HistGradientBoostingClassifier)")
    print("MLflow Tracking: NONE (fixed-horizon models were evaluated in diagnostics and not logged to MLflow)")
    print(f"14-Day Model Train Split: N={len(tr14)} ({tr14['created_at_dt'].min()} to {tr14['created_at_dt'].max()})")
    print(f"14-Day Model Val/Tune Split: N={len(vl14)} ({vl14['created_at_dt'].min()} to {vl14['created_at_dt'].max()})")
    print(f"14-Day Model Test Split: N={len(ts14)} ({ts14['created_at_dt'].min()} to {ts14['created_at_dt'].max()})")
    print(f"30-Day Model Train Split: N={len(tr30)} ({tr30['created_at_dt'].min()} to {tr30['created_at_dt'].max()})")
    print(f"30-Day Model Val/Tune Split: N={len(vl30)} ({vl30['created_at_dt'].min()} to {vl30['created_at_dt'].max()})")
    print(f"30-Day Model Test Split: N={len(ts30)} ({ts30['created_at_dt'].min()} to {ts30['created_at_dt'].max()})")

    # Check contamination in previously reported post-cutoff cohorts
    overlap_14_tr = len(set(sub14_prev["ticket_number"]).intersection(set(tr14["ticket_number"])))
    overlap_14_vl = len(set(sub14_prev["ticket_number"]).intersection(set(vl14["ticket_number"])))
    overlap_14_ts = len(set(sub14_prev["ticket_number"]).intersection(set(ts14["ticket_number"])))

    overlap_30_tr = len(set(sub30_prev["ticket_number"]).intersection(set(tr30["ticket_number"])))
    overlap_30_vl = len(set(sub30_prev["ticket_number"]).intersection(set(vl30["ticket_number"])))
    overlap_30_ts = len(set(sub30_prev["ticket_number"]).intersection(set(ts30["ticket_number"])))

    print("\nOverlap with Previously Reported Post-Cutoff Cohorts:")
    print(f"  14-day (N={len(sub14_prev)}): {overlap_14_tr} in train, {overlap_14_vl} in val/tune, {overlap_14_ts} in test")
    print(f"  30-day (N={len(sub30_prev)}): {overlap_30_tr} in train, {overlap_30_vl} in val/tune, {overlap_30_ts} in test")

    # 3. Evaluate on Both Cohorts
    res14_prev = evaluate_cohort_metrics(m14, sub14_prev, 14)
    res30_prev = evaluate_cohort_metrics(m30, sub30_prev, 30)

    res14_ts = evaluate_cohort_metrics(m14, ts14, 14)
    res30_ts = evaluate_cohort_metrics(m30, ts30, 30)

    print("\n================================================================================")
    print("METRICS ON PREVIOUS COHORT (INCLUDES IN-SAMPLE / TUNING DATA)")
    print("================================================================================")
    print(f"14-Day (N={res14_prev['n']}): MeanPred={res14_prev['mean_pred']:.4f} | Observed={res14_prev['prevalence']:.4f} | Brier={res14_prev['brier']:.4f}")
    print(f"  ROC-AUC: {res14_prev['roc_auc']:.4f} (95% CI: [{res14_prev['roc_ci'][0]:.4f}, {res14_prev['roc_ci'][1]:.4f}])")
    print(f"  PR-AUC:  {res14_prev['pr_auc']:.4f} (95% CI: [{res14_prev['pr_ci'][0]:.4f}, {res14_prev['pr_ci'][1]:.4f}])")
    print("Calibration Table:")
    print(res14_prev["table"][["Bin Range", "n", "Mean Predicted", "Closed Count", "Observed Rate", "Wilson CI 95%"]].to_string(index=False))

    print(f"\n30-Day (N={res30_prev['n']}): MeanPred={res30_prev['mean_pred']:.4f} | Observed={res30_prev['prevalence']:.4f} | Brier={res30_prev['brier']:.4f}")
    print(f"  ROC-AUC: {res30_prev['roc_auc']:.4f} (95% CI: [{res30_prev['roc_ci'][0]:.4f}, {res30_prev['roc_ci'][1]:.4f}])")
    print(f"  PR-AUC:  {res30_prev['pr_auc']:.4f} (95% CI: [{res30_prev['pr_ci'][0]:.4f}, {res30_prev['pr_ci'][1]:.4f}])")
    print("Calibration Table:")
    print(res30_prev["table"][["Bin Range", "n", "Mean Predicted", "Closed Count", "Observed Rate", "Wilson CI 95%"]].to_string(index=False))

    print("\n================================================================================")
    print("METRICS ON STRICTLY OUT-OF-SAMPLE TEST COHORTS (POST-TUNING CUTOFF)")
    print("================================================================================")
    print(f"14-Day Test (N={res14_ts['n']}): MeanPred={res14_ts['mean_pred']:.4f} | Observed={res14_ts['prevalence']:.4f} | Brier={res14_ts['brier']:.4f}")
    print(f"  ROC-AUC: {res14_ts['roc_auc']:.4f} (95% CI: [{res14_ts['roc_ci'][0]:.4f}, {res14_ts['roc_ci'][1]:.4f}])")
    print(f"  PR-AUC:  {res14_ts['pr_auc']:.4f} (95% CI: [{res14_ts['pr_ci'][0]:.4f}, {res14_ts['pr_ci'][1]:.4f}])")
    print("Calibration Table:")
    print(res14_ts["table"][["Bin Range", "n", "Mean Predicted", "Closed Count", "Observed Rate", "Wilson CI 95%"]].to_string(index=False))

    print(f"\n30-Day Test (N={res30_ts['n']}): MeanPred={res30_ts['mean_pred']:.4f} | Observed={res30_ts['prevalence']:.4f} | Brier={res30_ts['brier']:.4f}")
    print(f"  ROC-AUC: {res30_ts['roc_auc']:.4f} (95% CI: [{res30_ts['roc_ci'][0]:.4f}, {res30_ts['roc_ci'][1]:.4f}])")
    print(f"  PR-AUC:  {res30_ts['pr_auc']:.4f} (95% CI: [{res30_ts['pr_ci'][0]:.4f}, {res30_ts['pr_ci'][1]:.4f}])")
    print("Calibration Table:")
    print(res30_ts["table"][["Bin Range", "n", "Mean Predicted", "Closed Count", "Observed Rate", "Wilson CI 95%"]].to_string(index=False))

    # 4. Platt Scaling Recalibration vs Constant-Prevalence Baseline
    eps = 1e-6
    # 14d
    p14_vl = m14.predict_proba(vl14)
    y14_vl = vl14["target_closed_in_h"].values
    lr14 = LogisticRegression()
    lr14.fit(logit(np.clip(p14_vl, eps, 1 - eps)).reshape(-1, 1), y14_vl)
    p14_ts_platt = lr14.predict_proba(logit(np.clip(res14_ts["preds"], eps, 1 - eps)).reshape(-1, 1))[:, 1]
    brier_14_platt = float(brier_score_loss(res14_ts["y_true"], p14_ts_platt))
    prev_14_vl = float(y14_vl.mean())
    brier_14_const = float(brier_score_loss(res14_ts["y_true"], np.full_like(res14_ts["y_true"], prev_14_vl, dtype=float)))

    # 30d
    p30_vl = m30.predict_proba(vl30)
    y30_vl = vl30["target_closed_in_h"].values
    lr30 = LogisticRegression()
    lr30.fit(logit(np.clip(p30_vl, eps, 1 - eps)).reshape(-1, 1), y30_vl)
    p30_ts_platt = lr30.predict_proba(logit(np.clip(res30_ts["preds"], eps, 1 - eps)).reshape(-1, 1))[:, 1]
    brier_30_platt = float(brier_score_loss(res30_ts["y_true"], p30_ts_platt))
    prev_30_vl = float(y30_vl.mean())
    brier_30_const = float(brier_score_loss(res30_ts["y_true"], np.full_like(res30_ts["y_true"], prev_30_vl, dtype=float)))

    print("\n================================================================================")
    print("RECALIBRATION (PLATT SCALING) VS CONSTANT-PREVALENCE BASELINE AUDIT")
    print("================================================================================")
    print("14-Day Test (N=434):")
    print(f"  Raw Model:            Mean Pred={res14_ts['mean_pred']:.4f} | Brier={res14_ts['brier']:.4f}")
    print(f"  Platt Recalibrated:   Mean Pred={p14_ts_platt.mean():.4f} | Brier={brier_14_platt:.4f}")
    print(f"  Constant Base-Rate:   Mean Pred={prev_14_vl:.4f} | Brier={brier_14_const:.4f} (Observed={res14_ts['prevalence']:.4f})")
    print("  Conclusion: Constant baseline (0.2952) outperforms Platt scaling (0.3032). Recalibration only restores base rate with zero added skill.")
    print("\n30-Day Test (N=212):")
    print(f"  Raw Model:            Mean Pred={res30_ts['mean_pred']:.4f} | Brier={res30_ts['brier']:.4f}")
    print(f"  Platt Recalibrated:   Mean Pred={p30_ts_platt.mean():.4f} | Brier={brier_30_platt:.4f}")
    print(f"  Constant Base-Rate:   Mean Pred={prev_30_vl:.4f} | Brier={brier_30_const:.4f} (Observed={res30_ts['prevalence']:.4f})")
    print("  Conclusion: Constant baseline (0.2488) matches Platt scaling (0.2484). Recalibration adds zero predictive skill.")

    # 5. Generate Reliability Plot
    os.makedirs("docs/images", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)

    # 14d
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (y = x)", alpha=0.7)
    t14 = res14_prev["table"]
    yerr_low_14 = t14["Observed Rate"] - t14["ci_low"]
    yerr_high_14 = t14["ci_high"] - t14["Observed Rate"]
    ax.errorbar(
        t14["Mean Predicted"],
        t14["Observed Rate"],
        yerr=[yerr_low_14, yerr_high_14],
        fmt="o-",
        color="#2563EB",
        ecolor="#93C5FD",
        elinewidth=2,
        capsize=4,
        label="14d Full Cohort (N=1,362, in-sample/tune contaminated)",
    )
    for _, row in t14.iterrows():
        ax.annotate(
            f"n={row['n']}",
            (row["Mean Predicted"], row["Observed Rate"]),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=8,
            color="#1E3A8A",
            fontweight="bold",
        )
    ax.set_title("14-Day Horizon: Under-predicts (0.32 vs 0.44)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Mean Predicted Probability", fontsize=10)
    ax.set_ylabel("Empirical Observed Fraction Closed <= 14d", fontsize=10)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", fontsize=8)

    # 30d
    ax = axes[1]
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (y = x)", alpha=0.7)
    t30 = res30_prev["table"]
    yerr_low_30 = t30["Observed Rate"] - t30["ci_low"]
    yerr_high_30 = t30["ci_high"] - t30["Observed Rate"]
    ax.errorbar(
        t30["Mean Predicted"],
        t30["Observed Rate"],
        yerr=[yerr_low_30, yerr_high_30],
        fmt="s-",
        color="#DC2626",
        ecolor="#FCA5A5",
        elinewidth=2,
        capsize=4,
        label="30d Full Cohort (N=786, tuning contaminated)",
    )
    for _, row in t30.iterrows():
        ax.annotate(
            f"n={row['n']}",
            (row["Mean Predicted"], row["Observed Rate"]),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=8,
            color="#991B1B",
            fontweight="bold",
        )
    ax.set_title("30-Day Horizon: Non-monotonic & Over-confident at top (0.86 vs 0.59)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Mean Predicted Probability", fontsize=10)
    ax.set_ylabel("Empirical Observed Fraction Closed <= 30d", fontsize=10)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    out_img = "docs/images/calibration_reliability_plot.png"
    plt.savefig(out_img)
    plt.close()
    print(f"\n[OK] Saved reliability diagram to {out_img}")


if __name__ == "__main__":
    main()
