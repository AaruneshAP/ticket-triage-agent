"""Evaluates 14-day and 30-day cohort closure rates and monthly/weekly trends.

Saves empirical findings to docs/results/cohort_closure.json.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES_PATH = Path("data/interim/features.parquet")
OUTPUT_PATH = Path("docs/results/cohort_closure.json")


def run_cohort_evaluation() -> dict:
    df = pd.read_parquet(FEATURES_PATH)
    df["created_at_dt"] = pd.to_datetime(df["created_at"], utc=True)

    # 14-day test cohort:
    c14_start = pd.to_datetime("2026-10-02 05:25:14+00:00")
    c14_end = pd.to_datetime("2026-10-12 05:13:28+00:00")
    cohort14 = df[(df["created_at_dt"] >= c14_start) & (df["created_at_dt"] <= c14_end)].copy()
    cohort14["closed_within_14d"] = np.where(
        (cohort14["state"] == "closed") & (cohort14["duration_days"] <= 14), 1, 0
    )

    # 30-day test cohort:
    c30_start = pd.to_datetime("2026-09-16 05:16:02+00:00")
    c30_end = pd.to_datetime("2026-09-26 04:46:35+00:00")
    cohort30 = df[(df["created_at_dt"] >= c30_start) & (df["created_at_dt"] <= c30_end)].copy()
    cohort30["closed_within_30d"] = np.where(
        (cohort30["state"] == "closed") & (cohort30["duration_days"] <= 30), 1, 0
    )

    # Monthly breakdown
    df["year_month"] = df["created_at_dt"].dt.strftime("%Y-%m")
    monthly_stats = {}
    for ym, grp in df.groupby("year_month"):
        if ym >= "2026-05":
            n_tot = len(grp)
            n_closed = int((grp["state"] == "closed").sum())
            c14_n = int(((grp["state"] == "closed") & (grp["duration_days"] <= 14)).sum())
            c30_n = int(((grp["state"] == "closed") & (grp["duration_days"] <= 30)).sum())
            monthly_stats[ym] = {
                "total_tickets": n_tot,
                "closed_total": n_closed,
                "closure_rate": round(n_closed / n_tot, 4),
                "closed_within_14d": c14_n,
                "rate_14d": round(c14_n / n_tot, 4),
                "closed_within_30d": c30_n,
                "rate_30d": round(c30_n / n_tot, 4),
            }

    results = {
        "cohort_14d": {
            "window_start": str(c14_start),
            "window_end": str(c14_end),
            "total_tickets": len(cohort14),
            "closed_within_14d": int(cohort14["closed_within_14d"].sum()),
            "rate_within_14d": round(float(cohort14["closed_within_14d"].mean()), 4),
            "overall_closed": int((cohort14["state"] == "closed").sum()),
            "overall_closed_rate": round(float((cohort14["state"] == "closed").mean()), 4),
        },
        "cohort_30d": {
            "window_start": str(c30_start),
            "window_end": str(c30_end),
            "total_tickets": len(cohort30),
            "closed_within_30d": int(cohort30["closed_within_30d"].sum()),
            "rate_within_30d": round(float(cohort30["closed_within_30d"].mean()), 4),
            "overall_closed": int((cohort30["state"] == "closed").sum()),
            "overall_closed_rate": round(float((cohort30["state"] == "closed").mean()), 4),
        },
        "monthly_closure_rates": monthly_stats,
        "note": "The root cause of closure rate variation across cohorts was not investigated.",
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Cohort evaluation results saved to {OUTPUT_PATH}")
    return results


if __name__ == "__main__":
    res = run_cohort_evaluation()
    print(json.dumps(res, indent=2))
