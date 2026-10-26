"""Time-Based Dataset Splitting for Triage Models.

Enforces strictly chronological train/validation/test splits:
- Train: tickets created before T-60 days
- Validation: tickets created between T-60 days and T-30 days
- Test: tickets created in the last 30 days (T-30 to T)
"""

from typing import Tuple

import pandas as pd


def get_time_based_splits(
    df: pd.DataFrame,
    timestamp_col: str = "created_at",
    val_days: int = 30,
    test_days: int = 30,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset chronologically based on ticket creation timestamp."""
    df_sorted = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_sorted[timestamp_col]):
        df_sorted["split_dt"] = pd.to_datetime(df_sorted[timestamp_col], errors="coerce")
    else:
        df_sorted["split_dt"] = df_sorted[timestamp_col]

    df_sorted.sort_values("split_dt", inplace=True)
    df_sorted.reset_index(drop=True, inplace=True)

    max_dt = df_sorted["split_dt"].max()
    t_test_start = max_dt - pd.DateOffset(days=test_days)
    t_val_start = t_test_start - pd.DateOffset(days=val_days)

    train_df = df_sorted[df_sorted["split_dt"] < t_val_start].copy()
    val_df = df_sorted[
        (df_sorted["split_dt"] >= t_val_start) & (df_sorted["split_dt"] < t_test_start)
    ].copy()
    test_df = df_sorted[df_sorted["split_dt"] >= t_test_start].copy()

    # Drop temporary column if created
    for split_data in (train_df, val_df, test_df):
        if "split_dt" in split_data.columns and timestamp_col != "split_dt":
            split_data.drop(columns=["split_dt"], inplace=True)

    return train_df, val_df, test_df
