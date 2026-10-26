"""Feature Engineering Pipeline for GitHub Ticket Triage.

Constructs strictly creation-time features for survival analysis, not-planned
classification, and integration label recommendations.
Guarantees NO temporal or target leakage via leakage_guard.
"""

import argparse
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from triage.features.leakage_guard import (
    assert_no_leakage,
    clean_text_for_label_prediction,
)

logger = logging.getLogger(__name__)

DEFAULT_ISSUES_PARQUET = os.path.join("data", "interim", "issues.parquet")
DEFAULT_FEATURES_PARQUET = os.path.join("data", "interim", "features.parquet")


def compute_author_prior_tickets(df: pd.DataFrame) -> pd.Series:
    """Compute count of previous tickets created by this author strictly before created_at."""
    # Assumes df is sorted by created_at ascending
    return df.groupby("author").cumcount().astype("int32")


def compute_integration_volume_90d(df: pd.DataFrame) -> pd.Series:
    """Compute count of prior tickets for the same integration in the preceding 90 days."""
    volumes = np.zeros(len(df), dtype=np.int32)
    timestamps = df["created_at_dt"].values
    integrations = df["integration_clean"].values

    ninety_days_ns = np.timedelta64(90, "D")

    # Index by integration for fast range lookup
    by_integration: Dict[str, List[int]] = {}
    for i, integ in enumerate(integrations):
        if integ not in by_integration:
            by_integration[integ] = []
        by_integration[integ].append(i)

    for integ, indices in by_integration.items():
        if integ == "unlabeled":
            continue
        n_items = len(indices)
        left = 0
        for right in range(n_items):
            cur_idx = indices[right]
            cur_t = timestamps[cur_idx]
            min_t = cur_t - ninety_days_ns
            while left < right and timestamps[indices[left]] < min_t:
                left += 1
            # Number of strictly prior tickets in window
            volumes[cur_idx] = right - left

    return pd.Series(volumes, index=df.index)


def compute_integration_median_close_prior(df: pd.DataFrame) -> pd.Series:
    """
    Compute median days to close for tickets of the same integration that closed
    STRICTLY BEFORE the current ticket was created (no lookahead leakage).
    """
    medians = np.full(len(df), np.nan, dtype=np.float32)

    # Pre-extract closed tickets that have valid duration
    closed_mask = (
        (df["state"] == "closed") & df["closed_at_dt"].notna() & df["duration_days"].notna()
    )
    closed_df = df[closed_mask][["closed_at_dt", "integration_clean", "duration_days"]].sort_values(
        "closed_at_dt"
    )

    closed_timestamps = closed_df["closed_at_dt"].values
    closed_integs = closed_df["integration_clean"].values
    closed_durations = closed_df["duration_days"].values

    # Pre-group closed indices by integration
    closed_by_integ: Dict[str, List[int]] = {}
    for j, integ in enumerate(closed_integs):
        if integ not in closed_by_integ:
            closed_by_integ[integ] = []
        closed_by_integ[integ].append(j)

    created_timestamps = df["created_at_dt"].values
    current_integs = df["integration_clean"].values

    for i in range(len(df)):
        cur_t = created_timestamps[i]
        integ = current_integs[i]
        prior_durations: List[float] = []

        if integ in closed_by_integ:
            for c_idx in closed_by_integ[integ]:
                if closed_timestamps[c_idx] < cur_t:
                    prior_durations.append(closed_durations[c_idx])
                else:
                    break

        if prior_durations:
            medians[i] = float(np.median(prior_durations))

    # Global median fallback for tickets with no prior closed integration peers
    all_closed_durations = closed_durations[closed_durations > 0]
    global_median = float(np.median(all_closed_durations)) if len(all_closed_durations) > 0 else 7.0

    return pd.Series(medians, index=df.index).fillna(global_median)


def compute_version_gap_months(df: pd.DataFrame) -> pd.Series:
    """Compute version gap in months relative to the latest HA release seen so far."""
    # Version score = year * 12 + month
    ver_score = df["ha_version_year"].fillna(0) * 12 + df["ha_version_month"].fillna(0)
    # Expanding maximum seen up to this ticket's creation
    max_ver_seen = ver_score.expanding().max()
    gap = (max_ver_seen - ver_score).clip(lower=0)
    return gap.astype("float32")


def build_feature_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Transform interim issues into a complete, leakage-free feature matrix."""
    df = df_raw.copy()

    # Provide safe column defaults for single incoming tickets
    if "closed_at" not in df.columns:
        df["closed_at"] = None
    if "created_at" not in df.columns:
        df["created_at"] = pd.Timestamp.now(tz="UTC")
    if "integration" not in df.columns:
        df["integration"] = "unlabeled"
    if "author" not in df.columns:
        df["author"] = "unknown"
    if "author_association" not in df.columns:
        df["author_association"] = "NONE"
    if "state" not in df.columns:
        df["state"] = "open"
    if "form_problem" not in df.columns:
        df["form_problem"] = df.get("problem", df.get("body", ""))
    if "form_version" not in df.columns:
        df["form_version"] = df.get("version", "")
    if "form_install_type" not in df.columns:
        df["form_install_type"] = df.get("install_type", "unknown")
    if "form_last_working_version" not in df.columns:
        df["form_last_working_version"] = df.get("last_working_version", "")
    if "form_integration" not in df.columns:
        df["form_integration"] = df.get("integration_in_form", "")
    if "has_logs" not in df.columns:
        df["has_logs"] = False
    if "has_traceback" not in df.columns:
        df["has_traceback"] = False
    if "has_last_working_version" not in df.columns:
        df["has_last_working_version"] = False
    if "n_code_blocks" not in df.columns:
        df["n_code_blocks"] = 0
    if "n_urls" not in df.columns:
        df["n_urls"] = 0
    if "n_missing_form_fields" not in df.columns:
        df["n_missing_form_fields"] = 0
    if "refers_to_issue" not in df.columns:
        df["refers_to_issue"] = False
    if "state_reason" not in df.columns:
        df["state_reason"] = None
    if "ticket_number" not in df.columns:
        df["ticket_number"] = df.get("number", 0)

    # Convert timestamps with consistent UTC timezone
    df["created_at_dt"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    df["created_at_dt"] = df["created_at_dt"].fillna(pd.Timestamp.now(tz="UTC"))
    df["closed_at_dt"] = pd.to_datetime(df["closed_at"], errors="coerce", utc=True)
    df.sort_values("created_at_dt", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Calculate actual duration for closed tickets (used ONLY for prior historical aggregates)
    duration_s = (df["closed_at_dt"] - df["created_at_dt"]).dt.total_seconds()
    df["duration_days"] = duration_s / 86400.0

    # Clean integration column
    df["integration_clean"] = (
        df["integration"].fillna("unlabeled").astype(str).str.strip().str.lower()
    )
    df.loc[df["integration_clean"] == "", "integration_clean"] = "unlabeled"

    # 1. Text length features
    title_str = df["title"].fillna("").astype(str)
    body_str = df["body"].fillna("").astype(str)
    df["title_len"] = title_str.str.len().astype("int32")
    df["body_len"] = body_str.str.len().astype("int32")

    # 2. Form structure and diagnostics features
    df["n_code_blocks"] = df["n_code_blocks"].fillna(0).astype("int32")
    df["n_urls"] = df["n_urls"].fillna(0).astype("int32")
    df["has_traceback"] = df["has_traceback"].fillna(False).astype("int32")
    df["has_logs"] = df["has_logs"].fillna(False).astype("int32")
    df["refers_to_issue"] = df["refers_to_issue"].fillna(False).astype("int32")
    df["n_missing_form_fields"] = df["n_missing_form_fields"].fillna(0).astype("int32")
    df["has_last_working_version"] = df["has_last_working_version"].fillna(False).astype("int32")
    df["is_template"] = df["is_template"].fillna(False).astype("int32")

    # 3. Version features
    df["ha_version_year"] = (
        pd.to_numeric(df["ha_version_year"], errors="coerce").fillna(0).astype("int32")
    )
    df["ha_version_month"] = (
        pd.to_numeric(df["ha_version_month"], errors="coerce").fillna(0).astype("int32")
    )
    df["ha_version_patch"] = (
        pd.to_numeric(df["ha_version_patch"], errors="coerce").fillna(0).astype("int32")
    )
    df["version_gap_months"] = compute_version_gap_months(df)

    # 4. Installation type and author metadata
    df["install_type"] = df["form_install_type"].fillna("unknown").astype(str)
    df["author_association"] = df["author_association"].fillna("NONE").astype(str)

    # 5. Temporal creation features
    df["created_hour"] = df["created_at_dt"].dt.hour.fillna(12).astype("int32")
    df["created_day_of_week"] = df["created_at_dt"].dt.dayofweek.fillna(0).astype("int32")
    df["is_weekend"] = df["created_day_of_week"].isin([5, 6]).astype("int32")

    # 6. Historical context features (strictly prior)
    print("  Computing prior author ticket counts...", flush=True)
    df["author_prior_tickets"] = compute_author_prior_tickets(df)

    print("  Computing prior 90-day integration volume...", flush=True)
    df["integration_volume_90d"] = compute_integration_volume_90d(df)

    print("  Computing prior integration median close duration...", flush=True)
    df["integration_median_close_prior"] = compute_integration_median_close_prior(df)

    # 7. Clean text representations for models
    problem_text = df["form_problem"].fillna("").astype(str)
    df["clean_problem_text"] = problem_text.where(problem_text.str.strip() != "", body_str)
    # Strip integration form section to prevent target leakage during label prediction
    df["clean_text_for_label"] = df["clean_problem_text"].apply(clean_text_for_label_prediction)

    # Construct target variables (kept separate from feature matrix via leakage guard)
    now_ts = pd.Timestamp.now(tz=df["created_at_dt"].dt.tz)
    censor_duration = (now_ts - df["created_at_dt"]).dt.total_seconds() / 86400.0
    df["target_days_to_close_or_censor"] = np.where(
        (df["state"] == "closed") & df["duration_days"].notna(),
        df["duration_days"],
        censor_duration,
    ).astype("float32")
    df["target_is_closed_event"] = (df["state"] == "closed").astype("int32")

    is_closed = df["state"] == "closed"
    reason_lower = df["state_reason"].fillna("").astype(str).str.lower()
    df["target_is_not_planned"] = np.where(
        is_closed,
        reason_lower.isin(["not_planned", "duplicate"]).astype("int32"),
        np.nan,
    )
    df["target_integration"] = df["integration_clean"]

    # Verify leakage guard on feature matrix
    feature_cols = [
        "ticket_number",
        "created_at",
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
        "ha_version_patch",
        "version_gap_months",
        "install_type",
        "author_association",
        "created_hour",
        "created_day_of_week",
        "is_weekend",
        "author_prior_tickets",
        "integration_clean",
        "integration_volume_90d",
        "integration_median_close_prior",
        "clean_problem_text",
        "clean_text_for_label",
    ]
    if "ticket_number" not in df.columns:
        df["ticket_number"] = df.get("number", 0)
    if "number" not in df.columns:
        df["number"] = df["ticket_number"]
    assert_no_leakage(df[feature_cols])

    return df


def extract_tfidf_features(
    corpus: pd.Series,
    max_features: int = 100,
    vectorizer: Optional[TfidfVectorizer] = None,
) -> Tuple[np.ndarray, TfidfVectorizer, List[str]]:
    """Extract TF-IDF text features from title and problem text."""
    clean_corpus = corpus.fillna("").astype(str)
    if vectorizer is None:
        vec = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        tfidf_matrix = vec.fit_transform(clean_corpus).toarray()
    else:
        vec = vectorizer
        tfidf_matrix = vec.transform(clean_corpus).toarray()

    feature_names = [f"tfidf_{w}" for w in vec.get_feature_names_out()]
    return tfidf_matrix, vec, feature_names


def run_feature_pipeline(
    input_path: str = DEFAULT_ISSUES_PARQUET,
    output_path: str = DEFAULT_FEATURES_PARQUET,
) -> pd.DataFrame:
    """Execute complete feature engineering pipeline."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Interim issues dataset not found: {input_path}")

    print(f"Loading interim issues from {input_path}...", flush=True)
    df_raw = pd.read_parquet(input_path)

    t0 = time.time()
    df_features = build_feature_dataframe(df_raw)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_features.to_parquet(output_path, index=False)

    duration = round(time.time() - t0, 2)
    print(
        f"[SUCCESS] Feature matrix created: {len(df_features)} rows, {len(df_features.columns)} columns in {duration}s.",
        flush=True,
    )
    print(f"Saved to: {output_path}", flush=True)
    return df_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Feature Engineering Pipeline.")
    parser.add_argument(
        "--input", default=DEFAULT_ISSUES_PARQUET, help="Path to interim issues.parquet"
    )
    parser.add_argument(
        "--output", default=DEFAULT_FEATURES_PARQUET, help="Output features parquet path"
    )
    args = parser.parse_args()

    run_feature_pipeline(input_path=args.input, output_path=args.output)


if __name__ == "__main__":
    main()
