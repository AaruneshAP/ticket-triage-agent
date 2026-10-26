"""Leakage Guard: Enforce strict temporal and target leakage boundaries.

Rules:
1. Only information known at ticket creation timestamp (created_at) is allowed in features.
2. Forbidden columns: comments, comment counts, assignee, state, state_reason,
   closed_at, closed_by, closed_by_bot, updated_at, edits, and target variables.
3. For label prediction, the 'Integration causing the issue' field is bot-copied
   and must be stripped from raw text.
"""

import re
from typing import Collection, Iterable, List, Set, Union

import pandas as pd

FORBIDDEN_COLUMNS: Set[str] = {
    "comments",
    "comments_count",
    "comment_count",
    "assignee",
    "state",
    "state_reason",
    "closed_at",
    "closed_by",
    "closed_by_bot",
    "updated_at",
    "edits",
    "edit_count",
    "days_to_close_or_censor",
    "is_closed_event",
    "is_closed_without_fix",
}


class LeakageError(ValueError):
    """Raised when forbidden future or target information is detected in feature columns."""

    pass


def assert_no_leakage(columns_or_df: Union[pd.DataFrame, Collection[str]]) -> None:
    """Assert that none of the forbidden columns exist in the provided feature set."""
    if isinstance(columns_or_df, pd.DataFrame):
        cols = set(columns_or_df.columns)
    elif isinstance(columns_or_df, (list, tuple, set, pd.Index)):
        cols = set(columns_or_df)
    else:
        cols = set(list(columns_or_df))

    # Normalize lowercase column names for comparison
    col_map = {str(c).lower().strip(): str(c) for c in cols}
    leaked: List[str] = []
    for f in FORBIDDEN_COLUMNS:
        if f.lower() in col_map:
            leaked.append(col_map[f.lower()])

    if leaked:
        raise LeakageError(
            f"TEMPORAL LEAKAGE VIOLATION: Forbidden post-creation or target column(s) detected: {sorted(leaked)}.\n"
            f"Features must strictly only contain data known at ticket creation time."
        )


def clean_text_for_label_prediction(text: str) -> str:
    """
    Remove 'Integration causing the issue' section from form text.
    The Home Assistant bot applies the integration label directly copied from this field.
    Predicting integration label using this field represents severe target leakage.
    """
    if not text or not isinstance(text, str):
        return ""

    # Remove heading and content up to the next heading or end of string
    pattern = r"###\s*Integration causing the issue\s*[\r\n]+.*?(?=###|\Z)"
    cleaned = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)
    return cleaned.strip()


def validate_feature_matrix(df: pd.DataFrame, expected_targets: Iterable[str] = ()) -> None:
    """Validate a feature matrix, allowing specified explicit target columns during training."""
    allowed_targets = {t.lower() for t in expected_targets}
    feature_cols = [c for c in df.columns if str(c).lower() not in allowed_targets]
    assert_no_leakage(feature_cols)
