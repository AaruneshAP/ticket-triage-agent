"""Tests for Leakage Guard enforcement."""

import pandas as pd
import pytest

from triage.features.leakage_guard import (
    FORBIDDEN_COLUMNS,
    LeakageError,
    assert_no_leakage,
    clean_text_for_label_prediction,
    validate_feature_matrix,
)


def test_leakage_guard_rejects_forbidden_columns():
    for forbidden_col in FORBIDDEN_COLUMNS:
        df = pd.DataFrame(
            {
                "title_len": [42],
                "n_code_blocks": [1],
                forbidden_col: [100],
            }
        )
        with pytest.raises(LeakageError) as exc_info:
            assert_no_leakage(df)
        assert forbidden_col.lower() in str(exc_info.value).lower()


def test_leakage_guard_accepts_clean_creation_features():
    clean_cols = [
        "ticket_number",
        "title_len",
        "body_len",
        "n_code_blocks",
        "n_urls",
        "has_traceback",
        "refers_to_issue",
        "n_missing_form_fields",
        "has_last_working_version",
        "ha_version_year",
        "ha_version_month",
        "ha_version_patch",
        "version_gap_months",
        "install_type",
        "integration",
        "integration_volume_90d",
        "integration_median_close_prior",
        "author_association",
        "author_prior_tickets",
        "created_hour",
        "created_day_of_week",
        "is_weekend",
    ]
    df = pd.DataFrame(columns=clean_cols)
    # Should not raise
    assert_no_leakage(df)


def test_clean_text_for_label_prediction_strips_integration_field():
    raw_body = """
    ### The problem
    Integration fails to connect on reboot.

    ### What version of Home Assistant Core has the issue?
    2026.10.1

    ### Integration causing the issue
    matter

    ### Additional information
    Device is ESP32.
    """
    cleaned = clean_text_for_label_prediction(raw_body)
    assert "Integration causing the issue" not in cleaned
    assert "matter" not in cleaned
    assert "The problem" in cleaned
    assert "Integration fails to connect on reboot." in cleaned
    assert "Additional information" in cleaned


def test_validate_feature_matrix_with_target():
    df = pd.DataFrame(
        {
            "title_len": [50],
            "is_closed_without_fix": [1],
        }
    )
    # Target is allowed if explicitly specified
    validate_feature_matrix(df, expected_targets=["is_closed_without_fix"])

    # Fails if target is not declared
    with pytest.raises(LeakageError):
        validate_feature_matrix(df, expected_targets=[])
