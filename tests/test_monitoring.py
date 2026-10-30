"""Tests for MLOps Monitoring, Drift Detection, Performance Tracking, and Retraining."""

import numpy as np
import pandas as pd
import pytest

from triage.monitoring.drift import calculate_ks_test, calculate_psi, format_drift_table
from triage.monitoring.performance import evaluate_cohort_performance, format_performance_table
from triage.monitoring.retrain import evaluate_retraining_triggers, format_retraining_table


def test_psi_identical_distributions():
    np.random.seed(42)
    expected = np.random.normal(loc=10.0, scale=2.0, size=1000)
    actual = np.random.normal(loc=10.0, scale=2.0, size=1000)

    psi = calculate_psi(expected, actual, num_bins=10)
    assert psi < 0.10, f"Identical distributions should have low PSI, got {psi}"


def test_psi_shifted_distributions():
    np.random.seed(42)
    expected = np.random.normal(loc=10.0, scale=2.0, size=1000)
    actual = np.random.normal(loc=25.0, scale=2.0, size=1000)

    psi = calculate_psi(expected, actual, num_bins=10)
    assert psi >= 0.25, f"Severely shifted distributions should have high PSI, got {psi}"


def test_psi_binary_features_known_values():
    """Verify PSI calculates non-zero, exact values on binary features without collapsing to a single bin."""
    # Test 1: has_logs shifting from 0.44 to 0.56
    # 44 ones, 56 zeros vs 56 ones, 44 zeros
    expected_logs = np.array([1] * 44 + [0] * 56, dtype=float)
    actual_logs = np.array([1] * 56 + [0] * 44, dtype=float)

    psi_logs = calculate_psi(expected_logs, actual_logs)
    assert psi_logs > 0.0, f"Binary PSI for has_logs must not be 0.0000, got {psi_logs}"
    assert np.isclose(psi_logs, 0.0579, atol=0.005), f"Expected PSI ~0.0579 for has_logs shift, got {psi_logs}"

    # Test 2: has_traceback shifting from 0.28 to 0.33
    # 28 ones, 72 zeros vs 33 ones, 67 zeros
    expected_tb = np.array([1] * 28 + [0] * 72, dtype=float)
    actual_tb = np.array([1] * 33 + [0] * 67, dtype=float)

    psi_tb = calculate_psi(expected_tb, actual_tb)
    assert psi_tb > 0.0, f"Binary PSI for has_traceback must not be 0.0000, got {psi_tb}"
    assert np.isclose(psi_tb, 0.0118, atol=0.003), f"Expected PSI ~0.0118 for has_traceback shift, got {psi_tb}"


def test_ks_test_identical_distributions():
    np.random.seed(42)
    expected = np.random.normal(loc=5.0, scale=1.0, size=500)
    actual = np.random.normal(loc=5.0, scale=1.0, size=500)

    stat, p_val = calculate_ks_test(expected, actual)
    assert stat < 0.15, f"KS stat should be small for similar distributions, got {stat}"
    assert p_val > 0.01, f"p-value should indicate lack of significant difference, got {p_val}"


def test_format_drift_table():
    summary = {
        "features_analyzed": {
            "title_len": {
                "ref_mean": 75.0,
                "curr_mean": 72.0,
                "psi": 0.1437,
                "ks_statistic": 0.0926,
                "p_value": 0.0007,
                "alert_level": "warning",
            }
        }
    }
    table = format_drift_table(summary)
    assert "title_len" in table
    assert "0.1437" in table
    assert "WARNING" in table


def test_evaluate_retraining_triggers():
    drift_summary = {
        "reference_records": 1000,
        "current_records": 600,
        "features_analyzed": {
            "body_len": {"psi": 0.32, "alert_level": "alert"},
            "title_len": {"psi": 0.05, "alert_level": "normal"},
        },
        "overall_drift_alert": True,
    }
    decision = evaluate_retraining_triggers(drift_summary=drift_summary)
    assert decision["should_retrain"] is True
    assert decision["recommended_action"] == "TRIGGER_RETRAINING_PIPELINE"
    trigger_types = [t["trigger_type"] for t in decision["triggers"]]
    assert "feature_drift" in trigger_types
    assert "volume_threshold" in trigger_types


def test_evaluate_cohort_performance_mock():
    # Test cohort performance evaluation function with mock dataframe
    df = pd.DataFrame({
        "target_days_to_close_or_censor": [2.0, 5.0, 10.0, 15.0, 20.0, 30.0],
        "target_is_closed_event": [1, 1, 1, 1, 1, 1],
        "target_is_not_planned": [0, 0, 1, 0, 1, 0],
        "target_integration": ["mqtt", "zha", "mqtt", "hue", "zha", "mqtt"],
        "clean_text_for_label": ["mqtt text", "zha text", "mqtt text", "hue text", "zha text", "mqtt text"],
    })

    class MockModel:
        def predict(self, X):
            return np.array([2.0, 5.0, 10.0, 15.0, 20.0, 30.0])

        def predict_proba(self, X):
            return np.array([0.1, 0.2, 0.8, 0.1, 0.9, 0.2])

    metrics = evaluate_cohort_performance(df, surv_model=MockModel(), np_model=MockModel())
    assert metrics["n_tickets"] == 6
    assert metrics["survival_c_index"] is not None
    assert metrics["not_planned_pr_auc"] is not None
    assert metrics["not_planned_brier"] is not None


def test_format_performance_table():
    summary = {
        "cohort_metrics": [
            {
                "window_start": "2026-10-01",
                "window_end": "2026-10-15",
                "n_tickets": 100,
                "survival_c_index": 0.5821,
                "not_planned_pr_auc": 0.6214,
                "not_planned_brier": 0.1542,
                "label_accuracy": 0.7200,
            }
        ]
    }
    table = format_performance_table(summary)
    assert "2026-10-01 to 2026-10-15" in table
    assert "0.5821" in table
    assert "0.6214" in table


def test_format_retraining_table():
    summary = {
        "models_evaluated": {
            "survival_aft": {
                "primary_metric": "concordance_index",
                "champion_metric": 0.5426,
                "candidate_metric": 0.5426,
                "delta": 0.0,
                "margin_required": 0.01,
                "promoted": False,
            }
        }
    }
    table = format_retraining_table(summary)
    assert "survival_aft" in table
    assert "concordance_index" in table
    assert "NO (Kept Champion)" in table


def test_retrain_decision_promotion_consistency():
    """Verify that any row in fct_retrain_decisions with is_promoted=1 must have final_decision='PROMOTE'.

    Fails if is_promoted=1 and final_decision=REJECT.
    """
    from scripts.ops.publish_gold_monitoring_tables import prepare_gold_tables

    _, _, df_retrain = prepare_gold_tables()
    assert not df_retrain.empty, "df_retrain should not be empty"
    for _, row in df_retrain.iterrows():
        if row["is_promoted"] == 1:
            assert row["final_decision"] == "PROMOTE", (
                f"Row {row['path_name']} has is_promoted=1 but final_decision={row['final_decision']}"
            )
            assert row["final_decision"] != "REJECT", (
                f"Row {row['path_name']} has is_promoted=1 and final_decision=REJECT"
            )
        elif row["is_promoted"] == 0:
            assert row["final_decision"] == "REJECT", (
                f"Row {row['path_name']} has is_promoted=0 but final_decision={row['final_decision']}"
            )

    # Invariant assertion check: verify that a row with is_promoted=1 and final_decision=REJECT is strictly rejected
    def check_promotion_consistency(is_promoted: int, final_decision: str):
        if is_promoted == 1 and final_decision == "REJECT":
            raise AssertionError("Integrity violation: is_promoted=1 and final_decision=REJECT")

    with pytest.raises(AssertionError, match="Integrity violation: is_promoted=1 and final_decision=REJECT"):
        check_promotion_consistency(is_promoted=1, final_decision="REJECT")

