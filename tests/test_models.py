"""Unit tests for ML models, survival analysis, and forecasting."""

import numpy as np
import pandas as pd
import pytest

from triage.models.forecaster import WeeklyVolumeForecaster, prepare_weekly_series
from triage.models.labeler import IntegrationLabelRecommender
from triage.models.not_planned import NotPlannedClassifier
from triage.models.splits import get_time_based_splits
from triage.models.survival import MedianByIntegrationBaseline, XGBoostSurvivalAFT


@pytest.fixture
def sample_feature_df():
    n_rows = 60
    dates = pd.date_range("2026-05-01", periods=n_rows, freq="2D")
    df = pd.DataFrame(
        {
            "ticket_number": list(range(1, n_rows + 1)),
            "created_at": dates,
            "title_len": np.random.randint(10, 80, size=n_rows),
            "body_len": np.random.randint(50, 500, size=n_rows),
            "n_code_blocks": np.random.randint(0, 3, size=n_rows),
            "n_urls": np.random.randint(0, 2, size=n_rows),
            "has_traceback": np.random.binomial(1, 0.3, size=n_rows),
            "has_logs": np.random.binomial(1, 0.4, size=n_rows),
            "refers_to_issue": np.random.binomial(1, 0.2, size=n_rows),
            "n_missing_form_fields": np.random.randint(0, 4, size=n_rows),
            "has_last_working_version": np.random.binomial(1, 0.3, size=n_rows),
            "is_template": np.random.binomial(1, 0.8, size=n_rows),
            "ha_version_year": [2026] * n_rows,
            "ha_version_month": np.random.randint(5, 11, size=n_rows),
            "version_gap_months": np.random.uniform(0, 2, size=n_rows),
            "author_prior_tickets": np.random.randint(0, 5, size=n_rows),
            "created_hour": np.random.randint(0, 24, size=n_rows),
            "created_day_of_week": np.random.randint(0, 7, size=n_rows),
            "is_weekend": np.random.binomial(1, 0.28, size=n_rows),
            "integration_clean": ["matter", "mqtt", "hue", "unlabeled"] * (n_rows // 4),
            "integration_volume_90d": np.random.randint(1, 20, size=n_rows),
            "integration_median_close_prior": np.random.uniform(1.0, 10.0, size=n_rows),
            "clean_problem_text": ["device failed to turn on when triggered"] * n_rows,
            "clean_text_for_label": ["device failed to turn on when triggered"] * n_rows,
            "target_days_to_close_or_censor": np.random.uniform(0.5, 30.0, size=n_rows),
            "target_is_closed_event": np.random.binomial(1, 0.7, size=n_rows),
            "target_is_not_planned": np.random.binomial(1, 0.2, size=n_rows),
        }
    )
    return df


def test_chronological_splits(sample_feature_df):
    train, val, test = get_time_based_splits(
        sample_feature_df, timestamp_col="created_at", val_days=20, test_days=20
    )
    assert len(train) > 0
    assert len(val) > 0
    assert len(test) > 0
    assert train["created_at"].max() <= val["created_at"].min()
    assert val["created_at"].max() <= test["created_at"].min()


def test_survival_baseline_and_aft(sample_feature_df):
    baseline = MedianByIntegrationBaseline().fit(sample_feature_df)
    preds_base = baseline.predict(sample_feature_df)
    assert len(preds_base) == len(sample_feature_df)
    assert (preds_base > 0).all()

    aft = XGBoostSurvivalAFT(max_depth=2, n_estimators=10)
    aft.fit(
        X=sample_feature_df[
            ["title_len", "body_len", "integration_volume_90d", "integration_median_close_prior"]
        ],
        duration=sample_feature_df["target_days_to_close_or_censor"].values,
        event=sample_feature_df["target_is_closed_event"].values,
    )
    p50, p90 = aft.predict_percentiles(
        sample_feature_df[
            ["title_len", "body_len", "integration_volume_90d", "integration_median_close_prior"]
        ]
    )
    assert len(p50) == len(sample_feature_df)
    assert (p50 > 0).all()
    assert (p90 >= p50).all()


def test_not_planned_classifier_and_shap(sample_feature_df):
    clf = NotPlannedClassifier(max_features_tfidf=10, max_depth=2, n_estimators=10)
    clf.fit(sample_feature_df, sample_feature_df["target_is_not_planned"].values)

    probs = clf.predict_proba(sample_feature_df)
    assert len(probs) == len(sample_feature_df)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()

    # Explain first ticket
    explanations = clf.explain_ticket(sample_feature_df.iloc[[0]], top_k=3)
    assert len(explanations) == 3
    assert "feature" in explanations[0]
    assert "impact" in explanations[0]
    assert "direction" in explanations[0]


def test_label_recommender(sample_feature_df):
    labeler = IntegrationLabelRecommender(top_n_labels=3, max_features=50)
    labeler.fit(sample_feature_df)

    texts = sample_feature_df["clean_text_for_label"].head(5)
    labels, confs, candidates = labeler.predict_with_confidence(texts)
    assert len(labels) == 5
    assert len(confs) == 5
    assert (np.array(confs) >= 0.0).all() and (np.array(confs) <= 1.0).all()
    assert len(candidates[0]) == 3


def test_weekly_volume_forecaster(sample_feature_df):
    series = prepare_weekly_series(sample_feature_df, timestamp_col="created_at")
    assert len(series) > 0

    forecaster = WeeklyVolumeForecaster()
    ets_m, naive_m, mean_m, preds = forecaster.fit_and_evaluate(series, test_weeks=2)
    assert "mae" in ets_m
    assert "rmse" in ets_m
    assert len(preds) == 2

    future_preds = forecaster.forecast_future(steps=4)
    assert len(future_preds) == 4
