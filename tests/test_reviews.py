"""Tests for Review feedback persistence layer."""

import os

from triage.warehouse.reviews import load_recent_reviews, save_review


def test_save_and_load_review_local(tmp_path, monkeypatch):
    test_parquet = str(tmp_path / "reviews_test.parquet")
    monkeypatch.setattr("triage.warehouse.reviews.LOCAL_REVIEWS_PARQUET", test_parquet)
    monkeypatch.setattr("triage.warehouse.reviews.is_databricks_configured", lambda: False)

    rec = save_review(
        ticket_number=187117,
        decision="APPROVED",
        approved_draft="[AI draft, needs review] Thanks for reporting.",
        reviewer_notes="LGTM",
        suggested_label="shelly",
        reviewer="maintainer_test",
    )

    assert rec["ticket_number"] == 187117
    assert rec["decision"] == "APPROVED"
    assert rec["databricks_status"] == "skipped_not_configured"
    assert os.path.exists(test_parquet)

    df = load_recent_reviews(limit=10)
    assert len(df) == 1
    assert df.iloc[0]["ticket_number"] == 187117
    assert df.iloc[0]["decision"] == "APPROVED"
    assert df.iloc[0]["reviewer"] == "maintainer_test"
