import os
from unittest.mock import MagicMock

import pandas as pd

from triage.warehouse.loader import (
    filter_existing_records,
    is_databricks_configured,
    load_dataframe_to_databricks,
    run_warehouse_load,
)

FIXTURE_ISSUES = os.path.join(os.path.dirname(__file__), "fixtures", "sample_issues.parquet")
FIXTURE_COMMENTS = os.path.join(os.path.dirname(__file__), "fixtures", "sample_comments.parquet")


def test_is_databricks_configured_with_empty_env(monkeypatch):
    monkeypatch.setenv("DATABRICKS_HOST", "")
    monkeypatch.setenv("DATABRICKS_TOKEN", "")
    monkeypatch.setenv("DATABRICKS_HTTP_PATH", "")
    assert is_databricks_configured() is False


def test_warehouse_load_dry_run():
    # Validates fixture parquet files against schema without requiring active Databricks connection
    success = run_warehouse_load(
        issues_path=FIXTURE_ISSUES,
        comments_path=FIXTURE_COMMENTS,
        dry_run=True,
    )
    assert success is True


def test_filter_existing_records_comments():
    df = pd.DataFrame([
        {"comment_id": 101, "issue_number": 1, "body": "first"},
        {"comment_id": 102, "issue_number": 1, "body": "second"},
        {"comment_id": 103, "issue_number": 2, "body": "third"},
    ])
    # 101 and 102 already exist
    existing = {101, "102"}
    filtered = filter_existing_records(df, "comment_id", existing)
    assert len(filtered) == 1
    assert filtered.iloc[0]["comment_id"] == 103


def test_filter_existing_records_issues():
    df = pd.DataFrame([
        {"ticket_number": 180001, "title": "Issue A"},
        {"ticket_number": 180002, "title": "Issue B"},
    ])
    existing = {180001}
    filtered = filter_existing_records(df, "ticket_number", existing)
    assert len(filtered) == 1
    assert filtered.iloc[0]["ticket_number"] == 180002


def test_load_dataframe_dedupe_on_append():
    mock_cursor = MagicMock()
    # Mock cursor fetchall returning existing comment_ids 101 and 102
    mock_cursor.fetchall.return_value = [(101,), (102,)]

    df = pd.DataFrame([
        {"comment_id": 101, "issue_number": 1, "body": "repeat"},
        {"comment_id": 102, "issue_number": 1, "body": "repeat2"},
        {"comment_id": 103, "issue_number": 2, "body": "new_comment"},
    ])

    ok, num_batches, total_rows = load_dataframe_to_databricks(
        cursor=mock_cursor,
        df=df,
        table_name="raw_comments",
        columns=["comment_id", "issue_number", "body"],
        overwrite=False,
    )

    assert ok is True
    # Only 1 new row should be inserted
    assert total_rows == 1
    # Verify cursor executed SELECT DISTINCT and INSERT INTO
    calls = [c[0][0] for c in mock_cursor.execute.call_args_list]
    assert any("SELECT DISTINCT comment_id" in sql for sql in calls)
    assert any("INSERT INTO" in sql and "103" in sql for sql in calls)
