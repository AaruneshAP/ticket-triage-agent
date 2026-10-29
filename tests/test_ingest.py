"""Unit tests for ingestion pipelines (filtering PRs, deduplication, bot detection, upserts)."""

from triage.ingest.backfill import issues_to_dataframe, transform_issue_to_record
from triage.ingest.comments import extract_duplicate_reference, is_bot_user
from triage.ingest.incremental import upsert_issues


def test_transform_issue_drops_pr_flag():
    raw_issue = {
        "number": 1001,
        "title": "Zigbee sensor disconnects",
        "body": "### The problem\nSensor drops offline\n### What version of Home Assistant Core has the issue?\n2026.10.1",
        "state": "closed",
        "state_reason": "completed",
        "labels": [{"name": "integration: zha"}],
        "user": {"login": "testuser", "type": "User"},
        "closed_by": {"login": "home-assistant[bot]", "type": "Bot"},
        "created_at": "2026-10-01T12:00:00Z",
        "closed_at": "2026-10-02T12:00:00Z",
        "updated_at": "2026-10-02T12:00:00Z",
    }
    record = transform_issue_to_record(raw_issue)
    assert record["number"] == 1001
    assert record["integration"] == "zha"
    assert record["closed_by_bot"] is True
    assert record["is_template"] is True
    assert record["ha_version_year"] == 2026


def test_issues_to_dataframe_deduplicates():
    items = [
        {"number": 201, "title": "First", "body": "Body", "state": "open", "labels": []},
        {
            "number": 201,
            "title": "Duplicate of first",
            "body": "Body",
            "state": "open",
            "labels": [],
        },
        {"number": 202, "title": "Second", "body": "Body", "state": "open", "labels": []},
    ]
    df = issues_to_dataframe(items)
    assert len(df) == 2
    assert list(df["number"]) == [201, 202]


def test_is_bot_user():
    assert is_bot_user({"login": "github-actions[bot]", "type": "Bot"}) is True
    assert is_bot_user({"login": "home-assistant[bot]", "type": "Bot"}) is True
    assert is_bot_user({"login": "stale-bot", "type": "User"}) is True
    assert is_bot_user({"login": "frenck", "type": "User"}) is False
    assert is_bot_user(None) is False


def test_extract_duplicate_reference():
    assert extract_duplicate_reference("Duplicate of #18234") == 18234
    assert extract_duplicate_reference("Closing as duplicate of #9981") == 9981
    assert extract_duplicate_reference("Dup of #5555.") == 5555
    assert extract_duplicate_reference("No duplicate here.") is None


def test_upsert_issues(tmp_path):
    parquet_path = tmp_path / "test_issues.parquet"
    initial_items = [
        {"number": 1, "title": "Old title 1", "body": "body 1", "state": "open", "labels": []},
        {"number": 2, "title": "Old title 2", "body": "body 2", "state": "open", "labels": []},
    ]
    df_init = issues_to_dataframe(initial_items)
    df_init.to_parquet(parquet_path, index=False)

    updated_items = [
        {
            "number": 1,
            "title": "Updated title 1",
            "body": "body 1",
            "state": "closed",
            "labels": [],
        },
        {"number": 3, "title": "New issue 3", "body": "body 3", "state": "open", "labels": []},
    ]
    df_upserted = upsert_issues(updated_items, parquet_path=str(parquet_path))

    assert len(df_upserted) == 3
    assert set(df_upserted["number"]) == {1, 2, 3}
    row1 = df_upserted[df_upserted["number"] == 1].iloc[0]
    assert row1["title"] == "Updated title 1"
    assert row1["state"] == "closed"
