"""Tests for Sandbox GitHub Triage Bot and Strict Repository Allow-List Enforcement (Phase 10)."""

import os

import pytest

from triage.bot.sandbox_bot import (
    SandboxGitHubClient,
    StrictAllowlistViolation,
    assert_sandbox_write_allowed,
    generate_triage_draft_and_labels,
    triage_sandbox_issue,
)


def test_allowlist_guard_blocks_home_assistant_core(monkeypatch):
    """Verify that any write targeting home-assistant/core is strictly blocked."""
    monkeypatch.setenv("GITHUB_SANDBOX_REPO", "AaruneshAP/ticket-triage-sandbox")

    with pytest.raises(StrictAllowlistViolation) as exc_info:
        assert_sandbox_write_allowed("home-assistant/core")

    assert "strictly forbidden" in str(exc_info.value)
    assert "home-assistant/core" in str(exc_info.value)


def test_allowlist_guard_blocks_arbitrary_repos(monkeypatch):
    """Verify that arbitrary repositories are blocked by the allowlist guard."""
    monkeypatch.setenv("GITHUB_SANDBOX_REPO", "AaruneshAP/ticket-triage-sandbox")

    for forbidden in ["octocat/Hello-World", "other-user/ticket-triage-sandbox", "home-assistant/core"]:
        with pytest.raises(StrictAllowlistViolation):
            assert_sandbox_write_allowed(forbidden)


def test_allowlist_guard_permits_authorized_sandbox(monkeypatch):
    """Verify that the designated sandbox repository passes the allowlist guard."""
    monkeypatch.setenv("GITHUB_SANDBOX_REPO", "AaruneshAP/ticket-triage-sandbox")
    # Should not raise
    assert_sandbox_write_allowed("AaruneshAP/ticket-triage-sandbox")
    assert_sandbox_write_allowed("aaruneshap/ticket-triage-sandbox")  # Case insensitive


def test_generate_triage_draft_incomplete_issue():
    """Verify draft generation for an incomplete one-liner issue."""
    parsed = {
        "is_template": True,
        "problem": "It does not work.",
        "integration_in_form": "mqtt",
        "has_traceback": False,
        "has_logs": False,
        "version": "core-2026.8.0",
        "missing_fields": ["logs"],
    }
    res = generate_triage_draft_and_labels(parsed, raw_body="It does not work.", raw_title="MQTT broken")

    assert res["is_incomplete"] is True
    assert "needs-more-information" in res["labels"]
    assert "integration: mqtt" in res["labels"]
    assert "[AI draft, needs review]" in res["comment_text"]
    assert "Completeness Notice" in res["comment_text"]


def test_generate_triage_draft_complete_issue():
    """Verify draft generation for a complete issue with tracebacks."""
    parsed = {
        "is_template": True,
        "problem": "Traceback when toggling switch:\nTraceback (most recent call last):\nKeyError: 'state'",
        "integration_in_form": "switch",
        "has_traceback": True,
        "has_logs": True,
        "version": "core-2026.8.0",
        "missing_fields": [],
    }
    raw_b = "Traceback (most recent call last):\nKeyError: 'state'\nHome Assistant Core 2026.8.0"
    res = generate_triage_draft_and_labels(parsed, raw_body=raw_b, raw_title="Switch traceback error")

    assert res["is_incomplete"] is False
    assert "needs-more-information" not in res["labels"]
    assert "integration: switch" in res["labels"]
    assert "[AI draft, needs review]" in res["comment_text"]
    assert "Python traceback detected" in res["comment_text"]


@pytest.mark.integration
def test_live_sandbox_issue_lifecycle():
    """Live integration test: creates a test issue in sandbox repo, triages it, asserts comment, and closes it."""
    repo = os.getenv("GITHUB_SANDBOX_REPO") or os.getenv("SANDBOX_REPO")
    token = os.getenv("GITHUB_SANDBOX_TOKEN") or os.getenv("SANDBOX_TOKEN")

    if not (repo and token):
        pytest.skip("GITHUB_SANDBOX_REPO / TOKEN not set in environment; skipping live test.")

    client = SandboxGitHubClient(repo=repo, token=token)

    # 1. Create a test issue
    issue_body = (
        "### The problem\n"
        "Triage bot end-to-end integration test issue.\n\n"
        "### What version of Home Assistant Core has the issue?\n"
        "core-2026.8.0\n\n"
        "### Integration causing the issue\n"
        "switch\n\n"
        "### Traceback/logs\n"
        "```python\nTraceback (most recent call last):\n  File 'test.py', line 1, in <module>\nKeyError: 'switch_test'\n```"
    )
    issue = client.create_issue(title="[TEST] Automated Triage Bot Verification", body=issue_body)
    issue_num = issue["number"]
    assert issue_num > 0

    try:
        # 2. Run triage on the issue
        res = triage_sandbox_issue(issue_number=issue_num, dry_run=False)
        assert res["status"] == "live_triage_success"
        assert res["comment_id"] is not None
        assert "integration: switch" in res["labels_to_apply"]

        # 3. Verify issue state
        fetched = client.get_issue(issue_num)
        assert fetched["comments"] >= 1
    finally:
        # 4. Clean up: Close the test issue
        client.close_issue(issue_num)
        closed = client.get_issue(issue_num)
        assert closed["state"] == "closed"
