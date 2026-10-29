"""Unit tests for LangGraph agent, completeness rules, and triage pipeline."""

from triage.agent.completeness import (
    check_ticket_completeness,
    check_ticket_completeness_rules,
    is_bug_report_template,
    is_developer_or_bot_ticket,
)
from triage.agent.llm import filter_similar_tickets, generate_template_draft


def test_completeness_missing_version_and_logs():
    issue = {
        "title": "Shelly switch randomly disconnects",
        "body": "The switch disconnects when I turn it on.",
        "form_version": "_No response_",
        "has_logs": False,
        "has_traceback": False,
        "n_code_blocks": 0,
        "has_last_working_version": False,
    }
    missing = check_ticket_completeness(issue)
    assert any("version" in m.lower() for m in missing)
    assert any("logs" in m.lower() or "traceback" in m.lower() for m in missing)


def test_completeness_last_working_version_not_essential():
    # Ticket 178184: Last working version is optional and non-essential for triage
    issue = {
        "title": "Regression after upgrading: Z-Wave climate entities disappeared",
        "body": "This worked perfectly until the recent upgrade. Now it stopped working in the new build.",
        "form_version": "2026.10.0",
        "ha_version_year": 2026,
        "has_logs": True,
        "has_traceback": False,
        "n_code_blocks": 1,
        "has_last_working_version": False,
        "form_last_working_version": "",
    }
    missing = check_ticket_completeness_rules(issue)
    assert not any("last working version" in m.lower() for m in missing)


def test_completeness_system_information_version_not_flagged():
    # Tickets 177215 and 177224: Core version is inside ## System Information table
    sys_info = (
        "## System Information\n\n"
        "version | core-2026.7.4\n"
        "-- | --\n"
        "installation_type | Home Assistant Container\n"
        "dev | false\n"
    )
    issue = {
        "title": "Spotify playlist crashes integration",
        "form_problem": "When playing specific playlists, integration crashes with invalid value None.",
        "form_version": sys_info,
        "form_install_type": "Home Assistant Container",
        "has_logs": True,
        "has_traceback": True,
    }
    missing = check_ticket_completeness_rules(issue)
    assert not any("core version" in m.lower() for m in missing)


def test_completeness_short_text_one_liner_flagged():
    # Tickets 176864, 177225, 177617, 178141: brief one-liner with no logs/traceback
    issue = {
        "title": "Diivoo 2-zone water timer unstable",
        "form_problem": "The Diivoo 2-zone water timer is unstable. The integration frequently declares the device offline.",
        "form_version": "core-2026.6.3",
        "form_install_type": "Home Assistant OS",
        "has_logs": False,
        "has_traceback": False,
    }
    missing = check_ticket_completeness_rules(issue)
    assert any("too brief and lacks error logs" in m.lower() for m in missing)


def test_template_draft_generation_rules():
    missing_items = ["Diagnostic logs or Python traceback", "Exact Home Assistant Core version"]
    similar_issues = [
        {
            "ticket_number": 182001,
            "title": "Shelly Gen2 disconnects after reconnect",
            "similarity_score": 0.85,
            "integration": "shelly",
            "state": "closed",
        },
        {
            "ticket_number": 181950,
            "title": "Shelly integration timeout on reboot",
            "similarity_score": 0.72,
            "integration": "shelly",
            "state": "open",
        },
    ]
    draft = generate_template_draft(
        ticket_number=187999,
        missing_info=missing_items,
        similar_tickets=similar_issues,
        suggested_label="shelly",
        form_fields={"form_version": "unknown", "has_logs": False, "has_traceback": False},
    )

    # Must start with required prefix
    assert draft.startswith("[AI draft, needs review]")

    # Must request missing items
    assert "Diagnostic logs" in draft
    assert "Home Assistant Core version" in draft

    # Must reference similar issues
    assert "#182001" in draft
    assert "#181950" in draft
    assert "`shelly`" in draft

    # Must never promise a fix or release date
    assert "will fix" not in draft.lower()
    assert "fix will be released" not in draft.lower()
    assert "guarantee" not in draft.lower() or "does not guarantee" in draft.lower()
    # Must remove maintainer review commitments
    assert "a maintainer or code owner will review" not in draft.lower()
    assert "maintainers will review" not in draft.lower()


def test_filter_similar_tickets():
    similar = [
        {"ticket_number": 101, "title": "MQTT entity unavailable", "integration": "mqtt", "similarity_score": 0.88},
        {"ticket_number": 102, "title": "MQTT topic subscription fail", "integration": "mqtt", "similarity_score": 0.69},  # below 0.70
        {"ticket_number": 103, "title": "Shelly disconnects", "integration": "shelly", "similarity_score": 0.95},          # cross-integration
        {"ticket_number": 104, "title": "MQTT sensor missing", "integration": "", "similarity_score": 0.75},               # integration in title
    ]

    # Filter for mqtt with 0.70 threshold
    filtered = filter_similar_tickets(similar, current_integration="mqtt", min_score=0.70)
    ticket_nums = [t["ticket_number"] for t in filtered]

    # Should keep 101 (same integration >= 0.70) and 104 (fallback to title integration >= 0.70)
    assert 101 in ticket_nums
    assert 104 in ticket_nums
    # Should drop 102 (below 0.70 threshold)
    assert 102 not in ticket_nums
    # Should drop 103 (cross-integration dropped despite 0.95 score)
    assert 103 not in ticket_nums

    # If current integration is unknown or unlabeled, should drop all
    assert filter_similar_tickets(similar, current_integration=None) == []
    assert filter_similar_tickets(similar, current_integration="unlabeled") == []
    assert filter_similar_tickets(similar, current_integration="other") == []


def test_completeness_all_versions_flagged():
    bad_versions = ["all versions", "all", "any", "latest", "current", "na", "n/a", "recent", "not sure"]
    for bad_ver in bad_versions:
        issue = {
            "title": "Hue light does not respond",
            "form_problem": "### The problem\nWhen toggling the light in UI, nothing happens and it remains on.",
            "form_version": bad_ver,
            "form_install_type": "Home Assistant OS",
            "has_logs": True,
            "has_traceback": False,
        }
        missing = check_ticket_completeness_rules(issue)
        assert any("exact home assistant core version" in m.lower() for m in missing), f"Expected version flag for {bad_ver}"

    # Valid versions should not be flagged for version
    for good_ver in ["2026.10.1", "2026-4.3", "2025.1.0b2"]:
        issue = {
            "title": "Hue light does not respond",
            "form_problem": "### The problem\nWhen toggling the light in UI, nothing happens.",
            "form_version": good_ver,
            "form_install_type": "Home Assistant OS",
            "has_logs": True,
            "has_traceback": False,
        }
        missing = check_ticket_completeness_rules(issue)
        assert not any("exact home assistant core version" in m.lower() for m in missing), f"Did not expect version flag for {good_ver}"


def test_completeness_skips_bot_dashboards():
    bot_issues = [
        {"title": "Dependency Dashboard", "form_problem": "This issue contains a list of Renovate updates.", "form_version": "2026.10.0"},
        {"title": "Update dependency ruff to v0.7.0", "form_problem": "Renovate bot automated PR", "form_version": "2026.10.0"},
        {"title": "Automated security updates", "form_problem": "dependabot alerts summary", "form_version": "2026.10.0"},
        {"title": "Nightly release failed", "form_problem": "github-actions failed on build step", "form_version": "2026.10.0", "is_bot": True},
    ]
    for issue in bot_issues:
        assert is_developer_or_bot_ticket(issue) is True
        # Completeness rules should return empty list (skip completely)
        missing = check_ticket_completeness_rules(issue)
        assert missing == [], f"Expected bot issue {issue['title']} to be skipped"


def test_completeness_skips_developer_tasks():
    dev_issues = [
        {"title": "Migrate cast to async_setup_entry", "form_problem": "Migration task for core cast component", "form_version": "2026.10.0"},
        {"title": "Remove deprecated async_forward_entry_setup", "form_problem": "Part of cleanup effort", "form_version": "2026.10.0"},
        {"title": "Refactor bluetooth coordinator", "form_problem": "Optimize scanning latency", "form_version": "2026.10.0"},
        {"title": "Bump pydantic to 2.9", "form_problem": "Update library dependencies", "form_version": "2026.10.0"},
        {"title": "Clean up unneeded logger references", "form_problem": "Developer task parent issue", "form_version": "2026.10.0"},
    ]
    for issue in dev_issues:
        assert is_developer_or_bot_ticket(issue) is True
        missing = check_ticket_completeness_rules(issue)
        assert missing == [], f"Expected developer task {issue['title']} to be skipped"


def test_completeness_vague_symptom_and_screenshot_only():
    # Vague one-liner without logs
    vague_issue = {
        "title": "Entities are gone",
        "form_problem": "### The problem\nAll my entities disappeared from the dashboard.",
        "form_version": "2026.10.1",
        "form_install_type": "Home Assistant OS",
        "has_logs": False,
        "has_traceback": False,
    }
    missing_vague = check_ticket_completeness_rules(vague_issue)
    assert any("steps to reproduce" in m.lower() or "too brief" in m.lower() for m in missing_vague)

    # Screenshot-only issue
    screenshot_issue = {
        "title": "Error on dashboard",
        "form_problem": "### The problem\nSee attached screenshot:\n<img src='https://github.com/home-assistant/core/assets/123/abc.png' />",
        "form_version": "2026.10.1",
        "form_install_type": "Home Assistant OS",
        "has_logs": False,
        "has_traceback": False,
    }
    missing_screen = check_ticket_completeness_rules(screenshot_issue)
    assert any("steps to reproduce" in m.lower() or "too brief" in m.lower() for m in missing_screen)


def test_is_bug_report_template():
    structured_issue = {
        "title": "Bug in automation",
        "form_problem": "### The problem\nAutomation triggers twice.",
        "form_version": "2026.10.0",
        "form_install_type": "Home Assistant OS",
    }
    assert is_bug_report_template(structured_issue) is True

    unstructured_issue = {
        "title": "Question about docs",
        "body": "Where can I find documentation on zigbee pairing?",
        "form_version": "unknown",
        "form_install_type": "unknown",
    }
    assert is_bug_report_template(unstructured_issue) is False

