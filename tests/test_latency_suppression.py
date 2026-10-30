"""Tests ensuring all resolution-time estimates and probabilities remain suppressed.

Fails if any resolution-time probability, P50, or P90 appears in:
1. LangGraph output
2. Sandbox draft text / LLM draft responses
3. hf_space/index.html
"""

import re
from pathlib import Path

import pandas as pd

from triage.agent.graph import TriageAgentGraph
from triage.agent.llm import call_llm_draft


def test_langgraph_output_suppresses_resolution_metrics():
    """Verify LangGraph output contains no numeric P50/P90 or closure probabilities."""
    graph = TriageAgentGraph(models_dir="models")
    raw_issue = {
        "number": 188657,
        "title": "Zigbee coordinator disconnects after update",
        "body": (
            "### The problem\nCoordinator disconnects intermittently.\n\n"
            "### Home Assistant Core Version\ncore-2026.10.1\n\n"
            "### Integration causing the issue\nzha"
        ),
        "labels": [{"name": "integration: zha"}],
    }

    result = graph.run(raw_issue)

    # 1. Check predicted_days_to_close
    days_info = result.get("predicted_days_to_close", {})
    assert days_info.get("p50") is None, f"p50 must be None, found: {days_info.get('p50')}"
    assert days_info.get("p90") is None, f"p90 must be None, found: {days_info.get('p90')}"
    assert days_info.get("status") == "no reliable signal for resolution time"

    # 2. Check fixed_horizon_probabilities
    fh_info = result.get("fixed_horizon_probabilities", {})
    assert fh_info.get("p_closed_within_14d") is None, (
        f"p_closed_within_14d must be None, found: {fh_info.get('p_closed_within_14d')}"
    )
    assert fh_info.get("p_closed_within_30d") is None, (
        f"p_closed_within_30d must be None, found: {fh_info.get('p_closed_within_30d')}"
    )
    assert fh_info.get("status") == "no reliable signal for resolution time"

    # 3. Ensure no numeric resolution-time predictions appear in draft reply
    draft = result.get("draft_reply", "")
    for term in ["p50", "p90", "probability of closing", "expected resolution", "will be closed in"]:
        assert term not in draft.lower(), f"Forbidden resolution term '{term}' in draft reply:\n{draft}"


def test_sandbox_draft_text_suppresses_resolution_metrics():
    """Verify draft replies synthesized for triage sandbox have zero resolution predictions."""
    draft = call_llm_draft(
        ticket_number=188657,
        title="Zigbee coordinator disconnects",
        problem_text="Coordinator disconnects after update",
        missing_info=["Detailed reproduction steps", "Diagnostic logs or error traceback"],
        similar_tickets=[
            {
                "ticket_number": 181690,
                "title": "SONOFF ZBMINIL2 devices repeatedly become unavailable with ZBDongle-E and ZHA",
                "similarity": 0.82,
                "integration": "zha",
            }
        ],
        suggested_label="integration: zha",
        form_fields={"integration": "zha"},
    )

    forbidden_patterns = [
        r"\bp50\b",
        r"\bp90\b",
        r"probability of closing",
        r"expected resolution",
        r"closure probability",
        r"\b\d+(\.\d+)?\s*days?\b",
        r"closed within \d+ days",
    ]

    for pat in forbidden_patterns:
        match = re.search(pat, draft, re.IGNORECASE)
        assert not match, f"Draft text contains resolution-time estimate matching '{pat}': {match.group(0)}\nDraft:\n{draft}"


def test_hf_space_index_html_suppresses_resolution_metrics():
    """Verify hf_space/index.html contains no P50/P90 or resolution probability metrics."""
    html_path = Path("hf_space/index.html")
    assert html_path.exists(), "hf_space/index.html does not exist"
    html_content = html_path.read_text(encoding="utf-8")

    forbidden_patterns = [
        r"\bp50\b",
        r"\bp90\b",
        r"probability of closing",
        r"closed within 14",
        r"closed within 30",
        r"p_closed_within",
    ]

    for pat in forbidden_patterns:
        match = re.search(pat, html_content, re.IGNORECASE)
        assert not match, f"hf_space/index.html contains forbidden resolution term matching '{pat}': {match.group(0)}"


def test_sheets_feeds_suppress_resolution_metrics():
    """Verify Google Sheets feeds contain no predicted resolution-time columns or values (P50, P90, closure probabilities)."""
    from triage.monitoring.sheets_bridge import generate_sheets_feeds

    feeds = generate_sheets_feeds()
    assert feeds.get("status") == "success"

    forbidden_patterns = [
        r"\bp50\b",
        r"\bp90\b",
        r"predicted_time_to_close",
        r"probability of closing",
        r"closure_prob",
        r"p_closed_within",
    ]

    for feed_name, csv_path in feeds.get("files_generated", {}).items():
        df = pd.read_csv(csv_path)
        # Check column names
        for col in df.columns:
            for pat in forbidden_patterns:
                assert not re.search(pat, col, re.IGNORECASE), (
                    f"Feed '{feed_name}' contains forbidden predicted resolution-time column '{col}' matching '{pat}'"
                )
        # Check cell values
        for col in df.columns:
            for val in df[col].dropna().astype(str):
                for pat in forbidden_patterns:
                    assert not re.search(pat, val, re.IGNORECASE), (
                        f"Feed '{feed_name}' column '{col}' contains forbidden predicted resolution-time value '{val}' matching '{pat}'"
                    )

