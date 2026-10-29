"""Tests for sandbox_guard to ensure resolution latency estimates are blocked from posting."""

import pytest

from triage.agent.sandbox_guard import (
    ResolutionLeakageError,
    validate_sandbox_comment_no_resolution_time,
)


def test_clean_comment_allowed():
    """Verify that legitimate triage comments without latency estimates are permitted."""
    clean_comment = (
        "[AI draft, needs review]\n\n"
        "Thank you for reporting this issue. Based on triage analysis:\n"
        "- **Identified Component**: `zha`\n"
        "- **Diagnostics**: Traceback detected in logs.\n"
    )
    # Should not raise
    validate_sandbox_comment_no_resolution_time(clean_comment)


@pytest.mark.parametrize(
    "prohibited_text",
    [
        "Estimated resolution time is 3.5 days.",
        "Expected time to close: 14 days",
        "P50 resolution time is 2 days, P90 is 10 days.",
        "The closure probability is 0.75.",
        "Median time to close is 5 days.",
        "This ticket has a high probability of closing soon.",
    ],
)
def test_resolution_estimates_blocked(prohibited_text: str):
    """Verify that any resolution time or probability phrase raises ResolutionLeakageError."""
    with pytest.raises(ResolutionLeakageError) as exc_info:
        validate_sandbox_comment_no_resolution_time(prohibited_text)
    assert "Blocked comment containing prohibited resolution latency estimate" in str(exc_info.value)
