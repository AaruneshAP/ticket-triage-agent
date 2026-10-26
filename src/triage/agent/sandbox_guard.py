"""Guardrail to block posting of resolution latency estimates to GitHub sandbox issues."""

import re

RESOLUTION_TIME_REGEX = re.compile(
    r"(p50|p90|\b\d+\.?\d*\s*days?\b|probabilit|resolution\s*time|time\s*to\s*close)",
    re.IGNORECASE,
)


class ResolutionLeakageError(ValueError):
    """Raised when an automated comment contains uncalibrated resolution latency estimates."""
    pass


def validate_sandbox_comment_no_resolution_time(comment: str) -> None:
    """Validate that a comment contains no resolution latency estimates or probabilities.

    Raises:
        ResolutionLeakageError: If RESOLUTION_TIME_REGEX matches any substring.
    """
    if not comment:
        return
    match = RESOLUTION_TIME_REGEX.search(comment)
    if match:
        raise ResolutionLeakageError(
            f"Blocked comment containing prohibited resolution latency estimate: '{match.group(0)}'"
        )
