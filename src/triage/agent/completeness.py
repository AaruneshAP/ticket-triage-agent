"""Ticket Completeness & Diagnostic Quality Rules Engine.

Identifies missing critical information from incoming Home Assistant bug reports:
1. Missing or unparsed Home Assistant Core version.
2. Missing last working version when regression is claimed.
3. Missing traceback / error logs when an error or failure is reported.
4. Empty or terse problem description.
5. Unknown installation environment.
"""

import re
from typing import Any, Dict, List

ERROR_KEYWORDS = [
    "error",
    "traceback",
    "exception",
    "crash",
    "failed",
    "failure",
    "broken",
    "not working",
    "stopped working",
    "timeout",
    "unhandled",
    "disconnect",
    "fail",
]


def is_developer_or_bot_ticket(parsed_issue: Dict[str, Any]) -> bool:
    """Check if the ticket is a developer task, architectural issue, or bot dashboard."""
    title = str(parsed_issue.get("title") or "").strip().lower()
    text = str(
        parsed_issue.get("form_problem")
        or parsed_issue.get("problem")
        or parsed_issue.get("body")
        or parsed_issue.get("ticket_text")
        or parsed_issue.get("clean_problem_text")
        or ""
    ).strip().lower()
    full_str = f"{title} {text}"

    # Bot dashboards
    if any(bot in full_str for bot in ["dependency dashboard", "renovate", "dependabot", "github-actions"]):
        return True
    if bool(parsed_issue.get("is_bot")):
        return True

    # Developer task titles
    for prefix in ["migrate ", "remove ", "refactor ", "deprecate ", "bump ", "clean up "]:
        if title.startswith(prefix):
            return True

    # Explicit developer task notes
    if "developer task" in full_str or "parent task" in full_str or "parent tasks:" in full_str:
        return True

    return False


def is_bug_report_template(parsed_issue: Dict[str, Any]) -> bool:
    """Return True if issue is using the structured Home Assistant bug report form template."""
    if bool(parsed_issue.get("is_template")):
        return True

    text = str(
        parsed_issue.get("form_problem")
        or parsed_issue.get("problem")
        or parsed_issue.get("body")
        or parsed_issue.get("ticket_text")
        or parsed_issue.get("clean_problem_text")
        or ""
    )
    if "### The problem" in text or "### What version" in text or "### What type" in text:
        return True

    install = str(parsed_issue.get("form_install_type") or parsed_issue.get("install_type") or "").strip().lower()
    ver = str(parsed_issue.get("form_version") or parsed_issue.get("version") or "").strip().lower()
    if install and install not in ["nan", "none", "unknown", ""]:
        return True
    if ver and ver not in ["nan", "none", "unknown", ""]:
        return True

    return False


def check_ticket_completeness_rules(parsed_issue: Dict[str, Any]) -> List[str]:
    """
    Evaluate an issue for completeness using deterministic rules (First Pass).
    Rules apply ONLY to tickets using the bug-report template.
    Skips developer tasks, bot dashboards, and old-template issues.
    """
    # 1. Skip non-bug-report tickets
    if is_developer_or_bot_ticket(parsed_issue):
        return []
    if not is_bug_report_template(parsed_issue):
        return []

    missing_items: List[str] = []

    # 2. Version check (treat 'all versions' / 'latest' / etc. as missing; accept '2026-4.3' or '2026.4.3' or System Information block)
    raw_version = str(parsed_issue.get("form_version") or parsed_issue.get("version") or "").strip()
    problem_text = str(
        parsed_issue.get("form_problem")
        or parsed_issue.get("problem")
        or parsed_issue.get("body")
        or parsed_issue.get("ticket_text")
        or parsed_issue.get("clean_problem_text")
        or ""
    ).strip()

    # Check if version is inside a System Information markdown table (e.g., "version | core-2026.7.4")
    sys_match = re.search(
        r"version\s*\|\s*(core-20\d{2}[.-]\d{1,2}(?:\.[a-z0-9]+)?|20\d{2}[.-]\d{1,2}(?:\.[a-z0-9]+)?)",
        raw_version,
        re.I,
    )
    if not sys_match:
        sys_match = re.search(
            r"version\s*\|\s*(core-20\d{2}[.-]\d{1,2}(?:\.[a-z0-9]+)?|20\d{2}[.-]\d{1,2}(?:\.[a-z0-9]+)?)",
            problem_text,
            re.I,
        )

    if sys_match:
        ha_version = sys_match.group(1).lower()
    else:
        ha_version = raw_version.lower()

    if ha_version and ha_version not in ["nan", "none", "unknown", "_no response_", ""]:
        if re.search(r"\b(all versions|all|any|latest|current|na|n/a|not sure|recent)\b", ha_version):
            missing_items.append("Exact Home Assistant Core version (e.g., 2026.10.1)")
        else:
            if not re.search(r"20\d{2}[.-]\d{1,2}|core-20\d{2}[.-]\d{1,2}", ha_version):
                missing_items.append("Exact Home Assistant Core version (e.g., 2026.10.1)")
    else:
        # If not in form_version, check if mentioned in body text (e.g., 2026.4.3 or core-2026.7.4)
        if not re.search(r"20\d{2}[.-]\d{1,2}|core-20\d{2}[.-]\d{1,2}|0\.\d{2}", problem_text):
            missing_items.append("Exact Home Assistant Core version (e.g., 2026.10.1)")

    # 3. Vague-symptom / short-text check (missed one-liners with no log and no error)
    has_logs = bool(parsed_issue.get("has_logs", False))
    has_traceback = bool(parsed_issue.get("has_traceback", False))
    err_pattern = re.compile(
        r"Traceback|Logger:|File \".*?\", line|TypeError|KeyError|ValueError|AttributeError|Exception:|fatal:|API returned error code",
        re.I,
    )
    has_explicit_err = bool(err_pattern.search(problem_text)) or bool(
        err_pattern.search(str(parsed_issue.get("logs") or ""))
    )

    cleaned = re.sub(r"<img[\s\S]*?>", "", problem_text)
    cleaned = re.sub(r"!\[[^\]]*\]\([^\)]*\)", "", cleaned)
    cleaned = re.sub(r"https?://\S+", "", cleaned)
    cleaned = re.sub(r"###\s+The problem\s*(?:\([^)]*\))?\s*", "", cleaned)
    cleaned = re.sub(r"###\s+[A-Za-z0-9\s\?]+\s*", "", cleaned).strip()
    words = [w for w in cleaned.split() if len(w) > 1]

    is_screenshot_only = ("<img" in problem_text or "![" in problem_text) and len(words) < 20
    is_title_only = bool(
        re.search(r"\b(as in the title|see title|same as title|title says it all|as above)\b", cleaned, re.I)
    ) or (len(words) <= 5 and "title" in cleaned.lower())
    is_one_liner = (not has_traceback and not has_logs and not has_explicit_err) and (
        is_title_only or len(words) < 25
    )

    if is_screenshot_only or is_one_liner:
        missing_items.append(
            "Detailed description of the problem and steps to reproduce (summary is too brief and lacks error logs)"
        )

    # Note: Last working version is optional and non-essential for triage, so it is not flagged as missing.
    return missing_items


def check_ticket_completeness(parsed_issue: Dict[str, Any], use_llm: bool = True) -> List[str]:
    """
    Two-pass completeness check:
    1. Deterministic rules pass for core template requirements.
    2. Subtle-gap LLM pass for hardware/firmware, reproduction, or configuration omissions.
    """
    missing = check_ticket_completeness_rules(parsed_issue)

    if use_llm:
        try:
            from triage.agent.llm import detect_subtle_gaps

            title = str(parsed_issue.get("title", ""))
            prob = str(parsed_issue.get("form_problem") or parsed_issue.get("body", ""))
            subtle = detect_subtle_gaps(title, prob, parsed_issue)
            for s in subtle:
                if not any(s.lower() in m.lower() for m in missing):
                    missing.append(s)
        except Exception:
            pass

    return missing


def evaluate_completeness_on_sample(df: Any, n_samples: int = 50) -> Dict[str, Any]:
    """
    Evaluate completeness checks across a deterministic sample of tickets.
    """
    sample_df = df.head(n_samples)
    results = []
    has_missing_count = 0

    for _, row in sample_df.iterrows():
        missing = check_ticket_completeness(row.to_dict())
        if missing:
            has_missing_count += 1
        results.append(
            {
                "ticket_number": int(row.get("number", row.get("ticket_number", 0))),
                "missing_items": missing,
                "is_complete": len(missing) == 0,
            }
        )

    incomplete_pct = round(100.0 * has_missing_count / n_samples, 1)
    return {
        "sample_size": n_samples,
        "tickets_with_gaps": has_missing_count,
        "incomplete_percentage": incomplete_pct,
        "sample_details": results[:10],
    }
