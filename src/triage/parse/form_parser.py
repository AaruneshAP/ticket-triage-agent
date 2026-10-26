"""Form parsing and diagnostic extraction for GitHub issue bodies.

Extracts structured sections from standard issue forms, handles unstructured/hand-written
reports, removes target leakage sections for label prediction, and computes diagnostic indicators.
"""

import re
from typing import Any, Dict, Optional, Tuple

STANDARD_HEADINGS = [
    "The problem",
    "What version of Home Assistant Core has the issue?",
    "What was the last working version of Home Assistant Core?",
    "What type of installation are you running?",
    "Integration causing the issue",
    "Link to integration documentation on our website",
    "Anything in the logs that might be useful for us?",
    "Diagnostics information",
    "Example YAML snippet",
    "Additional information",
]

# Regex patterns for diagnostic metrics
TRACEBACK_PATTERN = re.compile(
    r"(Traceback \(most recent call last\):|Logger:\s+[a-zA-Z0-9_\.]+|File \"[^\"]+\", line \d+|Error:\s+|Exception:\s+)",
    re.IGNORECASE,
)
URL_PATTERN = re.compile(r"https?://[^\s\)\>\]]+", re.IGNORECASE)
CODE_BLOCK_PATTERN = re.compile(r"```[\s\S]*?```")
ISSUE_REF_PATTERN = re.compile(
    r"(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?|duplicate of|see)\s+#(\d+)|#(\d+)", re.IGNORECASE
)
HA_VERSION_PATTERN = re.compile(r"(\d{4})\.(\d{1,2})(?:\.(\d{1,2}))?")


def sanitize_field_value(value: Optional[str]) -> Optional[str]:
    """Return None if string is empty or contains standard GitHub null placeholder."""
    if not value:
        return None
    val = value.strip()
    if not val or val.lower() in ("_no response_", "none", "n/a", "no response", "null"):
        return None
    return val


def parse_form_sections(body: Optional[str]) -> Dict[str, Optional[str]]:
    """
    Parse markdown body with '### <heading>' into a dictionary of field names to content.
    If no headings are found (handwritten report), maps the entire body to 'The problem'.
    """
    if not body or not body.strip():
        return {h: None for h in STANDARD_HEADINGS}

    # Split by level 3 headers
    parts = re.split(r"^###\s+(.+)$", body, flags=re.MULTILINE)

    if len(parts) <= 1:
        # No '### ' headings found: treat entire body as problem statement
        sections: Dict[str, Optional[str]] = {h: None for h in STANDARD_HEADINGS}
        sections["The problem"] = sanitize_field_value(body)
        return sections

    sections = {h: None for h in STANDARD_HEADINGS}
    # First part before any heading (if any text exists)
    preamble = parts[0].strip()

    for i in range(1, len(parts), 2):
        heading = parts[i].strip()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        sections[heading] = sanitize_field_value(content)

    if preamble and not sections.get("The problem"):
        sections["The problem"] = sanitize_field_value(preamble)

    return sections


def is_template_body(body: Optional[str]) -> bool:
    """Return True if issue body contains at least two standard template headings."""
    if not body:
        return False
    found = sum(1 for h in STANDARD_HEADINGS if f"### {h}" in body)
    return found >= 2


def clean_text_for_label_prediction(body: Optional[str]) -> str:
    """
    Remove 'Integration causing the issue' and related link sections from the issue body.
    CRITICAL: Prevents target leakage when training or inferring label prediction models.
    """
    if not body:
        return ""

    # Remove the entire '### Integration causing the issue' block up to the next '### ' heading or EOF
    cleaned = re.sub(
        r"###\s+Integration causing the issue[\s\S]*?(?=(?:^###\s+|\Z))",
        "",
        body,
        flags=re.MULTILINE,
    )
    # Also remove link to documentation section if present
    cleaned = re.sub(
        r"###\s+Link to integration documentation on our website[\s\S]*?(?=(?:^###\s+|\Z))",
        "",
        cleaned,
        flags=re.MULTILINE,
    )
    return cleaned.strip()


def extract_ha_version_parts(
    version_str: Optional[str],
) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """Parse Home Assistant version string (e.g. '2026.10.2') into (year, month, patch)."""
    if not version_str:
        return None, None, None

    match = HA_VERSION_PATTERN.search(version_str)
    if not match:
        return None, None, None

    year = int(match.group(1))
    month = int(match.group(2))
    patch = int(match.group(3)) if match.group(3) else 0
    return year, month, patch


def parse_issue(body: Optional[str], title: Optional[str] = "") -> Dict[str, Any]:
    """
    Comprehensive parsing routine for an issue body and title.
    Returns structured fields and feature metrics without temporal/target leakage.
    """
    text = body or ""
    sections = parse_form_sections(text)
    is_tmpl = is_template_body(text)

    # Core fields
    problem = sections.get("The problem") or (text.strip() if not is_tmpl else None)
    version = sections.get("What version of Home Assistant Core has the issue?")
    last_working_version = sections.get("What was the last working version of Home Assistant Core?")
    install_type = sections.get("What type of installation are you running?")
    integration_in_form = sections.get("Integration causing the issue")
    logs = sections.get("Anything in the logs that might be useful for us?")
    diagnostics = sections.get("Diagnostics information")
    yaml_snippet = sections.get("Example YAML snippet")
    additional_info = sections.get("Additional information")

    # Diagnostic indicators
    code_blocks = CODE_BLOCK_PATTERN.findall(text)
    n_code_blocks = len(code_blocks)
    n_urls = len(URL_PATTERN.findall(text))
    has_tb = bool(TRACEBACK_PATTERN.search(text))
    has_logs = bool(logs and len(logs.strip()) > 15) or has_tb

    # Version parts
    year, month, patch = extract_ha_version_parts(version)

    # Missing fields count (among 6 core required fields in standard template)
    core_fields = [problem, version, install_type, integration_in_form, logs, diagnostics]
    n_missing = sum(1 for f in core_fields if not f)

    # Issue references (e.g. #12345)
    refs = set()
    for m in ISSUE_REF_PATTERN.finditer(f"{title} {text}"):
        ref_num = m.group(1) or m.group(2)
        if ref_num:
            refs.add(int(ref_num))

    return {
        "is_template": is_tmpl,
        "problem": problem,
        "version": version,
        "ha_version_year": year,
        "ha_version_month": month,
        "ha_version_patch": patch,
        "has_last_working_version": bool(last_working_version),
        "last_working_version": last_working_version,
        "install_type": install_type,
        "integration_in_form": integration_in_form,
        "logs": logs,
        "diagnostics": diagnostics,
        "yaml_snippet": yaml_snippet,
        "additional_info": additional_info,
        "has_traceback": has_tb,
        "has_logs": has_logs,
        "n_code_blocks": n_code_blocks,
        "n_urls": n_urls,
        "n_missing_form_fields": n_missing,
        "refers_to_issue": len(refs) > 0,
        "referenced_issues": sorted(list(refs)),
        "cleaned_text_no_target": clean_text_for_label_prediction(text),
    }
