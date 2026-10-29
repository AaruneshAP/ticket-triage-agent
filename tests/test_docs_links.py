"""Test markdown links and image references in tracked documentation files.

Ensures that no broken relative links or missing image assets exist in tracked .md files.
"""

import re
import subprocess
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parent.parent

# Matches markdown links [text](target) and images ![alt](target)
LINK_REGEX = re.compile(r"!?\[([^\]]*)\]\(([^)]+)\)")
# Matches HTML tags <img src="..."> and <a href="...">
HTML_SRC_REGEX = re.compile(r'<(?:img|a)\b[^>]*(?:src|href)=["\']([^"\']+)["\']', re.IGNORECASE)


def get_tracked_md_files() -> List[Path]:
    """Return all git-tracked .md files in the repository."""
    res = subprocess.run(
        ["git", "ls-files", "*.md"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        encoding="utf-8",
    )
    files = []
    for line in res.stdout.splitlines():
        line = line.strip()
        if line:
            p = REPO_ROOT / line
            if p.exists():
                files.append(p)
    return files


def test_no_broken_markdown_links():
    """Verify all relative links and image references resolve to existing files or directories."""
    tracked_files = get_tracked_md_files()
    assert tracked_files, "No tracked markdown files found."

    broken_links: List[str] = []

    for md_file in tracked_files:
        content = md_file.read_text(encoding="utf-8")
        lines = content.splitlines()

        for line_num, line in enumerate(lines, start=1):
            # 1. Check markdown links/images
            for m in LINK_REGEX.finditer(line):
                raw_url = m.group(2).strip()
                _check_url(raw_url, md_file, line_num, line, broken_links)

            # 2. Check HTML src/href attributes
            for m in HTML_SRC_REGEX.finditer(line):
                raw_url = m.group(1).strip()
                _check_url(raw_url, md_file, line_num, line, broken_links)

    if broken_links:
        failure_msg = f"Found {len(broken_links)} broken relative link(s) in documentation:\n"
        failure_msg += "\n".join(f"  - {err}" for err in broken_links)
        assert False, failure_msg


def _check_url(url: str, md_file: Path, line_num: int, line_text: str, broken_list: List[str]) -> None:
    # Ignore external URLs, anchors, and emails
    if url.startswith(("http://", "https://", "mailto:", "ftp:", "#")):
        return

    # Ignore template variables e.g. {{ ... }} or <...>
    if url.startswith("{{") or url.startswith("<"):
        return

    # Strip anchor and query params
    cleaned_target = url.split("#")[0].split("?")[0].strip()
    if not cleaned_target:
        # Was anchor-only link like (#section)
        return

    # If target is absolute relative to root (starts with /)
    if cleaned_target.startswith("/"):
        resolved_path = (REPO_ROOT / cleaned_target.lstrip("/")).resolve()
    else:
        resolved_path = (md_file.parent / cleaned_target).resolve()

    if not resolved_path.exists():
        rel_md = md_file.relative_to(REPO_ROOT)
        broken_list.append(
            f"{rel_md}:{line_num}: Target '{url}' not found on disk (resolved to: {resolved_path})"
        )
