"""Inspect all sandbox comments and issues in AaruneshAP/ticket-triage-sandbox.

Outputs structured metadata to docs/results/sandbox_comments.json without narrative.
"""

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List

RESULTS_PATH = Path("docs/results/sandbox_comments.json")

RESOLUTION_TIME_REGEX = re.compile(
    r"(p50|p90|\b\d+\.?\d*\s*days?\b|probabilit|resolution\s*time|time\s*to\s*close)",
    re.IGNORECASE,
)


def run_gh_api(endpoint: str) -> Any:
    cmd = ["gh", "api", endpoint, "--paginate"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return {"error": result.stderr.strip(), "returncode": result.returncode}
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"error": "JSON decode error", "raw": result.stdout[:200]}


def main() -> None:
    # 1. Fetch all comments
    comments_raw = run_gh_api("repos/AaruneshAP/ticket-triage-sandbox/issues/comments")
    comments_list: List[Dict[str, Any]] = []

    if isinstance(comments_raw, list):
        for c in comments_raw:
            body = c.get("body") or ""
            issue_url = c.get("issue_url") or ""
            issue_number = None
            if issue_url:
                try:
                    issue_number = int(issue_url.rstrip("/").split("/")[-1])
                except (ValueError, IndexError):
                    pass

            sha256_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
            regex_match = bool(RESOLUTION_TIME_REGEX.search(body))

            comments_list.append({
                "id": c.get("id"),
                "author": c.get("user", {}).get("login"),
                "created_at": c.get("created_at"),
                "issue_number": issue_number,
                "body_length": len(body),
                "sha256": sha256_hash,
                "resolution_regex_match": regex_match,
            })

    # 2. Fetch all issues (all states)
    issues_raw = run_gh_api("repos/AaruneshAP/ticket-triage-sandbox/issues?state=all")
    issue_numbers = []
    if isinstance(issues_raw, list):
        issue_numbers = [i.get("number") for i in issues_raw if isinstance(i, dict) and "number" in i]

    output_data = {
        "repository": "AaruneshAP/ticket-triage-sandbox",
        "issue_count": len(issue_numbers),
        "comments": sorted(comments_list, key=lambda x: (x["issue_number"] or 0, x["id"])),
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(json.dumps(output_data, indent=2))


if __name__ == "__main__":
    main()
