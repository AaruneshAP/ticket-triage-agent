"""
GitHub Issue Ingestion Script for home-assistant/core.

Fetches the latest open and closed issues, skips pull requests,
handles API rate limits, and exports both raw JSON and a flat CSV.
"""

import csv
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple

import requests
from dotenv import load_dotenv

# Load key-value pairs from local .env into os.environ
load_dotenv()

# Target repository coordinates
REPO_OWNER = "home-assistant"
REPO_NAME = "core"
GITHUB_API_BASE = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}"

# Desired number of actual issues per state
TARGET_ISSUES_PER_STATE = 200
# Maximum allowed items per page by GitHub REST API
ITEMS_PER_PAGE = 100


def get_auth_headers() -> Dict[str, str]:
    """
    Read GITHUB_TOKEN from environment and construct GitHub API request headers.
    Exits gracefully if token is missing.
    """
    token = os.getenv("GITHUB_TOKEN", "").strip()

    if not token:
        print("[ERROR] GITHUB_TOKEN is not set in your .env file.", file=sys.stderr)
        print("Please add 'GITHUB_TOKEN=your_token' to .env and try again.", file=sys.stderr)
        sys.exit(1)

    return {
        # Recommended API version header for GitHub REST API v3
        "X-GitHub-Api-Version": "2022-11-28",
        # Request standard JSON response format
        "Accept": "application/vnd.github+json",
        # Standard Bearer authorization header
        "Authorization": f"Bearer {token}",
        # Custom user agent identifies our client and follows GitHub API guidelines
        "User-Agent": "Ticket-Triage-Agent-Ingest/1.0",
    }


def check_rate_limit(headers: Dict[str, str]) -> None:
    """
    Inspect GitHub rate limit response headers and pause or warn if limit is nearly exhausted.
    """
    remaining_str = headers.get("X-RateLimit-Remaining")
    reset_str = headers.get("X-RateLimit-Reset")

    if remaining_str is not None:
        remaining = int(remaining_str)
        # Log status if quota drops below 50 requests
        if remaining < 50:
            print(f"  [RATE LIMIT WARNING] {remaining} requests remaining in quota.")

        # If quota is critically low, pause or halt
        if remaining <= 2 and reset_str is not None:
            reset_epoch = int(reset_str)
            wait_seconds = max(0, reset_epoch - int(time.time())) + 1
            if wait_seconds <= 60:
                print(
                    f"  [RATE LIMIT PAUSE] Limit almost reached. Sleeping {wait_seconds}s until reset..."
                )
                time.sleep(wait_seconds)
            else:
                print(
                    f"  [RATE LIMIT STOP] Limit almost reached. Reset in {wait_seconds}s. Stopping gracefully."
                )
                raise StopIteration("Rate limit nearly exhausted; stopping network calls.")


def fetch_issues_for_state(
    state: str, target_count: int, headers: Dict[str, str]
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Fetch up to target_count issues for a specific state ('open' or 'closed').
    Filters out pull requests (items containing a 'pull_request' key).

    Returns:
        (collected_issues, skipped_pr_count)
    """
    collected: List[Dict[str, Any]] = []
    skipped_prs = 0
    page = 1

    print(f"\n--> Fetching {target_count} '{state}' issues from {REPO_OWNER}/{REPO_NAME}...")

    while len(collected) < target_count:
        # Request parameters following GitHub REST API documentation
        params = {
            "state": state,
            "sort": "created",
            "direction": "desc",
            "per_page": ITEMS_PER_PAGE,
            "page": page,
        }

        # Issue the GET request to GitHub REST endpoint
        url = f"{GITHUB_API_BASE}/issues"
        response = requests.get(url, headers=headers, params=params, timeout=30)

        # Inspect rate limit headers on every response
        try:
            check_rate_limit(dict(response.headers))
        except StopIteration:
            # Stop pagination gracefully if quota is exhausted
            break

        # Check for HTTP errors (e.g. 401 Bad Credentials, 404 Repo Not Found)
        if response.status_code != 200:
            print(
                f"[ERROR] GitHub API returned status {response.status_code}: {response.text}",
                file=sys.stderr,
            )
            break

        items = response.json()
        # An empty list indicates no further issues exist in this state
        if not items:
            print(f"  No more {state} items returned by GitHub API at page {page}.")
            break

        page_issues_count = 0
        page_pr_count = 0

        for item in items:
            # GitHub's /issues endpoint returns both issues and PRs.
            # Real pull requests always have a 'pull_request' dictionary key.
            if "pull_request" in item:
                skipped_prs += 1
                page_pr_count += 1
                continue

            collected.append(item)
            page_issues_count += 1

            # Stop immediately once the target quota of pure issues is reached
            if len(collected) >= target_count:
                break

        print(
            f"  Page {page}: received {len(items)} items -> "
            f"+{page_issues_count} issues (total: {len(collected)}/{target_count}), "
            f"+{page_pr_count} PRs skipped"
        )

        # Increment page counter for next API call
        page += 1

    return collected, skipped_prs


def save_raw_json(issues: List[Dict[str, Any]], target_dir: str = "data/raw") -> str:
    """
    Save the un-modified JSON response list to data/raw/issues_<date>.json.
    """
    # Ensure nested target directories exist without throwing error if already present
    os.makedirs(target_dir, exist_ok=True)

    # Format date stamp as YYYYMMDD (e.g., issues_20261023.json)
    date_str = datetime.now().strftime("%Y%m%d")
    file_path = os.path.join(target_dir, f"issues_{date_str}.json")

    # Serialize complete JSON payload with indentation for human-readable inspection
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(issues, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Raw JSON saved to: {file_path} ({len(issues)} items)")
    return file_path


def flatten_issue(issue: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform a nested GitHub issue object into a flat dictionary matching the required CSV columns:
    number, title, body, state, state_reason, labels, author, author_association, assignee, comments, created_at, closed_at.
    """
    # Extract nested label names and join with pipe symbol '|'
    labels = issue.get("labels", [])
    label_names = [label["name"] for label in labels if isinstance(label, dict) and "name" in label]
    integration_name = ""
    for name in label_names:
        if name.lower().startswith("integration:"):
            integration_name = name.split(":", 1)[1].strip()
            break
    joined_labels = "|".join(label_names)

    # Safely extract author login from nested 'user' dict (may be None if user was deleted)
    user_obj = issue.get("user") or {}
    author = user_obj.get("login", "")

    # Safely extract assignee login from nested 'assignee' dict (frequently None on open tickets)
    assignee_obj = issue.get("assignee") or {}
    assignee = assignee_obj.get("login", "")

    return {
        "number": issue.get("number"),
        "title": issue.get("title", ""),
        # Replace None body with empty string so CSV cell contains clean text
        "body": issue.get("body") or "",
        "state": issue.get("state", ""),
        # state_reason indicates 'completed', 'not_planned', or 'reopened' (or None)
        "state_reason": issue.get("state_reason") or "",
        "labels": joined_labels,
        "integration": integration_name,
        "author": author,
        "author_association": issue.get("author_association", ""),
        "assignee": assignee,
        "comments": issue.get("comments", 0),
        "created_at": issue.get("created_at", ""),
        "closed_at": issue.get("closed_at") or "",
    }


def save_flattened_csv(issues: List[Dict[str, Any]], target_path: str = "data/issues.csv") -> str:
    """
    Transform issues to flat dictionaries and write them into data/issues.csv.
    """
    # Ensure parent directory 'data' exists
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    fieldnames = [
        "number",
        "title",
        "body",
        "state",
        "state_reason",
        "labels",
        "integration",
        "author",
        "author_association",
        "assignee",
        "comments",
        "created_at",
        "closed_at",
    ]

    # Write using utf-8 to preserve international characters in issue bodies
    with open(target_path, "w", encoding="utf-8", newline="") as f:
        # csv.QUOTE_ALL explicitly encapsulates every field in double quotes.
        # This protects multiline issue bodies containing commas, quotes, and newlines from parsing issues.
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()

        for issue in issues:
            flat_row = flatten_issue(issue)
            writer.writerow(flat_row)

    print(f"[OK] Flat CSV saved to: {target_path} ({len(issues)} rows)")
    return target_path


def print_summary(issues: List[Dict[str, Any]], total_skipped_prs: int) -> None:
    """
    Print a clean summary of ingested issues:
    - Total rows
    - Open vs Closed breakdown
    - Issues with an integration label
    - Skipped pull requests
    """
    total_rows = len(issues)
    open_count = sum(1 for issue in issues if issue.get("state") == "open")
    closed_count = sum(1 for issue in issues if issue.get("state") == "closed")

    # Count issues with at least one label containing 'integration' (e.g., 'integration: mqtt')
    integration_label_count = 0
    for issue in issues:
        labels = issue.get("labels", [])
        has_integration = any(
            isinstance(lbl, dict) and "integration" in lbl.get("name", "").lower() for lbl in labels
        )
        if has_integration:
            integration_label_count += 1

    integration_pct = (integration_label_count / total_rows * 100) if total_rows > 0 else 0

    print("\n" + "=" * 50)
    print("           INGESTION SUMMARY")
    print("=" * 50)
    print(f"Total rows collected:        {total_rows}")
    print(f"  - Open issues:             {open_count}")
    print(f"  - Closed issues:           {closed_count}")
    print(f"Issues with integration tag: {integration_label_count} ({integration_pct:.1f}%)")
    print(f"Pull requests skipped:       {total_skipped_prs}")
    print("=" * 50)


def main() -> None:
    """
    Main orchestration routine.
    """
    print("Starting GitHub issue ingestion...")
    headers = get_auth_headers()

    # Step 1: Fetch 200 open issues
    open_issues, open_prs_skipped = fetch_issues_for_state("open", TARGET_ISSUES_PER_STATE, headers)

    # Step 2: Fetch 200 closed issues
    closed_issues, closed_prs_skipped = fetch_issues_for_state(
        "closed", TARGET_ISSUES_PER_STATE, headers
    )

    # Combine both datasets (total 400 issues when API targets are reached)
    all_issues = open_issues + closed_issues
    total_skipped_prs = open_prs_skipped + closed_prs_skipped

    if not all_issues:
        print("[WARN] No issues were retrieved. Check your network or GitHub token permissions.")
        return

    # Step 3: Save raw JSON archive
    save_raw_json(all_issues)

    # Step 4: Save flattened CSV
    save_flattened_csv(all_issues)

    # Step 5: Print summary report
    print_summary(all_issues, total_skipped_prs)


if __name__ == "__main__":
    main()
