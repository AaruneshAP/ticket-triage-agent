"""
GitHub Historical Issue Backfill for home-assistant/core.

Fetches all issues (open and closed) created within the last N months,
filters out pull requests, checkpoints every 10 pages to allow resuming,
and exports both raw JSON and a flat CSV with full quoting.
"""

import csv
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple

import requests
from dotenv import load_dotenv

# Load credentials from .env
load_dotenv()

# ==========================================
# CONFIGURATION SETTINGS
# ==========================================
# Number of months of historical issues to backfill
MONTHS_BACK = 6

# Checkpoint frequency: save progress every N pages to survive interrupts
CHECKPOINT_EVERY_N_PAGES = 10

# Target repository coordinates
REPO_OWNER = "home-assistant"
REPO_NAME = "core"
GITHUB_API_BASE = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}"

# Maximum allowed items per page by GitHub REST API
ITEMS_PER_PAGE = 100

# File paths
CHECKPOINT_FILE = os.path.join("data", "raw", "backfill_checkpoint.json")
CSV_OUTPUT_PATH = os.path.join("data", "issues_6m.csv")


def get_cutoff_date(months: int) -> datetime:
    """
    Compute UTC datetime threshold for N months ago.
    Using average month length of 30.44 days to approximate calendar months.
    """
    days = int(months * 30.4375)
    return datetime.now(timezone.utc) - timedelta(days=days)


def parse_iso_datetime(date_str: str) -> datetime:
    """
    Parse an ISO 8601 timestamp string from GitHub API (e.g. '2026-05-12T14:32:00Z')
    into a timezone-aware UTC datetime object.
    """
    # Replace trailing 'Z' with explicit '+00:00' UTC offset for compatibility
    clean_str = date_str.replace("Z", "+00:00")
    return datetime.fromisoformat(clean_str)


def get_auth_headers() -> Dict[str, str]:
    """
    Read GITHUB_TOKEN from environment and construct GitHub API request headers.
    """
    token = os.getenv("GITHUB_TOKEN", "").strip()
    if not token:
        print(
            "[ERROR] GITHUB_TOKEN is not set in .env. Please configure it first.", file=sys.stderr
        )
        sys.exit(1)

    return {
        "X-GitHub-Api-Version": "2022-11-28",
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "User-Agent": "Ticket-Triage-Backfill/1.0",
    }


def load_checkpoint() -> Tuple[int, List[Dict[str, Any]], int]:
    """
    Attempt to load a previously saved checkpoint.
    Returns:
        (next_page_to_fetch, collected_issues_list, skipped_prs_count)
    """
    if not os.path.exists(CHECKPOINT_FILE):
        return 1, [], 0

    try:
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        next_page = data.get("next_page", 1)
        collected = data.get("collected_issues", [])
        skipped_prs = data.get("skipped_prs", 0)

        print(f"\n[RESUME] Found existing checkpoint at {CHECKPOINT_FILE}!")
        print(
            f"         Resuming from page {next_page} with {len(collected)} issues already collected.\n"
        )
        return next_page, collected, skipped_prs
    except Exception as e:
        print(f"[WARN] Failed to load checkpoint ({e}). Starting fresh from page 1.")
        return 1, [], 0


def save_checkpoint(next_page: int, collected: List[Dict[str, Any]], skipped_prs: int) -> None:
    """
    Write current progress to disk so runs can resume after interruptions.
    """
    os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
    payload = {
        "next_page": next_page,
        "collected_issues": collected,
        "skipped_prs": skipped_prs,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }

    # Write to a temporary file first then replace to avoid partial file corruption if interrupted mid-write
    temp_file = f"{CHECKPOINT_FILE}.tmp"
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(temp_file, CHECKPOINT_FILE)

    print(
        f"  [CHECKPOINT] Saved progress at page {next_page - 1} ({len(collected)} issues, {skipped_prs} PRs skipped)."
    )


def clear_checkpoint() -> None:
    """
    Remove temporary checkpoint file after the backfill completes successfully.
    """
    if os.path.exists(CHECKPOINT_FILE):
        try:
            os.remove(CHECKPOINT_FILE)
            print("[CLEANUP] Checkpoint file cleared after successful completion.")
        except OSError as e:
            print(f"[WARN] Could not remove checkpoint file: {e}")


def handle_rate_limits(response_headers: Dict[str, str]) -> None:
    """
    Check GitHub rate limit headers. If remaining requests are exhausted,
    sleep until the reset epoch timestamp.
    """
    remaining_str = response_headers.get("X-RateLimit-Remaining")
    reset_str = response_headers.get("X-RateLimit-Reset")

    if remaining_str is None or reset_str is None:
        return

    remaining = int(remaining_str)
    # If quota drops below 2 requests, pause until the reset window
    if remaining <= 2:
        reset_epoch = int(reset_str)
        current_epoch = int(time.time())
        # Sleep until reset plus a 2-second buffer
        wait_seconds = max(0, reset_epoch - current_epoch) + 2
        reset_time_str = datetime.fromtimestamp(reset_epoch, tz=timezone.utc).strftime(
            "%H:%M:%S UTC"
        )
        print(f"\n[RATE LIMIT REACHED] Sleeping for {wait_seconds}s (until {reset_time_str})...")
        time.sleep(wait_seconds)
        print("[RATE LIMIT RESET] Quota refreshed. Resuming requests.\n")


def fetch_backfill_issues(months: int, headers: Dict[str, str]) -> Tuple[List[Dict[str, Any]], int]:
    """
    Fetch all issues newer than `months` ago.
    Resumes from checkpoint if available, and checkpoints progress every 10 pages.
    """
    cutoff_date = get_cutoff_date(months)
    cutoff_display = cutoff_date.strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"--> Target: Fetch all issues created after {cutoff_display} (~{months} months ago)")

    # Restore from checkpoint or start fresh
    next_page, collected, skipped_prs = load_checkpoint()

    # Track seen issue numbers to guarantee no duplicate tickets if resuming
    seen_numbers = {item["number"] for item in collected if "number" in item}

    page = next_page
    pages_since_last_checkpoint = 0
    hit_cutoff = False

    while not hit_cutoff:
        params = {
            # state=all retrieves both open and closed issues in one chronological stream
            "state": "all",
            "sort": "created",
            "direction": "desc",
            "per_page": ITEMS_PER_PAGE,
            "page": page,
        }

        url = f"{GITHUB_API_BASE}/issues"
        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)
        except requests.RequestException as e:
            print(f"[NETWORK ERROR] Request failed on page {page}: {e}. Retrying in 10s...")
            time.sleep(10)
            continue

        # Inspect headers for rate limits and sleep if needed
        handle_rate_limits(dict(response.headers))

        if response.status_code == 403 or response.status_code == 429:
            print(f"[BLOCKED] Status {response.status_code}. Waiting 60 seconds...")
            time.sleep(60)
            continue

        if response.status_code != 200:
            print(
                f"[API ERROR] GitHub responded with {response.status_code}: {response.text}",
                file=sys.stderr,
            )
            break

        items = response.json()
        if not items:
            print(f"  No more items found at page {page}. Repository history exhausted.")
            break

        page_issues_added = 0
        page_prs_skipped = 0

        for item in items:
            created_str = item.get("created_at")
            if not created_str:
                continue

            item_created_at = parse_iso_datetime(created_str)

            # Because items are sorted by created descending, the first item older than cutoff
            # signals that all remaining items in this and future pages are also older.
            if item_created_at < cutoff_date:
                hit_cutoff = True
                print(
                    f"\n[CUTOFF REACHED] Encountered ticket #{item.get('number')} created at {created_str}."
                )
                print(f"                 This is older than cutoff {cutoff_display}. Stopping.")
                break

            # Skip pull requests
            if "pull_request" in item:
                skipped_prs += 1
                page_prs_skipped += 1
                continue

            # Deduplicate (useful when resuming)
            num = item.get("number")
            if num in seen_numbers:
                continue

            seen_numbers.add(num)
            collected.append(item)
            page_issues_added += 1

        print(
            f"  Page {page:02d}: +{page_issues_added} issues (total: {len(collected)}), "
            f"+{page_prs_skipped} PRs skipped | "
            f"Latest item on page: {items[-1].get('created_at', '')[:10]}"
        )

        page += 1
        pages_since_last_checkpoint += 1

        # Checkpoint every N pages (e.g., 10 pages) if cutoff has not yet ended the run
        if not hit_cutoff and pages_since_last_checkpoint >= CHECKPOINT_EVERY_N_PAGES:
            save_checkpoint(page, collected, skipped_prs)
            pages_since_last_checkpoint = 0

    return collected, skipped_prs


def flatten_issue(issue: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform a nested issue object into the standard flat 13-column schema.
    """
    labels = issue.get("labels", [])
    label_names = [lbl["name"] for lbl in labels if isinstance(lbl, dict) and "name" in lbl]
    joined_labels = "|".join(label_names)

    # Extract integration domain if an 'integration: <name>' label exists
    integration_name = ""
    for name in label_names:
        if name.lower().startswith("integration:"):
            integration_name = name.split(":", 1)[1].strip()
            break

    user_obj = issue.get("user") or {}
    assignee_obj = issue.get("assignee") or {}

    return {
        "number": issue.get("number"),
        "title": issue.get("title", ""),
        "body": issue.get("body") or "",
        "state": issue.get("state", ""),
        # state_reason records whether ticket was closed as completed, not_planned, etc.
        "state_reason": issue.get("state_reason") or "",
        "labels": joined_labels,
        "integration": integration_name,
        "author": user_obj.get("login", ""),
        "author_association": issue.get("author_association", ""),
        "assignee": assignee_obj.get("login", ""),
        "comments": issue.get("comments", 0),
        "created_at": issue.get("created_at", ""),
        "closed_at": issue.get("closed_at") or "",
    }


def save_raw_json(issues: List[Dict[str, Any]], target_dir: str = "data/raw") -> str:
    """
    Save the complete raw JSON responses to data/raw/backfill_<date>.json.
    """
    os.makedirs(target_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    output_path = os.path.join(target_dir, f"backfill_{date_str}.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(issues, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Raw JSON saved to: {output_path} ({len(issues)} items)")
    return output_path


def save_flattened_csv(issues: List[Dict[str, Any]], target_path: str = CSV_OUTPUT_PATH) -> str:
    """
    Flatten issues and write to CSV using csv.QUOTE_ALL for multiline integrity.
    """
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

    with open(target_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for issue in issues:
            writer.writerow(flatten_issue(issue))

    print(f"[OK] Flat CSV saved to: {target_path} ({len(issues)} rows)")
    # Clear the temporary checkpoint once everything is safely written
    clear_checkpoint()
    return target_path


def print_summary(issues: List[Dict[str, Any]], total_skipped_prs: int) -> None:
    """
    Print an exhaustive summary of the backfilled issues, including
    state breakdown and closed issue state_reason distribution.
    """
    from collections import Counter

    total_rows = len(issues)
    open_count = sum(1 for issue in issues if issue.get("state") == "open")
    closed_issues = [issue for issue in issues if issue.get("state") == "closed"]
    closed_count = len(closed_issues)

    # Count issues that have an integration label
    integration_count = 0
    for issue in issues:
        labels = issue.get("labels", [])
        if any(
            isinstance(lbl, dict) and "integration" in lbl.get("name", "").lower() for lbl in labels
        ):
            integration_count += 1

    integration_pct = (integration_count / total_rows * 100) if total_rows > 0 else 0

    # Count distribution of state_reason for closed issues
    # Older tickets or manual closes without reason return None/empty, categorized as 'unspecified'
    reason_counter = Counter(
        (issue.get("state_reason") or "unspecified").lower() for issue in closed_issues
    )

    print("\n" + "=" * 60)
    print(f"       HISTORICAL BACKFILL SUMMARY ({MONTHS_BACK} MONTHS)")
    print("=" * 60)
    print(f"Total Issues Ingested:           {total_rows}")
    print(f"  - Open issues:                 {open_count}")
    print(f"  - Closed issues:               {closed_count}")
    print(f"Issues with Integration Label:   {integration_count} ({integration_pct:.1f}%)")
    print(f"Pull Requests Skipped:           {total_skipped_prs}")
    print("\nClosed Issues State Reason Breakdown:")
    for reason, count in reason_counter.most_common():
        pct = (count / closed_count * 100) if closed_count > 0 else 0
        print(f"  - {reason:<28} {count:>5} ({pct:5.1f}%)")
    print("=" * 60)


def main() -> None:
    """
    Main orchestration routine.
    """
    print(f"Starting historical backfill for {REPO_OWNER}/{REPO_NAME}...")
    headers = get_auth_headers()

    # Step 1: Paged retrieval until cutoff date with checkpointing
    issues, skipped_prs = fetch_backfill_issues(MONTHS_BACK, headers)

    if not issues:
        print("[WARN] No issues were retrieved. Check your network or token permissions.")
        return

    # Step 2: Save raw JSON
    save_raw_json(issues)

    # Step 3: Save flat CSV (with csv.QUOTE_ALL)
    save_flattened_csv(issues)

    # Step 4: Display comprehensive metrics
    print_summary(issues, skipped_prs)


if __name__ == "__main__":
    main()
