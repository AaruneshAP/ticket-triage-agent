"""Historical Issue Backfill Pipeline for home-assistant/core.

Fetches historical issues within the specified window (default 6 months),
filters out PRs, handles rate limits, checkpoints progress every 10 pages,
saves immutable raw JSON in data/raw/, and generates normalized parquet in data/interim/.
"""

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from triage.ingest.client import UPSTREAM_OWNER, UPSTREAM_REPO, GitHubClient
from triage.parse.form_parser import parse_issue

DATA_RAW_DIR = os.path.join("data", "raw")
DATA_INTERIM_DIR = os.path.join("data", "interim")
CHECKPOINT_FILE = os.path.join(DATA_RAW_DIR, "backfill_checkpoint.json")
DEFAULT_PARQUET_PATH = os.path.join(DATA_INTERIM_DIR, "issues.parquet")
DEFAULT_CSV_PATH = os.path.join("data", "issues_6m.csv")
ITEMS_PER_PAGE = 100
CHECKPOINT_EVERY_N_PAGES = 10


def get_cutoff_date(months: int) -> datetime:
    """Compute UTC datetime threshold for N months ago."""
    days = int(months * 30.4375)
    return datetime.now(timezone.utc) - timedelta(days=days)


def parse_iso_datetime(date_str: str) -> datetime:
    """Parse ISO 8601 timestamp string from GitHub API."""
    return datetime.fromisoformat(date_str.replace("Z", "+00:00"))


def load_checkpoint() -> Tuple[int, List[Dict[str, Any]], int]:
    """Load previously saved checkpoint if exists."""
    if not os.path.exists(CHECKPOINT_FILE):
        return 1, [], 0

    try:
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        next_page = data.get("next_page", 1)
        collected = data.get("collected_issues", [])
        skipped_prs = data.get("skipped_prs", 0)
        print(
            f"[RESUME] Loaded checkpoint at page {next_page} with {len(collected)} issues collected."
        )
        return next_page, collected, skipped_prs
    except Exception as e:
        print(f"[WARN] Failed to load checkpoint ({e}). Starting fresh from page 1.")
        return 1, [], 0


def save_checkpoint(next_page: int, collected: List[Dict[str, Any]], skipped_prs: int) -> None:
    """Write current progress to disk to allow resumption."""
    os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
    payload = {
        "next_page": next_page,
        "collected_issues": collected,
        "skipped_prs": skipped_prs,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    tmp_path = f"{CHECKPOINT_FILE}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, CHECKPOINT_FILE)
    print(f"  [CHECKPOINT] Saved progress at page {next_page - 1} ({len(collected)} issues).")


def clear_checkpoint() -> None:
    """Remove checkpoint file upon successful completion."""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            os.remove(CHECKPOINT_FILE)
            print("[CLEANUP] Checkpoint cleared.")
        except OSError:
            pass


def fetch_backfill(
    months: int = 6,
    client: Optional[GitHubClient] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    """Fetch issues newer than cutoff date using GitHub REST API."""
    if client is None:
        client = GitHubClient()

    cutoff_date = get_cutoff_date(months)
    cutoff_str = cutoff_date.strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"--> Target: Backfill issues created after {cutoff_str} (~{months} months)")

    next_page, collected, skipped_prs = load_checkpoint()
    seen_numbers = {item["number"] for item in collected if "number" in item}

    page = next_page
    pages_since_checkpoint = 0
    hit_cutoff = False

    while not hit_cutoff:
        params = {
            "state": "all",
            "sort": "created",
            "direction": "desc",
            "per_page": ITEMS_PER_PAGE,
            "page": page,
        }
        endpoint = f"repos/{UPSTREAM_OWNER}/{UPSTREAM_REPO}/issues"

        try:
            resp = client.get(endpoint, params=params)
        except Exception as e:
            print(f"[ERROR] Request failed on page {page}: {e}. Retrying in 10s...")
            import time

            time.sleep(10)
            continue

        if resp.status_code != 200:
            print(f"[API ERROR] HTTP {resp.status_code}: {resp.text}", file=sys.stderr)
            break

        items = resp.json()
        if not items:
            print(f"  No more issues returned at page {page}. Ingestion finished.")
            break

        page_issues = 0
        for item in items:
            created_str = item.get("created_at")
            if not created_str:
                continue

            created_dt = parse_iso_datetime(created_str)
            if created_dt < cutoff_date:
                hit_cutoff = True
                print(
                    f"[CUTOFF REACHED] Issue #{item.get('number')} created at {created_str} is older than cutoff."
                )
                break

            # Filter out Pull Requests
            if "pull_request" in item:
                skipped_prs += 1
                continue

            num = item.get("number")
            if num in seen_numbers:
                continue

            seen_numbers.add(num)
            collected.append(item)
            page_issues += 1

        print(
            f"  Page {page:02d}: +{page_issues} issues (total: {len(collected)}), skipped PRs: {skipped_prs}"
        )
        page += 1
        pages_since_checkpoint += 1

        if not hit_cutoff and pages_since_checkpoint >= CHECKPOINT_EVERY_N_PAGES:
            save_checkpoint(page, collected, skipped_prs)
            pages_since_checkpoint = 0

    clear_checkpoint()
    return collected, skipped_prs


def transform_issue_to_record(issue: Dict[str, Any]) -> Dict[str, Any]:
    """Convert raw GitHub issue JSON object into a flat record with form parsed features."""
    labels = issue.get("labels", [])
    label_names = [lbl["name"] for lbl in labels if isinstance(lbl, dict) and "name" in lbl]
    joined_labels = "|".join(label_names)

    # Extract integration from label
    integration = ""
    for name in label_names:
        if name.lower().startswith("integration:"):
            parts = name.split(":", 1)
            if len(parts) == 2:
                integration = parts[1].strip()
                break

    # Author
    user = issue.get("user") or {}
    author = user.get("login", "")
    author_type = user.get("type", "User")

    # Assignee
    assignee = (issue.get("assignee") or {}).get("login", "") if issue.get("assignee") else None

    # Bot closure detection (stale bot = 'home-assistant[bot]' or other bots)
    closed_by = issue.get("closed_by") or {}
    closed_by_user = closed_by.get("login", "")
    closed_by_type = closed_by.get("type", "")
    is_closed_by_bot = bool(
        closed_by_type.lower() == "bot"
        or closed_by_user.endswith("[bot]")
        or closed_by_user.lower().endswith("-bot")
        or "stale" in closed_by_user.lower()
    )

    body = issue.get("body") or ""
    title = issue.get("title") or ""
    parsed = parse_issue(body, title)

    return {
        "number": issue.get("number"),
        "title": title,
        "body": body,
        "state": issue.get("state"),
        "state_reason": issue.get("state_reason"),
        "labels": joined_labels if joined_labels else None,
        "integration": integration if integration else None,
        "author": author,
        "author_type": author_type,
        "author_association": issue.get("author_association", "NONE"),
        "assignee": assignee,
        "comments": issue.get("comments", 0),
        "created_at": issue.get("created_at"),
        "closed_at": issue.get("closed_at"),
        "updated_at": issue.get("updated_at"),
        "closed_by": closed_by_user if closed_by_user else None,
        "closed_by_bot": is_closed_by_bot,
        # Form features
        "is_template": parsed["is_template"],
        "form_problem": parsed["problem"],
        "form_version": parsed["version"],
        "ha_version_year": parsed["ha_version_year"],
        "ha_version_month": parsed["ha_version_month"],
        "ha_version_patch": parsed["ha_version_patch"],
        "has_last_working_version": parsed["has_last_working_version"],
        "form_last_working_version": parsed["last_working_version"],
        "form_install_type": parsed["install_type"],
        "form_integration": parsed["integration_in_form"],
        "has_traceback": parsed["has_traceback"],
        "has_logs": parsed["has_logs"],
        "n_code_blocks": parsed["n_code_blocks"],
        "n_urls": parsed["n_urls"],
        "n_missing_form_fields": parsed["n_missing_form_fields"],
        "refers_to_issue": parsed["refers_to_issue"],
    }


def issues_to_dataframe(issues: List[Dict[str, Any]]) -> pd.DataFrame:
    """Transform list of raw issues into a validated DataFrame."""
    records = [transform_issue_to_record(issue) for issue in issues]
    df = pd.DataFrame(records)

    # Validate no duplicate numbers
    initial_len = len(df)
    df = df.drop_duplicates(subset=["number"]).reset_index(drop=True)
    if len(df) < initial_len:
        print(f"[DEDUP] Removed {initial_len - len(df)} duplicate issues.")

    return df


def print_ingestion_summary(df: pd.DataFrame, skipped_prs: int = 0) -> None:
    """Print standard ingestion summary for collected issues."""
    total_rows = len(df)
    open_count = int((df["state"] == "open").sum())
    closed_count = int((df["state"] == "closed").sum())
    labeled_count = int(df["integration"].notna().sum())
    labeled_pct = (labeled_count / total_rows * 100) if total_rows > 0 else 0
    bot_closed_count = int(df["closed_by_bot"].sum())

    closed_df = df[df["state"] == "closed"]
    reason_counts = Counter(closed_df["state_reason"].fillna("unspecified").str.lower())

    print("\n" + "=" * 65)
    print("                 INGESTION & BACKFILL SUMMARY")
    print("=" * 65)
    print(f"Total Unique Issues:             {total_rows}")
    print(f"  - Open Issues:                 {open_count}")
    print(f"  - Closed Issues:               {closed_count}")
    print(f"Issues with Integration Label:   {labeled_count} ({labeled_pct:.1f}%)")
    print(f"Closed by Stale/Triage Bot:      {bot_closed_count}")
    if skipped_prs > 0:
        print(f"Pull Requests Filtered Out:      {skipped_prs}")
    print("\nClosed Issues State Reason Breakdown:")
    for reason, count in reason_counts.most_common():
        pct = (count / closed_count * 100) if closed_count > 0 else 0
        print(f"  - {reason:<28} {count:>5} ({pct:5.1f}%)")
    print("=" * 65 + "\n")


def save_raw_json(issues: List[Dict[str, Any]]) -> str:
    """Save immutable raw JSON snapshot with timestamp."""
    os.makedirs(DATA_RAW_DIR, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(DATA_RAW_DIR, f"backfill_{timestamp}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(issues, f, indent=2, ensure_ascii=False)
    print(f"[RAW SAVED] Immutable JSON dump: {out_path} ({len(issues)} issues)")
    return out_path


def save_interim_parquet(df: pd.DataFrame, out_path: str = DEFAULT_PARQUET_PATH) -> None:
    """Save normalized dataset to Apache Parquet."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False, engine="pyarrow")
    print(f"[PARQUET SAVED] Interim dataset: {out_path} ({len(df)} rows)")


def save_csv(df: pd.DataFrame, out_path: str = DEFAULT_CSV_PATH) -> None:
    """Save flat CSV dataset."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[CSV SAVED] Flat CSV: {out_path} ({len(df)} rows)")


def run_backfill(
    months: int = 6,
    from_raw: Optional[str] = None,
    save_raw: bool = True,
) -> pd.DataFrame:
    """Orchestrate backfill pipeline."""
    if from_raw and os.path.exists(from_raw):
        print(f"[OFFLINE] Loading raw data from: {from_raw}")
        with open(from_raw, "r", encoding="utf-8") as f:
            issues = json.load(f)
        skipped_prs = 0
    else:
        issues, skipped_prs = fetch_backfill(months)
        if save_raw:
            save_raw_json(issues)

    df = issues_to_dataframe(issues)
    save_interim_parquet(df)
    save_csv(df)
    print_ingestion_summary(df, skipped_prs)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill historical issues for home-assistant/core."
    )
    parser.add_argument(
        "--months", type=int, default=6, help="Months of historical issues (default: 6)"
    )
    parser.add_argument(
        "--from-raw",
        type=str,
        default=None,
        help="Process an existing raw JSON file instead of querying API",
    )
    args = parser.parse_args()

    run_backfill(months=args.months, from_raw=args.from_raw)


if __name__ == "__main__":
    main()
