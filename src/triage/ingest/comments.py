"""Comments Ingestion Pipeline for home-assistant/core issues.

Fetches issue discussions with priority ordering (closed duplicate issues first
to establish duplicate gold evaluation sets), flags bot authors, extracts duplicate
references, and saves both immutable raw JSON and interim parquet datasets.
"""

import argparse
import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from triage.ingest.client import UPSTREAM_OWNER, UPSTREAM_REPO, GitHubClient

DATA_RAW_DIR = os.path.join("data", "raw")
DATA_INTERIM_DIR = os.path.join("data", "interim")
COMMENTS_CHECKPOINT_FILE = os.path.join(DATA_RAW_DIR, "comments_checkpoint.json")
DEFAULT_COMMENTS_PARQUET = os.path.join(DATA_INTERIM_DIR, "comments.parquet")
DEFAULT_DUPLICATES_PARQUET = os.path.join(DATA_INTERIM_DIR, "duplicate_pairs.parquet")

DUPLICATE_REF_PATTERN = re.compile(
    r"(?:duplicate\s+of|dup\s+of|superseded\s+by|closed\s+in\s+favor\s+of|same\s+issue\s+as|closing\s+as\s+duplicate\s+of)\s+#?(\d+)|(?:^|\s)#(\d+)",
    re.IGNORECASE,
)


def is_bot_user(user: Optional[Dict[str, Any]]) -> bool:
    """Return True if user is classified as a bot on GitHub."""
    if not user:
        return False
    user_type = (user.get("type") or "").lower()
    login = (user.get("login") or "").lower()
    return (
        user_type == "bot"
        or login.endswith("[bot]")
        or login.endswith("-bot")
        or login in ("github-actions", "home-assistant", "stale")
    )


def extract_duplicate_reference(comment_body: str) -> Optional[int]:
    """Parse comment body to find referenced canonical duplicate issue number."""
    if not comment_body:
        return None
    for match in DUPLICATE_REF_PATTERN.finditer(comment_body):
        num_str = match.group(1) or match.group(2)
        if num_str:
            try:
                return int(num_str)
            except ValueError:
                pass
    return None


def load_comments_checkpoint() -> Tuple[Set[int], List[Dict[str, Any]]]:
    """Load checkpoint of already processed issues and accumulated comments."""
    if not os.path.exists(COMMENTS_CHECKPOINT_FILE):
        return set(), []

    try:
        with open(COMMENTS_CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        processed = set(data.get("processed_issue_numbers", []))
        comments = data.get("comments", [])
        print(
            f"[RESUME] Comments checkpoint loaded: {len(processed)} issues already processed ({len(comments)} comments)."
        )
        return processed, comments
    except Exception as e:
        print(f"[WARN] Failed to load comments checkpoint ({e}). Starting fresh.")
        return set(), []


def save_comments_checkpoint(processed: Set[int], comments: List[Dict[str, Any]]) -> None:
    """Persist comments fetching checkpoint to disk."""
    os.makedirs(os.path.dirname(COMMENTS_CHECKPOINT_FILE), exist_ok=True)
    payload = {
        "processed_issue_numbers": sorted(list(processed)),
        "comments": comments,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    tmp_path = f"{COMMENTS_CHECKPOINT_FILE}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, COMMENTS_CHECKPOINT_FILE)


def fetch_comments_for_issue(
    issue_number: int,
    client: GitHubClient,
) -> List[Dict[str, Any]]:
    """Fetch all comments for a specific issue number."""
    endpoint = f"repos/{UPSTREAM_OWNER}/{UPSTREAM_REPO}/issues/{issue_number}/comments"
    params = {"per_page": 100}
    comments: List[Dict[str, Any]] = []
    page = 1

    while True:
        params["page"] = page
        try:
            resp = client.get(endpoint, params=params)
        except Exception as e:
            print(f"[WARN] Error fetching comments for #{issue_number} (page {page}): {e}")
            break

        if resp.status_code != 200:
            break

        items = resp.json()
        if not items or not isinstance(items, list):
            break

        comments.extend(items)
        if len(items) < 100:
            break
        page += 1

    return comments


def run_comments_ingestion(
    issues_parquet_path: str = os.path.join(DATA_INTERIM_DIR, "issues.parquet"),
    max_issues: Optional[int] = None,
    client: Optional[GitHubClient] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch comments for issues, prioritizing closed duplicates first.
    Generates data/interim/comments.parquet and data/interim/duplicate_pairs.parquet.
    """
    if client is None:
        client = GitHubClient()

    if not os.path.exists(issues_parquet_path):
        raise FileNotFoundError(
            f"Interim issues parquet not found at: {issues_parquet_path}. Run backfill first."
        )

    df_issues = pd.read_parquet(issues_parquet_path)
    # Order issues: closed duplicates first, then issues with comments > 0
    duplicate_mask = (df_issues["state_reason"] == "duplicate") | (
        df_issues["labels"].fillna("").str.contains("duplicate", case=False)
    )
    has_comments_mask = df_issues["comments"] > 0

    priority_duplicates = df_issues[duplicate_mask & has_comments_mask]["number"].tolist()
    other_issues = df_issues[~duplicate_mask & has_comments_mask]["number"].tolist()
    ordered_numbers = priority_duplicates + other_issues

    if max_issues:
        ordered_numbers = ordered_numbers[:max_issues]

    print(
        f"--> Target: Ingest comments for {len(ordered_numbers)} issues ({len(priority_duplicates)} duplicates prioritized)."
    )

    processed, collected_comments = load_comments_checkpoint()
    duplicate_pairs: List[Dict[str, Any]] = []

    count = 0
    checkpoint_frequency = 25

    for num in ordered_numbers:
        if num in processed:
            continue

        issue_comments = fetch_comments_for_issue(num, client)
        for c in issue_comments:
            c["issue_number"] = num
            collected_comments.append(c)

            # Check if this comment names a duplicate target
            body = c.get("body", "")
            dup_target = extract_duplicate_reference(body)
            if dup_target and dup_target != num:
                duplicate_pairs.append(
                    {
                        "query_issue": num,
                        "original_issue": dup_target,
                        "source": "comment",
                        "comment_author": (c.get("user") or {}).get("login"),
                        "created_at": c.get("created_at"),
                    }
                )

        processed.add(num)
        count += 1

        if count % 10 == 0 or count == len(ordered_numbers):
            print(
                f"  Processed {count}/{len(ordered_numbers)} issues | Total comments: {len(collected_comments)}"
            )

        if count % checkpoint_frequency == 0:
            save_comments_checkpoint(processed, collected_comments)

    # Final checkpoint save
    save_comments_checkpoint(processed, collected_comments)

    # Save immutable raw dump
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    os.makedirs(DATA_RAW_DIR, exist_ok=True)
    raw_comments_path = os.path.join(DATA_RAW_DIR, f"comments_{timestamp}.json")
    with open(raw_comments_path, "w", encoding="utf-8") as f:
        json.dump(collected_comments, f, indent=2, ensure_ascii=False)
    print(
        f"[RAW SAVED] Immutable comments dump: {raw_comments_path} ({len(collected_comments)} records)"
    )

    # Build comments DataFrame
    records = []
    for c in collected_comments:
        user = c.get("user") or {}
        records.append(
            {
                "comment_id": c.get("id"),
                "issue_number": c.get("issue_number"),
                "author": user.get("login"),
                "author_association": c.get("author_association", "NONE"),
                "is_bot": is_bot_user(user),
                "body": c.get("body"),
                "created_at": c.get("created_at"),
                "updated_at": c.get("updated_at"),
                "duplicate_of_issue": extract_duplicate_reference(c.get("body", "")),
            }
        )

    df_comments = pd.DataFrame(records)
    if not df_comments.empty:
        df_comments = df_comments.drop_duplicates(subset=["comment_id"]).reset_index(drop=True)
    os.makedirs(DATA_INTERIM_DIR, exist_ok=True)
    df_comments.to_parquet(DEFAULT_COMMENTS_PARQUET, index=False)
    print(f"[PARQUET SAVED] Interim comments: {DEFAULT_COMMENTS_PARQUET} ({len(df_comments)} rows)")

    # Build duplicate pairs DataFrame
    df_dups = pd.DataFrame(duplicate_pairs)
    if not df_dups.empty:
        df_dups = df_dups.drop_duplicates(subset=["query_issue", "original_issue"]).reset_index(
            drop=True
        )
    df_dups.to_parquet(DEFAULT_DUPLICATES_PARQUET, index=False)
    print(
        f"[PARQUET SAVED] Duplicate gold pairs: {DEFAULT_DUPLICATES_PARQUET} ({len(df_dups)} pairs)"
    )

    return df_comments, df_dups


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch comments for home-assistant/core issues.")
    parser.add_argument(
        "--max-issues", type=int, default=None, help="Limit number of issues to fetch comments for"
    )
    args = parser.parse_args()

    run_comments_ingestion(max_issues=args.max_issues)


if __name__ == "__main__":
    main()
