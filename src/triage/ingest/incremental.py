"""Incremental Ingestion Pipeline for home-assistant/core issues.

Polls GitHub API for issues updated since the last run ('since' parameter),
filters out PRs, transforms records, and upserts them into the interim dataset,
strictly deduplicating by issue number.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from triage.ingest.backfill import (
    DEFAULT_CSV_PATH,
    DEFAULT_PARQUET_PATH,
    print_ingestion_summary,
    save_csv,
    save_interim_parquet,
    transform_issue_to_record,
)
from triage.ingest.client import UPSTREAM_OWNER, UPSTREAM_REPO, GitHubClient

DATA_RAW_DIR = os.path.join("data", "raw")
DATA_INTERIM_DIR = os.path.join("data", "interim")
INCREMENTAL_CHECKPOINT_FILE = os.path.join(DATA_RAW_DIR, "incremental_checkpoint.json")


def get_latest_sync_timestamp(parquet_path: str = DEFAULT_PARQUET_PATH) -> Optional[str]:
    """Retrieve the latest updated_at or created_at timestamp from existing dataset or checkpoint."""
    if os.path.exists(INCREMENTAL_CHECKPOINT_FILE):
        try:
            with open(INCREMENTAL_CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                ts = data.get("last_sync_timestamp")
                if ts:
                    return ts
        except Exception:
            pass

    if os.path.exists(parquet_path):
        try:
            df = pd.read_parquet(parquet_path)
            if not df.empty and "updated_at" in df.columns:
                latest = df["updated_at"].dropna().max()
                if pd.notna(latest):
                    return str(latest)
            if not df.empty and "created_at" in df.columns:
                latest = df["created_at"].dropna().max()
                if pd.notna(latest):
                    return str(latest)
        except Exception as e:
            print(f"[WARN] Failed to read {parquet_path}: {e}")

    return None


def save_incremental_checkpoint(timestamp: str, items_synced: int) -> None:
    """Save latest sync timestamp."""
    os.makedirs(DATA_RAW_DIR, exist_ok=True)
    payload = {
        "last_sync_timestamp": timestamp,
        "items_synced": items_synced,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(INCREMENTAL_CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def fetch_incremental_issues(
    since_timestamp: str,
    client: Optional[GitHubClient] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    """Fetch all issues updated since given timestamp."""
    if client is None:
        client = GitHubClient()

    print(f"--> Target: Incremental fetch for issues updated since {since_timestamp}")
    endpoint = f"repos/{UPSTREAM_OWNER}/{UPSTREAM_REPO}/issues"
    page = 1
    collected: List[Dict[str, Any]] = []
    skipped_prs = 0

    while True:
        params = {
            "state": "all",
            "sort": "updated",
            "direction": "desc",
            "since": since_timestamp,
            "per_page": 100,
            "page": page,
        }

        try:
            resp = client.get(endpoint, params=params)
        except Exception as e:
            print(f"[ERROR] Incremental fetch failed at page {page}: {e}")
            break

        if resp.status_code != 200:
            print(f"[API ERROR] HTTP {resp.status_code}: {resp.text}", file=sys.stderr)
            break

        items = resp.json()
        if not items:
            break

        for item in items:
            if "pull_request" in item:
                skipped_prs += 1
                continue
            collected.append(item)

        print(
            f"  Page {page:02d}: +{len(items)} items processed ({len(collected)} issues, {skipped_prs} PRs skipped)"
        )
        if len(items) < 100:
            break
        page += 1

    return collected, skipped_prs


def upsert_issues(
    new_issues: List[Dict[str, Any]],
    parquet_path: str = DEFAULT_PARQUET_PATH,
) -> pd.DataFrame:
    """Upsert new/updated issues into existing parquet dataset and deduplicate."""
    new_records = [transform_issue_to_record(issue) for issue in new_issues]
    df_new = pd.DataFrame(new_records)

    if os.path.exists(parquet_path):
        df_existing = pd.read_parquet(parquet_path)
        # Identify existing numbers being updated vs new additions
        updated_numbers = set(df_new["number"])
        df_retained = df_existing[~df_existing["number"].isin(updated_numbers)]
        df_combined = pd.concat([df_retained, df_new], ignore_index=True)
    else:
        df_combined = df_new

    # Deduplicate strictly by issue number, keeping the latest record
    df_deduped = df_combined.drop_duplicates(subset=["number"], keep="last").reset_index(drop=True)
    return df_deduped


def run_incremental_sync(
    client: Optional[GitHubClient] = None,
    parquet_path: str = DEFAULT_PARQUET_PATH,
    fallback_hours: int = 24,
) -> pd.DataFrame:
    """Execute end-to-end incremental synchronization."""
    since = get_latest_sync_timestamp(parquet_path)
    if not since:
        # Fallback to last N hours if no existing timestamp found
        since = (datetime.now(timezone.utc) - pd.Timedelta(hours=fallback_hours)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        print(
            f"[INFO] No previous sync timestamp found. Defaulting to last {fallback_hours} hours: {since}"
        )

    items, skipped_prs = fetch_incremental_issues(since, client)
    now_iso = datetime.now(timezone.utc).isoformat()

    if not items:
        print("[INFO] No new issue updates found since last checkpoint.")
        save_incremental_checkpoint(now_iso, 0)
        return pd.read_parquet(parquet_path) if os.path.exists(parquet_path) else pd.DataFrame()

    # Save immutable raw dump of incremental batch
    os.makedirs(DATA_RAW_DIR, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    raw_path = os.path.join(DATA_RAW_DIR, f"incremental_{timestamp}.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)
    print(f"[RAW SAVED] Raw incremental batch: {raw_path} ({len(items)} issues)")

    # Upsert and persist
    df_updated = upsert_issues(items, parquet_path)
    save_interim_parquet(df_updated, parquet_path)
    save_csv(df_updated, DEFAULT_CSV_PATH)
    save_incremental_checkpoint(now_iso, len(items))

    print_ingestion_summary(df_updated, skipped_prs)
    return df_updated


def main() -> None:
    parser = argparse.ArgumentParser(description="Incremental issue synchronization.")
    parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="Fallback sync window in hours if no prior timestamp exists",
    )
    args = parser.parse_args()

    run_incremental_sync(fallback_hours=args.hours)


if __name__ == "__main__":
    main()
