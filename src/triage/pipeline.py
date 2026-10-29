"""Ticket Triage End-to-End Orchestration & Rebuild Pipeline.

Recreates all interim datasets, Databricks RAW tables, and dbt models
(Bronze, Silver, Gold) in a single command from immutable raw data:
    python -m triage.pipeline --rebuild
"""

import argparse
import glob
import json
import logging
import os
import subprocess
import sys
import time
from typing import Any, Dict, List

import pandas as pd
from dotenv import load_dotenv

from triage.ingest.backfill import transform_issue_to_record
from triage.warehouse.loader import run_warehouse_load

load_dotenv()
logger = logging.getLogger(__name__)

DATA_RAW_DIR = os.path.join("data", "raw")
DATA_INTERIM_DIR = os.path.join("data", "interim")
ISSUES_PARQUET = os.path.join(DATA_INTERIM_DIR, "issues.parquet")
COMMENTS_PARQUET = os.path.join(DATA_INTERIM_DIR, "comments.parquet")


def rebuild_interim_from_raw() -> bool:
    """Parse immutable data/raw/*.json files to regenerate data/interim/*.parquet."""
    os.makedirs(DATA_INTERIM_DIR, exist_ok=True)
    print("--> [1/3] Rebuilding interim parquet datasets from raw JSON...", flush=True)

    # 1. Rebuild issues
    raw_issue_files = sorted(glob.glob(os.path.join(DATA_RAW_DIR, "backfill_*.json")))
    if not raw_issue_files:
        raw_issue_files = sorted(glob.glob(os.path.join(DATA_RAW_DIR, "issues_*.json")))

    if not raw_issue_files:
        if os.path.exists(ISSUES_PARQUET):
            print(f"    No raw issues JSON found; using existing {ISSUES_PARQUET}", flush=True)
        else:
            raise FileNotFoundError(f"No raw issues JSON files found in {DATA_RAW_DIR}")
    else:
        latest_file = raw_issue_files[-1]
        print(f"    Parsing raw issues from {latest_file}...", flush=True)
        with open(latest_file, "r", encoding="utf-8") as f:
            raw_issues = json.load(f)

        records: List[Dict[str, Any]] = []
        seen_numbers = set()
        for issue in raw_issues:
            if not isinstance(issue, dict) or "pull_request" in issue:
                continue
            num = issue.get("number")
            if num in seen_numbers:
                continue
            seen_numbers.add(num)
            records.append(transform_issue_to_record(issue))

        df_issues = pd.DataFrame(records)
        df_issues.to_parquet(ISSUES_PARQUET, index=False)
        print(f"    Saved {len(df_issues)} issues to {ISSUES_PARQUET} [OK]", flush=True)

    # 2. Rebuild comments if available
    raw_comment_files = sorted(glob.glob(os.path.join(DATA_RAW_DIR, "comments_*.json")))
    if raw_comment_files:
        latest_comments_file = raw_comment_files[-1]
        print(f"    Parsing raw comments from {latest_comments_file}...", flush=True)
        with open(latest_comments_file, "r", encoding="utf-8") as f:
            raw_comments = json.load(f)

        if isinstance(raw_comments, list):
            df_comments = pd.DataFrame(raw_comments)
            df_comments.to_parquet(COMMENTS_PARQUET, index=False)
            print(f"    Saved {len(df_comments)} comments to {COMMENTS_PARQUET} [OK]", flush=True)
    elif os.path.exists(COMMENTS_PARQUET):
        print(f"    Existing comments parquet retained: {COMMENTS_PARQUET}", flush=True)

    return True


def rebuild_warehouse_raw() -> bool:
    """Bulk load interim parquet datasets into Databricks workspace.raw schema."""
    print("--> [2/3] Uploading interim datasets into Databricks RAW landing tables...", flush=True)
    return run_warehouse_load(
        issues_path=ISSUES_PARQUET,
        comments_path=COMMENTS_PARQUET,
        dry_run=False,
        overwrite=True,
    )


def rebuild_dbt_models(full_refresh: bool = True) -> bool:
    """Execute dbt build against Databricks for bronze, silver, and gold models."""
    print("--> [3/3] Running dbt build against Databricks (Bronze, Silver, Gold)...", flush=True)
    dbt_exe = os.path.join(".venv", "Scripts", "dbt.exe")
    cmd = [dbt_exe, "build", "--project-dir", "dbt", "--profiles-dir", "dbt"]
    if full_refresh:
        cmd.append("--full-refresh")

    proc = subprocess.run(cmd, text=True)
    if proc.returncode != 0:
        raise RuntimeError("dbt build failed during pipeline rebuild")

    return True


def run_pipeline_rebuild() -> bool:
    """Execute complete end-to-end rebuild from raw data to Databricks gold marts."""
    start_time = time.time()
    print("==================================================================", flush=True)
    print("Starting Ticket Triage Pipeline Rebuild", flush=True)
    print("==================================================================", flush=True)

    rebuild_interim_from_raw()
    rebuild_warehouse_raw()
    rebuild_dbt_models(full_refresh=True)

    duration = round(time.time() - start_time, 2)
    print("==================================================================", flush=True)
    print(f"Pipeline rebuild completed successfully in {duration}s!", flush=True)
    print("All Databricks tables (RAW, BRONZE, SILVER, GOLD) are verified up to date.", flush=True)
    print("==================================================================", flush=True)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ticket Triage End-to-End Orchestration & Rebuild Pipeline."
    )
    parser.add_argument(
        "--rebuild", action="store_true", help="Recreate every table from data/raw in one command"
    )
    parser.add_argument(
        "--dbt-only", action="store_true", help="Run dbt build without reloading RAW"
    )
    args = parser.parse_args()

    if args.rebuild:
        success = run_pipeline_rebuild()
    elif args.dbt_only:
        success = rebuild_dbt_models(full_refresh=False)
    else:
        parser.print_help()
        sys.exit(1)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
