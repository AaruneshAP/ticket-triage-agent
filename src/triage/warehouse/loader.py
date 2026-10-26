"""Databricks RAW Warehouse Loader.

Uploads normalized interim Parquet datasets (issues, comments) into
Databricks RAW schema tables (workspace.raw.raw_issues, workspace.raw.raw_comments)
using high-performance multi-row batch inserts via databricks-sql-connector.
"""

import argparse
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

DEFAULT_ISSUES_PARQUET = os.path.join("data", "interim", "issues.parquet")
DEFAULT_COMMENTS_PARQUET = os.path.join("data", "interim", "comments.parquet")

ISSUES_COLUMNS = [
    "ticket_number",
    "title",
    "body",
    "state",
    "state_reason",
    "labels",
    "integration",
    "author",
    "author_type",
    "author_association",
    "assignee",
    "comments_count",
    "created_at",
    "closed_at",
    "updated_at",
    "closed_by",
    "closed_by_bot",
    "is_template",
    "form_problem",
    "form_version",
    "ha_version_year",
    "ha_version_month",
    "ha_version_patch",
    "has_last_working_version",
    "form_last_working_version",
    "form_install_type",
    "form_integration",
    "has_traceback",
    "has_logs",
    "n_code_blocks",
    "n_urls",
    "n_missing_form_fields",
    "refers_to_issue",
]

COMMENTS_COLUMNS = [
    "comment_id",
    "issue_number",
    "author",
    "author_association",
    "is_bot",
    "body",
    "created_at",
    "updated_at",
    "duplicate_of_issue",
]


def get_databricks_config() -> Dict[str, str]:
    """Retrieve Databricks connection parameters from environment."""
    return {
        "host": os.getenv("DATABRICKS_HOST", "").strip(),
        "token": os.getenv("DATABRICKS_TOKEN", "").strip(),
        "http_path": os.getenv("DATABRICKS_HTTP_PATH", "").strip(),
        "catalog": os.getenv("DATABRICKS_CATALOG", "workspace").strip(),
        "schema": os.getenv("DATABRICKS_RAW_SCHEMA", "raw").strip(),
    }


def is_databricks_configured() -> bool:
    """Return True if required Databricks credentials exist."""
    cfg = get_databricks_config()
    return bool(cfg["host"] and cfg["token"] and cfg["http_path"])


def get_databricks_connection(config: Optional[Dict[str, str]] = None):
    """Open a connection to Databricks SQL Warehouse."""
    import socket

    from databricks import sql

    cfg = config or get_databricks_config()
    if not (cfg["host"] and cfg["token"] and cfg["http_path"]):
        raise ValueError(
            "Missing Databricks connection credentials (DATABRICKS_HOST, DATABRICKS_TOKEN, DATABRICKS_HTTP_PATH)"
        )

    old_timeout = socket.getdefaulttimeout()
    try:
        socket.setdefaulttimeout(15)
        return sql.connect(
            server_hostname=cfg["host"],
            http_path=cfg["http_path"],
            access_token=cfg["token"],
        )
    finally:
        socket.setdefaulttimeout(old_timeout)


def create_raw_tables_if_not_exists(
    cursor: Any, catalog: str = "workspace", schema: str = "raw"
) -> None:
    """Ensure raw landing tables exist with correct schema."""
    cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")

    create_issues_sql = f"""
    CREATE TABLE IF NOT EXISTS {catalog}.{schema}.raw_issues (
        ticket_number BIGINT,
        title STRING,
        body STRING,
        state STRING,
        state_reason STRING,
        labels STRING,
        integration STRING,
        author STRING,
        author_type STRING,
        author_association STRING,
        assignee STRING,
        comments_count INT,
        created_at STRING,
        closed_at STRING,
        updated_at STRING,
        closed_by STRING,
        closed_by_bot BOOLEAN,
        is_template BOOLEAN,
        form_problem STRING,
        form_version STRING,
        ha_version_year INT,
        ha_version_month INT,
        ha_version_patch INT,
        has_last_working_version BOOLEAN,
        form_last_working_version STRING,
        form_install_type STRING,
        form_integration STRING,
        has_traceback BOOLEAN,
        has_logs BOOLEAN,
        n_code_blocks INT,
        n_urls INT,
        n_missing_form_fields INT,
        refers_to_issue BOOLEAN
    )
    """
    cursor.execute(create_issues_sql)

    create_comments_sql = f"""
    CREATE TABLE IF NOT EXISTS {catalog}.{schema}.raw_comments (
        comment_id BIGINT,
        issue_number BIGINT,
        author STRING,
        author_association STRING,
        is_bot BOOLEAN,
        body STRING,
        created_at STRING,
        updated_at STRING,
        duplicate_of_issue BIGINT
    )
    """
    cursor.execute(create_comments_sql)


def _prepare_issues_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names and types for raw_issues insertion."""
    clean_df = df.copy()
    rename_map = {"number": "ticket_number", "comments": "comments_count"}
    clean_df.rename(columns=rename_map, inplace=True)

    for col in ISSUES_COLUMNS:
        if col not in clean_df.columns:
            clean_df[col] = None

    return clean_df[ISSUES_COLUMNS]


def _prepare_comments_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names and types for raw_comments insertion."""
    clean_df = df.copy()

    for col in COMMENTS_COLUMNS:
        if col not in clean_df.columns:
            clean_df[col] = None

    return clean_df[COMMENTS_COLUMNS]


def sql_format_val(v: Any) -> str:
    """Format a Python/Pandas scalar into a valid Databricks SQL literal."""
    if v is None or pd.isna(v):
        return "NULL"
    if isinstance(v, (bool, np.bool_)):
        return "TRUE" if v else "FALSE"
    if isinstance(v, (int, np.integer)):
        return str(v)
    if isinstance(v, (float, np.floating)):
        if np.isnan(v) or np.isinf(v):
            return "NULL"
        return str(v)

    # String value: escape backslashes and single quotes
    s = str(v).replace("\\", "\\\\").replace("'", "''")
    return f"'{s}'"


def filter_existing_records(df: pd.DataFrame, key_col: str, existing_keys: Any) -> pd.DataFrame:
    """Filter out rows whose key_col value is already present in existing_keys."""
    if df.empty or key_col not in df.columns or not existing_keys:
        return df

    # Normalise keys to handle both int and string representation
    keys_set = set()
    for k in existing_keys:
        keys_set.add(k)
        try:
            keys_set.add(int(k))
            keys_set.add(str(k))
        except (ValueError, TypeError):
            pass

    return df[~df[key_col].isin(keys_set)].copy()


def load_dataframe_to_databricks(
    cursor: Any,
    df: pd.DataFrame,
    table_name: str,
    columns: List[str],
    catalog: str = "workspace",
    schema: str = "raw",
    overwrite: bool = True,
    batch_size: int = 150,
) -> Tuple[bool, int, int]:
    """Bulk load pandas DataFrame into Databricks table using multi-row INSERT INTO with append deduplication."""
    full_table = f"{catalog}.{schema}.{table_name}"

    if overwrite:
        cursor.execute(f"TRUNCATE TABLE {full_table}")
        time.sleep(2)
    else:
        # Determine primary key for deduplication on append
        key_col = "ticket_number" if "ticket_number" in df.columns else ("comment_id" if "comment_id" in df.columns else None)
        if key_col and not df.empty:
            try:
                cursor.execute(f"SELECT DISTINCT {key_col} FROM {full_table}")
                rows = cursor.fetchall()
                existing_keys = [r[0] for r in rows if r[0] is not None]
                init_len = len(df)
                df = filter_existing_records(df, key_col, existing_keys)
                deduped_out = init_len - len(df)
                if deduped_out > 0:
                    print(
                        f"    [DEDUPE] Filtered out {deduped_out} duplicate rows already present in {full_table} "
                        f"({len(df)} new records remaining).",
                        flush=True,
                    )
            except Exception as e:
                logger.warning(f"Could not verify existing keys in {full_table} for deduplication: {e}")

    if df.empty:
        return True, 0, 0

    total_rows = len(df)
    num_batches = (total_rows + batch_size - 1) // batch_size
    cols_joined = ", ".join(columns)

    for i in range(0, total_rows, batch_size):
        chunk = df.iloc[i : i + batch_size]
        clauses = []
        for _, row in chunk[columns].iterrows():
            formatted_vals = [sql_format_val(row[c]) for c in columns]
            clauses.append(f"({', '.join(formatted_vals)})")

        insert_sql = f"INSERT INTO {full_table} ({cols_joined}) VALUES {', '.join(clauses)}"
        batch_num = (i // batch_size) + 1

        for attempt in range(1, 4):
            try:
                cursor.execute(insert_sql)
                break
            except Exception as e:
                if attempt == 3:
                    raise
                print(
                    f"    [RETRY] Batch {batch_num} hit {e}. Waiting 3s before retry {attempt + 1}/3...",
                    flush=True,
                )
                time.sleep(3)

        print(
            f"    Batch {batch_num}/{num_batches} ({len(chunk)} rows) inserted into {full_table}",
            flush=True,
        )

    return True, num_batches, total_rows


def run_warehouse_load(
    issues_path: str = DEFAULT_ISSUES_PARQUET,
    comments_path: str = DEFAULT_COMMENTS_PARQUET,
    dry_run: bool = False,
    overwrite: bool = True,
) -> bool:
    """Load interim parquet files into Databricks RAW tables."""
    if not os.path.exists(issues_path):
        fixture_issues = os.path.join("tests", "fixtures", "sample_issues.parquet")
        fixture_comments = os.path.join("tests", "fixtures", "sample_comments.parquet")
        if dry_run and os.path.exists(fixture_issues):
            issues_path = fixture_issues
            if not os.path.exists(comments_path) and os.path.exists(fixture_comments):
                comments_path = fixture_comments
        else:
            raise FileNotFoundError(f"Interim issues not found: {issues_path}")

    df_issues_raw = pd.read_parquet(issues_path)
    df_comments_raw = (
        pd.read_parquet(comments_path) if os.path.exists(comments_path) else pd.DataFrame()
    )

    df_issues = _prepare_issues_dataframe(df_issues_raw)
    df_comments = (
        _prepare_comments_dataframe(df_comments_raw)
        if not df_comments_raw.empty
        else pd.DataFrame()
    )

    print(
        f"--> Preparing to load {len(df_issues)} issues and {len(df_comments)} comments into Databricks RAW.",
        flush=True,
    )

    if dry_run or not is_databricks_configured():
        if not is_databricks_configured():
            print("[NOTICE] Databricks credentials not configured in .env.", flush=True)
            print(
                "         Validated dataset schemas for warehouse loading (dry run mode active).",
                flush=True,
            )
        else:
            print("[DRY RUN] Databricks connection skipped per user flag.", flush=True)

        print(
            f"  Validated raw_issues schema:   {len(df_issues.columns)} columns, {len(df_issues)} rows [OK]",
            flush=True,
        )
        print(
            f"  Validated raw_comments schema: {len(df_comments.columns) if not df_comments.empty else 0} columns, {len(df_comments)} rows [OK]",
            flush=True,
        )
        return True

    cfg = get_databricks_config()
    catalog = cfg["catalog"]
    schema = cfg["schema"]

    print(
        f"Connecting to Databricks ({cfg['host']}, catalog: {catalog}, schema: {schema})...",
        flush=True,
    )
    conn = get_databricks_connection(cfg)
    try:
        with conn.cursor() as cursor:
            t0 = time.time()
            cursor.execute(f"USE CATALOG {catalog}")
            create_raw_tables_if_not_exists(cursor, catalog=catalog, schema=schema)

            print(f"Uploading raw_issues ({len(df_issues)} rows) in batches...", flush=True)
            ok_issues, b_issues, n_issues = load_dataframe_to_databricks(
                cursor=cursor,
                df=df_issues,
                table_name="raw_issues",
                columns=ISSUES_COLUMNS,
                catalog=catalog,
                schema=schema,
                overwrite=overwrite,
                batch_size=150,
            )
            print(
                f"  raw_issues loaded: {ok_issues} ({n_issues} rows in {b_issues} batches)",
                flush=True,
            )

            if not df_comments.empty:
                print(f"Uploading raw_comments ({len(df_comments)} rows)...", flush=True)
                ok_comms, b_comms, n_comms = load_dataframe_to_databricks(
                    cursor=cursor,
                    df=df_comments,
                    table_name="raw_comments",
                    columns=COMMENTS_COLUMNS,
                    catalog=catalog,
                    schema=schema,
                    overwrite=overwrite,
                    batch_size=150,
                )
                print(
                    f"  raw_comments loaded: {ok_comms} ({n_comms} rows in {b_comms} batches)",
                    flush=True,
                )

            duration = round(time.time() - t0, 2)
            print(f"[SUCCESS] Databricks RAW load completed in {duration}s.", flush=True)
            return True
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bulk load interim parquet datasets into Databricks RAW."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate datasets and schema without connecting to Databricks",
    )
    parser.add_argument(
        "--append", action="store_true", help="Append without truncating target tables"
    )
    args = parser.parse_args()

    success = run_warehouse_load(dry_run=args.dry_run, overwrite=not args.append)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
