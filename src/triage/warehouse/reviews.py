"""Review Feedback Persistence Layer for Databricks and Local Fallback.

Stores human reviewer decisions (Approve, Edit, Reject) from the Streamlit
review app into Databricks `workspace.raw.reviews` and local Parquet backup.
"""

import datetime
import logging
import os
import uuid
from typing import Any, Dict

import pandas as pd

from triage.warehouse.loader import (
    get_databricks_config,
    get_databricks_connection,
    is_databricks_configured,
)

logger = logging.getLogger(__name__)

LOCAL_REVIEWS_PARQUET = os.path.join("data", "interim", "reviews.parquet")


def ensure_reviews_table_databricks(
    cursor: Any, catalog: str = "workspace", schema: str = "raw"
) -> None:
    """Ensure reviews table exists in Databricks."""
    cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
    create_sql = f"""
    CREATE TABLE IF NOT EXISTS {catalog}.{schema}.reviews (
        review_id STRING,
        ticket_number BIGINT,
        decision STRING,
        approved_draft STRING,
        reviewer_notes STRING,
        suggested_label STRING,
        reviewer STRING,
        reviewed_at TIMESTAMP
    )
    """
    cursor.execute(create_sql)


def save_review(
    ticket_number: int,
    decision: str,
    approved_draft: str,
    reviewer_notes: str = "",
    suggested_label: str = "",
    reviewer: str = "maintainer",
) -> Dict[str, Any]:
    """Save a human review decision to Databricks and local Parquet."""
    now = datetime.datetime.now(datetime.timezone.utc)
    review_record = {
        "review_id": str(uuid.uuid4())[:12],
        "ticket_number": int(ticket_number),
        "decision": str(decision).upper(),
        "approved_draft": str(approved_draft),
        "reviewer_notes": str(reviewer_notes),
        "suggested_label": str(suggested_label),
        "reviewer": str(reviewer),
        "reviewed_at": now.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # 1. Always save to local Parquet backup
    os.makedirs(os.path.dirname(LOCAL_REVIEWS_PARQUET), exist_ok=True)
    if os.path.exists(LOCAL_REVIEWS_PARQUET):
        try:
            existing_df = pd.read_parquet(LOCAL_REVIEWS_PARQUET)
            updated_df = pd.concat([existing_df, pd.DataFrame([review_record])], ignore_index=True)
        except Exception as e:
            logger.warning(f"Error reading existing reviews parquet: {e}. Starting fresh.")
            updated_df = pd.DataFrame([review_record])
    else:
        updated_df = pd.DataFrame([review_record])

    updated_df.to_parquet(LOCAL_REVIEWS_PARQUET, index=False)
    logger.info(f"Saved review for #{ticket_number} locally in {LOCAL_REVIEWS_PARQUET}")

    # 2. Save to Databricks if configured
    databricks_status = "skipped_not_configured"
    if is_databricks_configured():
        cfg = get_databricks_config()
        catalog = cfg["catalog"]
        schema = cfg["schema"]
        try:
            with get_databricks_connection(cfg) as conn:
                with conn.cursor() as cur:
                    ensure_reviews_table_databricks(cur, catalog, schema)
                    safe_draft = review_record["approved_draft"].replace("'", "''")
                    safe_notes = review_record["reviewer_notes"].replace("'", "''")
                    safe_label = review_record["suggested_label"].replace("'", "''")

                    insert_sql = f"""
                    INSERT INTO {catalog}.{schema}.reviews (
                        review_id, ticket_number, decision, approved_draft, reviewer_notes, suggested_label, reviewer, reviewed_at
                    ) VALUES (
                        '{review_record["review_id"]}',
                        {review_record["ticket_number"]},
                        '{review_record["decision"]}',
                        '{safe_draft}',
                        '{safe_notes}',
                        '{safe_label}',
                        '{review_record["reviewer"]}',
                        TIMESTAMP '{review_record["reviewed_at"]}'
                    )
                    """
                    cur.execute(insert_sql)
                    databricks_status = "persisted_to_databricks"
                    logger.info(
                        f"Persisted review for #{ticket_number} to Databricks {catalog}.{schema}.reviews"
                    )
        except Exception as e:
            databricks_status = f"databricks_error: {str(e)[:100]}"
            logger.warning(f"Could not persist review to Databricks: {e}")

    review_record["databricks_status"] = databricks_status
    return review_record


def load_recent_reviews(limit: int = 50) -> pd.DataFrame:
    """Load latest human review decisions from Databricks or local fallback."""
    if is_databricks_configured():
        cfg = get_databricks_config()
        catalog = cfg["catalog"]
        schema = cfg["schema"]
        try:
            with get_databricks_connection(cfg) as conn:
                with conn.cursor() as cur:
                    query = f"SELECT * FROM {catalog}.{schema}.reviews ORDER BY reviewed_at DESC LIMIT {limit}"
                    cur.execute(query)
                    rows = cur.fetchall()
                    if rows:
                        cols = [d[0] for d in cur.description]
                        return pd.DataFrame(rows, columns=cols)
        except Exception as e:
            logger.warning(
                f"Failed to fetch reviews from Databricks ({e}), falling back to local file."
            )

    if os.path.exists(LOCAL_REVIEWS_PARQUET):
        try:
            df = pd.read_parquet(LOCAL_REVIEWS_PARQUET)
            return df.tail(limit).iloc[::-1]
        except Exception as e:
            logger.error(f"Error loading local reviews: {e}")

    return pd.DataFrame(
        columns=[
            "review_id",
            "ticket_number",
            "decision",
            "approved_draft",
            "reviewer_notes",
            "suggested_label",
            "reviewer",
            "reviewed_at",
        ]
    )
