"""Test verifying incremental dbt merges against Databricks."""

import os
import subprocess

import pytest
from dotenv import load_dotenv

load_dotenv()


@pytest.mark.integration
def test_dbt_incremental_update():
    from databricks import sql

    host = os.getenv("DATABRICKS_HOST")
    token = os.getenv("DATABRICKS_TOKEN")
    http_path = os.getenv("DATABRICKS_HTTP_PATH")

    assert host and token and http_path, "Databricks credentials required for incremental test"

    with sql.connect(server_hostname=host, http_path=http_path, access_token=token) as conn:
        with conn.cursor() as cur:
            # 1. Fetch current ticket
            cur.execute(
                "SELECT ticket_number, title FROM workspace.silver.stg_issues WHERE ticket_number = 187117"
            )
            row = cur.fetchone()
            assert row is not None, "Ticket 187117 not found in silver.stg_issues"
            orig_title = row[1]

            # 2. Update raw_issues with test prefix and current timestamp
            test_title = "[INCREMENTAL_TEST] Matter integration failure"
            cur.execute(f"""
                UPDATE workspace.raw.raw_issues
                SET title = '{test_title}',
                    updated_at = CAST(current_timestamp() AS STRING)
                WHERE ticket_number = 187117
            """)

    # 3. Execute dbt build on stg_issues and fct_tickets incrementally
    dbt_exe = os.path.join(".venv", "Scripts", "dbt.exe")
    cmd = [
        dbt_exe,
        "build",
        "--select",
        "stg_issues",
        "fct_tickets",
        "--project-dir",
        "dbt",
        "--profiles-dir",
        "dbt",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f"dbt incremental build failed: {proc.stdout}\n{proc.stderr}"

    # 4. Verify merged record in silver.stg_issues and gold.fct_tickets
    with sql.connect(server_hostname=host, http_path=http_path, access_token=token) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT title FROM workspace.silver.stg_issues WHERE ticket_number = 187117"
            )
            updated_silver_title = cur.fetchone()[0]
            assert updated_silver_title == test_title, (
                f"Silver model not updated: got {updated_silver_title}"
            )

            cur.execute(
                "SELECT title_len FROM workspace.gold.fct_tickets WHERE ticket_number = 187117"
            )
            updated_gold_title_len = cur.fetchone()[0]
            assert updated_gold_title_len == len(test_title), (
                f"Gold model title_len mismatch: got {updated_gold_title_len}"
            )

            # Check that count remains exact (no duplication)
            cur.execute(
                "SELECT COUNT(*) FROM workspace.silver.stg_issues WHERE ticket_number = 187117"
            )
            count_val = cur.fetchone()[0]
            assert count_val == 1, f"Expected 1 record, got {count_val}"

            # 5. Restore original title in raw_issues and re-sync
            cur.execute(f"""
                UPDATE workspace.raw.raw_issues
                SET title = '{orig_title.replace("'", "''")}',
                    updated_at = CAST(current_timestamp() AS STRING)
                WHERE ticket_number = 187117
            """)

    # Final sync back
    proc_restore = subprocess.run(cmd, capture_output=True, text=True)
    assert proc_restore.returncode == 0
    print("dbt incremental build and merge verified successfully against Databricks!")


if __name__ == "__main__":
    test_dbt_incremental_update()
