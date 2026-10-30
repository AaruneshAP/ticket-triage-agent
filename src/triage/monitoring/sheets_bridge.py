"""Google Sheets & Looker Studio Data Bridge (Phase 9).

Generates scheduled, structured data feeds formatted for direct ingestion by
Google Sheets and Looker Studio dashboards:
1. daily_triage_kpi.csv: Daily operational metrics, SLA breaches, completeness rates.
2. integration_health.csv: Per-integration volume, failure rates, and median close times.
3. model_performance_feed.csv: Historical C-Index, PR-AUC, and Brier scores.
4. drift_monitoring_feed.csv: Real-time feature and prediction PSI stability metrics.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SHEETS_EXPORT_DIR = os.path.join("data", "exports", "sheets")
FEATURES_PATH = os.path.join("data", "interim", "features.parquet")
PERFORMANCE_PATH = os.path.join("data", "monitoring", "performance_history.json")
DRIFT_PATH = os.path.join("data", "monitoring", "drift_metrics.json")


def generate_sheets_feeds(
    features_path: str = FEATURES_PATH,
    export_dir: str = SHEETS_EXPORT_DIR,
) -> Dict[str, Any]:
    """Generate all CSV feeds formatted for Google Sheets / Looker Studio."""
    os.makedirs(export_dir, exist_ok=True)
    generated_files = {}

    actual_features_path = features_path
    if not os.path.exists(actual_features_path):
        fallback = os.path.join("hf_space", "data", "snapshot_tickets.parquet")
        if os.path.exists(fallback):
            actual_features_path = fallback
        else:
            logger.warning(f"Features parquet not found at {features_path}.")
            return {"status": "error", "message": f"File not found: {features_path}"}

    df = pd.read_parquet(actual_features_path)
    if "created_at_dt" not in df.columns:
        df["created_at_dt"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)

    # 1. Daily Triage KPI Feed
    df["date"] = df["created_at_dt"].dt.strftime("%Y-%m-%d")
    daily_groups = df.groupby("date")

    daily_rows = []
    for dt, group in daily_groups:
        n_tickets = len(group)
        n_closed = int(group["target_is_closed_event"].sum()) if "target_is_closed_event" in group else 0
        n_not_planned = int(group["target_is_not_planned"].sum()) if "target_is_not_planned" in group else 0
        n_incomplete = int((group["n_missing_form_fields"] > 0).sum()) if "n_missing_form_fields" in group else 0
        avg_body = float(np.round(group["body_len"].mean(), 1)) if "body_len" in group else 0.0

        daily_rows.append({
            "Date": dt,
            "Total_Tickets": n_tickets,
            "Closed_Tickets": n_closed,
            "Not_Planned_Tickets": n_not_planned,
            "Incomplete_Tickets": n_incomplete,
            "Completeness_Rate_Pct": round((1.0 - (n_incomplete / n_tickets if n_tickets else 0)) * 100, 1),
            "Not_Planned_Rate_Pct": round((n_not_planned / n_tickets if n_tickets else 0) * 100, 1),
            "Avg_Body_Length_Words": avg_body,
        })

    daily_df = pd.DataFrame(daily_rows).sort_values("Date")
    daily_file = os.path.join(export_dir, "daily_triage_kpi.csv")
    daily_df.to_csv(daily_file, index=False)
    generated_files["daily_kpi"] = daily_file

    # 2. Integration Health Feed
    integ_col = "integration_clean" if "integration_clean" in df.columns else "integration"
    integ_groups = df.groupby(integ_col)
    integ_rows = []

    for integ, group in integ_groups:
        if str(integ).lower() in ["unlabeled", "unknown", "none", "nan"]:
            continue
        n_tot = len(group)
        if n_tot < 5:
            continue
        n_np = int(group["target_is_not_planned"].sum()) if "target_is_not_planned" in group else 0
        n_incomp = int((group["n_missing_form_fields"] > 0).sum()) if "n_missing_form_fields" in group else 0
        has_tb = float(group["has_traceback"].mean() * 100) if "has_traceback" in group else 0.0
        med_dur = float(group["duration_days"].median()) if "duration_days" in group else 0.0

        integ_rows.append({
            "Integration": integ,
            "Total_Issues": n_tot,
            "Incomplete_Rate_Pct": round((n_incomp / n_tot) * 100, 1),
            "Not_Planned_Rate_Pct": round((n_np / n_tot) * 100, 1),
            "Has_Traceback_Pct": round(has_tb, 1),
            "Median_Days_To_Close": round(med_dur, 2) if not np.isnan(med_dur) else 0.0,
        })

    integ_df = pd.DataFrame(integ_rows).sort_values("Total_Issues", ascending=False)
    integ_file = os.path.join(export_dir, "integration_health.csv")
    integ_df.to_csv(integ_file, index=False)
    generated_files["integration_health"] = integ_file

    # 3. Model Performance Feed
    perf_rows = []
    perf_path = PERFORMANCE_PATH if os.path.exists(PERFORMANCE_PATH) else os.path.join("hf_space", "data", "performance_history.json")
    if os.path.exists(perf_path):
        try:
            with open(perf_path, "r", encoding="utf-8") as f:
                perf_data = json.load(f)
            for c in perf_data.get("cohort_metrics", []):
                perf_rows.append({
                    "Window_Start": c.get("window_start"),
                    "Window_End": c.get("window_end"),
                    "Split": c.get("split", "test"),
                    "Sample_Type": c.get("sample_type", "TEST"),
                    "Resolved_Tickets": c.get("n_tickets", 0),
                    "Survival_C_Index": c.get("survival_c_index", 0.0),
                    "Not_Planned_PR_AUC": c.get("not_planned_pr_auc", 0.0),
                    "Not_Planned_Brier": c.get("not_planned_brier", 0.0),
                    "Label_Accuracy": c.get("label_accuracy", 0.0),
                })
        except Exception as e:
            logger.warning(f"Could not parse performance history: {e}")

    perf_df = pd.DataFrame(perf_rows)
    perf_file = os.path.join(export_dir, "model_performance_feed.csv")
    perf_df.to_csv(perf_file, index=False)
    generated_files["model_performance"] = perf_file

    # 4. Drift Monitoring Feed
    drift_rows = []
    drift_path = DRIFT_PATH if os.path.exists(DRIFT_PATH) else os.path.join("hf_space", "data", "drift_metrics.json")
    if os.path.exists(drift_path):
        try:
            with open(drift_path, "r", encoding="utf-8") as f:
                drift_data = json.load(f)
            ts = drift_data.get("timestamp", datetime.now(timezone.utc).isoformat())
            for feat, data in drift_data.get("features_analyzed", {}).items():
                # Suppress predicted resolution-time metrics (P50, P90, time-to-close)
                if any(k in feat.lower() for k in ["predicted_time_to_close", "p50", "p90", "resolution_time", "closure_prob", "time_to_close"]):
                    continue
                drift_rows.append({
                    "Evaluated_At": ts,
                    "Feature_Or_Metric": feat,
                    "Ref_Mean": data.get("ref_mean", 0.0),
                    "Current_Mean": data.get("curr_mean", 0.0),
                    "PSI_Score": data.get("psi", 0.0),
                    "KS_Statistic": data.get("ks_statistic", 0.0),
                    "Alert_Status": data.get("alert_level", "normal").upper(),
                })
        except Exception as e:
            logger.warning(f"Could not parse drift metrics: {e}")

    drift_df = pd.DataFrame(drift_rows)
    drift_file = os.path.join(export_dir, "drift_monitoring_feed.csv")
    drift_df.to_csv(drift_file, index=False)
    generated_files["drift_monitoring"] = drift_file

    logger.info(f"Generated 4 Google Sheets feeds in {export_dir}")
    return {
        "status": "success",
        "export_dir": export_dir,
        "files_generated": generated_files,
    }


def sync_to_google_sheets(
    spreadsheet_id: str | None = None,
    service_account_path: str | None = None,
) -> Dict[str, Any]:
    """Sync the 4 generated CSV feeds to Google Sheets tabs and read back verification."""
    from google.oauth2 import service_account
    from googleapiclient.discovery import build

    sa_path = service_account_path or os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    sheet_id = spreadsheet_id or os.getenv("GOOGLE_SHEET_ID")

    if not sa_path or not sheet_id:
        logger.warning("GOOGLE_SERVICE_ACCOUNT_JSON or GOOGLE_SHEET_ID missing; skipping remote sync.")
        return {"status": "skipped", "message": "Missing credentials"}

    if not os.path.exists(sa_path):
        logger.error("Service account credentials file not found on disk.")
        return {"status": "error", "message": "Service account JSON file not found"}

    # Generate the feeds locally first
    feed_res = generate_sheets_feeds()
    if feed_res.get("status") != "success":
        return feed_res

    creds = service_account.Credentials.from_service_account_file(
        sa_path, scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )
    service = build("sheets", "v4", credentials=creds)

    # Inspect existing tabs
    metadata = service.spreadsheets().get(spreadsheetId=sheet_id).execute()
    existing_tabs = {s["properties"]["title"]: s["properties"]["sheetId"] for s in metadata.get("sheets", [])}

    tabs_to_sync = {
        "daily_triage_kpi": os.path.join(SHEETS_EXPORT_DIR, "daily_triage_kpi.csv"),
        "integration_health": os.path.join(SHEETS_EXPORT_DIR, "integration_health.csv"),
        "model_performance": os.path.join(SHEETS_EXPORT_DIR, "model_performance_feed.csv"),
        "drift_monitoring": os.path.join(SHEETS_EXPORT_DIR, "drift_monitoring_feed.csv"),
    }

    # Add any missing tabs
    add_requests = []
    for tab_name in tabs_to_sync:
        if tab_name not in existing_tabs:
            add_requests.append({"addSheet": {"properties": {"title": tab_name}}})

    if add_requests:
        service.spreadsheets().batchUpdate(spreadsheetId=sheet_id, body={"requests": add_requests}).execute()
        logger.info(f"Added {len(add_requests)} new tabs to Google Sheet.")

    # Write each feed to its tab
    row_counts = {}
    for tab_name, csv_path in tabs_to_sync.items():
        df = pd.read_csv(csv_path)
        df = df.fillna("")
        header = [list(df.columns)]
        data_rows = df.values.tolist()
        values = header + data_rows

        # Clear existing tab contents
        service.spreadsheets().values().clear(
            spreadsheetId=sheet_id, range=f"'{tab_name}'!A:Z", body={}
        ).execute()

        # Write fresh data
        service.spreadsheets().values().update(
            spreadsheetId=sheet_id,
            range=f"'{tab_name}'!A1",
            valueInputOption="USER_ENTERED",
            body={"values": values},
        ).execute()

        # Read back for verification
        res = service.spreadsheets().values().get(
            spreadsheetId=sheet_id, range=f"'{tab_name}'!A:Z"
        ).execute()
        read_rows = res.get("values", [])
        row_counts[tab_name] = {
            "total_rows": len(read_rows),
            "data_rows": len(read_rows) - 1 if read_rows else 0,
            "header": read_rows[0] if read_rows else [],
            "first_3_rows": read_rows[1:4] if len(read_rows) > 1 else [],
        }
        logger.info(f"Synced '{tab_name}': {len(read_rows)} total rows verified.")

    # Remove default empty Sheet1 if all target tabs are active
    if "Sheet1" in existing_tabs:
        try:
            service.spreadsheets().batchUpdate(
                spreadsheetId=sheet_id,
                body={"requests": [{"deleteSheet": {"sheetId": existing_tabs["Sheet1"]}}]},
            ).execute()
            logger.info("Removed empty default Sheet1 tab.")
        except Exception as e:
            logger.debug(f"Could not delete Sheet1: {e}")

    return {
        "status": "success",
        "verified_tabs": row_counts,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from dotenv import load_dotenv
    load_dotenv()
    res = sync_to_google_sheets()
    print(json.dumps(res, indent=2))

