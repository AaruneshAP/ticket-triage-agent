"""Publish monitoring, performance, and retrain decisions to Databricks workspace.gold tables and Power BI export directory."""

import json
import logging
import os
from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DRIFT_PATH = "data/monitoring/drift_metrics.json"
PERF_PATH = "data/monitoring/performance_history.json"
RETRAIN_PATH = "data/monitoring/promotion_decisions.json"
EXPORT_DIR = "data/exports/powerbi"


def prepare_gold_tables():
    # 1. fct_model_monitoring
    mon_records = []
    if os.path.exists(DRIFT_PATH):
        with open(DRIFT_PATH, "r", encoding="utf-8") as f:
            d = json.load(f)
        eval_ts = d.get("timestamp", datetime.now(timezone.utc).isoformat())
        for feat, info in d.get("features_analyzed", {}).items():
            mon_records.append({
                "feature_name": feat,
                "psi_score": float(info.get("psi", 0.0)),
                "alert_status": str(info.get("status", "NORMAL")),
                "is_alert": int(info.get("alert", False)),
                "evaluated_at": eval_ts,
            })
    df_monitoring = pd.DataFrame(mon_records)

    # 2. fct_model_performance
    perf_records = []
    if os.path.exists(PERF_PATH):
        with open(PERF_PATH, "r", encoding="utf-8") as f:
            d = json.load(f)
        eval_ts = d.get("timestamp", datetime.now(timezone.utc).isoformat())
        for c in d.get("cohort_metrics", []):
            perf_records.append({
                "window_start": str(c.get("window_start")),
                "window_end": str(c.get("window_end")),
                "days": int(c.get("days", 14)),
                "split": str(c.get("split", "test")),
                "sample_type": str(c.get("sample_type", "TEST")),
                "n_tickets": int(c.get("n_tickets", 0)),
                "survival_c_index": float(c.get("survival_c_index")) if c.get("survival_c_index") is not None else None,
                "survival_mae_days": float(c.get("survival_mae_days")) if c.get("survival_mae_days") is not None else None,
                "not_planned_pr_auc": float(c.get("not_planned_pr_auc")) if c.get("not_planned_pr_auc") is not None else None,
                "not_planned_brier": float(c.get("not_planned_brier")) if c.get("not_planned_brier") is not None else None,
                "label_accuracy": float(c.get("label_accuracy")) if c.get("label_accuracy") is not None else None,
                "label_macro_f1": float(c.get("label_macro_f1")) if c.get("label_macro_f1") is not None else None,
                "evaluated_at": eval_ts,
            })
    df_performance = pd.DataFrame(perf_records)

    # 3. fct_retrain_decisions
    retrain_records = []
    if os.path.exists(RETRAIN_PATH):
        with open(RETRAIN_PATH, "r", encoding="utf-8") as f:
            d = json.load(f)
        eval_ts = d.get("timestamp", datetime.now(timezone.utc).isoformat())
        for path_name, info in d.get("demonstration_paths", {}).items():
            size = info.get("grown_retraining_dataset_size") or info.get("training_dataset_size") or 0
            retrain_records.append({
                "path_name": str(path_name),
                "description": str(info.get("description", "")),
                "training_dataset_size": int(size),
                "metric_evaluated": str(info.get("metric_evaluated", "top_1_accuracy")),
                "champion_metric": float(info.get("champion_metric", 0.0)),
                "candidate_metric": float(info.get("candidate_metric", 0.0)),
                "delta": float(info.get("delta", 0.0)),
                "margin_required": float(info.get("margin_required", 0.01)),
                "is_promoted": int(info.get("promoted", False)),
                "final_decision": str(info.get("final_decision", "REJECT")),
                "evaluated_at": eval_ts,
            })
    df_retrain = pd.DataFrame(retrain_records)

    return df_monitoring, df_performance, df_retrain


def sync_to_databricks(df_monitoring, df_performance, df_retrain):
    from databricks import sql

    server_hostname = os.getenv("DATABRICKS_SERVER_HOSTNAME") or os.getenv("DATABRICKS_HOST", "").replace("https://", "").rstrip("/")
    http_path = os.getenv("DATABRICKS_HTTP_PATH")
    access_token = os.getenv("DATABRICKS_TOKEN")

    if not (server_hostname and http_path and access_token):
        logger.warning("Databricks credentials missing in .env; skipping remote sync.")
        return False

    logger.info("Connecting to Databricks SQL Warehouse...")
    conn = sql.connect(server_hostname=server_hostname, http_path=http_path, access_token=access_token)
    cursor = conn.cursor()

    try:
        # Ensure schema workspace.gold exists
        cursor.execute("CREATE SCHEMA IF NOT EXISTS workspace.gold")

        # 1. fct_model_monitoring
        logger.info(f"Creating workspace.gold.fct_model_monitoring ({len(df_monitoring)} rows)...")
        cursor.execute("""
            CREATE OR REPLACE TABLE workspace.gold.fct_model_monitoring (
                feature_name STRING,
                psi_score DOUBLE,
                alert_status STRING,
                is_alert INT,
                evaluated_at STRING
            )
        """)
        for _, r in df_monitoring.iterrows():
            cursor.execute(
                "INSERT INTO workspace.gold.fct_model_monitoring VALUES (?, ?, ?, ?, ?)",
                (r["feature_name"], r["psi_score"], r["alert_status"], r["is_alert"], r["evaluated_at"]),
            )

        # 2. fct_model_performance
        logger.info(f"Creating workspace.gold.fct_model_performance ({len(df_performance)} rows)...")
        cursor.execute("""
            CREATE OR REPLACE TABLE workspace.gold.fct_model_performance (
                window_start STRING,
                window_end STRING,
                days INT,
                split STRING,
                sample_type STRING,
                n_tickets INT,
                survival_c_index DOUBLE,
                survival_mae_days DOUBLE,
                not_planned_pr_auc DOUBLE,
                not_planned_brier DOUBLE,
                label_accuracy DOUBLE,
                label_macro_f1 DOUBLE,
                evaluated_at STRING
            )
        """)
        for _, r in df_performance.iterrows():
            cursor.execute(
                "INSERT INTO workspace.gold.fct_model_performance VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    r["window_start"], r["window_end"], r["days"], r["split"], r["sample_type"], r["n_tickets"],
                    r["survival_c_index"], r["survival_mae_days"], r["not_planned_pr_auc"],
                    r["not_planned_brier"], r["label_accuracy"], r["label_macro_f1"], r["evaluated_at"]
                ),
            )

        # 3. fct_retrain_decisions
        logger.info(f"Creating workspace.gold.fct_retrain_decisions ({len(df_retrain)} rows)...")
        cursor.execute("""
            CREATE OR REPLACE TABLE workspace.gold.fct_retrain_decisions (
                path_name STRING,
                description STRING,
                training_dataset_size INT,
                metric_evaluated STRING,
                champion_metric DOUBLE,
                candidate_metric DOUBLE,
                delta DOUBLE,
                margin_required DOUBLE,
                is_promoted INT,
                final_decision STRING,
                evaluated_at STRING
            )
        """)
        for _, r in df_retrain.iterrows():
            cursor.execute(
                "INSERT INTO workspace.gold.fct_retrain_decisions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    r["path_name"], r["description"], r["training_dataset_size"], r["metric_evaluated"],
                    r["champion_metric"], r["candidate_metric"], r["delta"], r["margin_required"],
                    r["is_promoted"], r["final_decision"], r["evaluated_at"]
                ),
            )

        logger.info("Successfully published all 3 monitoring tables to workspace.gold in Databricks!")
        return True
    finally:
        cursor.close()
        conn.close()


def main():
    os.makedirs(EXPORT_DIR, exist_ok=True)
    df_mon, df_perf, df_retrain = prepare_gold_tables()

    # Save CSV fallback files in data/exports/powerbi/
    df_mon.to_csv(os.path.join(EXPORT_DIR, "fct_model_monitoring.csv"), index=False)
    df_perf.to_csv(os.path.join(EXPORT_DIR, "fct_model_performance.csv"), index=False)
    df_retrain.to_csv(os.path.join(EXPORT_DIR, "fct_retrain_decisions.csv"), index=False)
    logger.info(f"Saved CSV fallback tables to {EXPORT_DIR}")

    # Sync live to Databricks SQL Warehouse
    sync_to_databricks(df_mon, df_perf, df_retrain)


if __name__ == "__main__":
    main()
