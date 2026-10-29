"""Airflow Host Bridge Daemon.

Enables Airflow Docker container tasks to execute host-level workloads that require:
1. Databricks SQL Warehouse connection without container network restrictions.
2. Full dbt-databricks CLI execution.
3. Live warehouse loading.
4. 4-Model triage scoring and drift monitoring using host virtual environment.

Communication is IPC via shared volume mount (data/.airflow_ipc).
"""

import json
import logging
import os
import subprocess
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (HostBridge) %(message)s",
)
logger = logging.getLogger("airflow_host_bridge")

IPC_DIR = os.path.join("data", ".airflow_ipc")
DBT_REQUEST = os.path.join(IPC_DIR, "dbt_build_request.json")
DBT_RESPONSE = os.path.join(IPC_DIR, "dbt_build_response.json")
LOAD_REQUEST = os.path.join(IPC_DIR, "load_raw_request.json")
LOAD_RESPONSE = os.path.join(IPC_DIR, "load_raw_response.json")
SCORE_REQUEST = os.path.join(IPC_DIR, "score_request.json")
SCORE_RESPONSE = os.path.join(IPC_DIR, "score_response.json")
DRIFT_REQUEST = os.path.join(IPC_DIR, "drift_request.json")
DRIFT_RESPONSE = os.path.join(IPC_DIR, "drift_response.json")


def handle_dbt_build(req: dict):
    logger.info("Received real dbt_build request. Executing 'dbt build' on host...")
    dbt_exe = os.path.join(".venv", "Scripts", "dbt.exe") if os.name == "nt" else "dbt"
    select_arg = req.get("select", "+fct_tickets mart_weekly_volume mart_model_monitoring")

    cmd = (
        [dbt_exe, "build", "--select"]
        + select_arg.split()
        + ["--project-dir", "dbt", "--profiles-dir", "dbt"]
    )

    start_t = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        dur = time.time() - start_t
        stdout = proc.stdout
        stderr = proc.stderr
        exit_code = proc.returncode

        pass_count = 0
        warn_count = 0
        error_count = 0
        for line in stdout.splitlines():
            if "Done. PASS=" in line:
                for part in line.split():
                    if part.startswith("PASS="):
                        pass_count = int(part.split("=")[1])
                    elif part.startswith("WARN="):
                        warn_count = int(part.split("=")[1])
                    elif part.startswith("ERROR="):
                        error_count = int(part.split("=")[1])

        resp = {
            "status": "success" if exit_code == 0 else "failed",
            "exit_code": exit_code,
            "duration_sec": round(dur, 2),
            "pass_count": pass_count,
            "warn_count": warn_count,
            "error_count": error_count,
            "models_count": 9,
            "tests_count": 15,
            "stdout": stdout,
            "stderr": stderr,
            "completed_at": time.time(),
        }
        logger.info(f"dbt build finished in {dur:.1f}s. Exit code: {exit_code}, PASS={pass_count}, WARN={warn_count}, ERROR={error_count}")
    except Exception as e:
        logger.error(f"dbt build execution error: {e}")
        resp = {
            "status": "error",
            "exit_code": 1,
            "duration_sec": round(time.time() - start_t, 2),
            "error": str(e),
            "stdout": "",
            "stderr": str(e),
            "pass_count": 0,
            "warn_count": 0,
            "error_count": 1,
            "completed_at": time.time(),
        }

    with open(DBT_RESPONSE, "w", encoding="utf-8") as f:
        json.dump(resp, f, indent=2)

    if os.path.exists(DBT_REQUEST):
        os.remove(DBT_REQUEST)


def handle_load_raw(req: dict):
    logger.info("Received real load_raw request. Executing warehouse load on host...")
    start_t = time.time()
    try:
        from triage.warehouse.loader import run_warehouse_load
        issues_p = os.path.join("data", "interim", "issues.parquet")
        comments_p = os.path.join("data", "interim", "comments.parquet")
        res = run_warehouse_load(issues_path=issues_p, comments_path=comments_p, overwrite=False)
        dur = time.time() - start_t
        resp = {
            "status": "success",
            "duration_sec": round(dur, 2),
            "result": res,
            "completed_at": time.time(),
        }
        logger.info(f"Warehouse load completed in {dur:.1f}s: {res}")
    except Exception as e:
        logger.warning(f"Warehouse load error: {e}")
        resp = {
            "status": "error",
            "duration_sec": round(time.time() - start_t, 2),
            "error": str(e),
            "completed_at": time.time(),
        }

    with open(LOAD_RESPONSE, "w", encoding="utf-8") as f:
        json.dump(resp, f, indent=2)

    if os.path.exists(LOAD_REQUEST):
        os.remove(LOAD_REQUEST)


def handle_scoring(req: dict):
    logger.info("Received score_tickets request. Executing 4-model triage scoring on host...")
    start_t = time.time()
    try:
        from triage.models.scoring import score_new_tickets

        base_data = "data"
        p_path = os.path.join(base_data, "interim", "issues.parquet")
        scoring_ckpt_path = os.path.join(base_data, "interim", "scoring_checkpoint.json")
        scored_out = os.path.join(base_data, "interim", "scored_tickets.jsonl")
        models_dir = "models"

        res = score_new_tickets(
            issues_path=p_path,
            scoring_ckpt_path=scoring_ckpt_path,
            scored_out_path=scored_out,
            models_dir=models_dir,
        )
        resp = {
            "status": res.get("status", "success"),
            "duration_sec": res.get("duration_sec", round(time.time() - start_t, 2)),
            "scored_count": res.get("scored_count", 0),
            "survival_count": res.get("survival_count", 0),
            "not_planned_count": res.get("not_planned_count", 0),
            "label_recommender_count": res.get("label_recommender_count", 0),
            "completeness_count": res.get("completeness_count", 0),
            "incomplete_count": res.get("incomplete_count", 0),
            "sample_ticket_ids": res.get("sample_ticket_ids", []),
            "completed_at": time.time(),
        }
    except Exception as e:
        logger.error(f"Scoring error: {e}")
        resp = {
            "status": "error",
            "duration_sec": round(time.time() - start_t, 2),
            "error": str(e),
            "completed_at": time.time(),
        }

    with open(SCORE_RESPONSE, "w", encoding="utf-8") as f:
        json.dump(resp, f, indent=2)

    if os.path.exists(SCORE_REQUEST):
        os.remove(SCORE_REQUEST)


def handle_drift(req: dict):
    logger.info("Received drift_check request. Executing drift analysis on host...")
    start_t = time.time()
    try:
        from triage.monitoring.drift import run_drift_analysis
        summary = run_drift_analysis()
        dur = time.time() - start_t
        resp = {
            "status": "success",
            "duration_sec": round(dur, 2),
            "overall_drift_alert": summary.get("overall_drift_alert", False),
            "features_analyzed": len(summary.get("features_analyzed", {})),
            "completed_at": time.time(),
        }
        logger.info(f"Drift check completed in {dur:.1f}s: alert={summary.get('overall_drift_alert')}")
    except Exception as e:
        logger.error(f"Drift check error: {e}")
        resp = {
            "status": "error",
            "duration_sec": round(time.time() - start_t, 2),
            "error": str(e),
            "completed_at": time.time(),
        }

    with open(DRIFT_RESPONSE, "w", encoding="utf-8") as f:
        json.dump(resp, f, indent=2)

    if os.path.exists(DRIFT_REQUEST):
        os.remove(DRIFT_REQUEST)


def main():
    os.makedirs(IPC_DIR, exist_ok=True)
    logger.info(f"Airflow Host Bridge Daemon initialized. Monitoring {IPC_DIR}...")

    # Clean old requests
    for p in [DBT_REQUEST, LOAD_REQUEST, SCORE_REQUEST, DRIFT_REQUEST]:
        if os.path.exists(p):
            os.remove(p)

    while True:
        try:
            if os.path.exists(DBT_REQUEST):
                try:
                    with open(DBT_REQUEST, "r", encoding="utf-8") as f:
                        req = json.load(f)
                    handle_dbt_build(req)
                except Exception as e:
                    logger.error(f"Error processing dbt request: {e}")
                    if os.path.exists(DBT_REQUEST):
                        os.remove(DBT_REQUEST)

            if os.path.exists(LOAD_REQUEST):
                try:
                    with open(LOAD_REQUEST, "r", encoding="utf-8") as f:
                        req = json.load(f)
                    handle_load_raw(req)
                except Exception as e:
                    logger.error(f"Error processing load request: {e}")
                    if os.path.exists(LOAD_REQUEST):
                        os.remove(LOAD_REQUEST)

            if os.path.exists(SCORE_REQUEST):
                try:
                    with open(SCORE_REQUEST, "r", encoding="utf-8") as f:
                        req = json.load(f)
                    handle_scoring(req)
                except Exception as e:
                    logger.error(f"Error processing scoring request: {e}")
                    if os.path.exists(SCORE_REQUEST):
                        os.remove(SCORE_REQUEST)

            if os.path.exists(DRIFT_REQUEST):
                try:
                    with open(DRIFT_REQUEST, "r", encoding="utf-8") as f:
                        req = json.load(f)
                    handle_drift(req)
                except Exception as e:
                    logger.error(f"Error processing drift request: {e}")
                    if os.path.exists(DRIFT_REQUEST):
                        os.remove(DRIFT_REQUEST)

            time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Host bridge stopped.")
            break
        except Exception as e:
            logger.error(f"Bridge loop unexpected notice: {e}")
            time.sleep(2)


if __name__ == "__main__":
    main()
