import json
import subprocess
import time


def get_airflow_pid() -> str:
    cmd = "for p in /proc/[0-9]*/cmdline; do if tr '\\0' ' ' < $p 2>/dev/null | grep -q 'airflow scheduler'; then echo $p | cut -d/ -f3; break; fi; done"
    res = subprocess.run(["wsl", "-d", "docker-desktop", "sh", "-c", cmd], capture_output=True, text=True)
    pid = res.stdout.strip()
    if pid and pid.isdigit():
        return pid
    return "1220"


def trigger_dag(pid: str):
    trigger_script = f"""
export AIRFLOW__DATABASE__SQL_ALCHEMY_CONN='postgresql+psycopg2://airflow:airflow@postgres:5432/airflow'
export AIRFLOW_HOME='/opt/airflow'
nsenter -t {pid} -m -u -i -n -p /home/airflow/.local/bin/airflow dags trigger triage_hourly
"""
    res = subprocess.run(["wsl", "-d", "docker-desktop", "sh", "-c", trigger_script], capture_output=True, text=True, timeout=25)
    print("Trigger output:\n", res.stdout)
    if res.stderr:
        print("Trigger stderr:\n", res.stderr)


def poll_dag(pid: str) -> dict:
    poll_script = f"""
export AIRFLOW__DATABASE__SQL_ALCHEMY_CONN='postgresql+psycopg2://airflow:airflow@postgres:5432/airflow'
export AIRFLOW_HOME='/opt/airflow'
nsenter -t {pid} -m -u -i -n -p /home/airflow/.local/bin/python - << 'EOF'
import json
from airflow.models import DagRun, TaskInstance
from airflow.utils.session import create_session

with create_session() as session:
    dr = session.query(DagRun).filter(DagRun.dag_id == 'triage_hourly').order_by(DagRun.execution_date.desc()).first()
    if not dr:
        print(json.dumps({{"status": "no_run"}}))
    else:
        tis = session.query(TaskInstance).filter(TaskInstance.dag_id == 'triage_hourly', TaskInstance.run_id == dr.run_id).all()
        tasks = []
        for ti in tis:
            dur = (ti.end_date - ti.start_date).total_seconds() if (ti.end_date and ti.start_date) else None
            tasks.append({{
                "task_id": ti.task_id,
                "state": ti.state,
                "start_date": ti.start_date.isoformat() if ti.start_date else None,
                "end_date": ti.end_date.isoformat() if ti.end_date else None,
                "duration_sec": round(dur, 2) if dur is not None else None,
                "try_number": ti.try_number
            }})
        print(json.dumps({{
            "status": "found",
            "run_id": dr.run_id,
            "state": dr.state,
            "start_date": dr.start_date.isoformat() if dr.start_date else None,
            "end_date": dr.end_date.isoformat() if dr.end_date else None,
            "tasks": tasks
        }}))
EOF
"""
    res = subprocess.run(["wsl", "-d", "docker-desktop", "sh", "-c", poll_script], capture_output=True, text=True, timeout=25)
    out = res.stdout.strip()
    lines = [raw_line for raw_line in out.splitlines() if raw_line.startswith("{") and "run_id" in raw_line]
    if lines:
        try:
            return json.loads(lines[-1])
        except Exception:
            pass
    return {}


def main():
    pid = get_airflow_pid()
    print(f"=== Identified Airflow Scheduler Container PID: {pid} ===")

    print("=== Step 1: Triggering fresh triage_hourly run ===")
    trigger_dag(pid)

    print("=== Step 2: Monitoring DAG run execution (no manual task clears/retries) ===")
    start_t = time.time()
    completed = False

    while time.time() - start_t < 360:
        time.sleep(5)
        data = poll_dag(pid)
        if not data:
            continue
        run_id = data.get("run_id")
        state = data.get("state")
        tasks = data.get("tasks", [])
        task_summary = ", ".join([f"{t['task_id']}:{t['state']}" for t in tasks])
        elapsed = int(time.time() - start_t)
        print(f"[{elapsed}s] DAG run {run_id} | State: {state} | Tasks: {task_summary}")

        if state in ["success", "failed"]:
            completed = True
            print(f"\nDAG run finished with state: {state} in {elapsed} seconds.")
            print("\n=== Per-Task Timing Summary ===")
            for t in sorted(tasks, key=lambda x: x.get("start_date") or ""):
                print(f"  Task: {t['task_id']:<25} | State: {t['state']:<10} | Start: {t['start_date']} | End: {t['end_date']} | Duration: {t['duration_sec']}s")
            break

    if not completed:
        print("Execution monitoring timed out.")


if __name__ == "__main__":
    main()
