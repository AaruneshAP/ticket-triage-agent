import subprocess

script = """
export AIRFLOW__DATABASE__SQL_ALCHEMY_CONN='postgresql+psycopg2://airflow:airflow@postgres:5432/airflow'
export AIRFLOW_HOME='/opt/airflow'
nsenter -t 1134 -m -u -i -n -p /home/airflow/.local/bin/python - << 'EOF'
from airflow.models import DagRun, TaskInstance, Log, TaskFail
from airflow.utils.session import create_session

with create_session() as session:
    ti_count = session.query(TaskInstance).filter(TaskInstance.dag_id == 'triage_hourly').delete(synchronize_session=False)
    tf_count = session.query(TaskFail).filter(TaskFail.dag_id == 'triage_hourly').delete(synchronize_session=False)
    dr_count = session.query(DagRun).filter(DagRun.dag_id == 'triage_hourly').delete(synchronize_session=False)
    session.commit()
    print(f"Cleared triage_hourly history: {dr_count} DAG runs, {ti_count} TaskInstances, {tf_count} TaskFails deleted.")
EOF
"""
res = subprocess.run(["wsl", "-d", "docker-desktop", "sh", "-c", script], capture_output=True, text=True, timeout=20)
print(res.stdout)
if res.stderr:
    print("STDERR:", res.stderr)
