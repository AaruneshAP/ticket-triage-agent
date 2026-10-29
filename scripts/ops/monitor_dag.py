import subprocess
import time

script = '''
export AIRFLOW__DATABASE__SQL_ALCHEMY_CONN='postgresql+psycopg2://airflow:airflow@postgres:5432/airflow'
export AIRFLOW_HOME='/opt/airflow'
nsenter -t 1134 -m -u -i -n -p /home/airflow/.local/bin/python -c "
from airflow.models import DagRun, TaskInstance
from airflow.utils.session import create_session

with create_session() as session:
    dr = session.query(DagRun).filter(DagRun.dag_id == 'triage_hourly', DagRun.run_id == 'manual__2026-10-28T12:16:55+00:00').first()
    if dr:
        print('DAG_RUN_STATE:', dr.state)
        tis = session.query(TaskInstance).filter(TaskInstance.dag_id == 'triage_hourly', TaskInstance.run_id == 'manual__2026-10-28T12:16:55+00:00').all()
        for ti in tis:
            dur = round((ti.end_date - ti.start_date).total_seconds(), 2) if ti.end_date and ti.start_date else None
            print(f'TASK: {ti.task_id} | STATE: {ti.state} | DURATION: {dur}s | START: {ti.start_date} | END: {ti.end_date}')
"
'''

def main():
    for attempt in range(25):
        res = subprocess.run(['wsl', '-d', 'docker-desktop', 'sh', '-c', script], capture_output=True, text=True, timeout=25)
        output = res.stdout.strip()
        print(f"\n--- Check {attempt+1} ---")
        for line in output.splitlines():
            if 'DAG_RUN_STATE' in line or 'TASK:' in line:
                print(" ", line)
        if 'DAG_RUN_STATE: success' in output:
            print("\n🎉 TRIAGE_HOURLY RUN SUCCEEDED END-TO-END!")
            break
        if 'DAG_RUN_STATE: failed' in output:
            print("\n❌ TRIAGE_HOURLY RUN FAILED!")
            break
        time.sleep(5)

if __name__ == '__main__':
    main()
