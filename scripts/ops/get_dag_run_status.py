import subprocess

script = '''
export AIRFLOW__DATABASE__SQL_ALCHEMY_CONN='postgresql+psycopg2://airflow:airflow@postgres:5432/airflow'
export AIRFLOW_HOME='/opt/airflow'
nsenter -t 1134 -m -u -i -n -p /home/airflow/.local/bin/python -c "
from airflow.models import DagRun, TaskInstance
from airflow.utils.session import create_session

with create_session() as session:
    dr = session.query(DagRun).filter(DagRun.dag_id == 'triage_hourly', DagRun.run_id == 'manual__2026-10-28T12:16:55+00:00').first()
    if dr:
        print('DAG_RUN_STATE:', dr.state, '| START:', dr.start_date, '| END:', dr.end_date)
        tis = session.query(TaskInstance).filter(TaskInstance.dag_id == 'triage_hourly', TaskInstance.run_id == 'manual__2026-10-28T12:16:55+00:00').order_by(TaskInstance.start_date).all()
        for ti in tis:
            dur = round((ti.end_date - ti.start_date).total_seconds(), 2) if ti.end_date and ti.start_date else None
            dur_str = f'{dur}s' if dur is not None else 'None'
            s_date = str(ti.start_date) if ti.start_date else 'Not Started'
            e_date = str(ti.end_date) if ti.end_date else 'Running/Queued'
            state_str = str(ti.state or 'none')
            print(f'  {ti.task_id:<22} | {state_str:<10} | {dur_str:<9} | {s_date} -> {e_date}')
"
'''

def main():
    res = subprocess.run(['wsl', '-d', 'docker-desktop', 'sh', '-c', script], capture_output=True, text=True, timeout=25)
    print(res.stdout)
    if res.stderr:
        print('STDERR:', res.stderr)

if __name__ == '__main__':
    main()
