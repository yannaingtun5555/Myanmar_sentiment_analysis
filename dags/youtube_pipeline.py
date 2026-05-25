from __future__ import annotations

import os
import shutil
import sys
from datetime import datetime, timedelta

import psycopg2
import yaml
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "config.yml")
BOT_SCRIPT = os.path.join(PROJECT_ROOT, "src", "bot", "telegram_request_bot.py")
PYENV_DATA_ENG_PYTHON = os.environ.get(
    "AIRFLOW_PYTHON_BIN",
    shutil.which("python") or "python",
)

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from pipeline.fetch_comment import main as fetch_main
from pipeline.preprocess import main as preprocess_main
from pipeline.predict_model import main as predict_main
from pipeline.save_results import main as save_main


default_args = {
    "owner": "data_team",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_from_project_root(fn):
    """Ensure script-style modules using relative paths run correctly in Airflow."""
    old_cwd = os.getcwd()
    os.chdir(PROJECT_ROOT)
    try:
        return fn()
    finally:
        os.chdir(old_cwd)


def has_new_request() -> str:
    cfg = load_config()
    db = cfg["database"]

    conn = psycopg2.connect(
        host=db["host"],
        port=db["port"],
        database=db["database"],
        user=db["user"],
        password=db["password"],
    )

    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM requests WHERE status = 'NEW' LIMIT 1")
            row = cur.fetchone()
            return "fetch_comments" if row else "stop_pipeline"
    finally:
        conn.close()


telegram_dag = DAG(
    dag_id="telegram_bot_service",
    default_args=default_args,
    description="Run Telegram request bot service",
    schedule_interval=None,
    catchup=False,
    max_active_runs=1,
    tags=["telegram", "bot"],
)

if os.path.exists(BOT_SCRIPT):
    start_telegram_bot = BashOperator(
        task_id="start_telegram_bot",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            f"{PYENV_DATA_ENG_PYTHON} {BOT_SCRIPT}"
        ),
        dag=telegram_dag,
    )
else:
    start_telegram_bot = BashOperator(
        task_id="start_telegram_bot",
        bash_command=f"echo 'Bot script not found: {BOT_SCRIPT}' && exit 1",
        dag=telegram_dag,
    )


pipeline_dag = DAG(
    dag_id="youtube_sentiment_pipeline",
    default_args=default_args,
    description="Fetch -> preprocess -> predict -> save when NEW request exists",
    schedule_interval="* * * * *",
    catchup=False,
    max_active_runs=1,
    tags=["youtube", "sentiment", "pipeline"],
)

check_request = BranchPythonOperator(
    task_id="check_request",
    python_callable=has_new_request,
    dag=pipeline_dag,
)

stop_pipeline = EmptyOperator(
    task_id="stop_pipeline",
    dag=pipeline_dag,
)

fetch_comments = PythonOperator(
    task_id="fetch_comments",
    python_callable=run_from_project_root,
    op_args=[fetch_main],
    dag=pipeline_dag,
)

preprocess_comments = PythonOperator(
    task_id="preprocess_comments",
    python_callable=run_from_project_root,
    op_args=[preprocess_main],
    dag=pipeline_dag,
)

predict_comments = PythonOperator(
    task_id="predict_comments",
    python_callable=run_from_project_root,
    op_args=[predict_main],
    dag=pipeline_dag,
)

save_results = PythonOperator(
    task_id="save_results",
    python_callable=run_from_project_root,
    op_args=[save_main],
    dag=pipeline_dag,
)

check_request >> stop_pipeline
check_request >> fetch_comments >> preprocess_comments >> predict_comments >> save_results
