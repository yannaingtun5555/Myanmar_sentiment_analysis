# ~/Desktop/myanmar_sentiment/airflow/dags/test_dag.py
"""
SIMPLE TEST DAG - Verify Airflow is working
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator

def print_hello():
    """Simple hello function"""
    print("=" * 50)
    print("Hello from Airflow!")
    print(f"Current time: {datetime.now()}")
    print("=" * 50)
    return "Hello World!"

def print_world():
    """Another simple function"""
    print("World! Airflow is running correctly")
    return "Done"

# DAG definition
default_args = {
    'owner': 'ynt',
    'depends_on_past': False,
    'start_date': datetime(2026, 5, 22),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
}

dag = DAG(
    'test_airflow_dag',
    default_args=default_args,
    description='Simple test DAG',
    schedule_interval='*/1 * * * *',  # Run every minute
    catchup=False,
    tags=['test'],
)

# Create tasks
start = DummyOperator(task_id='start', dag=dag)

task1 = PythonOperator(
    task_id='print_hello',
    python_callable=print_hello,
    dag=dag,
)

task2 = PythonOperator(
    task_id='print_world',
    python_callable=print_world,
    dag=dag,
)

end = DummyOperator(task_id='end', dag=dag)

# Set task dependencies
start >> task1 >> task2 >> end

print("✅ Test DAG loaded successfully!")