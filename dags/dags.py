from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# import functions từ code của bạn
from src.data_process.preprocess import preprocess
from src.data_process.feature_engineer import run_fe


default_args = {
    "start_date": datetime(2026, 1, 1),
}

with DAG(
    dag_id="ml_data_pipeline",
    schedule=None,   # chạy manual
    catchup=False,
    default_args=default_args,
    tags=["ml", "data_pipeline"]
) as dag:

    preprocess_task = PythonOperator(
        task_id="preprocess",
        python_callable=preprocess
    )

    feature_engineer_task = PythonOperator(
        task_id="feature_engineering",
        python_callable=run_fe
    )
        # DAG flow
    preprocess_task >> feature_engineer_task