from datetime import timedelta

import pendulum
from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator

from scripts import airflow_pipeline as pipeline


def choose_drift_path(**context):
    result = context["ti"].xcom_pull(task_ids="drift_check")

    if isinstance(result, dict) and result.get("drift_detected") is True:
        return "conditional_retrain"

    return "no_retraining_needed"


with DAG(
    dag_id="fraudguard_drift_monitor",
    description="Scheduled drift-check placeholder for the Phase 8 implementation",
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    schedule="0 3 * * 1",
    catchup=False,
    max_active_runs=1,
    default_args={
        "owner": "fraudguard",
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
    tags=["fraudguard", "drift"],
) as dag:

    drift = PythonOperator(
        task_id="drift_check",
        python_callable=pipeline.drift_check,
    )

    branch = BranchPythonOperator(
        task_id="drift_decision",
        python_callable=choose_drift_path,
    )

    retrain = PythonOperator(
        task_id="conditional_retrain",
        python_callable=pipeline.conditional_retrain,
    )

    no_retraining = EmptyOperator(
        task_id="no_retraining_needed",
    )

    drift >> branch >> [retrain, no_retraining]
