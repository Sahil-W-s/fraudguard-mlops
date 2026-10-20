from datetime import timedelta
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
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
    description="Detect feature drift and trigger gated retraining when drift is detected",
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

    retrain = TriggerDagRunOperator(
    task_id="conditional_retrain",
    trigger_dag_id="fraudguard_training_pipeline",
    conf={
        "trigger_reason": "drift_detected",
        "drift_report": (
            "{{ ti.xcom_pull(task_ids='drift_check')['report_path'] }}"
        ),
    },
    wait_for_completion=False,
)

    no_retraining = EmptyOperator(
        task_id="no_retraining_needed",
    )

    drift >> branch >> [retrain, no_retraining]
