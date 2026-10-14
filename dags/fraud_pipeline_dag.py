from __future__ import annotations

import logging
from datetime import timedelta

import pendulum
from airflow import DAG
from airflow.exceptions import AirflowException
from airflow.operators.python import PythonOperator, BranchPythonOperator

from scripts import airflow_pipeline as pipeline


logger = logging.getLogger(__name__)


def run_evaluation(**context):
    ti = context["ti"]
    candidate = ti.xcom_pull(task_ids="train_candidate")
    if not candidate:
        raise AirflowException("Training did not return candidate metadata.")
    return pipeline.evaluate_candidate(candidate)


def run_quality_gate(**context):
    ti = context["ti"]
    candidate = ti.xcom_pull(task_ids="train_candidate")
    if not candidate:
        raise AirflowException("Candidate metadata is missing.")
    return pipeline.quality_gate(candidate)


def choose_promotion_path(**context):
    ti = context["ti"]
    gate = ti.xcom_pull(task_ids="quality_gate")

    if not isinstance(gate, dict) or "passed" not in gate:
        raise AirflowException(
            "Quality gate result is missing; refusing promotion."
        )

    if gate["passed"] is True:
        return "register_candidate"

    return "reject_candidate"


def run_registration(**context):
    ti = context["ti"]
    candidate = ti.xcom_pull(task_ids="train_candidate")
    gate = ti.xcom_pull(task_ids="quality_gate")

    if not candidate:
        raise AirflowException("Candidate metadata is missing.")

    return pipeline.register_candidate(candidate, gate)


def run_rejection(**context):
    ti = context["ti"]
    candidate = ti.xcom_pull(task_ids="train_candidate")
    gate = ti.xcom_pull(task_ids="quality_gate")

    if not candidate:
        raise AirflowException("Candidate metadata is missing.")

    return pipeline.reject_candidate(candidate, gate)


default_args = {
    "owner": "fraudguard",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="fraudguard_training_pipeline",
    description="FraudGuard ingestion, training, evaluation and gated promotion",
    default_args=default_args,
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    schedule="0 2 * * 1",
    catchup=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(hours=12),
    tags=["fraudguard", "mlops", "training"],
) as dag:

    ingest = PythonOperator(
        task_id="ingest_data",
        python_callable=pipeline.ingest_data,
    )

    validate = PythonOperator(
        task_id="validate_data",
        python_callable=pipeline.validate_data,
    )

    feature_engineer = PythonOperator(
        task_id="feature_engineering",
        python_callable=pipeline.engineer_features,
    )

    train = PythonOperator(
        task_id="train_candidate",
        python_callable=pipeline.train_candidate,
        execution_timeout=timedelta(hours=8),
    )

    evaluate = PythonOperator(
        task_id="evaluate_candidate",
        python_callable=run_evaluation,
        execution_timeout=timedelta(hours=2),
    )

    gate = PythonOperator(
        task_id="quality_gate",
        python_callable=run_quality_gate,
        execution_timeout=timedelta(hours=2),
    )

    branch = BranchPythonOperator(
        task_id="promotion_decision",
        python_callable=choose_promotion_path,
    )

    register = PythonOperator(
        task_id="register_candidate",
        python_callable=run_registration,
    )

    reject = PythonOperator(
        task_id="reject_candidate",
        python_callable=run_rejection,
    )

    ingest >> validate >> feature_engineer >> train >> evaluate >> gate >> branch
    branch >> [register, reject]
