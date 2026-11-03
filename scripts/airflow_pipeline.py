from __future__ import annotations
import math
import logging
import os

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.metrics import average_precision_score

from src.validation.validate_data import main as validate_main

from scripts import run_experiments as experiments


logger = logging.getLogger(__name__)

MODEL_NAME = experiments.REGISTERED_MODEL_NAME
EXPERIMENT_NAME = experiments.EXPERIMENT_NAME


def configure_mlflow() -> None:
    mlflow.set_tracking_uri(experiments.TRACKING_URI)


def ingest_data() -> dict:
    """Verify that the raw datasets are available for validation."""

    from src.ingestion.load_data import (
        TRANSACTION_PATH,
        IDENTITY_PATH,
    )

    for dataset_path in (TRANSACTION_PATH, IDENTITY_PATH):
        if not dataset_path.is_file():
            raise FileNotFoundError(
                f"Required dataset not found: {dataset_path}"
            )

        if dataset_path.stat().st_size == 0:
            raise ValueError(
                f"Required dataset is empty: {dataset_path}"
            )

    result = {
        "status": "success",
        "stage": "ingestion",
        "transaction_file_bytes": TRANSACTION_PATH.stat().st_size,
        "identity_file_bytes": IDENTITY_PATH.stat().st_size,
    }

    logger.info("Raw dataset availability check passed: %s", result)
    return result


def validate_data() -> dict:
    validate_main()
    return {"status": "success", "stage": "validation"}


def engineer_features() -> dict:
    data = experiments.load_splits()

    result = {
        "status": "success",
        "training_rows": len(data["X_train"]),
        "validation_rows": len(data["X_valid"]),
        "test_rows": len(data["X_test"]),
        "feature_count": len(data["feature_pipeline"].feature_names),
        "split_strategy": "chronological_70_15_15",
    }

    logger.info("Feature engineering summary: %s", result)
    return result


def train_candidate() -> dict:
    configure_mlflow()
    mlflow.set_experiment(EXPERIMENT_NAME)

    data = experiments.load_splits()
    lineage = experiments.get_lineage()

    model_names = [
        "logistic_regression",
        "lightgbm",
        "random_forest",
    ]

    results = [
        experiments.train_experiment(name, data, lineage)
        for name in model_names
    ]

    winner = max(
        results,
        key=lambda item: item["validation_average_precision"],
    )

    logger.info(
        "Selected candidate %s with validation AP %.6f",
        winner["model_name"],
        winner["validation_average_precision"],
    )

    return {
        "model_name": winner["model_name"],
        "run_id": winner["run_id"],
        "validation_average_precision": float(
            winner["validation_average_precision"]
        ),
        "validation_recall_at_precision_target": float(
            winner["validation_recall_at_precision_target"]
        ),
        "threshold": float(winner["threshold"]),
    }


def evaluate_candidate(candidate: dict) -> dict:
    configure_mlflow()

    data = experiments.load_splits()
    lineage = experiments.get_lineage()

    summary = experiments.evaluate_winner(
        winner=candidate,
        data=data,
        lineage=lineage,
        register_model=False,
    )

    test_ap = float(
        summary["test_metrics"]["average_precision"]
    )

    logger.info(
        "Candidate %s held-out AP: %.6f",
        candidate["run_id"],
        test_ap,
    )

    return {
        "run_id": candidate["run_id"],
        "model_name": candidate["model_name"],
        "test_average_precision": test_ap,
        "validation_average_precision": float(
            candidate["validation_average_precision"]
        ),
    }


def quality_gate(candidate: dict) -> dict:
    """Compare candidate and champion on validation data only."""
    configure_mlflow()

    minimum_improvement = float(
        os.getenv("FRAUDGUARD_MIN_AP_IMPROVEMENT", "0.001")
    )

    if minimum_improvement < 0:
        raise ValueError(
            "FRAUDGUARD_MIN_AP_IMPROVEMENT cannot be negative"
        )

    data = experiments.load_splits()
    y_valid = data["y_valid"]
    X_valid = data["X_valid"]

    client = MlflowClient(
        tracking_uri=experiments.TRACKING_URI
    )

    # Fail closed if the champion is absent or cannot be loaded.
    champion_uri = f"models:/{MODEL_NAME}@champion"
    champion = mlflow.sklearn.load_model(champion_uri)

    candidate_uri = f"runs:/{candidate['run_id']}/model"
    challenger = mlflow.sklearn.load_model(candidate_uri)

    # Compare both models on exactly the same validation rows.
    champion_probabilities = champion.predict_proba(X_valid)[:, 1]
    candidate_probabilities = challenger.predict_proba(X_valid)[:, 1]

    champion_ap = float(
        average_precision_score(y_valid, champion_probabilities)
    )
    candidate_ap = float(
        average_precision_score(y_valid, candidate_probabilities)
    )

    required_ap = champion_ap + minimum_improvement
    passed = candidate_ap >= required_ap

    result = {
        "passed": passed,
        "candidate_run_id": candidate["run_id"],
        "evaluation_split": "validation",
        "candidate_average_precision": candidate_ap,
        "champion_average_precision": champion_ap,
        "required_average_precision": required_ap,
        "minimum_improvement": minimum_improvement,
    }

    client.log_metric(
        candidate["run_id"],
        "quality_gate_candidate_average_precision",
        candidate_ap,
    )
    client.log_metric(
        candidate["run_id"],
        "quality_gate_champion_average_precision",
        champion_ap,
    )
    client.set_tag(
        candidate["run_id"],
        "quality_gate_passed",
        str(passed).lower(),
    )

    logger.info("Quality gate result: %s", result)
    return result



def register_candidate(candidate: dict, gate: dict) -> dict:
    """Register and promote a candidate only after a valid quality gate."""

    # Validate candidate metadata.
    if not isinstance(candidate, dict) or not candidate.get("run_id"):
        raise ValueError(
            "Candidate registration blocked: invalid candidate metadata"
        )

    # Validate the quality-gate decision.
    if not isinstance(gate, dict) or gate.get("passed") is not True:
        raise ValueError(
            "Candidate registration blocked: quality gate did not pass"
        )

    # Ensure the gate approved this exact candidate.
    if gate.get("candidate_run_id") != candidate["run_id"]:
        raise ValueError(
            "Candidate registration blocked: quality-gate run ID mismatch"
        )

    if gate.get("evaluation_split") != "validation":
        raise ValueError(
            "Candidate registration blocked: unexpected evaluation split"
        )

    # Validate all metrics before touching the registry.
    try:
        candidate_ap = float(gate["candidate_average_precision"])
        champion_ap = float(gate["champion_average_precision"])
        required_ap = float(gate["required_average_precision"])
        minimum_improvement = float(gate["minimum_improvement"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Candidate registration blocked: invalid gate metrics"
        ) from exc

    metrics = (
        candidate_ap,
        champion_ap,
        required_ap,
        minimum_improvement,
    )

    if not all(math.isfinite(value) for value in metrics):
        raise ValueError(
            "Candidate registration blocked: non-finite gate metrics"
        )

    if (
        not 0.0 <= candidate_ap <= 1.0
        or not 0.0 <= champion_ap <= 1.0
        or not 0.0 <= required_ap <= 1.0
        or minimum_improvement < 0.0
    ):
        raise ValueError(
            "Candidate registration blocked: metrics outside valid ranges"
        )

    if not math.isclose(
        required_ap,
        champion_ap + minimum_improvement,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "Candidate registration blocked: inconsistent gate threshold"
        )

    if candidate_ap < required_ap:
        raise ValueError(
            "Candidate registration blocked: candidate is below threshold"
        )

    # All gate checks passed. Registry mutations start here.
    configure_mlflow()

    client = MlflowClient(
        tracking_uri=experiments.TRACKING_URI
    )

    registered = mlflow.register_model(
        model_uri=f"runs:/{candidate['run_id']}/model",
        name=MODEL_NAME,
        await_registration_for=120,
    )

    client.set_registered_model_alias(
        MODEL_NAME,
        "champion",
        registered.version,
    )

    client.update_model_version(
        name=MODEL_NAME,
        version=registered.version,
        description=(
            "Promoted by the FraudGuard Airflow quality gate. "
            f"Candidate AP={candidate_ap:.6f}; "
            f"previous champion AP={champion_ap:.6f}."
        ),
    )

    logger.info(
        "Registered model version %s and updated champion alias",
        registered.version,
    )

    return {
        "status": "registered",
        "model_name": MODEL_NAME,
        "version": str(registered.version),
        "alias": "champion",
    }



def reject_candidate(candidate: dict, gate: dict) -> dict:
    if not isinstance(gate, dict):
        gate = {}

    logger.warning(
        "Candidate rejected. Candidate AP=%s, champion AP=%s. "
        "The champion alias remains unchanged.",
        gate.get("candidate_average_precision"),
        gate.get("champion_average_precision"),
    )

    client = MlflowClient(
        tracking_uri=experiments.TRACKING_URI
    )
    client.set_tag(
        candidate["run_id"],
        "promotion_status",
        "rejected",
    )

    return {
        "status": "rejected",
        "run_id": candidate["run_id"],
        "quality_gate": gate,
        "champion_changed": False,
    }


def drift_check() -> dict:
    """Run the real feature-drift detector for the latest request batch."""
    from scripts.drift_detection import run_drift_check

    result = run_drift_check()

    logger.info(
        "Drift check result: status=%s, detected=%s, "
        "drifted_features=%s/%s",
        result.get("status"),
        result.get("drift_detected"),
        result.get("drifted_feature_count"),
        result.get("feature_count"),
    )

    return result


def conditional_retrain() -> dict:
    # Phase 6 placeholder. Do not trigger training automatically yet.
    logger.warning(
        "Retraining requested by drift workflow; "
        "automatic retraining is not implemented yet."
    )
    return {
        "status": "not_implemented",
    }
