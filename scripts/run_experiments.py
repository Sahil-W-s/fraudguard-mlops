from __future__ import annotations
import os
import json
import logging
import subprocess
import tempfile
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.build_features import TARGET_COLUMN, TIME_COLUMN
from src.features.feature_pipeline import FeaturePipeline
from src.inference.model_bundle import FraudModelBundle


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"
MODELS_DIR = ROOT / "models"

TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)
EXPERIMENT_NAME = "FraudGuard-Fraud-Detection"
REGISTERED_MODEL_NAME = "FraudGuardFraudDetector"

PRECISION_TARGET = 0.80
RANDOM_STATE = 42

DOCS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Lineage
# ---------------------------------------------------------------------

def git_output(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        PermissionError,
    ):
        return "unknown"


def read_dvc_hash(filename: str) -> str:
    path = ROOT / "data" / "raw" / filename

    if not path.exists():
        return "not_available"

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("- md5:"):
            return line.split(":", 1)[1].strip()

    return "not_available"


def get_lineage() -> dict:
    return {
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_worktree_dirty": bool(git_output("status", "--porcelain")),
        "transaction_dvc_md5": read_dvc_hash(
            "train_transaction.csv.dvc"
        ),
        "identity_dvc_md5": read_dvc_hash(
            "train_identity.csv.dvc"
        ),
        "dataset": "IEEE-CIS Fraud Detection",
        "split_strategy": "chronological_70_15_15",
        "feature_list": "docs/validated_selected_features.csv",
    }


# ---------------------------------------------------------------------
# Data loading and splitting
# ---------------------------------------------------------------------

def load_splits():
    """Load selected model features and create chronological splits."""
    logger.info("Loading selected IEEE-CIS features...")

    transaction_path = ROOT / "data" / "raw" / "train_transaction.csv"
    identity_path = ROOT / "data" / "raw" / "train_identity.csv"
    feature_path = ROOT / "docs" / "validated_selected_features.csv"

    feature_names = (
        pd.read_csv(feature_path)["feature"].astype(str).tolist()
    )

    # Read CSV headers only to determine which table contains each feature.
    transaction_columns = set(
        pd.read_csv(transaction_path, nrows=0).columns
    )
    identity_columns = set(
        pd.read_csv(identity_path, nrows=0).columns
    )

    required = set(feature_names) | {
        "TransactionID",
        TIME_COLUMN,
        TARGET_COLUMN,
    }

    missing = required - transaction_columns - identity_columns
    if missing:
        raise ValueError(
            f"Required training columns are missing: {sorted(missing)}"
        )

    # The join key, timestamp, and target must come from the transaction table.
    transaction_required = {
        "TransactionID",
        TIME_COLUMN,
        TARGET_COLUMN,
    }

    transaction_features = [
        name for name in feature_names
        if name in transaction_columns
    ]
    identity_features = [
        name for name in feature_names
        if name in identity_columns and name not in transaction_columns
    ]

    transaction_usecols = sorted(
        transaction_required | set(transaction_features)
    )
    identity_usecols = sorted(
        {"TransactionID"} | set(identity_features)
    )

    transaction_df = pd.read_csv(
        transaction_path,
        usecols=transaction_usecols,
    )
    identity_df = pd.read_csv(
        identity_path,
        usecols=identity_usecols,
    )

    if transaction_df["TransactionID"].duplicated().any():
        raise ValueError("Duplicate TransactionIDs in transaction data.")

    if identity_df["TransactionID"].duplicated().any():
        raise ValueError("Duplicate TransactionIDs in identity data.")

    df = transaction_df.merge(
        identity_df,
        on="TransactionID",
        how="left",
        validate="one_to_one",
        sort=False,
    )

    del transaction_df, identity_df

    df.sort_values(TIME_COLUMN, inplace=True, ignore_index=True)

    n = len(df)
    train_end = int(n * 0.70)
    valid_end = int(n * 0.85)

    # Slices retain the chronological ordering without copying the full rows.
    train_df = df.iloc[:train_end]
    valid_df = df.iloc[train_end:valid_end]
    test_df = df.iloc[valid_end:]

    y_train = train_df[TARGET_COLUMN].astype(int)
    y_valid = valid_df[TARGET_COLUMN].astype(int)
    y_test = test_df[TARGET_COLUMN].astype(int)

    feature_pipeline = FeaturePipeline(feature_names=feature_names)

    X_train = feature_pipeline.fit_transform(train_df)
    X_valid = feature_pipeline.transform(valid_df)
    X_test = feature_pipeline.transform(test_df)

    logger.info(
        "Split sizes: train=%s, validation=%s, test=%s",
        len(train_df), len(valid_df), len(test_df),
    )
    logger.info("Selected feature count: %s", X_train.shape[1])
    logger.info("Training fraud prevalence: %.5f", y_train.mean())

    return {
        "train_df": train_df,
        "valid_df": valid_df,
        "test_df": test_df,
        "y_train": y_train,
        "y_valid": y_valid,
        "y_test": y_test,
        "X_train": X_train,
        "X_valid": X_valid,
        "X_test": X_test,
        "feature_pipeline": feature_pipeline,
    }

# ---------------------------------------------------------------------
# Preprocessing for non-LightGBM models
# ---------------------------------------------------------------------

def make_preprocessor(
    numeric_columns: list[str],
    categorical_columns: list[str],
    scale_numeric: bool,
) -> ColumnTransformer:
    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median")),
    ]

    if scale_numeric:
        numeric_steps.append(
            ("scaler", StandardScaler())
        )

    numeric_pipeline = Pipeline(numeric_steps)

    categorical_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=10,
                    sparse_output=True,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


# ---------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------

def choose_threshold(
    y_true,
    probabilities: np.ndarray,
    precision_target: float = PRECISION_TARGET,
) -> tuple[float, float, float]:
    """
    Select the threshold with maximum recall among thresholds
    meeting the required validation precision.

    If no threshold meets the target, select the threshold with
    the highest achievable precision and record that limitation.
    """
    precision, recall, thresholds = precision_recall_curve(
        y_true,
        probabilities,
    )

    # The final precision/recall point has no corresponding threshold.
    precision_at_threshold = precision[:-1]
    recall_at_threshold = recall[:-1]

    eligible = np.flatnonzero(
        precision_at_threshold >= precision_target
    )

    if len(eligible):
        best_idx = eligible[
            np.argmax(recall_at_threshold[eligible])
        ]
    else:
        best_idx = int(np.argmax(precision_at_threshold))
        logger.warning(
            "No validation threshold reached precision %.2f; "
            "using the highest-precision threshold.",
            precision_target,
        )

    return (
        float(thresholds[best_idx]),
        float(precision_at_threshold[best_idx]),
        float(recall_at_threshold[best_idx]),
    )


def calculate_metrics(
    y_true,
    probabilities: np.ndarray,
    threshold: float,
) -> dict:
    predictions = (
        probabilities >= threshold
    ).astype(int)

    return {
        "average_precision": float(
            average_precision_score(y_true, probabilities)
        ),
        "roc_auc": float(
            roc_auc_score(y_true, probabilities)
        ),
        "log_loss": float(
            log_loss(y_true, probabilities, labels=[0, 1])
        ),
        "precision": float(
            precision_score(
                y_true, predictions, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_true, predictions, zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                y_true, predictions, zero_division=0
            )
        ),
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),
    }


# ---------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------

def save_confusion_matrix(
    y_true,
    probabilities: np.ndarray,
    threshold: float,
    output_path: Path,
    title: str,
) -> None:
    predictions = (
        probabilities >= threshold
    ).astype(int)

    matrix = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Legitimate", "Fraud"],
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    display.plot(
        ax=ax,
        values_format="d",
        colorbar=False,
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    # Also preserve the exact counts as machine-readable JSON.
    counts_path = output_path.with_suffix(".json")
    tn, fp, fn, tp = matrix.ravel()

    counts_path.write_text(
        json.dumps(
            {
                "threshold": float(threshold),
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def save_feature_importance(
    estimator,
    model_name: str,
    feature_names: list[str],
    output_path: Path,
) -> None:
    if model_name == "logistic_regression":
        values = np.abs(
            np.asarray(estimator.coef_).reshape(-1)
        )
    else:
        values = np.asarray(estimator.feature_importances_)

    if len(values) != len(feature_names):
        logger.warning(
            "Feature-importance dimensions do not match for %s.",
            model_name,
        )
        return

    order = np.argsort(values)[-30:]
    fig, ax = plt.subplots(figsize=(10, 9))

    ax.barh(
        [feature_names[i] for i in order],
        values[order],
    )
    ax.set_title(f"Top 30 feature importances: {model_name}")
    ax.set_xlabel(
        "Absolute coefficient"
        if model_name == "logistic_regression"
        else "Feature importance"
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------

def train_experiment(
    model_name: str,
    data: dict,
    lineage: dict,
) -> dict:
    X_train = data["X_train"]
    X_valid = data["X_valid"]
    y_train = data["y_train"]
    y_valid = data["y_valid"]
    feature_pipeline = data["feature_pipeline"]

    numeric_columns = feature_pipeline.numeric_columns_
    categorical_columns = feature_pipeline.categorical_columns_

    preprocessor = None

    if model_name == "logistic_regression":
        preprocessor = make_preprocessor(
            numeric_columns,
            categorical_columns,
            scale_numeric=True,
        )

        X_train_model = preprocessor.fit_transform(X_train)
        X_valid_model = preprocessor.transform(X_valid)

        estimator = LogisticRegression(
            class_weight="balanced",
            solver="saga",
            max_iter=100,
            tol=1e-3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        params = {
            "class_weight": "balanced",
            "solver": "saga",
            "max_iter": 100,
            "tol": 1e-3,
        }

    elif model_name == "lightgbm":
        X_train_model = X_train
        X_valid_model = X_valid

        estimator = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=1500,
            learning_rate=0.05,
            num_leaves=31,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
        )

        params = {
            "objective": "binary",
            "n_estimators": 1500,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "class_weight": "balanced",
            "early_stopping_rounds": 50,
        }

    elif model_name == "random_forest":
        preprocessor = make_preprocessor(
            numeric_columns,
            categorical_columns,
            scale_numeric=False,
        )

        X_train_model = preprocessor.fit_transform(X_train)
        X_valid_model = preprocessor.transform(X_valid)

        estimator = RandomForestClassifier(
            n_estimators=200,
            max_depth=18,
            min_samples_leaf=10,
            max_features="sqrt",
            class_weight="balanced_subsample",
            max_samples=0.5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        params = {
            "n_estimators": 200,
            "max_depth": 18,
            "min_samples_leaf": 10,
            "max_features": "sqrt",
            "class_weight": "balanced_subsample",
            "max_samples": 0.5,
        }

    else:
        raise ValueError(f"Unknown model: {model_name}")

    with mlflow.start_run(run_name=model_name) as active_run:
        run_id = active_run.info.run_id

        mlflow.set_tags(
            {
                "model_family": model_name,
                "dataset": lineage["dataset"],
                "split_strategy": lineage["split_strategy"],
                "git_commit": lineage["git_commit"],
                "git_worktree_dirty": str(
                    lineage["git_worktree_dirty"]
                ).lower(),
                "transaction_dvc_md5": lineage[
                    "transaction_dvc_md5"
                ],
                "identity_dvc_md5": lineage[
                    "identity_dvc_md5"
                ],
                "primary_metric": "validation_average_precision",
            }
        )

        mlflow.log_params(params)
        mlflow.log_param("precision_target", PRECISION_TARGET)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("feature_count", X_train.shape[1])
        mlflow.log_param("training_rows", len(y_train))
        mlflow.log_param("validation_rows", len(y_valid))
        mlflow.log_param("validation_fraud_rate", float(y_valid.mean()))

        logger.info("Training %s...", model_name)

        if model_name == "lightgbm":
            estimator.fit(
                X_train_model,
                y_train,
                eval_set=[(X_valid_model, y_valid)],
                eval_metric="auc",
                categorical_feature=categorical_columns,
                callbacks=[
                    lgb.early_stopping(
                        stopping_rounds=50,
                        verbose=True,
                    ),
                ],
            )

            mlflow.log_param(
                "best_iteration",
                int(estimator.best_iteration_ or 1500),
            )

        else:
            estimator.fit(X_train_model, y_train)

        probabilities = estimator.predict_proba(
            X_valid_model
        )[:, 1]

        threshold, achieved_precision, achieved_recall = (
            choose_threshold(y_valid, probabilities)
        )

        metrics = calculate_metrics(
            y_valid,
            probabilities,
            threshold,
        )

        mlflow.log_metrics(
            {
                f"validation_{key}": value
                for key, value in metrics.items()
            }
        )
        mlflow.log_metric(
            "validation_recall_at_precision_target",
            achieved_recall,
        )
        mlflow.log_metric(
            "validation_precision_at_selected_threshold",
            achieved_precision,
        )
        mlflow.log_metric(
            "validation_selected_threshold",
            threshold,
        )

        bundle = FraudModelBundle(
            model_name=model_name,
            estimator=estimator,
            feature_pipeline=feature_pipeline,
            preprocessor=preprocessor,
            threshold=threshold,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)

            save_confusion_matrix(
                y_valid,
                probabilities,
                threshold,
                temp / "validation_confusion_matrix.png",
                f"{model_name}: validation confusion matrix",
            )

            if preprocessor is None:
                importance_names = feature_pipeline.feature_names
            else:
                importance_names = (
                    preprocessor.get_feature_names_out().tolist()
                )

            save_feature_importance(
                estimator,
                model_name,
                importance_names,
                temp / "feature_importance.png",
            )

            mlflow.log_artifacts(
                temp_dir,
                artifact_path="validation_artifacts",
            )

        # Log the full preprocessing + estimator bundle.
        # The code path makes the custom wrapper available when loading.
        sample_input = data["train_df"].head(5)
        sample_output = bundle.predict_proba(sample_input)

        signature = infer_signature(
            sample_input,
            sample_output,
        )

                # Do not expose the target or transaction identifier
        # in the model's serving signature.
        sample_input = (
            data["train_df"]
            .drop(
                columns=[TARGET_COLUMN, "TransactionID"],
                errors="ignore",
            )
            .head(5)
        )

        sample_output = bundle.predict_proba(sample_input)

        signature = infer_signature(
            sample_input,
            sample_output,
        )

        mlflow.sklearn.log_model(
            sk_model=bundle,
            name="model",
            signature=signature,
            input_example=sample_input,
            code_paths=[str(ROOT / "src")],
            skops_trusted_types=[
                "numpy.dtype",
                "src.features.feature_pipeline.FeaturePipeline",
                "src.inference.model_bundle.FraudModelBundle",
                "collections.OrderedDict",
                "lightgbm.basic.Booster",
                "lightgbm.sklearn.LGBMClassifier",
                "sklearn.tree._tree.Tree",
                "sklearn.compose._column_transformer._RemainderColsList",
            ],
        )

        logger.info(
            "%s validation AP=%.6f, recall=%.6f, precision=%.6f",
            model_name,
            metrics["average_precision"],
            achieved_recall,
            achieved_precision,
        )

        return {
            "model_name": model_name,
            "run_id": run_id,
            "validation_average_precision": metrics[
                "average_precision"
            ],
            "validation_recall_at_precision_target": achieved_recall,
            "validation_precision_at_selected_threshold": (
                achieved_precision
            ),
            "threshold": threshold,
            "params": params,
        }


# ---------------------------------------------------------------------
# Final evaluation and registration
# ---------------------------------------------------------------------

def evaluate_winner(
    winner: dict,
    data: dict,
    lineage: dict,
    register_model: bool = True,
) -> dict:
    run_id = winner["run_id"]

    # Rebuild the winner's complete pipeline from its logged artifact.
    model_uri = f"runs:/{run_id}/model"
    bundle = mlflow.sklearn.load_model(model_uri)

    test_df = data["test_df"]
    y_test = data["y_test"]

    probabilities = bundle.predict_proba(test_df)[:, 1]

    test_metrics = calculate_metrics(
        y_test,
        probabilities,
        bundle.threshold,
    )

    client = MlflowClient(tracking_uri=TRACKING_URI)

    for key, value in test_metrics.items():
        client.log_metric(
            run_id,
            f"test_{key}",
            float(value),
        )

    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)

        save_confusion_matrix(
            y_test,
            probabilities,
            bundle.threshold,
            temp / "test_confusion_matrix.png",
            "Selected model: held-out test confusion matrix",
        )

        client.log_artifacts(
            run_id,
            temp_dir,
            artifact_path="test_artifacts",
        )

        # Airflow evaluates candidates before allowing promotion.
    # Candidate-only mode must never change the champion alias.
    if not register_model:
        summary = {
            "experiment_name": EXPERIMENT_NAME,
            "registered_model_name": REGISTERED_MODEL_NAME,
            "registered_model_version": None,
            "registered_model_alias": None,
            "candidate_only": True,
            "winning_model": winner["model_name"],
            "winning_run_id": run_id,
            "validation_average_precision": (
                winner["validation_average_precision"]
            ),
            "validation_recall_at_precision_target": (
                winner["validation_recall_at_precision_target"]
            ),
            "decision_threshold": bundle.threshold,
            "precision_target": PRECISION_TARGET,
            "test_metrics": test_metrics,
            "lineage": lineage,
        }

        summary_path = DOCS_DIR / "airflow_candidate_evaluation.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

        logger.info(
            "Candidate evaluated without registration. Test metrics: %s",
            test_metrics,
        )
        logger.info("Candidate summary saved to %s", summary_path)

        return summary


    # Register only the validation-selected model.
    registered = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME,
        await_registration_for=120,
    )

    client.set_registered_model_alias(
        REGISTERED_MODEL_NAME,
        "champion",
        registered.version,
    )

    client.update_model_version(
        name=REGISTERED_MODEL_NAME,
        version=registered.version,
        description=(
            f"Selected by validation Average Precision. "
            f"Validation AP={winner['validation_average_precision']:.6f}. "
            f"Test AP={test_metrics['average_precision']:.6f}."
        ),
    )

    version_tags = {
        "git_commit": lineage["git_commit"],
        "transaction_dvc_md5": lineage["transaction_dvc_md5"],
        "identity_dvc_md5": lineage["identity_dvc_md5"],
        "selection_metric": "validation_average_precision",
        "validation_average_precision": str(
            winner["validation_average_precision"]
        ),
        "test_average_precision": str(
            test_metrics["average_precision"]
        ),
        "precision_target": str(PRECISION_TARGET),
    }

    for key, value in version_tags.items():
        client.set_model_version_tag(
            REGISTERED_MODEL_NAME,
            registered.version,
            key,
            value,
        )

    summary = {
        "experiment_name": EXPERIMENT_NAME,
        "registered_model_name": REGISTERED_MODEL_NAME,
        "registered_model_version": str(registered.version),
        "registered_model_alias": "champion",
        "winning_model": winner["model_name"],
        "winning_run_id": run_id,
        "validation_average_precision": (
            winner["validation_average_precision"]
        ),
        "validation_recall_at_precision_target": (
            winner["validation_recall_at_precision_target"]
        ),
        "decision_threshold": bundle.threshold,
        "precision_target": PRECISION_TARGET,
        "test_metrics": test_metrics,
        "lineage": lineage,
    }

    summary_path = DOCS_DIR / "phase3_experiment_summary.json"

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    logger.info("Registered model version: %s", registered.version)
    logger.info("Assigned alias: champion")
    logger.info("Test metrics: %s", test_metrics)
    logger.info("Summary saved to %s", summary_path)

    return summary


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    lineage = get_lineage()
    data = load_splits()

    model_names = [
        "logistic_regression",
        "lightgbm",
        "random_forest",
    ]

    results = []

    for model_name in model_names:
        results.append(
            train_experiment(model_name, data, lineage)
        )

    results.sort(
        key=lambda item: item["validation_average_precision"],
        reverse=True,
    )

    print("\nValidation ranking:")
    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. {result['model_name']}: "
            f"AP={result['validation_average_precision']:.6f}, "
            f"recall@precision={result['validation_recall_at_precision_target']:.6f}"
        )

    winner = results[0]

    print(
        "\nSelected using validation Average Precision:",
        winner["model_name"],
    )

    summary = evaluate_winner(winner, data, lineage)

    print("\nPhase 3 summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()