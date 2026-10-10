
from pathlib import Path
from typing import Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, create_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_LIST_PATH = (
    PROJECT_ROOT / "docs" / "validated_selected_features.csv"
)

# Fallback feature types from the verified fitted pipeline.
CATEGORICAL_FALLBACK = {
    "DeviceInfo", "R_emaildomain", "P_emaildomain", "id_31",
    "card6", "id_33", "id_30", "M6", "ProductCD", "M5", "M4",
}


def load_selected_features() -> list[str]:
    if not FEATURE_LIST_PATH.exists():
        raise FileNotFoundError(
            f"Selected feature list not found: {FEATURE_LIST_PATH}"
        )

    feature_list = pd.read_csv(FEATURE_LIST_PATH)

    if "feature" not in feature_list.columns:
        raise ValueError("Feature list must contain a 'feature' column.")

    features = feature_list["feature"].astype(str).tolist()

    if not features or len(features) != len(set(features)):
        raise ValueError("Selected features must be non-empty and unique.")

    return features


FEATURE_NAMES = load_selected_features()


def load_feature_types() -> tuple[set[str], set[str]]:
    """Load feature types from the registered fitted pipeline."""
    try:
        import mlflow
        import mlflow.sklearn
        import os

        mlflow.set_tracking_uri(
            os.getenv(
                "MLFLOW_TRACKING_URI",
                "http://127.0.0.1:5000",
            )
        )
        model = mlflow.sklearn.load_model(
            os.getenv(
                "MLFLOW_MODEL_URI",
                "models:/FraudGuardFraudDetector@champion",
            )
        )
        pipeline = model.feature_pipeline

        if pipeline.feature_names != FEATURE_NAMES:
            raise RuntimeError(
                "Registered pipeline features do not match the API schema."
            )

        return (
            set(pipeline.numeric_columns_),
            set(pipeline.categorical_columns_),
        )
    except Exception:
        # Schema construction remains possible when MLflow is offline.
        # The API itself will report model-load failure during startup.
        return (
            set(FEATURE_NAMES) - CATEGORICAL_FALLBACK,
            CATEGORICAL_FALLBACK,
        )


NUMERIC_FEATURES, CATEGORICAL_FEATURES = load_feature_types()

if NUMERIC_FEATURES | CATEGORICAL_FEATURES != set(FEATURE_NAMES):
    raise RuntimeError("Feature types do not cover the selected features.")

if NUMERIC_FEATURES & CATEGORICAL_FEATURES:
    raise RuntimeError("Feature types overlap.")


field_definitions = {}

for feature in FEATURE_NAMES:
    if feature in NUMERIC_FEATURES:
        field_definitions[feature] = (Optional[float], ...)
    else:
        field_definitions[feature] = (Optional[str], ...)


TransactionRequest = create_model(
    "TransactionRequest",
    __config__=ConfigDict(extra="forbid"),
    **field_definitions,
)


class PredictionResponse(BaseModel):
    fraud_probability: float
    predicted_fraud: int
    threshold: float


class FeatureContribution(BaseModel):
    feature: str
    value: float
    direction: str


class ExplanationResponse(PredictionResponse):
    top_features: list[FeatureContribution]
