
import os

import mlflow
import numpy as np
import pandas as pd

from src.api.schemas import FEATURE_NAMES


def test_registered_bundle_pipeline_and_predictions():
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://127.0.0.1:5000",
    )
    model_uri = os.getenv(
        "MLFLOW_MODEL_URI",
        "models:/FraudGuardFraudDetector@champion",
    )

    mlflow.set_tracking_uri(tracking_uri)
    model = mlflow.sklearn.load_model(model_uri)

    pipeline = model.feature_pipeline

    assert pipeline.is_fitted_
    assert pipeline.feature_names == FEATURE_NAMES

    # Synthetic input is used for an interface consistency check.
    # It is not a substitute for testing a real transaction.
    row = pd.DataFrame(
        [{feature: 1.0 for feature in FEATURE_NAMES}]
    )

    transformed = pipeline.transform(row)

    assert transformed.columns.tolist() == FEATURE_NAMES

    bundle_probability = model.predict_proba(row)[0, 1]

    assert np.isfinite(bundle_probability)
    assert 0.0 <= bundle_probability <= 1.0

    # Verify the bundle's threshold-based decision.
    expected = int(bundle_probability >= model.threshold)
    assert int(model.predict(row)[0]) == expected
