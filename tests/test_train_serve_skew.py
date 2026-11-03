
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from src.api.schemas import FEATURE_NAMES
from src.features.feature_pipeline import FeaturePipeline
from src.inference.model_bundle import FraudModelBundle


def test_registered_bundle_pipeline_and_predictions(tmp_path):
    # Build a small fitted pipeline using the production feature list.
    training_data = pd.DataFrame(
        [{feature: 1.0 for feature in FEATURE_NAMES},
         {feature: 2.0 for feature in FEATURE_NAMES}]
    )

    pipeline = FeaturePipeline(feature_names=FEATURE_NAMES)
    pipeline.fit(training_data)

    # Create a deterministic classifier for an interface test.
    estimator = DummyClassifier(strategy="prior")
    transformed = pipeline.transform(training_data)
    estimator.fit(transformed, [0, 1])

    model = FraudModelBundle(
        model_name="FraudGuardFraudDetector",
        estimator=estimator,
        feature_pipeline=pipeline,
        threshold=0.5,
    )


    tracking_db = tmp_path / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{tracking_db.as_posix()}")

    with mlflow.start_run():
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            skops_trusted_types=[
                "src.features.feature_pipeline.FeaturePipeline",
                "src.inference.model_bundle.FraudModelBundle",
            ],
        )
        run_id = mlflow.active_run().info.run_id

    model_uri = f"runs:/{run_id}/model"
    loaded_model = mlflow.sklearn.load_model(model_uri)

    loaded_pipeline = loaded_model.feature_pipeline

    assert loaded_pipeline.is_fitted_
    assert loaded_pipeline.feature_names == FEATURE_NAMES

    # Synthetic input checks the model interface, not fraud performance.
    row = pd.DataFrame(
        [{feature: 1.0 for feature in FEATURE_NAMES}]
    )

    transformed_row = loaded_pipeline.transform(row)

    assert transformed_row.columns.tolist() == FEATURE_NAMES

    bundle_probability = loaded_model.predict_proba(row)[0, 1]

    assert np.isfinite(bundle_probability)
    assert 0.0 <= bundle_probability <= 1.0

    expected = int(bundle_probability >= loaded_model.threshold)
    assert int(loaded_model.predict(row)[0]) == expected
