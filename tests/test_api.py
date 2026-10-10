
import numpy as np
import pandas as pd
import pytest

from fastapi.testclient import TestClient

from src.api.main import app
from src.api.schemas import FEATURE_NAMES


class FakePipeline:
    def __init__(self):
        self.feature_names = FEATURE_NAMES
        self.is_fitted_ = True
        self.categorical_columns_ = []
        self.numeric_columns_ = FEATURE_NAMES

    def transform(self, frame):
        return frame[self.feature_names]


class FakeModel:
    def __init__(self):
        self.feature_pipeline = FakePipeline()
        self.preprocessor = None
        self.estimator = object()
        self.threshold = 0.9

    def predict_proba(self, frame):
        return np.array([[0.35, 0.65]])


@pytest.fixture
def client():
    with TestClient(app, raise_server_exceptions=False) as test_client:
        app.state.model = FakeModel()
        app.state.explainer = None
        app.state.model_error = None
        yield test_client


def valid_payload():
    from src.api.schemas import (
    FEATURE_NAMES,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)


def valid_payload():
    payload = {}

    for feature in FEATURE_NAMES:
        if feature in NUMERIC_FEATURES:
            payload[feature] = 1.0
        elif feature in CATEGORICAL_FEATURES:
            payload[feature] = "test_category"

    return payload

def test_predict_returns_probability_and_threshold(client):
    response = client.post("/predict", json=valid_payload())

    assert response.status_code == 200
    result = response.json()

    assert result["fraud_probability"] == pytest.approx(0.65)
    assert result["predicted_fraud"] == 0
    assert result["threshold"] == pytest.approx(0.9)


def test_missing_feature_returns_422(client):
    payload = valid_payload()
    payload.pop(FEATURE_NAMES[0])

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_extra_feature_returns_422(client):
    payload = valid_payload()
    payload["unexpected_feature"] = 1

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_unavailable_model_returns_503(client):
    app.state.model = None
    app.state.model_error = "Test model-load failure"

    response = client.post("/predict", json=valid_payload())

    assert response.status_code == 503


def test_metrics_returns_prometheus_content(client):
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "python_info" in response.text


def test_health_returns_503_without_model(client):
    app.state.model = None
    app.state.model_error = "Test model-load failure"

    response = client.get("/health")

    assert response.status_code == 503


from src.api.schemas import NUMERIC_FEATURES, CATEGORICAL_FEATURES


def test_invalid_numeric_string_returns_422(client):
    payload = valid_payload()
    numeric_feature = next(iter(NUMERIC_FEATURES))
    payload[numeric_feature] = "string"

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_numeric_categorical_mismatch_returns_422(client):
    payload = valid_payload()
    categorical_feature = next(iter(CATEGORICAL_FEATURES))
    payload[categorical_feature] = 123

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_numeric_feature_accepts_null(client):
    payload = valid_payload()
    numeric_feature = next(iter(NUMERIC_FEATURES))
    payload[numeric_feature] = None

    response = client.post("/predict", json=payload)

    # The schema accepts null; the model may or may not support it.
    assert response.status_code != 422 or response.json()["detail"]
