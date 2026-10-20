
import logging
import os
from contextlib import asynccontextmanager

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import shap

from fastapi import FastAPI, HTTPException, Query, Request
from time import perf_counter
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.responses import Response

from src.api.schemas import (
    FEATURE_NAMES,
    TransactionRequest,
    PredictionResponse,
    ExplanationResponse,
    FeatureContribution,
)
from src.api.prediction_store import (
    initialize_store,
    record_prediction,
    get_prediction_stats,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# HTTP request metrics
HTTP_REQUESTS = Counter(
    "fraudguard_http_requests_total",
    "Total HTTP requests handled by the FraudGuard API.",
    ["method", "endpoint", "status_code"],
)

HTTP_REQUEST_DURATION = Histogram(
    "fraudguard_http_request_duration_seconds",
    "HTTP request duration in seconds.",
    ["method", "endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

# Model inference latency
INFERENCE_DURATION = Histogram(
    "fraudguard_inference_duration_seconds",
    "Model inference duration in seconds.",
    ["operation"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

# Prediction class distribution
PREDICTIONS = Counter(
    "fraudguard_predictions_total",
    "Total successful predictions returned by the prediction endpoint.",
    ["predicted_fraud"],
)

# Prediction monitoring and delayed-label evaluation
PREDICTED_FRAUD_RATE = Gauge(
    "fraudguard_predicted_fraud_rate",
    "Fraction of all recorded predictions classified as fraud.",
)

LABELED_PREDICTIONS = Gauge(
    "fraudguard_labeled_predictions",
    "Number of predictions with an assigned ground-truth label.",
)

ACTUAL_FRAUD_RATE = Gauge(
    "fraudguard_actual_fraud_rate",
    "Fraction of labeled predictions whose ground-truth label is fraud.",
)

EVALUATION_PRECISION = Gauge(
    "fraudguard_evaluation_precision",
    "Precision over predictions with available ground-truth labels.",
)

EVALUATION_RECALL = Gauge(
    "fraudguard_evaluation_recall",
    "Recall over predictions with available ground-truth labels.",
)

TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)
MODEL_URI = os.getenv(
    "MLFLOW_MODEL_URI",
    "models:/FraudGuardFraudDetector@champion",
)


def load_model():
    mlflow.set_tracking_uri(TRACKING_URI)
    model = mlflow.sklearn.load_model(MODEL_URI)

    if not hasattr(model, "feature_pipeline"):
        raise TypeError("Registered model has no feature pipeline.")

    pipeline = model.feature_pipeline

    if not pipeline.is_fitted_:
        raise RuntimeError("Registered feature pipeline is not fitted.")

    if list(pipeline.feature_names) != FEATURE_NAMES:
        raise RuntimeError(
            "Serving feature list does not match the registered "
            "model's fitted feature list."
        )

    if not hasattr(model, "estimator"):
        raise TypeError("Registered model has no estimator.")

    if not hasattr(model.estimator, "predict_proba"):
        raise TypeError("Registered estimator has no predict_proba().")

    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = None
    app.state.explainer = None
    app.state.model_error = None

    try:
        initialize_store()
        logger.info("Prediction store initialized.")
    except Exception:
        logger.exception("Prediction store initialization failed.")
        raise

    try:
        app.state.model = load_model()
        logger.info("Loaded registered model: %s", MODEL_URI)

        try:
            app.state.explainer = shap.TreeExplainer(
                app.state.model.estimator
            )
            logger.info("SHAP TreeExplainer initialized.")
        except Exception:
            logger.exception("SHAP initialization failed.")

    except Exception as exc:
        app.state.model_error = str(exc)
        logger.exception("Model loading failed.")

    yield


app = FastAPI(
    title="FraudGuard API",
    description=(
        "Fraud detection API backed by the MLflow champion model. "
        "Invalid request bodies intentionally return HTTP 422."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

@app.middleware("http")
async def observe_http_requests(request: Request, call_next):
    # Avoid counting Prometheus scrapes as application traffic.
    if request.url.path == "/metrics":
        return await call_next(request)

    start_time = perf_counter()
    status_code = 500

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        elapsed = perf_counter() - start_time

        route = request.scope.get("route")
        endpoint = getattr(route, "path", request.url.path)

        HTTP_REQUESTS.labels(
            method=request.method,
            endpoint=endpoint,
            status_code=str(status_code),
        ).inc()

        HTTP_REQUEST_DURATION.labels(
            method=request.method,
            endpoint=endpoint,
        ).observe(elapsed)

def refresh_evaluation_metrics() -> None:
    stats = get_prediction_stats()

    total = stats["total_predictions"]
    labeled = stats["labeled_predictions"]
    tp = stats["true_positives"]
    fp = stats["false_positives"]
    fn = stats["false_negatives"]

    PREDICTED_FRAUD_RATE.set(
        stats["predicted_fraud_count"] / total if total else 0
    )

    LABELED_PREDICTIONS.set(labeled)

    ACTUAL_FRAUD_RATE.set(
        stats["actual_fraud_count"] / labeled if labeled else 0
    )

    # Do not report precision or recall as zero when no labels exist.
    EVALUATION_PRECISION.set(
        tp / (tp + fp) if tp + fp else float("nan")
    )

    EVALUATION_RECALL.set(
        tp / (tp + fn) if tp + fn else float("nan")
    )

def get_model(request: Request):
    model = request.app.state.model

    if model is None:
        raise HTTPException(
            status_code=503,
            detail={
                "message": "Fraud model is unavailable.",
                "reason": request.app.state.model_error,
            },
        )

    return model


def predict_one(
    model,
    row: pd.DataFrame,
    operation: str = "predict",
):
    start_time = perf_counter()

    try:
        probability = float(model.predict_proba(row)[0, 1])
    finally:
        INFERENCE_DURATION.labels(
            operation=operation
        ).observe(perf_counter() - start_time)

    threshold = float(model.threshold)
    prediction = int(probability >= threshold)

    return probability, prediction, threshold


@app.get("/health")
def health(request: Request):
    model = request.app.state.model

    if model is None:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "unhealthy",
                "model_loaded": False,
                "reason": request.app.state.model_error,
            },
        )

    registry_reachable = False

    try:
        from mlflow import MlflowClient

        client = MlflowClient(tracking_uri=TRACKING_URI)
        client.get_model_version_by_alias(
            "FraudGuardFraudDetector",
            "champion",
        )
        registry_reachable = True
    except Exception:
        logger.exception("MLflow registry health check failed.")

    if not registry_reachable:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "degraded",
                "model_loaded": True,
                "mlflow_reachable": False,
            },
        )

    return {
        "status": "healthy",
        "model_loaded": True,
        "mlflow_reachable": True,
        "model_uri": MODEL_URI,
        "feature_count": len(FEATURE_NAMES),
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    payload: TransactionRequest,
    request: Request,
):
    model = get_model(request)
    row = pd.DataFrame([payload.model_dump()])

    try:
        probability, prediction, threshold = predict_one(
            model, row
        )
    except Exception as exc:
        logger.exception("Prediction failed.")
        raise HTTPException(
            status_code=422,
            detail=f"Unable to process transaction features: {exc}",
        ) from exc

    PREDICTIONS.labels(
        predicted_fraud=str(prediction)
    ).inc()

    try:
        record_prediction(
            fraud_probability=probability,
            predicted_fraud=prediction,
            threshold=threshold,
        )
        refresh_evaluation_metrics()
    except Exception:
        logger.exception("Failed to persist prediction.")
        raise HTTPException(
            status_code=503,
            detail="Prediction logging is temporarily unavailable.",
        )

    return PredictionResponse(
        fraud_probability=probability,
        predicted_fraud=prediction,
        threshold=threshold,
    )


@app.post(
    "/explain",
    response_model=ExplanationResponse,
)
def explain(
    payload: TransactionRequest,
    request: Request,
    top_n: int = Query(default=10, ge=1, le=80),
):
    model = get_model(request)
    explainer = request.app.state.explainer

    if explainer is None:
        raise HTTPException(
            status_code=503,
            detail="SHAP explainer is unavailable.",
        )

    row = pd.DataFrame([payload.model_dump()])

    try:
        probability, prediction, threshold = predict_one(
            model,
            row,
            operation="explain",
        )

        transformed = model.feature_pipeline.transform(row)

        if model.preprocessor is not None:
            transformed = model.preprocessor.transform(transformed)

        shap_values = explainer.shap_values(transformed)

        # SHAP versions may return either a list of class arrays
        # or a NumPy array.
        if isinstance(shap_values, list):
            contributions = np.asarray(shap_values[1])[0]
        else:
            contributions = np.asarray(shap_values)

            if contributions.ndim == 3:
                contributions = contributions[0, :, 1]
            elif contributions.ndim == 2:
                contributions = contributions[0]

        feature_names = list(transformed.columns)

        if len(feature_names) != len(contributions):
            raise RuntimeError(
                "SHAP output does not match transformed features."
            )

        ranked = sorted(
            zip(feature_names, contributions),
            key=lambda item: abs(float(item[1])),
            reverse=True,
        )[:top_n]

        top_features = [
            FeatureContribution(
                feature=name,
                value=float(value),
                direction=(
                    "increases model output"
                    if value > 0
                    else "decreases model output"
                    if value < 0
                    else "no contribution"
                ),
            )
            for name, value in ranked
        ]

    except Exception as exc:
        logger.exception("SHAP explanation failed.")
        raise HTTPException(
            status_code=503,
            detail=f"Unable to generate explanation: {exc}",
        ) from exc

    return ExplanationResponse(
        fraud_probability=probability,
        predicted_fraud=prediction,
        threshold=threshold,
        top_features=top_features,
    )


@app.get("/metrics")
def metrics():
    try:
        refresh_evaluation_metrics()
    except Exception:
        logger.exception("Failed to refresh evaluation metrics.")

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )