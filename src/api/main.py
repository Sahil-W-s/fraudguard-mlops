
import logging
import os
from contextlib import asynccontextmanager

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import shap

from fastapi import FastAPI, HTTPException, Query, Request
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from src.api.schemas import (
    FEATURE_NAMES,
    TransactionRequest,
    PredictionResponse,
    ExplanationResponse,
    FeatureContribution,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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


def predict_one(model, row: pd.DataFrame):
    probability = float(model.predict_proba(row)[0, 1])
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
            model, row
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
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
