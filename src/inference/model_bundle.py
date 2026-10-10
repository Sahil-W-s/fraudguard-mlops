from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin


class FraudModelBundle(ClassifierMixin, BaseEstimator):
    """Complete FraudGuard preprocessing + classifier bundle."""

    def __init__(
        self,
        model_name,
        estimator,
        feature_pipeline,
        preprocessor=None,
        threshold=0.5,
    ):
        self.model_name = model_name
        self.estimator = estimator
        self.feature_pipeline = feature_pipeline
        self.preprocessor = preprocessor
        self.threshold = threshold

    def predict_proba(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        features = self.feature_pipeline.transform(X)

        if self.preprocessor is not None:
            features = self.preprocessor.transform(features)

        return self.estimator.predict_proba(features)

    def predict(self, X):
        probabilities = self.predict_proba(X)[:, 1]
        return (probabilities >= self.threshold).astype(int)