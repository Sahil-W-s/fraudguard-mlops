from pathlib import Path

import joblib
import pandas as pd

from src.features.feature_pipeline import FeaturePipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "fraud_model.joblib"
PIPELINE_PATH = PROJECT_ROOT / "models" / "feature_pipeline.joblib"


class FraudPredictor:
    """Load saved artifacts and predict fraud probabilities."""

    def __init__(
        self,
        model_path: str | Path = MODEL_PATH,
        pipeline_path: str | Path = PIPELINE_PATH,
    ):
        self.model_path = Path(model_path)
        self.pipeline_path = Path(pipeline_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found: {self.model_path}"
            )

        if not self.pipeline_path.exists():
            raise FileNotFoundError(
                f"Feature pipeline not found: {self.pipeline_path}"
            )

        self.model = joblib.load(self.model_path)
        self.pipeline = FeaturePipeline.load(self.pipeline_path)

        if hasattr(self.model, "feature_name_"):
            model_features = list(self.model.feature_name_)
            if model_features != self.pipeline.feature_names:
                raise ValueError(
                    "Model and pipeline feature orders do not match."
                )

    def predict_proba(self, transactions: pd.DataFrame) -> pd.Series:
        """
        Return the estimated fraud probability for each transaction.

        Input must contain the raw columns required by the fitted pipeline.
        """
        if transactions.empty:
            return pd.Series(
                index=transactions.index,
                dtype="float64",
                name="fraud_probability",
            )

        features = self.pipeline.transform(transactions)

        probabilities = self.model.predict_proba(
            features,
            num_iteration=self.model.best_iteration_,
        )[:, 1]

        return pd.Series(
            probabilities,
            index=transactions.index,
            name="fraud_probability",
        )

    def predict(
        self,
        transactions: pd.DataFrame,
        threshold: float = 0.5,
    ) -> pd.DataFrame:
        """
        Return fraud probabilities and threshold-based predictions.

        The default threshold is a placeholder, not a calibrated
        or business-approved operating threshold.
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1.")

        probabilities = self.predict_proba(transactions)

        return pd.DataFrame(
            {
                "fraud_probability": probabilities,
                "predicted_fraud": (
                    probabilities >= threshold
                ).astype("int8"),
            },
            index=transactions.index,
        )