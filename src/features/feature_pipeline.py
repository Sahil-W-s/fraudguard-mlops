from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEATURE_LIST = (
    PROJECT_ROOT / "docs" / "validated_selected_features.csv"
)


class FeaturePipeline:
    """Reusable, fitted feature transformation pipeline."""

    def __init__(
        self,
        feature_names: Optional[list[str]] = None,
    ):
        self.feature_names = feature_names
        self.categorical_columns_: list[str] = []
        self.numeric_columns_: list[str] = []
        self.category_levels_: dict[str, list[str]] = {}
        self.is_fitted_ = False

    @staticmethod
    def _is_categorical(series: pd.Series) -> bool:
        dtype = series.dtype

        return (
            is_object_dtype(dtype)
            or is_string_dtype(dtype)
            or isinstance(dtype, pd.CategoricalDtype)
        )

    def _validate_columns(self, df: pd.DataFrame) -> None:
        if self.feature_names is None:
            raise ValueError(
                "Feature names are not configured."
            )

        if not df.columns.is_unique:
            raise ValueError(
                "Input DataFrame contains duplicate column names."
            )

        missing = [
            column
            for column in self.feature_names
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required features: {missing}"
            )

    def fit(self, df: pd.DataFrame):
        """Learn feature types and categorical levels from training data."""

        if self.feature_names is None:
            feature_list = pd.read_csv(DEFAULT_FEATURE_LIST)

            if "feature" not in feature_list.columns:
                raise ValueError(
                    "Feature list must contain a 'feature' column."
                )

            self.feature_names = (
                feature_list["feature"].astype(str).tolist()
            )

        if not self.feature_names:
            raise ValueError("Feature list cannot be empty.")

        if len(self.feature_names) != len(set(self.feature_names)):
            raise ValueError(
                "Feature list contains duplicate names."
            )

        self._validate_columns(df)

        self.categorical_columns_ = []
        self.numeric_columns_ = []
        self.category_levels_ = {}

        X = df[self.feature_names]

        for column in self.feature_names:
            series = X[column]

            if self._is_categorical(series):
                self.categorical_columns_.append(column)

                values = series.astype("string")
                levels = values.dropna().unique().tolist()

                self.category_levels_[column] = sorted(levels)

            elif (
                is_numeric_dtype(series.dtype)
                or is_bool_dtype(series.dtype)
            ):
                self.numeric_columns_.append(column)

            else:
                raise TypeError(
                    f"Unsupported dtype for {column}: "
                    f"{series.dtype}"
                )

        self.is_fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return features in the exact training column order."""

        if not self.is_fitted_:
            raise RuntimeError(
                "Call fit() before transform()."
            )

        self._validate_columns(df)

        X = df[self.feature_names].copy()

        for column in self.categorical_columns_:
            values = X[column].astype("string")
            known_categories = self.category_levels_[column]

            # Map unseen categories to missing before categorical conversion.
            values = values.where(
                values.isna() | values.isin(known_categories),
                pd.NA,
            )

            X[column] = pd.Categorical(
                values,
                categories=known_categories,
            )

        for column in self.numeric_columns_:
            X[column] = pd.to_numeric(
                X[column],
                errors="raise",
            )

        return X

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit the pipeline and transform the same training data."""

        return self.fit(df).transform(df)

    def save(self, path: str | Path) -> None:
        """Persist the fitted pipeline."""

        if not self.is_fitted_:
            raise RuntimeError(
                "Cannot save an unfitted pipeline."
            )

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "FeaturePipeline":
        """Load a previously fitted pipeline."""

        pipeline = joblib.load(path)

        if not isinstance(pipeline, cls):
            raise TypeError(
                "Saved artifact is not a FeaturePipeline."
            )

        if not pipeline.is_fitted_:
            raise RuntimeError(
                "Saved pipeline has not been fitted."
            )

        return pipeline