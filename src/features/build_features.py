"""
FraudGuard feature engineering utilities.

Phase 2.1:
- Identify target, identifier, and temporal columns
- Classify numerical and categorical features
- Generate a feature inventory
- Provide deterministic feature-selection helpers

Important:
This module intentionally does NOT perform final feature selection yet.
That will happen in Phase 2.2.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


# ---------------------------------------------------------------------------
# Dataset contract
# ---------------------------------------------------------------------------

TARGET_COLUMN = "isFraud"
ID_COLUMNS = ["TransactionID"]
TIME_COLUMN = "TransactionDT"


# ---------------------------------------------------------------------------
# Column identification
# ---------------------------------------------------------------------------

def validate_required_columns(
    df: pd.DataFrame,
    required_columns: Iterable[str] | None = None,
) -> None:
    """
    Validate that required columns exist in the dataframe.

    Parameters
    ----------
    df:
        Input dataframe.
    required_columns:
        Columns that must be present.

    Raises
    ------
    ValueError
        If one or more required columns are missing.
    """

    if required_columns is None:
        required_columns = [
            TARGET_COLUMN,
            *ID_COLUMNS,
            TIME_COLUMN,
        ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )


def identify_feature_columns(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    id_columns: Iterable[str] = ID_COLUMNS,
) -> list[str]:
    """
    Return columns eligible for feature engineering.

    Target and identifier columns are excluded.

    Parameters
    ----------
    df:
        Input dataframe.
    target_column:
        Target/label column.
    id_columns:
        Identifier columns.

    Returns
    -------
    list[str]
        Candidate feature columns.
    """

    excluded = {
        target_column,
        *id_columns,
    }

    return [
        column
        for column in df.columns
        if column not in excluded
    ]


# ---------------------------------------------------------------------------
# Feature type identification
# ---------------------------------------------------------------------------

def identify_numeric_features(
    df: pd.DataFrame,
    feature_columns: Iterable[str],
) -> list[str]:
    """
    Identify numerical feature columns.
    """

    feature_df = df[list(feature_columns)]

    return feature_df.select_dtypes(
        include=["number"]
    ).columns.tolist()


def identify_categorical_features(
    df: pd.DataFrame,
    feature_columns: Iterable[str],
) -> list[str]:
    """
    Identify categorical/object feature columns.
    """

    feature_df = df[list(feature_columns)]

    return feature_df.select_dtypes(
        include=["object", "str", "category", "bool"]
    ).columns.tolist()


# ---------------------------------------------------------------------------
# Missingness analysis
# ---------------------------------------------------------------------------

def calculate_missingness(
    df: pd.DataFrame,
    feature_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    Calculate missing-value statistics for features.

    Returns a dataframe containing:
    - column name
    - dtype
    - number of missing values
    - missing percentage
    - number of unique values
    """

    if feature_columns is None:
        feature_columns = identify_feature_columns(df)

    rows = []

    for column in feature_columns:
        missing_count = int(df[column].isna().sum())
        total_count = len(df)

        missing_percentage = (
            missing_count / total_count * 100
            if total_count > 0
            else 0.0
        )

        rows.append(
            {
                "feature": column,
                "dtype": str(df[column].dtype),
                "missing_count": missing_count,
                "missing_percentage": missing_percentage,
                "unique_values": int(df[column].nunique(dropna=True)),
            }
        )

    result = pd.DataFrame(rows)

    if not result.empty:
        result = result.sort_values(
            "missing_percentage",
            ascending=False,
        ).reset_index(drop=True)

    return result


# ---------------------------------------------------------------------------
# Feature inventory
# ---------------------------------------------------------------------------

def build_feature_inventory(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a complete feature inventory.

    Each column is classified as:
    - target
    - identifier
    - time
    - numerical
    - categorical
    - other
    """

    validate_required_columns(df)

    feature_columns = identify_feature_columns(df)

    numeric_columns = set(
        identify_numeric_features(
            df,
            feature_columns,
        )
    )

    categorical_columns = set(
        identify_categorical_features(
            df,
            feature_columns,
        )
    )

    missingness = calculate_missingness(
        df,
        feature_columns,
    )

    missingness_lookup = (
        missingness.set_index("feature")
        if not missingness.empty
        else pd.DataFrame()
    )

    rows = []

    for column in df.columns:

        if column == TARGET_COLUMN:
            role = "target"

        elif column in ID_COLUMNS:
            role = "identifier"

        elif column == TIME_COLUMN:
            role = "time"

        elif column in numeric_columns:
            role = "numerical"

        elif column in categorical_columns:
            role = "categorical"

        else:
            role = "other"

        if column in missingness_lookup.index:
            missing_count = int(
                missingness_lookup.loc[
                    column,
                    "missing_count",
                ]
            )

            missing_percentage = float(
                missingness_lookup.loc[
                    column,
                    "missing_percentage",
                ]
            )

            unique_values = int(
                missingness_lookup.loc[
                    column,
                    "unique_values",
                ]
            )

        else:
            missing_count = int(df[column].isna().sum())

            missing_percentage = (
                missing_count / len(df) * 100
                if len(df) > 0
                else 0.0
            )

            unique_values = int(
                df[column].nunique(dropna=True)
            )

        rows.append(
            {
                "feature": column,
                "dtype": str(df[column].dtype),
                "role": role,
                "missing_count": missing_count,
                "missing_percentage": missing_percentage,
                "unique_values": unique_values,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Candidate feature filtering
# ---------------------------------------------------------------------------

def get_candidate_features(
    inventory: pd.DataFrame,
) -> list[str]:
    """
    Return columns that are currently eligible for feature selection.

    Target, identifiers, and time column are excluded from the initial
    predictive feature candidate set.

    Note:
    TransactionDT is deliberately excluded here because it will be handled
    separately for chronological splitting.
    """

    excluded_roles = {
        "target",
        "identifier",
        "time",
    }

    return inventory.loc[
        ~inventory["role"].isin(excluded_roles),
        "feature",
    ].tolist()


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_feature_inventory(
    inventory: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Save feature inventory as CSV.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    inventory.to_csv(
        output_path,
        index=False,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def analyze_features(
    df: pd.DataFrame,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Analyze the dataframe and optionally save its feature inventory.

    This function is intentionally deterministic.
    """

    inventory = build_feature_inventory(df)

    if output_path is not None:
        save_feature_inventory(
            inventory,
            output_path,
        )

    return inventory