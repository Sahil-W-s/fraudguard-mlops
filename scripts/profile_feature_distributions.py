from pathlib import Path

import pandas as pd

from src.ingestion.load_data import (
    load_data,
    validate_join_key,
    join_data,
)

from src.features.build_features import (
    TARGET_COLUMN,
    ID_COLUMNS,
    TIME_COLUMN,
    identify_feature_columns,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_PATH = (
    PROJECT_ROOT
    / "docs"
    / "feature_distribution_profile.csv"
)


def dominant_value_ratio(series: pd.Series) -> float:
    """
    Return the proportion of non-null observations
    belonging to the most frequent value.
    """

    non_null = series.dropna()

    if len(non_null) == 0:
        return 1.0

    return float(
        non_null.value_counts(normalize=True).iloc[0]
    )


def profile_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Profile all predictive candidate features using
    the actual joined IEEE-CIS dataset.
    """

    feature_columns = [
    c for c in identify_feature_columns(df)
    if c != TIME_COLUMN
]

    rows = []

    for column in feature_columns:
        series = df[column]

        non_null = series.dropna()

        missing_percentage = (
            series.isna().mean() * 100
        )

        unique_values = (
            series.nunique(dropna=True)
        )

        dominant_ratio = dominant_value_ratio(
            series
        )

        if pd.api.types.is_numeric_dtype(series):
            variance = series.var()
        else:
            variance = None

        rows.append(
            {
                "feature": column,
                "dtype": str(series.dtype),
                "missing_percentage": missing_percentage,
                "unique_values": unique_values,
                "dominant_value_ratio": dominant_ratio,
                "variance": variance,
                "non_null_count": len(non_null),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "dominant_value_ratio",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def main():
    print("Loading IEEE-CIS datasets...")

    transaction_df, identity_df = load_data()

    print("Validating join key...")

    validate_join_key(
        transaction_df,
        identity_df,
    )

    print("Joining datasets...")

    joined_df = join_data(
        transaction_df,
        identity_df,
    )

    print(
        f"Joined dataset shape: {joined_df.shape}"
    )

    print("Profiling feature distributions...")

    profile = profile_features(
        joined_df
    )

    profile.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"\nProfile saved to:\n{OUTPUT_PATH}"
    )

    print("\nPotential near-constant features:")

    near_constant = profile[
        profile["dominant_value_ratio"] >= 0.995
    ]

    print(
        f"Count: {len(near_constant)}"
    )

    if not near_constant.empty:
        print(
            near_constant[
                [
                    "feature",
                    "dtype",
                    "missing_percentage",
                    "unique_values",
                    "dominant_value_ratio",
                ]
            ]
            .head(50)
            .to_string(index=False)
        )

    print("\nHighest dominant-value ratios:")

    print(
        profile[
            [
                "feature",
                "dtype",
                "missing_percentage",
                "unique_values",
                "dominant_value_ratio",
            ]
        ]
        .head(30)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()