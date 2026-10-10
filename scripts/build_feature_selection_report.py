from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "docs"

IMPORTANCE_PATH = DOCS_DIR / "lightgbm_feature_importance.csv"
DISTRIBUTION_PATH = DOCS_DIR / "feature_distribution_profile.csv"
SELECTION_PATH = DOCS_DIR / "feature_selection_report.csv"

OUTPUT_PATH = DOCS_DIR / "combined_feature_selection_report.csv"


def main():
    importance = pd.read_csv(IMPORTANCE_PATH)
    distribution = pd.read_csv(DISTRIBUTION_PATH)
    selection = pd.read_csv(SELECTION_PATH)

    # Exclude time and identifiers from predictive candidates.
    excluded = {"TransactionDT", "TransactionID", "isFraud"}

    report = importance.merge(
        distribution,
        on="feature",
        how="left",
        suffixes=("", "_distribution"),
    )

    # Bring in missingness-based selection flags where available.
    selection_columns = [
        column
        for column in selection.columns
        if column == "feature"
        or column not in report.columns
    ]

    report = report.merge(
        selection[selection_columns],
        on="feature",
        how="left",
        suffixes=("", "_selection"),
    )

    report = report[~report["feature"].isin(excluded)].copy()

    report["cumulative_gain_percentage"] = (
        report["gain_importance_percentage"].cumsum()
    )

    report["near_constant"] = (
        report["dominant_value_ratio"] >= 0.995
    )

    report["high_missingness"] = (
        report["missing_percentage"] > 90
    )

    report = report.sort_values(
        "gain_rank"
    ).reset_index(drop=True)

    report.to_csv(OUTPUT_PATH, index=False)

    print(f"Report saved to: {OUTPUT_PATH}")
    print(f"Features evaluated: {len(report)}")

    print("\nTop 50 features with combined diagnostics:")

    display_columns = [
        "gain_rank",
        "feature",
        "gain_importance_percentage",
        "cumulative_gain_percentage",
        "missing_percentage",
        "unique_values",
        "dominant_value_ratio",
        "near_constant",
        "high_missingness",
    ]

    display_columns = [
        column
        for column in display_columns
        if column in report.columns
    ]

    print(
        report[display_columns]
        .head(50)
        .to_string(index=False)
    )

    print("\nCumulative gain coverage:")

    for n in [10, 20, 30, 50, 80, 100]:
        if len(report) >= n:
            coverage = report.iloc[n - 1][
                "cumulative_gain_percentage"
            ]
            print(f"Top {n:3} features: {coverage:.2f}%")

    print("\nHighly ranked near-constant features:")
    print(
        report.loc[
            (report["gain_rank"] <= 100)
            & report["near_constant"],
            ["gain_rank", "feature", "gain_importance_percentage"],
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()