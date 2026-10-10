from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INVENTORY_PATH = (
    PROJECT_ROOT
    / "docs"
    / "feature_inventory.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "docs"
    / "feature_selection_report.csv"
)


def build_selection_report(inventory: pd.DataFrame) -> pd.DataFrame:
    """
    Build an evidence-based feature-selection analysis.

    This stage does not remove features. It only identifies
    potential filtering candidates for Phase 2.2.
    """

    report = inventory.copy()

    # Predictive candidates only.
    # Target, identifier and time columns are handled separately.
    report["candidate_feature"] = (
        ~report["role"].isin(
            ["target", "identifier", "time"]
        )
    )

    report["missing_gt_90"] = (
        report["missing_percentage"] > 90.0
    )

    report["missing_gt_95"] = (
        report["missing_percentage"] > 95.0
    )

    # A feature with <= 1 unique non-null value carries
    # essentially no usable variation.
    report["constant_or_single_value"] = (
        report["unique_values"] <= 1
    )

    # Low-cardinality indicator.
    # This is descriptive only; we do NOT automatically remove
    # these features because categorical fraud signals can be useful.
    report["low_cardinality"] = (
        report["unique_values"] <= 5
    )

    # A deliberately conservative near-constant flag.
    # With only inventory information available at this stage,
    # we cannot calculate the dominant-value percentage, so
    # this remains False until raw-data analysis is performed.
    report["near_constant"] = False

    return report


def print_summary(report: pd.DataFrame) -> None:
    candidates = report[
        report["candidate_feature"]
    ]

    print("\n" + "=" * 70)
    print("PHASE 2.2.1 — FEATURE SELECTION ANALYSIS")
    print("=" * 70)

    print(f"\nTotal columns: {len(report)}")
    print(f"Candidate features: {len(candidates)}")

    print("\nFeature roles:")
    print(
        report["role"]
        .value_counts()
        .to_string()
    )

    print("\nMissingness:")
    print(
        f"Candidate features >90% missing: "
        f"{candidates['missing_gt_90'].sum()}"
    )

    print(
        f"Candidate features >95% missing: "
        f"{candidates['missing_gt_95'].sum()}"
    )

    print(
        f"Candidate features <=1 unique value: "
        f"{candidates['constant_or_single_value'].sum()}"
    )

    print(
        f"Candidate features <=5 unique values: "
        f"{candidates['low_cardinality'].sum()}"
    )

    print("\nCandidate features by type:")

    print(
        candidates["role"]
        .value_counts()
        .to_string()
    )

    print("\nTop 30 candidate features by missingness:")

    columns = [
        "feature",
        "role",
        "dtype",
        "missing_percentage",
        "unique_values",
        "missing_gt_90",
        "missing_gt_95",
        "constant_or_single_value",
        "low_cardinality",
    ]

    print(
        candidates[columns]
        .sort_values(
            "missing_percentage",
            ascending=False,
        )
        .head(30)
        .to_string(index=False)
    )


def main():
    if not INVENTORY_PATH.exists():
        raise FileNotFoundError(
            f"Feature inventory not found: {INVENTORY_PATH}"
        )

    print(
        f"Loading feature inventory:\n"
        f"{INVENTORY_PATH}"
    )

    inventory = pd.read_csv(
        INVENTORY_PATH
    )

    report = build_selection_report(
        inventory
    )

    report.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"\nFeature-selection report saved to:\n"
        f"{OUTPUT_PATH}"
    )

    print_summary(report)


if __name__ == "__main__":
    main()