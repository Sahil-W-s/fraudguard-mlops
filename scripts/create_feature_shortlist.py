from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "docs"

REPORT_PATH = DOCS_DIR / "combined_feature_selection_report.csv"
OUTPUT_PATH = DOCS_DIR / "selected_features.csv"

N_FEATURES = 80


def main():
    report = pd.read_csv(REPORT_PATH)

    report = report.sort_values("gain_rank").reset_index(drop=True)

    selected = report.head(N_FEATURES).copy()

    selected = selected[
        [
            "gain_rank",
            "feature",
            "gain_importance",
            "gain_importance_percentage",
            "missing_percentage",
            "unique_values",
            "dominant_value_ratio",
        ]
    ]

    selected.to_csv(OUTPUT_PATH, index=False)

    print(f"Selected features: {len(selected)}")
    print(f"Saved to: {OUTPUT_PATH}")

    print("\nSelected feature names:")
    print(selected["feature"].tolist())

    print(
        "\nCumulative gain coverage: "
        f"{selected['gain_importance_percentage'].sum():.2f}%"
    )


if __name__ == "__main__":
    main()