from pathlib import Path

from src.ingestion.load_data import (
    load_data,
    validate_join_key,
    join_data,
)

from src.features.build_features import analyze_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_PATH = (
    PROJECT_ROOT
    / "docs"
    / "feature_inventory.csv"
)


def main():
    print("Loading IEEE-CIS datasets...")

    transaction_df, identity_df = load_data()

    print("Validating join key...")
    validate_join_key(
        transaction_df,
        identity_df,
    )

    print("Joining transaction and identity data...")
    joined_df = join_data(
        transaction_df,
        identity_df,
    )

    print(
        f"Joined dataset shape: {joined_df.shape}"
    )

    print("Building feature inventory...")

    inventory = analyze_features(
        joined_df,
        output_path=OUTPUT_PATH,
    )

    print(
        f"Feature inventory saved to:\n"
        f"{OUTPUT_PATH}"
    )

    print("\nFeature role summary:")
    print(
        inventory["role"]
        .value_counts()
        .to_string()
    )

    print("\nTop 20 features by missingness:")
    print(
        inventory[
            [
                "feature",
                "role",
                "dtype",
                "missing_percentage",
                "unique_values",
            ]
        ]
        .sort_values(
            "missing_percentage",
            ascending=False,
        )
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()