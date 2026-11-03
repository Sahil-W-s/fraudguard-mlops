from pathlib import Path

from src.ingestion.load_data import (
    load_data,
    validate_join_key,
    join_data,
)
from src.features.build_features import TIME_COLUMN
from src.features.feature_pipeline import FeaturePipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = PROJECT_ROOT / "models" / "feature_pipeline.joblib"


def main():
    print("Loading IEEE-CIS datasets...")

    transaction_df, identity_df = load_data()
    validate_join_key(transaction_df, identity_df)
    df = join_data(transaction_df, identity_df)

    print(f"Joined dataset shape: {df.shape}")

    # Fit preprocessing only on the earliest 70% of transactions.
    df = df.sort_values(TIME_COLUMN).reset_index(drop=True)
    train_end = int(len(df) * 0.70)
    train_df = df.iloc[:train_end]

    print(f"Pipeline fitting rows: {len(train_df):,}")
    print(
        f"Training time range: "
        f"{train_df[TIME_COLUMN].min()} -> "
        f"{train_df[TIME_COLUMN].max()}"
    )

    pipeline = FeaturePipeline()
    X_train = pipeline.fit_transform(train_df)

    pipeline.save(ARTIFACT_PATH)

    print(f"\nPipeline saved to: {ARTIFACT_PATH}")
    print(f"Selected features: {len(pipeline.feature_names)}")
    print(f"Categorical features: {len(pipeline.categorical_columns_)}")
    print(f"Numeric features: {len(pipeline.numeric_columns_)}")
    print(f"Transformed training shape: {X_train.shape}")

    # Verify that the saved artifact can be loaded.
    loaded_pipeline = FeaturePipeline.load(ARTIFACT_PATH)
    X_reloaded = loaded_pipeline.transform(train_df)

    assert X_reloaded.columns.tolist() == X_train.columns.tolist()
    assert X_reloaded.shape == X_train.shape

    print("Artifact reload verification passed.")


if __name__ == "__main__":
    main()