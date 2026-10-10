from pathlib import Path

import lightgbm as lgb
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
OUTPUT_PATH = PROJECT_ROOT / "docs" / "lightgbm_feature_importance.csv"


def prepare_features(df: pd.DataFrame):
    """
    Prepare candidate features for LightGBM.

    Excludes the target, identifier, and TransactionDT.
    Converts all string-like columns to pandas categorical dtype.
    """

    feature_columns = [
        column
        for column in identify_feature_columns(df)
        if column != TIME_COLUMN
    ]

    X = df[feature_columns].copy()
    y = df[TARGET_COLUMN].copy()

    categorical_features = []

    for column in X.columns:
        if (
            pd.api.types.is_object_dtype(X[column].dtype)
            or pd.api.types.is_string_dtype(X[column].dtype)
            or isinstance(X[column].dtype, pd.CategoricalDtype)
        ):
            X[column] = X[column].astype("category")
            categorical_features.append(column)

    # Fail early if any unsupported data types remain.
    invalid_columns = [
        column
        for column in X.columns
        if not (
            pd.api.types.is_numeric_dtype(X[column].dtype)
            or pd.api.types.is_bool_dtype(X[column].dtype)
            or isinstance(X[column].dtype, pd.CategoricalDtype)
        )
    ]

    if invalid_columns:
        raise TypeError(
            f"Unsupported feature dtypes: {invalid_columns}"
        )

    return X, y, categorical_features


def chronological_split(df: pd.DataFrame, train_fraction: float = 0.80):
    """
    Split the dataset chronologically using TransactionDT.
    """

    sorted_df = df.sort_values(TIME_COLUMN).reset_index(drop=True)

    split_index = int(len(sorted_df) * train_fraction)

    train_df = sorted_df.iloc[:split_index].copy()
    validation_df = sorted_df.iloc[split_index:].copy()

    return train_df, validation_df


def train_model(
    X_train,
    y_train,
    X_valid,
    y_valid,
    categorical_features,
):
    """
    Train a lightweight LightGBM model for feature importance.
    """

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="auc",
        categorical_feature=categorical_features,
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=30,
                verbose=True,
            )
        ],
    )

    return model


def build_importance_report(model, feature_columns):
    """
    Create a feature importance report using both:
    - split importance
    - gain importance
    """

    importance_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "split_importance": model.booster_.feature_importance(
                importance_type="split"
            ),
            "gain_importance": model.booster_.feature_importance(
                importance_type="gain"
            ),
        }
    )

    importance_df["gain_importance_percentage"] = (
        importance_df["gain_importance"]
        / importance_df["gain_importance"].sum()
        * 100
    )

    importance_df["split_importance_percentage"] = (
        importance_df["split_importance"]
        / importance_df["split_importance"].sum()
        * 100
    )

    importance_df = importance_df.sort_values(
        "gain_importance",
        ascending=False,
    ).reset_index(drop=True)

    importance_df["gain_rank"] = (
        importance_df.index + 1
    )

    return importance_df


def main():
    print("Loading IEEE-CIS datasets...")

    transaction_df, identity_df = load_data()

    print("Validating join key...")
    validate_join_key(transaction_df, identity_df)

    print("Joining datasets...")
    df = join_data(transaction_df, identity_df)

    print(f"Joined dataset shape: {df.shape}")

    print("\nCreating chronological train/validation split...")

    train_df, validation_df = chronological_split(df)

    print(f"Training rows:   {len(train_df):,}")
    print(f"Validation rows: {len(validation_df):,}")

    print(
        f"Training TransactionDT range: "
        f"{train_df[TIME_COLUMN].min()} -> "
        f"{train_df[TIME_COLUMN].max()}"
    )

    print(
        f"Validation TransactionDT range: "
        f"{validation_df[TIME_COLUMN].min()} -> "
        f"{validation_df[TIME_COLUMN].max()}"
    )

    print("\nPreparing LightGBM features...")

    X_train, y_train, categorical_features = prepare_features(
        train_df
    )

    X_valid, y_valid, _ = prepare_features(
        validation_df
    )

    print(f"Candidate features: {X_train.shape[1]}")
    print(
        f"Categorical features: "
        f"{len(categorical_features)}"
    )

    print("\nTraining LightGBM...")

    model = train_model(
        X_train,
        y_train,
        X_valid,
        y_valid,
        categorical_features,
    )

    print(
        f"\nBest iteration: "
        f"{model.best_iteration_}"
    )

    print("\nGenerating feature importance report...")

    importance_df = build_importance_report(
        model,
        X_train.columns.tolist(),
    )

    importance_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\nFeature importance report saved to:")
    print(OUTPUT_PATH)

    print("\nTop 30 features by gain importance:")

    display_columns = [
        "gain_rank",
        "feature",
        "gain_importance",
        "gain_importance_percentage",
        "split_importance",
    ]

    print(
        importance_df[display_columns]
        .head(30)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()