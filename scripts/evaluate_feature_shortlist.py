from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    log_loss,
    roc_auc_score,
)

from src.ingestion.load_data import (
    load_data,
    validate_join_key,
    join_data,
)
from src.features.build_features import (
    TARGET_COLUMN,
    TIME_COLUMN,
    identify_feature_columns,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "docs"

IMPORTANCE_OUTPUT = DOCS_DIR / "validated_feature_importance.csv"
SHORTLIST_OUTPUT = DOCS_DIR / "validated_selected_features.csv"
METRICS_OUTPUT = DOCS_DIR / "feature_selection_evaluation.csv"

N_FEATURES = 80
RANDOM_STATE = 42


def split_chronologically(df):
    df = df.sort_values(TIME_COLUMN).reset_index(drop=True)

    n = len(df)
    train_end = int(n * 0.70)
    valid_end = int(n * 0.85)

    train = df.iloc[:train_end].copy()
    valid = df.iloc[train_end:valid_end].copy()
    test = df.iloc[valid_end:].copy()

    print(f"Training rows:   {len(train):,}")
    print(f"Validation rows: {len(valid):,}")
    print(f"Test rows:       {len(test):,}")

    print(
        "Time boundaries:",
        f"train max={train[TIME_COLUMN].max()},",
        f"validation={valid[TIME_COLUMN].min()}–"
        f"{valid[TIME_COLUMN].max()},",
        f"test min={test[TIME_COLUMN].min()}",
    )

    return train, valid, test


def prepare_frames(train, valid, test, feature_columns):
    X_train = train[feature_columns].copy()
    X_valid = valid[feature_columns].copy()
    X_test = test[feature_columns].copy()

    categorical_features = []

    for column in feature_columns:
        dtype = X_train[column].dtype

        is_categorical = (
            pd.api.types.is_object_dtype(dtype)
            or pd.api.types.is_string_dtype(dtype)
            or isinstance(dtype, pd.CategoricalDtype)
        )

        if is_categorical:
            X_train[column] = X_train[column].astype("category")
            categories = X_train[column].cat.categories

            for X in (X_valid, X_test):
                X[column] = (
                    X[column]
                    .astype("category")
                    .cat.set_categories(categories)
                )

            categorical_features.append(column)

    return X_train, X_valid, X_test, categorical_features


def make_model():
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )


def fit_model(model, X_train, y_train, X_valid, y_valid, categorical):
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="auc",
        categorical_feature=categorical,
        callbacks=[
            lgb.early_stopping(30, verbose=True),
        ],
    )
    return model


def evaluate_model(name, model, X_test, y_test, feature_count):
    probabilities = model.predict_proba(
        X_test,
        num_iteration=model.best_iteration_,
    )[:, 1]

    return {
        "model": name,
        "feature_count": feature_count,
        "best_iteration": model.best_iteration_,
        "roc_auc": roc_auc_score(y_test, probabilities),
        "average_precision": average_precision_score(
            y_test, probabilities
        ),
        "log_loss": log_loss(y_test, probabilities),
    }


def main():
    print("Loading IEEE-CIS datasets...")

    transaction_df, identity_df = load_data()
    validate_join_key(transaction_df, identity_df)
    df = join_data(transaction_df, identity_df)

    train, valid, test = split_chronologically(df)

    y_train = train[TARGET_COLUMN]
    y_valid = valid[TARGET_COLUMN]
    y_test = test[TARGET_COLUMN]

    feature_columns = [
        column
        for column in identify_feature_columns(df)
        if column != TIME_COLUMN
    ]

    print(f"\nFull feature count: {len(feature_columns)}")

    (
        X_train,
        X_valid,
        X_test,
        categorical,
    ) = prepare_frames(
        train,
        valid,
        test,
        feature_columns,
    )

    print(f"Categorical feature count: {len(categorical)}")

    # Train the full model and derive feature importance.
    print("\nTraining full-feature model...")

    full_model = fit_model(
        make_model(),
        X_train,
        y_train,
        X_valid,
        y_valid,
        categorical,
    )

    importance = pd.DataFrame({
        "feature": feature_columns,
        "gain_importance": (
            full_model.booster_.feature_importance(
                importance_type="gain"
            )
        ),
        "split_importance": (
            full_model.booster_.feature_importance(
                importance_type="split"
            )
        ),
    }).sort_values(
        "gain_importance",
        ascending=False,
    ).reset_index(drop=True)

    importance["gain_rank"] = importance.index + 1
    importance.to_csv(IMPORTANCE_OUTPUT, index=False)

    selected_features = importance.head(N_FEATURES)["feature"].tolist()

    pd.DataFrame({
        "gain_rank": range(1, len(selected_features) + 1),
        "feature": selected_features,
    }).to_csv(SHORTLIST_OUTPUT, index=False)

    print(f"\nSaved importance report: {IMPORTANCE_OUTPUT}")
    print(f"Saved fresh shortlist: {SHORTLIST_OUTPUT}")
    print(f"Selected {len(selected_features)} features")

    # Evaluate the full model on the untouched test period.
    results = [
        evaluate_model(
            "full",
            full_model,
            X_test,
            y_test,
            len(feature_columns),
        )
    ]

    # Train and evaluate the reduced model with the same time split.
    print("\nTraining reduced-feature model...")

    X_train_reduced = X_train[selected_features]
    X_valid_reduced = X_valid[selected_features]
    X_test_reduced = X_test[selected_features]

    reduced_categorical = [
        column
        for column in categorical
        if column in selected_features
    ]

    reduced_model = fit_model(
        make_model(),
        X_train_reduced,
        y_train,
        X_valid_reduced,
        y_valid,
        reduced_categorical,
    )

    results.append(
        evaluate_model(
            "top_80",
            reduced_model,
            X_test_reduced,
            y_test,
            len(selected_features),
        )
    )

    metrics = pd.DataFrame(results)
    metrics.to_csv(METRICS_OUTPUT, index=False)

    print(f"\nSaved evaluation metrics: {METRICS_OUTPUT}")
    print("\nFinal chronological test results:")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()