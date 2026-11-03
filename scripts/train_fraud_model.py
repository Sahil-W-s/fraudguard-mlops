import json
from pathlib import Path

import joblib
import lightgbm as lgb
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
)
from src.features.feature_pipeline import FeaturePipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"

PIPELINE_PATH = MODELS_DIR / "feature_pipeline.joblib"
MODEL_PATH = MODELS_DIR / "fraud_model.joblib"
METRICS_PATH = DOCS_DIR / "baseline_model_metrics.json"


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading IEEE-CIS datasets...")

    transaction_df, identity_df = load_data()
    validate_join_key(transaction_df, identity_df)
    df = join_data(transaction_df, identity_df)

    df = df.sort_values(TIME_COLUMN).reset_index(drop=True)

    # Chronological 70% / 15% / 15% split.
    n = len(df)
    train_end = int(n * 0.70)
    valid_end = int(n * 0.85)

    train_df = df.iloc[:train_end].copy()
    valid_df = df.iloc[train_end:valid_end].copy()
    test_df = df.iloc[valid_end:].copy()

    print(f"Training rows:   {len(train_df):,}")
    print(f"Validation rows: {len(valid_df):,}")
    print(f"Test rows:       {len(test_df):,}")

    y_train = train_df[TARGET_COLUMN]
    y_valid = valid_df[TARGET_COLUMN]
    y_test = test_df[TARGET_COLUMN]

    # Fit preprocessing exclusively on training data.
    pipeline = FeaturePipeline()
    X_train = pipeline.fit_transform(train_df)
    X_valid = pipeline.transform(valid_df)
    X_test = pipeline.transform(test_df)

    print(f"Features: {X_train.shape[1]}")
    print(f"Categorical features: {len(pipeline.categorical_columns_)}")

    print("\nTraining LightGBM classifier...")

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=3000,
        learning_rate=0.05,
        num_leaves=31,
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
        categorical_feature=pipeline.categorical_columns_,
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=50,
                verbose=True,
            )
        ],
    )

    # Evaluate once on the later test period.
    probabilities = model.predict_proba(
        X_test,
        num_iteration=model.best_iteration_,
    )[:, 1]

    metrics = {
        "model": "LightGBM",
        "feature_count": int(X_train.shape[1]),
        "training_rows": int(len(train_df)),
        "validation_rows": int(len(valid_df)),
        "test_rows": int(len(test_df)),
        "best_iteration": int(model.best_iteration_),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "average_precision": float(
            average_precision_score(y_test, probabilities)
        ),
        "log_loss": float(log_loss(y_test, probabilities)),
        "feature_names": pipeline.feature_names,
    }

    # Save the fitted preprocessing and classifier together as artifacts.
    pipeline.save(PIPELINE_PATH)
    joblib.dump(model, MODEL_PATH)

    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print("\nSaved pipeline:", PIPELINE_PATH)
    print("Saved model:", MODEL_PATH)
    print("Saved metrics:", METRICS_PATH)

    print("\nChronological test metrics:")
    for metric in (
        "feature_count",
        "best_iteration",
        "roc_auc",
        "average_precision",
        "log_loss",
    ):
        print(f"{metric}: {metrics[metric]}")

    # Verify both artifacts can be loaded.
    loaded_pipeline = FeaturePipeline.load(PIPELINE_PATH)
    loaded_model = joblib.load(MODEL_PATH)

    reloaded_test = loaded_pipeline.transform(test_df)
    reloaded_probabilities = loaded_model.predict_proba(
        reloaded_test,
        num_iteration=loaded_model.best_iteration_,
    )[:, 1]

    assert reloaded_test.columns.tolist() == X_test.columns.tolist()
    assert len(reloaded_probabilities) == len(y_test)
    assert (0 <= reloaded_probabilities).all()
    assert (reloaded_probabilities <= 1).all()

    print("\nArtifact reload verification passed.")


if __name__ == "__main__":
    main()