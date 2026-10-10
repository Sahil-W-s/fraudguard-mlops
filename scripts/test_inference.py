from src.ingestion.load_data import (
    load_data,
    validate_join_key,
    join_data,
)
from src.features.build_features import TIME_COLUMN
from src.inference.predict import FraudPredictor


def main():
    print("Loading inference artifacts...")
    predictor = FraudPredictor()

    print("Loading sample transactions...")
    transaction_df, identity_df = load_data()
    validate_join_key(transaction_df, identity_df)
    df = join_data(transaction_df, identity_df)

    df = df.sort_values(TIME_COLUMN).reset_index(drop=True)

    # Use five later transactions as an inference smoke test.
    sample = df.tail(5).copy()

    predictions = predictor.predict(sample)

    print("\nSample predictions:")
    print(predictions.to_string())

    assert len(predictions) == 5
    assert predictions["fraud_probability"].between(0, 1).all()

    print("\nInference smoke test passed.")


if __name__ == "__main__":
    main()