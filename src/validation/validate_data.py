from pathlib import Path
import logging
import sys
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRANSACTION_PATH = PROJECT_ROOT / "data" / "raw" / "train_transaction.csv"
IDENTITY_PATH = PROJECT_ROOT / "data" / "raw" / "train_identity.csv"


# Validation thresholds
MAX_NULL_RATE = 0.99
EXPECTED_FRAUD_RATE = 0.035
FRAUD_RATE_TOLERANCE = 0.01


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def validate_files(transaction_path, identity_path):
    """Check that the required datasets exist."""

    if not transaction_path.exists():
        raise FileNotFoundError(
            f"Transaction dataset not found: {transaction_path}"
        )

    if not identity_path.exists():
        raise FileNotFoundError(
            f"Identity dataset not found: {identity_path}"
        )

    logger.info("Required dataset files found.")


def validate_schema(transaction_df, identity_df):
    """Validate expected columns and critical data types."""

    logger.info("Validating schema...")

    required_transaction_columns = {
        "TransactionID",
        "isFraud",
        "TransactionAmt",
        "TransactionDT",
    }

    required_identity_columns = {
        "TransactionID",
    }

    missing_transaction = (
        required_transaction_columns
        - set(transaction_df.columns)
    )

    missing_identity = (
        required_identity_columns
        - set(identity_df.columns)
    )

    if missing_transaction:
        raise ValueError(
            "Missing transaction columns: "
            f"{sorted(missing_transaction)}"
        )

    if missing_identity:
        raise ValueError(
            "Missing identity columns: "
            f"{sorted(missing_identity)}"
        )

    if not pd.api.types.is_numeric_dtype(
        transaction_df["TransactionAmt"]
    ):
        raise TypeError(
            "TransactionAmt must be numeric."
        )

    if not pd.api.types.is_numeric_dtype(
        transaction_df["TransactionDT"]
    ):
        raise TypeError(
            "TransactionDT must be numeric."
        )

    if not pd.api.types.is_numeric_dtype(
        transaction_df["isFraud"]
    ):
        raise TypeError(
            "isFraud must be numeric."
        )

    logger.info("Schema validation passed.")


def validate_ranges(transaction_df):
    """Validate important numerical ranges."""

    logger.info("Validating value ranges...")

    if (transaction_df["TransactionAmt"] <= 0).any():
        invalid_count = int(
            (transaction_df["TransactionAmt"] <= 0).sum()
        )

        raise ValueError(
            f"TransactionAmt contains {invalid_count} "
            "non-positive values."
        )

    valid_labels = {0, 1}

    actual_labels = set(
        transaction_df["isFraud"].dropna().unique()
    )

    if not actual_labels.issubset(valid_labels):
        raise ValueError(
            f"isFraud contains unexpected values: "
            f"{sorted(actual_labels - valid_labels)}"
        )

    logger.info("Range validation passed.")


def validate_null_rates(transaction_df, identity_df):
    """Validate that no column exceeds the absolute null-rate limit."""

    logger.info("Validating null rates...")

    datasets = {
        "transaction": transaction_df,
        "identity": identity_df,
    }

    failures = []

    for dataset_name, df in datasets.items():

        null_rates = df.isna().mean()

        for column, null_rate in null_rates.items():

            if null_rate > MAX_NULL_RATE:
                failures.append(
                    (
                        dataset_name,
                        column,
                        null_rate * 100,
                    )
                )

    if failures:

        formatted_failures = "\n".join(
            f"{dataset}.{column}: {rate:.2f}% null"
            for dataset, column, rate in failures
        )

        raise ValueError(
            "Null-rate validation failed:\n"
            + formatted_failures
        )

    logger.info(
        "Null-rate validation passed. "
        "Maximum allowed null rate: %.2f%%",
        MAX_NULL_RATE * 100,
    )


def validate_class_balance(transaction_df):
    """Validate that the fraud rate has not shifted drastically."""

    logger.info("Validating class balance...")

    fraud_rate = transaction_df["isFraud"].mean()

    lower_bound = EXPECTED_FRAUD_RATE - FRAUD_RATE_TOLERANCE
    upper_bound = EXPECTED_FRAUD_RATE + FRAUD_RATE_TOLERANCE

    logger.info(
        "Observed fraud rate: %.4f%%",
        fraud_rate * 100,
    )

    logger.info(
        "Expected range: %.2f%% - %.2f%%",
        lower_bound * 100,
        upper_bound * 100,
    )

    if not lower_bound <= fraud_rate <= upper_bound:
        raise ValueError(
            "Class balance validation failed. "
            f"Observed fraud rate: {fraud_rate:.4f}"
        )

    logger.info("Class balance validation passed.")


def validate_duplicate_ids(transaction_df, identity_df):
    """Ensure TransactionID is unique in both tables."""

    logger.info("Validating TransactionID uniqueness...")

    transaction_duplicates = (
        transaction_df["TransactionID"].duplicated().sum()
    )

    identity_duplicates = (
        identity_df["TransactionID"].duplicated().sum()
    )

    if transaction_duplicates > 0:
        raise ValueError(
            f"Transaction table contains "
            f"{transaction_duplicates} duplicate TransactionIDs."
        )

    if identity_duplicates > 0:
        raise ValueError(
            f"Identity table contains "
            f"{identity_duplicates} duplicate TransactionIDs."
        )

    logger.info("TransactionID uniqueness validation passed.")


def main(transaction_path=TRANSACTION_PATH):
    """Run all Phase 1 data validation checks."""

    logger.info("Starting Phase 1 data validation.")

    validate_files(
        transaction_path,
        IDENTITY_PATH,
    )

    logger.info("Loading datasets...")

    transaction_df = pd.read_csv(
        transaction_path
    )

    identity_df = pd.read_csv(
        IDENTITY_PATH
    )

    logger.info(
        "Transaction dataset: %s rows x %s columns",
        transaction_df.shape[0],
        transaction_df.shape[1],
    )

    logger.info(
        "Identity dataset: %s rows x %s columns",
        identity_df.shape[0],
        identity_df.shape[1],
    )

    validate_schema(
        transaction_df,
        identity_df,
    )

    validate_ranges(
        transaction_df
    )

    validate_null_rates(
        transaction_df,
        identity_df,
    )

    validate_class_balance(
        transaction_df
    )

    validate_duplicate_ids(
        transaction_df,
        identity_df,
    )

    logger.info(
        "ALL DATA VALIDATION CHECKS PASSED."
    )


if __name__ == "__main__":
    import sys

    transaction_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else TRANSACTION_PATH
    )

    main(transaction_path)