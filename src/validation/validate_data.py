
from pathlib import Path
import logging

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRANSACTION_PATH = PROJECT_ROOT / "data" / "raw" / "train_transaction.csv"
IDENTITY_PATH = PROJECT_ROOT / "data" / "raw" / "train_identity.csv"

# Validation thresholds
MAX_NULL_RATE = 0.99
EXPECTED_FRAUD_RATE = 0.035
FRAUD_RATE_TOLERANCE = 0.01
CHUNK_SIZE = 25_000


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def validate_files(transaction_path, identity_path):
    """Check that the required datasets exist and are non-empty."""

    for path in (transaction_path, identity_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required dataset not found: {path}")

        if path.stat().st_size == 0:
            raise ValueError(f"Required dataset is empty: {path}")

    logger.info("Required dataset files found.")


def validate_schema(transaction_df, identity_df):
    """Validate expected columns and critical data types."""

    required_transaction_columns = {
        "TransactionID",
        "isFraud",
        "TransactionAmt",
        "TransactionDT",
    }
    required_identity_columns = {"TransactionID"}

    missing_transaction = required_transaction_columns - set(transaction_df.columns)
    missing_identity = required_identity_columns - set(identity_df.columns)

    if missing_transaction:
        raise ValueError(
            f"Missing transaction columns: {sorted(missing_transaction)}"
        )

    if missing_identity:
        raise ValueError(
            f"Missing identity columns: {sorted(missing_identity)}"
        )

    for column in ("TransactionAmt", "TransactionDT", "isFraud"):
        if not pd.api.types.is_numeric_dtype(transaction_df[column]):
            raise TypeError(f"{column} must be numeric.")

    logger.info("Schema validation passed.")


def _check_null_rates(df, dataset_name, totals, failures):
    """Accumulate per-column null counts without retaining previous chunks."""

    for column in df.columns:
        totals[dataset_name]["rows"] += 0  # Row count is accumulated separately.
        totals[dataset_name]["nulls"].setdefault(column, 0)
        totals[dataset_name]["nulls"][column] += int(df[column].isna().sum())


def _validate_dataset(
    path,
    dataset_name,
    required_columns,
    transaction=False,
    check_class_balance=False,
):
    """Validate a CSV in chunks, maintaining global counts and ID uniqueness."""

    logger.info("Validating %s dataset in chunks...", dataset_name)

    rows = 0
    null_counts = {}
    seen_ids = set()
    fraud_count = 0
    min_transaction_amount = None
    schema_checked = False

    for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE):
        missing = required_columns - set(chunk.columns)
        if missing:
            raise ValueError(
                f"Missing {dataset_name} columns: {sorted(missing)}"
            )

        if not schema_checked and transaction:
            validate_schema(
                chunk,
                pd.DataFrame(columns=["TransactionID"]),
            )
            schema_checked = True

        rows += len(chunk)

        for column in chunk.columns:
            null_counts[column] = (
                null_counts.get(column, 0)
                + int(chunk[column].isna().sum())
            )

        ids = chunk["TransactionID"]

        if ids.isna().any():
            raise ValueError(
                f"{dataset_name} contains missing TransactionIDs."
            )

        duplicates_in_chunk = ids.duplicated()
        if duplicates_in_chunk.any():
            raise ValueError(
                f"{dataset_name} contains duplicate TransactionIDs."
            )

        overlap = set(ids).intersection(seen_ids)
        if overlap:
            raise ValueError(
                f"{dataset_name} contains duplicate TransactionIDs "
                "across chunks."
            )

        seen_ids.update(ids.tolist())

        if transaction:
            amounts = chunk["TransactionAmt"]

            invalid_amounts = amounts <= 0
            if invalid_amounts.any():
                invalid_count = int(invalid_amounts.sum())
                raise ValueError(
                    f"TransactionAmt contains {invalid_count} "
                    "non-positive values."
                )

            labels = set(chunk["isFraud"].dropna().unique())
            if not labels.issubset({0, 1}):
                raise ValueError(
                    f"isFraud contains unexpected values: "
                    f"{sorted(labels - {0, 1})}"
                )

            fraud_count += int(chunk["isFraud"].sum())

    if rows == 0:
        raise ValueError(f"{dataset_name} dataset contains no rows.")

    null_rates = {
        column: count / rows
        for column, count in null_counts.items()
    }

    failures = [
        (column, rate)
        for column, rate in null_rates.items()
        if rate > MAX_NULL_RATE
    ]

    if failures:
        details = "\n".join(
            f"{dataset_name}.{column}: {rate * 100:.2f}% null"
            for column, rate in failures
        )
        raise ValueError(f"Null-rate validation failed:\n{details}")

    if transaction and check_class_balance:
        fraud_rate = fraud_count / rows
        lower = EXPECTED_FRAUD_RATE - FRAUD_RATE_TOLERANCE
        upper = EXPECTED_FRAUD_RATE + FRAUD_RATE_TOLERANCE

        logger.info("Observed fraud rate: %.4f%%", fraud_rate * 100)
        logger.info(
            "Expected range: %.2f%% - %.2f%%",
            lower * 100,
            upper * 100,
        )

        if not lower <= fraud_rate <= upper:
            raise ValueError(
                "Class balance validation failed. "
                f"Observed fraud rate: {fraud_rate:.4f}"
            )

    logger.info(
        "%s validation passed: %s rows, %s columns.",
        dataset_name.capitalize(),
        rows,
        len(null_counts),
    )

    return rows


def main(transaction_path=TRANSACTION_PATH):
    """Run data validation with bounded-memory CSV reads."""

    logger.info("Starting Phase 1 data validation.")

    validate_files(transaction_path, IDENTITY_PATH)

    transaction_required = {
        "TransactionID",
        "isFraud",
        "TransactionAmt",
        "TransactionDT",
    }

    identity_required = {"TransactionID"}

    transaction_rows = _validate_dataset(
        transaction_path,
        "transaction",
        transaction_required,
        transaction=True,
        check_class_balance=True,
    )

    identity_rows = _validate_dataset(
        IDENTITY_PATH,
        "identity",
        identity_required,
    )

    logger.info(
        "ALL DATA VALIDATION CHECKS PASSED. "
        "Transaction rows: %s; identity rows: %s.",
        transaction_rows,
        identity_rows,
    )


if __name__ == "__main__":
    import sys

    transaction_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else TRANSACTION_PATH
    )

    main(transaction_path)
