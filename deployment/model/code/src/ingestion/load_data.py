from pathlib import Path
import logging

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRANSACTION_PATH = PROJECT_ROOT / "data" / "raw" / "train_transaction.csv"
IDENTITY_PATH = PROJECT_ROOT / "data" / "raw" / "train_identity.csv"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def load_data():
    """Load the IEEE-CIS transaction and identity datasets."""

    logger.info("Loading transaction data...")
    transaction_df = pd.read_csv(TRANSACTION_PATH)

    logger.info(
        "Transaction rows: %s | columns: %s",
        len(transaction_df),
        len(transaction_df.columns),
    )

    logger.info("Loading identity data...")
    identity_df = pd.read_csv(IDENTITY_PATH)

    logger.info(
        "Identity rows: %s | columns: %s",
        len(identity_df),
        len(identity_df.columns),
    )

    return transaction_df, identity_df


def validate_join_key(transaction_df, identity_df):
    """Validate the TransactionID columns before joining."""

    if "TransactionID" not in transaction_df.columns:
        raise ValueError(
            "TransactionID missing from transaction dataset."
        )

    if "TransactionID" not in identity_df.columns:
        raise ValueError(
            "TransactionID missing from identity dataset."
        )

    transaction_duplicates = transaction_df["TransactionID"].duplicated().sum()
    identity_duplicates = identity_df["TransactionID"].duplicated().sum()

    logger.info(
        "Duplicate TransactionIDs in transaction table: %s",
        transaction_duplicates,
    )

    logger.info(
        "Duplicate TransactionIDs in identity table: %s",
        identity_duplicates,
    )

    if transaction_duplicates > 0:
        raise ValueError(
            "Transaction table contains duplicate TransactionIDs."
        )

    if identity_duplicates > 0:
        raise ValueError(
            "Identity table contains duplicate TransactionIDs."
        )


def join_data(transaction_df, identity_df):
    """Perform a left join using TransactionID."""

    logger.info("Joining transaction and identity data...")

    transaction_rows_before = len(transaction_df)

    joined_df = transaction_df.merge(
        identity_df,
        on="TransactionID",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    transaction_rows_after = len(joined_df)

    matched_transactions = int(
        (joined_df["_merge"] == "both").sum()
    )

    identityless_transactions = int(
        (joined_df["_merge"] == "left_only").sum()
    )

    joined_df.drop(columns=["_merge"], inplace=True)

    logger.info(
        "Transaction rows before join: %s",
        transaction_rows_before,
    )

    logger.info(
        "Rows after left join: %s",
        transaction_rows_after,
    )

    logger.info(
        "Transactions with identity data: %s",
        matched_transactions,
    )

    logger.info(
        "Transactions without identity data: %s",
        identityless_transactions,
    )

    identityless_percentage = (
        identityless_transactions
        / transaction_rows_before
        * 100
    )

    logger.info(
        "Identity-less transactions: %.2f%%",
        identityless_percentage,
    )

    if transaction_rows_before != transaction_rows_after:
        raise ValueError(
            "Row count changed after left join. "
            "Investigate duplicate or invalid TransactionIDs."
        )

    return joined_df


def main():
    """Run the IEEE-CIS ingestion pipeline."""

    logger.info("Starting IEEE-CIS data ingestion")

    if not TRANSACTION_PATH.exists():
        raise FileNotFoundError(
            f"Transaction dataset not found: {TRANSACTION_PATH}"
        )

    if not IDENTITY_PATH.exists():
        raise FileNotFoundError(
            f"Identity dataset not found: {IDENTITY_PATH}"
        )

    transaction_df, identity_df = load_data()

    validate_join_key(transaction_df, identity_df)

    joined_df = join_data(transaction_df, identity_df)

    logger.info(
        "Final joined dataset: %s rows x %s columns",
        joined_df.shape[0],
        joined_df.shape[1],
    )

    logger.info(
        "IEEE-CIS ingestion completed successfully."
    )


if __name__ == "__main__":
    main()