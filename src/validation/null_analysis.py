from pathlib import Path
import logging

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRANSACTION_PATH = PROJECT_ROOT / "data" / "raw" / "train_transaction.csv"
IDENTITY_PATH = PROJECT_ROOT / "data" / "raw" / "train_identity.csv"

REPORT_DIR = PROJECT_ROOT / "docs" / "data_analysis"
NULL_REPORT_PATH = REPORT_DIR / "null_analysis.csv"
SIGNAL_REPORT_PATH = REPORT_DIR / "high_null_fraud_signal.csv"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def load_datasets():
    """Load the raw IEEE-CIS datasets."""

    logger.info("Loading transaction dataset...")
    transaction_df = pd.read_csv(TRANSACTION_PATH)

    logger.info("Loading identity dataset...")
    identity_df = pd.read_csv(IDENTITY_PATH)

    return transaction_df, identity_df


def calculate_null_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate missing-value statistics for every column."""

    total_rows = len(df)

    null_count = df.isna().sum()
    null_percentage = (null_count / total_rows) * 100

    statistics = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": df.dtypes.astype(str).values,
            "total_rows": total_rows,
            "missing_count": null_count.values,
            "missing_percentage": null_percentage.values,
        }
    )

    return statistics.sort_values(
        by="missing_percentage",
        ascending=False,
    )


def get_high_null_columns(transaction_df, identity_df):
    """Return columns with more than 90% missing values."""

    transaction_stats = calculate_null_statistics(transaction_df)
    identity_stats = calculate_null_statistics(identity_df)

    transaction_high_null = transaction_stats[
        transaction_stats["missing_percentage"] > 90
    ]["column"].tolist()

    identity_high_null = identity_stats[
        identity_stats["missing_percentage"] > 90
    ]["column"].tolist()

    return transaction_high_null, identity_high_null


def analyze_fraud_signal(
    transaction_df: pd.DataFrame,
    identity_df: pd.DataFrame,
    transaction_columns: list[str],
    identity_columns: list[str],
):
    """
    Analyze whether missingness in high-null features
    is associated with fraud.
    """

    logger.info("Analyzing fraud signal in high-null features...")

    # Add identity columns to the transaction table.
    identity_subset = identity_df[
        ["TransactionID"] + identity_columns
    ].copy()

    merged_df = transaction_df.merge(
        identity_subset,
        on="TransactionID",
        how="left",
        validate="one_to_one",
    )

    overall_fraud_rate = merged_df["isFraud"].mean() * 100

    logger.info(
        "Overall fraud rate: %.4f%%",
        overall_fraud_rate,
    )

    results = []

    all_columns = transaction_columns + identity_columns

    for column in all_columns:

        missing_mask = merged_df[column].isna()
        present_mask = ~missing_mask

        missing_count = int(missing_mask.sum())
        present_count = int(present_mask.sum())

        missing_fraud_count = int(
            merged_df.loc[missing_mask, "isFraud"].sum()
        )

        present_fraud_count = int(
            merged_df.loc[present_mask, "isFraud"].sum()
        )

        missing_fraud_rate = (
            missing_fraud_count / missing_count * 100
            if missing_count > 0
            else 0
        )

        present_fraud_rate = (
            present_fraud_count / present_count * 100
            if present_count > 0
            else 0
        )

        results.append(
            {
                "column": column,
                "missing_count": missing_count,
                "present_count": present_count,
                "missing_fraud_count": missing_fraud_count,
                "present_fraud_count": present_fraud_count,
                "missing_fraud_rate": missing_fraud_rate,
                "present_fraud_rate": present_fraud_rate,
                "fraud_rate_difference": (
                    missing_fraud_rate - present_fraud_rate
                ),
            }
        )

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="missing_fraud_rate",
        ascending=False,
    )

    return results_df


def main():
    """Run complete null and missingness-signal analysis."""

    logger.info("Starting null-value analysis.")

    if not TRANSACTION_PATH.exists():
        raise FileNotFoundError(
            f"Transaction dataset not found: {TRANSACTION_PATH}"
        )

    if not IDENTITY_PATH.exists():
        raise FileNotFoundError(
            f"Identity dataset not found: {IDENTITY_PATH}"
        )

    transaction_df, identity_df = load_datasets()

    # ---------------------------------------------------------
    # 1. Calculate complete null statistics
    # ---------------------------------------------------------

    transaction_stats = calculate_null_statistics(
        transaction_df
    )

    identity_stats = calculate_null_statistics(
        identity_df
    )

    transaction_stats.insert(
        0,
        "dataset",
        "transaction",
    )

    identity_stats.insert(
        0,
        "dataset",
        "identity",
    )

    all_statistics = pd.concat(
        [
            transaction_stats,
            identity_stats,
        ],
        ignore_index=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_statistics.to_csv(
        NULL_REPORT_PATH,
        index=False,
    )

    # ---------------------------------------------------------
    # 2. Identify >90% null columns
    # ---------------------------------------------------------

    transaction_high_null, identity_high_null = (
        get_high_null_columns(
            transaction_df,
            identity_df,
        )
    )

    logger.info(
        "Transaction columns >90%% null: %s",
        len(transaction_high_null),
    )

    logger.info(
        "Identity columns >90%% null: %s",
        len(identity_high_null),
    )

    logger.info(
        "High-null transaction columns: %s",
        transaction_high_null,
    )

    logger.info(
        "High-null identity columns: %s",
        identity_high_null,
    )

    # ---------------------------------------------------------
    # 3. Analyze relationship between missingness and fraud
    # ---------------------------------------------------------

    signal_results = analyze_fraud_signal(
        transaction_df,
        identity_df,
        transaction_high_null,
        identity_high_null,
    )

    signal_results.to_csv(
        SIGNAL_REPORT_PATH,
        index=False,
    )

    logger.info(
        "High-null fraud signal report saved to: %s",
        SIGNAL_REPORT_PATH,
    )

    # ---------------------------------------------------------
    # 4. Print results
    # ---------------------------------------------------------

    logger.info(
        "\nHigh-null feature fraud analysis:\n%s",
        signal_results.to_string(index=False),
    )

    logger.info(
        "Null-value analysis completed successfully."
    )


if __name__ == "__main__":
    main()