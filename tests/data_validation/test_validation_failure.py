from pathlib import Path
import sys

import pandas as pd


# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Make src importable
sys.path.insert(0, str(PROJECT_ROOT))

from src.validation.validate_data import main


RAW_TRANSACTION = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "train_transaction.csv"
)

TEST_DIR = PROJECT_ROOT / "tests" / "data_validation"
CORRUPTED_FILE = TEST_DIR / "corrupted_transaction.csv"


def main_test():
    print("Creating corrupted test dataset...")

    # Take only 100 rows from the real dataset
    df = pd.read_csv(
        RAW_TRANSACTION,
        nrows=100,
    )

    original_value = df.loc[0, "TransactionAmt"]

    # Deliberately corrupt the dataset
    df.loc[0, "TransactionAmt"] = -100

    df.to_csv(
        CORRUPTED_FILE,
        index=False,
    )

    print(
        f"Original TransactionAmt: {original_value}"
    )

    print(
        f"Corrupted TransactionAmt: "
        f"{df.loc[0, 'TransactionAmt']}"
    )

    try:
        # Run the actual validator
        main(CORRUPTED_FILE)

    except ValueError as error:

        if "non-positive" in str(error):
            print(
                "PASS: Validator correctly rejected "
                "the corrupted dataset."
            )
        else:
            raise AssertionError(
                "Validator failed for an unexpected reason."
            )

    else:
        raise AssertionError(
            "Validator accepted corrupted data!"
        )

    finally:
        if CORRUPTED_FILE.exists():
            CORRUPTED_FILE.unlink()

        print(
            "Temporary corrupted dataset removed."
        )


if __name__ == "__main__":
    main_test()