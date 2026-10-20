
"""Assign reproducible synthetic labels to previously unlabeled predictions.

These labels are for monitoring-pipeline demonstrations only. They are not
ground truth and must not be used to evaluate real-world model performance.
"""

import argparse
import hashlib
import logging

from src.api.prediction_store import (
    assign_label,
    get_unlabeled_predictions,
    initialize_store,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def simulated_label(prediction_id: str, fraud_probability: float) -> int:
    """Generate a deterministic synthetic label for demonstration purposes.

    This uses a stable hash so rerunning the script produces the same
    label for each prediction. The label is intentionally synthetic.
    """
    digest = hashlib.sha256(prediction_id.encode("utf-8")).digest()
    noise = int.from_bytes(digest[:4], "big") / (2**32 - 1)

    # A simple illustrative rule, not a calibrated probability model.
    return int(noise < fraud_probability)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1000)
    args = parser.parse_args()

    if args.limit < 1:
        parser.error("--limit must be at least 1")

    initialize_store()
    predictions = get_unlabeled_predictions(limit=args.limit)

    if not predictions:
        logger.info("No unlabeled predictions found.")
        return

    assigned = 0

    for item in predictions:
        label = simulated_label(
            item["prediction_id"],
            float(item["fraud_probability"]),
        )

        if assign_label(
            prediction_id=item["prediction_id"],
            true_label=label,
            label_source="simulated",
        ):
            assigned += 1

    logger.info(
        "Assigned %d synthetic labels from %d eligible predictions.",
        assigned,
        len(predictions),
    )
    logger.warning(
        "Synthetic labels are for pipeline testing only; "
        "they do not establish real model precision or recall."
    )


if __name__ == "__main__":
    main()
