import os
import sqlite3
from pathlib import Path
from typing import Any
from uuid import uuid4

DB_PATH = Path(
    os.getenv(
        "FRAUDGUARD_PREDICTION_DB",
        str(Path(__file__).resolve().parents[2] / "data" / "monitoring" / "predictions.db"),
    )
)


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_store() -> None:
    with connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                fraud_probability REAL NOT NULL,
                predicted_fraud INTEGER NOT NULL CHECK (predicted_fraud IN (0, 1)),
                threshold REAL NOT NULL,
                true_label INTEGER CHECK (
                    true_label IS NULL OR true_label IN (0, 1)
                ),
                label_timestamp TEXT,
                label_source TEXT
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_predictions_timestamp
            ON predictions(timestamp)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_predictions_label_timestamp
            ON predictions(label_timestamp)
            """
        )


def record_prediction(
    fraud_probability: float,
    predicted_fraud: int,
    threshold: float,
) -> str:
    from datetime import datetime, timezone

    prediction_id = str(uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    with connect() as connection:
        connection.execute(
            """
            INSERT INTO predictions (
                prediction_id, timestamp, fraud_probability,
                predicted_fraud, threshold
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                prediction_id,
                timestamp,
                float(fraud_probability),
                int(predicted_fraud),
                float(threshold),
            ),
        )

    return prediction_id


def assign_label(
    prediction_id: str,
    true_label: int,
    label_source: str = "simulated",
) -> bool:
    from datetime import datetime, timezone

    if true_label not in (0, 1):
        raise ValueError("true_label must be 0 or 1")

    with connect() as connection:
        cursor = connection.execute(
            """
            UPDATE predictions
            SET true_label = ?, label_timestamp = ?, label_source = ?
            WHERE prediction_id = ? AND true_label IS NULL
            """,
            (
                int(true_label),
                datetime.now(timezone.utc).isoformat(),
                label_source,
                prediction_id,
            ),
        )

    return cursor.rowcount == 1


def get_prediction_stats() -> dict[str, Any]:
    with connect() as connection:
        row = connection.execute(
            """
            SELECT
                COUNT(*) AS total_predictions,
                COALESCE(SUM(predicted_fraud), 0) AS predicted_fraud_count,
                COUNT(true_label) AS labeled_predictions,
                COALESCE(
                    SUM(CASE WHEN true_label = 1 THEN 1 ELSE 0 END), 0
                ) AS actual_fraud_count,
                COALESCE(
                    SUM(
                        CASE
                            WHEN true_label IS NOT NULL
                             AND predicted_fraud = 1
                             AND true_label = 1
                            THEN 1 ELSE 0
                        END
                    ), 0
                ) AS true_positives,
                COALESCE(
                    SUM(
                        CASE
                            WHEN true_label IS NOT NULL
                             AND predicted_fraud = 1
                             AND true_label = 0
                            THEN 1 ELSE 0
                        END
                    ), 0
                ) AS false_positives,
                COALESCE(
                    SUM(
                        CASE
                            WHEN true_label IS NOT NULL
                             AND predicted_fraud = 0
                             AND true_label = 1
                            THEN 1 ELSE 0
                        END
                    ), 0
                ) AS false_negatives
            FROM predictions
            """
        ).fetchone()

    return dict(row)


def get_unlabeled_predictions(limit: int = 1000) -> list[dict[str, Any]]:
    with connect() as connection:
        rows = connection.execute(
            """
            SELECT prediction_id, timestamp, fraud_probability, predicted_fraud
            FROM predictions
            WHERE true_label IS NULL
            ORDER BY timestamp ASC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return [dict(row) for row in rows]
