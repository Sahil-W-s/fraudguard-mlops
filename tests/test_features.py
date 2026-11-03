import pandas as pd
import pytest

from src.features.build_features import (
    TARGET_COLUMN,
    ID_COLUMNS,
    build_feature_inventory,
    calculate_missingness,
    get_candidate_features,
    identify_categorical_features,
    identify_feature_columns,
    identify_numeric_features,
    validate_required_columns,
)


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4],
            "TransactionDT": [100, 200, 300, 400],
            "TransactionAmt": [10.5, 20.0, 15.5, 100.0],
            "ProductCD": ["W", "C", "W", "H"],
            "card1": [1000, 2000, 3000, None],
            "isFraud": [0, 1, 0, 0],
        }
    )


def test_required_columns_are_present(sample_dataframe):
    validate_required_columns(sample_dataframe)


def test_missing_required_column_is_detected(sample_dataframe):
    df = sample_dataframe.drop(columns=["TransactionDT"])

    with pytest.raises(ValueError):
        validate_required_columns(df)


def test_target_is_not_a_feature(sample_dataframe):
    features = identify_feature_columns(sample_dataframe)

    assert TARGET_COLUMN not in features


def test_identifier_is_not_a_feature(sample_dataframe):
    features = identify_feature_columns(sample_dataframe)

    for column in ID_COLUMNS:
        assert column not in features


def test_numeric_features_are_detected(sample_dataframe):
    features = identify_feature_columns(sample_dataframe)

    numeric = identify_numeric_features(
        sample_dataframe,
        features,
    )

    assert "TransactionAmt" in numeric
    assert "card1" in numeric


def test_categorical_features_are_detected(sample_dataframe):
    features = identify_feature_columns(sample_dataframe)

    categorical = identify_categorical_features(
        sample_dataframe,
        features,
    )

    assert "ProductCD" in categorical


def test_missingness_is_calculated(sample_dataframe):
    result = calculate_missingness(sample_dataframe)

    card1 = result.loc[
        result["feature"] == "card1"
    ].iloc[0]

    assert card1["missing_count"] == 1
    assert card1["missing_percentage"] == 25.0


def test_inventory_contains_expected_roles(sample_dataframe):
    inventory = build_feature_inventory(
        sample_dataframe
    )

    roles = dict(
        zip(
            inventory["feature"],
            inventory["role"],
        )
    )

    assert roles["isFraud"] == "target"
    assert roles["TransactionID"] == "identifier"
    assert roles["TransactionDT"] == "time"
    assert roles["TransactionAmt"] == "numerical"
    assert roles["ProductCD"] == "categorical"


def test_candidate_features_exclude_target_and_identifiers(
    sample_dataframe,
):
    inventory = build_feature_inventory(
        sample_dataframe
    )

    candidates = get_candidate_features(
        inventory
    )

    assert "isFraud" not in candidates
    assert "TransactionID" not in candidates
    assert "TransactionDT" not in candidates