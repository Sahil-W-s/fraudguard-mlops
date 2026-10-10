import pandas as pd
import pytest

from src.features.feature_pipeline import FeaturePipeline


@pytest.fixture
def training_data():
    return pd.DataFrame({
        "amount": [10.5, 20.0, None, 40.0],
        "product": ["A", "B", "A", None],
        "isFraud": [0, 1, 0, 1],
    })


def test_fit_transform_preserves_feature_order(training_data):
    pipeline = FeaturePipeline(
        feature_names=["product", "amount"]
    )

    transformed = pipeline.fit_transform(training_data)

    assert transformed.columns.tolist() == [
        "product",
        "amount",
    ]
    assert len(transformed) == len(training_data)


def test_categorical_levels_are_learned_from_training(
    training_data,
):
    pipeline = FeaturePipeline(
        feature_names=["product", "amount"]
    )

    pipeline.fit(training_data)

    inference = pd.DataFrame({
        "amount": [50.0],
        "product": ["UNSEEN_PRODUCT"],
    })

    transformed = pipeline.transform(inference)

    assert pd.isna(transformed.loc[0, "product"])
    assert transformed["product"].dtype.name == "category"


def test_numeric_missing_values_are_preserved(training_data):
    pipeline = FeaturePipeline(
        feature_names=["amount", "product"]
    )

    transformed = pipeline.fit_transform(training_data)

    assert transformed["amount"].isna().sum() == 1


def test_missing_required_feature_raises_error(training_data):
    pipeline = FeaturePipeline(
        feature_names=["amount", "product"]
    ).fit(training_data)

    incomplete = pd.DataFrame({
        "amount": [100.0],
    })

    with pytest.raises(
        ValueError,
        match="Missing required features",
    ):
        pipeline.transform(incomplete)


def test_transform_requires_fit(training_data):
    pipeline = FeaturePipeline(
        feature_names=["amount", "product"]
    )

    with pytest.raises(
        RuntimeError,
        match="Call fit",
    ):
        pipeline.transform(training_data)


def test_save_and_load_pipeline(tmp_path, training_data):
    pipeline = FeaturePipeline(
        feature_names=["amount", "product"]
    ).fit(training_data)

    artifact = tmp_path / "feature_pipeline.joblib"
    pipeline.save(artifact)

    loaded = FeaturePipeline.load(artifact)

    original_result = pipeline.transform(training_data)
    loaded_result = loaded.transform(training_data)

    pd.testing.assert_frame_equal(
        original_result,
        loaded_result,
    )