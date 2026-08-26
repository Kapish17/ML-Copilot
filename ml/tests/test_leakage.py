"""Tests proving that preprocessing never learns anything from the test set.

Leakage is the failure mode that quietly inflates every later metric, so it is
checked directly rather than assumed: each test compares a learned statistic
against the training rows alone, or shows that changing the test data cannot
change what was fitted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.features.config import PreprocessingConfig
from ml.pipelines.preparation import prepare_dataset
from ml.pipelines.preprocessing import build_preprocessor
from ml.pipelines.result import PreparedDataset
from ml.tests.factories import SEED, classification_frame

NUMERIC_BRANCH = "numeric"
CATEGORICAL_BRANCH = "categorical"


def _numeric_values_pipeline(prepared: PreparedDataset):
    """Return the impute-then-scale pipeline of the numeric branch.

    The numeric branch is a ``FeatureUnion`` of transformed values and missing
    indicators; the learned statistics live in the ``values`` half.
    """
    branch = prepared.preprocessor.named_transformers_[NUMERIC_BRANCH]
    return dict(branch.transformer_list)["values"]


def _encoder(prepared: PreparedDataset):
    """Return the fitted one-hot encoder."""
    branch = prepared.preprocessor.named_transformers_[CATEGORICAL_BRANCH]
    return branch.named_steps["encode"]


def _config(**overrides: object) -> PreprocessingConfig:
    """Configuration for the churn dataset used throughout this module."""
    config = PreprocessingConfig(
        target_column="churn",
        numeric_columns=("age", "monthly_charges"),
        categorical_columns=("contract",),
        excluded_columns=("notes", "plan", "payment_method"),
        identifier_columns=("customer_id",),
        boolean_columns=("is_active",),
        datetime_columns=("signup_date",),
        task_type="classification",
    )
    return config.with_overrides(**overrides) if overrides else config


def test_imputer_learns_the_training_median_only() -> None:
    """The fill value comes from the training rows, not the whole dataset."""
    frame = classification_frame()
    prepared = prepare_dataset(frame, _config())
    imputer = _numeric_values_pipeline(prepared).named_steps["impute"]

    train_median = frame.loc[prepared.X_train.index, "age"].median()
    full_median = frame["age"].median()

    assert imputer.statistics_[0] == pytest.approx(train_median)
    assert train_median != pytest.approx(full_median), (
        "the dataset must be arranged so the two medians differ, otherwise "
        "this test cannot tell them apart"
    )


def test_scaler_learns_the_training_statistics_only() -> None:
    """Centring and scaling are derived from the training rows alone."""
    frame = classification_frame()
    prepared = prepare_dataset(frame, _config())
    scaler = _numeric_values_pipeline(prepared).named_steps["scale"]

    train_values = frame.loc[prepared.X_train.index, "monthly_charges"]
    full_values = frame["monthly_charges"]

    assert scaler.mean_[1] == pytest.approx(train_values.mean())
    assert scaler.mean_[1] != pytest.approx(full_values.mean())


def test_scaled_test_data_is_not_recentred() -> None:
    """The test set is transformed, never re-fitted, so it is not centred."""
    prepared = prepare_dataset(classification_frame(), _config())

    assert prepared.X_train["monthly_charges"].mean() == pytest.approx(0.0, abs=1e-9)
    assert prepared.X_test["monthly_charges"].mean() != pytest.approx(0.0, abs=1e-6)


def _single_occurrence_frame(rows: int = 40) -> pd.DataFrame:
    """A dataset where every category occurs exactly once.

    Because no category is repeated, the categories in the test half are
    guaranteed to be absent from the training half, which makes the encoder
    tests below impossible to satisfy by accident.
    """
    rng = np.random.default_rng(SEED)
    return pd.DataFrame(
        {
            "code": [f"cat_{index}" for index in range(rows)],
            "value": rng.normal(0, 1, rows).round(3),
            "price": rng.normal(100, 10, rows).round(2),
        }
    )


def _single_occurrence_config() -> PreprocessingConfig:
    """Configuration for the single-occurrence dataset."""
    return PreprocessingConfig(
        target_column="price",
        numeric_columns=("value",),
        categorical_columns=("code",),
        task_type="regression",
    )


def test_encoder_learns_training_categories_only() -> None:
    """Categories seen only in the test set never become features."""
    frame = _single_occurrence_frame()
    prepared = prepare_dataset(frame, _single_occurrence_config())

    train_categories = set(frame.loc[prepared.X_train.index, "code"])
    test_only = set(frame.loc[prepared.X_test.index, "code"]) - train_categories

    assert test_only, "the fixture must produce categories unique to the test set"
    assert set(_encoder(prepared).categories_[0]) == train_categories
    for category in test_only:
        assert f"code_{category}" not in prepared.feature_names


def test_unseen_test_categories_encode_as_all_zeros() -> None:
    """Unknown categories are handled quietly rather than raising."""
    frame = _single_occurrence_frame()
    prepared = prepare_dataset(frame, _single_occurrence_config())

    one_hot_columns = [
        name for name in prepared.feature_names if name.startswith("code_")
    ]
    assert prepared.X_test[one_hot_columns].to_numpy().sum() == 0.0


def test_changing_the_test_rows_cannot_change_what_was_fitted() -> None:
    """The decisive check: test values have no influence on the pipeline.

    The split depends only on the target and the seed, so corrupting the
    feature values of the test rows leaves the same rows on each side. If any
    statistic were fitted on the full dataset, the learned parameters or the
    transformed training data would move. Neither does.
    """
    frame = classification_frame()
    baseline = prepare_dataset(frame, _config())

    corrupted = frame.copy()
    test_rows = baseline.X_test.index
    corrupted.loc[test_rows, "age"] = 10_000.0
    corrupted.loc[test_rows, "monthly_charges"] = -10_000.0
    corrupted.loc[test_rows, "contract"] = "Corrupted-contract"

    after = prepare_dataset(corrupted, _config())

    assert list(after.X_test.index) == list(test_rows), "the split must be unchanged"
    assert after.feature_names == baseline.feature_names
    assert np.allclose(
        after.X_train.to_numpy(), baseline.X_train.to_numpy()
    ), "training features changed, so something was fitted on test data"

    baseline_imputer = _numeric_values_pipeline(baseline).named_steps["impute"]
    after_imputer = _numeric_values_pipeline(after).named_steps["impute"]
    assert np.allclose(baseline_imputer.statistics_, after_imputer.statistics_)

    baseline_scaler = _numeric_values_pipeline(baseline).named_steps["scale"]
    after_scaler = _numeric_values_pipeline(after).named_steps["scale"]
    assert np.allclose(baseline_scaler.mean_, after_scaler.mean_)
    assert np.allclose(baseline_scaler.scale_, after_scaler.scale_)

    assert list(_encoder(after).categories_[0]) == list(
        _encoder(baseline).categories_[0]
    )


def test_missing_indicators_are_decided_at_fit_time() -> None:
    """A gap that appears only after fitting cannot add a new feature.

    Fitting on complete data and transforming data with gaps is exactly the
    situation at inference time: the feature layout must not change, and the
    gap must be filled with a statistic learned during fitting.
    """
    config = PreprocessingConfig(target_column="y", numeric_columns=("value",))
    training = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0, 5.0]})
    later = pd.DataFrame({"value": [1.0, np.nan]})

    preprocessor = build_preprocessor(config)
    fitted_columns = list(preprocessor.fit_transform(training).columns)
    transformed = preprocessor.transform(later)

    assert "missingindicator_value" not in fitted_columns
    assert list(transformed.columns) == fitted_columns
    assert transformed["value"].isna().sum() == 0


def test_the_target_is_never_passed_to_the_preprocessor() -> None:
    """No feature transformer ever sees the values it is meant to predict."""
    prepared = prepare_dataset(classification_frame(), _config())
    seen_columns = list(prepared.preprocessor.feature_names_in_)

    assert "churn" not in seen_columns
    assert seen_columns == list(prepared.config.feature_columns)
