"""Tests for the sklearn preprocessing pipeline itself.

These exercise ``build_preprocessor`` directly, fitting on one frame and
transforming another, which is the same fit/transform contract the dataset
preparation step relies on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from ml.features.config import PreprocessingConfig
from ml.pipelines.preprocessing import build_preprocessor, feature_names_of
from ml.tests.factories import categorical_frame, numeric_frame


def _fitted(config: PreprocessingConfig, frame: pd.DataFrame):
    """Return a preprocessor fitted on ``frame`` and its transformed output."""
    preprocessor = build_preprocessor(config)
    transformed = preprocessor.fit_transform(frame)
    return preprocessor, transformed


def test_numeric_features_are_imputed_and_scaled() -> None:
    """Numbers come out complete and standardised."""
    config = PreprocessingConfig(
        target_column="label", numeric_columns=("value", "other")
    )
    _, transformed = _fitted(config, numeric_frame())

    assert transformed["value"].isna().sum() == 0
    assert transformed["value"].mean() == pytest.approx(0.0, abs=1e-9)
    assert transformed["value"].std(ddof=0) == pytest.approx(1.0)


def test_missing_numeric_values_use_the_median_by_default() -> None:
    """The default numeric imputation is the median of the fitted data."""
    config = PreprocessingConfig(
        target_column="label", numeric_columns=("value",), scaling_strategy="none"
    )
    frame = numeric_frame()
    _, transformed = _fitted(config, frame)

    expected = frame["value"].median()
    assert transformed.loc[2, "value"] == pytest.approx(expected)


def test_mean_imputation_can_be_selected() -> None:
    """Mean imputation is available when explicitly chosen."""
    config = PreprocessingConfig(
        target_column="label",
        numeric_columns=("value",),
        numeric_imputation="mean",
        scaling_strategy="none",
    )
    frame = numeric_frame()
    _, transformed = _fitted(config, frame)

    assert transformed.loc[2, "value"] == pytest.approx(frame["value"].mean())


def test_missing_indicator_is_added_and_left_unscaled() -> None:
    """A gap in the data becomes a readable 0/1 feature of its own."""
    config = PreprocessingConfig(target_column="label", numeric_columns=("value",))
    _, transformed = _fitted(config, numeric_frame())

    assert "missingindicator_value" in transformed.columns
    assert set(transformed["missingindicator_value"].unique()) == {0.0, 1.0}
    assert transformed.loc[2, "missingindicator_value"] == 1.0


def test_missing_indicators_can_be_disabled() -> None:
    """The indicator is configurable, not mandatory."""
    config = PreprocessingConfig(
        target_column="label", numeric_columns=("value",), add_missing_indicators=False
    )
    _, transformed = _fitted(config, numeric_frame())

    assert list(transformed.columns) == ["value"]


def test_columns_without_gaps_get_no_indicator() -> None:
    """Indicators are only created for columns that actually had gaps."""
    config = PreprocessingConfig(
        target_column="label", numeric_columns=("value", "other")
    )
    _, transformed = _fitted(config, numeric_frame())

    assert "missingindicator_value" in transformed.columns
    assert "missingindicator_other" not in transformed.columns


def test_scaling_can_be_switched_off() -> None:
    """With scaling disabled the numbers keep their original values."""
    config = PreprocessingConfig(
        target_column="label", numeric_columns=("other",), scaling_strategy="none"
    )
    frame = numeric_frame()
    _, transformed = _fitted(config, frame)

    assert transformed["other"].tolist() == frame["other"].tolist()


def test_minmax_scaling_is_available() -> None:
    """Min-max scaling maps the fitted range onto 0..1."""
    config = PreprocessingConfig(
        target_column="label", numeric_columns=("other",), scaling_strategy="minmax"
    )
    _, transformed = _fitted(config, numeric_frame())

    assert transformed["other"].min() == pytest.approx(0.0)
    assert transformed["other"].max() == pytest.approx(1.0)


def test_categorical_features_are_one_hot_encoded() -> None:
    """Each category becomes its own readable column."""
    config = PreprocessingConfig(
        target_column="label", categorical_columns=("contract",)
    )
    _, transformed = _fitted(config, categorical_frame())

    assert list(transformed.columns) == [
        "contract_Month-to-month",
        "contract_One-year",
        "contract_Two-year",
    ]
    assert set(transformed.to_numpy().ravel()) <= {0.0, 1.0}
    assert transformed.sum(axis=1).tolist() == [1.0] * len(transformed)


def test_missing_categories_use_the_most_frequent_value() -> None:
    """The default categorical imputation fills the commonest category."""
    config = PreprocessingConfig(
        target_column="label", categorical_columns=("contract",)
    )
    _, transformed = _fitted(config, categorical_frame())

    assert transformed.loc[3, "contract_Month-to-month"] == 1.0


def test_constant_categorical_imputation_creates_its_own_category() -> None:
    """Filling with a constant keeps missingness visible as a category."""
    config = PreprocessingConfig(
        target_column="label",
        categorical_columns=("contract",),
        categorical_imputation="constant",
    )
    _, transformed = _fitted(config, categorical_frame())

    assert "contract_Unknown" in transformed.columns
    assert transformed.loc[3, "contract_Unknown"] == 1.0


def test_unknown_categories_are_handled_at_transform_time() -> None:
    """A category never seen during fitting must not break inference."""
    config = PreprocessingConfig(
        target_column="label", categorical_columns=("contract",)
    )
    preprocessor, _ = _fitted(config, categorical_frame())

    unseen = pd.DataFrame({"contract": ["Lifetime", "One-year"]})
    transformed = preprocessor.transform(unseen)

    assert list(transformed.columns) == list(feature_names_of(preprocessor))
    assert transformed.iloc[0].tolist() == [0.0, 0.0, 0.0], "unseen category is all zeros"
    assert transformed.loc[1, "contract_One-year"] == 1.0


def test_boolean_features_become_zero_and_one() -> None:
    """Booleans are cast rather than one-hot encoded."""
    frame = pd.DataFrame({"is_active": [True, False, None], "label": ["a", "b", "a"]})
    config = PreprocessingConfig(target_column="label", boolean_columns=("is_active",))
    _, transformed = _fitted(config, frame)

    assert transformed["is_active"].tolist()[:2] == [1.0, 0.0]
    assert transformed["is_active"].isna().sum() == 0
    assert "missingindicator_is_active" in transformed.columns


def test_datetime_features_are_expanded() -> None:
    """Datetime columns contribute calendar components, not raw timestamps."""
    frame = pd.DataFrame(
        {
            "signup_date": ["2023-01-05", "2022-07-19", "2024-03-02"],
            "label": ["a", "b", "a"],
        }
    )
    config = PreprocessingConfig(target_column="label", datetime_columns=("signup_date",))
    _, transformed = _fitted(config, frame)

    assert list(transformed.columns) == [
        "signup_date_year",
        "signup_date_month",
        "signup_date_day",
        "signup_date_day_of_week",
    ]
    assert transformed.loc[0, "signup_date_year"] == 2023


def test_feature_names_survive_every_branch() -> None:
    """Names stay traceable to their source column across all four branches."""
    frame = pd.DataFrame(
        {
            "monthly_charges": [10.0, 20.0, 30.0],
            "contract": ["Month-to-month", "One-year", "Two-year"],
            "is_active": [True, False, True],
            "signup_date": ["2023-01-05", "2022-07-19", "2024-03-02"],
            "label": ["a", "b", "a"],
        }
    )
    config = PreprocessingConfig(
        target_column="label",
        numeric_columns=("monthly_charges",),
        categorical_columns=("contract",),
        boolean_columns=("is_active",),
        datetime_columns=("signup_date",),
    )
    preprocessor, transformed = _fitted(config, frame)

    assert feature_names_of(preprocessor) == (
        "monthly_charges",
        "contract_Month-to-month",
        "contract_One-year",
        "contract_Two-year",
        "is_active",
        "signup_date_year",
        "signup_date_month",
        "signup_date_day",
        "signup_date_day_of_week",
    )
    assert list(transformed.columns) == list(feature_names_of(preprocessor))


def test_only_configured_columns_reach_the_pipeline() -> None:
    """Everything not in a feature group is dropped, including the target."""
    config = PreprocessingConfig(target_column="label", numeric_columns=("other",))
    _, transformed = _fitted(config, numeric_frame())

    assert list(transformed.columns) == ["other"]


def test_preprocessor_starts_unfitted() -> None:
    """``build_preprocessor`` returns a fresh, unfitted estimator."""
    config = PreprocessingConfig(target_column="label", numeric_columns=("other",))
    with pytest.raises(NotFittedError):
        build_preprocessor(config).transform(numeric_frame())


def test_transform_output_is_stable_after_fitting() -> None:
    """Transforming the fitted frame again reproduces the same values."""
    config = PreprocessingConfig(target_column="label", numeric_columns=("other",))
    frame = numeric_frame()
    preprocessor, first = _fitted(config, frame)
    second = preprocessor.transform(frame)

    assert np.allclose(first.to_numpy(), second.to_numpy())
