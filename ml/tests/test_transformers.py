"""Tests for the custom datetime transformer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from ml.features.transformers import (
    DatetimeComponentExtractor,
    parse_datetime,
    to_float_frame,
)
from ml.features.types import DatetimeComponent


@pytest.fixture
def dates() -> pd.DataFrame:
    """Three timestamps written as text, as a CSV would deliver them."""
    return pd.DataFrame(
        {"signup_date": ["2023-01-05", "2022-07-19", "2024-03-02"]}
    )


def test_extracts_the_default_components(dates: pd.DataFrame) -> None:
    """Year, month, day and day of week are produced by default."""
    result = DatetimeComponentExtractor().fit_transform(dates)

    assert list(result.columns) == [
        "signup_date_year",
        "signup_date_month",
        "signup_date_day",
        "signup_date_day_of_week",
    ]
    assert result.loc[0, "signup_date_year"] == 2023
    assert result.loc[0, "signup_date_month"] == 1
    assert result.loc[0, "signup_date_day"] == 5
    assert result.loc[0, "signup_date_day_of_week"] == 3, "2023-01-05 was a Thursday"


def test_components_are_configurable(dates: pd.DataFrame) -> None:
    """Only the requested components are produced."""
    extractor = DatetimeComponentExtractor(
        components=(DatetimeComponent.YEAR, DatetimeComponent.QUARTER)
    )
    result = extractor.fit_transform(dates)

    assert list(result.columns) == ["signup_date_year", "signup_date_quarter"]
    assert result.loc[1, "signup_date_quarter"] == 3


def test_components_accept_plain_strings(dates: pd.DataFrame) -> None:
    """Component names may be given as strings, as a stored config would."""
    result = DatetimeComponentExtractor(components=("year", "month")).fit_transform(dates)
    assert list(result.columns) == ["signup_date_year", "signup_date_month"]


def test_feature_names_match_the_output(dates: pd.DataFrame) -> None:
    """``get_feature_names_out`` agrees with the transformed columns."""
    extractor = DatetimeComponentExtractor().fit(dates)
    assert list(extractor.get_feature_names_out()) == list(
        extractor.transform(dates).columns
    )


def test_unparsable_values_become_missing() -> None:
    """A value that is not a date becomes NaN for the imputer to handle."""
    frame = pd.DataFrame({"day": ["2023-01-05", "not a date", None]})
    result = DatetimeComponentExtractor(components=("year",)).fit_transform(frame)

    assert result.loc[0, "day_year"] == 2023
    assert bool(np.isnan(result.loc[1, "day_year"]))
    assert bool(np.isnan(result.loc[2, "day_year"]))


def test_already_parsed_datetimes_pass_through() -> None:
    """A real datetime column needs no parsing."""
    frame = pd.DataFrame({"day": pd.to_datetime(["2020-02-29", "2021-12-31"])})
    result = DatetimeComponentExtractor(components=("year", "day")).fit_transform(frame)

    assert result["day_year"].tolist() == [2020.0, 2021.0]
    assert result["day_day"].tolist() == [29.0, 31.0]


def test_transform_before_fit_is_refused(dates: pd.DataFrame) -> None:
    """The transformer follows the sklearn contract."""
    with pytest.raises(NotFittedError):
        DatetimeComponentExtractor().transform(dates)


def test_parse_datetime_returns_nat_for_nonsense() -> None:
    """The parser never raises on unusable input."""
    parsed = parse_datetime(pd.Series(["nonsense", "also nonsense"]))
    assert parsed.isna().all()


def test_to_float_frame_preserves_names_and_missing_values() -> None:
    """Boolean casting keeps column names and leaves gaps for the imputer."""
    frame = pd.DataFrame({"flag": [True, False, None]}, dtype="object")
    result = to_float_frame(frame)

    assert list(result.columns) == ["flag"]
    assert result["flag"].tolist()[:2] == [1.0, 0.0]
    assert bool(result["flag"].isna().iloc[2])
