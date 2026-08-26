"""Custom sklearn transformers used by the preprocessing pipeline.

Only what sklearn does not already provide lives here. Everything else —
imputation, scaling, one-hot encoding — uses the stock estimators, so the
pipeline stays a standard sklearn object that can be inspected, cloned and
later persisted with the model.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from pandas.api import types as pdt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ml.features.types import DatetimeComponent

#: How each supported component is read off a parsed datetime column.
_COMPONENT_ACCESSORS = {
    DatetimeComponent.YEAR: lambda values: values.dt.year,
    DatetimeComponent.MONTH: lambda values: values.dt.month,
    DatetimeComponent.DAY: lambda values: values.dt.day,
    DatetimeComponent.DAY_OF_WEEK: lambda values: values.dt.dayofweek,
    DatetimeComponent.QUARTER: lambda values: values.dt.quarter,
    DatetimeComponent.HOUR: lambda values: values.dt.hour,
}


def to_float_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Cast a frame to floats, keeping column names and missing values.

    Used to turn boolean columns into the ``0.0`` / ``1.0`` encoding the rest
    of the pipeline expects, without deciding what to do about missing values;
    that remains the imputer's job.
    """
    return pd.DataFrame(frame).astype("float64")


def parse_datetime(values: pd.Series) -> pd.Series:
    """Parse a column into datetimes, turning unparsable entries into ``NaT``.

    Only columns the configuration lists as datetimes reach this function, so
    arbitrary text is never coerced into dates by accident.
    """
    if pdt.is_datetime64_any_dtype(values):
        return values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return pd.to_datetime(values, errors="coerce", format="mixed")
        except (ValueError, TypeError):
            return pd.to_datetime(values, errors="coerce")


class DatetimeComponentExtractor(BaseEstimator, TransformerMixin):
    """Expand datetime columns into numeric calendar components.

    A raw timestamp is useless to most estimators, while its year, month, day
    and day of week are ordinary numeric features. Each input column produces
    one output column per configured component, named ``<column>_<component>``
    so the origin of every feature stays readable.

    Unparsable or missing timestamps become ``NaN`` and are left for the
    imputer that follows in the pipeline.
    """

    def __init__(
        self,
        components: Sequence[DatetimeComponent | str] = (
            DatetimeComponent.YEAR,
            DatetimeComponent.MONTH,
            DatetimeComponent.DAY,
            DatetimeComponent.DAY_OF_WEEK,
        ),
    ) -> None:
        """Store the components to extract, unchanged, as sklearn requires."""
        self.components = components

    def _resolved_components(self) -> tuple[DatetimeComponent, ...]:
        """Return the configured components as enum members."""
        return tuple(DatetimeComponent(component) for component in self.components)

    @staticmethod
    def _as_frame(X: pd.DataFrame | Iterable) -> pd.DataFrame:
        """Return the input as a DataFrame without copying when possible."""
        return X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

    def fit(self, X: pd.DataFrame, y: object = None) -> DatetimeComponentExtractor:
        """Record the input columns; no statistics are learned from the data.

        Args:
            X: Datetime columns to expand.
            y: Ignored, present for the sklearn API.

        Returns:
            DatetimeComponentExtractor: The fitted transformer.
        """
        frame = self._as_frame(X)
        self.feature_names_in_ = np.asarray(
            [str(column) for column in frame.columns], dtype=object
        )
        self.n_features_in_ = frame.shape[1]
        self._components_ = self._resolved_components()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Expand each datetime column into its configured components.

        Args:
            X: Datetime columns, in the same order as at fit time.

        Returns:
            pandas.DataFrame: One float column per column-component pair.
        """
        check_is_fitted(self, "feature_names_in_")
        frame = self._as_frame(X)
        extracted: dict[str, pd.Series] = {}
        for position, name in enumerate(self.feature_names_in_):
            source = frame[name] if name in frame.columns else frame.iloc[:, position]
            parsed = parse_datetime(source)
            for component in self._components_:
                accessor = _COMPONENT_ACCESSORS[component]
                extracted[f"{name}_{component.value}"] = accessor(parsed).astype(
                    "float64"
                )
        return pd.DataFrame(extracted, index=frame.index)

    def get_feature_names_out(
        self, input_features: Sequence[str] | None = None
    ) -> np.ndarray:
        """Return the generated feature names, in output order.

        Args:
            input_features: Ignored when the transformer is already fitted.

        Returns:
            numpy.ndarray: Names of the form ``<column>_<component>``.
        """
        check_is_fitted(self, "feature_names_in_")
        return np.asarray(
            [
                f"{name}_{component.value}"
                for name in self.feature_names_in_
                for component in self._components_
            ],
            dtype=object,
        )
