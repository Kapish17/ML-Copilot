"""Vocabulary shared by the configuration, pipeline and result objects.

The string values mirror the semantic types produced by dataset profiling, so
a profile can be mapped onto a preprocessing configuration without either side
importing the other.
"""

from __future__ import annotations

from enum import Enum


class FeatureType(str, Enum):
    """How a column is treated by the preprocessing pipeline."""

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    DATETIME = "datetime"


class TaskType(str, Enum):
    """The kind of supervised problem the target represents."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    AUTO = "auto"


class ScalingStrategy(str, Enum):
    """How numeric features are rescaled."""

    STANDARD = "standard"
    MINMAX = "minmax"
    NONE = "none"


class NumericImputation(str, Enum):
    """How missing numeric values are filled."""

    MEDIAN = "median"
    MEAN = "mean"
    CONSTANT = "constant"


class CategoricalImputation(str, Enum):
    """How missing categorical values are filled."""

    MOST_FREQUENT = "most_frequent"
    CONSTANT = "constant"


class DatetimeComponent(str, Enum):
    """Calendar components that can be extracted from a datetime column."""

    YEAR = "year"
    MONTH = "month"
    DAY = "day"
    DAY_OF_WEEK = "day_of_week"
    QUARTER = "quarter"
    HOUR = "hour"


class ColumnRole(str, Enum):
    """The role a column plays once the configuration is resolved."""

    TARGET = "target"
    FEATURE = "feature"
    IDENTIFIER = "identifier"
    EXCLUDED = "excluded"


class ExclusionReason(str, Enum):
    """Why a column was not used as a feature.

    Every reason is reported back to the caller, so a column never disappears
    from the feature set without an explanation.
    """

    DECLARED_TARGET = "declared_target"
    EXCLUDED_BY_CALLER = "excluded_by_caller"
    IDENTIFIER_BY_CALLER = "identifier_by_caller"
    PROFILE_POSSIBLE_ID = "profile_possible_id"
    PROFILE_SUSPICIOUS = "profile_suspicious"
    CONSTANT_COLUMN = "constant_column"
    NO_VALUES = "no_values"
    FREE_TEXT = "free_text"
    HIGH_CARDINALITY = "high_cardinality"
    UNSUPPORTED_TYPE = "unsupported_type"
