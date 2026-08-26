"""The preprocessing configuration and its validation.

``PreprocessingConfig`` is the single description of how a dataset should be
turned into model-ready arrays: which column plays which role, how missing
values are filled, how features are encoded and scaled, and how the data is
split. It is a plain frozen dataclass with no dependency on pandas, sklearn or
the web layer, so it can be built by hand, derived from a dataset profile, or
later deserialised from a stored experiment definition.

Defaults are chosen to be safe rather than clever: median imputation, standard
scaling, one-hot encoding that tolerates unseen categories, and no resampling
of any kind.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from ml.errors import (
    ConfigurationError,
    DuplicateColumnAssignmentError,
    EmptyFeatureSetError,
    MissingTargetError,
    TargetLeakageError,
    UnknownColumnError,
)
from ml.features.types import (
    CategoricalImputation,
    DatetimeComponent,
    FeatureType,
    NumericImputation,
    ScalingStrategy,
    TaskType,
)

DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_MAX_CATEGORICAL_CARDINALITY = 50
DEFAULT_MAX_CLASSIFICATION_CLASSES = 20
DEFAULT_CATEGORICAL_FILL_VALUE = "Unknown"
DEFAULT_NUMERIC_FILL_VALUE = 0.0
DEFAULT_DATETIME_COMPONENTS: tuple[DatetimeComponent, ...] = (
    DatetimeComponent.YEAR,
    DatetimeComponent.MONTH,
    DatetimeComponent.DAY,
    DatetimeComponent.DAY_OF_WEEK,
)

#: Fewest rows that can still be divided into a usable train and test set.
MIN_ROWS_FOR_SPLIT = 4


def _as_tuple(values: Iterable[str] | None) -> tuple[str, ...]:
    """Normalise an optional sequence of column names into a tuple."""
    return tuple(values) if values is not None else ()


def _coerce(enum_type: type, value: Any, field_name: str) -> Any:
    """Convert a plain string into the matching enum member.

    Args:
        enum_type: The enum the value belongs to.
        value: An enum member or its string value.
        field_name: Field name, used in the error message.

    Returns:
        The enum member.

    Raises:
        ConfigurationError: If the value is not a valid member.
    """
    try:
        return enum_type(value)
    except ValueError as exc:
        allowed = ", ".join(member.value for member in enum_type)
        raise ConfigurationError(
            f"Invalid {field_name}: {value!r}. Allowed values: {allowed}.",
            details={"field": field_name, "value": str(value), "allowed": allowed},
        ) from exc


@dataclass(frozen=True)
class PreprocessingConfig:
    """How a dataset is turned into model-ready training and test data."""

    target_column: str

    # Feature groups. Empty groups are simply skipped by the pipeline.
    numeric_columns: tuple[str, ...] = ()
    categorical_columns: tuple[str, ...] = ()
    boolean_columns: tuple[str, ...] = ()
    datetime_columns: tuple[str, ...] = ()

    # Columns that are deliberately kept out of the feature set.
    identifier_columns: tuple[str, ...] = ()
    excluded_columns: tuple[str, ...] = ()

    # Modelling intent, used to decide whether stratification applies.
    task_type: TaskType = TaskType.AUTO

    # Transformation strategies.
    scaling_strategy: ScalingStrategy = ScalingStrategy.STANDARD
    numeric_imputation: NumericImputation = NumericImputation.MEDIAN
    categorical_imputation: CategoricalImputation = CategoricalImputation.MOST_FREQUENT
    numeric_fill_value: float = DEFAULT_NUMERIC_FILL_VALUE
    categorical_fill_value: str = DEFAULT_CATEGORICAL_FILL_VALUE
    add_missing_indicators: bool = True
    datetime_components: tuple[DatetimeComponent, ...] = field(
        default=DEFAULT_DATETIME_COMPONENTS
    )

    # Thresholds used when a configuration is inferred from a dataset profile.
    max_categorical_cardinality: int = DEFAULT_MAX_CATEGORICAL_CARDINALITY
    max_classification_classes: int = DEFAULT_MAX_CLASSIFICATION_CLASSES

    # Splitting.
    test_size: float = DEFAULT_TEST_SIZE
    random_state: int = DEFAULT_RANDOM_STATE

    def __post_init__(self) -> None:
        """Normalise sequences to tuples and plain strings to enum members."""
        object.__setattr__(self, "numeric_columns", _as_tuple(self.numeric_columns))
        object.__setattr__(
            self, "categorical_columns", _as_tuple(self.categorical_columns)
        )
        object.__setattr__(self, "boolean_columns", _as_tuple(self.boolean_columns))
        object.__setattr__(self, "datetime_columns", _as_tuple(self.datetime_columns))
        object.__setattr__(
            self, "identifier_columns", _as_tuple(self.identifier_columns)
        )
        object.__setattr__(self, "excluded_columns", _as_tuple(self.excluded_columns))
        object.__setattr__(
            self,
            "datetime_components",
            tuple(
                _coerce(DatetimeComponent, component, "datetime_components")
                for component in self.datetime_components
            ),
        )
        object.__setattr__(
            self, "task_type", _coerce(TaskType, self.task_type, "task_type")
        )
        object.__setattr__(
            self,
            "scaling_strategy",
            _coerce(ScalingStrategy, self.scaling_strategy, "scaling_strategy"),
        )
        object.__setattr__(
            self,
            "numeric_imputation",
            _coerce(NumericImputation, self.numeric_imputation, "numeric_imputation"),
        )
        object.__setattr__(
            self,
            "categorical_imputation",
            _coerce(
                CategoricalImputation,
                self.categorical_imputation,
                "categorical_imputation",
            ),
        )

    @property
    def feature_columns(self) -> tuple[str, ...]:
        """Every column used as a feature, in a stable, predictable order."""
        return (
            *self.numeric_columns,
            *self.categorical_columns,
            *self.boolean_columns,
            *self.datetime_columns,
        )

    @property
    def feature_groups(self) -> dict[FeatureType, tuple[str, ...]]:
        """Feature columns keyed by the branch of the pipeline that handles them."""
        return {
            FeatureType.NUMERIC: self.numeric_columns,
            FeatureType.CATEGORICAL: self.categorical_columns,
            FeatureType.BOOLEAN: self.boolean_columns,
            FeatureType.DATETIME: self.datetime_columns,
        }

    def feature_type_of(self, column: str) -> FeatureType | None:
        """Return the pipeline branch handling ``column``, or ``None``."""
        for feature_type, columns in self.feature_groups.items():
            if column in columns:
                return feature_type
        return None

    def with_overrides(self, **overrides: Any) -> PreprocessingConfig:
        """Return a copy with explicit values replacing the inferred ones.

        This is how a caller overrides anything derived from a dataset profile,
        for example ``config.with_overrides(scaling_strategy="none")``.

        Args:
            **overrides: Any field of the configuration.

        Returns:
            PreprocessingConfig: A new, validated configuration.

        Raises:
            ConfigurationError: If a field name is unknown or a value invalid.
        """
        allowed = {item.name for item in self.__dataclass_fields__.values()}
        unknown = sorted(set(overrides) - allowed)
        if unknown:
            raise ConfigurationError(
                "Unknown configuration field(s): " + ", ".join(unknown) + ".",
                details={"unknown_fields": unknown},
            )
        return replace(self, **overrides)


def validate_config(config: PreprocessingConfig, columns: Sequence[str]) -> None:
    """Check a configuration against the columns of a dataset.

    The checks are ordered from most to least fundamental so the first error a
    caller sees is the most useful one.

    Args:
        config: The configuration to validate.
        columns: Column names present in the dataset.

    Raises:
        MissingTargetError: The target column is blank or not in the dataset.
        TargetLeakageError: The target was also assigned to a feature group.
        DuplicateColumnAssignmentError: A column is in two feature groups.
        UnknownColumnError: A configured column is not in the dataset.
        EmptyFeatureSetError: No feature columns remain.
        ConfigurationError: A split parameter is out of range.
    """
    available = list(columns)

    if not config.target_column or not config.target_column.strip():
        raise MissingTargetError("A target column must be configured.")
    if config.target_column not in available:
        raise MissingTargetError(
            f"Target column '{config.target_column}' is not in the dataset.",
            details={
                "target_column": config.target_column,
                "available_columns": available,
            },
        )

    if config.target_column in config.feature_columns:
        raise TargetLeakageError(
            f"Target column '{config.target_column}' is also configured as a "
            "feature. The target must never be preprocessed as a feature.",
            details={"target_column": config.target_column},
        )

    seen: set[str] = set()
    duplicates: list[str] = []
    for column in config.feature_columns:
        if column in seen and column not in duplicates:
            duplicates.append(column)
        seen.add(column)
    if duplicates:
        raise DuplicateColumnAssignmentError(
            "Column(s) assigned to more than one feature group: "
            + ", ".join(duplicates)
            + ".",
            details={"columns": duplicates},
        )

    configured = {
        *config.feature_columns,
        *config.identifier_columns,
        *config.excluded_columns,
    }
    missing = sorted(column for column in configured if column not in available)
    if missing:
        raise UnknownColumnError(
            "Configured column(s) not present in the dataset: "
            + ", ".join(missing)
            + ".",
            details={"columns": missing, "available_columns": available},
        )

    if not config.feature_columns:
        raise EmptyFeatureSetError(
            "The configuration selects no feature columns, so there is nothing "
            "to preprocess.",
            details={"target_column": config.target_column},
        )

    if not 0.0 < config.test_size < 1.0:
        raise ConfigurationError(
            f"test_size must be between 0 and 1 (exclusive), got {config.test_size}.",
            details={"test_size": config.test_size},
        )
