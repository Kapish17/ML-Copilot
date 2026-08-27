"""Construction of the sklearn preprocessing pipeline.

The pipeline is a plain ``ColumnTransformer`` holding one branch per feature
group. Using sklearn's own abstraction rather than a bespoke framework is a
deliberate choice: it gives correct fit/transform semantics for free, which is
what keeps test data out of every learned statistic, and it means the whole
preprocessor can later be stored alongside a trained model.

Nothing here touches the target column — only the columns listed in the
configuration's feature groups are ever passed in.
"""

from __future__ import annotations

import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
)

from ml.features.config import PreprocessingConfig
from ml.features.transformers import DatetimeComponentExtractor, to_float_frame
from ml.features.types import (
    CategoricalImputation,
    FeatureType,
    NumericImputation,
    ScalingStrategy,
)

PASSTHROUGH = "passthrough"

#: Imputation applied to the calendar components extracted from datetimes.
DATETIME_COMPONENT_IMPUTATION = "median"


def _build_scaler(strategy: ScalingStrategy) -> object:
    """Return the scaler for a strategy, or a passthrough step.

    Scaling is skipped for one-hot, boolean and datetime-component features:
    those are already on a comparable, bounded scale, and rescaling them only
    makes the resulting feature values harder to read.
    """
    if strategy is ScalingStrategy.STANDARD:
        return StandardScaler()
    if strategy is ScalingStrategy.MINMAX:
        return MinMaxScaler()
    return PASSTHROUGH


def _build_numeric_imputer(config: PreprocessingConfig) -> SimpleImputer:
    """Return the imputer for numeric features."""
    if config.numeric_imputation is NumericImputation.CONSTANT:
        return SimpleImputer(
            strategy="constant", fill_value=config.numeric_fill_value
        )
    return SimpleImputer(strategy=config.numeric_imputation.value)


def _missing_indicator_branch() -> Pipeline:
    """Flag which values were missing, as 0/1 floats.

    ``features="missing-only"`` means the indicator columns are decided at fit
    time from the training data alone: a column that had no gaps in training
    produces no indicator, even if the test set has gaps later.
    ``error_on_new=False`` is what allows that case to pass through quietly
    instead of raising, which matters at inference time.
    """
    return Pipeline(
        [
            ("flag", MissingIndicator(features="missing-only", error_on_new=False)),
            ("cast", FunctionTransformer(to_float_frame, feature_names_out="one-to-one")),
        ]
    )


def _with_missing_indicators(
    values: Pipeline, config: PreprocessingConfig
) -> Pipeline | FeatureUnion:
    """Append missing indicators beside transformed values, when enabled.

    The indicators sit outside the scaling branch on purpose: they are already
    0/1, and rescaling them would only make the output harder to read.
    """
    if not config.add_missing_indicators:
        return values
    return FeatureUnion(
        [("values", values), ("missing", _missing_indicator_branch())],
        verbose_feature_names_out=False,
    )


def _build_categorical_imputer(config: PreprocessingConfig) -> SimpleImputer:
    """Return the imputer for categorical features.

    With the constant strategy, missing values become their own category and
    the one-hot encoder keeps that information as a dedicated column, so no
    separate missing indicator is needed on this branch.
    """
    if config.categorical_imputation is CategoricalImputation.CONSTANT:
        return SimpleImputer(
            strategy="constant", fill_value=config.categorical_fill_value
        )
    return SimpleImputer(strategy=CategoricalImputation.MOST_FREQUENT.value)


def build_numeric_pipeline(config: PreprocessingConfig) -> Pipeline | FeatureUnion:
    """Impute missing numbers, scale them, and flag what was missing."""
    values = Pipeline(
        [
            ("impute", _build_numeric_imputer(config)),
            ("scale", _build_scaler(config.scaling_strategy)),
        ]
    )
    return _with_missing_indicators(values, config)


def build_categorical_pipeline(config: PreprocessingConfig) -> Pipeline:
    """Impute missing categories, then one-hot encode them.

    ``handle_unknown="ignore"`` is what makes the fitted pipeline safe at
    inference time: a category never seen during training produces all-zero
    indicators instead of an error.
    """
    return Pipeline(
        [
            ("impute", _build_categorical_imputer(config)),
            (
                "encode",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    dtype="float64",
                ),
            ),
        ]
    )


def build_boolean_pipeline(config: PreprocessingConfig) -> Pipeline:
    """Cast booleans to 0/1 floats, then impute with the most frequent value.

    Booleans are not scaled: they already live on the same 0/1 scale as the
    one-hot features they sit beside.
    """
    values = Pipeline(
        [("impute", SimpleImputer(strategy=CategoricalImputation.MOST_FREQUENT.value))]
    )
    return Pipeline(
        [
            (
                "cast",
                FunctionTransformer(to_float_frame, feature_names_out="one-to-one"),
            ),
            ("features", _with_missing_indicators(values, config)),
        ]
    )


def build_datetime_pipeline(config: PreprocessingConfig) -> Pipeline:
    """Expand datetimes into calendar components, then impute the gaps."""
    return Pipeline(
        [
            ("components", DatetimeComponentExtractor(config.datetime_components)),
            ("impute", SimpleImputer(strategy=DATETIME_COMPONENT_IMPUTATION)),
        ]
    )


#: Builder for each feature group, in the order branches are applied.
_BRANCH_BUILDERS = {
    FeatureType.NUMERIC: build_numeric_pipeline,
    FeatureType.CATEGORICAL: build_categorical_pipeline,
    FeatureType.BOOLEAN: build_boolean_pipeline,
    FeatureType.DATETIME: build_datetime_pipeline,
}


def build_preprocessor(config: PreprocessingConfig) -> ColumnTransformer:
    """Assemble the preprocessing pipeline described by a configuration.

    Only non-empty feature groups become branches, so a purely numeric dataset
    produces a pipeline with a single branch rather than empty placeholders.
    Columns outside the feature groups are dropped, which means identifiers,
    excluded columns and the target can never reach a transformer.

    Args:
        config: The resolved preprocessing configuration.

    Returns:
        sklearn.compose.ColumnTransformer: An unfitted preprocessor whose
        output is a ``pandas.DataFrame`` with meaningful column names.
    """
    branches = [
        (feature_type.value, _BRANCH_BUILDERS[feature_type](config), list(columns))
        for feature_type, columns in config.feature_groups.items()
        if columns
    ]
    preprocessor = ColumnTransformer(
        transformers=branches,
        remainder="drop",
        verbose_feature_names_out=False,
    )
    preprocessor.set_output(transform="pandas")
    return preprocessor


def clone_preprocessor(preprocessor: ColumnTransformer) -> ColumnTransformer:
    """Return an unfitted copy of a preprocessor with the same configuration.

    ``sklearn.base.clone`` copies constructor parameters but not the output
    configuration set by ``set_output``, so that is reapplied here. Used when a
    full ``Pipeline(preprocessing, estimator)`` needs its own preprocessing
    step to fit on the training rows.
    """
    copy = clone(preprocessor)
    copy.set_output(transform="pandas")
    return copy


def feature_names_of(preprocessor: ColumnTransformer) -> tuple[str, ...]:
    """Return the output feature names of a fitted preprocessor.

    Names survive every transformation: a scaled numeric column keeps its own
    name, a one-hot column becomes ``<column>_<category>``, a datetime becomes
    ``<column>_<component>`` and a missing indicator becomes
    ``missingindicator_<column>``. Later commits rely on this for feature
    importance and explanation.
    """
    return tuple(str(name) for name in preprocessor.get_feature_names_out())


def transform_frame(
    preprocessor: ColumnTransformer, frame: pd.DataFrame
) -> pd.DataFrame:
    """Apply a fitted preprocessor and guarantee a named DataFrame result."""
    transformed = preprocessor.transform(frame)
    if isinstance(transformed, pd.DataFrame):
        return transformed
    return pd.DataFrame(  # pragma: no cover - pandas output is configured above
        transformed, columns=list(feature_names_of(preprocessor)), index=frame.index
    )
