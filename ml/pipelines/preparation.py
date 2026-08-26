"""Turning a standardised DataFrame into model-ready training and test data.

This module is the entry point of the ML layer. It accepts a
``pandas.DataFrame`` — never a file, a path or an upload — so any future
ingestion adapter (Excel, JSON, Parquet, SQL, an API) can feed exactly the same
pipeline once it produces a DataFrame.

The order of operations is what prevents leakage:

1. validate the configuration against the dataset,
2. separate the target from the features,
3. split into train and test,
4. **fit** the preprocessor on the training features only,
5. **transform** the test features with those already-learned statistics.

No step ever sees the test set before it is fitted, and the target is never
passed to a feature transformer.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from ml.errors import InsufficientDataError, MLError
from ml.evaluation.splitting import (
    DatasetSplit,
    describe_target,
    resolve_task_type,
    split_dataset,
)
from ml.features.config import (
    MIN_ROWS_FOR_SPLIT,
    PreprocessingConfig,
    validate_config,
)
from ml.features.decisions import ColumnDecision
from ml.features.types import ColumnRole
from ml.pipelines.preprocessing import (
    build_preprocessor,
    feature_names_of,
    transform_frame,
)
from ml.pipelines.result import PreparedDataset


def _decisions_from_config(
    config: PreprocessingConfig, columns: Sequence[str]
) -> tuple[ColumnDecision, ...]:
    """Describe every dataset column from the configuration alone.

    Used when the caller built a configuration by hand rather than inferring it
    from a profile, so a prepared dataset always explains what happened to each
    column even without profiling context.
    """
    decisions: list[ColumnDecision] = []
    for column in columns:
        if column == config.target_column:
            decisions.append(
                ColumnDecision(
                    column=column,
                    role=ColumnRole.TARGET,
                    reason="Declared as the target column.",
                )
            )
            continue

        feature_type = config.feature_type_of(column)
        if feature_type is not None:
            decisions.append(
                ColumnDecision(
                    column=column,
                    role=ColumnRole.FEATURE,
                    reason=(
                        f"Configured as a {feature_type.value} feature; handled by "
                        f"the {feature_type.value} branch."
                    ),
                    feature_type=feature_type,
                )
            )
            continue

        if column in config.identifier_columns:
            role, reason = ColumnRole.IDENTIFIER, "Marked as an identifier."
        elif column in config.excluded_columns:
            role, reason = ColumnRole.EXCLUDED, "Excluded explicitly by the caller."
        else:
            role, reason = (
                ColumnRole.EXCLUDED,
                "Not assigned to any feature group by the configuration.",
            )
        decisions.append(ColumnDecision(column=column, role=role, reason=reason))
    return tuple(decisions)


def separate_target(
    frame: pd.DataFrame, config: PreprocessingConfig
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a dataset into feature columns and the target column.

    The feature frame is built by selecting the configured feature columns
    rather than by dropping the target, so the target cannot reach a
    transformer even if the configuration is later extended.

    Args:
        frame: The full dataset.
        config: A configuration already validated against ``frame``.

    Returns:
        tuple[pandas.DataFrame, pandas.Series]: Features and target.
    """
    features = frame.loc[:, list(config.feature_columns)].copy()
    target = frame[config.target_column].copy()
    return features, target


def _drop_missing_target(
    frame: pd.DataFrame, target_column: str
) -> tuple[pd.DataFrame, int]:
    """Remove rows whose target is missing and report how many were removed.

    A supervised model cannot learn from a row with no label. The count is
    returned and surfaced in the result, so the removal is visible rather than
    silent.
    """
    missing = frame[target_column].isna()
    dropped = int(missing.sum())
    return (frame.loc[~missing] if dropped else frame), dropped


def prepare_dataset(
    frame: pd.DataFrame,
    config: PreprocessingConfig,
    *,
    decisions: Sequence[ColumnDecision] = (),
) -> PreparedDataset:
    """Prepare a dataset for model training.

    Args:
        frame: A standardised dataset. Any ingestion format is acceptable as
            long as it has already been turned into a DataFrame.
        config: The preprocessing configuration to apply.
        decisions: Optional per-column reasoning, as produced by
            :func:`ml.features.inference.infer_configuration`. When omitted,
            decisions are derived from the configuration itself.

    Returns:
        PreparedDataset: Transformed train and test features, the untouched
        target arrays, the fitted preprocessor and a full account of the run.

    Raises:
        MLError: If ``frame`` is not a DataFrame.
        ConfigurationError: If the configuration does not fit the dataset.
        InsufficientDataError: If too few labelled rows remain to split.
    """
    if not isinstance(frame, pd.DataFrame):
        raise MLError(
            "Preprocessing expects a pandas DataFrame. Convert the source data "
            "with an ingestion adapter before calling the ML layer.",
            details={"received_type": type(frame).__name__},
        )

    validate_config(config, list(frame.columns))

    labelled, dropped = _drop_missing_target(frame, config.target_column)
    if labelled.shape[0] < MIN_ROWS_FOR_SPLIT:
        raise InsufficientDataError(
            f"Only {labelled.shape[0]} row(s) have a target value; at least "
            f"{MIN_ROWS_FOR_SPLIT} are needed to build a train/test split.",
            details={
                "labelled_row_count": int(labelled.shape[0]),
                "rows_dropped_missing_target": dropped,
                "minimum_rows": MIN_ROWS_FOR_SPLIT,
            },
        )

    features, target = separate_target(labelled, config)
    task_type = resolve_task_type(
        target, config.task_type, config.max_classification_classes
    )

    split: DatasetSplit = split_dataset(
        features,
        target,
        test_size=config.test_size,
        random_state=config.random_state,
        task_type=task_type,
    )

    preprocessor = build_preprocessor(config)
    # Fitted on the training features alone: no test row and no target value
    # contributes to any imputation, scaling or encoding statistic.
    X_train = preprocessor.fit_transform(split.X_train)
    X_test = transform_frame(preprocessor, split.X_test)

    return PreparedDataset(
        config=config,
        preprocessor=preprocessor,
        X_train=X_train,
        X_test=X_test,
        y_train=split.y_train,
        y_test=split.y_test,
        feature_names=feature_names_of(preprocessor),
        feature_groups={
            feature_type.value: columns
            for feature_type, columns in config.feature_groups.items()
            if columns
        },
        selected_columns=config.feature_columns,
        excluded_columns=config.excluded_columns,
        identifier_columns=config.identifier_columns,
        column_decisions=tuple(decisions)
        or _decisions_from_config(config, list(frame.columns)),
        task_type=task_type,
        train_row_count=int(split.X_train.shape[0]),
        test_row_count=int(split.X_test.shape[0]),
        rows_dropped_missing_target=dropped,
        stratified=split.stratified,
        stratification_note=split.stratification_note,
        target_overall=describe_target(target, task_type),
        target_train=describe_target(split.y_train, task_type),
        target_test=describe_target(split.y_test, task_type),
    )
