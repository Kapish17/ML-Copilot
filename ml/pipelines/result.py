"""The result of preparing a dataset for training.

``PreparedDataset`` deliberately mixes two kinds of thing: the model-ready
arrays and fitted preprocessor that a training step needs, and a plain summary
that a caller can serialise. :meth:`PreparedDataset.summary` is the boundary —
it returns only JSON-friendly values, so a fitted sklearn estimator or a
DataFrame never has to leak into an API response.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer

from ml.evaluation.splitting import TargetDistribution
from ml.features.config import PreprocessingConfig
from ml.features.decisions import ColumnDecision
from ml.features.types import TaskType


@dataclass(frozen=True)
class PreparedDataset:
    """Model-ready data plus a full account of how it was produced.

    Both the transformed features (``X_train``, ``X_test``) and the untouched
    source columns behind them (``X_train_raw``, ``X_test_raw``) are kept. The
    transformed frames are what an estimator consumes directly; the raw frames
    are what a full ``Pipeline(preprocessing, estimator)`` is fitted on, so a
    trained model can accept raw feature rows later instead of pre-transformed
    matrices.
    """

    config: PreprocessingConfig
    preprocessor: ColumnTransformer
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    X_train_raw: pd.DataFrame
    X_test_raw: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: tuple[str, ...]
    feature_groups: dict[str, tuple[str, ...]]
    selected_columns: tuple[str, ...]
    excluded_columns: tuple[str, ...]
    identifier_columns: tuple[str, ...]
    column_decisions: tuple[ColumnDecision, ...]
    task_type: TaskType
    train_row_count: int
    test_row_count: int
    rows_dropped_missing_target: int
    stratified: bool
    stratification_note: str | None
    target_overall: TargetDistribution
    target_train: TargetDistribution
    target_test: TargetDistribution

    @property
    def feature_count(self) -> int:
        """Number of features after encoding and expansion."""
        return len(self.feature_names)

    def summary(self) -> dict[str, Any]:
        """Return a serialisable description of the prepared dataset.

        The fitted preprocessor and the data frames are intentionally left out:
        they are ML-internal objects, and anything crossing an API boundary
        should be built from this dictionary instead.
        """
        return {
            "target_column": self.config.target_column,
            "task_type": self.task_type.value,
            "feature_count": self.feature_count,
            "feature_names": list(self.feature_names),
            "feature_groups": {
                group: list(columns) for group, columns in self.feature_groups.items()
            },
            "selected_columns": list(self.selected_columns),
            "excluded_columns": list(self.excluded_columns),
            "identifier_columns": list(self.identifier_columns),
            "column_decisions": [
                decision.as_dict() for decision in self.column_decisions
            ],
            "split": {
                "train_row_count": self.train_row_count,
                "test_row_count": self.test_row_count,
                "test_size": self.config.test_size,
                "random_state": self.config.random_state,
                "stratified": self.stratified,
                "stratification_note": self.stratification_note,
                "rows_dropped_missing_target": self.rows_dropped_missing_target,
            },
            "target_distribution": {
                "overall": self.target_overall.as_dict(),
                "train": self.target_train.as_dict(),
                "test": self.target_test.as_dict(),
            },
            "preprocessing": {
                "scaling_strategy": self.config.scaling_strategy.value,
                "numeric_imputation": self.config.numeric_imputation.value,
                "categorical_imputation": self.config.categorical_imputation.value,
                "add_missing_indicators": self.config.add_missing_indicators,
                "datetime_components": [
                    component.value for component in self.config.datetime_components
                ],
            },
        }
