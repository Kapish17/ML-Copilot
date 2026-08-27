"""The result of training one model.

``TrainedModel`` holds both halves of the outcome: the fitted sklearn pipeline
that makes predictions, and a plain description of how it scored.
:meth:`TrainedModel.summary` is the boundary between them — it returns only
JSON-friendly values and never touches the pipeline, so a fitted estimator has
no way of reaching an API response.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline

from ml.evaluation.metrics import EvaluationMetrics, MetricDefinition
from ml.features.types import TaskType
from ml.models.baselines import BaselineComparison, BaselineResult
from ml.models.spec import ModelSpec

PREPROCESSING_STEP = "preprocessing"
MODEL_STEP = "model"


@dataclass(frozen=True)
class DatasetInfo:
    """What the model was trained on, in numbers."""

    target_column: str
    task_type: TaskType
    train_row_count: int
    test_row_count: int
    raw_feature_columns: tuple[str, ...]
    transformed_feature_count: int
    stratified: bool

    def as_dict(self) -> dict[str, Any]:
        """Render the dataset information as plain, JSON-friendly values."""
        return {
            "target_column": self.target_column,
            "task_type": self.task_type.value,
            "train_row_count": self.train_row_count,
            "test_row_count": self.test_row_count,
            "raw_feature_columns": list(self.raw_feature_columns),
            "transformed_feature_count": self.transformed_feature_count,
            "stratified": self.stratified,
        }


@dataclass(frozen=True)
class TrainedModel:
    """A fitted model, its scores, and the context needed to read them."""

    spec: ModelSpec
    display_name: str
    task_type: TaskType
    pipeline: Pipeline
    metrics: EvaluationMetrics
    baseline: BaselineResult
    baseline_comparison: BaselineComparison
    primary_metric: MetricDefinition
    feature_names: tuple[str, ...]
    dataset: DatasetInfo
    training_seconds: float

    @property
    def model_name(self) -> str:
        """The registry identifier of the trained estimator."""
        return self.spec.model_name

    @property
    def primary_metric_value(self) -> float | None:
        """The score models are ranked by, or ``None`` if unavailable."""
        return self.metrics.get(self.primary_metric.key)

    @property
    def estimator(self):
        """The fitted estimator, without the preprocessing around it."""
        return self.pipeline.named_steps[MODEL_STEP]

    @property
    def preprocessor(self):
        """The preprocessing step fitted inside this model's pipeline."""
        return self.pipeline.named_steps[PREPROCESSING_STEP]

    def predict(self, features: pd.DataFrame):
        """Predict from **raw** feature rows.

        The same preprocessing that was fitted during training is applied
        first, so callers pass the original columns rather than a transformed
        matrix.

        Args:
            features: Raw feature columns, as they appear in the source dataset.

        Returns:
            numpy.ndarray: One prediction per row.
        """
        return self.pipeline.predict(features)

    def predict_proba(self, features: pd.DataFrame):
        """Predict class probabilities from raw feature rows.

        Raises:
            AttributeError: If the underlying estimator has no ``predict_proba``.
        """
        return self.pipeline.predict_proba(features)

    def summary(self) -> dict[str, Any]:
        """Return a serialisable description of the trained model.

        The fitted pipeline is deliberately omitted: it is an ML-internal
        object, and anything crossing an API boundary should be built from this
        dictionary instead.
        """
        return {
            "model_name": self.model_name,
            "display_name": self.display_name,
            "task_type": self.task_type.value,
            "spec": self.spec.as_dict(),
            "primary_metric": {
                "key": self.primary_metric.key,
                "display_name": self.primary_metric.display_name,
                "direction": self.primary_metric.direction.value,
                "value": self.primary_metric_value,
            },
            "metrics": self.metrics.as_dict(),
            "baseline": self.baseline.as_dict(),
            "baseline_comparison": self.baseline_comparison.as_dict(),
            "training_seconds": self.training_seconds,
            "feature_count": len(self.feature_names),
            "feature_names": list(self.feature_names),
            "dataset": self.dataset.as_dict(),
        }
