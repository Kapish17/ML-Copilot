"""Training one model on a prepared dataset.

The trained artefact is a full ``Pipeline(preprocessing, estimator)``:

    raw feature rows -> preprocessing -> estimator -> prediction

The preprocessing step is a fresh copy of the one the preprocessing layer
configured, fitted here on the training rows alone and carried inside the
trained model. That is what lets a caller hand the finished model raw columns
later — a transformed matrix is never the only artefact.

Leakage discipline is inherited from the split, not re-established here: the
pipeline is fitted on ``X_train_raw`` only, and the test rows are seen for the
first time when predictions are made.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np
from sklearn.pipeline import Pipeline

from ml.errors import InsufficientDataError, ModelTrainingError
from ml.evaluation.metrics import evaluate_predictions, resolve_primary_metric
from ml.features.types import TaskType
from ml.models.baselines import BaselineResult, compare_to_baseline, evaluate_baseline
from ml.models.registry import ModelRegistry, default_registry
from ml.models.result import MODEL_STEP, PREPROCESSING_STEP, DatasetInfo, TrainedModel
from ml.models.spec import ModelSpec, build_estimator, get_model_spec, validate_spec
from ml.pipelines.preprocessing import clone_preprocessor
from ml.pipelines.result import PreparedDataset


def build_pipeline(prepared: PreparedDataset, estimator) -> Pipeline:
    """Wrap an estimator behind the dataset's preprocessing.

    The preprocessing step is unfitted: it learns its statistics when the
    pipeline is fitted, from the training rows it is given.

    Args:
        prepared: The dataset produced by the preprocessing layer.
        estimator: An unfitted estimator.

    Returns:
        sklearn.pipeline.Pipeline: A pipeline that takes raw feature rows.
    """
    return Pipeline(
        [
            (PREPROCESSING_STEP, clone_preprocessor(prepared.preprocessor)),
            (MODEL_STEP, estimator),
        ]
    )


def _class_scores(
    pipeline: Pipeline, features, task_type: TaskType
) -> tuple[np.ndarray | None, list | None]:
    """Return predicted probabilities and their class order, when available.

    Models without ``predict_proba`` simply report no scores, and ROC-AUC is
    then marked unavailable rather than being approximated.
    """
    if task_type is not TaskType.CLASSIFICATION:
        return None, None
    if not hasattr(pipeline, "predict_proba"):
        return None, None
    try:
        return pipeline.predict_proba(features), list(pipeline.classes_)
    except (AttributeError, NotImplementedError):  # pragma: no cover - defensive
        return None, None


def _dataset_info(prepared: PreparedDataset) -> DatasetInfo:
    """Summarise the dataset a model was trained on."""
    return DatasetInfo(
        target_column=prepared.config.target_column,
        task_type=prepared.task_type,
        train_row_count=prepared.train_row_count,
        test_row_count=prepared.test_row_count,
        raw_feature_columns=prepared.config.feature_columns,
        transformed_feature_count=len(prepared.feature_names),
        stratified=prepared.stratified,
    )


def _require_usable_split(prepared: PreparedDataset) -> None:
    """Refuse to train when either side of the split is empty."""
    if prepared.train_row_count == 0 or prepared.test_row_count == 0:
        raise InsufficientDataError(
            "Training needs a non-empty training set and a non-empty test set; "
            f"got {prepared.train_row_count} training and "
            f"{prepared.test_row_count} test rows.",
            details={
                "train_row_count": prepared.train_row_count,
                "test_row_count": prepared.test_row_count,
            },
        )


def train_model(
    prepared: PreparedDataset,
    spec: ModelSpec | str,
    *,
    registry: ModelRegistry | None = None,
    baseline: BaselineResult | None = None,
) -> TrainedModel:
    """Train one model on a prepared dataset and evaluate it.

    The steps are: validate the request against the registry and the dataset's
    task, build the estimator, wrap it behind the preprocessing, fit on the
    training rows, predict the untouched test rows, score them, and compare
    against the naive baseline.

    Args:
        prepared: The dataset produced by the preprocessing layer.
        spec: A model specification, or a registry identifier for the defaults.
        registry: Registry to resolve the model in; the default when omitted.
        baseline: A baseline already evaluated for this dataset. Passing one
            avoids re-fitting it for every model in a comparison run.

    Returns:
        TrainedModel: The fitted pipeline, its scores and their context.

    Raises:
        UnknownModelError: The model is not in the registry.
        IncompatibleTaskError: The model does not solve the dataset's task.
        InvalidHyperparameterError: A hyperparameter is not accepted.
        InvalidMetricError: The requested primary metric does not exist.
        InsufficientDataError: Either side of the split is empty.
        ModelTrainingError: The estimator failed while fitting or predicting.
    """
    active = registry or default_registry()
    resolved_spec = (
        get_model_spec(spec, registry=active) if isinstance(spec, str) else spec
    )
    definition = validate_spec(resolved_spec, prepared.task_type, registry=active)
    _require_usable_split(prepared)

    primary_metric = resolve_primary_metric(
        prepared.task_type, resolved_spec.primary_metric
    )
    estimator = build_estimator(
        definition,
        resolved_spec,
        fallback_random_state=prepared.config.random_state,
    )
    pipeline = build_pipeline(prepared, estimator)

    started = perf_counter()
    try:
        pipeline.fit(prepared.X_train_raw, prepared.y_train)
    except Exception as exc:  # noqa: BLE001 - any estimator failure is reported alike
        raise ModelTrainingError(
            f"Model '{definition.identifier}' failed while training: {exc}",
            details={
                "model_name": definition.identifier,
                "error_type": type(exc).__name__,
            },
        ) from exc
    training_seconds = perf_counter() - started

    try:
        predictions = pipeline.predict(prepared.X_test_raw)
        scores, score_labels = _class_scores(
            pipeline, prepared.X_test_raw, prepared.task_type
        )
    except Exception as exc:  # noqa: BLE001 - as above
        raise ModelTrainingError(
            f"Model '{definition.identifier}' failed while predicting: {exc}",
            details={
                "model_name": definition.identifier,
                "error_type": type(exc).__name__,
            },
        ) from exc

    metrics = evaluate_predictions(
        prepared.task_type,
        prepared.y_test,
        predictions,
        y_score=scores,
        score_labels=score_labels,
    )
    reference = baseline if baseline is not None else evaluate_baseline(prepared)

    return TrainedModel(
        spec=resolved_spec,
        display_name=definition.display_name,
        task_type=prepared.task_type,
        pipeline=pipeline,
        metrics=metrics,
        baseline=reference,
        baseline_comparison=compare_to_baseline(
            primary_metric,
            metrics.get(primary_metric.key),
            reference.metrics.get(primary_metric.key),
        ),
        primary_metric=primary_metric,
        feature_names=prepared.feature_names,
        dataset=_dataset_info(prepared),
        training_seconds=training_seconds,
    )
