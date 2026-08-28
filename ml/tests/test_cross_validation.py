"""Tests for cross-validation over the training data."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold

from ml.errors import IncompatibleTaskError, InvalidFoldCountError, UnknownModelError
from ml.evaluation.cross_validation import (
    DEFAULT_FOLDS,
    FoldStatus,
    build_splitter,
    cross_validate_model,
    cross_validate_pipeline,
    summarise_metric,
    validate_fold_count,
)
from ml.evaluation.metrics import MetricDirection
from ml.features.config import PreprocessingConfig
from ml.features.types import TaskType
from ml.models.training import build_pipeline, clone_pipeline
from ml.pipelines.preparation import prepare_dataset
from ml.pipelines.result import PreparedDataset
from ml.tests.factories import rare_class_frame

SEED = 42


class _AlwaysFails(BaseEstimator, ClassifierMixin):
    """An estimator that raises when fitted, to exercise fold error handling."""

    def fit(self, X, y=None):  # noqa: ANN001, ANN201 - sklearn signature
        """Fail loudly, as a broken estimator would."""
        raise RuntimeError("this fold's estimator failed")

    def predict(self, X):  # noqa: ANN001, ANN201 - sklearn signature
        """Never reached."""
        raise RuntimeError("this fold's estimator failed")


# --------------------------------------------------------------------------
# Splitters
# --------------------------------------------------------------------------


def test_classification_uses_stratified_kfold() -> None:
    """Classification folds keep the class proportions."""
    splitter = build_splitter(TaskType.CLASSIFICATION, folds=5, random_state=SEED)

    assert isinstance(splitter, StratifiedKFold)
    assert splitter.n_splits == 5
    assert splitter.shuffle is True
    assert splitter.random_state == SEED


def test_regression_uses_plain_kfold() -> None:
    """A continuous target has no classes to balance."""
    splitter = build_splitter(TaskType.REGRESSION, folds=4, random_state=SEED)

    assert isinstance(splitter, KFold)
    assert not isinstance(splitter, StratifiedKFold)
    assert splitter.n_splits == 4
    assert splitter.shuffle is True


def test_stratification_keeps_the_minority_class_in_every_fold(
    imbalanced_prepared: PreparedDataset,
) -> None:
    """Every validation fold contains the rare class, which is the point."""
    target = imbalanced_prepared.y_train
    splitter = build_splitter(TaskType.CLASSIFICATION, folds=5, random_state=SEED)

    for _, validation_index in splitter.split(
        imbalanced_prepared.X_train_raw, target
    ):
        assert "minority" in set(target.iloc[validation_index])


# --------------------------------------------------------------------------
# Fold-count validation
# --------------------------------------------------------------------------


def test_fewer_than_two_folds_is_rejected(
    classification_prepared: PreparedDataset,
) -> None:
    """One fold is not cross-validation."""
    with pytest.raises(InvalidFoldCountError) as exc_info:
        validate_fold_count(
            classification_prepared.y_train, task_type=TaskType.CLASSIFICATION, folds=1
        )
    assert exc_info.value.details["reason"] == "too_few_folds"


def test_more_folds_than_rows_is_rejected() -> None:
    """A dataset cannot be divided into more parts than it has rows."""
    with pytest.raises(InvalidFoldCountError) as exc_info:
        validate_fold_count(
            pd.Series([1.0, 2.0, 3.0]), task_type=TaskType.REGRESSION, folds=5
        )
    assert exc_info.value.details["reason"] == "more_folds_than_rows"


def test_a_class_smaller_than_the_fold_count_is_rejected() -> None:
    """A rare class fails clearly rather than producing misleading folds."""
    frame = rare_class_frame(rows=40, rare_count=3)
    config = PreprocessingConfig(
        target_column="outcome",
        numeric_columns=("measure", "noise"),
        task_type="classification",
    )
    prepared = prepare_dataset(frame, config)

    with pytest.raises(InvalidFoldCountError) as exc_info:
        cross_validate_model(prepared, "logistic_regression", folds=5)

    details = exc_info.value.details
    assert details["reason"] == "class_smaller_than_folds"
    assert details["smallest_class"] == "rare"
    assert details["smallest_class_count"] < 5
    assert "rare" in str(exc_info.value)


def test_a_workable_minority_class_is_accepted(
    imbalanced_prepared: PreparedDataset,
) -> None:
    """Imbalance alone is fine when every class has enough rows."""
    result = cross_validate_model(imbalanced_prepared, "logistic_regression", folds=5)

    assert len(result.successful_folds) == 5
    assert result.mean_primary_metric is not None


# --------------------------------------------------------------------------
# Fold mechanics
# --------------------------------------------------------------------------


def test_cross_validation_runs_the_requested_number_of_folds(
    classification_prepared: PreparedDataset,
) -> None:
    """Every fold is run, and together they cover the training rows once."""
    result = cross_validate_model(classification_prepared, "logistic_regression", folds=4)

    assert result.folds == 4
    assert len(result.fold_results) == 4
    assert [fold.fold for fold in result.fold_results] == [1, 2, 3, 4]
    assert sum(fold.validation_size for fold in result.fold_results) == (
        classification_prepared.train_row_count
    )


def test_default_fold_count_is_five(classification_prepared: PreparedDataset) -> None:
    """Five folds unless the caller says otherwise."""
    result = cross_validate_model(classification_prepared, "logistic_regression")

    assert DEFAULT_FOLDS == 5
    assert result.folds == 5


def test_each_fold_reports_the_full_metric_set(
    classification_prepared: PreparedDataset,
) -> None:
    """Fold-level metrics reuse the existing metric definitions."""
    result = cross_validate_model(classification_prepared, "logistic_regression", folds=3)

    for fold in result.successful_folds:
        assert fold.metrics is not None
        assert set(fold.metrics.values) == {
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
        }
        assert fold.metrics.classification is not None
        assert fold.training_seconds is not None and fold.training_seconds >= 0


def test_regression_folds_report_regression_metrics(
    regression_prepared: PreparedDataset,
) -> None:
    """Cross-validation works the same way for a continuous target."""
    result = cross_validate_model(regression_prepared, "linear_regression", folds=4)

    assert result.task_type is TaskType.REGRESSION
    assert len(result.successful_folds) == 4
    for fold in result.successful_folds:
        assert fold.metrics is not None
        assert set(fold.metrics.values) == {"mae", "mse", "rmse", "r2"}
    assert result.confusion_matrix is None


def test_mean_and_standard_deviation_match_the_fold_values(
    classification_prepared: PreparedDataset,
) -> None:
    """The aggregate is exactly the arithmetic over the folds that ran."""
    result = cross_validate_model(classification_prepared, "logistic_regression", folds=5)
    values = [fold.metrics.get("f1") for fold in result.successful_folds]

    assert result.mean_primary_metric == pytest.approx(float(np.mean(values)))
    assert result.std_primary_metric == pytest.approx(float(np.std(values)))
    summary = result.aggregates["f1"]
    assert summary.values == tuple(values)
    assert summary.minimum == pytest.approx(min(values))
    assert summary.maximum == pytest.approx(max(values))
    assert summary.fold_count == 5


def test_summarise_metric_handles_an_empty_set() -> None:
    """A metric no fold produced summarises to nothing, not to zero."""
    summary = summarise_metric("f1", [])

    assert summary.mean is None
    assert summary.std is None
    assert summary.fold_count == 0


def test_pooled_confusion_matrix_covers_the_training_rows(
    classification_prepared: PreparedDataset,
) -> None:
    """Each training row is validated exactly once, so the matrix sums to it."""
    result = cross_validate_model(classification_prepared, "logistic_regression", folds=5)

    assert result.confusion_matrix is not None
    total = sum(sum(row) for row in result.confusion_matrix)
    assert total == classification_prepared.train_row_count
    assert result.class_labels == ("no", "yes")


def test_primary_metric_direction_comes_from_the_shared_definition(
    classification_prepared: PreparedDataset, regression_prepared: PreparedDataset
) -> None:
    """There is one source of truth for whether higher is better."""
    classification = cross_validate_model(
        classification_prepared, "logistic_regression", folds=3
    )
    regression = cross_validate_model(regression_prepared, "linear_regression", folds=3)

    assert classification.primary_metric.key == "f1"
    assert classification.primary_metric.direction is MetricDirection.HIGHER_IS_BETTER
    assert regression.primary_metric.key == "rmse"
    assert regression.primary_metric.direction is MetricDirection.LOWER_IS_BETTER


def test_primary_metric_can_be_overridden(
    regression_prepared: PreparedDataset,
) -> None:
    """Ranking does not have to use the task default."""
    result = cross_validate_model(
        regression_prepared, "linear_regression", folds=3, primary_metric="r2"
    )

    assert result.primary_metric.key == "r2"
    assert result.mean_primary_metric == pytest.approx(
        result.aggregates["r2"].mean
    )


# --------------------------------------------------------------------------
# Reproducibility and failures
# --------------------------------------------------------------------------


def test_cross_validation_is_reproducible(
    classification_prepared: PreparedDataset,
) -> None:
    """The same dataset, model, folds and seed give the same fold scores."""
    first = cross_validate_model(classification_prepared, "random_forest_classifier", folds=4)
    second = cross_validate_model(classification_prepared, "random_forest_classifier", folds=4)

    assert [fold.metrics.values for fold in first.successful_folds] == [
        fold.metrics.values for fold in second.successful_folds
    ]
    assert first.mean_primary_metric == second.mean_primary_metric
    assert first.std_primary_metric == second.std_primary_metric


def test_a_failing_fold_does_not_stop_the_others(
    classification_prepared: PreparedDataset,
) -> None:
    """One broken fold is recorded; the run continues and still aggregates."""
    calls = {"count": 0}

    def factory():
        """Return a working pipeline except on the second fold."""
        calls["count"] += 1
        estimator = (
            _AlwaysFails() if calls["count"] == 2 else LogisticRegression(max_iter=1000)
        )
        return clone_pipeline(build_pipeline(classification_prepared, estimator))

    fold_results, seconds = cross_validate_pipeline(
        factory,
        classification_prepared.X_train_raw,
        classification_prepared.y_train,
        task_type=TaskType.CLASSIFICATION,
        folds=3,
        random_state=SEED,
    )

    statuses = [fold.status for fold in fold_results]
    assert statuses == [FoldStatus.SUCCEEDED, FoldStatus.FAILED, FoldStatus.SUCCEEDED]
    failed = fold_results[1]
    assert failed.error is not None and "failed" in failed.error
    assert failed.error_type == "RuntimeError"
    assert failed.metrics is None
    assert seconds >= 0


def test_a_run_where_every_fold_fails_is_reported_not_raised(
    classification_prepared: PreparedDataset,
) -> None:
    """The errors are recorded so a comparison can report them."""

    def factory():
        """Always return a broken pipeline."""
        return clone_pipeline(
            build_pipeline(classification_prepared, _AlwaysFails())
        )

    fold_results, _ = cross_validate_pipeline(
        factory,
        classification_prepared.X_train_raw,
        classification_prepared.y_train,
        task_type=TaskType.CLASSIFICATION,
        folds=3,
        random_state=SEED,
    )

    assert all(not fold.succeeded for fold in fold_results)
    assert all(fold.error for fold in fold_results)


def test_invalid_model_is_rejected_before_any_fold_runs(
    classification_prepared: PreparedDataset,
) -> None:
    """A bad request fails immediately, not once per fold."""
    with pytest.raises(UnknownModelError):
        cross_validate_model(classification_prepared, "xgboost", folds=3)


def test_incompatible_model_is_rejected(
    classification_prepared: PreparedDataset,
) -> None:
    """A regressor cannot be cross-validated on a classification dataset."""
    with pytest.raises(IncompatibleTaskError):
        cross_validate_model(classification_prepared, "linear_regression", folds=3)


# --------------------------------------------------------------------------
# Leakage: the test set, and the folds themselves
# --------------------------------------------------------------------------


def test_cross_validation_never_reads_the_test_set(
    classification_prepared: PreparedDataset,
) -> None:
    """Replacing the whole test set changes nothing about the CV result.

    The prepared dataset is copied with its test attributes destroyed. If any
    part of cross-validation read them, the fold scores would move or the run
    would fail. Neither happens.
    """
    from dataclasses import replace

    baseline = cross_validate_model(classification_prepared, "logistic_regression", folds=4)

    poisoned = replace(
        classification_prepared,
        X_test_raw=classification_prepared.X_test_raw.iloc[:0],
        X_test=classification_prepared.X_test.iloc[:0],
        y_test=classification_prepared.y_test.iloc[:0],
        test_row_count=0,
    )
    after = cross_validate_model(poisoned, "logistic_regression", folds=4)

    assert [fold.metrics.values for fold in after.successful_folds] == [
        fold.metrics.values for fold in baseline.successful_folds
    ]
    assert after.mean_primary_metric == baseline.mean_primary_metric


def _numeric_imputer(pipeline):
    """Return the fitted numeric imputer inside a training pipeline."""
    branch = pipeline.named_steps["preprocessing"].named_transformers_["numeric"]
    return dict(branch.transformer_list)["values"].named_steps["impute"]


def _recording_factory(prepared: PreparedDataset, store: list):
    """Build a pipeline factory that keeps every pipeline it creates."""

    def factory():
        pipeline = clone_pipeline(
            build_pipeline(prepared, LogisticRegression(max_iter=1000))
        )
        store.append(pipeline)
        return pipeline

    return factory


def test_preprocessing_is_fitted_on_each_fold_alone(
    classification_prepared: PreparedDataset,
) -> None:
    """Each fold's imputer learns from that fold's training rows only.

    The statistics are checked against the median of exactly the rows the fold
    was given, and against the median of the whole training set, which they do
    not match.
    """
    features = classification_prepared.X_train_raw
    target = classification_prepared.y_train
    pipelines: list = []

    cross_validate_pipeline(
        _recording_factory(classification_prepared, pipelines),
        features,
        target,
        task_type=TaskType.CLASSIFICATION,
        folds=3,
        random_state=SEED,
    )

    splitter = build_splitter(TaskType.CLASSIFICATION, folds=3, random_state=SEED)
    fold_statistics = []
    for pipeline, (train_index, _) in zip(
        pipelines, splitter.split(features, target), strict=True
    ):
        expected = features.iloc[train_index]["income"].median()
        statistic = _numeric_imputer(pipeline).statistics_[0]
        assert statistic == pytest.approx(expected)
        fold_statistics.append(statistic)

    whole_training_median = features["income"].median()
    assert len(set(fold_statistics)) == 3, "each fold learned its own statistic"
    assert any(
        value != pytest.approx(whole_training_median) for value in fold_statistics
    ), "the folds cannot all have learned from the whole training set"


def test_validation_rows_cannot_influence_their_own_folds_preprocessing(
    classification_prepared: PreparedDataset,
) -> None:
    """Corrupting a fold's validation rows leaves that fold's fit untouched.

    The corrupted rows are validation data for fold 1 and training data for the
    others, so fold 1's statistics must be identical and the other folds' must
    change. That is only true if each fold fits on its own training rows.
    """
    features = classification_prepared.X_train_raw
    target = classification_prepared.y_train
    splits = list(
        build_splitter(TaskType.CLASSIFICATION, folds=3, random_state=SEED).split(
            features, target
        )
    )

    corrupted = features.copy()
    corrupted.iloc[splits[0][1], corrupted.columns.get_loc("income")] = 1e9

    original_pipelines: list = []
    corrupted_pipelines: list = []
    for frame, store in ((features, original_pipelines), (corrupted, corrupted_pipelines)):
        cross_validate_pipeline(
            _recording_factory(classification_prepared, store),
            frame,
            target,
            task_type=TaskType.CLASSIFICATION,
            folds=3,
            random_state=SEED,
        )

    original = [_numeric_imputer(item).statistics_[0] for item in original_pipelines]
    after = [_numeric_imputer(item).statistics_[0] for item in corrupted_pipelines]

    assert after[0] == pytest.approx(original[0]), (
        "fold 1 only ever saw these rows as validation data"
    )
    assert after[1] != pytest.approx(original[1])
    assert after[2] != pytest.approx(original[2])


# --------------------------------------------------------------------------
# Serialisation and packaging
# --------------------------------------------------------------------------


def test_summary_is_json_safe(classification_prepared: PreparedDataset) -> None:
    """The summary is plain data; no estimator is serialised."""
    result = cross_validate_model(classification_prepared, "logistic_regression", folds=3)
    summary = result.summary()

    json.dumps(summary)
    assert "pipeline" not in summary
    assert summary["folds"] == 3
    assert summary["successful_fold_count"] == 3
    assert summary["failed_fold_count"] == 0
    assert summary["evaluated_on"] == "training_folds"
    assert summary["primary_metric"]["key"] == "f1"
    assert summary["primary_metric"]["direction"] == "higher_is_better"
    assert len(summary["fold_results"]) == 3
    assert summary["aggregates"]["f1"]["mean"] == result.mean_primary_metric


def test_cross_validation_module_imports_on_its_own() -> None:
    """Importing the module first must not trip a package import cycle.

    ``ml.evaluation.cross_validation`` builds pipelines from ``ml.models``,
    which in turn reaches back for cross-validation. This guards the
    arrangement that keeps that from becoming an import-order trap.
    """
    completed = subprocess.run(
        [sys.executable, "-c", "import ml.evaluation.cross_validation"],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
