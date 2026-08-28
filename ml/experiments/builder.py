"""Assembling one experiment record from what the pipeline already produced.

Nothing here recomputes anything. Preprocessing, cross-validated selection,
final evaluation and explanation each already return a structured result with a
serialisable summary; this module composes them, adds the dataset fingerprint
and the environment, and hands back an :class:`~ml.experiments.run.ExperimentRun`.

That is deliberate. If the record disagreed with the objects it came from —
because it recalculated a metric its own way — the history would be worse than
useless. Composition keeps one source of truth for every number.

The dataset profile is optional. When it is supplied its quality findings are
carried into the record, because "this run was on data with three columns above
the missingness threshold" is exactly the kind of context that matters months
later, and exactly what a retrieval layer will want to index.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from ml.experiments.fingerprint import fingerprint_dataset
from ml.experiments.identity import configuration_hash, generate_experiment_id
from ml.experiments.run import (
    DatasetSection,
    EvaluationSection,
    ExperimentRun,
    ExplainabilitySection,
    PreprocessingSection,
    SelectionSection,
    capture_environment,
)
from ml.explainability.results import GlobalExplanation
from ml.models.selection import ModelSelectionResult
from ml.pipelines.result import PreparedDataset

#: Feature importances kept in a record. Enough to explain a model, few enough
#: that a wide dataset does not turn history into a data dump.
DEFAULT_MAX_FEATURE_IMPORTANCES = 50


def _quality_issues(profile: Any | None) -> tuple[dict[str, Any], ...]:
    """Extract the profiler's findings, if a profile was supplied.

    Read structurally rather than by type, so the experiment layer stays
    independent of the profiling implementation.
    """
    quality = getattr(profile, "quality", None)
    issues = getattr(quality, "issues", ()) if quality is not None else ()
    return tuple(
        {
            "code": str(getattr(issue, "code", "")),
            "severity": str(
                getattr(getattr(issue, "severity", ""), "value", getattr(issue, "severity", ""))
            ),
            "columns": [str(column) for column in getattr(issue, "columns", ())],
            "message": str(getattr(issue, "message", "")),
        }
        for issue in issues
    )


def _dataset_section(
    frame: pd.DataFrame,
    prepared: PreparedDataset,
    *,
    profile: Any | None,
    source_format: str | None,
) -> tuple[DatasetSection, str]:
    """Build the dataset section and return it with the fingerprint value."""
    fingerprint = fingerprint_dataset(frame)
    section = DatasetSection(
        fingerprint=fingerprint.value,
        fingerprint_algorithm=fingerprint.algorithm,
        row_count=fingerprint.row_count,
        column_count=fingerprint.column_count,
        target_column=prepared.config.target_column,
        task_type=prepared.task_type.value,
        columns=fingerprint.columns,
        dtypes=fingerprint.dtypes,
        source_format=source_format,
        data_quality_issues=_quality_issues(profile),
    )
    return section, fingerprint.value


def _preprocessing_section(prepared: PreparedDataset) -> PreprocessingSection:
    """Build the preprocessing section from the prepared dataset's summary."""
    summary = prepared.summary()
    split = summary["split"]
    return PreprocessingSection(
        config=dict(summary["preprocessing"]),
        feature_groups={
            key: list(values) for key, values in summary["feature_groups"].items()
        },
        selected_columns=tuple(summary["selected_columns"]),
        excluded_columns=tuple(summary["excluded_columns"]),
        identifier_columns=tuple(summary["identifier_columns"]),
        transformed_feature_names=tuple(summary["feature_names"]),
        column_decisions=tuple(summary["column_decisions"]),
        train_row_count=int(split["train_row_count"]),
        test_row_count=int(split["test_row_count"]),
        test_size=split["test_size"],
        random_state=split["random_state"],
        stratified=bool(split["stratified"]),
        stratification_note=split["stratification_note"],
        rows_dropped_missing_target=int(split["rows_dropped_missing_target"]),
    )


def _selection_section(summary: Mapping[str, Any]) -> SelectionSection:
    """Build the selection section from the selection result's summary."""
    selection = summary["selection"]
    metric = selection["primary_metric"]
    candidates = tuple(dict(item) for item in selection["candidates"])
    return SelectionSection(
        strategy=str(summary["strategy"]),
        folds=summary.get("folds"),
        primary_metric=str(metric["key"]),
        primary_metric_direction=metric.get("direction"),
        candidate_models=tuple(str(item["model_name"]) for item in candidates),
        candidates=candidates,
        selected_model=str(selection["winner"]),
        selection_score=selection.get("winner_score"),
        selection_score_std=selection.get("winner_score_std"),
        scored_on=selection.get("scored_on"),
        uses_test_data=bool(selection.get("uses_test_data", False)),
    )


def _evaluation_section(summary: Mapping[str, Any]) -> EvaluationSection:
    """Build the evaluation section from the selection result's summary."""
    final = summary["final_evaluation"]
    metrics = final["metrics"]
    baseline = final["baseline"]
    return EvaluationSection(
        primary_metric=str(final["primary_metric"]["key"]),
        primary_metric_value=final["primary_metric"]["value"],
        metrics=dict(metrics["values"]),
        unavailable_metrics=dict(metrics["unavailable"]),
        baseline_identifier=baseline.get("identifier"),
        baseline_metrics=dict(baseline["metrics"]["values"]),
        baseline_comparison=dict(final["baseline_comparison"]),
        classification_details=metrics.get("classification"),
        test_row_count=int(final["test_row_count"]),
        is_unbiased=bool(final["is_unbiased"]),
    )


def _explainability_section(
    explanation: GlobalExplanation, *, max_importances: int
) -> ExplainabilitySection:
    """Build the explainability section, keeping the record small."""
    summary = explanation.summary()
    importances = summary["feature_importances"]
    warnings = list(summary["warnings"])
    if len(importances) > max_importances:
        warnings.append(
            f"Stored the top {max_importances} of {len(importances)} feature "
            "importances to keep the experiment record small."
        )
        importances = importances[:max_importances]

    return ExplainabilitySection(
        status=str(summary["status"]),
        method=str(summary["method"]),
        explainer=summary.get("explainer"),
        aggregation=summary.get("aggregation"),
        explained_output=summary.get("explained_output"),
        feature_importances=tuple(dict(item) for item in importances),
        sample_count=int(summary["sample_count"]),
        feature_count=int(summary["feature_count"]),
        reason=summary.get("reason"),
        warnings=tuple(warnings),
    )


def configuration_components(
    *,
    fingerprint: str,
    prepared: PreparedDataset,
    selection: SelectionSection,
) -> dict[str, Any]:
    """Collect the inputs that define this experiment's configuration.

    Only inputs: which data, prepared how, which models were offered, how the
    winner was to be chosen, and with which seed. Outcomes are excluded on
    purpose, so re-running the same setup produces the same hash — which is
    what makes repeated runs findable and a reproducibility claim checkable.
    """
    summary = prepared.summary()
    return {
        "dataset_fingerprint": fingerprint,
        "target_column": summary["target_column"],
        "task_type": summary["task_type"],
        "preprocessing": summary["preprocessing"],
        "feature_groups": summary["feature_groups"],
        "selected_columns": summary["selected_columns"],
        "excluded_columns": summary["excluded_columns"],
        "test_size": summary["split"]["test_size"],
        "random_state": summary["split"]["random_state"],
        "selection_strategy": selection.strategy,
        "folds": selection.folds,
        "primary_metric": selection.primary_metric,
        "candidate_models": sorted(selection.candidate_models),
    }


def create_experiment_run(
    frame: pd.DataFrame,
    prepared: PreparedDataset,
    selection: ModelSelectionResult,
    *,
    name: str,
    explanation: GlobalExplanation | None = None,
    profile: Any | None = None,
    description: str | None = None,
    tags: Sequence[str] = (),
    source_format: str | None = None,
    created_at: datetime | None = None,
    max_feature_importances: int = DEFAULT_MAX_FEATURE_IMPORTANCES,
) -> ExperimentRun:
    """Compose one experiment record from the pipeline's existing results.

    Args:
        frame: The dataset the run used, for its content fingerprint. Only the
            fingerprint is kept — the data itself is never stored.
        prepared: The preprocessing result from Commit 3.
        selection: The cross-validated selection and final evaluation from
            Commit 5.
        name: A short human label for the run.
        explanation: The global explanation from Commit 6, when one was made.
        profile: The dataset profile from Commit 2, when one is to hand. Its
            quality findings are recorded as context.
        description: Longer free text about the run.
        tags: Labels for grouping and later retrieval.
        source_format: Where the data came from — ``"csv"``, ``"parquet"``, a
            database name. Recorded as context only; runs are identified by
            fingerprint, never by format or path.
        created_at: When the run happened; now, in UTC, when omitted.
        max_feature_importances: How many ranked features to keep.

    Returns:
        ExperimentRun: The record, ready to be saved.
    """
    moment = created_at or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)

    summary = selection.summary()
    dataset, fingerprint = _dataset_section(
        frame, prepared, profile=profile, source_format=source_format
    )
    preprocessing = _preprocessing_section(prepared)
    selection_section = _selection_section(summary)
    evaluation = _evaluation_section(summary)

    config_hash = configuration_hash(
        configuration_components(
            fingerprint=fingerprint,
            prepared=prepared,
            selection=selection_section,
        )
    )

    return ExperimentRun(
        experiment_id=generate_experiment_id(config_hash, created_at=moment),
        configuration_hash=config_hash,
        created_at=moment,
        name=name,
        description=description,
        tags=tuple(str(tag) for tag in tags),
        dataset=dataset,
        preprocessing=preprocessing,
        selection=selection_section,
        evaluation=evaluation,
        explainability=(
            _explainability_section(
                explanation, max_importances=max_feature_importances
            )
            if explanation is not None
            else None
        ),
        environment=capture_environment(prepared.config.random_state),
    )
