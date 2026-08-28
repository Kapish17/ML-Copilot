"""What one experiment is, written down.

An ``ExperimentRun`` is the record of a complete pass through the pipeline:
which data, prepared how, which models were considered, which one won and why,
how it scored on data it had never seen, and what the explanations said. It is
metadata and results only — no dataset, no fitted pipeline, no explainer.

The record is written in sections that mirror the layers that produced them, so
a reader can find what they need without knowing the code, and a future
retrieval layer can index each section on its own.

Every record carries a ``schema_version``. Reading a record written under a
version this code does not understand fails with a clear error rather than
producing a half-populated object — a stored history is only useful if it can
be trusted.
"""

from __future__ import annotations

import platform
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ml.errors import InvalidExperimentRecordError, UnsupportedSchemaVersionError
from ml.experiments.serialization import to_jsonable

#: The version of this record format. Bump it when a change would stop older
#: readers from understanding a new file.
EXPERIMENT_SCHEMA_VERSION = "1.0"
#: Versions this code can read.
SUPPORTED_SCHEMA_VERSIONS = frozenset({EXPERIMENT_SCHEMA_VERSION})


def _require(payload: Mapping[str, Any], key: str, *, where: str) -> Any:
    """Return a required field, or explain precisely what is missing."""
    if not isinstance(payload, Mapping):
        raise InvalidExperimentRecordError(
            f"The '{where}' section must be an object, not "
            f"{type(payload).__name__}.",
            details={"section": where},
        )
    if key not in payload:
        raise InvalidExperimentRecordError(
            f"The '{where}' section is missing the required field '{key}'.",
            details={"section": where, "field": key},
        )
    return payload[key]


def _typed(
    payload: Mapping[str, Any], key: str, kind: type | tuple[type, ...], *, where: str
) -> Any:
    """Return a required field, checking its type."""
    value = _require(payload, key, where=where)
    if not isinstance(value, kind):
        expected = getattr(kind, "__name__", str(kind))
        raise InvalidExperimentRecordError(
            f"Field '{key}' in section '{where}' should be {expected}, not "
            f"{type(value).__name__}.",
            details={"section": where, "field": key},
        )
    return value


def _strings(value: Any) -> tuple[str, ...]:
    """Coerce a stored list into a tuple of strings."""
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        return (str(value),)
    return tuple(str(item) for item in value)


def _mapping(value: Any) -> dict[str, Any]:
    """Coerce a stored mapping into a plain dictionary."""
    return dict(value) if isinstance(value, Mapping) else {}


def _records(value: Any) -> tuple[dict[str, Any], ...]:
    """Coerce a stored list of objects into a tuple of dictionaries."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(dict(item) for item in value if isinstance(item, Mapping))


@dataclass(frozen=True)
class DatasetSection:
    """Which data the experiment ran on, identified by content."""

    fingerprint: str
    row_count: int
    column_count: int
    target_column: str
    task_type: str
    columns: tuple[str, ...] = ()
    dtypes: dict[str, str] = field(default_factory=dict)
    fingerprint_algorithm: str = "sha256"
    source_format: str | None = None
    data_quality_issues: tuple[dict[str, Any], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Render the section as plain, JSON-friendly values."""
        return {
            "fingerprint": self.fingerprint,
            "fingerprint_algorithm": self.fingerprint_algorithm,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "target_column": self.target_column,
            "task_type": self.task_type,
            "columns": list(self.columns),
            "dtypes": dict(self.dtypes),
            "source_format": self.source_format,
            "data_quality_issues": [dict(item) for item in self.data_quality_issues],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DatasetSection:
        """Rebuild the section from a stored record."""
        where = "dataset"
        return cls(
            fingerprint=str(_typed(payload, "fingerprint", str, where=where)),
            row_count=int(_typed(payload, "row_count", int, where=where)),
            column_count=int(_typed(payload, "column_count", int, where=where)),
            target_column=str(_typed(payload, "target_column", str, where=where)),
            task_type=str(_typed(payload, "task_type", str, where=where)),
            columns=_strings(payload.get("columns")),
            dtypes={str(k): str(v) for k, v in _mapping(payload.get("dtypes")).items()},
            fingerprint_algorithm=str(payload.get("fingerprint_algorithm", "sha256")),
            source_format=payload.get("source_format"),
            data_quality_issues=_records(payload.get("data_quality_issues")),
        )


@dataclass(frozen=True)
class PreprocessingSection:
    """How the data was turned into features, and what was left out."""

    config: dict[str, Any] = field(default_factory=dict)
    feature_groups: dict[str, list[str]] = field(default_factory=dict)
    selected_columns: tuple[str, ...] = ()
    excluded_columns: tuple[str, ...] = ()
    identifier_columns: tuple[str, ...] = ()
    transformed_feature_names: tuple[str, ...] = ()
    column_decisions: tuple[dict[str, Any], ...] = ()
    train_row_count: int = 0
    test_row_count: int = 0
    test_size: float | None = None
    random_state: int | None = None
    stratified: bool = False
    stratification_note: str | None = None
    rows_dropped_missing_target: int = 0

    def as_dict(self) -> dict[str, Any]:
        """Render the section as plain, JSON-friendly values."""
        return {
            "config": dict(self.config),
            "feature_groups": {
                key: list(values) for key, values in self.feature_groups.items()
            },
            "selected_columns": list(self.selected_columns),
            "excluded_columns": list(self.excluded_columns),
            "identifier_columns": list(self.identifier_columns),
            "transformed_feature_names": list(self.transformed_feature_names),
            "column_decisions": [dict(item) for item in self.column_decisions],
            "train_row_count": self.train_row_count,
            "test_row_count": self.test_row_count,
            "test_size": self.test_size,
            "random_state": self.random_state,
            "stratified": self.stratified,
            "stratification_note": self.stratification_note,
            "rows_dropped_missing_target": self.rows_dropped_missing_target,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PreprocessingSection:
        """Rebuild the section from a stored record."""
        where = "preprocessing"
        groups = _mapping(_require(payload, "feature_groups", where=where))
        return cls(
            config=_mapping(payload.get("config")),
            feature_groups={key: list(_strings(value)) for key, value in groups.items()},
            selected_columns=_strings(payload.get("selected_columns")),
            excluded_columns=_strings(payload.get("excluded_columns")),
            identifier_columns=_strings(payload.get("identifier_columns")),
            transformed_feature_names=_strings(payload.get("transformed_feature_names")),
            column_decisions=_records(payload.get("column_decisions")),
            train_row_count=int(_typed(payload, "train_row_count", int, where=where)),
            test_row_count=int(_typed(payload, "test_row_count", int, where=where)),
            test_size=payload.get("test_size"),
            random_state=payload.get("random_state"),
            stratified=bool(payload.get("stratified", False)),
            stratification_note=payload.get("stratification_note"),
            rows_dropped_missing_target=int(payload.get("rows_dropped_missing_target", 0)),
        )


@dataclass(frozen=True)
class SelectionSection:
    """Which models were tried and how the winner was chosen.

    Under cross-validated selection these scores come from the training folds
    alone, which is recorded in ``scored_on`` so a later reader never mistakes
    them for test results.
    """

    strategy: str
    primary_metric: str
    selected_model: str
    primary_metric_direction: str | None = None
    folds: int | None = None
    candidate_models: tuple[str, ...] = ()
    candidates: tuple[dict[str, Any], ...] = ()
    selection_score: float | None = None
    selection_score_std: float | None = None
    scored_on: str | None = None
    uses_test_data: bool = False

    def as_dict(self) -> dict[str, Any]:
        """Render the section as plain, JSON-friendly values."""
        return {
            "strategy": self.strategy,
            "folds": self.folds,
            "primary_metric": self.primary_metric,
            "primary_metric_direction": self.primary_metric_direction,
            "candidate_models": list(self.candidate_models),
            "candidates": [dict(item) for item in self.candidates],
            "selected_model": self.selected_model,
            "selection_score": self.selection_score,
            "selection_score_std": self.selection_score_std,
            "scored_on": self.scored_on,
            "uses_test_data": self.uses_test_data,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SelectionSection:
        """Rebuild the section from a stored record."""
        where = "selection"
        return cls(
            strategy=str(_typed(payload, "strategy", str, where=where)),
            primary_metric=str(_typed(payload, "primary_metric", str, where=where)),
            selected_model=str(_typed(payload, "selected_model", str, where=where)),
            primary_metric_direction=payload.get("primary_metric_direction"),
            folds=payload.get("folds"),
            candidate_models=_strings(payload.get("candidate_models")),
            candidates=_records(payload.get("candidates")),
            selection_score=payload.get("selection_score"),
            selection_score_std=payload.get("selection_score_std"),
            scored_on=payload.get("scored_on"),
            uses_test_data=bool(payload.get("uses_test_data", False)),
        )


@dataclass(frozen=True)
class EvaluationSection:
    """How the chosen model scored on data it had never seen."""

    primary_metric: str
    primary_metric_value: float | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    unavailable_metrics: dict[str, str] = field(default_factory=dict)
    baseline_identifier: str | None = None
    baseline_metrics: dict[str, float] = field(default_factory=dict)
    baseline_comparison: dict[str, Any] = field(default_factory=dict)
    classification_details: dict[str, Any] | None = None
    test_row_count: int = 0
    is_unbiased: bool = False

    def as_dict(self) -> dict[str, Any]:
        """Render the section as plain, JSON-friendly values."""
        return {
            "primary_metric": self.primary_metric,
            "primary_metric_value": self.primary_metric_value,
            "metrics": dict(self.metrics),
            "unavailable_metrics": dict(self.unavailable_metrics),
            "baseline_identifier": self.baseline_identifier,
            "baseline_metrics": dict(self.baseline_metrics),
            "baseline_comparison": dict(self.baseline_comparison),
            "classification_details": self.classification_details,
            "test_row_count": self.test_row_count,
            "is_unbiased": self.is_unbiased,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> EvaluationSection:
        """Rebuild the section from a stored record."""
        where = "evaluation"
        return cls(
            primary_metric=str(_typed(payload, "primary_metric", str, where=where)),
            primary_metric_value=payload.get("primary_metric_value"),
            metrics=_mapping(_require(payload, "metrics", where=where)),
            unavailable_metrics=_mapping(payload.get("unavailable_metrics")),
            baseline_identifier=payload.get("baseline_identifier"),
            baseline_metrics=_mapping(payload.get("baseline_metrics")),
            baseline_comparison=_mapping(payload.get("baseline_comparison")),
            classification_details=payload.get("classification_details"),
            test_row_count=int(payload.get("test_row_count", 0)),
            is_unbiased=bool(payload.get("is_unbiased", False)),
        )


@dataclass(frozen=True)
class ExplainabilitySection:
    """What the explanations said about the chosen model."""

    status: str
    method: str
    explainer: str | None = None
    aggregation: str | None = None
    explained_output: str | None = None
    feature_importances: tuple[dict[str, Any], ...] = ()
    sample_count: int = 0
    feature_count: int = 0
    reason: str | None = None
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Render the section as plain, JSON-friendly values."""
        return {
            "status": self.status,
            "method": self.method,
            "explainer": self.explainer,
            "aggregation": self.aggregation,
            "explained_output": self.explained_output,
            "feature_importances": [dict(item) for item in self.feature_importances],
            "sample_count": self.sample_count,
            "feature_count": self.feature_count,
            "reason": self.reason,
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ExplainabilitySection:
        """Rebuild the section from a stored record."""
        where = "explainability"
        return cls(
            status=str(_typed(payload, "status", str, where=where)),
            method=str(_typed(payload, "method", str, where=where)),
            explainer=payload.get("explainer"),
            aggregation=payload.get("aggregation"),
            explained_output=payload.get("explained_output"),
            feature_importances=_records(payload.get("feature_importances")),
            sample_count=int(payload.get("sample_count", 0)),
            feature_count=int(payload.get("feature_count", 0)),
            reason=payload.get("reason"),
            warnings=_strings(payload.get("warnings")),
        )


@dataclass(frozen=True)
class EnvironmentSection:
    """What it took to produce this result, for anyone reproducing it.

    Deliberately excluded: hostname, user name, working directory, environment
    variables and anything else that identifies a machine or could carry a
    credential. Library versions and the interpreter are what reproduction
    needs; the rest is someone's private business.
    """

    python_version: str
    platform: str
    packages: dict[str, str] = field(default_factory=dict)
    random_state: int | None = None

    def as_dict(self) -> dict[str, Any]:
        """Render the section as plain, JSON-friendly values."""
        return {
            "python_version": self.python_version,
            "platform": self.platform,
            "packages": dict(self.packages),
            "random_state": self.random_state,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> EnvironmentSection:
        """Rebuild the section from a stored record."""
        where = "environment"
        return cls(
            python_version=str(_typed(payload, "python_version", str, where=where)),
            platform=str(payload.get("platform", "")),
            packages={
                str(k): str(v) for k, v in _mapping(payload.get("packages")).items()
            },
            random_state=payload.get("random_state"),
        )


def _package_version(name: str) -> str | None:
    """Return an installed package's version, or ``None`` if it is absent."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:  # pragma: no cover - all are installed here
        return None


def capture_environment(random_state: int | None = None) -> EnvironmentSection:
    """Record the interpreter and library versions behind a run.

    Args:
        random_state: The seed the run used, kept beside the versions because
            reproducing a result needs both.

    Returns:
        EnvironmentSection: Versions and platform, with nothing identifying.
    """
    packages = {
        name: value
        for name in ("pandas", "numpy", "scikit-learn", "scipy", "shap")
        if (value := _package_version(name)) is not None
    }
    return EnvironmentSection(
        python_version=platform.python_version(),
        platform=f"{platform.system()}-{platform.machine()}",
        packages=packages,
        random_state=random_state,
    )


@dataclass(frozen=True)
class ExperimentRun:
    """One complete experiment, as it will be remembered."""

    experiment_id: str
    configuration_hash: str
    created_at: datetime
    name: str
    dataset: DatasetSection
    preprocessing: PreprocessingSection
    selection: SelectionSection
    evaluation: EvaluationSection
    environment: EnvironmentSection
    explainability: ExplainabilitySection | None = None
    description: str | None = None
    tags: tuple[str, ...] = ()
    schema_version: str = EXPERIMENT_SCHEMA_VERSION

    @property
    def selected_model(self) -> str:
        """The model this run settled on."""
        return self.selection.selected_model

    @property
    def task_type(self) -> str:
        """The kind of problem this run solved."""
        return self.dataset.task_type

    @property
    def primary_metric(self) -> str:
        """The metric the run was judged by."""
        return self.evaluation.primary_metric

    def headline(self) -> dict[str, Any]:
        """A one-line view of the run, for listings and comparisons."""
        return {
            "experiment_id": self.experiment_id,
            "created_at": self.created_at.isoformat(),
            "name": self.name,
            "dataset_fingerprint": self.dataset.fingerprint,
            "task_type": self.task_type,
            "target_column": self.dataset.target_column,
            "selected_model": self.selected_model,
            "strategy": self.selection.strategy,
            "primary_metric": self.primary_metric,
            "selection_score": self.selection.selection_score,
            "test_score": self.evaluation.primary_metric_value,
        }

    def to_dict(self) -> dict[str, Any]:
        """Render the whole run as a JSON-safe record."""
        payload = {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "configuration_hash": self.configuration_hash,
            "created_at": self.created_at,
            "name": self.name,
            "description": self.description,
            "tags": list(self.tags),
            "dataset": self.dataset.as_dict(),
            "preprocessing": self.preprocessing.as_dict(),
            "selection": self.selection.as_dict(),
            "evaluation": self.evaluation.as_dict(),
            "explainability": (
                self.explainability.as_dict() if self.explainability else None
            ),
            "environment": self.environment.as_dict(),
        }
        return to_jsonable(payload)

    @classmethod
    def from_dict(cls, payload: Any) -> ExperimentRun:
        """Rebuild a run from a stored record, validating as it goes.

        Args:
            payload: The parsed JSON record.

        Returns:
            ExperimentRun: The reconstructed run.

        Raises:
            InvalidExperimentRecordError: The record is not an object, or a
                required field is missing or of the wrong type.
            UnsupportedSchemaVersionError: The record was written under a
                schema version this code cannot read.
        """
        if not isinstance(payload, Mapping):
            raise InvalidExperimentRecordError(
                f"An experiment record must be an object, not "
                f"{type(payload).__name__}."
            )

        version = payload.get("schema_version")
        if version not in SUPPORTED_SCHEMA_VERSIONS:
            raise UnsupportedSchemaVersionError(
                f"Experiment schema version {version!r} cannot be read by this "
                f"version of ML Copilot, which supports "
                f"{sorted(SUPPORTED_SCHEMA_VERSIONS)}.",
                details={
                    "found": version,
                    "supported": sorted(SUPPORTED_SCHEMA_VERSIONS),
                },
            )

        where = "experiment"
        raw_created = _typed(payload, "created_at", str, where=where)
        try:
            created_at = datetime.fromisoformat(raw_created)
        except ValueError as exc:
            raise InvalidExperimentRecordError(
                f"'created_at' is not an ISO 8601 timestamp: {raw_created!r}"
            ) from exc
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)

        explainability = payload.get("explainability")
        return cls(
            experiment_id=str(_typed(payload, "experiment_id", str, where=where)),
            configuration_hash=str(
                _typed(payload, "configuration_hash", str, where=where)
            ),
            created_at=created_at,
            name=str(_typed(payload, "name", str, where=where)),
            dataset=DatasetSection.from_dict(_require(payload, "dataset", where=where)),
            preprocessing=PreprocessingSection.from_dict(
                _require(payload, "preprocessing", where=where)
            ),
            selection=SelectionSection.from_dict(
                _require(payload, "selection", where=where)
            ),
            evaluation=EvaluationSection.from_dict(
                _require(payload, "evaluation", where=where)
            ),
            environment=EnvironmentSection.from_dict(
                _require(payload, "environment", where=where)
            ),
            explainability=(
                ExplainabilitySection.from_dict(explainability)
                if isinstance(explainability, Mapping)
                else None
            ),
            description=payload.get("description"),
            tags=_strings(payload.get("tags")),
            schema_version=str(version),
        )
