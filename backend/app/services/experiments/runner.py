"""Orchestration of one complete experiment.

The runner is the application service the API is an adapter around. It owns the
order of operations and nothing else: every step below is a call into a package
built in an earlier commit, and no statistic, split, score or explanation is
computed here.

::

    DataFrame -> profile -> infer configuration -> apply overrides
              -> prepare (leakage-safe split) -> validate candidates
              -> cross-validate and select -> retrain winner
              -> single untouched-test evaluation -> SHAP explanation
              -> ExperimentRun -> ExperimentStore

The entry point that matters is :meth:`ExperimentRunner.run_frame`, which takes
a **standardised DataFrame**. Files enter through :meth:`run_upload`, which
does nothing but ask the dataset service for that DataFrame — so CSV, Excel and
JSON uploads all arrive here as the same object and run through the identical
pipeline. The format is carried onto the record as context and is never read by
anything that makes a decision. Adding Parquet or a SQL source would mean
another adapter in the dataset service and no change here. **Parquet, SQL,
databases, cloud storage and URL ingestion are not implemented.**

Execution is synchronous: one HTTP request runs the pipeline and waits. The
limits in :class:`~app.core.config.Settings` exist to bound how long that can
take. Nothing about the design assumes it, though — :meth:`run_frame` is a
plain function of its arguments with no shared state, so moving it onto a
worker later is a change of caller, not of runner.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from app.core.config import Settings
from app.core.errors import DatasetTooLargeError
from app.services.datasets import DatasetProfilingService
from app.services.datasets.validation import AsyncReadable
from app.services.experiments.options import ExperimentOptions
from ml.artifacts import ModelArtifactStore, build_metadata
from ml.errors import ExplainabilityError, MissingTargetError, ModelArtifactError
from ml.evaluation.metrics import get_metric
from ml.experiments import ExperimentRun, create_experiment_run
from ml.experiments.run import ModelArtifactSection
from ml.experiments.store import ExperimentStore
from ml.explainability import ExplanationConfig, GlobalExplanation, explain_global
from ml.features.inference import infer_configuration
from ml.models.registry import ModelRegistry, default_registry
from ml.models.selection import ModelSelectionResult, select_and_evaluate_best_model
from ml.models.spec import ModelSpec, validate_spec
from ml.pipelines.preparation import prepare_dataset
from ml.pipelines.result import PreparedDataset

logger = logging.getLogger(__name__)

#: Recorded when the caller hands over a DataFrame directly and says nothing
#: about where it came from. An upload always reports its real format.
SOURCE_FORMAT = "csv"

#: Used when the caller names neither a target column nor a run name.
FALLBACK_RUN_NAME = "untitled experiment"


@dataclass(frozen=True)
class ExperimentArtifacts:
    """The live objects one run produced, for the caller that asked to keep them.

    **This is memory, not persistence.** Commit 7 deliberately does not write
    a fitted estimator to disk, and nothing here changes that: these objects
    exist only for as long as the caller holds this result, are never
    serialised, never stored, and never reachable from
    :meth:`ExperimentRunResult.as_dict`. Once the process ends they are gone,
    and the experiment record — which is what is persisted — still contains no
    model.

    The one caller that wants them is an in-process orchestrator that has just
    run an experiment and wants to explain it in the same breath. Asking for
    them is explicit (``retain_artifacts=True``) so that no caller keeps a
    fitted pipeline alive by accident.
    """

    #: The fitted model, as :mod:`ml.explainability` expects it.
    trained_model: Any
    #: Raw training features, the natural reference rows for an explanation.
    X_reference: Any
    #: Training targets, needed only by the permutation fallback.
    y_reference: Any


@dataclass(frozen=True)
class ExperimentRunResult:
    """One executed experiment, plus what the caller should know about it.

    ``record`` is the stored :class:`~ml.experiments.run.ExperimentRun`; it is
    already free of estimators, explainers and data. ``warnings`` collects
    everything the run decided on the caller's behalf or could not do.
    """

    record: ExperimentRun
    warnings: tuple[str, ...]
    stored: bool
    duration_seconds: float
    #: Live fitted objects, present only when the caller asked for them. Left
    #: out of :meth:`as_dict` deliberately — it renders what is *stored*, and
    #: a fitted model is not.
    artifacts: ExperimentArtifacts | None = None

    def as_dict(self) -> dict[str, Any]:
        """Render the whole result as plain JSON-safe values."""
        payload = self.record.to_dict()
        payload["warnings"] = list(self.warnings)
        payload["execution"] = {
            "duration_seconds": self.duration_seconds,
            "stored": self.stored,
            "mode": "synchronous",
        }
        return payload


class ExperimentRunner:
    """Run an experiment end to end and record what happened."""

    def __init__(
        self,
        settings: Settings,
        store: ExperimentStore,
        dataset_service: DatasetProfilingService,
        *,
        registry: ModelRegistry | None = None,
        artifact_store: ModelArtifactStore | None = None,
    ) -> None:
        """Wire the runner to its collaborators.

        Args:
            settings: Active application settings, supplying every limit.
            store: Where the resulting record is saved. Any implementation of
                the storage interface will do; the local JSON store is the
                only one that exists. **MLflow is not implemented.**
            dataset_service: The ingestion and profiling adapter.
            registry: Model registry to resolve candidates in.
            artifact_store: Where the winning fitted model is persisted, so it
                can be predicted from later. Optional: with no store, a run
                still completes and is recorded exactly as before, and the
                record simply carries no model section. That is what a caller
                with nowhere to write — a script, a test of the pipeline
                itself — gets, and it is also the state every run made before
                Commit 22 is in.
        """
        self._settings = settings
        self._store = store
        self._datasets = dataset_service
        self._registry = registry or default_registry()
        self._artifacts = artifact_store

    # -- Entry points ------------------------------------------------------

    async def run_upload(
        self,
        upload: AsyncReadable,
        filename: str | None,
        options: ExperimentOptions,
        content_type: str | None = None,
    ) -> ExperimentRunResult:
        """Run an experiment on an uploaded dataset file.

        CSV, Excel and JSON all arrive here; the dataset service turns each
        into the same standardised DataFrame, and everything after that line
        is identical. The upload is validated and parsed in memory and is
        never written to disk; only its fingerprint, shape and format survive,
        inside the record.

        Args:
            upload: The incoming file object.
            filename: Client-supplied filename, used only for its extension.
            options: The experiment configuration.
            content_type: The client's declared media type, if any.

        Returns:
            ExperimentRunResult: The stored record and its warnings.

        Raises:
            DatasetError: If the upload fails validation or cannot be parsed.
            MLError: If the experiment itself cannot be run.
        """
        validated = options.validated(self._settings)
        loaded = await self._datasets.load_upload(upload, filename, content_type)
        return self.run_frame(
            loaded.frame,
            validated,
            dataset_label=loaded.filename,
            source_format=loaded.source_format,
        )

    def run_content(
        self,
        filename: str | None,
        content: bytes,
        options: ExperimentOptions,
        content_type: str | None = None,
    ) -> ExperimentRunResult:
        """Run an experiment on dataset bytes already in memory."""
        validated = options.validated(self._settings)
        loaded = self._datasets.load_content(filename, content, content_type)
        return self.run_frame(
            loaded.frame,
            validated,
            dataset_label=loaded.filename,
            source_format=loaded.source_format,
        )

    def run_frame(
        self,
        frame: pd.DataFrame,
        options: ExperimentOptions,
        *,
        dataset_label: str = "dataset",
        source_format: str = SOURCE_FORMAT,
        created_at: datetime | None = None,
        retain_artifacts: bool = False,
    ) -> ExperimentRunResult:
        """Run an experiment on a standardised DataFrame.

        This is the format-agnostic operation a future agent would call: it
        takes data and a configuration, and returns a structured result with
        no filesystem, pandas or sklearn detail in it.

        Args:
            frame: The dataset, already standardised by an ingestion adapter.
            options: The experiment configuration. Assumed already validated
                by :meth:`~app.services.experiments.options.ExperimentOptions.validated`
                when it arrives from one of the entry points above.
            dataset_label: Display label for the data; never used as a path.
            source_format: Where the data came from, recorded as context. Runs
                are identified by content fingerprint, never by format.
            created_at: When the run happened; now, in UTC, when omitted.
            retain_artifacts: Keep the fitted model and its reference rows on
                the result, in memory only. Off by default. Nothing about
                what is *stored* changes either way — the record contains no
                model in both cases; this only decides whether the caller
                keeps a reference to the live objects after the call returns.

        Returns:
            ExperimentRunResult: The stored record and its warnings.

        Raises:
            DatasetTooLargeError: If the dataset exceeds the experiment limits.
            MLError: If any stage of the pipeline rejects the request.
        """
        started = time.perf_counter()
        warnings: list[str] = []

        self._check_size(frame)
        target = self._resolve_target(frame, options, warnings)

        # A run is the longest thing this service does and the only one that
        # holds its request open for seconds or minutes. Two lines — one when
        # it starts, one when it finishes — turn "the API is hanging" into "it
        # is cross-validating six models on 40,000 rows".
        #
        # Shape, strategy and column *names* only. No cell value is logged
        # here or anywhere below.
        logger.info(
            "Experiment started: %d rows x %d columns, target=%r, strategy=%s, "
            "models=%s",
            len(frame),
            frame.shape[1],
            target,
            options.strategy,
            len(options.models) or "all",
        )

        profile = self._datasets.profile_frame(
            frame,
            filename=dataset_label,
            target_column=target,
            source_format=source_format,
        )
        prepared = self._prepare(frame, profile, options, target)
        self._validate_candidates(options, prepared)

        selection = select_and_evaluate_best_model(
            prepared,
            models=list(options.models) or None,
            registry=self._registry,
            primary_metric=options.primary_metric,
            strategy=options.strategy,
            folds=options.folds or self._settings.default_cv_folds,
        )

        explanation = self._explain(selection, prepared, options, warnings)
        if explanation is not None and explanation.warnings:
            warnings.extend(explanation.warnings)

        record = create_experiment_run(
            frame,
            prepared,
            selection,
            name=options.resolved_name(f"{dataset_label} · {target}"),
            explanation=explanation,
            profile=profile,
            description=options.description,
            tags=options.tags,
            source_format=source_format,
            created_at=created_at or datetime.now(timezone.utc),
            max_feature_importances=self._settings.explanation_top_features,
        )

        # Only now, with a complete record in hand, is there a model worth
        # keeping: every step above succeeded, the winner was chosen on
        # cross-validated scores and measured once on the untouched test set.
        # A run that failed anywhere earlier raised long before this line, so
        # a failed experiment can never leave a usable model behind.
        record = replace(
            record, model_artifact=self._persist_model(record, prepared, selection)
        )
        self._store.save(record)

        artifacts = (
            ExperimentArtifacts(
                trained_model=selection.final_model,
                X_reference=prepared.X_train_raw,
                y_reference=prepared.y_train,
            )
            if retain_artifacts
            else None
        )

        duration = round(time.perf_counter() - started, 3)
        logger.info(
            "Experiment finished in %.1fs: %s selected on %s, stored as %s "
            "(explanation=%s, warnings=%d)",
            duration,
            record.selection.selected_model,
            record.selection.primary_metric,
            record.experiment_id,
            record.explainability.status if record.explainability else "skipped",
            len(warnings),
        )

        return ExperimentRunResult(
            record=record,
            warnings=tuple(warnings),
            stored=True,
            duration_seconds=duration,
            artifacts=artifacts,
        )

    # -- Steps -------------------------------------------------------------

    def _persist_model(
        self,
        record: ExperimentRun,
        prepared: PreparedDataset,
        selection: ModelSelectionResult,
    ) -> ModelArtifactSection | None:
        """Save the winning fitted pipeline, and describe it on the record.

        What is written is the **same object the experiment fitted** — the
        preprocessing that learned its statistics from the training rows, with
        the winning estimator behind it. Nothing is rebuilt, refitted or
        reconstructed, which is what makes a prediction made next month
        comparable to the score this run reported.

        A failure here is not a failure of the experiment. The run happened,
        the numbers are real, and the record is worth keeping; what is lost is
        the ability to predict from it later. So the failure is logged and
        ``None`` is returned, which reads back as "this run has no model" —
        the same state as every run made before Commit 22.

        Args:
            record: The finished record, for its experiment id.
            prepared: The dataset the pipeline was fitted on.
            selection: The finished selection, carrying the winner.

        Returns:
            ModelArtifactSection | None: The section to attach, or ``None``
            when no model was stored.
        """
        if self._artifacts is None:
            return None

        try:
            metadata = build_metadata(
                experiment_id=record.experiment_id,
                prepared=prepared,
                selection=selection,
                created_at=record.created_at,
            )
            self._artifacts.save(
                record.experiment_id, selection.final_model.pipeline, metadata
            )
        except (ModelArtifactError, OSError) as exc:
            logger.warning(
                "Could not persist the model for %s: %s — the experiment is "
                "recorded without one and cannot be predicted from",
                record.experiment_id,
                type(exc).__name__,
            )
            return None

        return ModelArtifactSection(
            stored=True,
            model_name=metadata.model_name,
            task_type=metadata.task_type.value,
            target_column=metadata.target_column,
            feature_names=metadata.feature_names,
            class_labels=tuple(
                str(label) for label in metadata.public_summary()["classes"]
            ),
            artifact_schema_version=metadata.schema_version,
            created_at=metadata.created_at.isoformat(),
        )

    def _check_size(self, frame: pd.DataFrame) -> None:
        """Refuse a dataset too large to run synchronously."""
        rows, columns = frame.shape
        if rows > self._settings.max_experiment_rows:
            raise DatasetTooLargeError(
                f"An experiment may use at most "
                f"{self._settings.max_experiment_rows} rows, this dataset has "
                f"{rows}.",
                details={
                    "row_count": rows,
                    "max_rows": self._settings.max_experiment_rows,
                },
            )
        if columns > self._settings.max_experiment_feature_columns:
            raise DatasetTooLargeError(
                f"An experiment may use at most "
                f"{self._settings.max_experiment_feature_columns} columns, this "
                f"dataset has {columns}.",
                details={
                    "column_count": columns,
                    "max_columns": self._settings.max_experiment_feature_columns,
                },
            )

    def _resolve_target(
        self,
        frame: pd.DataFrame,
        options: ExperimentOptions,
        warnings: list[str],
    ) -> str:
        """Decide which column is the target.

        When the caller names one it is used. When they do not, the **last
        column** is used — the ordinary convention for a tabular dataset — and
        the choice is reported as a warning rather than made silently.

        This is a naming convention, not detection: nothing examines the data
        to decide which column is worth predicting, and **automatic target
        detection is not implemented**.

        Raises:
            MissingTargetError: If the dataset has no columns at all.
        """
        if options.target_column:
            return options.target_column

        columns = list(frame.columns)
        if not columns:
            raise MissingTargetError(
                "The dataset has no columns, so there is nothing to predict.",
                details={"available_columns": []},
            )
        chosen = str(columns[-1])
        warnings.append(
            f"No target column was given, so the last column '{chosen}' was "
            "used by convention. Automatic target detection is not "
            "implemented; name the column explicitly to be sure."
        )
        return chosen

    def _prepare(
        self,
        frame: pd.DataFrame,
        profile: Any,
        options: ExperimentOptions,
        target: str,
    ) -> PreparedDataset:
        """Infer a configuration from the profile, apply overrides, prepare."""
        inferred = infer_configuration(
            profile,
            target_column=target,
            excluded_columns=options.excluded_columns,
            identifier_columns=options.identifier_columns,
        )
        overrides = options.preprocessing_overrides
        config = (
            inferred.config.with_overrides(**overrides)
            if overrides
            else inferred.config
        )
        return prepare_dataset(frame, config, decisions=inferred.decisions)

    def _validate_candidates(
        self, options: ExperimentOptions, prepared: PreparedDataset
    ) -> None:
        """Check the requested models and metric against the detected task.

        Model comparison tolerates a candidate that fails, which is right when
        one of six models happens to blow up and wrong when the caller asked
        for a regressor on a classification problem — that is a mistake in the
        request and should be answered as one, not quietly dropped from the
        results. The ML layer's own validators do the checking, so there is
        exactly one definition of which model solves which task.

        Raises:
            UnknownModelError: If a requested model is not in the registry.
            IncompatibleTaskError: If a model does not solve the detected task.
            InvalidMetricError: If the metric does not exist for the task.
        """
        if options.primary_metric:
            get_metric(options.primary_metric, prepared.task_type)

        for model_name in options.models:
            validate_spec(
                ModelSpec(
                    model_name=model_name, primary_metric=options.primary_metric
                ),
                prepared.task_type,
                registry=self._registry,
            )

    def _explain(
        self,
        selection: ModelSelectionResult,
        prepared: PreparedDataset,
        options: ExperimentOptions,
        warnings: list[str],
    ) -> GlobalExplanation | None:
        """Explain the winning model, or record why there is no explanation.

        A model the explainers cannot handle already comes back as a
        structured "unavailable" result rather than an exception, so the only
        thing caught here is a genuine failure — and a failed explanation
        must not throw away a completed experiment.
        """
        if not options.explain:
            warnings.append("Explanation was skipped at the caller's request.")
            return None

        config = ExplanationConfig(
            max_reference_rows=self._settings.explanation_reference_rows,
            max_explanation_rows=self._settings.explanation_rows,
        )
        try:
            return explain_global(
                selection.final_model,
                prepared.X_train_raw,
                prepared.y_train,
                config=config,
                top_n=self._settings.explanation_top_features,
            )
        except ExplainabilityError as exc:
            logger.warning("Explanation failed for %s: %s", selection.selected_model_name, exc)
            warnings.append(
                "The model was trained and evaluated, but no explanation could "
                f"be produced: {exc}"
            )
            return None


def run_experiment(
    frame: pd.DataFrame,
    *,
    settings: Settings,
    store: ExperimentStore,
    dataset_service: DatasetProfilingService,
    artifact_store: ModelArtifactStore | None = None,
    target_column: str | None = None,
    models: Sequence[str] = (),
    dataset_label: str = "dataset",
    source_format: str | None = None,
    retain_artifacts: bool = False,
    **option_fields: Any,
) -> ExperimentRunResult:
    """Run one experiment on a DataFrame, as a single function call.

    A thin convenience wrapper over :class:`ExperimentRunner` for callers that
    have data and a few choices rather than an assembled request — a script, a
    notebook, or the agent's ``run_experiment`` tool. It returns the same
    structured result the HTTP endpoint serialises, so no caller ever needs to
    know about pandas, sklearn or where records are kept.

    This is the callable the agent layer is wired to. It satisfies that
    layer's executor protocol structurally, which is what lets ``agent/`` run
    real experiments without importing this package.

    ``source_format`` is bound by the caller that knows it — the request that
    ingested the upload — rather than guessed here, so a run the agent starts
    on a spreadsheet is recorded as having come from one. The agent itself
    never supplies it and never sees it.
    """
    options = ExperimentOptions(
        target_column=target_column, models=tuple(models), **option_fields
    ).validated(settings)
    runner = ExperimentRunner(
        settings, store, dataset_service, artifact_store=artifact_store
    )
    return runner.run_frame(
        frame,
        options,
        dataset_label=dataset_label,
        source_format=source_format or SOURCE_FORMAT,
        retain_artifacts=retain_artifacts,
    )
