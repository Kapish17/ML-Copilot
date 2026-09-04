"""Schemas describing the experiment API.

These models are the public contract of the experiment endpoints, and they are
also the last line of defence for requirement that responses be JSON-safe. Each
section mirrors one section of the stored
:class:`~ml.experiments.run.ExperimentRun`, and responses are built by
validating ``record.to_dict()`` against :class:`ExperimentRecord` — so a
sklearn estimator, a numpy array, a DataFrame or a SHAP explainer could not
reach a client even if something upstream tried to put one there: it would fail
validation instead of being serialised.

Free-form sections (the preprocessing configuration, per-column decisions, the
metric dictionaries) are typed as plain JSON containers rather than being
re-declared field by field. Their contents belong to the ML layer, and copying
them here would create a second definition to keep in step with the first.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

JsonValue = Any


# ---------------------------------------------------------------------------
# Requests
#
# The run endpoint's request is multipart — a file plus configuration fields —
# and is declared in ``app/api/v1/experiment_form.py``, because FastAPI cannot
# flatten a model into a form that also carries a file. JSON-bodied requests
# are modelled here as usual.
# ---------------------------------------------------------------------------


class ExperimentCompareRequest(BaseModel):
    """The experiments to rank against one another."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "experiment_ids": [
                    "exp_84a8d53a1f5f_20260828T134457Z_e420",
                    "exp_90e8e969804e_20260828T134459Z_de07",
                ]
            }
        },
    )

    experiment_ids: list[str] = Field(
        ...,
        min_length=2,
        description=(
            "At least two stored experiment ids. They must share one task and "
            "one primary metric; runs judged by different metrics are refused "
            "rather than ranked against each other."
        ),
    )


# ---------------------------------------------------------------------------
# Record sections
# ---------------------------------------------------------------------------


class ExperimentDataset(BaseModel):
    """How the data was identified, without holding any of it."""

    fingerprint: str = Field(
        ..., description="Content hash of the dataset; independent of filename."
    )
    fingerprint_algorithm: str
    row_count: int
    column_count: int
    target_column: str
    task_type: str
    columns: list[str]
    dtypes: dict[str, str]
    source_format: str | None = None
    data_quality_issues: list[JsonValue] = Field(default_factory=list)


class ExperimentPreprocessing(BaseModel):
    """What became a feature, what did not, and how the split was made."""

    config: dict[str, JsonValue]
    feature_groups: dict[str, list[str]]
    selected_columns: list[str]
    excluded_columns: list[str]
    identifier_columns: list[str]
    transformed_feature_names: list[str]
    column_decisions: list[dict[str, JsonValue]]
    train_row_count: int
    test_row_count: int
    test_size: float
    random_state: int
    stratified: bool
    stratification_note: str | None = None
    rows_dropped_missing_target: int


class ExperimentSelection(BaseModel):
    """Which models were tried, which won, and on what data it was judged."""

    strategy: str
    folds: int | None = None
    primary_metric: str
    primary_metric_direction: str
    candidate_models: list[str]
    candidates: list[dict[str, JsonValue]]
    selected_model: str
    selection_score: float | None = None
    selection_score_std: float | None = None
    scored_on: str = Field(
        ..., description="'training_folds' or 'held_out_test_set'."
    )
    uses_test_data: bool


class ExperimentEvaluation(BaseModel):
    """The single measurement on data the model had never seen."""

    primary_metric: str
    primary_metric_value: float | None = None
    metrics: dict[str, JsonValue]
    unavailable_metrics: dict[str, str] = Field(
        default_factory=dict,
        description="Metrics that could not be computed, and why.",
    )
    baseline_identifier: str | None = None
    baseline_metrics: dict[str, JsonValue] = Field(default_factory=dict)
    baseline_comparison: dict[str, JsonValue] = Field(default_factory=dict)
    classification_details: dict[str, JsonValue] | None = None
    test_row_count: int
    is_unbiased: bool = Field(
        ...,
        description=(
            "True when the test set played no part in choosing the model. "
            "False under the holdout strategy, where the score is optimistic."
        ),
    )


class ExperimentExplainability(BaseModel):
    """What the explanation found, or why there is none."""

    status: str
    method: str
    explainer: str | None = None
    aggregation: str | None = None
    explained_output: str | None = None
    feature_importances: list[dict[str, JsonValue]] = Field(default_factory=list)
    sample_count: int = 0
    feature_count: int = 0
    reason: str | None = None
    warnings: list[str] = Field(default_factory=list)


class ExperimentModelArtifact(BaseModel):
    """That this run's winning model was persisted, and what it expects.

    A note about what happened when the run finished, not a promise about now.
    Ask `GET /api/v1/experiments/{id}/model` for whether a prediction can
    actually be made today — an artifact can be deleted and a volume can be
    wiped, and this section would still say a model was written.

    Column **names**, kinds and counts only. No cell value and no row: the
    uploaded dataset is not stored, and neither is any part of it.
    """

    stored: bool
    model_name: str
    task_type: str
    target_column: str
    feature_names: list[str] = Field(default_factory=list)
    feature_count: int = 0
    class_labels: list[str] = Field(default_factory=list)
    artifact_schema_version: str = ""
    created_at: str | None = None


class ExperimentEnvironment(BaseModel):
    """What a reproduction attempt would need.

    Interpreter, platform, library versions and the seed — and deliberately no
    hostname, username, path or environment variable.
    """

    python_version: str
    platform: str
    packages: dict[str, str]
    random_state: int | None = None


class ExperimentRecord(BaseModel):
    """One stored experiment, exactly as the store holds it."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str
    experiment_id: str
    configuration_hash: str
    created_at: datetime
    name: str
    description: str | None = None
    tags: list[str] = Field(default_factory=list)
    dataset: ExperimentDataset
    preprocessing: ExperimentPreprocessing
    selection: ExperimentSelection
    evaluation: ExperimentEvaluation
    explainability: ExperimentExplainability | None = None
    model_artifact: ExperimentModelArtifact | None = None
    environment: ExperimentEnvironment


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


class ExperimentExecution(BaseModel):
    """How the run was carried out, as opposed to what it found."""

    duration_seconds: float
    stored: bool = Field(
        ..., description="True when the record was written to the store."
    )
    mode: str = Field(
        "synchronous",
        description=(
            "Execution mode. Only 'synchronous' exists: no queue, worker or "
            "background execution is implemented."
        ),
    )


class ExperimentRunResponse(ExperimentRecord):
    """A completed experiment, plus what the run decided or could not do."""

    warnings: list[str] = Field(
        default_factory=list,
        description=(
            "Choices made on the caller's behalf and anything that could not "
            "be done, such as a target picked by convention or an explanation "
            "that was unavailable."
        ),
    )
    execution: ExperimentExecution


class ExperimentHeadline(BaseModel):
    """A one-line view of a run, for listings."""

    experiment_id: str
    created_at: datetime
    name: str
    dataset_fingerprint: str
    task_type: str
    target_column: str
    selected_model: str
    strategy: str
    primary_metric: str
    selection_score: float | None = None
    test_score: float | None = None


class ExperimentListResponse(BaseModel):
    """Stored experiments matching a query."""

    count: int = Field(..., description="Number of experiments returned.")
    limit: int = Field(..., description="Page size the query was capped at.")
    experiments: list[ExperimentHeadline] = Field(default_factory=list)


class ComparisonRowModel(BaseModel):
    """One run's line in a comparison, ranked best first."""

    experiment_id: str
    created_at: str
    name: str
    model_name: str
    strategy: str
    selection_score: float | None = None
    selection_score_std: float | None = None
    test_score: float | None = None
    baseline_score: float | None = None
    improvement: float | None = None


class ExperimentComparisonResponse(BaseModel):
    """Several runs ranked on the one metric they share."""

    task_type: str
    primary_metric: str
    direction: str = Field(
        ..., description="'higher_is_better' or 'lower_is_better'."
    )
    higher_is_better: bool
    run_count: int
    best_experiment_id: str | None = None
    runs: list[ComparisonRowModel] = Field(default_factory=list)
    table: str = Field(
        ..., description="The same ranking rendered as a readable text table."
    )


class ModelInfo(BaseModel):
    """One model the registry can build."""

    identifier: str
    display_name: str
    task_type: str
    supports_probabilities: bool | None = None
    supports_random_state: bool | None = None
    default_parameters: dict[str, JsonValue] = Field(default_factory=dict)
    description: str | None = None


class ExperimentCapabilitiesResponse(BaseModel):
    """What may be asked for in an experiment request.

    Exists so a client does not have to hard-code the model identifiers,
    metrics or limits that the server already knows.
    """

    models: list[ModelInfo]
    metrics: dict[str, list[str]] = Field(
        ..., description="Available metric keys per task type."
    )
    strategies: list[str]
    sort_keys: list[str]
    limits: dict[str, JsonValue]
    supported_dataset_extensions: list[str]
