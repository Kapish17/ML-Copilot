"""Running an experiment, through the runner that already exists.

There is no training code in this module and there must never be. The whole
pipeline — profiling, leakage-safe splitting, candidate validation,
cross-validated selection, one untouched-test evaluation, SHAP, and writing
the record — is the experiment runner's, and this tool's job is to decide
whether the planner may ask for it and with what.

**What a planner may choose is a short list.** A dataset by name, a target
column, up to a few models *by identifier from the existing registry*, a
metric, a fold count, and a label. That is the whole surface.

**What a planner may not do is more interesting.** It cannot supply an
estimator, a class name, a dotted import path, a hyperparameter object, a
callable, a preprocessing pipeline, a file path or a random seed dressed up as
a model. None of those are declared fields, and an undeclared field is a
rejected call rather than an ignored one. A model name that is not already in
the registry is rejected for the same reason a misspelt one is: the allowed
values are read from the registry itself, so "what the agent may train" and
"what the system already supports" cannot drift apart.

The fitted model that comes out of a run is kept in memory only, for the
explanation tool, and only for this run. See :mod:`agent.tools.artifacts` —
nothing about Commit 7's decision not to persist models has changed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from agent.schemas import (
    INTEGER,
    STRING,
    STRING_LIST,
    ArgumentField,
    ArgumentSchema,
)
from agent.tools.base import BaseTool, ToolResult

#: Models one call may ask to compare. The runner has its own limit from
#: application settings; this is a second, smaller bound so a planner cannot
#: turn one question into the longest run the settings would permit.
MAX_MODELS_PER_CALL = 4
#: Bounds on the fold count a planner may request.
MIN_FOLDS = 2
MAX_FOLDS = 10
#: Feature importances carried into an observation.
MAX_REPORTED_FEATURES = 10


def summarise_run(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Reduce a stored experiment record to what a planner can act on.

    The full record carries every fold's metrics, the whole preprocessing
    configuration and the environment. A planner needs the identity, the
    winner, the scores and the headline importances; the rest would spend the
    context budget without changing any decision. The identifier is preserved
    exactly, because it is what the answer will cite and what a person will
    look the run up by.
    """
    dataset = payload.get("dataset") or {}
    selection = payload.get("selection") or {}
    evaluation = payload.get("evaluation") or {}
    explainability = payload.get("explainability") or {}

    summary: dict[str, Any] = {
        "status": "ok",
        "experiment_id": payload.get("experiment_id"),
        "name": payload.get("name"),
        "created_at": payload.get("created_at"),
        "task_type": dataset.get("task_type"),
        "target_column": dataset.get("target_column"),
        "dataset_fingerprint": dataset.get("fingerprint"),
        "row_count": dataset.get("row_count"),
        "selected_model": selection.get("selected_model"),
        "selection_strategy": selection.get("strategy"),
        "selection_score": selection.get("selection_score"),
        "primary_metric": evaluation.get("primary_metric"),
        "primary_metric_value": evaluation.get("primary_metric_value"),
        "test_metrics": evaluation.get("metrics") or {},
        "warnings": list(payload.get("warnings") or []),
    }

    candidates = selection.get("candidates") or selection.get("comparison") or []
    if isinstance(candidates, list):
        summary["candidates"] = [
            {
                "model": entry.get("model_name") or entry.get("model"),
                "score": entry.get("score") or entry.get("mean_score"),
                "std": entry.get("std") or entry.get("standard_deviation"),
                "status": entry.get("status"),
            }
            for entry in candidates
            if isinstance(entry, dict)
        ]

    importances = explainability.get("feature_importances") or []
    if isinstance(importances, list) and importances:
        summary["top_features"] = [
            {
                "feature": entry.get("feature"),
                "importance": entry.get("importance"),
                "rank": entry.get("rank"),
            }
            for entry in importances[:MAX_REPORTED_FEATURES]
            if isinstance(entry, dict)
        ]
        summary["explanation_method"] = explainability.get("method")
    else:
        summary["top_features"] = []
        summary["explanation_method"] = explainability.get("method")

    return summary


class RunExperimentTool(BaseTool):
    """Run one experiment through the existing runner and store it."""

    tool_name = "run_experiment"
    tool_description = (
        "Train and compare models on one of the datasets available to this "
        "session, using cross-validation to select a winner, evaluate it once "
        "on a held-out test set, and store the run. Returns the stored "
        "experiment: its id, the selected model, the scores and the headline "
        "feature importances. Use this to answer which model performs best, "
        "or how well a target can be predicted. Only models the system "
        "already supports can be requested."
    )

    def __init__(
        self,
        source: Any,
        executor: Callable[..., Any],
        *,
        available_models: Callable[[], Sequence[str]] | Sequence[str] = (),
        available_metrics: Callable[[], Sequence[str]] | Sequence[str] = (),
        artifacts: Any = None,
    ) -> None:
        """Wire the tool to its dataset names, the runner and the registries.

        Args:
            source: Where a named dataset comes from.
            executor: The existing runner, as a callable taking the dataset
                and safe keyword options. Supplied by the caller so this
                package need not import the service layer.
            available_models: The model identifiers a planner may name —
                read from the existing model registry, not listed here.
            available_metrics: The metric names a planner may name.
            artifacts: Optional cache that keeps the fitted model in memory
                for the explanation tool. In-memory only; see
                :mod:`agent.tools.artifacts`.
        """
        super().__init__()
        self._source = source
        self._executor = executor
        self._models = available_models
        self._metrics = available_metrics
        self._artifacts = artifacts

    @staticmethod
    def _resolve(values: Callable[[], Sequence[str]] | Sequence[str]) -> list[str]:
        """Read an allowed-value list that may be a callable or a sequence."""
        return list(values() if callable(values) else values)

    @property
    def schema(self) -> ArgumentSchema:
        """The complete, deliberately short set of choices a planner has."""
        return ArgumentSchema(
            fields=(
                ArgumentField(
                    name="dataset",
                    type=STRING,
                    description="Name of the dataset to run the experiment on.",
                    required=True,
                    max_length=200,
                    choices_provider=lambda: list(self._source.names()),
                ),
                ArgumentField(
                    name="target_column",
                    type=STRING,
                    description=(
                        "Column to predict. Omit only if the dataset's "
                        "convention makes it obvious; the run will say what "
                        "it picked."
                    ),
                    max_length=200,
                ),
                ArgumentField(
                    name="models",
                    type=STRING_LIST,
                    description=(
                        "Model identifiers to compare. Must be models the "
                        "system already supports; anything else is rejected. "
                        "Omit to let the system choose the candidates that "
                        "suit the detected task."
                    ),
                    max_items=MAX_MODELS_PER_CALL,
                    choices_provider=lambda: self._resolve(self._models),
                ),
                ArgumentField(
                    name="primary_metric",
                    type=STRING,
                    description="Metric to judge the models by.",
                    max_length=80,
                    choices_provider=lambda: self._resolve(self._metrics),
                ),
                ArgumentField(
                    name="folds",
                    type=INTEGER,
                    description="Cross-validation folds.",
                    minimum=MIN_FOLDS,
                    maximum=MAX_FOLDS,
                ),
                ArgumentField(
                    name="name",
                    type=STRING,
                    description="A short label for the run, for later reference.",
                    max_length=120,
                ),
            )
        )

    def run(self, arguments: Mapping[str, Any]) -> ToolResult:
        """Run the experiment and summarise the stored record."""
        dataset = arguments["dataset"]
        try:
            frame = self._source.get(dataset)
        except KeyError:
            return ToolResult.unavailable(
                "unknown_dataset",
                message=(
                    f"No dataset named '{dataset}' is available to this session. "
                    f"Available: {', '.join(self._source.names()) or '(none)'}."
                ),
            )

        options: dict[str, Any] = {"dataset_label": dataset}
        for field_name in ("target_column", "primary_metric", "folds", "name"):
            if arguments.get(field_name) is not None:
                options[field_name] = arguments[field_name]
        if arguments.get("models"):
            options["models"] = tuple(arguments["models"])
        if self._artifacts is not None:
            options["retain_artifacts"] = True

        result = self._executor(frame, **options)
        payload = result.as_dict()
        summary = summarise_run(payload)

        # Kept in memory, for this run only, so the explanation tool can work
        # on a model that has not been written down anywhere.
        artifacts = getattr(result, "artifacts", None)
        experiment_id = summary.get("experiment_id")
        if self._artifacts is not None and artifacts is not None and experiment_id:
            self._artifacts.put(experiment_id, artifacts)
            summary["explainable_now"] = True
        else:
            summary["explainable_now"] = False

        return ToolResult(output=summary)


__all__ = [
    "MAX_FOLDS",
    "MAX_MODELS_PER_CALL",
    "MAX_REPORTED_FEATURES",
    "MIN_FOLDS",
    "RunExperimentTool",
    "summarise_run",
]
