"""Explaining a model — when the model still exists, and saying so when it does not.

This is the tool where it would be easiest to lie, so it is the one worth
reading closely.

Commit 7 decided not to persist fitted models. An experiment record holds the
dataset fingerprint, the configuration, the scores and — if one was computed
at run time — a stored global feature-importance summary. It does not hold the
estimator. That decision stands: nothing in this commit writes a model to
disk, and nothing here pretends one was written.

So the tool answers in three different ways, and the difference between them
is the point:

**Recomputed.** The experiment ran in *this* session, its fitted model is
still in memory (see :mod:`agent.tools.artifacts`), and the existing
explainability service is called for real — a fresh global explanation, or a
per-row explanation, computed now.

**From the stored record.** The experiment is older, the model is gone, but
the run stored a global importance summary when it happened. That summary is
real: it was produced by the same explainability layer at the time, and
returning it requires no model. It is labelled ``stored_record`` so nobody
mistakes it for a live computation.

**Unavailable.** Anything that genuinely needs the estimator and cannot have
it: a per-row explanation of an older run, or a global explanation of a run
that never stored one. The answer is a structured result with
``reason = "fitted_model_not_persisted"`` and an explanation of why. No SHAP
value is ever invented, estimated, or carried over from a different run.

The application does now persist a successful run's fitted pipeline, and
``POST /api/v1/experiments/{id}/predict`` uses it. **This tool deliberately
does not.** Reaching a stored artifact means deserialising a pickle, and this
package holds the tightest boundary in the project: it imports no filesystem
access, no ``scikit-learn`` and nothing from the backend, and it is handed the
models it may explain rather than going to find them. A live explanation of a
historical run would be a genuine feature, and it belongs on the side of that
boundary that already owns the artifact store — not inside the agent.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, is_dataclass
from typing import Any

from agent.observations import ensure_json_safe
from agent.schemas import (
    INTEGER,
    STRING,
    ArgumentField,
    ArgumentSchema,
)
from agent.tools.base import BaseTool, ToolResult

#: The two things this tool can be asked for.
SCOPE_GLOBAL = "global"
SCOPE_PREDICTION = "prediction"
SCOPES: tuple[str, ...] = (SCOPE_GLOBAL, SCOPE_PREDICTION)

#: Where an explanation came from. Reported on every available result so a
#: reader never has to guess whether SHAP ran just now.
SOURCE_RECOMPUTED = "recomputed"
SOURCE_STORED_RECORD = "stored_record"

#: Why an explanation could not be produced.
REASON_NOT_PERSISTED = "fitted_model_not_persisted"
REASON_NOT_RECORDED = "explanation_not_recorded"
REASON_UNKNOWN_EXPERIMENT = "unknown_experiment"

#: Feature entries carried into an observation.
MAX_REPORTED_FEATURES = 15
#: Largest row index a planner may ask to explain.
MAX_ROW_INDEX = 10_000

#: Said in full on every unavailable result, because a short reason code is
#: not enough for a reader deciding whether the system is broken.
NOT_PERSISTED_MESSAGE = (
    "This experiment's fitted model is not available. Experiment records "
    "store the dataset fingerprint, the configuration, the scores and the "
    "recorded feature importances, but deliberately not the trained "
    "estimator, so a model cannot be explained after the process that "
    "trained it has ended. Run the experiment again in this session to "
    "explain it live."
)


def _dataclass_payload(value: Any) -> dict[str, Any]:
    """Render an explanation object as plain values, whatever type it is.

    The explainability layer returns frozen dataclasses; the experiment store
    returns a record with its own renderer. Both are handled without this
    module importing either, and the result is passed through
    :func:`~agent.observations.ensure_json_safe` so an enum becomes its value
    and a numpy scalar becomes a number.
    """
    for method in ("as_dict", "to_dict", "model_dump"):
        renderer = getattr(value, method, None)
        if callable(renderer):
            rendered = renderer()
            if isinstance(rendered, dict):
                return ensure_json_safe(rendered)

    if is_dataclass(value) and not isinstance(value, type):
        rendered = ensure_json_safe(asdict(value))
        if isinstance(rendered, dict):
            return rendered

    return {}


def _importances(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Pull the ranked feature importances out of a rendered explanation."""
    entries = payload.get("feature_importances") or []
    return [
        {
            "feature": entry.get("feature"),
            "importance": entry.get("importance"),
            "rank": entry.get("rank"),
        }
        for entry in entries[:MAX_REPORTED_FEATURES]
        if isinstance(entry, dict)
    ]


def _contributions(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Pull the signed per-feature contributions out of a local explanation."""
    entries = payload.get("feature_contributions") or []
    return [
        {
            "feature": entry.get("feature"),
            "contribution": entry.get("contribution"),
            "direction": entry.get("direction"),
            "rank": entry.get("rank"),
        }
        for entry in entries[:MAX_REPORTED_FEATURES]
        if isinstance(entry, dict)
    ]


class ExplainExperimentTool(BaseTool):
    """Explain a model's behaviour using the existing explainability layer."""

    tool_name = "explain_experiment"
    tool_description = (
        "Explain which features drive a stored experiment's selected model, "
        "or why one row received the prediction it did. Explanations describe "
        "model behaviour and association, never causation. An experiment run "
        "in this session can be explained live; an older one can only report "
        "the feature importances recorded when it ran, because fitted models "
        "are not persisted."
    )

    def __init__(
        self,
        *,
        artifacts: Any = None,
        lookup: Any = None,
        explain_global: Callable[..., Any] | None = None,
        explain_prediction: Callable[..., Any] | None = None,
        top_n: int = MAX_REPORTED_FEATURES,
    ) -> None:
        """Wire the tool to the in-memory models, the store and the service.

        Args:
            artifacts: Cache of fitted models from experiments this run
                performed. Without it, only stored summaries are available.
            lookup: The existing experiment store, read only.
            explain_global: The existing global explanation function.
            explain_prediction: The existing per-row explanation function.
            top_n: How many features to keep in a result.
        """
        super().__init__()
        self._artifacts = artifacts
        self._lookup = lookup
        self._explain_global = explain_global
        self._explain_prediction = explain_prediction
        self._top_n = top_n

    @property
    def schema(self) -> ArgumentSchema:
        """Which experiment, what kind of explanation, and which row."""
        return ArgumentSchema(
            fields=(
                ArgumentField(
                    name="experiment_id",
                    type=STRING,
                    description=(
                        "The experiment to explain. Must be an id that "
                        "appeared in an earlier observation or exists in the "
                        "experiment store."
                    ),
                    required=True,
                    max_length=200,
                ),
                ArgumentField(
                    name="scope",
                    type=STRING,
                    description=(
                        "'global' for the features that drive the model "
                        "overall; 'prediction' for why one row scored as it "
                        "did. A per-row explanation needs the fitted model, "
                        "so it is only possible for an experiment run in this "
                        "session."
                    ),
                    default=SCOPE_GLOBAL,
                    choices=SCOPES,
                    max_length=40,
                ),
                ArgumentField(
                    name="row_index",
                    type=INTEGER,
                    description=(
                        "Which row of the experiment's own training data to "
                        "explain, when scope is 'prediction'."
                    ),
                    default=0,
                    minimum=0,
                    maximum=MAX_ROW_INDEX,
                ),
            )
        )

    # -- The three answers -------------------------------------------------

    def _live_global(self, experiment_id: str, artifacts: Any) -> ToolResult:
        """Compute a fresh global explanation for a model still in memory."""
        explanation = self._explain_global(
            artifacts.trained_model,
            artifacts.X_reference,
            artifacts.y_reference,
            top_n=self._top_n,
        )
        payload = _dataclass_payload(explanation)
        status = str(payload.get("status", "")) or "unknown"

        if status != "available":
            # The explainability layer already answers "I cannot explain this
            # model" as a structured result. It is passed through as one.
            return ToolResult.unavailable(
                str(payload.get("reason") or "explanation_unavailable"),
                experiment_id=experiment_id,
                scope=SCOPE_GLOBAL,
                message=str(
                    payload.get("reason")
                    or "The explainability layer could not explain this model."
                ),
                explanation_status=status,
            )

        return ToolResult(
            output={
                "status": "ok",
                "experiment_id": experiment_id,
                "scope": SCOPE_GLOBAL,
                "source": SOURCE_RECOMPUTED,
                "method": payload.get("method"),
                "model_name": payload.get("model_name"),
                "task_type": payload.get("task_type"),
                "sample_count": payload.get("sample_count"),
                "feature_importances": _importances(payload),
                "warnings": list(payload.get("warnings") or []),
                "interpretation_note": (
                    "Importance describes model behaviour and association, "
                    "not causation."
                ),
            }
        )

    def _live_prediction(
        self, experiment_id: str, artifacts: Any, row_index: int
    ) -> ToolResult:
        """Explain one row of a model still in memory."""
        reference = artifacts.X_reference
        try:
            available_rows = len(reference)
        except TypeError:  # pragma: no cover - defensive
            available_rows = 0
        if row_index >= available_rows:
            return ToolResult.unavailable(
                "row_out_of_range",
                experiment_id=experiment_id,
                scope=SCOPE_PREDICTION,
                message=(
                    f"Row {row_index} is outside this experiment's "
                    f"{available_rows} reference rows."
                ),
            )

        explanation = self._explain_prediction(
            artifacts.trained_model,
            reference.iloc[[row_index]],
            background=reference,
            top_n=self._top_n,
        )
        payload = _dataclass_payload(explanation)
        status = str(payload.get("status", "")) or "unknown"

        if status != "available":
            return ToolResult.unavailable(
                str(payload.get("reason") or "explanation_unavailable"),
                experiment_id=experiment_id,
                scope=SCOPE_PREDICTION,
                message=str(
                    payload.get("reason")
                    or "The explainability layer could not explain this prediction."
                ),
                explanation_status=status,
            )

        return ToolResult(
            output={
                "status": "ok",
                "experiment_id": experiment_id,
                "scope": SCOPE_PREDICTION,
                "source": SOURCE_RECOMPUTED,
                "row_index": row_index,
                "method": payload.get("method"),
                "model_name": payload.get("model_name"),
                "task_type": payload.get("task_type"),
                "prediction": payload.get("prediction"),
                "predicted_class": payload.get("predicted_class"),
                "base_value": payload.get("base_value"),
                "feature_contributions": _contributions(payload),
                "warnings": list(payload.get("warnings") or []),
                "interpretation_note": (
                    "Contributions describe how this model responded to this "
                    "row, not why the outcome happened."
                ),
            }
        )

    def _stored_summary(self, experiment_id: str) -> ToolResult:
        """Return the importances the run recorded, or say there are none."""
        if self._lookup is None or not self._lookup.exists(experiment_id):
            return ToolResult.unavailable(
                REASON_UNKNOWN_EXPERIMENT,
                experiment_id=experiment_id,
                message=f"No experiment is stored under the id '{experiment_id}'.",
            )

        record = self._lookup.get(experiment_id)
        payload = _dataclass_payload(record)
        explainability = payload.get("explainability") or {}
        importances = _importances(explainability)

        if not importances:
            return ToolResult.unavailable(
                REASON_NOT_RECORDED,
                experiment_id=experiment_id,
                scope=SCOPE_GLOBAL,
                message=(
                    "This experiment recorded no feature importances when it "
                    "ran, and its fitted model was not persisted, so there is "
                    "nothing to explain it from. " + NOT_PERSISTED_MESSAGE
                ),
            )

        return ToolResult(
            output={
                "status": "ok",
                "experiment_id": experiment_id,
                "scope": SCOPE_GLOBAL,
                "source": SOURCE_STORED_RECORD,
                "method": explainability.get("method"),
                "model_name": (payload.get("selection") or {}).get("selected_model"),
                "task_type": (payload.get("dataset") or {}).get("task_type"),
                "feature_importances": importances,
                "warnings": [
                    "These importances were recorded when the experiment ran. "
                    "The fitted model was not persisted, so nothing was "
                    "recomputed."
                ],
                "interpretation_note": (
                    "Importance describes model behaviour and association, "
                    "not causation."
                ),
            }
        )

    def run(self, arguments: Mapping[str, Any]) -> ToolResult:
        """Explain the experiment, or say precisely why it cannot be explained."""
        experiment_id = arguments["experiment_id"]
        scope = arguments.get("scope", SCOPE_GLOBAL)
        row_index = int(arguments.get("row_index", 0) or 0)

        artifacts = self._artifacts.get(experiment_id) if self._artifacts else None

        if artifacts is not None:
            if scope == SCOPE_PREDICTION and self._explain_prediction is not None:
                return self._live_prediction(experiment_id, artifacts, row_index)
            if scope == SCOPE_GLOBAL and self._explain_global is not None:
                return self._live_global(experiment_id, artifacts)

        if scope == SCOPE_PREDICTION:
            # There is no honest fallback. A per-row explanation is a
            # computation over the estimator, and the estimator is gone.
            return ToolResult.unavailable(
                REASON_NOT_PERSISTED,
                experiment_id=experiment_id,
                scope=SCOPE_PREDICTION,
                message=(
                    "A per-row explanation requires the fitted model. "
                    + NOT_PERSISTED_MESSAGE
                ),
                explainable_experiments=list(
                    self._artifacts.experiment_ids() if self._artifacts else ()
                ),
            )

        return self._stored_summary(experiment_id)


__all__ = [
    "MAX_REPORTED_FEATURES",
    "MAX_ROW_INDEX",
    "NOT_PERSISTED_MESSAGE",
    "REASON_NOT_PERSISTED",
    "REASON_NOT_RECORDED",
    "REASON_UNKNOWN_EXPERIMENT",
    "SCOPES",
    "SCOPE_GLOBAL",
    "SCOPE_PREDICTION",
    "SOURCE_RECOMPUTED",
    "SOURCE_STORED_RECORD",
    "ExplainExperimentTool",
]
