"""Naming datasets, and profiling the one that was named.

The security design of this tool is in its *input*, not its output. A planner
does not say where a dataset is; it says which registered dataset it means.
:class:`InMemoryDatasetSource` is the whole of the addressing scheme — a
mapping from a short name to data the application already holds — and there is
no other way for a dataset to enter a tool.

That is what makes "no arbitrary filesystem access" a structural fact rather
than a filter. There is no path parsing to defeat, no allowlist of directories
to escape, no ``..`` to normalise, because a path is never accepted in the
first place. ``"../../etc/passwd"``, ``"C:\\Users\\me\\keys.txt"`` and
``"htpasswd"`` are all simply names that were never registered, and all three
get the same answer: no such dataset, here are the ones there are.

The profiling itself is entirely the existing dataset service's. This module
selects the fields worth putting in an observation and drops the rest — a full
profile of a wide dataset would spend a large part of the context budget on
column statistics no planner is going to read.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from agent.schemas import ArgumentField, ArgumentSchema, STRING
from agent.tools.base import BaseTool, ProfilingService, ToolResult

#: Columns summarised individually before the list is truncated. A planner
#: needs the shape of the data and the notable columns, not all 900 of them.
MAX_PROFILED_COLUMNS = 40
#: Quality findings carried into an observation.
MAX_QUALITY_ISSUES = 12


class InMemoryDatasetSource:
    """The datasets one agent run may name.

    A deliberately tiny object: a name-to-data mapping and nothing else. It
    holds whatever standardised in-memory dataset the application already
    produced — from an upload, a fixture, or a caller with data in hand — and
    it never reads, writes or resolves anything on a filesystem.
    """

    def __init__(self, datasets: Mapping[str, Any] | None = None) -> None:
        """Register the datasets available to this run."""
        self._datasets: dict[str, Any] = dict(datasets or {})

    def add(self, name: str, frame: Any) -> InMemoryDatasetSource:
        """Register one dataset under a name."""
        self._datasets[name] = frame
        return self

    def names(self) -> Sequence[str]:
        """Every registered dataset name."""
        return tuple(self._datasets)

    def get(self, name: str) -> Any:
        """Return the dataset registered under a name.

        Raises:
            KeyError: If nothing is registered under that name.
        """
        return self._datasets[name]


def _model_payload(value: Any) -> Any:
    """Render a profile object as plain values, whatever type it is.

    The profiling service returns Pydantic models today. Reading them through
    ``model_dump`` when it exists, and falling back to ``as_dict`` or the
    object itself, keeps this tool from being coupled to that choice.
    """
    for method in ("model_dump", "as_dict", "to_dict"):
        renderer = getattr(value, method, None)
        if callable(renderer):
            return renderer(mode="json") if method == "model_dump" else renderer()
    return value


def summarise_profile(profile: Any) -> dict[str, Any]:
    """Reduce a full dataset profile to what a planner can act on.

    Keeps the shape, the target, the inferred task, the quality findings and a
    per-column summary; drops the distributions, histograms and per-column
    statistics that would dominate the context budget.
    """
    payload = _model_payload(profile)
    if not isinstance(payload, dict):  # pragma: no cover - defensive
        return {"status": "unreadable_profile"}

    dataset = payload.get("dataset") or {}
    target = payload.get("target")
    quality = payload.get("quality") or {}
    columns = payload.get("columns") or []

    summary: dict[str, Any] = {
        "status": "ok",
        "dataset_label": payload.get("filename"),
        "rows": dataset.get("row_count"),
        "columns": dataset.get("column_count"),
        "duplicate_row_count": dataset.get("duplicate_row_count"),
        "missing_cell_percentage": dataset.get("missing_cell_percentage"),
        "column_type_counts": dataset.get("column_type_counts") or {},
    }

    if isinstance(target, dict):
        summary["target"] = {
            "name": target.get("name"),
            "inferred_type": target.get("inferred_type"),
            "missing_count": target.get("missing_count"),
            "class_balance": target.get("class_balance"),
        }
        summary["inferred_task"] = target.get("task_suggestion")
        summary["inferred_task_reason"] = target.get("task_reason")
    else:
        summary["target"] = None
        # Said explicitly rather than left absent: "no target was given" and
        # "the task could not be inferred" are different things, and a planner
        # deciding whether to run an experiment needs to tell them apart.
        summary["inferred_task"] = None
        summary["inferred_task_reason"] = (
            "No target column was given, so no modelling task was inferred."
        )

    issues = quality.get("issues") or []
    summary["quality_issue_count"] = quality.get("issue_count", len(issues))
    summary["quality_issues"] = [
        {
            "code": issue.get("code"),
            "severity": issue.get("severity"),
            "column": issue.get("column"),
            "message": issue.get("message"),
        }
        for issue in issues[:MAX_QUALITY_ISSUES]
    ]

    summary["features"] = [
        {
            "name": column.get("name"),
            "inferred_type": column.get("inferred_type"),
            "missing_percentage": column.get("missing_percentage"),
            "unique_count": column.get("unique_count"),
            "is_constant": column.get("is_constant"),
        }
        for column in columns[:MAX_PROFILED_COLUMNS]
    ]
    if len(columns) > MAX_PROFILED_COLUMNS:
        summary["features_truncated"] = True
        summary["features_omitted"] = len(columns) - MAX_PROFILED_COLUMNS

    return summary


class DatasetProfileTool(BaseTool):
    """Profile a registered dataset using the existing profiling service."""

    tool_name = "dataset_profile"
    tool_description = (
        "Profile one of the datasets available to this session: its shape, "
        "column types, missing values, data-quality findings and — when a "
        "target column is named — the modelling task that target implies. "
        "Call this before running an experiment when the task type or the "
        "column names are not already known. Datasets are addressed by name; "
        "file paths and URLs are not accepted."
    )

    def __init__(self, source: Any, profiler: ProfilingService) -> None:
        """Wire the tool to the dataset names and the profiling service."""
        super().__init__()
        self._source = source
        self._profiler = profiler

    @property
    def schema(self) -> ArgumentSchema:
        """Two names: which dataset, and optionally which target column.

        ``dataset`` is restricted to the registered names, resolved at call
        time so the planner is told what actually exists rather than what
        existed when the tool was built.
        """
        return ArgumentSchema(
            fields=(
                ArgumentField(
                    name="dataset",
                    type=STRING,
                    description="Name of the dataset to profile.",
                    required=True,
                    max_length=200,
                    choices_provider=lambda: list(self._source.names()),
                ),
                ArgumentField(
                    name="target_column",
                    type=STRING,
                    description=(
                        "Optional column to analyse as the prediction target. "
                        "Naming it is what produces the inferred task type."
                    ),
                    max_length=200,
                ),
            )
        )

    def run(self, arguments: Mapping[str, Any]) -> ToolResult:
        """Profile the named dataset."""
        name = arguments["dataset"]
        try:
            frame = self._source.get(name)
        except KeyError:
            return ToolResult.unavailable(
                "unknown_dataset",
                message=(
                    f"No dataset named '{name}' is available to this session. "
                    f"Available: {', '.join(self._source.names()) or '(none)'}."
                ),
            )

        profile = self._profiler.profile_frame(
            frame,
            filename=name,
            target_column=arguments.get("target_column"),
        )
        return ToolResult(output=summarise_profile(profile))


__all__ = [
    "MAX_PROFILED_COLUMNS",
    "MAX_QUALITY_ISSUES",
    "DatasetProfileTool",
    "InMemoryDatasetSource",
    "summarise_profile",
]
