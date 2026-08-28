"""The storage contract, and how history is queried.

Everything above this module talks to ``ExperimentStore``, never to a file
path. That is the whole point of the interface: today the only implementation
writes JSON to a directory, and tomorrow one could write to PostgreSQL or hand
runs to MLflow — **neither of which is implemented** — without a single caller
changing.

Sorting by the primary metric is the one operation that can quietly go wrong.
An F1 of 0.82 and an RMSE of 5276 are not comparable, and ordering them
together would produce a confident nonsense ranking. So the sort refuses a
mixed set rather than guessing, and where a set *is* comparable it takes the
direction from the same metric definitions the rest of the project uses.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from ml.errors import IncomparableExperimentsError
from ml.evaluation.metrics import MetricDirection, get_metric
from ml.experiments.run import ExperimentRun
from ml.features.types import TaskType


class ExperimentSortKey(str, Enum):
    """How a list of runs should be ordered."""

    CREATED_AT = "created_at"
    PRIMARY_METRIC = "primary_metric"
    MODEL_NAME = "model_name"


@dataclass(frozen=True)
class ExperimentQuery:
    """Which runs to return, and in what order.

    Every filter is optional and they combine with "and". ``descending`` means
    "best or newest first": for ``CREATED_AT`` the most recent run leads, for
    ``MODEL_NAME`` the ordering is reversed alphabetically, and for
    ``PRIMARY_METRIC`` the best-scoring run leads — which is the largest F1 but
    the smallest RMSE, read from the metric's own declared direction.
    """

    dataset_fingerprint: str | None = None
    target_column: str | None = None
    task_type: str | None = None
    model_name: str | None = None
    selection_strategy: str | None = None
    primary_metric: str | None = None
    tags: tuple[str, ...] = ()
    sort_by: ExperimentSortKey = ExperimentSortKey.CREATED_AT
    descending: bool = True
    limit: int | None = None

    def matches(self, run: ExperimentRun) -> bool:
        """Return True when a run satisfies every filter set on this query."""
        checks = (
            (self.dataset_fingerprint, run.dataset.fingerprint),
            (self.target_column, run.dataset.target_column),
            (self.task_type, run.task_type),
            (self.model_name, run.selected_model),
            (self.selection_strategy, run.selection.strategy),
            (self.primary_metric, run.primary_metric),
        )
        if any(wanted is not None and wanted != actual for wanted, actual in checks):
            return False
        return set(self.tags).issubset(set(run.tags))


def shared_metric_direction(runs: Sequence[ExperimentRun]) -> MetricDirection:
    """Return the shared direction of the runs' primary metric.

    Also the compatibility check: a set of runs judged by different metrics, or
    solving different tasks, cannot be ranked together at all.

    Raises:
        IncomparableExperimentsError: If the runs do not share one metric.
    """
    metrics = {run.primary_metric for run in runs}
    if len(metrics) > 1:
        raise IncomparableExperimentsError(
            "These runs cannot be ranked together: they were judged by "
            "different metrics (" + ", ".join(sorted(metrics)) + "). Filter by "
            "task_type or primary_metric first.",
            details={"primary_metrics": sorted(metrics)},
        )

    metric = metrics.pop()
    tasks = {run.task_type for run in runs}
    if len(tasks) > 1:
        raise IncomparableExperimentsError(
            "These runs cannot be ranked together: they solve different tasks "
            "(" + ", ".join(sorted(tasks)) + ").",
            details={"task_types": sorted(tasks)},
        )
    return get_metric(metric, TaskType(tasks.pop())).direction


def sort_runs(
    runs: Sequence[ExperimentRun], query: ExperimentQuery
) -> tuple[ExperimentRun, ...]:
    """Order runs according to a query.

    Args:
        runs: The runs to order.
        query: The sort key and direction.

    Returns:
        tuple[ExperimentRun, ...]: The ordered runs.

    Raises:
        IncomparableExperimentsError: If sorting by primary metric was asked
            for and the runs do not share one metric and task.
    """
    if not runs:
        return ()

    if query.sort_by is ExperimentSortKey.MODEL_NAME:
        ordered = sorted(runs, key=lambda run: run.selected_model)
        return tuple(reversed(ordered)) if query.descending else tuple(ordered)

    if query.sort_by is ExperimentSortKey.PRIMARY_METRIC:
        direction = shared_metric_direction(runs)
        higher_is_better = direction is MetricDirection.HIGHER_IS_BETTER

        def rank(run: ExperimentRun) -> tuple[int, float]:
            """Order by score, keeping unscored runs at the end."""
            value = run.evaluation.primary_metric_value
            if value is None:
                return (1, 0.0)
            best_first = -value if higher_is_better else value
            return (0, best_first if query.descending else -best_first)

        return tuple(sorted(runs, key=rank))

    ordered = sorted(runs, key=lambda run: run.created_at)
    return tuple(reversed(ordered)) if query.descending else tuple(ordered)


def apply_query(
    runs: Sequence[ExperimentRun], query: ExperimentQuery | None
) -> tuple[ExperimentRun, ...]:
    """Filter, sort and truncate a set of runs.

    Shared by every store implementation so that filtering behaves identically
    whatever the backend.
    """
    active = query or ExperimentQuery()
    matched = [run for run in runs if active.matches(run)]
    ordered = sort_runs(matched, active)
    return ordered[: active.limit] if active.limit is not None else ordered


@runtime_checkable
class ExperimentStore(Protocol):
    """Where experiment history lives.

    The ML layer depends on this and nothing more, so a filesystem, a database
    or a tracking service are interchangeable behind it.
    """

    def save(self, run: ExperimentRun) -> str:
        """Persist a run and return its identifier."""
        ...

    def get(self, experiment_id: str) -> ExperimentRun:
        """Load one run, raising if it is missing or unreadable."""
        ...

    def exists(self, experiment_id: str) -> bool:
        """Return whether a run is stored under this identifier."""
        ...

    def list(self, query: ExperimentQuery | None = None) -> tuple[ExperimentRun, ...]:
        """Return the stored runs matching a query."""
        ...

    def delete(self, experiment_id: str) -> bool:
        """Remove a run, returning whether there was one to remove."""
        ...
