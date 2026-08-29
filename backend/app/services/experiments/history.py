"""Reading and comparing stored experiments.

Every question this service answers is answered by Commit 7's storage layer:
:class:`~ml.experiments.store.ExperimentQuery` does the filtering and sorting,
and :func:`~ml.experiments.comparison.compare_experiments` does the ranking.
There is no second query language and no separate database access here — the
service exists to bound what an HTTP caller may ask for and to keep the routes
free of ML imports.

**No database is implemented.** Records are the local JSON files described in
``ml/experiments/local_store.py``.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.core.config import Settings
from ml.errors import ConfigurationError
from ml.experiments import (
    ExperimentComparison,
    ExperimentQuery,
    ExperimentRun,
    ExperimentSortKey,
    compare_experiments,
)
from ml.experiments.store import ExperimentStore

#: Sort keys an HTTP caller may name, mirroring ``ExperimentSortKey``.
SORT_KEYS = tuple(key.value for key in ExperimentSortKey)
#: Sort directions, in the words a query string uses.
SORT_ORDERS = ("desc", "asc")


class ExperimentHistoryService:
    """Query, fetch and compare the experiments a store holds."""

    def __init__(self, settings: Settings, store: ExperimentStore) -> None:
        """Bind the service to its settings and storage backend."""
        self._settings = settings
        self._store = store

    def get(self, experiment_id: str) -> ExperimentRun:
        """Return one stored experiment.

        Args:
            experiment_id: The run's identifier.

        Returns:
            ExperimentRun: The stored record.

        Raises:
            InvalidExperimentIdError: If the identifier is malformed or would
                point outside the store.
            ExperimentNotFoundError: If nothing is stored under it.
            MalformedExperimentError: If the stored record is unreadable.
        """
        return self._store.get(experiment_id)

    def list(
        self,
        *,
        dataset_fingerprint: str | None = None,
        target_column: str | None = None,
        task_type: str | None = None,
        model_name: str | None = None,
        selection_strategy: str | None = None,
        primary_metric: str | None = None,
        tags: Sequence[str] = (),
        sort_by: str | None = None,
        order: str | None = None,
        limit: int | None = None,
    ) -> tuple[ExperimentRun, ...]:
        """Return stored experiments matching the given filters.

        Filters combine with "and"; every one of them is optional. Sorting by
        ``primary_metric`` reads each metric's own declared direction, so
        "best first" means the largest F1 but the smallest RMSE.

        Raises:
            ConfigurationError: If a sort key, order or limit is unusable.
        """
        query = ExperimentQuery(
            dataset_fingerprint=dataset_fingerprint or None,
            target_column=target_column or None,
            task_type=task_type or None,
            model_name=model_name or None,
            selection_strategy=selection_strategy or None,
            primary_metric=primary_metric or None,
            tags=tuple(tags),
            sort_by=self._sort_key(sort_by),
            descending=self._descending(order),
            limit=self._limit(limit),
        )
        return self._store.list(query)

    def compare(self, experiment_ids: Sequence[str]) -> ExperimentComparison:
        """Rank several stored experiments against each other.

        Args:
            experiment_ids: The runs to compare. Each must exist.

        Returns:
            ExperimentComparison: The ranking and its shared metric.

        Raises:
            ConfigurationError: If too few or too many ids were given.
            ExperimentNotFoundError: If one of the ids is not stored.
            IncomparableExperimentsError: If the runs do not share one metric
                and one task, which is what stops an RMSE being ranked against
                an F1.
        """
        unique = tuple(dict.fromkeys(str(item).strip() for item in experiment_ids if str(item).strip()))
        if len(unique) < 2:
            raise ConfigurationError(
                "Comparing experiments needs at least two distinct experiment ids.",
                details={"experiment_id_count": len(unique)},
            )
        if len(unique) > self._settings.max_comparison_experiments:
            raise ConfigurationError(
                f"At most {self._settings.max_comparison_experiments} experiments "
                f"may be compared at once, got {len(unique)}.",
                details={
                    "experiment_id_count": len(unique),
                    "maximum": self._settings.max_comparison_experiments,
                },
            )
        return compare_experiments([self._store.get(item) for item in unique])

    # -- Query parameter handling -----------------------------------------

    def _sort_key(self, sort_by: str | None) -> ExperimentSortKey:
        """Resolve a sort key name, defaulting to newest first."""
        if not sort_by:
            return ExperimentSortKey.CREATED_AT
        try:
            return ExperimentSortKey(sort_by)
        except ValueError as exc:
            raise ConfigurationError(
                f"Unknown sort key '{sort_by}'. Available: " + ", ".join(SORT_KEYS) + ".",
                details={"sort_by": sort_by, "available": list(SORT_KEYS)},
            ) from exc

    def _descending(self, order: str | None) -> bool:
        """Resolve a sort order, defaulting to 'best or newest first'."""
        if not order:
            return True
        normalised = order.strip().lower()
        if normalised not in SORT_ORDERS:
            raise ConfigurationError(
                f"Unknown sort order '{order}'. Available: " + ", ".join(SORT_ORDERS) + ".",
                details={"order": order, "available": list(SORT_ORDERS)},
            )
        return normalised == "desc"

    def _limit(self, limit: int | None) -> int:
        """Clamp the page size, defaulting to the configured page limit."""
        if limit is None:
            return self._settings.experiment_page_limit
        if limit < 1 or limit > self._settings.max_experiment_page_limit:
            raise ConfigurationError(
                f"limit must be between 1 and "
                f"{self._settings.max_experiment_page_limit}, got {limit}.",
                details={
                    "limit": limit,
                    "maximum": self._settings.max_experiment_page_limit,
                },
            )
        return limit
