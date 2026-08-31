"""Holding a just-trained model long enough to explain it.

Commit 7 made a deliberate choice: an experiment record contains the dataset
fingerprint, the configuration, the scores and the feature importances, and
**no fitted model**. Nothing in this commit changes that, and this module is
the part of the design that has to be read carefully so that it is not
mistaken for a change.

What this cache holds is a reference to objects that already exist in memory
because the experiment ran a moment ago in this process. It:

- is created per agent run and thrown away with it,
- is never serialised, never written to disk, never put in the execution
  state, and never rendered into an observation,
- holds a small, fixed number of entries and evicts the oldest,
- and is keyed by an experiment identifier the run itself produced.

The consequence is the honest one. An experiment this run performed can be
explained live, because its model is still in memory. An experiment from last
week cannot, because its model was never written down — and the explanation
tool says exactly that, with a reason code, rather than producing something
plausible.

The objects inside are opaque here: this module never imports pandas,
scikit-learn or SHAP, and never inspects what it is holding. It stores them
and hands them back.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

#: Experiments whose fitted objects one run may hold at once. A run cannot
#: make more than a handful of experiments within its tool budget, and the
#: cap means a long-lived orchestrator cannot accumulate fitted models.
DEFAULT_MAX_ENTRIES = 4


class ExperimentArtifactCache:
    """In-memory, run-scoped fitted objects, keyed by experiment id."""

    def __init__(self, *, max_entries: int = DEFAULT_MAX_ENTRIES) -> None:
        """Create an empty cache holding at most ``max_entries`` experiments."""
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1.")
        self._max_entries = max_entries
        self._entries: OrderedDict[str, Any] = OrderedDict()

    def __len__(self) -> int:
        """How many experiments are currently held."""
        return len(self._entries)

    def __contains__(self, experiment_id: object) -> bool:
        """Whether this run can still explain that experiment live."""
        return isinstance(experiment_id, str) and experiment_id in self._entries

    def put(self, experiment_id: str, artifacts: Any) -> None:
        """Hold one experiment's fitted objects, evicting the oldest if full."""
        if not isinstance(experiment_id, str) or not experiment_id:
            raise ValueError("An experiment id is required.")
        self._entries[experiment_id] = artifacts
        self._entries.move_to_end(experiment_id)
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)

    def get(self, experiment_id: str) -> Any | None:
        """Return the held objects, or ``None`` when there are none.

        ``None`` is the normal answer for any experiment this run did not
        perform, and the explanation tool treats it as such.
        """
        if not isinstance(experiment_id, str):
            return None
        return self._entries.get(experiment_id)

    def experiment_ids(self) -> tuple[str, ...]:
        """Which experiments can currently be explained live."""
        return tuple(self._entries)

    def clear(self) -> None:
        """Drop everything held. Called when a run finishes."""
        self._entries.clear()


__all__ = ["DEFAULT_MAX_ENTRIES", "ExperimentArtifactCache"]
