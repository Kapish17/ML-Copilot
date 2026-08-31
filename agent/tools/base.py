"""The tool contract, and the protocols tools reach the rest of the system by.

A tool is the *only* way this agent can affect anything. There is no escape
hatch beside it: no generic ``execute``, no code evaluation, no shell, no
filesystem handle, no HTTP client. Whatever the system can be asked to do is
the union of the registered tools' declared capabilities, and that union is
readable in one place — :func:`agent.tools.build_default_registry`.

Every tool is four things:

``name``      a stable identifier the planner may ask for
``description``  what it is for, in the words a planner will read
``schema``    the declared arguments, which are validated before ``run``
``run``       the work, expressed entirely in terms of existing services

The last point is the one to hold on to. A tool orchestrates; it does not
compute. Profiling belongs to the dataset service, training and selection to
the experiment runner, ranking to the retrieval service, SHAP to the
explainability layer. A tool that started computing would be a second
implementation of something already tested, and the two would drift.

**Why protocols rather than imports.** The services live in ``app.services``,
``rag`` and ``ml``. If this package imported them, the agent would depend on
the web layer and — transitively — on pandas, scikit-learn and SHAP, and a
later HTTP endpoint would close a loop between the backend and the agent. So
the collaborators are declared here structurally: a tool asks for something
with the right method, and the caller supplies the real service. The wiring
happens where both are already known; the agent stays a layer that knows how
to orchestrate without knowing what it is orchestrating with.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from agent.schemas import ArgumentSchema

# ---------------------------------------------------------------------------
# The tool contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolResult:
    """What a tool produced, before it becomes an observation.

    ``available`` is the honest middle ground between success and failure: the
    tool ran correctly and is reporting that the thing asked for does not
    exist. An explanation for a model that was never persisted is the case
    this exists for. It must never be dressed up as either a result or a
    crash.
    """

    output: dict[str, Any]
    available: bool = True
    #: Why the result is unavailable. A stable code, e.g.
    #: ``fitted_model_not_persisted``.
    reason: str | None = None
    #: Citation identifiers this result contributes to the answer's evidence.
    citations: tuple[str, ...] = ()

    @classmethod
    def unavailable(cls, reason: str, **output: Any) -> ToolResult:
        """Build a structured "cannot do this, and here is why" result."""
        payload = dict(output)
        payload.setdefault("status", "unavailable")
        payload.setdefault("reason", reason)
        return cls(output=payload, available=False, reason=reason)


@runtime_checkable
class Tool(Protocol):
    """Something the agent may be asked to run."""

    @property
    def name(self) -> str:
        """Stable identifier, e.g. ``"dataset_profile"``."""
        ...  # pragma: no cover - protocol

    @property
    def description(self) -> str:
        """What the tool does, written for the planner to read."""
        ...  # pragma: no cover - protocol

    @property
    def schema(self) -> ArgumentSchema:
        """The declared arguments. Validated before :meth:`run` is called."""
        ...  # pragma: no cover - protocol

    def run(self, arguments: Mapping[str, Any]) -> ToolResult:
        """Do the work.

        Args:
            arguments: Already validated against :attr:`schema`. A tool never
                receives an argument the schema did not approve, so it may
                read its fields without re-checking their types.

        Returns:
            ToolResult: JSON-safe output, or a structured unavailable result.

        Raises:
            Exception: Anything a tool raises is caught by the orchestrator,
                logged with its cause and turned into a failed observation
                carrying an authored message. A tool is not required to catch
                everything itself, but it must not put a path, a credential or
                a vendor message into an exception it raises deliberately.
        """
        ...  # pragma: no cover - protocol


class BaseTool:
    """Convenience base giving a tool its name, description and schema."""

    #: Subclasses set these three.
    tool_name: str = ""
    tool_description: str = ""

    def __init__(self) -> None:
        """Refuse a subclass that forgot to name or describe itself."""
        if not self.tool_name or not self.tool_description:
            raise ValueError(
                f"{type(self).__name__} must define tool_name and tool_description."
            )

    @property
    def name(self) -> str:
        """Stable identifier the planner may ask for."""
        return self.tool_name

    @property
    def description(self) -> str:
        """What the tool does, as the planner will read it."""
        return self.tool_description

    @property
    def schema(self) -> ArgumentSchema:
        """The declared arguments. Subclasses override."""
        return ArgumentSchema()

    def definition(self) -> dict[str, Any]:
        """Render the tool as the planner will be shown it."""
        return {
            "name": self.name,
            "description": self.description,
            "arguments": self.schema.as_dict()["fields"],
        }

    def run(self, arguments: Mapping[str, Any]) -> ToolResult:  # pragma: no cover
        """Do the work. Subclasses override."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# What the tools need from the rest of the system
# ---------------------------------------------------------------------------


@runtime_checkable
class DatasetSource(Protocol):
    """Where a named dataset comes from.

    The indirection is the point. A tool never receives a path, a URL or file
    content from the planner — it receives a *name*, which this resolves to a
    dataset the application already has. A planner that asks for
    ``"../../etc/passwd"`` or ``"C:\\keys.txt"`` is asking for a name that is
    not registered, and gets the same "unknown dataset" answer it would get
    for any typo. There is no path handling to get wrong because no path ever
    reaches the tool.
    """

    def names(self) -> Sequence[str]:
        """Every dataset name currently available."""
        ...  # pragma: no cover - protocol

    def get(self, name: str) -> Any:
        """Return the standardised in-memory dataset registered under a name.

        Raises:
            KeyError: If the name is not registered.
        """
        ...  # pragma: no cover - protocol


@runtime_checkable
class ProfilingService(Protocol):
    """Structural view of the existing dataset profiling service."""

    def profile_frame(
        self, frame: Any, *, filename: str = ..., target_column: str | None = ...
    ) -> Any:
        """Profile a standardised dataset."""
        ...  # pragma: no cover - protocol


@runtime_checkable
class ExperimentExecutor(Protocol):
    """Structural view of the existing experiment runner.

    One method, matching :meth:`app.services.experiments.runner.ExperimentRunner.run_frame`
    closely enough that the real runner satisfies it, and narrow enough that a
    test double is a few lines.
    """

    def run_frame(self, frame: Any, options: Any, **kwargs: Any) -> Any:
        """Run one experiment on a standardised dataset."""
        ...  # pragma: no cover - protocol


@runtime_checkable
class RetrievalLike(Protocol):
    """Structural view of the existing retrieval service.

    Search only. There is deliberately no ``index``, ``delete`` or ``save`` on
    this protocol: the agent has no way to name an operation that would modify
    the knowledge base, because no such operation is declared anywhere it can
    reach.
    """

    def search(self, question: str, **kwargs: Any) -> Any:
        """Find the passages that bear on a question."""
        ...  # pragma: no cover - protocol


@runtime_checkable
class ExperimentLookup(Protocol):
    """Structural view of the existing experiment store, reading only."""

    def exists(self, experiment_id: str) -> bool:
        """Whether a run is stored under this identifier."""
        ...  # pragma: no cover - protocol

    def get(self, experiment_id: str) -> Any:
        """Return one stored run."""
        ...  # pragma: no cover - protocol


__all__ = [
    "BaseTool",
    "DatasetSource",
    "ExperimentExecutor",
    "ExperimentLookup",
    "ProfilingService",
    "RetrievalLike",
    "Tool",
    "ToolResult",
]
