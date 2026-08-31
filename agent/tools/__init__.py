"""The tools, and the one function that decides which of them exist.

``base``            the tool contract and the protocols tools reach services by
``artifacts``       in-memory fitted models, for explaining a run just made
``datasets``        naming a dataset, and profiling the one named
``experiments``     running an experiment through the existing runner
``knowledge``       searching documentation and experiment history
``explainability``  explaining a model, or saying why it cannot be

:func:`build_default_registry` is the complete answer to "what can this agent
do". Four entries, all optional depending on what the caller wired up. There
is no fifth registered elsewhere, no dynamic discovery and no way to add one
at runtime — a capability arrives by editing this function.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from agent.registry import ToolRegistry
from agent.tools.artifacts import ExperimentArtifactCache
from agent.tools.base import BaseTool, Tool, ToolResult
from agent.tools.datasets import DatasetProfileTool, InMemoryDatasetSource
from agent.tools.experiments import RunExperimentTool
from agent.tools.explainability import ExplainExperimentTool
from agent.tools.knowledge import SearchKnowledgeTool


def build_default_registry(
    *,
    source: Any = None,
    profiler: Any = None,
    executor: Callable[..., Any] | None = None,
    retrieval: Any = None,
    lookup: Any = None,
    explain_global: Callable[..., Any] | None = None,
    explain_prediction: Callable[..., Any] | None = None,
    artifacts: ExperimentArtifactCache | None = None,
    available_models: Callable[[], Sequence[str]] | Sequence[str] = (),
    available_metrics: Callable[[], Sequence[str]] | Sequence[str] = (),
    source_types: Callable[[], Sequence[str]] | Sequence[str] = (),
    max_top_k: int = 10,
    max_query_length: int = 2_000,
) -> ToolRegistry:
    """Build the registry from whichever services the caller supplied.

    A tool is registered only when the collaborators it needs are present, so
    an agent wired with retrieval but no experiment runner simply has fewer
    tools — and the planner is told about fewer tools, because the planner is
    shown this registry and nothing else. There is no tool that exists but is
    hidden, and none that is hidden but reachable.

    Returns:
        ToolRegistry: The complete allowlist for one agent.
    """
    registry = ToolRegistry()

    if source is not None and profiler is not None:
        registry.register(DatasetProfileTool(source, profiler))

    if source is not None and executor is not None:
        registry.register(
            RunExperimentTool(
                source,
                executor,
                available_models=available_models,
                available_metrics=available_metrics,
                artifacts=artifacts,
            )
        )

    if retrieval is not None:
        registry.register(
            SearchKnowledgeTool(
                retrieval,
                max_top_k=max_top_k,
                max_query_length=max_query_length,
                source_types=source_types,
            )
        )

    if lookup is not None or artifacts is not None:
        registry.register(
            ExplainExperimentTool(
                artifacts=artifacts,
                lookup=lookup,
                explain_global=explain_global,
                explain_prediction=explain_prediction,
            )
        )

    return registry


__all__ = [
    "BaseTool",
    "DatasetProfileTool",
    "ExperimentArtifactCache",
    "ExplainExperimentTool",
    "InMemoryDatasetSource",
    "RunExperimentTool",
    "SearchKnowledgeTool",
    "Tool",
    "ToolResult",
    "build_default_registry",
]
