"""Shared fixtures for the agent suite.

Everything here is offline and deterministic. The planner is scripted, the
services are doubles, and the one fixture that builds a real retrieval index
does it over this repository's own documentation into a temporary directory.
No test needs a credential, and none touches the network.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from agent.config import AgentConfig
from agent.orchestrator import AgentOrchestrator
from agent.planners.fake import FakePlanner
from agent.registry import ToolRegistry
from agent.tests.factories import (
    FakeExecutor,
    FakeProfiler,
    FakeRetrieval,
    FakeStore,
    documentation_results,
    experiment_payload,
)
from agent.tools import build_default_registry
from agent.tools.artifacts import ExperimentArtifactCache
from agent.tools.datasets import InMemoryDatasetSource

#: The model identifiers the fake registry offers. Deliberately a subset of
#: the real registry's, so a test can assert that a plausible-but-unoffered
#: name is refused.
AVAILABLE_MODELS: tuple[str, ...] = (
    "logistic_regression",
    "random_forest_classifier",
    "linear_regression",
)
AVAILABLE_METRICS: tuple[str, ...] = ("f1", "accuracy", "roc_auc", "rmse")
SOURCE_TYPES: tuple[str, ...] = ("project_documentation", "experiment")


@pytest.fixture
def config() -> AgentConfig:
    """Default budgets."""
    return AgentConfig()


@pytest.fixture
def dataset_source() -> InMemoryDatasetSource:
    """One named dataset, standing in for data the application already has."""
    return InMemoryDatasetSource({"sales": object()})


@pytest.fixture
def retrieval() -> FakeRetrieval:
    """A retrieval service returning two real-looking documentation passages."""
    return FakeRetrieval(documentation_results())


@pytest.fixture
def executor() -> FakeExecutor:
    """An experiment runner returning one fixed stored record."""
    return FakeExecutor()


@pytest.fixture
def store() -> FakeStore:
    """An experiment store holding the record the fake runner produces."""
    payload = experiment_payload()
    return FakeStore({payload["experiment_id"]: payload})


@pytest.fixture
def artifacts() -> ExperimentArtifactCache:
    """An empty in-memory cache of fitted models."""
    return ExperimentArtifactCache()


@pytest.fixture
def registry(
    dataset_source: InMemoryDatasetSource,
    retrieval: FakeRetrieval,
    executor: FakeExecutor,
    store: FakeStore,
    artifacts: ExperimentArtifactCache,
) -> ToolRegistry:
    """The full four-tool registry, wired to doubles."""
    return build_default_registry(
        source=dataset_source,
        profiler=FakeProfiler(),
        executor=executor,
        retrieval=retrieval,
        lookup=store,
        artifacts=artifacts,
        available_models=AVAILABLE_MODELS,
        available_metrics=AVAILABLE_METRICS,
        source_types=SOURCE_TYPES,
    )


@pytest.fixture
def build_agent(registry: ToolRegistry, artifacts: ExperimentArtifactCache):
    """Build an orchestrator around a scripted planner.

    Returns a factory so a test can script the exact plan it is about, which
    is the whole reason the fake planner exists.
    """

    def factory(
        steps: Sequence[Any] = (),
        *,
        answer: str | Sequence[str] | None = None,
        error: Exception | None = None,
        answer_error: Exception | None = None,
        config: AgentConfig | None = None,
        tools: ToolRegistry | None = None,
    ) -> tuple[AgentOrchestrator, FakePlanner]:
        """Return an orchestrator and the planner driving it."""
        planner = FakePlanner(
            steps, answer=answer, error=error, answer_error=answer_error
        )
        orchestrator = AgentOrchestrator(
            planner,
            tools if tools is not None else registry,
            config=config or AgentConfig(),
            artifacts=artifacts,
        )
        return orchestrator, planner

    return factory
