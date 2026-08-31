"""Shared FastAPI dependencies.

Services are built per request from the active settings. Resolving settings
through a dependency keeps them overridable in tests without touching the
process environment.
"""

from __future__ import annotations

from functools import lru_cache, partial
from typing import Annotated

from fastapi import Depends

from agent.config import AgentConfig, config_from_env as agent_config_from_env
from agent.planner import LLMPlanner
from agent.registry import ToolRegistry
from agent.tools import build_default_registry
from agent.tools.artifacts import ExperimentArtifactCache
from agent.tools.datasets import InMemoryDatasetSource
from app.core.config import Settings, get_settings
from app.services.agent import AgentService
from app.services.datasets import DatasetProfilingService
from app.services.experiments import ExperimentHistoryService, ExperimentRunner
from app.services.experiments.runner import run_experiment
from app.services.knowledge import KnowledgeService
from app.services.knowledge.filters import KNOWN_SOURCE_TYPES
from llm.config import LLMConfig, config_from_env as llm_config_from_env
from llm.providers import LLMProvider, build_llm_provider
from llm.service import RAGAnswerService
from ml.evaluation.metrics import CLASSIFICATION_METRICS, REGRESSION_METRICS
from ml.experiments import LocalExperimentStore
from ml.experiments.store import ExperimentStore
from ml.explainability import explain_global, explain_prediction
from ml.models.registry import default_registry
from rag.config import RagConfig, config_from_env as rag_config_from_env
from rag.retrieval import RetrievalService
from rag.stores import LocalVectorStore

SettingsDep = Annotated[Settings, Depends(get_settings)]


def get_dataset_service(settings: SettingsDep) -> DatasetProfilingService:
    """Provide a dataset profiling service bound to the active settings."""
    return DatasetProfilingService(settings)


DatasetServiceDep = Annotated[DatasetProfilingService, Depends(get_dataset_service)]


def get_experiment_store(settings: SettingsDep) -> ExperimentStore:
    """Provide the experiment store the service layer should use.

    The return type is the storage *interface*, so a future MLflow or database
    backend is a change to this one function. **Only the local JSON store is
    implemented.** Constructing it has no side effect; the directory is created
    on the first write.
    """
    return LocalExperimentStore(settings.experiment_store_dir)


ExperimentStoreDep = Annotated[ExperimentStore, Depends(get_experiment_store)]


def get_experiment_runner(
    settings: SettingsDep,
    store: ExperimentStoreDep,
    datasets: DatasetServiceDep,
) -> ExperimentRunner:
    """Provide an experiment runner wired to its collaborators."""
    return ExperimentRunner(settings, store, datasets)


ExperimentRunnerDep = Annotated[ExperimentRunner, Depends(get_experiment_runner)]


def get_experiment_history(
    settings: SettingsDep, store: ExperimentStoreDep
) -> ExperimentHistoryService:
    """Provide the experiment history service for the active store."""
    return ExperimentHistoryService(settings, store)


ExperimentHistoryDep = Annotated[
    ExperimentHistoryService, Depends(get_experiment_history)
]


# ---------------------------------------------------------------------------
# Knowledge: retrieval and grounded answers
#
# The two configurations come from ``rag/`` and ``llm/`` rather than being
# copied into ``Settings``: they already read ``RAG_*`` and ``LLM_*`` from the
# environment, and duplicating them here would give the project two places to
# change one number. ``create_app`` can override either, which is how a test
# points the index at a temporary directory.
#
# Everything below is lazy. Building a retrieval service reads no index and
# building a provider builds no client, so resolving these dependencies costs
# nothing and the application starts with no credential and no index present.
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_rag_config() -> RagConfig:
    """Provide the retrieval configuration.

    Cached because it only reads the environment, and because the embedding
    provider built from it is worth reusing across requests rather than
    reconstructing per call.
    """
    return rag_config_from_env()


RagConfigDep = Annotated[RagConfig, Depends(get_rag_config)]


@lru_cache(maxsize=1)
def get_llm_config() -> LLMConfig:
    """Provide the language-model configuration.

    Holds the *name* of the variable carrying the API key, never the key —
    the provider reads the environment at the moment it authenticates.
    """
    return llm_config_from_env()


LLMConfigDep = Annotated[LLMConfig, Depends(get_llm_config)]


def get_retrieval_service(config: RagConfigDep) -> RetrievalService:
    """Provide a retrieval service over the configured index.

    Constructing it neither reads the index nor loads the embedding provider;
    both happen on first use.
    """
    return RetrievalService(config, store=LocalVectorStore(config.index_dir))


RetrievalServiceDep = Annotated[RetrievalService, Depends(get_retrieval_service)]


def get_llm_provider(config: LLMConfigDep) -> LLMProvider:
    """Provide the language-model provider.

    No SDK is imported and no credential is read here. A provider with no key
    reports itself unready, which is what lets the application serve
    everything except ``/ask`` without one.
    """
    return build_llm_provider(config)


LLMProviderDep = Annotated[LLMProvider, Depends(get_llm_provider)]


def get_answer_service(
    config: LLMConfigDep,
    retrieval: RetrievalServiceDep,
    provider: LLMProviderDep,
) -> RAGAnswerService:
    """Provide the grounded answer service.

    ``propagate_retrieval_errors`` is on: over HTTP, "there is no relevant
    evidence" and "the retrieval system is broken" must reach the client as
    different things — a 200 with a status, and a 503 someone has to act on.
    """
    return RAGAnswerService(
        config,
        retriever=retrieval,
        provider=provider,
        propagate_retrieval_errors=True,
    )


AnswerServiceDep = Annotated[RAGAnswerService, Depends(get_answer_service)]


def get_knowledge_service(
    rag_config: RagConfigDep,
    retrieval: RetrievalServiceDep,
    answering: AnswerServiceDep,
) -> KnowledgeService:
    """Provide the knowledge service the search and ask routes delegate to."""
    return KnowledgeService(rag_config, retrieval=retrieval, answering=answering)


KnowledgeServiceDep = Annotated[KnowledgeService, Depends(get_knowledge_service)]


# ---------------------------------------------------------------------------
# The agent
#
# This is where the four independent packages are wired together, and it is
# the only place that knows they exist at once. ``agent/`` reaches every
# service through a structural protocol rather than an import, which is what
# lets it stay free of FastAPI, pandas, scikit-learn, SHAP and any SDK — and
# what makes this function, rather than that package, responsible for saying
# which concrete service goes where.
#
# Everything stays lazy. A registry is a handful of small objects; no index is
# read, no SDK imported and no credential looked at until a request needs one,
# so the application starts with none of them present.
# ---------------------------------------------------------------------------


def get_dataset_source() -> InMemoryDatasetSource:
    """Provide the datasets one agent run may name.

    **Empty by default, deliberately.** Uploads are never kept (see
    ``backend/README.md``), and this endpoint takes a JSON body rather than a
    file, so nothing populates this in the default wiring — which means
    ``dataset_profile`` and ``run_experiment`` are simply not registered, and
    the planner is told about the two tools that are. A caller sees that in
    ``tools_available`` rather than having a tool fail on them.

    It is a dependency rather than a constant so a test — and a later commit
    that gives the endpoint a dataset to work on — can supply one without
    touching anything else.
    """
    return InMemoryDatasetSource()


DatasetSourceDep = Annotated[InMemoryDatasetSource, Depends(get_dataset_source)]


@lru_cache(maxsize=1)
def get_agent_config() -> AgentConfig:
    """Provide the agent's limits, from ``AGENT_*``.

    Cached because it only reads the environment. These are the server's
    ceiling: a request may lower any of them and raise none.
    """
    return agent_config_from_env()


AgentConfigDep = Annotated[AgentConfig, Depends(get_agent_config)]


def get_agent_artifacts() -> ExperimentArtifactCache:
    """Provide the in-memory cache of fitted models for one run.

    Per request, and cleared by the orchestrator when the run ends, so an
    experiment performed while answering a question can be explained in the
    same answer and nothing outlives the question that made it. **Nothing is
    written to disk**: Commit 7's decision not to persist fitted models
    stands.
    """
    return ExperimentArtifactCache()


AgentArtifactsDep = Annotated[ExperimentArtifactCache, Depends(get_agent_artifacts)]


def get_agent_registry(
    settings: SettingsDep,
    rag_config: RagConfigDep,
    source: DatasetSourceDep,
    datasets: DatasetServiceDep,
    store: ExperimentStoreDep,
    retrieval: RetrievalServiceDep,
    artifacts: AgentArtifactsDep,
) -> ToolRegistry:
    """Provide the complete allowlist of tools one agent run may use.

    Each tool is registered only when the service it wraps is available, so
    the set is honest about what this deployment can actually do. The model
    and metric names a request may ask for are read from the existing
    registries rather than listed again here, so "what the agent may train"
    and "what the system supports" cannot drift apart.

    A dataset source with nothing in it counts as absent. Registering
    ``dataset_profile`` and ``run_experiment`` against no datasets would
    advertise two tools whose every call fails with "no such dataset", which
    wastes a planner's turns and misleads a client reading
    ``tools_available``.
    """
    return build_default_registry(
        source=source if source.names() else None,
        profiler=datasets,
        executor=partial(
            run_experiment,
            settings=settings,
            store=store,
            dataset_service=datasets,
        ),
        retrieval=retrieval,
        lookup=store,
        artifacts=artifacts,
        explain_global=explain_global,
        explain_prediction=explain_prediction,
        available_models=lambda: list(default_registry().identifiers()),
        available_metrics=[
            definition.key
            for definition in CLASSIFICATION_METRICS + REGRESSION_METRICS
        ],
        source_types=tuple(KNOWN_SOURCE_TYPES),
        max_top_k=rag_config.max_top_k,
        max_query_length=rag_config.max_query_length,
    )


AgentRegistryDep = Annotated[ToolRegistry, Depends(get_agent_registry)]


def get_agent_planner(
    provider: LLMProviderDep, config: AgentConfigDep, llm_config: LLMConfigDep
) -> LLMPlanner:
    """Provide the planner, over the same provider abstraction ``/ask`` uses.

    No SDK is imported and no credential is read here. A planner whose
    provider has no key reports itself unready, which is what lets the
    application serve everything except this endpoint without one.
    """
    return LLMPlanner(provider, config=config, model=llm_config.model)


AgentPlannerDep = Annotated[LLMPlanner, Depends(get_agent_planner)]


def get_agent_service(
    planner: AgentPlannerDep,
    registry: AgentRegistryDep,
    config: AgentConfigDep,
    artifacts: AgentArtifactsDep,
) -> AgentService:
    """Provide the agent service the endpoint delegates to."""
    return AgentService(
        planner=planner, registry=registry, config=config, artifacts=artifacts
    )


AgentServiceDep = Annotated[AgentService, Depends(get_agent_service)]
