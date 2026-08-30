"""Shared FastAPI dependencies.

Services are built per request from the active settings. Resolving settings
through a dependency keeps them overridable in tests without touching the
process environment.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from app.core.config import Settings, get_settings
from app.services.datasets import DatasetProfilingService
from app.services.experiments import ExperimentHistoryService, ExperimentRunner
from app.services.knowledge import KnowledgeService
from llm.config import LLMConfig, config_from_env as llm_config_from_env
from llm.providers import LLMProvider, build_llm_provider
from llm.service import RAGAnswerService
from ml.experiments import LocalExperimentStore
from ml.experiments.store import ExperimentStore
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
