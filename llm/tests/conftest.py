"""Shared fixtures for the language-model test suite.

Every fixture is offline. No test in this suite reads a credential, builds an
SDK client or contacts anything; the provider is a deterministic fake and the
evidence is written by hand.
"""

from __future__ import annotations

import pytest

from llm.config import LLMConfig
from llm.context import build_context
from llm.providers.fake import FakeLLMProvider
from llm.service import RAGAnswerService
from llm.tests.factories import (
    FakeRetriever,
    documentation_results,
    experiment_results,
    mixed_results,
)


@pytest.fixture
def config() -> LLMConfig:
    """A configuration using the fake provider and temperature zero."""
    return LLMConfig(provider="fake", model="fake-model", temperature=0.0)


@pytest.fixture
def documentation_context(config: LLMConfig):
    """Evidence built from two documentation passages."""
    return build_context(documentation_results(), config)


@pytest.fixture
def experiment_context(config: LLMConfig):
    """Evidence built from two experiment passages."""
    return build_context(experiment_results(), config)


@pytest.fixture
def documentation_retriever() -> FakeRetriever:
    """A retriever returning documentation evidence."""
    return FakeRetriever(documentation_results())


@pytest.fixture
def experiment_retriever() -> FakeRetriever:
    """A retriever returning experiment evidence."""
    return FakeRetriever(experiment_results())


@pytest.fixture
def mixed_retriever() -> FakeRetriever:
    """A retriever returning both documentation and experiment evidence."""
    return FakeRetriever(mixed_results())


@pytest.fixture
def empty_retriever() -> FakeRetriever:
    """A retriever that finds nothing."""
    return FakeRetriever(())


def make_service(
    config: LLMConfig,
    retriever: FakeRetriever,
    provider: FakeLLMProvider,
) -> RAGAnswerService:
    """Assemble an answer service from a retriever and a provider."""
    return RAGAnswerService(config, retriever=retriever, provider=provider)


@pytest.fixture
def service_factory(config: LLMConfig):
    """Build an answer service from a retriever and a provider."""

    def build(retriever: FakeRetriever, provider: FakeLLMProvider) -> RAGAnswerService:
        """Return a service wired to the given collaborators."""
        return make_service(config, retriever, provider)

    return build
