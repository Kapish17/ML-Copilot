"""An optional smoke test against a real provider.

**This is not part of the normal test suite.** It is skipped unless a
credential is already configured *and* the test is explicitly opted into, so
the ordinary run stays offline, free and deterministic.

To run it::

    export LLM_API_KEY=...          # already required for the provider
    export RUN_LLM_INTEGRATION=1    # explicit opt-in
    pytest llm/tests/test_real_provider.py -m external

It exists to catch the things a fake cannot: that the SDK call signature is
right, that the model returns something, and — the part worth having — that a
real model given real evidence produces citations the validator accepts.

The key is never printed, asserted on, or included in any output. The
assertions below deliberately check the *shape* of what came back rather than
its content, because a real model's wording is not deterministic and a test
that pinned it would fail for the wrong reasons.
"""

from __future__ import annotations

import os

import pytest

from llm.answers import AnswerStatus
from llm.config import LLMConfig, config_from_env
from llm.messages import GenerationRequest, build_messages
from llm.providers.openai_provider import OpenAIProvider
from llm.service import RAGAnswerService
from llm.tests.factories import FakeRetriever, documentation_results

#: Set to "1" to opt in. A configured key alone is not enough — a developer
#: with a key in their environment should not start spending it by running the
#: suite.
OPT_IN_VARIABLE = "RUN_LLM_INTEGRATION"

pytestmark = pytest.mark.external


def _enabled() -> bool:
    """Whether the opt-in and a credential are both present."""
    if os.getenv(OPT_IN_VARIABLE, "").strip() != "1":
        return False
    return LLMConfig().has_api_key


requires_credentials = pytest.mark.skipif(
    not _enabled(),
    reason=(
        f"Set {OPT_IN_VARIABLE}=1 and configure an API key to run the real "
        "provider smoke test. The rest of the suite runs offline."
    ),
)


@requires_credentials
def test_a_real_provider_answers() -> None:
    """The SDK call works and returns usable text."""
    config = config_from_env(provider="openai")
    provider = OpenAIProvider(config)

    result = provider.generate(
        GenerationRequest(
            messages=build_messages(
                "Answer in exactly one short sentence.",
                "What does the F1 score measure?",
            ),
            model=config.model,
            temperature=0.0,
            max_output_tokens=60,
            timeout_seconds=config.timeout_seconds,
        )
    )

    assert result.text.strip()
    assert result.provider == "openai"
    assert result.latency_seconds is not None
    assert "sk-" not in result.text


@requires_credentials
def test_a_real_model_produces_citations_the_validator_accepts() -> None:
    """The part a fake cannot check: that the prompt actually works.

    Asserts the shape of the outcome, not its wording. A real model's prose is
    not deterministic, and pinning it would make this fail for reasons that
    say nothing about the system.
    """
    config = config_from_env(provider="openai")
    service = RAGAnswerService(
        config,
        retriever=FakeRetriever(documentation_results()),
        provider=OpenAIProvider(config),
    )

    answer = service.answer("How does this project prevent data leakage?")

    assert answer.status in (
        AnswerStatus.GROUNDED,
        AnswerStatus.INSUFFICIENT_EVIDENCE,
    ), f"unexpected status: {answer.status.value}"
    assert answer.rejected_citations == (), "the model invented a citation"
    if answer.status is AnswerStatus.GROUNDED:
        assert answer.citations
        assert set(answer.citation_ids) <= set(answer.allowed_citations)


@requires_credentials
def test_a_real_model_declines_when_the_evidence_does_not_cover_the_question() -> None:
    """Given evidence about preprocessing, asked about something else."""
    config = config_from_env(provider="openai")
    service = RAGAnswerService(
        config,
        retriever=FakeRetriever(documentation_results()),
        provider=OpenAIProvider(config),
    )

    answer = service.answer(
        "What was the closing share price of Acme Corporation last Tuesday?"
    )

    assert answer.status is not AnswerStatus.GROUNDED
    assert answer.rejected_citations == ()


def test_the_optional_test_is_off_by_default() -> None:
    """A guard on the guard: the suite must not start spending money.

    Runs always, and asserts that the opt-in is required — so a change that
    accidentally enables these tests fails here rather than on someone's bill.
    """
    if os.getenv(OPT_IN_VARIABLE, "").strip() == "1":
        pytest.skip("integration tests are explicitly enabled in this environment")

    assert _enabled() is False
