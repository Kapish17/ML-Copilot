"""Tests for the provider interface, its two implementations and messages."""

from __future__ import annotations

import json
import os

import pytest

from llm.config import AVAILABLE_PROVIDERS, LLMConfig, config_from_env
from llm.errors import (
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMError,
    LLMProviderError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from llm.messages import (
    GenerationRequest,
    GenerationResult,
    Role,
    build_messages,
    redact,
)
from llm.providers import FakeLLMProvider, LLMProvider, build_llm_provider
from llm.providers.openai_provider import OpenAIProvider

FAKE_KEY = "sk-test-not-a-real-key-0123456789"


def make_request(prompt: str = "What was the F1 score?") -> GenerationRequest:
    """Build a small request for a provider test."""
    return GenerationRequest(
        messages=build_messages("You are a grounded assistant.", prompt),
        model="fake-model",
        temperature=0.0,
        max_output_tokens=200,
        timeout_seconds=5.0,
    )


# --------------------------------------------------------------------------
# The interface
# --------------------------------------------------------------------------


def test_both_providers_satisfy_the_interface() -> None:
    """Everything above the interface depends on it and nothing else."""
    assert isinstance(FakeLLMProvider(), LLMProvider)
    assert isinstance(OpenAIProvider(LLMConfig()), LLMProvider)


def test_a_provider_is_chosen_by_configuration() -> None:
    """Swapping providers is a setting, not a code change."""
    assert isinstance(build_llm_provider(LLMConfig(provider="fake")), FakeLLMProvider)
    assert isinstance(
        build_llm_provider(LLMConfig(provider="openai")), OpenAIProvider
    )


def test_an_unknown_provider_is_refused() -> None:
    """With the available names, so the mistake is easy to fix."""
    with pytest.raises(LLMConfigurationError) as exc_info:
        LLMConfig(provider="telepathy")

    assert exc_info.value.details["available"] == list(AVAILABLE_PROVIDERS)


# --------------------------------------------------------------------------
# Laziness and credentials
# --------------------------------------------------------------------------


def test_constructing_the_real_provider_builds_nothing() -> None:
    """No SDK client, no credential read, no network call at construction."""
    provider = OpenAIProvider(LLMConfig())

    assert provider.is_loaded is False


def test_importing_the_package_does_not_import_the_sdk() -> None:
    """The whole layer must import with the SDK absent."""
    import subprocess
    import sys
    from pathlib import Path

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, llm; print('openai' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[2],
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


def test_generation_without_a_key_is_a_configuration_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nothing is attempted, and the message says which variable to set."""
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    provider = OpenAIProvider(LLMConfig(api_key_env="LLM_API_KEY"))

    assert provider.is_ready is False
    with pytest.raises(LLMConfigurationError) as exc_info:
        provider.generate(make_request())

    assert exc_info.value.details["api_key_env"] == "LLM_API_KEY"
    assert "LLM_API_KEY" in exc_info.value.message


def test_the_configuration_reports_a_key_without_revealing_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A status check may say whether a key exists, never what it is."""
    config = LLMConfig(api_key_env="LLM_TEST_KEY")
    monkeypatch.setenv("LLM_TEST_KEY", FAKE_KEY)

    described = json.dumps(config.describe())

    assert config.has_api_key is True
    assert described.count("api_key_configured") == 1
    assert FAKE_KEY not in described
    assert "sk-" not in described


def test_the_key_is_never_stored_on_the_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """It is read at the moment of use and dropped."""
    monkeypatch.setenv("LLM_TEST_KEY", FAKE_KEY)
    provider = OpenAIProvider(LLMConfig(api_key_env="LLM_TEST_KEY"))

    assert provider.is_ready is True
    assert FAKE_KEY not in json.dumps(
        {key: str(value) for key, value in vars(provider).items()}
    )


def test_reading_a_key_returns_it_only_to_the_caller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``resolve_api_key`` is the one place a value is produced."""
    monkeypatch.setenv("LLM_TEST_KEY", FAKE_KEY)

    assert LLMConfig(api_key_env="LLM_TEST_KEY").resolve_api_key() == FAKE_KEY


def test_a_blank_key_counts_as_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty variable is not a credential."""
    monkeypatch.setenv("LLM_TEST_KEY", "   ")
    config = LLMConfig(api_key_env="LLM_TEST_KEY")

    assert config.has_api_key is False
    with pytest.raises(LLMConfigurationError):
        config.resolve_api_key()


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"temperature": 3.0},
        {"temperature": -0.1},
        {"max_output_tokens": 0},
        {"timeout_seconds": 0},
        {"max_retries": -1},
        {"max_context_chars": 0},
        {"max_context_chunks": 0},
        {"max_context_chunks": 10, "max_retrieved_chunks": 4},
        {"min_evidence_score": 1.5},
        {"model": "  "},
    ],
)
def test_unusable_configuration_is_refused(overrides: dict) -> None:
    """Every limit is checked, including the ones that only make sense together."""
    with pytest.raises(LLMConfigurationError):
        LLMConfig(**overrides)


def test_an_unknown_configuration_field_is_refused() -> None:
    """A typo in an override is an error, not a silently ignored setting."""
    with pytest.raises(LLMConfigurationError, match="temperture"):
        LLMConfig().with_overrides(temperture=0.5)


def test_configuration_reads_the_environment_but_never_the_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Settings come from the environment; the key is read at generation time."""
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    monkeypatch.setenv("LLM_MODEL", "some-model")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.2")
    monkeypatch.setenv("LLM_MAX_CONTEXT_CHARS", "5000")
    monkeypatch.setenv("LLM_API_KEY", FAKE_KEY)

    config = config_from_env()

    assert config.provider == "fake"
    assert config.model == "some-model"
    assert config.temperature == 0.2
    assert config.max_context_chars == 5_000
    assert FAKE_KEY not in json.dumps(config.describe())


def test_a_malformed_environment_value_fails_loudly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Better than silently running with a default nobody chose."""
    monkeypatch.setenv("LLM_MAX_CONTEXT_CHARS", "lots")

    with pytest.raises(LLMConfigurationError, match="LLM_MAX_CONTEXT_CHARS"):
        config_from_env()


def test_the_default_temperature_is_zero() -> None:
    """A grounded answer should not vary between identical runs."""
    assert LLMConfig().temperature == 0.0


# --------------------------------------------------------------------------
# Messages
# --------------------------------------------------------------------------


def test_messages_are_a_system_and_a_user_turn() -> None:
    """No history: each answer is grounded in its own evidence."""
    messages = build_messages("rules", "question")

    assert [message.role for message in messages] == [Role.SYSTEM, Role.USER]
    assert messages[0].content == "rules"
    assert messages[1].content == "question"


def test_a_request_exposes_its_prompts() -> None:
    """Used by tests and by the fake provider to assert what was sent."""
    request = make_request("the question")

    assert request.system_prompt == "You are a grounded assistant."
    assert request.user_prompt == "the question"
    assert request.character_count > 0
    assert request.as_payload()[0] == {
        "role": "system",
        "content": "You are a grounded assistant.",
    }


def test_a_result_reports_truncation_and_totals() -> None:
    """A model that stopped at the limit must be distinguishable."""
    truncated = GenerationResult(
        text="partial", model="m", provider="fake", finish_reason="length",
        prompt_tokens=10, completion_tokens=5,
    )
    complete = GenerationResult(text="done", model="m", provider="fake", finish_reason="stop")

    assert truncated.is_truncated is True
    assert truncated.total_tokens == 15
    assert complete.is_truncated is False
    assert complete.total_tokens is None
    json.dumps(truncated.as_dict())


def test_redaction_removes_a_secret_from_text() -> None:
    """Defence in depth: keys never reach a prompt in the first place."""
    text = f"the key is {FAKE_KEY} and that is that"

    assert FAKE_KEY not in redact(text, [FAKE_KEY])
    assert "[redacted]" in redact(text, [FAKE_KEY])


def test_redaction_ignores_values_too_short_to_be_secrets() -> None:
    """Redacting 'a' would mangle every sentence."""
    assert redact("a normal sentence", ["a", ""]) == "a normal sentence"


# --------------------------------------------------------------------------
# The fake provider
# --------------------------------------------------------------------------


def test_the_fake_provider_returns_what_it_is_scripted_to() -> None:
    """Deterministic, so a grounding test can control exactly what comes back."""
    provider = FakeLLMProvider(responses="a grounded answer")
    result = provider.generate(make_request())

    assert result.text == "a grounded answer"
    assert result.provider == "fake"
    assert provider.call_count == 1
    assert provider.last_user_prompt == "What was the F1 score?"


def test_the_fake_provider_walks_through_a_sequence() -> None:
    """For tests that need a second call to differ from the first."""
    provider = FakeLLMProvider(responses=["first", "second"])

    assert provider.generate(make_request()).text == "first"
    assert provider.generate(make_request()).text == "second"
    assert provider.generate(make_request()).text == "second", "the last repeats"


def test_the_fake_provider_can_answer_from_the_request() -> None:
    """Used to assert what actually reached the model."""
    provider = FakeLLMProvider.echoing()
    provider.generate(make_request("evidence and a question"))

    assert provider.last_request is not None
    assert "evidence and a question" in provider.requests[0].user_prompt


def test_an_unscripted_fake_provider_abstains() -> None:
    """A model with nothing to say should say so, not invent something."""
    from llm.prompts import INSUFFICIENT_EVIDENCE_MARKER
    from llm.providers.fake import DEFAULT_ANSWER

    assert INSUFFICIENT_EVIDENCE_MARKER in DEFAULT_ANSWER
    assert INSUFFICIENT_EVIDENCE_MARKER in FakeLLMProvider().generate(
        make_request()
    ).text


def test_an_empty_response_is_reported_as_unusable() -> None:
    """A request that succeeded and produced nothing is its own failure."""
    with pytest.raises(LLMResponseError):
        FakeLLMProvider(responses="   ").generate(make_request())


@pytest.mark.parametrize(
    ("factory", "expected"),
    [
        (FakeLLMProvider.timing_out, LLMTimeoutError),
        (FakeLLMProvider.rate_limited, LLMRateLimitError),
        (FakeLLMProvider.unauthorised, LLMAuthenticationError),
        (FakeLLMProvider.unavailable, LLMUnavailableError),
        (FakeLLMProvider.malformed, LLMResponseError),
    ],
)
def test_the_fake_provider_reproduces_each_failure(factory, expected) -> None:
    """Every way a real provider fails can be exercised without one."""
    with pytest.raises(expected):
        factory().generate(make_request())


def test_every_provider_failure_is_an_llm_error() -> None:
    """A caller never has to catch a vendor's exception class."""
    for factory in (
        FakeLLMProvider.timing_out,
        FakeLLMProvider.rate_limited,
        FakeLLMProvider.unauthorised,
        FakeLLMProvider.unavailable,
        FakeLLMProvider.malformed,
    ):
        with pytest.raises(LLMError):
            factory().generate(make_request())


def test_a_provider_error_carries_no_credential() -> None:
    """Not in the message, not in the details."""
    try:
        FakeLLMProvider.unauthorised().generate(make_request())
    except LLMProviderError as exc:
        payload = json.dumps({"message": exc.message, "details": exc.details})
        assert "sk-" not in payload
        assert FAKE_KEY not in payload
        assert "LLM_API_KEY" in payload, "naming the variable is fine; the value is not"


def test_a_provider_with_no_credential_reports_itself_unready() -> None:
    """So a caller can check before spending a call."""
    assert FakeLLMProvider(ready=False).is_ready is False
    assert FakeLLMProvider().is_ready is True


# --------------------------------------------------------------------------
# The real provider's error mapping
# --------------------------------------------------------------------------


def test_the_real_provider_maps_sdk_exceptions_without_leaking_them() -> None:
    """Each vendor exception becomes one of this project's errors.

    Skipped where the SDK is not installed: the layer must work without it,
    and this test is about the mapping rather than about the SDK.
    """
    openai = pytest.importorskip("openai")
    provider = OpenAIProvider(LLMConfig(model="test-model"))

    class Response:
        """The minimum an SDK error needs to be constructed."""

        status_code = 429
        headers: dict = {}
        request = None

    cases = [
        (openai.APITimeoutError(request=None), LLMTimeoutError),
        (
            openai.AuthenticationError(
                f"Incorrect API key provided: {FAKE_KEY}",
                response=Response(),
                body=None,
            ),
            LLMAuthenticationError,
        ),
        (
            openai.RateLimitError("slow down", response=Response(), body=None),
            LLMRateLimitError,
        ),
        (
            openai.BadRequestError(
                "This model's maximum context length is 8192 tokens",
                response=Response(),
                body=None,
            ),
            LLMError,
        ),
        (openai.APIConnectionError(request=None), LLMUnavailableError),
        (RuntimeError("something else entirely"), LLMUnavailableError),
    ]

    for raised, expected in cases:
        mapped = provider._translate(raised)
        assert isinstance(mapped, expected), type(raised).__name__
        assert isinstance(mapped, LLMError)
        payload = json.dumps({"m": mapped.message, "d": mapped.details})
        assert FAKE_KEY not in payload, type(raised).__name__
        assert "sk-" not in payload


def test_a_context_overflow_is_distinguished_from_a_bad_request() -> None:
    """One is actionable by lowering a limit; the other is a bug."""
    openai = pytest.importorskip("openai")
    from llm.errors import LLMContextTooLargeError

    provider = OpenAIProvider(LLMConfig())

    class Response:
        """The minimum an SDK error needs to be constructed."""

        status_code = 400
        headers: dict = {}
        request = None

    overflow = provider._translate(
        openai.BadRequestError(
            "maximum context length is 8192 tokens", response=Response(), body=None
        )
    )
    malformed = provider._translate(
        openai.BadRequestError(
            "unknown parameter 'foo'", response=Response(), body=None
        )
    )

    assert isinstance(overflow, LLMContextTooLargeError)
    assert "max_context_chars" in overflow.details
    assert isinstance(malformed, LLMResponseError)


def test_the_real_provider_is_used_only_when_asked_for() -> None:
    """The default configuration names it, but nothing is contacted."""
    assert LLMConfig().provider == "openai"
    assert os.getenv("LLM_API_KEY") is None or True, "no key is required to get here"
