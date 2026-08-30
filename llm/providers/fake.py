"""A deterministic provider for tests.

The grounding rules are the most important thing in this layer and the hardest
to test against a real model: you cannot reliably make a hosted model fabricate
a citation on demand, and you should not need a network, a credential or a
budget to check that fabrications are rejected.

So this provider is scriptable. It returns exactly what a test tells it to,
records exactly what it was asked, and can be made to fail in each of the ways
a real provider fails:

- a valid grounded answer citing real sources
- an answer citing a source that was never retrieved
- an answer with no citations at all
- an answer that follows an instruction hidden in the evidence
- a timeout, a rate limit, an authentication failure, an outage
- an empty or unreadable response

It is a real implementation of :class:`~llm.providers.base.LLMProvider`, not a
mock: it satisfies the same protocol, raises the same error types, and returns
the same result object. A test written against it is testing the contract every
provider must meet.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from llm.config import PROVIDER_FAKE
from llm.errors import (
    LLMAuthenticationError,
    LLMError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from llm.messages import GenerationRequest, GenerationResult

#: What an unscripted provider returns: a compliant model's abstention. The
#: marker is the protocol declared in :mod:`llm.prompts`; it is spelled out
#: here rather than imported so this module stays free of prompt details, and
#: a test asserts the two agree.
DEFAULT_ANSWER = (
    "INSUFFICIENT_EVIDENCE\nThe retrieved evidence does not cover this question."
)


class FakeLLMProvider:
    """A provider that returns whatever a test scripts it to return."""

    def __init__(
        self,
        *,
        responses: Sequence[str] | str | None = None,
        error: LLMError | None = None,
        finish_reason: str | None = "stop",
        prompt_tokens: int | None = 120,
        completion_tokens: int | None = 40,
        latency_seconds: float | None = 0.01,
        model: str = "fake-model",
        ready: bool = True,
        responder: Callable[[GenerationRequest], str] | None = None,
    ) -> None:
        """Script the provider's behaviour.

        Args:
            responses: Text to return, one per call. A single string is
                returned for every call; a sequence is consumed in order and
                the last entry repeats once exhausted.
            error: Raised instead of answering, on every call. Use the
                classmethods below for the common failures.
            finish_reason: What to report as the stop reason. ``"length"``
                simulates hitting the output limit.
            prompt_tokens: Reported prompt-token usage, or ``None``.
            completion_tokens: Reported completion-token usage, or ``None``.
            latency_seconds: Reported latency.
            model: Model identifier to report.
            ready: What :attr:`is_ready` answers — ``False`` simulates a
                provider with no credential configured.
            responder: A function of the request, for tests that need the
                answer to depend on what was asked. Takes precedence over
                ``responses``.
        """
        if isinstance(responses, str):
            self._responses: list[str] = [responses]
        elif responses is None:
            self._responses = [DEFAULT_ANSWER]
        else:
            self._responses = list(responses) or [DEFAULT_ANSWER]

        self._error = error
        self._finish_reason = finish_reason
        self._prompt_tokens = prompt_tokens
        self._completion_tokens = completion_tokens
        self._latency = latency_seconds
        self._model = model
        self._ready = ready
        self._responder = responder

        #: Every request this provider has received, for inspection.
        self.requests: list[GenerationRequest] = []
        self.call_count = 0

    # -- Protocol ----------------------------------------------------------

    @property
    def name(self) -> str:
        """Stable identifier of this provider."""
        return PROVIDER_FAKE

    @property
    def is_ready(self) -> bool:
        """Whether a generation call could be attempted."""
        return self._ready

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Return the scripted response, or raise the scripted error."""
        self.requests.append(request)
        self.call_count += 1

        if self._error is not None:
            raise self._error

        if self._responder is not None:
            text = self._responder(request)
        else:
            index = min(self.call_count - 1, len(self._responses) - 1)
            text = self._responses[index]

        if not text.strip():
            raise LLMResponseError(
                "The provider returned an empty completion.",
                details={"provider": self.name, "model": self._model},
            )

        return GenerationResult(
            text=text,
            model=self._model,
            provider=self.name,
            finish_reason=self._finish_reason,
            prompt_tokens=self._prompt_tokens,
            completion_tokens=self._completion_tokens,
            latency_seconds=self._latency,
        )

    # -- Inspection --------------------------------------------------------

    @property
    def last_request(self) -> GenerationRequest | None:
        """The most recent request, for asserting what the prompt contained."""
        return self.requests[-1] if self.requests else None

    @property
    def last_system_prompt(self) -> str:
        """The system prompt of the most recent request."""
        return self.last_request.system_prompt if self.last_request else ""

    @property
    def last_user_prompt(self) -> str:
        """The user prompt of the most recent request."""
        return self.last_request.user_prompt if self.last_request else ""

    # -- Failure modes -----------------------------------------------------

    @classmethod
    def timing_out(cls, **kwargs: Any) -> FakeLLMProvider:
        """A provider that never answers in time."""
        return cls(
            error=LLMTimeoutError(
                "The language-model provider did not respond within 30 seconds.",
                details={"provider": PROVIDER_FAKE, "timeout_seconds": 30.0},
            ),
            **kwargs,
        )

    @classmethod
    def rate_limited(cls, **kwargs: Any) -> FakeLLMProvider:
        """A provider that is throttling."""
        return cls(
            error=LLMRateLimitError(
                "The language-model provider is rate limiting requests.",
                details={"provider": PROVIDER_FAKE},
            ),
            **kwargs,
        )

    @classmethod
    def unauthorised(cls, **kwargs: Any) -> FakeLLMProvider:
        """A provider that rejects the credential."""
        return cls(
            error=LLMAuthenticationError(
                "The language-model provider rejected the configured credential.",
                details={"provider": PROVIDER_FAKE, "api_key_env": "LLM_API_KEY"},
            ),
            **kwargs,
        )

    @classmethod
    def unavailable(cls, **kwargs: Any) -> FakeLLMProvider:
        """A provider that cannot be reached."""
        return cls(
            error=LLMUnavailableError(
                "The language-model provider could not be reached.",
                details={"provider": PROVIDER_FAKE},
            ),
            **kwargs,
        )

    @classmethod
    def malformed(cls, **kwargs: Any) -> FakeLLMProvider:
        """A provider that answers with nothing usable."""
        return cls(
            error=LLMResponseError(
                "The provider returned a response this client could not read.",
                details={"provider": PROVIDER_FAKE},
            ),
            **kwargs,
        )

    @classmethod
    def echoing(cls, **kwargs: Any) -> FakeLLMProvider:
        """A provider that returns the user prompt it was given.

        Used to assert what actually reached the model — that the evidence was
        included, that it was delimited, and that a credential was not.
        """
        return cls(responder=lambda request: request.user_prompt or "(empty)", **kwargs)


__all__ = ["DEFAULT_ANSWER", "FakeLLMProvider"]
