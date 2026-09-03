"""The OpenAI-compatible provider.

Uses the official ``openai`` SDK's chat-completions API. That API rather than
a vendor-specific one is the whole point: the same code, with ``base_url``
pointed elsewhere, talks to Azure OpenAI, vLLM, Ollama, LM Studio, OpenRouter,
Together and anything else that speaks the shape. One provider implementation
therefore covers hosted models and a model running on the developer's own
machine, which matters for a project whose other layers all work offline.

Everything is lazy. The module imports without the SDK installed; constructing
a provider builds no client, reads no credential and contacts nothing; the
first :meth:`OpenAIProvider.generate` call does all three. The client is then
kept and reused, because building one per request would pay for a TLS pool
setup on every question.

**Credentials.** The key is read from the environment at the moment a client
is built, handed to the SDK, and never stored on this object, put into a
message, echoed in an error or logged. Provider exceptions are caught and
replaced with this project's own errors, so an SDK exception carrying a
request URL or an authorisation header cannot escape.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from llm.config import LLMConfig, PROVIDER_OPENAI
from llm.errors import (
    LLMAuthenticationError,
    LLMContextTooLargeError,
    LLMDependencyError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from llm.messages import GenerationRequest, GenerationResult

logger = logging.getLogger(__name__)

INSTALL_HINT = (
    "The OpenAI-compatible provider needs the SDK: pip install openai. "
    "Retrieval works without it; only answer generation needs it."
)

#: Substrings that mark a bad request as a context-window problem rather than
#: a malformed one. Matched case-insensitively against the provider's message.
_CONTEXT_MARKERS = (
    "context length",
    "context_length",
    "maximum context",
    "too many tokens",
    "reduce the length",
    "string too long",
)


class OpenAIProvider:
    """Generates completions through an OpenAI-compatible chat API."""

    def __init__(self, config: LLMConfig) -> None:
        """Configure the provider without building anything.

        Args:
            config: Supplies the model, the endpoint, the timeout, the retry
                budget and the *name* of the environment variable holding the
                key. The key itself is not read here.
        """
        self._config = config
        self._client: Any | None = None

    @property
    def name(self) -> str:
        """Stable identifier of this provider."""
        return PROVIDER_OPENAI

    @property
    def model(self) -> str:
        """The model this provider will ask for."""
        return self._config.model

    @property
    def is_ready(self) -> bool:
        """Whether a credential is configured.

        Does not contact the provider and does not reveal the credential. A
        key that is present but invalid still reads as ready here; only a real
        call can tell the difference, and it raises
        :class:`~llm.errors.LLMAuthenticationError` when it does.
        """
        return self._config.has_api_key

    @property
    def is_loaded(self) -> bool:
        """Whether an SDK client has actually been built yet."""
        return self._client is not None

    # -- Client ------------------------------------------------------------

    def _build_client(self) -> Any:
        """Create the SDK client on first use.

        The credential is checked before the SDK is imported. A missing key is
        the far more common problem and the more actionable message, and
        checking it first costs nothing.

        Raises:
            LLMConfigurationError: If no API key is configured.
            LLMDependencyError: If the SDK is not installed.
        """
        # Raises LLMConfigurationError when unset. The value is passed
        # straight to the SDK and never bound to an attribute here.
        api_key = self._config.resolve_api_key()

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMDependencyError(
                f"The 'openai' package is not installed. {INSTALL_HINT}",
                details={"provider": self.name, "model": self._config.model},
            ) from exc

        options: dict[str, Any] = {
            "api_key": api_key,
            "timeout": self._config.timeout_seconds,
            # The SDK retries transient failures itself; letting it do so
            # keeps the backoff in one place and bounded by configuration.
            "max_retries": self._config.max_retries,
        }
        if self._config.base_url:
            options["base_url"] = self._config.base_url
        return OpenAI(**options)

    def _client_or_build(self) -> Any:
        """Return the cached client, building it the first time."""
        if self._client is None:
            self._client = self._build_client()
        return self._client

    # -- Generation --------------------------------------------------------

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate a completion.

        Args:
            request: The messages and settings for this call.

        Returns:
            GenerationResult: The generated text and its metadata.

        Raises:
            LLMError: Any failure, mapped to this project's error types. No
                SDK exception escapes.
        """
        client = self._client_or_build()
        started = time.perf_counter()

        try:
            response = client.chat.completions.create(
                model=request.model or self._config.model,
                messages=request.as_payload(),
                temperature=request.temperature,
                max_completion_tokens=request.max_output_tokens,
                timeout=request.timeout_seconds,
            )
        except Exception as exc:  # noqa: BLE001 - every SDK failure is mapped
            translated = self._translate(exc)
            # The outcome of a call that leaves this process is worth recording
            # either way. The exception *type* and this project's own error
            # code say what happened; the provider's message may quote the
            # request back and is left to the error envelope.
            logger.warning(
                "Generation failed after %.2fs: %s -> %s (provider=%s model=%s)",
                time.perf_counter() - started,
                type(exc).__name__,
                type(translated).__name__,
                self.name,
                request.model or self._config.model,
            )
            raise translated from None

        latency = round(time.perf_counter() - started, 3)
        result = self._read_response(response, request=request, latency=latency)
        # **No prompt and no completion.** Both are the sensitive halves of
        # this call — the prompt carries retrieved passages and the user's
        # question, the completion is the answer — so what is recorded is the
        # shape of the exchange, not its content.
        logger.info(
            "Generation succeeded in %.2fs: provider=%s model=%s finish=%s "
            "prompt_tokens=%s completion_tokens=%s",
            latency,
            result.provider,
            result.model,
            result.finish_reason,
            result.prompt_tokens,
            result.completion_tokens,
        )
        return result

    def _read_response(
        self, response: Any, *, request: GenerationRequest, latency: float
    ) -> GenerationResult:
        """Turn an SDK response into a result, or explain why it cannot.

        Raises:
            LLMResponseError: If the response has no usable completion. A
                request that succeeded at the transport level and produced
                nothing is a distinct failure from one that never connected,
                and the caller is told which.
        """
        try:
            choices = response.choices or []
            choice = choices[0]
            text = (choice.message.content or "").strip()
            finish_reason = getattr(choice, "finish_reason", None)
        except (AttributeError, IndexError, TypeError) as exc:
            raise LLMResponseError(
                "The provider returned a response this client could not read.",
                details={
                    "provider": self.name,
                    "model": request.model,
                    "reason": type(exc).__name__,
                },
            ) from None

        if not text:
            raise LLMResponseError(
                "The provider returned an empty completion.",
                details={
                    "provider": self.name,
                    "model": request.model,
                    "finish_reason": finish_reason,
                },
            )

        usage = getattr(response, "usage", None)
        return GenerationResult(
            text=text,
            model=getattr(response, "model", request.model) or request.model,
            provider=self.name,
            finish_reason=finish_reason,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            latency_seconds=latency,
        )

    # -- Error mapping -----------------------------------------------------

    def _translate(self, exc: Exception) -> Exception:
        """Map an SDK exception onto this project's error types.

        The SDK's own message is not passed through. It can carry a request
        URL, headers or an echoed payload, and nothing in it is worth the risk
        — so each case gets a message written here, and ``details`` carries
        only the provider, the model and, where the SDK offers one, a status
        code.
        """
        context = {"provider": self.name, "model": self._config.model}

        try:
            import openai
        except ImportError:  # pragma: no cover - only if the SDK vanished
            return LLMUnavailableError(
                "The language-model provider could not be reached.",
                details=context,
            )

        if isinstance(exc, openai.APITimeoutError):
            return LLMTimeoutError(
                "The language-model provider did not respond within "
                f"{self._config.timeout_seconds:g} seconds.",
                details={**context, "timeout_seconds": self._config.timeout_seconds},
            )
        if isinstance(exc, openai.AuthenticationError):
            return LLMAuthenticationError(
                "The language-model provider rejected the configured "
                "credential. Check the value of "
                f"{self._config.api_key_env}.",
                details={**context, "api_key_env": self._config.api_key_env},
            )
        if isinstance(exc, openai.PermissionDeniedError):
            return LLMAuthenticationError(
                "The configured credential is not permitted to use this model.",
                details=context,
            )
        if isinstance(exc, openai.RateLimitError):
            return LLMRateLimitError(
                "The language-model provider is rate limiting requests, or a "
                "quota is exhausted.",
                details=context,
            )
        if isinstance(exc, openai.BadRequestError):
            message = str(exc).lower()
            if any(marker in message for marker in _CONTEXT_MARKERS):
                return LLMContextTooLargeError(
                    "The request exceeded the model's context window. Lower "
                    "max_context_chars or max_context_chunks and try again.",
                    details={
                        **context,
                        "max_context_chars": self._config.max_context_chars,
                        "max_context_chunks": self._config.max_context_chunks,
                    },
                )
            return LLMResponseError(
                "The language-model provider rejected the request as malformed.",
                details=context,
            )
        if isinstance(exc, openai.NotFoundError):
            return LLMUnavailableError(
                f"The model '{self._config.model}' is not available to this "
                "credential or endpoint.",
                details=context,
            )
        if isinstance(exc, (openai.APIConnectionError, openai.InternalServerError)):
            return LLMUnavailableError(
                "The language-model provider could not be reached.",
                details=context,
            )
        if isinstance(exc, openai.APIStatusError):
            return LLMUnavailableError(
                "The language-model provider returned an unexpected status.",
                details={**context, "status_code": getattr(exc, "status_code", None)},
            )

        logger.warning(
            "Unmapped %s from the %s provider", type(exc).__name__, self.name
        )
        return LLMUnavailableError(
            "The language-model provider failed in an unexpected way.",
            details=context,
        )


__all__ = ["OpenAIProvider"]
