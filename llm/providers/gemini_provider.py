"""The Gemini provider.

Uses Google's official ``google-genai`` SDK (``pip install google-genai``),
which is the SDK Google's own documentation currently recommends for the
Gemini API — not the older, deprecated ``google-generativeai`` package. It
talks to Google's own endpoint; no ``base_url`` is needed, unlike the OpenAI
provider, which exists precisely to be pointed at other endpoints.

Existing for one practical reason: Gemini's "flash" models are, at the time
of writing, usable within the Gemini API's free tier, inside Google's
published rate and request-per-day limits — see
https://ai.google.dev/gemini-api/docs/pricing for current figures, since a
quota is Google's to change and this module makes no promise about it. This
is an alternative to OpenAI, not a replacement: ``LLM_PROVIDER`` chooses
between them, and everything above this module — grounding, the agent, the
answer service — depends on :class:`~llm.providers.base.LLMProvider` and
never on which one is running.

Same three obligations as every provider (see ``llm/providers/base.py``):
lazy construction, typed failures, and no credential leakage. Also the same
shape as ``openai_provider.py`` on purpose — a reader who understands one
understands the other.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from llm.config import LLMConfig, PROVIDER_GEMINI
from llm.errors import (
    LLMAuthenticationError,
    LLMContextTooLargeError,
    LLMDependencyError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from llm.messages import GenerationRequest, GenerationResult, Role

logger = logging.getLogger(__name__)

INSTALL_HINT = (
    "The Gemini provider needs the SDK: pip install google-genai. "
    "Retrieval works without it; only answer generation needs it."
)

#: Substrings that mark a bad request as a context-window problem rather than
#: a malformed one. Matched case-insensitively against the provider's message.
#: Gemini phrases this differently from OpenAI, so the list is its own.
_CONTEXT_MARKERS = (
    "context length",
    "context_length",
    "maximum context",
    "too many tokens",
    "token count exceeds",
    "exceeds the maximum number of tokens",
    "reduce the length",
    "input token count",
)

#: Gemini's finish reasons, translated to the vocabulary the rest of this
#: layer already speaks (``GenerationResult.is_truncated`` checks for
#: ``"length"``, matching what the OpenAI provider reports). Anything not
#: listed falls back to its own name, lower-cased, so a reader still learns
#: something even for a reason this mapping has not seen yet.
_FINISH_REASONS = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
}


def _finish_reason(raw: Any) -> str | None:
    """Normalise a Gemini finish reason to this project's vocabulary."""
    if raw is None:
        return None
    # google.genai.types.FinishReason is a str Enum; ``.value`` gives the
    # plain "STOP" rather than "FinishReason.STOP".
    name = str(getattr(raw, "value", raw))
    return _FINISH_REASONS.get(name, name.lower())


class GeminiProvider:
    """Generates completions through the Gemini API."""

    def __init__(self, config: LLMConfig) -> None:
        """Configure the provider without building anything.

        Args:
            config: Supplies the model, the timeout, the retry budget and the
                *name* of the environment variable holding the key. The key
                itself is not read here.
        """
        self._config = config
        self._client: Any | None = None

    @property
    def name(self) -> str:
        """Stable identifier of this provider."""
        return PROVIDER_GEMINI

    @property
    def model(self) -> str:
        """The model this provider will ask for."""
        return self._config.model

    @property
    def is_ready(self) -> bool:
        """Whether a credential is configured.

        Does not contact the provider and does not reveal the credential. A
        key that is present but invalid still reads as ready here; only a
        real call can tell the difference, and it raises
        :class:`~llm.errors.LLMAuthenticationError` when it does.
        """
        return self._config.has_api_key

    @property
    def is_loaded(self) -> bool:
        """Whether an SDK client has actually been built yet."""
        return self._client is not None

    # -- Client --------------------------------------------------------

    def _build_client(self) -> Any:
        """Create the SDK client on first use.

        The credential is checked before the SDK is imported. A missing key
        is the far more common problem and the more actionable message, and
        checking it first costs nothing.

        Raises:
            LLMConfigurationError: If no API key is configured.
            LLMDependencyError: If the SDK is not installed.
        """
        # Raises LLMConfigurationError when unset. The value is passed
        # straight to the SDK and never bound to an attribute here.
        api_key = self._config.resolve_api_key()

        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise LLMDependencyError(
                f"The 'google-genai' package is not installed. {INSTALL_HINT}",
                details={"provider": self.name, "model": self._config.model},
            ) from exc

        http_options: dict[str, Any] = {
            # Milliseconds — the SDK's own unit, unlike the seconds this
            # project's configuration uses everywhere else.
            "timeout": int(self._config.timeout_seconds * 1000),
            # Left unset, this SDK makes exactly one attempt and gives up —
            # unlike the OpenAI SDK, which retries a transient failure on its
            # own once handed ``max_retries``. Gemini's own API returns a
            # plain 5xx under load routinely enough (observed directly: a
            # ``ServerError`` a few seconds into an otherwise normal request)
            # that leaving this unset turns a load blip into a 502 for every
            # caller. ``max_retries`` is retries *after* the first attempt,
            # matching what the rest of this project's configuration means by
            # the name, so the attempt count here is one more than that.
            "retry_options": types.HttpRetryOptions(
                attempts=self._config.max_retries + 1,
                http_status_codes=(408, 429, 500, 502, 503, 504),
            ),
        }
        # Optional, and only for someone routing Gemini traffic elsewhere —
        # a corporate proxy, a regional endpoint. The official API needs
        # nothing here.
        if self._config.base_url:
            http_options["base_url"] = self._config.base_url

        return genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(**http_options),
        )

    def _client_or_build(self) -> Any:
        """Return the cached client, building it the first time."""
        if self._client is None:
            self._client = self._build_client()
        return self._client

    # -- Generation ------------------------------------------------------

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

        from google.genai import types

        contents = [
            types.Content(
                role="model" if message.role is Role.ASSISTANT else "user",
                parts=[types.Part.from_text(text=message.content)],
            )
            for message in request.messages
            if message.role is not Role.SYSTEM
        ]
        config = types.GenerateContentConfig(
            system_instruction=request.system_prompt or None,
            temperature=request.temperature,
            max_output_tokens=request.max_output_tokens,
        )

        try:
            response = client.models.generate_content(
                model=request.model or self._config.model,
                contents=contents,
                config=config,
            )
        except Exception as exc:  # noqa: BLE001 - every SDK failure is mapped
            translated = self._translate(exc)
            # The outcome of a call that leaves this process is worth
            # recording either way. The exception *type* and this project's
            # own error code say what happened; the provider's message may
            # quote the request back and is left to the error envelope.
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
                and the caller is told which. Includes the case, specific to
                this SDK, where reading ``response.text`` itself raises
                because the model returned no text part — a safety block, for
                instance — rather than an empty string.
        """
        candidates = getattr(response, "candidates", None) or []
        finish_reason = _finish_reason(
            getattr(candidates[0], "finish_reason", None) if candidates else None
        )

        try:
            text = (response.text or "").strip()
        except (ValueError, AttributeError, IndexError, TypeError) as exc:
            raise LLMResponseError(
                "The provider returned a response this client could not read.",
                details={
                    "provider": self.name,
                    "model": request.model,
                    "reason": type(exc).__name__,
                    "finish_reason": finish_reason,
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

        usage = getattr(response, "usage_metadata", None)
        return GenerationResult(
            text=text,
            model=getattr(response, "model_version", request.model) or request.model,
            provider=self.name,
            finish_reason=finish_reason,
            prompt_tokens=getattr(usage, "prompt_token_count", None),
            completion_tokens=getattr(usage, "candidates_token_count", None),
            latency_seconds=latency,
        )

    # -- Error mapping -----------------------------------------------------

    def _translate(self, exc: Exception) -> Exception:
        """Map an SDK exception onto this project's error types.

        The SDK's own message is not passed through. It can carry a request
        URL, headers or an echoed payload, and nothing in it is worth the
        risk — so each case gets a message written here, and ``details``
        carries only the provider, the model and, where the SDK offers one, a
        status code.
        """
        context = {"provider": self.name, "model": self._config.model}

        try:
            from google.genai import errors as genai_errors
        except ImportError:  # pragma: no cover - only if the SDK vanished
            return LLMUnavailableError(
                "The language-model provider could not be reached.",
                details=context,
            )

        if isinstance(exc, genai_errors.ClientError):
            code = getattr(exc, "code", None)
            status = str(getattr(exc, "status", "") or "")

            if code in (401, 403) or status in ("UNAUTHENTICATED", "PERMISSION_DENIED"):
                return LLMAuthenticationError(
                    "The language-model provider rejected the configured "
                    "credential. Check the value of "
                    f"{self._config.api_key_env}.",
                    details={**context, "api_key_env": self._config.api_key_env},
                )
            if code == 429 or status == "RESOURCE_EXHAUSTED":
                return LLMRateLimitError(
                    "The language-model provider is rate limiting requests, "
                    "or a quota is exhausted.",
                    details=context,
                )
            if code == 404 or status == "NOT_FOUND":
                return LLMUnavailableError(
                    f"The model '{self._config.model}' is not available to "
                    "this credential or endpoint.",
                    details=context,
                )

            message = str(getattr(exc, "message", "") or str(exc)).lower()
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
                "The language-model provider rejected the request as "
                "malformed.",
                details={**context, "status_code": code},
            )

        if isinstance(exc, genai_errors.ServerError):
            return LLMUnavailableError(
                "The language-model provider could not be reached.",
                details={**context, "status_code": getattr(exc, "code", None)},
            )

        if isinstance(exc, genai_errors.APIError):
            return LLMUnavailableError(
                "The language-model provider returned an unexpected status.",
                details={**context, "status_code": getattr(exc, "code", None)},
            )

        # Errors below the SDK's own hierarchy: a request that never got a
        # response at all — DNS failure, connection refused, the timeout set
        # on the client above expiring. These arrive as whatever the
        # underlying HTTP client raises, so they are told apart by name
        # rather than by ``isinstance`` against a library this module does
        # not import.
        type_name = type(exc).__name__
        if "Timeout" in type_name:
            return LLMTimeoutError(
                "The language-model provider did not respond within "
                f"{self._config.timeout_seconds:g} seconds.",
                details={**context, "timeout_seconds": self._config.timeout_seconds},
            )
        if "Connect" in type_name or "Network" in type_name:
            return LLMUnavailableError(
                "The language-model provider could not be reached.",
                details=context,
            )

        logger.warning(
            "Unmapped %s from the %s provider", type(exc).__name__, self.name
        )
        return LLMUnavailableError(
            "The language-model provider failed in an unexpected way.",
            details=context,
        )


__all__ = ["GeminiProvider"]
