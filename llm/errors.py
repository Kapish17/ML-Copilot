"""Errors raised by the language-model layer.

Plain Python exceptions with no HTTP meaning, matching ``ml/errors.py`` and
``rag/errors.py``. A future API layer translates them; this package never
imports a web framework.

Two rules hold for every error here, and they are the reason provider
exceptions are wrapped rather than propagated.

**No credential ever appears in an error.** Not in the message, not in the
details, not in the chained cause's text. A provider SDK is free to put a
request URL or an authorisation header in its own exception; this layer never
passes one through.

**No raw provider exception reaches a caller.** The SDK's class names and
internals are its business. Each is mapped onto one of the classes below, so a
caller can handle "rate limited" or "timed out" without importing the SDK, and
so swapping providers does not change what callers catch.
"""

from __future__ import annotations

from typing import Any


class LLMError(Exception):
    """Base class for every failure in the language-model layer."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        """Store the message and any structured context.

        Args:
            message: Explanation written for a person reading a log or an API
                response. Never contains a stack trace, a credential or a
                filesystem path.
            details: Machine-readable context — the provider name, the model,
                a retry count. Never a key.
        """
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = details or {}


class LLMConfigurationError(LLMError):
    """The layer is not configured well enough to generate anything.

    Raised for an unknown provider name, an out-of-range setting, and — the
    common case — a missing API key. Deliberately distinct from the failures
    below: nothing was attempted, so nothing was spent, and the fix is local.
    """


class LLMProviderError(LLMError):
    """Base class for a failure that came from the provider."""


class LLMAuthenticationError(LLMProviderError):
    """The provider rejected the credential.

    A key that is present but wrong, expired or lacking permission. The key
    itself is never echoed back, so the message says what happened and not
    what was sent.
    """


class LLMTimeoutError(LLMProviderError):
    """The provider did not answer within the configured timeout."""


class LLMRateLimitError(LLMProviderError):
    """The provider is throttling, or a quota is exhausted."""


class LLMUnavailableError(LLMProviderError):
    """The provider could not be reached, or returned a server error."""


class LLMContextTooLargeError(LLMProviderError):
    """The request exceeded the model's context window.

    Distinct from the other provider errors because it is actionable: the
    caller can lower ``max_context_chars`` or ``max_context_chunks`` and try
    again, rather than waiting for someone else's service to recover.
    """


class LLMResponseError(LLMProviderError):
    """The provider answered, but not with something usable.

    An empty completion, a response with no choices, a truncated payload. The
    request succeeded at the transport level and still produced nothing to
    ground, which is a different problem from the connection failing.
    """


class LLMDependencyError(LLMConfigurationError):
    """The provider's SDK is not installed.

    A configuration problem rather than a provider problem: nothing was
    contacted, and the fix is an install.
    """


__all__ = [
    "LLMAuthenticationError",
    "LLMConfigurationError",
    "LLMContextTooLargeError",
    "LLMDependencyError",
    "LLMError",
    "LLMProviderError",
    "LLMRateLimitError",
    "LLMResponseError",
    "LLMTimeoutError",
    "LLMUnavailableError",
]
