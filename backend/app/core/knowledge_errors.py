"""Translation of retrieval and language-model exceptions into the API contract.

``rag/`` and ``llm/`` raise plain Python exceptions with no HTTP meaning — that
independence is deliberate, and this module is the single place where it is
bridged for those two layers. It is the sibling of :mod:`app.core.ml_errors`,
which does the same job for ``ml/``, and it reuses that module's path
sanitiser so there is one definition of "scrub this before it leaves".

The statuses are chosen around one distinction that matters more than the
others:

**A question that was processed but could not be answered is not an error.**
``insufficient_evidence`` and ``grounding_failed`` are *results*. They travel
as 200 with a status field, because the request was valid, the work was done,
and the honest outcome is that the answer is not trustworthy. Returning 5xx
for them would tell a client to retry something that will fail identically.

What *is* an error:

- **502** — the provider failed. Someone else's service timed out, throttled
  or fell over. Retrying later may work.
- **503** — the layer is not configured to answer at all. No API key, no SDK,
  or the index has never been built. Nothing will work until a human acts.
- **4xx** — the request itself is wrong.

Three things are scrubbed on the way out, and the reasons are worth stating.

**Credentials.** No message or detail from ``llm/`` contains one — the layer
is built that way — but this module also never passes a provider's own
exception text through, so a vendor message that echoed an authorisation
header could not reach a client even if one appeared.

**Filesystem paths.** A "not found" that names the index directory is useful
in a log and has no business in an HTTP response.

**Internal failures.** Anything mapped to 5xx gets a written message; the
class name and the details stay in the log.
"""

from __future__ import annotations

from typing import Any

from app.core.ml_errors import sanitise_details
from llm.errors import (
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMContextTooLargeError,
    LLMDependencyError,
    LLMError,
    LLMProviderError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from rag.errors import (
    ConfigurationError as RagConfigurationError,
    CorruptIndexError,
    EmbeddingDimensionError,
    EmbeddingError,
    EmbeddingProviderUnavailableError,
    IngestionError,
    RagError,
    RetrievalError,
    SourceNotFoundError,
    UnsafeSourceError,
    VectorStoreError,
)

_GENERIC_MESSAGE = "The request could not be completed."

#: ``exception type -> (api code, http status)``, most specific first. Order
#: matters: the first class an exception is an instance of wins, so subclasses
#: must precede the bases they refine.
_RAG_ERROR_MAPPING: tuple[tuple[type[Exception], str, int], ...] = (
    # The caller's request.
    (RagConfigurationError, "invalid_search_request", 400),
    (RetrievalError, "invalid_search_request", 400),
    (SourceNotFoundError, "source_not_found", 404),
    (UnsafeSourceError, "unsafe_source", 400),
    # The retrieval infrastructure. 503 rather than 500 because the index can
    # be rebuilt: the service is unavailable, not permanently broken, and the
    # distinction tells an operator what to do.
    (CorruptIndexError, "retrieval_index_unavailable", 503),
    (EmbeddingDimensionError, "retrieval_index_unavailable", 503),
    (EmbeddingProviderUnavailableError, "embedding_provider_unavailable", 503),
    (EmbeddingError, "embedding_error", 500),
    (VectorStoreError, "retrieval_store_error", 500),
    (IngestionError, "ingestion_error", 500),
    (RagError, "retrieval_error", 500),
)

_LLM_ERROR_MAPPING: tuple[tuple[type[Exception], str, int], ...] = (
    # Not configured to answer. Nothing was attempted and nothing was spent;
    # a human has to set something before any request will work.
    (LLMDependencyError, "llm_not_configured", 503),
    (LLMConfigurationError, "llm_not_configured", 503),
    # The provider failed. Someone else's service; retrying later may work.
    (LLMAuthenticationError, "llm_authentication_failed", 502),
    (LLMTimeoutError, "llm_timeout", 502),
    (LLMRateLimitError, "llm_rate_limited", 502),
    (LLMUnavailableError, "llm_unavailable", 502),
    (LLMResponseError, "llm_response_error", 502),
    # Actionable by the caller: lower the context limits and try again.
    (LLMContextTooLargeError, "llm_context_too_large", 400),
    (LLMProviderError, "llm_provider_error", 502),
    (LLMError, "llm_error", 502),
)

_ERROR_MAPPING = _RAG_ERROR_MAPPING + _LLM_ERROR_MAPPING

#: Statuses whose message is authored for a client. A 5xx describes a failure
#: on this side, so its message is replaced with a written one.
_CLIENT_FACING_MAX_STATUS = 499

#: 5xx statuses whose message is still safe and useful to show. A missing API
#: key or an unbuilt index is a configuration problem the operator must fix,
#: and saying which one saves them guessing — none of these messages contains
#: a credential, a path or a provider's own text.
_INFORMATIVE_STATUSES = frozenset({502, 503})


def translate_knowledge_error(exc: Exception) -> tuple[str, int, str, dict[str, Any]]:
    """Map a retrieval or language-model exception onto the API contract.

    Args:
        exc: The exception raised by ``rag`` or ``llm``.

    Returns:
        tuple: ``(code, status_code, message, details)`` ready for the
        envelope. A 500 gets a generic message and no details; a 502 or 503
        keeps its own, because those messages are written for an operator and
        say what to do.
    """
    for error_type, code, status_code in _ERROR_MAPPING:
        if isinstance(exc, error_type):
            break
    else:
        code, status_code = "internal_error", 500

    if status_code > _CLIENT_FACING_MAX_STATUS and status_code not in _INFORMATIVE_STATUSES:
        return code, status_code, _GENERIC_MESSAGE, {}

    message = getattr(exc, "message", None) or str(exc) or _GENERIC_MESSAGE
    details = sanitise_details(getattr(exc, "details", None))
    return code, status_code, message, details


def is_client_error(exc: Exception) -> bool:
    """Return whether the exception maps to a 4xx status."""
    return translate_knowledge_error(exc)[1] <= _CLIENT_FACING_MAX_STATUS


__all__ = [
    "LLMError",
    "RagError",
    "is_client_error",
    "translate_knowledge_error",
]
