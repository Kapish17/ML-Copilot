"""Failures that belong to the API's view of the knowledge layers.

Two conditions that ``rag/`` and ``llm/`` have no opinion about, because they
only matter to something serving requests.

An **unbuilt index** is not an error to a library caller — an empty store
returns no results, which is a truthful answer to "search this". Over HTTP it
is a different thing entirely: a client asking a reasonable question would be
told "no relevant evidence" when the truth is that nothing has been indexed
yet, and would go away believing the system has no answer rather than that it
has not been set up. So it is reported as a 503 that says what to run.

The same reasoning applies to a **provider with no credential**. The layer
already refuses to generate, and returns a structured
``configuration_error``; this error exists so the API can refuse *before*
retrieval work is done, and answer with the status the specification asks for.
"""

from __future__ import annotations

from app.core.errors import MLCopilotError


class KnowledgeError(MLCopilotError):
    """Base class for API-level failures of the knowledge endpoints."""

    code = "knowledge_error"
    status_code = 500


class IndexNotBuiltError(KnowledgeError):
    """No retrieval index has been built yet.

    Deliberately distinct from "the index holds nothing relevant". The first
    means a human has to run the indexer; the second means the question has no
    answer here. Collapsing them would send a client away with the wrong
    conclusion.
    """

    code = "retrieval_index_not_built"
    status_code = 503


class AnsweringUnavailableError(KnowledgeError):
    """Answer generation is not configured.

    Raised before any retrieval work happens, so a request that cannot
    possibly succeed does not spend an embedding pass first.
    """

    code = "llm_not_configured"
    status_code = 503


#: How an :class:`~llm.answers.Answer` failure becomes an HTTP status.
#:
#: The answer service *returns* provider and configuration failures rather
#: than raising them, which is right for a library: a caller reads a field
#: instead of catching something. Over HTTP they are genuine errors — the
#: request did not produce an answer and the client should not read one — so
#: they are converted here.
#:
#: Note what is **not** in this table. ``grounded``,
#: ``insufficient_evidence`` and ``grounding_failed`` are results, not
#: failures: the request was valid, the work was done, and the outcome is
#: reported in the body with a 200.
ANSWER_FAILURE_MAPPING: dict[str, tuple[str, int]] = {
    # Not configured to answer at all — a human has to set something.
    "llm_configuration": ("llm_not_configured", 503),
    "llm_dependency": ("llm_not_configured", 503),
    # The provider failed. Someone else's service; retrying may work.
    "llm_timeout": ("llm_timeout", 502),
    "llm_rate_limit": ("llm_rate_limited", 502),
    "llm_authentication": ("llm_authentication_failed", 502),
    "llm_unavailable": ("llm_unavailable", 502),
    "llm_response": ("llm_response_error", 502),
    "llm_model": ("llm_provider_error", 502),
    # Actionable by the caller: ask for less evidence.
    "llm_context_too_large": ("llm_context_too_large", 400),
}

#: Used when an answer failed with a code this table does not know.
DEFAULT_ANSWER_FAILURE = ("llm_provider_error", 502)


class AnsweringFailedError(KnowledgeError):
    """An answer request failed rather than producing a result.

    Carries the code and status from :data:`ANSWER_FAILURE_MAPPING`, so a
    timeout and a missing credential reach the client as different things
    without this module needing a class per failure.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str,
        status_code: int,
        details: dict | None = None,
    ) -> None:
        """Build the error with the status its cause maps to."""
        super().__init__(message, details=details)
        # Instance attributes shadow the class ones the handler reads.
        self.code = code
        self.status_code = status_code

    @classmethod
    def from_answer_code(
        cls, error_code: str | None, message: str, details: dict | None = None
    ) -> AnsweringFailedError:
        """Build the error for an answer's ``error_code``."""
        code, status_code = ANSWER_FAILURE_MAPPING.get(
            error_code or "", DEFAULT_ANSWER_FAILURE
        )
        return cls(message, code=code, status_code=status_code, details=details)


__all__ = [
    "ANSWER_FAILURE_MAPPING",
    "AnsweringFailedError",
    "AnsweringUnavailableError",
    "IndexNotBuiltError",
    "KnowledgeError",
]
