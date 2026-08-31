"""Translation of agent-layer failures into the API contract.

``agent/`` raises plain Python exceptions with no HTTP meaning, exactly as
``ml/``, ``rag/`` and ``llm/`` do. This module is the single place where that
independence is bridged, and the sibling of :mod:`app.core.ml_errors` and
:mod:`app.core.knowledge_errors`.

There are two quite different things to map, and keeping them apart is most of
the design.

**Exceptions the orchestrator raises.** A configuration it cannot run under, a
tool registered twice — programming or setup problems, mapped below.

**Failures the orchestrator *returns*.** This is the larger and more important
set. :class:`~agent.results.AgentResult` reports a run that could not be
completed as a status with an ``error_code`` rather than as an exception,
because for a library caller reading a field beats catching something. Over
HTTP that is the wrong shape: no answer was produced and a client must not be
able to read one out of the body. So :data:`RUN_FAILURE_MAPPING` converts those
codes — and *only* those — into HTTP errors.

What is deliberately **not** in that table is the point of it. ``completed``,
``partial``, ``insufficient_evidence`` and ``grounding_failed`` are results:
the request was valid, the work was done, and the outcome is reported in the
body with a 200. An agent that searched, found nothing relevant, and said so
has done its job. Returning 5xx for that would tell a client to retry
something that will fail identically, and would hide an honest answer behind
an error.

Nothing a provider wrote reaches a client through here. The agent layer has
already replaced every vendor exception with an authored message under a
stable code; this module maps the code and passes the authored message.
"""

from __future__ import annotations

from typing import Any

from agent.errors import (
    AgentConfigurationError,
    AgentError,
    BudgetExhaustedError,
    DuplicateToolError,
    MalformedPlanError,
    PlannerError,
    PlannerProviderError,
    PlannerUnavailableError,
    ToolError,
    ToolExecutionError,
    ToolValidationError,
    UnknownToolError,
)
from app.core.ml_errors import sanitise_details

_GENERIC_MESSAGE = "The request could not be completed."

#: ``exception type -> (api code, http status)``, most specific first. Order
#: matters: the first class an exception is an instance of wins, so subclasses
#: must precede the bases they refine.
_AGENT_ERROR_MAPPING: tuple[tuple[type[Exception], str, int], ...] = (
    # Not configured to plan at all. Nothing was attempted and nothing spent;
    # a human has to set a credential before any request will work.
    (PlannerUnavailableError, "llm_not_configured", 503),
    # The planner's provider failed, or produced something that was not a
    # decision. Both are upstream problems with the model service, and both
    # may succeed on a retry.
    (PlannerProviderError, "agent_provider_error", 502),
    (MalformedPlanError, "agent_planner_error", 502),
    (PlannerError, "agent_planner_error", 502),
    # The agent is set up wrongly on this side. A 400 rather than a 500
    # because the message says which limit or tool is at fault and someone can
    # act on it.
    (AgentConfigurationError, "invalid_agent_configuration", 400),
    (DuplicateToolError, "invalid_agent_configuration", 400),
    # These three normally never escape the orchestrator — an unknown tool and
    # an invalid argument set become rejected *observations* so the planner
    # can correct itself, and a failing tool becomes a failed observation.
    # They are mapped anyway, for a caller that drives the registry directly.
    (UnknownToolError, "unknown_tool", 400),
    (ToolValidationError, "invalid_tool_arguments", 400),
    (ToolExecutionError, "tool_execution_failed", 502),
    (ToolError, "tool_error", 500),
    # A budget ran out. Normally returned as a partial result rather than
    # raised; mapped for completeness.
    (BudgetExhaustedError, "agent_budget_exhausted", 400),
    (AgentError, "agent_error", 500),
)

#: How an :class:`~agent.results.AgentResult` *failure* becomes an HTTP status.
#:
#: Keyed by the ``error_code`` a failed run reports. A code this table does not
#: know falls back to :data:`DEFAULT_RUN_FAILURE`, which treats it as an
#: upstream planner problem — the safer guess, since every failure the
#: orchestrator can produce today comes from the planner.
#:
#: Note again what is absent: no entry for a successful run in any of its four
#: outcomes. Those never reach this table.
RUN_FAILURE_MAPPING: dict[str, tuple[str, int]] = {
    "planner_unavailable": ("llm_not_configured", 503),
    "planner_provider_error": ("agent_provider_error", 502),
    "malformed_plan": ("agent_planner_error", 502),
    "agent_configuration_error": ("invalid_agent_configuration", 400),
}

#: Used when a run failed with a code the table above does not know.
DEFAULT_RUN_FAILURE = ("agent_planner_error", 502)

#: Statuses whose message is authored for a client.
_CLIENT_FACING_MAX_STATUS = 499

#: 5xx statuses whose own message is still safe and useful to show. A missing
#: credential or a provider timeout is something an operator must act on, and
#: saying which saves them guessing — none of these messages contains a
#: credential, a path, or a provider's own text.
_INFORMATIVE_STATUSES = frozenset({502, 503})


def translate_agent_error(exc: Exception) -> tuple[str, int, str, dict[str, Any]]:
    """Map an agent-layer exception onto the API contract.

    Args:
        exc: The exception raised by ``agent/``.

    Returns:
        tuple: ``(code, status_code, message, details)`` ready for the
        envelope. A 500 gets a generic message and no details; a 502 or 503
        keeps its own, because those are written for an operator.
    """
    for error_type, code, status_code in _AGENT_ERROR_MAPPING:
        if isinstance(exc, error_type):
            break
    else:
        code, status_code = "internal_error", 500

    if status_code > _CLIENT_FACING_MAX_STATUS and status_code not in _INFORMATIVE_STATUSES:
        return code, status_code, _GENERIC_MESSAGE, {}

    message = getattr(exc, "message", None) or str(exc) or _GENERIC_MESSAGE
    details = sanitise_details(getattr(exc, "details", None))
    return code, status_code, message, details


def translate_run_failure(error_code: str | None) -> tuple[str, int]:
    """Map a failed run's ``error_code`` onto an API code and status."""
    return RUN_FAILURE_MAPPING.get(error_code or "", DEFAULT_RUN_FAILURE)


def is_client_error(exc: Exception) -> bool:
    """Return whether the exception maps to a 4xx status."""
    return translate_agent_error(exc)[1] <= _CLIENT_FACING_MAX_STATUS


__all__ = [
    "DEFAULT_RUN_FAILURE",
    "RUN_FAILURE_MAPPING",
    "AgentError",
    "is_client_error",
    "translate_agent_error",
    "translate_run_failure",
]
