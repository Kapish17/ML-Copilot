"""Running the bounded agent for an HTTP request, as an application service.

``budgets``  what a request may ask for, and the one direction it may move
``errors``   the two refusals only an HTTP caller cares about
``service``  ``AgentService`` — one question, one run

Nothing in this package imports FastAPI, so the same service is drivable from
a script, a test or a future worker; the HTTP route is one caller among
several. It computes nothing: planning, tool selection, execution and
grounding all belong to ``agent/``, and the capabilities being orchestrated
belong to ``ml/``, ``rag/`` and ``llm/``.
"""

from app.services.agent.budgets import LOWERABLE_BUDGETS, resolve_config
from app.services.agent.errors import (
    AgentBudgetError,
    AgentServiceError,
    AgentUnavailableError,
)
from app.services.agent.service import (
    NOT_CONFIGURED_MESSAGE,
    AgentRunFailedError,
    AgentService,
)

__all__ = [
    "LOWERABLE_BUDGETS",
    "NOT_CONFIGURED_MESSAGE",
    "AgentBudgetError",
    "AgentRunFailedError",
    "AgentService",
    "AgentServiceError",
    "AgentUnavailableError",
    "resolve_config",
]
