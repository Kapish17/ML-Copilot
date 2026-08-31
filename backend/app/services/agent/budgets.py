"""What a request may ask for, and the one direction it may move.

A client may make an agent run *smaller* — fewer tool calls, fewer planning
turns, less accumulated evidence. It may never make one larger. That is the
whole policy, and it is worth stating why it is a policy rather than a
clamp.

The server's limits exist because an agent run costs real work: model calls,
cross-validated training, SHAP. If a request could raise them, they would not
be limits, they would be defaults — and the thing that decides how much a
question costs would be whoever is asking.

**Rejected, not capped.** A request for a hundred tool calls could be quietly
reduced to six and run. It is refused instead, because a client that believes
it was granted a hundred and receives a partial result after six has no way to
tell what happened. A 422 naming the limit is the shorter conversation, and it
is the same choice the knowledge endpoints made for ``top_k``.
"""

from __future__ import annotations

from typing import Any

from agent.config import AgentConfig
from app.services.agent.errors import AgentBudgetError

#: The fields a request may lower, mapped to the words used in a message.
LOWERABLE_BUDGETS: dict[str, str] = {
    "max_tool_calls": "tool calls",
    "max_iterations": "planning steps",
    "max_context_chars": "characters of observed material",
}


def resolve_config(
    server_config: AgentConfig, requested: dict[str, Any] | None
) -> AgentConfig:
    """Build the configuration one run will use.

    Args:
        server_config: The limits the server is configured with. These are the
            ceiling; nothing a request says can raise them.
        requested: Budgets from the request body. ``None`` and absent fields
            both mean "use the server's".

    Returns:
        AgentConfig: The server's configuration with any *lower* requested
        values applied.

    Raises:
        AgentBudgetError: If a requested value exceeds the server's limit. The
            message names the limit, so a client can retry with something the
            server will grant.
    """
    overrides: dict[str, Any] = {}

    for field, description in LOWERABLE_BUDGETS.items():
        value = (requested or {}).get(field)
        if value is None:
            continue

        ceiling = getattr(server_config, field)
        if value > ceiling:
            raise AgentBudgetError(
                f"'{field}' may be at most {ceiling} — the server allows no "
                f"more than {ceiling} {description} for one question. "
                "A request may lower a limit, never raise it.",
                details={"field": field, "requested": value, "maximum": ceiling},
            )
        overrides[field] = value

    if not overrides:
        return server_config

    # ``max_observation_chars`` must not exceed the context budget, so lowering
    # the context also lowers it. Done here rather than left to the caller
    # because an otherwise valid request would be refused by the config's own
    # validation, with a message about a field the client never sent.
    context = overrides.get("max_context_chars")
    if context is not None and context < server_config.max_observation_chars:
        overrides["max_observation_chars"] = context

    return server_config.with_overrides(**overrides)


__all__ = ["LOWERABLE_BUDGETS", "resolve_config"]
