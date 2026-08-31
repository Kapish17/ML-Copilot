"""The two moves a planner is allowed to make, and how its text becomes one.

A planner's whole vocabulary is this:

``{"action": "tool", "tool": "<name>", "arguments": {...}}``
``{"action": "final"}``

That is the entire protocol. There is no third action, so there is nothing to
add one — no "execute", no "python", no "shell", no "fetch". A response that
does not parse into one of these two is a
:class:`~agent.errors.MalformedPlanError`, and malformed is where it stops:
this module never falls back to reading the text as an instruction, never
extracts a code block and never guesses at intent.

That last point is the security property worth stating plainly. If a planner
replies with

.. code-block:: text

    ```python
    import os
    os.environ["LLM_API_KEY"]
    ```

there is no branch of this parser that reaches an interpreter. It is text that
does not parse as a decision, so it is rejected as malformed — the same
outcome as an empty reply or a half-written brace. Nothing in this package
imports :mod:`subprocess`, calls :func:`eval` or :func:`exec`, or resolves a
string to a callable.

JSON is the wire format because it is data. It is read with
:func:`json.loads`, which cannot execute anything, never with a literal
evaluator or a template.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from agent.errors import MalformedPlanError

#: The only two actions that exist.
ACTION_TOOL = "tool"
ACTION_FINAL = "final"
ACTIONS: tuple[str, ...] = (ACTION_TOOL, ACTION_FINAL)

#: Longest planner response this module will even attempt to parse. A decision
#: is a short object; anything longer is a planner that has started writing
#: prose, and reading further only spends time.
MAX_PLAN_CHARS = 20_000

#: Strips a ```json fence when a model wraps its object in one. This is
#: formatting, not content: the fence is removed and the *inside* still has to
#: parse as one of the two actions. It is not a path to running a code block —
#: a ```python fence whose body is not a decision object is still malformed.
_FENCE = re.compile(r"^\s*```(?:json|JSON)?\s*(?P<body>.*?)\s*```\s*$", re.DOTALL)

#: Finds the outermost JSON object when a model adds a sentence around it.
_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class PlanStep:
    """One decision: call a named tool, or stop and answer."""

    action: str
    tool: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    #: A short, optional note the planner may give about the choice. Recorded
    #: as metadata, never as reasoning: it is one line about *what* was chosen,
    #: not a trace of how. See ``agent/README.md`` on chain-of-thought.
    reason: str | None = None

    @property
    def is_tool_call(self) -> bool:
        """True when this step asks for a tool."""
        return self.action == ACTION_TOOL

    @property
    def is_final(self) -> bool:
        """True when this step asks to stop and answer."""
        return self.action == ACTION_FINAL

    def as_dict(self) -> dict[str, Any]:
        """Render the step as plain JSON-safe values."""
        payload: dict[str, Any] = {"action": self.action}
        if self.tool is not None:
            payload["tool"] = self.tool
        if self.arguments:
            payload["arguments"] = self.arguments
        if self.reason:
            payload["reason"] = self.reason
        return payload


#: How long a planner's optional note may be before it is cut. Short by
#: design: it is a label, not an explanation.
MAX_REASON_CHARS = 240

#: Longest string that will be treated as a tool name at all. Registered names
#: are short identifiers; a long one is a planner putting text somewhere it
#: does not belong.
MAX_TOOL_NAME_CHARS = 100


def _strip_fence(text: str) -> str:
    """Remove a surrounding code fence, if there is one."""
    match = _FENCE.match(text)
    return match.group("body") if match else text


def parse_plan(text: str) -> PlanStep:
    """Turn a planner's response into a decision, or refuse it.

    Args:
        text: Exactly what the planner produced.

    Returns:
        PlanStep: The requested tool call, or the request to answer.

    Raises:
        MalformedPlanError: If the response is empty, too long, not JSON, not
            an object, missing its action, or naming an action that does not
            exist. Every one of these is reported as malformed rather than
            interpreted — including a response that is entirely code, prose or
            an instruction.
    """
    if not isinstance(text, str) or not text.strip():
        raise MalformedPlanError(
            "The planner returned nothing. A decision must be a JSON object "
            "with an 'action' of 'tool' or 'final'.",
            details={"reason": "empty"},
        )
    if len(text) > MAX_PLAN_CHARS:
        raise MalformedPlanError(
            "The planner's response was too long to be a decision.",
            details={"reason": "too_long", "length": len(text)},
        )

    candidate = _strip_fence(text.strip())
    try:
        payload = json.loads(candidate)
    except (ValueError, TypeError):
        match = _OBJECT.search(candidate)
        if match is None:
            raise MalformedPlanError(
                "The planner's response was not a decision. Expected a JSON "
                "object with an 'action' of 'tool' or 'final'; no tool was "
                "called and nothing was executed.",
                details={"reason": "not_json"},
            ) from None
        try:
            payload = json.loads(match.group(0))
        except (ValueError, TypeError):
            raise MalformedPlanError(
                "The planner's response was not a decision. Expected a JSON "
                "object with an 'action' of 'tool' or 'final'; no tool was "
                "called and nothing was executed.",
                details={"reason": "not_json"},
            ) from None

    if not isinstance(payload, dict):
        raise MalformedPlanError(
            "A decision must be a JSON object.",
            details={"reason": "not_an_object"},
        )

    action = payload.get("action")
    if not isinstance(action, str) or action.strip().lower() not in ACTIONS:
        raise MalformedPlanError(
            "A decision must name an action of 'tool' or 'final'. "
            f"Got: {action!r}.",
            details={"reason": "unknown_action", "action": str(action)[:80]},
        )
    action = action.strip().lower()

    reason = payload.get("reason")
    reason = reason.strip()[:MAX_REASON_CHARS] if isinstance(reason, str) else None

    if action == ACTION_FINAL:
        return PlanStep(action=ACTION_FINAL, reason=reason or None)

    tool = payload.get("tool") or payload.get("tool_name")
    if not isinstance(tool, str) or not tool.strip():
        raise MalformedPlanError(
            "A tool decision must name the tool to call.",
            details={"reason": "missing_tool"},
        )
    if len(tool) > MAX_TOOL_NAME_CHARS:
        # A registered name is short. An over-long one is rejected here rather
        # than echoed back through a "no such tool" message, which is the one
        # place a planner could otherwise put arbitrary text into the record.
        raise MalformedPlanError(
            "A tool name that long is not a tool name.",
            details={"reason": "tool_name_too_long", "length": len(tool)},
        )

    arguments = payload.get("arguments", payload.get("input", {}))
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        raise MalformedPlanError(
            "Tool arguments must be an object of named values.",
            details={"reason": "invalid_arguments", "tool": tool.strip()[:80]},
        )

    return PlanStep(
        action=ACTION_TOOL,
        tool=tool.strip(),
        arguments=arguments,
        reason=reason or None,
    )


__all__ = [
    "ACTIONS",
    "ACTION_FINAL",
    "ACTION_TOOL",
    "MAX_PLAN_CHARS",
    "MAX_REASON_CHARS",
    "MAX_TOOL_NAME_CHARS",
    "PlanStep",
    "parse_plan",
]
