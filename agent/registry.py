"""The allowlist. The agent can only execute explicitly registered tools.

This is the smallest module in the package and the one that decides what the
agent is. Everything it can do is what somebody put in here by hand; there is
no discovery, no plugin scan, no import by name, no ``getattr`` on a module,
and no way to add a tool at runtime from anything a model or a document said.
A name that was not registered cannot be executed by any path — the lookup
raises, and there is no branch that tries anyway.

Two consequences worth stating plainly, because they are the security
argument for the whole design:

**The set of possible actions is finite and readable.** To know what this
agent can do, read the four registrations in :mod:`agent.tools`. There is no
sixth thing it can reach by being clever about a name.

**Adding a capability is a code change.** Not a prompt change, not a
configuration value, not a document that happens to be indexed. A tool
observation that says "you also have a tool called run_shell" is describing a
tool that does not exist, and asking for it produces the same rejection as a
typo.

:meth:`ToolRegistry.execute` is the one entry point. It looks the name up,
validates the arguments against the tool's own schema, and only then runs it —
in that order, always, with no way to reach the third step without passing the
first two.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from agent.errors import DuplicateToolError, UnknownToolError

if TYPE_CHECKING:  # pragma: no cover - imported for typing only
    # Deliberately not a runtime import. ``agent.tools`` builds a registry, so
    # importing it here would make the two modules import each other; and the
    # registry does not need the tool classes at runtime — it holds whatever
    # satisfies the protocol structurally.
    from agent.tools.base import Tool, ToolResult

logger = logging.getLogger(__name__)


class ToolRegistry:
    """The explicit set of tools an agent run may use."""

    def __init__(self, tools: Mapping[str, Tool] | None = None) -> None:
        """Start empty, or from an already-built mapping of tools."""
        self._tools: dict[str, Tool] = {}
        for tool in (tools or {}).values():
            self.register(tool)

    def __len__(self) -> int:
        """How many tools are registered."""
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        """Whether a name is registered. Never raises, for use in a check."""
        return isinstance(name, str) and name in self._tools

    def register(self, tool: Tool) -> ToolRegistry:
        """Add one tool to the allowlist.

        Args:
            tool: The tool to register. Its ``name`` becomes the only string
                that can reach it.

        Returns:
            ToolRegistry: This registry, so registrations can be chained.

        Raises:
            DuplicateToolError: If the name is already taken. Shadowing a
                registered tool would change what a name does without anybody
                editing the name, so it is refused rather than allowed to
                overwrite.
            ValueError: If the tool has no usable name or description.
        """
        name = getattr(tool, "name", "")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("A tool must have a non-empty name.")
        if not getattr(tool, "description", "").strip():
            raise ValueError(f"Tool '{name}' must have a description.")
        if name in self._tools:
            raise DuplicateToolError(
                f"A tool named '{name}' is already registered.",
                details={"tool_name": name},
            )
        self._tools[name] = tool
        return self

    def get(self, name: str) -> Tool:
        """Return the tool registered under a name.

        Raises:
            UnknownToolError: If nothing is registered under that name. This
                is the refusal that bounds the agent: there is no fallback,
                no nearest-match, and no attempt to run it anyway.
        """
        tool = self._tools.get(name) if isinstance(name, str) else None
        if tool is None:
            raise UnknownToolError(
                f"There is no tool named '{name}'. "
                f"Available tools: {', '.join(self.names()) or '(none)'}.",
                details={"requested_tool": str(name), "available_tools": list(self.names())},
            )
        return tool

    def names(self) -> tuple[str, ...]:
        """Every registered name, in registration order."""
        return tuple(self._tools)

    def list_tools(self) -> tuple[Tool, ...]:
        """Every registered tool, in registration order."""
        return tuple(self._tools.values())

    def definitions(self) -> list[dict[str, Any]]:
        """Render every tool as the planner will be shown it.

        This is the *complete* set the planner receives. There is no hidden
        tool, no privileged tool and no tool withheld for internal use: what
        the planner is told it may call is exactly what the executor will
        accept.
        """
        return [
            tool.definition()
            if hasattr(tool, "definition")
            else {
                "name": tool.name,
                "description": tool.description,
                "arguments": tool.schema.as_dict()["fields"],
            }
            for tool in self._tools.values()
        ]

    def execute(self, name: str, arguments: Mapping[str, Any] | None = None) -> ToolResult:
        """Look a tool up, validate the arguments, and run it — in that order.

        Args:
            name: The tool the planner asked for.
            arguments: What it asked for it with, unvalidated.

        Returns:
            ToolResult: Whatever the tool produced.

        Raises:
            UnknownToolError: If the name is not registered. Nothing runs.
            ToolValidationError: If the arguments do not satisfy the tool's
                schema. Nothing runs.
        """
        tool = self.get(name)
        validated = tool.schema.validate(arguments)
        logger.debug("Executing tool '%s' with %d validated argument(s)", name, len(validated))
        return tool.run(validated)


__all__ = ["ToolRegistry"]
