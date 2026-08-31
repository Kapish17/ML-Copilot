"""The allowlist, and what it refuses.

These are the tests that make "the agent can only execute explicitly
registered tools" a fact rather than an intention. Most of them are about a
name *not* being there.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent.errors import DuplicateToolError, ToolValidationError, UnknownToolError
from agent.registry import ToolRegistry
from agent.schemas import STRING, ArgumentField, ArgumentSchema
from agent.tools.base import BaseTool, Tool, ToolResult


class EchoTool(BaseTool):
    """A minimal tool, for testing the registry rather than any behaviour."""

    tool_name = "echo"
    tool_description = "Return the text it was given."

    def __init__(self, name: str = "echo") -> None:
        """Allow the name to vary, so duplicates can be tested."""
        self.tool_name = name
        super().__init__()
        self.calls: list[dict[str, Any]] = []

    @property
    def schema(self) -> ArgumentSchema:
        """One required string."""
        return ArgumentSchema(
            fields=(
                ArgumentField(
                    name="text", type=STRING, description="Text to echo.", required=True
                ),
            )
        )

    def run(self, arguments: Any) -> ToolResult:
        """Record the call and echo."""
        self.calls.append(dict(arguments))
        return ToolResult(output={"text": arguments["text"]})


def test_a_registered_tool_can_be_looked_up() -> None:
    """The ordinary case."""
    tool = EchoTool()
    registry = ToolRegistry().register(tool)

    assert registry.get("echo") is tool
    assert "echo" in registry
    assert len(registry) == 1


def test_listing_reports_every_registered_tool() -> None:
    """A caller can see the whole allowlist."""
    registry = ToolRegistry().register(EchoTool("one")).register(EchoTool("two"))

    assert registry.names() == ("one", "two")
    assert [tool.name for tool in registry.list_tools()] == ["one", "two"]


def test_definitions_are_what_the_planner_is_shown() -> None:
    """Name, description and declared arguments — and nothing live."""
    registry = ToolRegistry().register(EchoTool())
    definition = registry.definitions()[0]

    assert definition["name"] == "echo"
    assert definition["description"]
    assert definition["arguments"][0]["name"] == "text"
    assert definition["arguments"][0]["required"] is True


def test_an_unknown_tool_name_is_refused() -> None:
    """The single most important refusal in the package."""
    registry = ToolRegistry().register(EchoTool())

    with pytest.raises(UnknownToolError) as caught:
        registry.get("run_shell")

    assert caught.value.code == "unknown_tool"
    assert "run_shell" in caught.value.details["requested_tool"]
    # The available tools are named, so a planner can correct itself.
    assert caught.value.details["available_tools"] == ["echo"]


@pytest.mark.parametrize(
    "name",
    [
        "shell",
        "python",
        "exec",
        "eval",
        "subprocess",
        "read_file",
        "write_file",
        "http_get",
        "fetch_url",
        "os.system",
        "echo ",
        "ECHO",
        "",
    ],
)
def test_no_plausible_alias_reaches_a_registered_tool(name: str) -> None:
    """There is no fuzzy match, no normalisation and no fallback.

    A trailing space and a different case are as unknown as ``subprocess``.
    Being forgiving about names would be the first step towards executing
    something nobody registered.
    """
    registry = ToolRegistry().register(EchoTool())

    with pytest.raises(UnknownToolError):
        registry.execute(name, {"text": "hello"})


def test_executing_an_unknown_tool_runs_nothing() -> None:
    """The refusal happens before any tool code is reached."""
    tool = EchoTool()
    registry = ToolRegistry().register(tool)

    with pytest.raises(UnknownToolError):
        registry.execute("not_a_tool", {"text": "hello"})

    assert tool.calls == []


def test_registering_the_same_name_twice_is_refused() -> None:
    """Shadowing would change what a name does without anyone editing it."""
    registry = ToolRegistry().register(EchoTool())

    with pytest.raises(DuplicateToolError):
        registry.register(EchoTool())


def test_a_tool_without_a_name_or_description_is_refused() -> None:
    """A planner cannot choose sensibly between undescribed options."""

    class Nameless:
        name = ""
        description = "something"
        schema = ArgumentSchema()

        def run(self, arguments: Any) -> ToolResult:
            return ToolResult(output={})

    with pytest.raises(ValueError):
        ToolRegistry().register(Nameless())  # type: ignore[arg-type]


def test_execute_validates_before_running() -> None:
    """Invalid arguments never reach the tool."""
    tool = EchoTool()
    registry = ToolRegistry().register(tool)

    with pytest.raises(ToolValidationError):
        registry.execute("echo", {"text": 42})

    assert tool.calls == []


def test_execute_passes_validated_arguments_through() -> None:
    """The happy path, and the proof that normalisation reaches the tool."""
    tool = EchoTool()
    registry = ToolRegistry().register(tool)

    result = registry.execute("echo", {"text": "  hello  "})

    assert result.output == {"text": "hello"}
    assert tool.calls == [{"text": "hello"}]


def test_a_registry_has_no_generic_execute_tool(registry: ToolRegistry) -> None:
    """The default registry offers four capabilities and no escape hatch."""
    assert set(registry.names()) == {
        "dataset_profile",
        "run_experiment",
        "search_knowledge",
        "explain_experiment",
    }

    for forbidden in ("execute", "run", "python", "shell", "bash", "eval", "http"):
        assert forbidden not in registry.names()


def test_the_real_tools_satisfy_the_tool_protocol(registry: ToolRegistry) -> None:
    """Structural conformance, checked rather than assumed."""
    for tool in registry.list_tools():
        assert isinstance(tool, Tool)
