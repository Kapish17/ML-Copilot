"""Parsing a planner's response into one of exactly two decisions.

The parser is the narrowest point in the system: everything a model says
passes through it, and only two shapes get out. Most of these tests are about
things that should *not* get out.
"""

from __future__ import annotations

import json

import pytest

from agent.errors import MalformedPlanError
from agent.plans import (
    ACTION_FINAL,
    ACTION_TOOL,
    MAX_PLAN_CHARS,
    MAX_TOOL_NAME_CHARS,
    parse_plan,
)


def test_a_tool_decision_is_parsed() -> None:
    """The ordinary case."""
    step = parse_plan(
        json.dumps(
            {"action": "tool", "tool": "search_knowledge", "arguments": {"query": "x"}}
        )
    )

    assert step.action == ACTION_TOOL
    assert step.is_tool_call is True
    assert step.tool == "search_knowledge"
    assert step.arguments == {"query": "x"}


def test_a_final_decision_is_parsed() -> None:
    """The other one."""
    step = parse_plan('{"action": "final"}')

    assert step.is_final is True
    assert step.tool is None


def test_a_fenced_object_is_read() -> None:
    """Models wrap JSON in a fence often enough to be worth handling."""
    step = parse_plan('```json\n{"action": "final"}\n```')

    assert step.is_final is True


def test_an_object_embedded_in_prose_is_read() -> None:
    """Formatting noise around a real decision is not a failure."""
    step = parse_plan(
        'Here is my choice:\n{"action": "tool", "tool": "x", "arguments": {}}\nThanks.'
    )

    assert step.tool == "x"


def test_a_tool_name_under_an_alternative_key_is_read() -> None:
    """``tool_name`` and ``input`` are common alternatives."""
    step = parse_plan(
        '{"action": "tool", "tool_name": "search_knowledge", "input": {"query": "x"}}'
    )

    assert step.tool == "search_knowledge"
    assert step.arguments == {"query": "x"}


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "I think we should load the data first.",
        "```python\nprint('hi')\n```",
        "{not json",
        "[1, 2, 3]",
        '"final"',
        "42",
        '{"tool": "search_knowledge"}',
        '{"action": "run"}',
        '{"action": "execute", "code": "print(1)"}',
        '{"action": null}',
        '{"action": "tool"}',
        '{"action": "tool", "tool": ""}',
        '{"action": "tool", "tool": "x", "arguments": "query=x"}',
        '{"action": "tool", "tool": "x", "arguments": [1, 2]}',
    ],
)
def test_anything_that_is_not_a_decision_is_malformed(text: str) -> None:
    """One outcome for every shape of wrong: rejected, never interpreted."""
    with pytest.raises(MalformedPlanError):
        parse_plan(text)


def test_a_fenced_python_block_is_malformed_not_executed() -> None:
    """The fence is stripped, and the inside still has to be a decision."""
    with pytest.raises(MalformedPlanError) as caught:
        parse_plan('```python\nimport os\nos.environ["LLM_API_KEY"]\n```')

    assert caught.value.details["reason"] in {"not_json", "not_an_object"}


def test_an_enormous_response_is_refused_without_being_parsed() -> None:
    """A decision is short; reading further only spends time."""
    with pytest.raises(MalformedPlanError) as caught:
        parse_plan("x" * (MAX_PLAN_CHARS + 1))

    assert caught.value.details["reason"] == "too_long"


def test_an_over_long_tool_name_is_refused() -> None:
    """So a planner cannot put arbitrary text into the run's record."""
    with pytest.raises(MalformedPlanError) as caught:
        parse_plan(
            json.dumps({"action": "tool", "tool": "x" * (MAX_TOOL_NAME_CHARS + 1)})
        )

    assert caught.value.details["reason"] == "tool_name_too_long"


def test_the_action_is_case_insensitive_and_trimmed() -> None:
    """Formatting variance is not a security boundary."""
    assert parse_plan('{"action": " FINAL "}').action == ACTION_FINAL


def test_an_optional_reason_is_kept_but_bounded() -> None:
    """A label, not a reasoning trace."""
    step = parse_plan(
        json.dumps({"action": "final", "reason": "y" * 1_000})
    )

    assert step.reason is not None
    assert len(step.reason) <= 240


def test_a_decision_renders_as_plain_values() -> None:
    """It goes into a record, so it must serialise."""
    step = parse_plan(
        '{"action": "tool", "tool": "x", "arguments": {"a": 1}, "reason": "why"}'
    )

    json.dumps(step.as_dict())
