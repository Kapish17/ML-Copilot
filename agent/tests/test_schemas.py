"""Argument validation — the boundary between model output and running code.

Every test here is a way a tool call could be wrong, and the assertion is
always the same in spirit: it is rejected before anything runs, and the
rejection says which field and why.
"""

from __future__ import annotations

import pytest

from agent.errors import ToolValidationError
from agent.schemas import (
    ABSOLUTE_MAX_LIST_ITEMS,
    ABSOLUTE_MAX_STRING_LENGTH,
    BOOLEAN,
    INTEGER,
    NUMBER,
    STRING,
    STRING_LIST,
    ArgumentField,
    ArgumentSchema,
)


def schema(*fields: ArgumentField) -> ArgumentSchema:
    """Build a schema from fields, for brevity."""
    return ArgumentSchema(fields=fields)


def test_a_valid_call_is_normalised() -> None:
    """Strings are stripped and defaults filled in."""
    validated = schema(
        ArgumentField("query", STRING, "What to search for.", required=True),
        ArgumentField("top_k", INTEGER, "How many.", default=5, minimum=1, maximum=10),
    ).validate({"query": "  leakage  "})

    assert validated == {"query": "leakage", "top_k": 5}


def test_an_undeclared_field_is_rejected_not_ignored() -> None:
    """The difference between "not supported" and "silently not supported"."""
    with pytest.raises(ToolValidationError) as caught:
        schema(ArgumentField("query", STRING, "Query.", required=True)).validate(
            {"query": "x", "api_key": "sk-secret"}
        )

    assert caught.value.details["unknown_arguments"] == ["api_key"]


@pytest.mark.parametrize(
    "field_name",
    [
        "api_key",
        "base_url",
        "system_prompt",
        "model",
        "temperature",
        "code",
        "command",
        "path",
        "url",
        "skip_validation",
    ],
)
def test_no_smuggled_setting_survives_validation(field_name: str) -> None:
    """None of these is a declared field, so none of them takes effect."""
    with pytest.raises(ToolValidationError):
        schema(ArgumentField("query", STRING, "Query.", required=True)).validate(
            {"query": "x", field_name: "anything"}
        )


def test_a_missing_required_field_is_rejected() -> None:
    """A tool never runs with a required argument absent."""
    with pytest.raises(ToolValidationError) as caught:
        schema(ArgumentField("query", STRING, "Query.", required=True)).validate({})

    assert caught.value.details["field"] == "query"


def test_a_wrongly_typed_field_is_rejected() -> None:
    """A number where a string was declared."""
    with pytest.raises(ToolValidationError):
        schema(ArgumentField("query", STRING, "Query.", required=True)).validate(
            {"query": 42}
        )


def test_a_boolean_is_not_accepted_as_a_number() -> None:
    """``bool`` subclasses ``int``, so an unguarded check would let it past."""
    with pytest.raises(ToolValidationError):
        schema(ArgumentField("top_k", INTEGER, "How many.")).validate({"top_k": True})


def test_numeric_bounds_are_enforced() -> None:
    """Above the maximum and below the minimum both fail."""
    spec = schema(ArgumentField("top_k", INTEGER, "How many.", minimum=1, maximum=10))

    with pytest.raises(ToolValidationError):
        spec.validate({"top_k": 0})
    with pytest.raises(ToolValidationError) as caught:
        spec.validate({"top_k": 999_999})

    assert caught.value.details["maximum"] == 10


def test_a_float_is_accepted_where_a_number_is_declared() -> None:
    """And normalised to a float."""
    validated = schema(
        ArgumentField("threshold", NUMBER, "Minimum score.", minimum=0.0, maximum=1.0)
    ).validate({"threshold": 1})

    assert validated == {"threshold": 1.0}


def test_a_string_longer_than_the_limit_is_rejected() -> None:
    """Query length is bounded, as it is for the HTTP callers."""
    spec = schema(ArgumentField("query", STRING, "Query.", required=True, max_length=100))

    with pytest.raises(ToolValidationError) as caught:
        spec.validate({"query": "x" * 500})

    assert caught.value.details["max_length"] == 100


def test_no_string_can_exceed_the_absolute_limit() -> None:
    """Even a field that declares no maximum of its own is bounded."""
    spec = schema(ArgumentField("note", STRING, "A note."))

    with pytest.raises(ToolValidationError):
        spec.validate({"note": "x" * (ABSOLUTE_MAX_STRING_LENGTH + 1)})


def test_an_empty_required_string_is_rejected() -> None:
    """Whitespace is not a query."""
    with pytest.raises(ToolValidationError):
        schema(ArgumentField("query", STRING, "Query.", required=True)).validate(
            {"query": "   "}
        )


def test_choices_restrict_a_string_to_a_declared_set() -> None:
    """This is how a model name is held to the existing registry."""
    spec = schema(
        ArgumentField("scope", STRING, "Scope.", choices=("global", "prediction"))
    )

    assert spec.validate({"scope": "global"}) == {"scope": "global"}
    with pytest.raises(ToolValidationError) as caught:
        spec.validate({"scope": "everything"})

    assert caught.value.details["allowed_values"] == ["global", "prediction"]


def test_choices_can_be_resolved_at_call_time() -> None:
    """So the allowed values follow the live registry, not a stale copy."""
    available = ["alpha"]
    spec = schema(
        ArgumentField(
            "model", STRING, "Model.", choices_provider=lambda: list(available)
        )
    )

    with pytest.raises(ToolValidationError):
        spec.validate({"model": "beta"})

    available.append("beta")
    assert spec.validate({"model": "beta"}) == {"model": "beta"}


def test_a_list_field_validates_every_entry() -> None:
    """Types, emptiness and membership are all checked per entry."""
    spec = schema(
        ArgumentField(
            "models", STRING_LIST, "Models.", max_items=3, choices=("a", "b")
        )
    )

    assert spec.validate({"models": [" a ", "b"]}) == {"models": ["a", "b"]}

    for bad in ([1, 2], ["a", ""], ["a", "c"], ["a", "b", "a", "b"], "a"):
        with pytest.raises(ToolValidationError):
            spec.validate({"models": bad})


def test_a_list_cannot_exceed_the_absolute_item_limit() -> None:
    """Even without a declared maximum."""
    spec = schema(ArgumentField("tags", STRING_LIST, "Tags."))

    with pytest.raises(ToolValidationError):
        spec.validate({"tags": ["x"] * (ABSOLUTE_MAX_LIST_ITEMS + 1)})


def test_arguments_must_be_an_object() -> None:
    """A list, a string or a number is not a set of named arguments."""
    spec = schema(ArgumentField("query", STRING, "Query."))

    for bad in ([], "query=x", 7):
        with pytest.raises(ToolValidationError):
            spec.validate(bad)  # type: ignore[arg-type]


def test_no_arguments_at_all_is_fine_when_nothing_is_required() -> None:
    """A tool with only optional fields can be called bare."""
    assert schema(ArgumentField("note", STRING, "A note.")).validate(None) == {}


def test_an_explicit_null_falls_back_to_the_default() -> None:
    """Models write ``null`` for "not specified" more often than they omit."""
    validated = schema(
        ArgumentField("top_k", INTEGER, "How many.", default=5)
    ).validate({"top_k": None})

    assert validated == {"top_k": 5}


def test_a_boolean_field_accepts_only_booleans() -> None:
    """And not the strings a model might write instead."""
    spec = schema(ArgumentField("explain", BOOLEAN, "Explain?"))

    # An explicitly supplied False is kept. Only an absent or null value falls
    # back to the default, so "the caller said no" survives.
    assert spec.validate({"explain": False}) == {"explain": False}
    assert spec.validate({"explain": True}) == {"explain": True}
    with pytest.raises(ToolValidationError):
        spec.validate({"explain": "yes"})


def test_the_declaration_renders_without_live_objects() -> None:
    """What the planner sees is data: no callables, no closures."""
    import json

    rendered = schema(
        ArgumentField(
            "model", STRING, "Model.", choices_provider=lambda: ["a", "b"]
        )
    ).as_dict()

    assert json.loads(json.dumps(rendered))["fields"][0]["allowed_values"] == ["a", "b"]
