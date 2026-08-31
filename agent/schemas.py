"""Typed argument schemas, and the validation a tool call must survive.

A tool call arrives as text a language model wrote. Between that text and any
code that does real work stands this module, and nothing else. It is
deliberately small and deliberately boring: a field has a declared type, a
declared requirement, and — where a value could be abused — declared bounds or
a declared set of allowed values. Anything not declared is rejected.

Three properties matter more than the mechanics:

**Rejection is the default.** An unknown field is an error, not something
ignored. A model that submits ``{"query": "...", "api_key": "..."}`` is told
the call is invalid; the key is not quietly dropped and the call is not
quietly run without it. Silently ignoring a field is how a caller comes to
believe a setting took effect.

**Values are checked, not just types.** ``top_k`` is bounded, a query has a
maximum length, a model name must be one the existing registry already knows.
A well-typed string is not automatically a safe one.

**Nothing here interprets text as code.** No ``eval``, no ``exec``, no import
by name, no dotted-path lookup, no format string built from model output. The
only thing this module does with a model's words is compare them against a
declaration.

Pydantic is deliberately not used. This package must stay free of the web
layer's dependencies, and the validation needed here — a few dozen fields with
plain types and bounds — is small enough that owning it keeps the agent's
trust boundary readable in one file.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from agent.errors import ToolValidationError

#: Every type a tool argument may have. Anything outside this set — a nested
#: object, a callable, an arbitrary class — cannot be declared, which means a
#: planner has no way to submit one.
FieldType = str

STRING: FieldType = "string"
INTEGER: FieldType = "integer"
NUMBER: FieldType = "number"
BOOLEAN: FieldType = "boolean"
STRING_LIST: FieldType = "string_list"

_SCALAR_TYPES: dict[str, tuple[type, ...]] = {
    STRING: (str,),
    INTEGER: (int,),
    NUMBER: (int, float),
    BOOLEAN: (bool,),
}

#: A generous ceiling on any single string argument, applied even when a field
#: declares no maximum of its own. It exists so that no path through this
#: module can accept an unbounded amount of model-written text.
ABSOLUTE_MAX_STRING_LENGTH = 8_000
#: A ceiling on how many entries a list argument may hold.
ABSOLUTE_MAX_LIST_ITEMS = 50


@dataclass(frozen=True)
class ArgumentField:
    """One declared argument of one tool."""

    name: str
    type: FieldType
    description: str
    required: bool = False
    default: Any = None
    #: Numeric bounds, inclusive.
    minimum: float | None = None
    maximum: float | None = None
    #: String and list length bounds.
    max_length: int | None = None
    max_items: int | None = None
    #: The complete set of acceptable values. When set, anything else is
    #: rejected — this is how a model name is restricted to the existing
    #: registry rather than trusted because it looks plausible.
    choices: tuple[str, ...] | None = None
    #: Resolved when the acceptable values are only knowable at call time,
    #: e.g. the columns of the dataset actually loaded. Takes precedence over
    #: ``choices``.
    choices_provider: Callable[[], Sequence[str]] | None = None

    def allowed_values(self) -> tuple[str, ...] | None:
        """The acceptable values right now, or ``None`` when unrestricted."""
        if self.choices_provider is not None:
            return tuple(self.choices_provider())
        return self.choices

    def as_dict(self) -> dict[str, Any]:
        """Render the declaration as the planner will be shown it.

        This is what a planner sees, so it carries the description and the
        constraints but never a callable or any other live object.
        """
        payload: dict[str, Any] = {
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "required": self.required,
        }
        if self.default is not None:
            payload["default"] = self.default
        for key, value in (
            ("minimum", self.minimum),
            ("maximum", self.maximum),
            ("max_length", self.max_length),
            ("max_items", self.max_items),
        ):
            if value is not None:
                payload[key] = value
        allowed = self.allowed_values()
        if allowed is not None:
            payload["allowed_values"] = list(allowed)
        return payload


def _fail(message: str, **details: Any) -> ToolValidationError:
    """Build a validation failure with structured, already-safe details."""
    return ToolValidationError(message, details=details)


def _validate_scalar(spec: ArgumentField, value: Any) -> Any:
    """Check and normalise one scalar argument."""
    expected = _SCALAR_TYPES[spec.type]

    # bool is a subclass of int in Python, so an unguarded isinstance check
    # would accept True where a count was asked for.
    if spec.type in {INTEGER, NUMBER} and isinstance(value, bool):
        raise _fail(
            f"'{spec.name}' must be a {spec.type}, not a boolean.", field=spec.name
        )
    if not isinstance(value, expected):
        raise _fail(
            f"'{spec.name}' must be a {spec.type}.",
            field=spec.name,
            expected_type=spec.type,
        )

    if spec.type == STRING:
        text = value.strip()
        if spec.required and not text:
            raise _fail(f"'{spec.name}' must not be empty.", field=spec.name)
        limit = min(spec.max_length or ABSOLUTE_MAX_STRING_LENGTH, ABSOLUTE_MAX_STRING_LENGTH)
        if len(text) > limit:
            raise _fail(
                f"'{spec.name}' may be at most {limit} characters, got {len(text)}.",
                field=spec.name,
                length=len(text),
                max_length=limit,
            )
        allowed = spec.allowed_values()
        if allowed is not None and text not in allowed:
            raise _fail(
                f"'{spec.name}' must be one of: {', '.join(allowed) or '(none available)'}.",
                field=spec.name,
                allowed_values=list(allowed),
            )
        return text

    if spec.type == NUMBER:
        value = float(value)
    if spec.minimum is not None and value < spec.minimum:
        raise _fail(
            f"'{spec.name}' must be at least {spec.minimum}.",
            field=spec.name,
            minimum=spec.minimum,
        )
    if spec.maximum is not None and value > spec.maximum:
        raise _fail(
            f"'{spec.name}' must be at most {spec.maximum}.",
            field=spec.name,
            maximum=spec.maximum,
        )
    return value


def _validate_list(spec: ArgumentField, value: Any) -> list[str]:
    """Check and normalise one list-of-strings argument."""
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise _fail(f"'{spec.name}' must be a list of strings.", field=spec.name)

    limit = min(spec.max_items or ABSOLUTE_MAX_LIST_ITEMS, ABSOLUTE_MAX_LIST_ITEMS)
    if len(value) > limit:
        raise _fail(
            f"'{spec.name}' may hold at most {limit} entries, got {len(value)}.",
            field=spec.name,
            max_items=limit,
        )

    allowed = spec.allowed_values()
    items: list[str] = []
    for entry in value:
        if not isinstance(entry, str):
            raise _fail(f"Every entry of '{spec.name}' must be a string.", field=spec.name)
        text = entry.strip()
        if not text:
            raise _fail(f"'{spec.name}' must not contain empty entries.", field=spec.name)
        if len(text) > ABSOLUTE_MAX_STRING_LENGTH:
            raise _fail(
                f"An entry of '{spec.name}' is too long.", field=spec.name
            )
        if allowed is not None and text not in allowed:
            raise _fail(
                f"'{spec.name}' may only contain: "
                f"{', '.join(allowed) or '(none available)'}.",
                field=spec.name,
                allowed_values=list(allowed),
            )
        items.append(text)
    return items


@dataclass(frozen=True)
class ArgumentSchema:
    """The complete declared argument surface of one tool."""

    fields: tuple[ArgumentField, ...] = field(default=())

    def field_names(self) -> tuple[str, ...]:
        """Every declared field name."""
        return tuple(spec.name for spec in self.fields)

    def as_dict(self) -> dict[str, Any]:
        """Render the schema as the planner will be shown it."""
        return {"fields": [spec.as_dict() for spec in self.fields]}

    def validate(self, arguments: Mapping[str, Any] | None) -> dict[str, Any]:
        """Validate a set of arguments, or refuse them.

        Args:
            arguments: What the planner submitted. ``None`` is treated as an
                empty mapping, so a tool whose fields are all optional can be
                called with no arguments at all.

        Returns:
            dict[str, Any]: The validated, normalised values — strings
            stripped, numbers coerced, defaults filled in. This is the only
            object a tool ever sees.

        Raises:
            ToolValidationError: If the arguments are not a mapping, carry a
                field that is not declared, omit a required field, or hold a
                value of the wrong type, the wrong size or outside the
                allowed set.
        """
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, Mapping):
            raise _fail("Tool arguments must be an object of named values.")

        declared = {spec.name: spec for spec in self.fields}

        # Rejected, never ignored. A planner that submits `api_key` is told
        # the call is invalid rather than having the field silently removed.
        unknown = sorted(str(key) for key in arguments if key not in declared)
        if unknown:
            raise _fail(
                "Unknown argument(s): " + ", ".join(unknown) + ". "
                "Only the declared arguments of this tool are accepted.",
                unknown_arguments=unknown,
                allowed_arguments=list(declared),
            )

        validated: dict[str, Any] = {}
        for name, spec in declared.items():
            if name not in arguments or arguments[name] is None:
                if spec.required:
                    raise _fail(f"'{name}' is required.", field=name)
                if spec.default is not None:
                    validated[name] = spec.default
                continue

            value = arguments[name]
            if spec.type == STRING_LIST:
                validated[name] = _validate_list(spec, value)
            else:
                validated[name] = _validate_scalar(spec, value)

        return validated


__all__ = [
    "ABSOLUTE_MAX_LIST_ITEMS",
    "ABSOLUTE_MAX_STRING_LENGTH",
    "BOOLEAN",
    "INTEGER",
    "NUMBER",
    "STRING",
    "STRING_LIST",
    "ArgumentField",
    "ArgumentSchema",
]
