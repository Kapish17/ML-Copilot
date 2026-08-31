"""What came back from a tool, as a record rather than a return value.

An observation is the only way information enters an agent run. Everything the
planner learns after its first turn, everything the final answer is built
from, and everything a caller sees afterwards passes through this type — which
makes it the right place to enforce the two rules that matter.

**An observation is data, never an instruction.** A retrieved document may say
"ignore your previous instructions"; an experiment description is whatever
someone typed when they ran it. Both arrive here as content. This module does
not act on any of it, and the planner is told explicitly that observations
cannot change its rules. Nothing in a tool's output is ever parsed as a
command, a tool name or a permission.

**An observation is JSON-safe and free of internals.** A tool returns plain
values — no DataFrame, no fitted pipeline, no SHAP explainer, no provider
object. :func:`ensure_json_safe` is the backstop: a value that is not
JSON-legal is replaced by a description of its type rather than serialised, so
a mistake upstream becomes a visible placeholder instead of a leak.

Inputs are recorded too, in summary form: enough to see what was asked,
truncated so a long argument cannot fill the state, and passed through the
same safety check.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

#: Longest recorded value of any single summarised input argument.
MAX_INPUT_VALUE_CHARS = 200
#: Appended to anything this module shortens.
TRUNCATION_MARKER = "…[truncated]"


class ObservationStatus(str, Enum):
    """How one tool call turned out."""

    #: The tool ran and produced a result.
    OK = "ok"
    #: The tool ran and honestly reported that it could not do the work —
    #: an explanation with no fitted model, a search with no index. A result,
    #: not a breakdown, and the planner is expected to carry on.
    UNAVAILABLE = "unavailable"
    #: The call never ran: the tool is not registered, or the arguments did
    #: not validate.
    REJECTED = "rejected"
    #: The tool ran and raised.
    FAILED = "failed"

    @property
    def produced_result(self) -> bool:
        """True when the planner has something it can use."""
        return self is ObservationStatus.OK


def ensure_json_safe(value: Any, *, _depth: int = 0) -> Any:
    """Return ``value`` reduced to plain JSON-legal values.

    Dictionaries and sequences are walked; strings, booleans, integers and
    finite floats pass through; ``None`` passes through. A non-finite float
    becomes ``None``, because ``NaN`` and ``Infinity`` are not JSON. Anything
    else — a DataFrame, an estimator, an explainer, an SDK client — is
    replaced by ``"<Type>"``.

    That replacement is deliberate. Rendering the object's ``repr`` would put
    an address, a file path or a fitted model's parameters into the state; the
    type name says enough to debug with and carries nothing.
    """
    if _depth > 12:
        return "<nested too deeply>"

    # Checked before the primitives: a ``class X(str, Enum)`` member *is* a
    # str, so an isinstance check would pass it through as the member object,
    # whose str() is "X.MEMBER" rather than its value.
    if isinstance(value, Enum):
        return ensure_json_safe(value.value, _depth=_depth + 1)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {
            str(key): ensure_json_safe(item, _depth=_depth + 1)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [ensure_json_safe(item, _depth=_depth + 1) for item in value]

    # A numpy scalar is not a Python number, and rendering one as "<int64>"
    # would lose a real value — a prediction, a score. ``item()`` is the
    # documented way to get the Python equivalent, and asking for it by
    # duck-typing avoids importing numpy into this package.
    unwrap = getattr(value, "item", None)
    if callable(unwrap):
        try:
            unwrapped = unwrap()
        except Exception:  # noqa: BLE001 - anything unexpected falls through
            unwrapped = None
        if isinstance(unwrapped, (str, bool, int, float)):
            return ensure_json_safe(unwrapped, _depth=_depth + 1)

    return f"<{type(value).__name__}>"


def summarise_arguments(arguments: dict[str, Any] | None) -> dict[str, Any]:
    """Reduce validated arguments to a short, safe record of what was asked."""
    summary: dict[str, Any] = {}
    for name, value in (arguments or {}).items():
        safe = ensure_json_safe(value)
        if isinstance(safe, str) and len(safe) > MAX_INPUT_VALUE_CHARS:
            safe = safe[:MAX_INPUT_VALUE_CHARS] + TRUNCATION_MARKER
        summary[name] = safe
    return summary


@dataclass(frozen=True)
class Observation:
    """One executed — or refused — tool call, as it will be remembered."""

    call_id: str
    tool_name: str
    status: ObservationStatus
    #: What was asked, summarised. Never the raw planner text.
    input_summary: dict[str, Any] = field(default_factory=dict)
    #: What came back. Always JSON-safe; empty for a rejected call.
    output: dict[str, Any] = field(default_factory=dict)
    #: Present only when the call did not produce a result. An authored
    #: message: never a stack trace, a vendor message or a path.
    error: str | None = None
    #: Stable code for the failure, when there was one.
    error_code: str | None = None
    duration_ms: float | None = None
    #: Citation identifiers this observation contributed, if any. Only the
    #: knowledge tool produces these, and they are the only identifiers the
    #: final answer may cite.
    citations: tuple[str, ...] = ()

    @property
    def succeeded(self) -> bool:
        """True when the tool produced a usable result."""
        return self.status.produced_result

    def as_dict(self) -> dict[str, Any]:
        """Render the observation as plain JSON-safe values."""
        payload: dict[str, Any] = {
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "status": self.status.value,
            "input_summary": ensure_json_safe(self.input_summary),
            "output": ensure_json_safe(self.output),
        }
        if self.error is not None:
            payload["error"] = self.error
        if self.error_code is not None:
            payload["error_code"] = self.error_code
        if self.duration_ms is not None:
            payload["duration_ms"] = self.duration_ms
        if self.citations:
            payload["citations"] = list(self.citations)
        return payload


__all__ = [
    "MAX_INPUT_VALUE_CHARS",
    "TRUNCATION_MARKER",
    "Observation",
    "ObservationStatus",
    "ensure_json_safe",
    "summarise_arguments",
]
