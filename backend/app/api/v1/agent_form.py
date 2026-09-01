"""The multipart contract of a dataset-bearing agent request.

A dataset is a file, and a file cannot travel inside a JSON body, so the
dataset-aware endpoint takes ``multipart/form-data``: the dataset as an upload
and the question and budgets as form fields. FastAPI cannot flatten a Pydantic
model into a form when the same request also carries a file — Commit 8 proved
that with a test — so the fields are declared as the dependency below: one
place, fully described, and each appears in the generated OpenAPI schema.

The dependency's only job is to turn form fields into a plain, FastAPI-free
request object the service layer works with. It validates what can be checked
from the value alone — a question that is blank or over-long, a budget that is
not a positive integer, a field the endpoint does not define. What depends on
the server's configured limits is the budget policy's
(:mod:`app.services.agent.budgets`), and what depends on the file is the
dataset service's, which already owns extension, size and parsing. No limit is
redefined here.

**Undeclared fields are rejected, not ignored.** A JSON body gets that from
``extra="forbid"``; a form does not, because a multipart body has no schema of
its own — so it is done explicitly. The two endpoints must agree, or a caller
would find that a field refused on one is quietly accepted on the other.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any

from fastapi import Depends, Form, Request
from fastapi.exceptions import RequestValidationError

from app.schemas.agent import MAX_QUESTION_LENGTH

#: Every field this endpoint defines. Anything else in the body is refused.
ALLOWED_FORM_FIELDS: frozenset[str] = frozenset(
    {"file", "question", "max_tool_calls", "max_iterations", "max_context_chars"}
)


@dataclass(frozen=True)
class AgentAskForm:
    """The non-file half of a dataset-bearing agent request."""

    question: str
    max_tool_calls: int | None = None
    max_iterations: int | None = None
    max_context_chars: int | None = None

    def budgets(self) -> dict[str, Any]:
        """The requested limits, without the fields that were not supplied."""
        requested = {
            "max_tool_calls": self.max_tool_calls,
            "max_iterations": self.max_iterations,
            "max_context_chars": self.max_context_chars,
        }
        return {name: value for name, value in requested.items() if value is not None}


def _invalid(field: str, message: str) -> RequestValidationError:
    """Build the validation failure the shared handler renders as a 422.

    Raised rather than returned so the multipart endpoint answers in exactly
    the shape the JSON one does. The offending *value* is deliberately not
    included: a refused field is refused, and quoting what was in it only puts
    it back on the wire.
    """
    return RequestValidationError(
        [{"type": "value_error", "loc": ("body", field), "msg": message}]
    )


async def agent_ask_form(
    request: Request,
    question: Annotated[
        str,
        Form(
            description="What to answer, in the caller's own words.",
            min_length=1,
            max_length=MAX_QUESTION_LENGTH,
            examples=[
                "Analyse this dataset, find the best model, and explain why."
            ],
        ),
    ],
    max_tool_calls: Annotated[
        int | None,
        Form(
            description=(
                "Most tool calls this question may cause. May only lower the "
                "server's limit; a larger value is rejected."
            ),
            ge=1,
            examples=[4],
        ),
    ] = None,
    max_iterations: Annotated[
        int | None,
        Form(
            description=(
                "Most planning steps this question may take. May only lower "
                "the server's limit."
            ),
            ge=1,
        ),
    ] = None,
    max_context_chars: Annotated[
        int | None,
        Form(
            description=(
                "Most observed material this run may accumulate, in "
                "characters. May only lower the server's limit."
            ),
            ge=1,
        ),
    ] = None,
) -> AgentAskForm:
    """Assemble the question and budgets from the submitted form.

    Raises:
        RequestValidationError: If the question is blank once stripped, or the
            body carries a field this endpoint does not define. The first
            matters because ``min_length`` counts characters, so ``"   "``
            satisfies it and a run started on that would spend a planning call
            to be told there was nothing to answer.
    """
    # The form is already parsed and cached by the time this runs, so reading
    # it again costs nothing.
    submitted = set((await request.form()).keys())
    unknown = sorted(submitted - ALLOWED_FORM_FIELDS)
    if unknown:
        raise _invalid(
            unknown[0],
            "Unknown field(s): "
            + ", ".join(unknown)
            + ". Only the declared fields of this endpoint are accepted.",
        )

    text = question.strip()
    if not text:
        raise _invalid("question", "A question is required.")

    return AgentAskForm(
        question=text,
        max_tool_calls=max_tool_calls,
        max_iterations=max_iterations,
        max_context_chars=max_context_chars,
    )


AgentAskFormDep = Annotated[AgentAskForm, Depends(agent_ask_form)]


__all__ = [
    "ALLOWED_FORM_FIELDS",
    "AgentAskForm",
    "AgentAskFormDep",
    "agent_ask_form",
]
