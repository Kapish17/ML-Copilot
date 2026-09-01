"""Schemas describing the agent API.

These models are the public contract of ``POST /api/v1/agent/ask``, and they
are also the last line of defence for the requirement that responses be
JSON-safe. The response is built by validating the structured
:class:`~agent.results.AgentResult` the orchestrator already produces, so a
DataFrame, a fitted pipeline, a SHAP explainer or a provider object could not
reach a client even if something upstream tried to put one there: it would
fail validation rather than be serialised.

What a request may *not* contain is as much a part of the contract as what it
may. There is no field for a system prompt, a provider endpoint, an API key, a
model name, a tool list, a registry, an estimator, a filesystem path or a
switch that turns off grounding or citation validation. The server is
authoritative over every one of those, and ``extra="forbid"`` means an attempt
to supply one is a 422 rather than a silently ignored field.

What a request *may* vary is how small the run is. Three budgets, each of
which may only be lowered — see :mod:`app.services.agent.budgets`.

Two things are deliberately absent from the response as well. There is no
``chain_of_thought`` field and no prompt: what comes back is which tool was
chosen, the validated arguments, what the tool returned, and the answer. How
the planner decided is not returned, not stored and not logged.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

JsonValue = Any

#: A question long enough to be a real analysis request and short enough that
#: a pasted document is refused. Enforced by the schema so an over-long body
#: never reaches the planner.
MAX_QUESTION_LENGTH = 2_000


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class AgentAskRequest(BaseModel):
    """A question for the agent, and optionally a smaller budget to answer it in."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "question": (
                    "Analyse the customers dataset, tell me which model "
                    "performed best, and explain why."
                ),
                "max_tool_calls": 4,
            }
        },
    )

    question: str = Field(
        ...,
        min_length=1,
        max_length=MAX_QUESTION_LENGTH,
        description="What to answer, in the caller's own words.",
        examples=["Which model performs best on the customers data, and why?"],
    )
    max_tool_calls: int | None = Field(
        None,
        ge=1,
        description=(
            "Most tool calls this question may cause. May only lower the "
            "server's limit; a larger value is rejected."
        ),
        examples=[4],
    )
    max_iterations: int | None = Field(
        None,
        ge=1,
        description=(
            "Most planning steps this question may take. May only lower the "
            "server's limit."
        ),
    )
    max_context_chars: int | None = Field(
        None,
        ge=1,
        description=(
            "Most observed material this run may accumulate, in characters. "
            "May only lower the server's limit."
        ),
    )

    @field_validator("question")
    @classmethod
    def _reject_a_blank_question(cls, value: str) -> str:
        """Strip the question, and refuse one that is only whitespace.

        ``min_length`` counts characters, so ``"   "`` satisfies it. A run
        started on that would spend a planning call to be told there was
        nothing to answer.
        """
        text = value.strip()
        if not text:
            raise ValueError("A question is required.")
        return text

    def budgets(self) -> dict[str, Any]:
        """The requested limits, without the fields that were not supplied."""
        requested = {
            "max_tool_calls": self.max_tool_calls,
            "max_iterations": self.max_iterations,
            "max_context_chars": self.max_context_chars,
        }
        return {name: value for name, value in requested.items() if value is not None}


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------


class AgentToolCall(BaseModel):
    """One tool the agent chose to run, and how it went.

    The arguments shown are the *validated* ones — what the tool actually
    received after the schema approved it, not the raw text the planner wrote.
    A call that was rejected records the argument names only, because its
    values never passed validation.
    """

    call_id: str = Field(
        ..., description="Internal reference for this call, e.g. 'call-01'."
    )
    tool_name: str
    status: str = Field(
        ...,
        description=(
            "'ok' — the tool produced a result. 'unavailable' — it ran and "
            "honestly reported it could not do the work. 'rejected' — the "
            "call never ran: the tool is not registered, or the arguments did "
            "not validate. 'failed' — the tool raised."
        ),
        examples=["ok"],
    )
    arguments: dict[str, JsonValue] = Field(default_factory=dict)
    duration_ms: float | None = None


class AgentObservation(BaseModel):
    """What one tool call returned.

    Untrusted content: a retrieved passage was written by whoever could add a
    document, and an experiment's name is whatever someone typed. It is data
    for the client to read, exactly as it was data for the planner.
    """

    call_id: str
    tool_name: str
    status: str
    input_summary: dict[str, JsonValue] = Field(default_factory=dict)
    output: dict[str, JsonValue] = Field(default_factory=dict)
    error: str | None = Field(
        None,
        description=(
            "Present only when the call produced no result. An authored "
            "message — never a stack trace, a provider's words or a path."
        ),
    )
    error_code: str | None = None
    duration_ms: float | None = None
    citations: list[str] = Field(default_factory=list)


class AgentCitationModel(BaseModel):
    """One source backing part of an answer.

    Only the identifier came from the model; the title, the reference and the
    score were read from the passage that was actually retrieved, so they are
    trustworthy even when the prose is not.
    """

    citation_id: str
    source_type: str = ""
    source_title: str = ""
    source_reference: str = Field(
        "", description="A repository-relative path, or an experiment id."
    )
    score: float | None = None


class AgentDatasetInfo(BaseModel):
    """The uploaded dataset, as far as a response describes it.

    Facts about the data, and none of the data. **Uploaded datasets are
    processed in memory for the request and are never persisted as raw data by
    the agent** — so there is no identifier to fetch one back with, no path,
    and no rows. What is here is the shape, the column names, the display
    filename and the content fingerprint that identifies it in any experiment
    it produced.
    """

    name: str = Field(
        ...,
        description=(
            "The name the agent addressed the dataset by. A constant: a "
            "client's filename never becomes an identifier."
        ),
        examples=["uploaded_dataset"],
    )
    filename: str = Field(
        ...,
        description=(
            "The submitted filename, reduced to a bare name. Display metadata "
            "only — nothing resolves it to a location."
        ),
        examples=["customers.csv"],
    )
    source_format: str = Field(
        ...,
        description=(
            "How the upload was read: 'csv', 'xlsx' or 'json'. The agent "
            "itself is not told — every format becomes the same standardised "
            "table before the run starts, so the answer does not depend on "
            "which one it was."
        ),
        examples=["csv"],
    )
    fingerprint: str = Field(
        ...,
        description=(
            "Content fingerprint of the standardised data. The canonical "
            "identity, and what any experiment from this dataset is filed "
            "under. Computed from the table, not the file, so the same data "
            "uploaded as CSV, Excel or JSON fingerprints identically."
        ),
        examples=["86494cff7a45cb7f"],
    )
    row_count: int
    column_count: int
    columns: list[str]
    persisted: bool = Field(
        False,
        description=(
            "Always false. The dataset lived in memory for this request only."
        ),
    )


class AgentAskResponse(BaseModel):
    """One agent run, as the caller receives it.

    **The status is not decoration.** Only ``completed`` may be presented to a
    user as an answer; the others say, in order, that something was missing,
    that nothing supported an answer, or that the text cannot be trusted.
    """

    question: str
    status: str = Field(
        ...,
        description=(
            "'completed' — supported by the observations, every citation "
            "real. 'partial' — real work done, but something is missing: a "
            "tool was unavailable, or a budget ran out. "
            "'insufficient_evidence' — nothing observed supports an answer. "
            "'grounding_failed' — the answer cited a source that was never "
            "retrieved, or cited nothing while evidence existed. 'failed' is "
            "reached only as an HTTP error."
        ),
        examples=["completed"],
    )
    final_answer: str = Field(
        ...,
        description=(
            "The generated text. For 'grounding_failed' this is what the "
            "model wrote, returned so a human can see what happened — it is "
            "not an answer."
        ),
    )
    is_answer: bool = Field(
        ..., description="True only when the answer may be presented as one."
    )
    tool_calls: list[AgentToolCall] = Field(default_factory=list)
    observations: list[AgentObservation] = Field(default_factory=list)
    citations: list[AgentCitationModel] = Field(default_factory=list)
    citation_ids: list[str] = Field(default_factory=list)
    rejected_citations: list[str] = Field(
        default_factory=list,
        description=(
            "Identifiers the model produced that were never retrieved, and "
            "experiment ids no tool returned. Reported rather than quietly "
            "removed — a fabricated source is the most important thing to "
            "know about an answer, and it is never repaired."
        ),
    )
    allowed_citations: list[str] = Field(
        default_factory=list,
        description="Exactly what the model was permitted to cite, for audit.",
    )
    experiment_ids: list[str] = Field(
        default_factory=list,
        description="Experiments this run created or read, in order.",
    )
    warnings: list[str] = Field(default_factory=list)
    iterations: int = Field(..., description="Planning steps taken.")
    tool_call_count: int = Field(..., description="Tool calls made, including failures.")
    tools_available: list[str] = Field(
        default_factory=list,
        description=(
            "The tools the agent could choose from for this run. A tool is "
            "registered only when the service it wraps is available, so this "
            "is how a client sees that, for example, no dataset was supplied."
        ),
    )
    dataset: AgentDatasetInfo | None = Field(
        None,
        description=(
            "The dataset this request supplied, when it supplied one. Facts "
            "about it only — never rows, and never a location."
        ),
    )
    error_code: str | None = Field(
        None, description="Stable code when a budget stopped the run."
    )
    duration_ms: float | None = None


class AgentStatusResponse(BaseModel):
    """What the agent endpoint can currently do.

    Reports whether a credential is configured, never what it is, and names no
    filesystem location.
    """

    agent_available: bool = Field(
        ..., description="False when no language-model credential is configured."
    )
    tools: list[str] = Field(
        default_factory=list, description="The tools currently registered."
    )
    dataset_upload_supported: bool = Field(
        True,
        description=(
            "Whether POST /api/v1/agent/ask-with-dataset accepts a dataset. "
            "When it does, a request that uploads one gets the two "
            "dataset-dependent tools in addition to those listed above."
        ),
    )
    supported_dataset_formats: list[str] = Field(
        default_factory=list,
        description=(
            "The upload formats that endpoint accepts. The agent's reasoning "
            "does not depend on which one a caller uses — every format "
            "becomes the same standardised table before the run begins."
        ),
        examples=[["csv", "xlsx", "json"]],
    )
    max_tool_calls: int
    max_iterations: int
    max_context_chars: int
    max_answer_length: int


__all__ = [
    "MAX_QUESTION_LENGTH",
    "AgentAskRequest",
    "AgentAskResponse",
    "AgentCitationModel",
    "AgentDatasetInfo",
    "AgentObservation",
    "AgentStatusResponse",
    "AgentToolCall",
]
