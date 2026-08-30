"""Schemas describing the knowledge API.

These models are the public contract of ``/search`` and ``/ask``, and they are
also the last line of defence for the requirement that responses be JSON-safe.
Responses are built by validating the structured objects that ``rag/`` and
``llm/`` already produce, so an embedding vector, a numpy array, a provider
object or a fitted estimator could not reach a client even if something
upstream tried to put one there: it would fail validation rather than be
serialised.

What a request may *not* contain is as much a part of the contract as what it
may. There is no field for a system prompt, a provider endpoint, an API key, a
model name, or a switch that turns off grounding or citation validation. The
server is authoritative over safety settings; a caller varies how much
evidence to look at and where to look for it, and nothing else.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.services.knowledge.filters import KNOWN_SOURCE_TYPES

JsonValue = Any


# ---------------------------------------------------------------------------
# Shared request fields
# ---------------------------------------------------------------------------


class KnowledgeFilters(BaseModel):
    """Metadata a request may narrow the evidence by.

    Applied by the retrieval layer **before** ranking, so asking for the five
    best classification experiments searches classification experiments rather
    than ranking everything and discarding the rest.
    """

    model_config = ConfigDict(extra="forbid")

    source_types: list[str] = Field(
        default_factory=list,
        description=(
            "Restrict to these kinds of source. An unknown value is rejected "
            "rather than silently matching nothing."
        ),
        examples=[list(KNOWN_SOURCE_TYPES)],
    )
    task_type: str | None = Field(
        None, description="'classification' or 'regression'.", examples=["classification"]
    )
    dataset_fingerprint: str | None = Field(
        None,
        description=(
            "Content fingerprint of a dataset. Finds every run on the same "
            "data however the file was named."
        ),
        examples=["86494cff7a45cb7f"],
    )
    target_column: str | None = Field(
        None, description="Only evidence about runs predicting this column."
    )
    selected_model: str | None = Field(
        None,
        description="Only evidence about runs whose winner was this model.",
        examples=["random_forest_classifier"],
    )
    primary_metric: str | None = Field(
        None, description="Only evidence about runs judged by this metric.", examples=["f1"]
    )
    experiment_id: str | None = Field(
        None, description="Only evidence from this experiment."
    )

    def as_metadata(self) -> dict[str, Any]:
        """Return the named metadata fields, without the source types."""
        return {
            "task_type": self.task_type,
            "dataset_fingerprint": self.dataset_fingerprint,
            "target_column": self.target_column,
            "selected_model": self.selected_model,
            "primary_metric": self.primary_metric,
            "experiment_id": self.experiment_id,
        }


class _KnowledgeRequest(BaseModel):
    """Fields shared by both knowledge endpoints."""

    model_config = ConfigDict(extra="forbid")

    top_k: int | None = Field(
        None,
        description=(
            "Most passages to consider. Capped by the server's configured "
            "maximum; omit for the default."
        ),
        ge=1,
        examples=[5],
    )
    similarity_threshold: float | None = Field(
        None,
        description=(
            "Minimum cosine similarity a passage must reach. Raise it to "
            "trade recall for precision."
        ),
        ge=-1.0,
        le=1.0,
    )
    filters: KnowledgeFilters = Field(
        default_factory=KnowledgeFilters,
        description="Metadata narrowing, applied before ranking.",
    )


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class SearchRequest(_KnowledgeRequest):
    """A search of the indexed documentation and experiment history."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "query": "How is data leakage prevented?",
                "top_k": 5,
                "filters": {"source_types": ["project_documentation"]},
            }
        },
    )

    query: str = Field(
        ...,
        min_length=1,
        description="What to search for, in the caller's own words.",
        examples=["How is data leakage prevented?"],
    )


class SearchResult(BaseModel):
    """One retrieved passage, with its score and its attribution."""

    rank: int = Field(..., description="Position in the ranking; 1 is best.")
    score: float = Field(..., description="Cosine similarity to the query.")
    content: str = Field(..., description="The passage itself.")
    document_id: str
    chunk_id: str
    source_type: str = Field(
        ..., description="'project_documentation' or 'experiment'."
    )
    source_title: str
    source_reference: str = Field(
        ..., description="A repository-relative path, or an experiment id."
    )
    citation_id: str = Field(
        ...,
        description=(
            "Stable reference to this passage, e.g. "
            "'docs:ml-readme#leakage-prevention'. Resolvable by hand."
        ),
    )
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    """Evidence found for one query."""

    query: str
    results: list[SearchResult] = Field(default_factory=list)
    result_count: int = Field(..., description="Number of passages returned.")
    top_k: int = Field(..., description="The limit actually applied.")
    similarity_threshold: float = Field(
        ..., description="The threshold actually applied."
    )
    similarity_metric: str = Field(
        "cosine", description="How the scores were computed."
    )
    candidate_count: int = Field(
        ...,
        description=(
            "Passages the filter admitted before ranking. Distinguishes 'the "
            "index has nothing about this' from 'the filter excluded "
            "everything'."
        ),
    )
    citations: list[str] = Field(
        default_factory=list, description="The citation of each result, in order."
    )


# ---------------------------------------------------------------------------
# Ask
# ---------------------------------------------------------------------------


class AskRequest(_KnowledgeRequest):
    """A question to answer from retrieved evidence."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "question": "Which model was selected and what did it score?",
                "top_k": 6,
                "filters": {
                    "source_types": ["experiment"],
                    "task_type": "classification",
                },
            }
        },
    )

    question: str = Field(
        ...,
        min_length=1,
        description="What to answer, in the caller's own words.",
        examples=["How does this project prevent data leakage?"],
    )


class AnswerCitation(BaseModel):
    """One source backing part of an answer.

    Only the identifier came from the model; everything else was looked up
    from the evidence actually retrieved, so a citation's title and score are
    trustworthy even when the prose is not.
    """

    citation_id: str
    source_type: str
    source_title: str
    source_reference: str
    relevance_score: float
    excerpt: str = Field(
        "", description="A short opening extract of the cited passage."
    )


class AnswerMetadataModel(BaseModel):
    """How an answer was produced.

    Enough to audit a call and nothing more: no prompt, no raw response, no
    credential, no filesystem path.
    """

    provider: str
    model: str
    retrieved_count: int
    context_count: int = Field(
        ..., description="Passages actually placed in the model's context."
    )
    context_truncated: bool = Field(
        ..., description="True when evidence was shortened or left out."
    )
    context_characters: int
    approximate_context_tokens: int = Field(
        ..., description="A chars/4 heuristic, not a tokeniser."
    )
    below_threshold_count: int
    latency_seconds: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None


class AskResponse(BaseModel):
    """A grounded answer, or an honest account of why there is not one.

    **The status is not decoration.** Only ``grounded`` may be presented to a
    user as an answer; the others say, in order, that there was nothing to
    ground in, that the model produced text that cannot be trusted, that the
    provider failed, or that answering is not configured.
    """

    question: str
    answer: str = Field(
        ...,
        description=(
            "The generated text. For 'grounding_failed' this is what the "
            "model wrote, returned so a human can see what happened — it is "
            "not an answer."
        ),
    )
    status: str = Field(
        ...,
        description=(
            "'grounded' — backed by evidence with valid citations. "
            "'insufficient_evidence' — nothing worth grounding in, or the "
            "model declined. 'grounding_failed' — the text cited a source "
            "that was not retrieved, or cited nothing at all. "
            "'provider_error' / 'configuration_error' — reached only as HTTP "
            "errors."
        ),
        examples=["grounded"],
    )
    is_grounded: bool = Field(
        ..., description="True only when the answer may be presented as one."
    )
    citations: list[AnswerCitation] = Field(default_factory=list)
    citation_ids: list[str] = Field(default_factory=list)
    rejected_citations: list[str] = Field(
        default_factory=list,
        description=(
            "Identifiers the model produced that were not in the retrieved "
            "evidence. Reported rather than quietly removed — a fabricated "
            "source is the most important thing to know about an answer."
        ),
    )
    allowed_citations: list[str] = Field(
        default_factory=list,
        description="Exactly what the model was permitted to cite, for audit.",
    )
    warnings: list[str] = Field(default_factory=list)
    error_code: str | None = None
    metadata: AnswerMetadataModel


class KnowledgeStatusResponse(BaseModel):
    """What the knowledge endpoints can currently do.

    Reports whether a credential is configured, never what it is, and names no
    filesystem location.
    """

    search_available: bool
    answering_available: bool = Field(
        ..., description="False when no language-model credential is configured."
    )
    index_built: bool = Field(
        ..., description="False when nothing has been indexed yet."
    )
    similarity_metric: str
    default_top_k: int
    max_top_k: int
    max_query_length: int
    source_types: list[str] = Field(
        default_factory=lambda: list(KNOWN_SOURCE_TYPES),
        description="Source types a filter may name.",
    )


__all__ = [
    "AnswerCitation",
    "AnswerMetadataModel",
    "AskRequest",
    "AskResponse",
    "KnowledgeFilters",
    "KnowledgeStatusResponse",
    "SearchRequest",
    "SearchResponse",
    "SearchResult",
]
