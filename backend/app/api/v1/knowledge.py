"""Knowledge endpoints: search the project's memory, and answer from it.

Each handler does the same three things and nothing else: take the request,
hand it to the knowledge service, validate the structured result against a
response schema. The retrieval lives in ``rag/``, the generation and grounding
in ``llm/``, and the orchestration in
:mod:`app.services.knowledge.service` — no route here embeds a passage, builds
a prompt, filters a list or checks a citation.

Failures propagate. Retrieval errors, language-model errors and the two
API-level refusals are all turned into the one documented envelope by the
application's exception handlers, so no handler builds an error response by
hand.

One thing worth saying plainly, because it decides the status codes: an answer
that could not be grounded is a **result**, not a failure. The request was
valid, the work was done, and the honest outcome is that the answer is not
trustworthy — so it is a 200 carrying a status, not a 5xx telling a client to
retry something that will fail identically.
"""

from __future__ import annotations

from fastapi import APIRouter, status

from app.api.dependencies import KnowledgeServiceDep
from app.api.security import UNAUTHORIZED_RESPONSE, Protected
from app.schemas.errors import ErrorResponse
from app.schemas.knowledge import (
    AskRequest,
    AskResponse,
    KnowledgeStatusResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
)

router = APIRouter(tags=["knowledge"])

_SEARCH_ERRORS: dict[int | str, dict[str, object]] = {
    # Every route below is protected, so each can be refused before it
    # runs. Documented here rather than left for a reader to discover.
    **UNAUTHORIZED_RESPONSE,
    status.HTTP_400_BAD_REQUEST: {
        "model": ErrorResponse,
        "description": (
            "The query is blank or too long, a limit is out of range, or a "
            "filter names a source type that does not exist."
        ),
    },
    status.HTTP_422_UNPROCESSABLE_CONTENT: {
        "model": ErrorResponse,
        "description": "The request body does not match the schema.",
    },
    status.HTTP_503_SERVICE_UNAVAILABLE: {
        "model": ErrorResponse,
        "description": (
            "The retrieval index has not been built, or exists but cannot be "
            "read. Distinct from an empty result, which is a 200."
        ),
    },
}

_ASK_ERRORS: dict[int | str, dict[str, object]] = {
    # Every route below is protected, so each can be refused before it
    # runs. Documented here rather than left for a reader to discover.
    **UNAUTHORIZED_RESPONSE,
    **_SEARCH_ERRORS,
    status.HTTP_502_BAD_GATEWAY: {
        "model": ErrorResponse,
        "description": (
            "The language-model provider failed: timeout, rate limit, outage "
            "or an unusable response."
        ),
    },
    status.HTTP_503_SERVICE_UNAVAILABLE: {
        "model": ErrorResponse,
        "description": (
            "Answer generation is not configured (no API key), or the "
            "retrieval index is unavailable."
        ),
    },
}


@router.post(
    "/search",
    dependencies=[Protected],
    response_model=SearchResponse,
    responses=_SEARCH_ERRORS,
    summary="Search project documentation and experiment history",
)
def search_knowledge(
    knowledge: KnowledgeServiceDep, request: SearchRequest
) -> SearchResponse:
    """Return the passages that bear on a query, ranked best first.

    Searches two kinds of knowledge this project produces about itself: the
    documentation that says how the system works, and the stored experiment
    records that say what was actually run. Every passage comes back with a
    stable citation that resolves by hand — a documentation file and heading,
    or an experiment id the experiment API will fetch.

    **The similarity metric is cosine.** Metadata filters are applied *before*
    ranking, so asking for five classification experiments searches
    classification experiments rather than ranking everything and discarding
    the rest.

    No answer is generated here and no language model is involved, so this
    endpoint needs no credential. An empty result is a truthful 200: it means
    nothing relevant was found, not that anything went wrong.
    """
    response = knowledge.search(
        request.query,
        top_k=request.top_k,
        similarity_threshold=request.similarity_threshold,
        source_types=request.filters.source_types,
        metadata=request.filters.as_metadata(),
    )
    return SearchResponse(
        query=response.question,
        results=[
            SearchResult(
                rank=result.rank,
                score=result.score,
                content=result.content,
                document_id=result.document_id,
                chunk_id=result.chunk_id,
                source_type=result.source_type,
                source_title=result.source_title,
                source_reference=result.source_reference,
                citation_id=result.citation,
                metadata=result.metadata,
            )
            for result in response.results
        ],
        result_count=len(response.results),
        top_k=response.top_k,
        similarity_threshold=response.similarity_threshold,
        similarity_metric=response.similarity_metric,
        candidate_count=response.candidate_count,
        citations=list(response.citations),
    )


@router.post(
    "/ask",
    dependencies=[Protected],
    response_model=AskResponse,
    responses=_ASK_ERRORS,
    summary="Answer a question from retrieved evidence, with citations",
)
def ask_knowledge(
    knowledge: KnowledgeServiceDep, request: AskRequest
) -> AskResponse:
    """Answer a question from the project's own documentation and history.

    **Evidence-grounded answers; the LLM is not the source of truth.** The
    model's knowledge is used to explain what an F1 score *means*, never to
    supply what this project *scored*. Every project-specific claim must come
    from a retrieved passage, every citation is checked against the passages
    actually supplied, and an answer citing a source that was not retrieved is
    rejected rather than quietly cleaned up.

    Read `status` before using `answer`:

    * `grounded` — backed by evidence, valid citations, no fabrications. The
      only status a caller may present to a user as an answer.
    * `insufficient_evidence` — retrieval found nothing worth grounding in, or
      the model said the evidence does not cover the question. No claim is
      being made. **HTTP 200**: the request was processed correctly.
    * `grounding_failed` — the model cited a source that was not retrieved, or
      cited nothing at all. The text is returned so a human can see what
      happened; it is not an answer. **HTTP 200** for the same reason.

    Safety settings belong to the server. A request may vary how much evidence
    to look at and where to look for it — it cannot supply a system prompt, a
    provider endpoint, a credential or a model, and it cannot switch off
    grounding or citation validation.

    Requires a language-model credential; without one this returns 503 while
    `POST /api/v1/search` continues to work.
    """
    answer = knowledge.ask(
        request.question,
        top_k=request.top_k,
        similarity_threshold=request.similarity_threshold,
        source_types=request.filters.source_types,
        metadata=request.filters.as_metadata(),
    )
    return AskResponse.model_validate(answer.as_dict())


@router.get(
    "/knowledge/status",
    response_model=KnowledgeStatusResponse,
    summary="Report what the knowledge endpoints can currently do",
)
def knowledge_status(knowledge: KnowledgeServiceDep) -> KnowledgeStatusResponse:
    """Describe whether search and answering are available, and their limits.

    Exists so a client can tell "answering is not configured" from "the
    question had no answer" before asking, and so it need not hard-code the
    limits the server already knows. Reports whether a credential is
    configured, never what it is.
    """
    return KnowledgeStatusResponse.model_validate(knowledge.describe())
