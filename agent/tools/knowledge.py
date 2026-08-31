"""Searching the knowledge base — and only searching it.

This tool is the agent's read path into the retrieval layer built in Commit 9.
It is worth being explicit about what is *absent*: there is no tool to index,
re-index, delete, edit or add a document, and no argument that names a file, a
directory or a URL. The agent has no way to express a change to the knowledge
base because no such operation is declared anywhere it can reach.

The output carries citation identifiers, and those identifiers matter more
than the text beside them. They are the only strings the final answer may
cite; the grounding check compares what the model wrote against exactly this
set. An identifier that never came back from a search is a fabrication, and
saying "the retrieved evidence" while citing something else is precisely the
failure the check exists to catch.

Embeddings never leave the retrieval layer. What comes back here is text,
scores and attribution.

Two limits are the retrieval layer's own rather than this tool's invention:
``top_k`` is capped by the RAG configuration, and a query is bounded by the
configured maximum length. Reusing them means a planner and an HTTP client are
held to the same rules by the same code.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from agent.schemas import (
    INTEGER,
    STRING,
    STRING_LIST,
    ArgumentField,
    ArgumentSchema,
)
from agent.tools.base import BaseTool, ToolResult

#: Characters of any single retrieved passage carried into an observation.
#: Long enough to answer from, short enough that a handful of passages cannot
#: exhaust the run's context budget.
MAX_PASSAGE_CHARS = 1_200
#: Appended to a passage this module shortens.
TRUNCATION_MARKER = "…[truncated]"


def _passage(result: Any, limit: int) -> dict[str, Any]:
    """Render one retrieved passage as plain values, without its vector."""
    content = getattr(result, "content", "") or ""
    truncated = len(content) > limit
    if truncated:
        content = content[: limit - len(TRUNCATION_MARKER)] + TRUNCATION_MARKER

    return {
        "citation_id": getattr(result, "citation", None),
        "rank": getattr(result, "rank", None),
        "score": getattr(result, "score", None),
        "source_type": getattr(result, "source_type", None),
        "source_title": getattr(result, "source_title", None),
        "source_reference": getattr(result, "source_reference", None),
        "content": content,
        "truncated": truncated,
    }


class SearchKnowledgeTool(BaseTool):
    """Search project documentation and experiment history for evidence."""

    tool_name = "search_knowledge"
    tool_description = (
        "Search the project's own documentation and its stored experiment "
        "history, and return the passages that bear on a question, each with "
        "a citation id. Use this to answer questions about how the system "
        "works, what an earlier experiment found, or what a term means in "
        "this project. Returns evidence, never an answer, and cannot modify "
        "anything it searches."
    )

    def __init__(
        self,
        retrieval: Any,
        *,
        max_top_k: int = 10,
        max_query_length: int = 2_000,
        source_types: Callable[[], Sequence[str]] | Sequence[str] = (),
        max_passage_chars: int = MAX_PASSAGE_CHARS,
    ) -> None:
        """Wire the tool to the retrieval service and its own limits.

        Args:
            retrieval: The existing retrieval service. Only ``search`` is
                used, and only ``search`` is declared on the protocol.
            max_top_k: Largest ``top_k`` a planner may request. Comes from the
                RAG configuration so both callers share one limit.
            max_query_length: Longest query a planner may submit, likewise.
            source_types: The kinds of source a planner may filter by.
            max_passage_chars: Characters of each passage to carry forward.
        """
        super().__init__()
        self._retrieval = retrieval
        self._max_top_k = max(1, int(max_top_k))
        self._max_query_length = max(1, int(max_query_length))
        self._source_types = source_types
        self._max_passage_chars = max(120, int(max_passage_chars))

    @property
    def schema(self) -> ArgumentSchema:
        """A query, how much of it to return, and where to look."""
        return ArgumentSchema(
            fields=(
                ArgumentField(
                    name="query",
                    type=STRING,
                    description="What to search for, in plain language.",
                    required=True,
                    max_length=self._max_query_length,
                ),
                ArgumentField(
                    name="top_k",
                    type=INTEGER,
                    description="How many passages to return.",
                    minimum=1,
                    maximum=self._max_top_k,
                ),
                ArgumentField(
                    name="source_types",
                    type=STRING_LIST,
                    description=(
                        "Restrict the search to these kinds of source. Omit "
                        "to search everything."
                    ),
                    max_items=8,
                    choices_provider=lambda: list(
                        self._source_types()
                        if callable(self._source_types)
                        else self._source_types
                    ),
                ),
            )
        )

    def run(self, arguments: Mapping[str, Any]) -> ToolResult:
        """Search, and return the evidence with its citation identifiers."""
        response = self._retrieval.search(
            arguments["query"],
            top_k=arguments.get("top_k"),
            source_types=tuple(arguments.get("source_types") or ()),
        )

        results = list(getattr(response, "results", ()) or ())
        passages = [_passage(item, self._max_passage_chars) for item in results]
        citations = tuple(
            passage["citation_id"]
            for passage in passages
            if isinstance(passage["citation_id"], str) and passage["citation_id"]
        )

        output: dict[str, Any] = {
            "status": "ok",
            "query": arguments["query"],
            "result_count": len(passages),
            "results": passages,
            "citations": list(citations),
        }
        if not passages:
            # A truthful empty result, not a failure. The planner is expected
            # to say so rather than fill the gap from the model's own memory.
            output["status"] = "no_results"
            output["message"] = (
                "Nothing in the indexed documentation or experiment history "
                "matched this query."
            )

        return ToolResult(output=output, citations=citations)


__all__ = ["MAX_PASSAGE_CHARS", "TRUNCATION_MARKER", "SearchKnowledgeTool"]
