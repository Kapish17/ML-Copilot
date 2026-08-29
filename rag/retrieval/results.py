"""What a retrieval returns.

A :class:`RetrievalResult` is one piece of evidence: the passage, how well it
matched, and everything needed to attribute it. A :class:`RetrievalResponse`
is the whole answer to one query — the evidence, and what was asked.

Everything here is **evidence, not prose**. There is no field for a summary, a
conclusion or an answer, and that is deliberate: this layer's job is to find
the right passages and say where they came from. Turning them into an answer
is a later commit's job, and giving it somewhere to put one now would invite
ungrounded text into the index. **No LLM generation is implemented.**

The serialised form is the shape a future model would be handed::

    {
      "question": "How does ML Copilot prevent data leakage?",
      "results": [
        {"rank": 1, "score": 0.71, "content": "...",
         "citation": "docs:ml-readme#leakage-prevention",
         "source_type": "project_documentation", "metadata": {...}}
      ]
    }

No embedding vector appears in any of it. A vector is an implementation
detail of the search, it is large, and it means nothing to a reader.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from rag.documents import Chunk


@dataclass(frozen=True)
class RetrievalResult:
    """One retrieved passage, with its score and its attribution."""

    rank: int
    score: float
    content: str
    document_id: str
    chunk_id: str
    source_type: str
    source_title: str
    source_reference: str
    citation: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_chunk(cls, chunk: Chunk, *, rank: int, score: float) -> RetrievalResult:
        """Build a result from a stored chunk and its similarity."""
        return cls(
            rank=rank,
            score=float(score),
            content=chunk.content,
            document_id=chunk.document_id,
            chunk_id=chunk.chunk_id,
            source_type=chunk.source_type,
            source_title=chunk.source_title,
            source_reference=chunk.source_reference,
            citation=chunk.citation,
            metadata=dict(chunk.metadata),
        )

    @property
    def heading(self) -> str | None:
        """The heading this passage sits under, when it has one."""
        value = self.metadata.get("heading")
        return str(value) if value else None

    @property
    def experiment_id(self) -> str | None:
        """The experiment this passage came from, when it came from one."""
        value = self.metadata.get("experiment_id")
        return str(value) if value else None

    def as_dict(self) -> dict[str, Any]:
        """Render the result as plain JSON-safe values.

        Deliberately without the embedding: the caller gets the text, the
        score and the attribution, which is everything an answer can be built
        or checked from.
        """
        return {
            "rank": self.rank,
            "score": self.score,
            "content": self.content,
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "source_type": self.source_type,
            "source_title": self.source_title,
            "source_reference": self.source_reference,
            "citation": self.citation,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RetrievalResponse:
    """The evidence found for one question, and how it was looked for."""

    question: str
    results: tuple[RetrievalResult, ...] = ()
    top_k: int = 0
    similarity_threshold: float = 0.0
    similarity_metric: str = "cosine"
    filter_applied: Mapping[str, Any] = field(default_factory=dict)
    #: How many chunks the filter admitted before ranking. Distinguishes "the
    #: index has nothing about this" from "the filter excluded everything".
    candidate_count: int = 0

    def __len__(self) -> int:
        """How many results were returned."""
        return len(self.results)

    def __iter__(self):
        """Iterate the results, best first."""
        return iter(self.results)

    @property
    def is_empty(self) -> bool:
        """True when nothing met the query and its threshold."""
        return not self.results

    @property
    def citations(self) -> tuple[str, ...]:
        """The citation of every result, in rank order, de-duplicated."""
        return tuple(dict.fromkeys(result.citation for result in self.results))

    def by_source_type(self, source_type: str) -> tuple[RetrievalResult, ...]:
        """The results from one kind of source, in rank order."""
        return tuple(
            result for result in self.results if result.source_type == source_type
        )

    def as_dict(self) -> dict[str, Any]:
        """Render the whole response as plain JSON-safe values."""
        return {
            "question": self.question,
            "top_k": self.top_k,
            "similarity_threshold": self.similarity_threshold,
            "similarity_metric": self.similarity_metric,
            "filter": dict(self.filter_applied),
            "candidate_count": self.candidate_count,
            "result_count": len(self.results),
            "citations": list(self.citations),
            "results": [result.as_dict() for result in self.results],
        }

    def as_evidence(self) -> list[dict[str, Any]]:
        """Render just the evidence, in the shape a future model would read.

        One entry per passage: what it says, where it came from, and how well
        it matched. Nothing else — the reasoning, the synthesis and the prose
        belong to whatever consumes this, not to the retriever.
        """
        return [
            {
                "content": result.content,
                "source": result.citation,
                "source_title": result.source_title,
                "score": result.score,
                "metadata": dict(result.metadata),
            }
            for result in self.results
        ]


def rank_results(hits: Sequence[tuple[Chunk, float]]) -> tuple[RetrievalResult, ...]:
    """Turn ``(chunk, score)`` pairs into ranked results, best first."""
    return tuple(
        RetrievalResult.from_chunk(chunk, rank=position, score=score)
        for position, (chunk, score) in enumerate(hits, start=1)
    )


__all__ = ["RetrievalResponse", "RetrievalResult", "rank_results"]
