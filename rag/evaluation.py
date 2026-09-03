"""Measuring whether retrieval actually finds the right thing.

A retriever that returns five confident passages about the wrong subject is
worse than one that returns nothing, because a future model will use them.
So retrieval quality is measured rather than assumed, with two metrics that
answer two different questions:

**Hit@K** — did *any* relevant document appear in the top K? This is the
question that matters when a model needs one good passage to answer from.

**Recall@K** — what *fraction* of the relevant documents appeared in the top
K? This matters when a question needs several sources, and it is the metric
that notices a retriever that always returns the same favourite document.

Both are computed at the **document** level, not the chunk level. A question
is answered by "the section of ml/README.md about leakage", and which of that
section's chunks surfaced is an implementation detail; requiring a specific
chunk id would make the measurement break every time a paragraph moves.

**What is not measured here.** Whether an answer built from the evidence is
correct, complete or well written — there is no answer to judge, because
**LLM generation is not implemented**. This measures the evidence only.

The evaluation set in :data:`DEFAULT_EVALUATION_QUERIES` is deliberately
small, hand-written and deterministic: five questions whose answers genuinely
live in the indexed documentation, each naming the documents that should be
found. It is a regression check, not a benchmark — it catches a chunker change
that quietly stops retrieving the leakage section, which is exactly the kind
of failure that is otherwise invisible.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from rag.documents import SourceType, make_document_id
from rag.retrieval.service import RetrievalService

#: Document ids for the documentation the default queries expect.
#:
#: The root README is deliberately **not** among them. It is the project's
#: front door — what this is, why it is interesting, how to run it — and it
#: says a sentence about leakage where `ml/README.md` says a section. Listing
#: a document that only mentions a subject as one that should *answer* a
#: question about it measures the wrong thing: recall then falls whenever the
#: front page is edited, which is not a retrieval regression.
ML_README = make_document_id(SourceType.PROJECT_DOCUMENTATION.value, "ml/README.md")
BACKEND_README = make_document_id(
    SourceType.PROJECT_DOCUMENTATION.value, "backend/README.md"
)
RAG_README = make_document_id(SourceType.PROJECT_DOCUMENTATION.value, "rag/README.md")
ARCHITECTURE_DOC = make_document_id(
    SourceType.PROJECT_DOCUMENTATION.value, "docs/ARCHITECTURE.md"
)
API_DOC = make_document_id(SourceType.PROJECT_DOCUMENTATION.value, "docs/API.md")


@dataclass(frozen=True)
class EvaluationQuery:
    """One question and the documents that should answer it.

    ``relevant_document_ids`` is a set of alternatives: a query is a hit when
    *any* of them is retrieved, and recall is measured against all of them.
    ``required_metadata`` records a filter the query should be run with, so
    the evaluation can exercise the hybrid path as well as plain search.
    """

    question: str
    relevant_document_ids: tuple[str, ...]
    source_types: tuple[str, ...] = ()
    required_metadata: dict[str, Any] = field(default_factory=dict)
    note: str = ""


#: Five deterministic questions whose answers live in the project's own
#: documentation. Experiment-specific questions are built at evaluation time
#: from whatever runs exist, since their ids are not known in advance.
DEFAULT_EVALUATION_QUERIES: tuple[EvaluationQuery, ...] = (
    EvaluationQuery(
        question="How does ML Copilot prevent data leakage during preprocessing?",
        relevant_document_ids=(ML_README, ARCHITECTURE_DOC),
        source_types=(SourceType.PROJECT_DOCUMENTATION.value,),
        note="A section in the ML README, a paragraph in the architecture doc.",
    ),
    EvaluationQuery(
        question=(
            "What is the difference between cross-validation and the final "
            "test evaluation?"
        ),
        relevant_document_ids=(ML_README, ARCHITECTURE_DOC),
        source_types=(SourceType.PROJECT_DOCUMENTATION.value,),
        note="Both explain why the test set is measured once.",
    ),
    EvaluationQuery(
        question="What preprocessing is applied to categorical columns?",
        relevant_document_ids=(ML_README, ARCHITECTURE_DOC),
        source_types=(SourceType.PROJECT_DOCUMENTATION.value,),
        note="One-hot encoding and cardinality limits.",
    ),
    EvaluationQuery(
        question="Which HTTP endpoints run an experiment and list past experiments?",
        relevant_document_ids=(BACKEND_README, API_DOC),
        source_types=(SourceType.PROJECT_DOCUMENTATION.value,),
        note="The endpoint tables, and the API reference.",
    ),
    EvaluationQuery(
        question="How are documents chunked and embedded for retrieval?",
        relevant_document_ids=(RAG_README, ARCHITECTURE_DOC),
        source_types=(SourceType.PROJECT_DOCUMENTATION.value,),
        note="The RAG layer's own documentation, and the architecture doc.",
    ),
)


@dataclass(frozen=True)
class QueryOutcome:
    """What one query retrieved, and whether it was right."""

    question: str
    expected: tuple[str, ...]
    retrieved: tuple[str, ...]
    hit: bool
    recall: float
    top_score: float | None
    citations: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Render the outcome as plain JSON-safe values."""
        return {
            "question": self.question,
            "expected_document_ids": list(self.expected),
            "retrieved_document_ids": list(self.retrieved),
            "hit": self.hit,
            "recall": self.recall,
            "top_score": self.top_score,
            "citations": list(self.citations),
        }


@dataclass(frozen=True)
class EvaluationReport:
    """Retrieval quality over a set of queries."""

    k: int
    outcomes: tuple[QueryOutcome, ...]

    @property
    def query_count(self) -> int:
        """How many queries were evaluated."""
        return len(self.outcomes)

    @property
    def hit_rate(self) -> float:
        """Hit@K: the share of queries that found at least one right document."""
        if not self.outcomes:
            return 0.0
        return sum(1 for outcome in self.outcomes if outcome.hit) / len(self.outcomes)

    @property
    def recall(self) -> float:
        """Recall@K: the mean share of relevant documents retrieved."""
        if not self.outcomes:
            return 0.0
        return sum(outcome.recall for outcome in self.outcomes) / len(self.outcomes)

    @property
    def misses(self) -> tuple[QueryOutcome, ...]:
        """The queries that found nothing relevant — the ones worth reading."""
        return tuple(outcome for outcome in self.outcomes if not outcome.hit)

    def as_dict(self) -> dict[str, Any]:
        """Render the whole report as plain JSON-safe values."""
        return {
            "k": self.k,
            "query_count": self.query_count,
            "hit_at_k": self.hit_rate,
            "recall_at_k": self.recall,
            "miss_count": len(self.misses),
            "outcomes": [outcome.as_dict() for outcome in self.outcomes],
        }

    def as_text(self) -> str:
        """Render the report as a short readable summary."""
        lines = [
            f"Retrieval evaluation over {self.query_count} queries at k={self.k}",
            f"  Hit@{self.k}:    {self.hit_rate:.2%}",
            f"  Recall@{self.k}: {self.recall:.2%}",
        ]
        if self.misses:
            lines.append("  Missed:")
            lines.extend(f"    - {outcome.question}" for outcome in self.misses)
        return "\n".join(lines)


def evaluate_query(
    service: RetrievalService, query: EvaluationQuery, *, k: int
) -> QueryOutcome:
    """Run one evaluation query and score what came back.

    Args:
        service: The retrieval service to exercise.
        query: The question and the documents it should find.
        k: How many results to consider.

    Returns:
        QueryOutcome: What was retrieved and whether it was relevant.
    """
    response = service.search(
        query.question,
        top_k=k,
        source_types=query.source_types,
        equals=query.required_metadata or None,
    )
    retrieved = tuple(dict.fromkeys(result.document_id for result in response.results))
    expected = set(query.relevant_document_ids)
    found = expected.intersection(retrieved)

    return QueryOutcome(
        question=query.question,
        expected=query.relevant_document_ids,
        retrieved=retrieved,
        hit=bool(found),
        recall=len(found) / len(expected) if expected else 0.0,
        top_score=response.results[0].score if response.results else None,
        citations=response.citations,
    )


def evaluate_retrieval(
    service: RetrievalService,
    queries: Iterable[EvaluationQuery] = DEFAULT_EVALUATION_QUERIES,
    *,
    k: int = 5,
) -> EvaluationReport:
    """Measure Hit@K and Recall@K over a set of queries.

    Args:
        service: The retrieval service to exercise.
        queries: The evaluation set; the default documentation questions
            when omitted.
        k: How many results each query may return.

    Returns:
        EvaluationReport: Per-query outcomes and the two aggregate metrics.
    """
    outcomes = tuple(evaluate_query(service, query, k=k) for query in queries)
    return EvaluationReport(k=k, outcomes=outcomes)


def experiment_queries(
    runs: Sequence[Any], *, limit: int = 3
) -> tuple[EvaluationQuery, ...]:
    """Build evaluation queries from experiments that actually exist.

    An experiment's id is not known when the evaluation set is written, so
    these queries are generated from the stored runs: for each one, a
    question naming it and the document that must be retrieved to answer it.

    Args:
        runs: ``ExperimentRun`` records.
        limit: How many runs to build queries for.

    Returns:
        tuple[EvaluationQuery, ...]: Two queries per run — which model was
        selected, and which features mattered.
    """
    queries: list[EvaluationQuery] = []
    for run in list(runs)[:limit]:
        document_id = make_document_id(
            SourceType.EXPERIMENT.value, run.experiment_id
        )
        queries.append(
            EvaluationQuery(
                question=(
                    f"Which model was selected in experiment {run.experiment_id} "
                    "and what did it score on the test set?"
                ),
                relevant_document_ids=(document_id,),
                source_types=(SourceType.EXPERIMENT.value,),
                note="Selection and final evaluation sections.",
            )
        )
        queries.append(
            EvaluationQuery(
                question=(
                    "What features influenced the selected model in experiment "
                    f"{run.experiment_id}?"
                ),
                relevant_document_ids=(document_id,),
                source_types=(SourceType.EXPERIMENT.value,),
                note="Explainability section.",
            )
        )
    return tuple(queries)


__all__ = [
    "DEFAULT_EVALUATION_QUERIES",
    "EvaluationQuery",
    "EvaluationReport",
    "QueryOutcome",
    "evaluate_query",
    "evaluate_retrieval",
    "experiment_queries",
]
