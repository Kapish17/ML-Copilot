"""Checking the final answer against what was actually observed.

This is Commit 10's grounding, applied to a different kind of evidence. The
extraction and validation are literally the same functions —
:func:`llm.grounding.extract_citations` and
:func:`llm.grounding.validate_citations` — because two implementations of
"is this citation real" would eventually disagree, and the one that mattered
would be whichever ran first.

What changes is where the allowed set comes from. In Commit 10 it was the
passages placed in the prompt. Here it is every citation identifier that came
back from a ``search_knowledge`` observation during the run. Same rule, wider
window: a citation is valid exactly when the run actually retrieved it.

Fabricated citations are reported, never repaired. Guessing which real source
a model *meant* would turn an obvious failure into a subtle one — the answer
would look cited, and the citation would point somewhere the model never read.

There is a second kind of grounding here that Commit 10 did not need, because
Commit 10 only ever had retrieved text. An agent also produces *results*:
experiment ids, scores, feature names. Those are not citable in the RAG sense
— they came from a tool, not a document — so the check for them is different:
:func:`unsupported_experiment_ids` looks for experiment identifiers in the
answer that no observation produced, which is the cheapest reliable signal
that a model has started inventing runs.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from agent.observations import Observation
from agent.results import AgentCitation
from llm.context import ContextItem, EvidenceContext
from llm.grounding import GroundingReport, validate_citations

#: The line the answer prompt asks for when nothing supports an answer. Read
#: as a declared protocol rather than guessed at from phrasing, exactly as the
#: answer service does in Commit 10.
ABSTENTION_MARKER = "INSUFFICIENT_EVIDENCE"

#: Experiment identifiers, as Commit 7 forms them. Used only to notice an id
#: in an answer that no observation produced.
EXPERIMENT_ID_PATTERN = re.compile(r"\bexp_[A-Za-z0-9_\-]+\b")


def evidence_from_observations(observations: Sequence[Observation]) -> EvidenceContext:
    """Build the evidence context the grounding check runs against.

    Every retrieved passage the run saw becomes one
    :class:`~llm.context.ContextItem`, so the allowed citation set is exactly
    what was retrieved — no more, and no less.
    """
    items: list[ContextItem] = []
    for observation in observations:
        if not observation.succeeded:
            continue
        results = observation.output.get("results")
        if not isinstance(results, list):
            continue
        for entry in results:
            if not isinstance(entry, dict):
                continue
            citation = entry.get("citation_id")
            if not isinstance(citation, str) or not citation:
                continue
            items.append(
                ContextItem(
                    index=len(items) + 1,
                    citation=citation,
                    score=float(entry.get("score") or 0.0),
                    content=str(entry.get("content") or ""),
                    source_type=str(entry.get("source_type") or ""),
                    source_title=str(entry.get("source_title") or ""),
                    source_reference=str(entry.get("source_reference") or ""),
                    truncated=bool(entry.get("truncated")),
                )
            )

    return EvidenceContext(items=tuple(items), retrieved_count=len(items))


def build_agent_citations(
    citation_ids: Sequence[str], context: EvidenceContext
) -> tuple[AgentCitation, ...]:
    """Build citation objects from identifiers and the evidence behind them.

    Only the identifier comes from the model; the title, reference and score
    are read from the passage that was actually retrieved.
    """
    by_id = {item.citation: item for item in context.items}
    citations: list[AgentCitation] = []
    for citation_id in citation_ids:
        item = by_id.get(citation_id)
        if item is None:  # pragma: no cover - validated ids are always present
            continue
        citations.append(
            AgentCitation(
                citation_id=citation_id,
                source_type=item.source_type,
                source_title=item.source_title,
                source_reference=item.source_reference,
                score=item.score,
            )
        )
    return tuple(citations)


def check_citations(text: str, context: EvidenceContext) -> GroundingReport:
    """Validate an answer's citations against the retrieved evidence.

    A direct call into the LLM layer's own validator. This module deliberately
    adds nothing to it — the point is that the agent and the ask endpoint
    apply the same rule.
    """
    return validate_citations(text, context)


def is_abstention(text: str) -> bool:
    """Whether the answer declared that the evidence was not enough."""
    return any(
        line.strip() == ABSTENTION_MARKER for line in (text or "").splitlines()
    )


def observed_experiment_ids(observations: Sequence[Observation]) -> set[str]:
    """Every experiment identifier the run actually produced or read."""
    found: set[str] = set()
    for observation in observations:
        identifier = observation.output.get("experiment_id")
        if isinstance(identifier, str) and identifier:
            found.add(identifier)
        for entry in observation.output.get("results") or []:
            if isinstance(entry, dict):
                reference = entry.get("source_reference")
                if isinstance(reference, str) and reference.startswith("exp_"):
                    found.add(reference)
    return found


def unsupported_experiment_ids(
    text: str, observations: Sequence[Observation]
) -> tuple[str, ...]:
    """Experiment ids in the answer that no observation produced.

    An invented experiment id is a fabricated *result*, which is worse than a
    fabricated citation: it looks like a record a person can go and read, and
    there is nothing there.
    """
    observed = observed_experiment_ids(observations)
    mentioned = EXPERIMENT_ID_PATTERN.findall(text or "")
    return tuple(
        dict.fromkeys(item for item in mentioned if item not in observed)
    )


def has_reportable_results(observations: Sequence[Observation]) -> bool:
    """Whether any tool produced something an answer could be built from."""
    return any(observation.succeeded for observation in observations)


def strip_abstention_marker(text: str) -> str:
    """Remove the protocol line, keeping whatever the model said around it."""
    lines = [
        line
        for line in (text or "").splitlines()
        if line.strip() != ABSTENTION_MARKER
    ]
    return "\n".join(lines).strip()


def coerce_length(text: str, limit: int) -> tuple[str, bool]:
    """Cut an over-long answer at the configured limit.

    Returns the text and whether it was cut. A model that ignores the length
    it was asked for is trimmed rather than trusted, because the limit is
    there to bound what a caller receives, not to be politely requested.
    """
    if not isinstance(text, str):  # pragma: no cover - defensive
        return "", False
    if len(text) <= limit:
        return text, False
    return text[:limit].rstrip() + "…", True


def render_observations_for_answer(
    observations: Sequence[Observation],
) -> list[dict[str, Any]]:
    """Render observations as the answer step will be shown them."""
    return [observation.as_dict() for observation in observations]


__all__ = [
    "ABSTENTION_MARKER",
    "EXPERIMENT_ID_PATTERN",
    "build_agent_citations",
    "check_citations",
    "coerce_length",
    "evidence_from_observations",
    "has_reportable_results",
    "is_abstention",
    "observed_experiment_ids",
    "render_observations_for_answer",
    "strip_abstention_marker",
    "unsupported_experiment_ids",
]
