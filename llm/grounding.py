"""Checking that an answer is actually backed by the evidence.

The prompt asks a model to cite only what it was given. This module makes that
true. The difference matters: a prompt is a request, and a request is followed
most of the time, which is exactly the failure mode that makes ungrounded
systems dangerous — they are right often enough to be trusted and wrong often
enough to mislead.

So every generated answer is checked:

1. Extract every citation-shaped identifier from the text.
2. Split them into ones that appear in the retrieved evidence and ones that do
   not.
3. If any do not, the answer **fails**. It is not silently cleaned.
4. If none do, and evidence was available, the answer **fails** — text with no
   citations is not a grounded answer, whatever it says.

**Fabricated citations are never repaired.** A citation of
``experiment:exp_999`` when ``experiment:exp_123`` was retrieved might be a
typo, or might be a model inventing a run that does not exist; guessing which
would mean attaching a real source to a claim it may not support, and a wrong
citation that looks right is worse than an obvious failure. The identifier is
reported as rejected and the answer is marked ungrounded.

**Extraction is deliberately conservative about what counts as a citation.**
Prose is full of ``word:word`` shapes — "note: this", "ratio:0.5" — and
treating those as fabricated citations would fail honest answers. The rule:

- Anything in square brackets matching the citation grammar is a citation
  attempt, whatever its prefix. ``[paper:smith2020]`` is a fabrication, not a
  coincidence.
- Outside brackets, only identifiers whose prefix is a known citation prefix
  count — the prefixes used by the retrieval layer, plus any appearing in the
  evidence actually supplied.

That keeps ordinary prose from failing while still catching every plausible
fabrication.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from llm.answers import Citation
from llm.context import EvidenceContext
from rag.citations import CITATION_PREFIXES

#: Characters of a cited passage kept as an excerpt on the citation.
EXCERPT_LENGTH = 240

#: The grammar of a citation: a lowercase prefix, a colon, a body of
#: identifier characters, and an optional ``#fragment``. Matches
#: ``docs:ml-readme#preprocessing`` and
#: ``experiment:exp_84a8d53a1f5f_20260828T134457Z_e420``.
_BODY = r"[A-Za-z0-9][A-Za-z0-9._-]*(?:#[A-Za-z0-9._-]+)?"
_CITATION = rf"(?P<prefix>[a-z][a-z0-9_]{{1,19}}):(?P<body>{_BODY})"

#: A citation inside square brackets — an unambiguous citation attempt.
BRACKETED_PATTERN = re.compile(rf"\[\s*{_CITATION}\s*\]")
#: A bare citation, accepted only when its prefix is known.
BARE_PATTERN = re.compile(rf"(?<![\w\[]){_CITATION}")

#: Prefixes the retrieval layer produces. Bare identifiers using one of these
#: count as citations even without brackets.
KNOWN_PREFIXES: frozenset[str] = frozenset(CITATION_PREFIXES.values()) | {"source"}


@dataclass(frozen=True)
class GroundingReport:
    """The result of checking one answer against its evidence."""

    #: Citations the model used that were genuinely retrieved, in the order
    #: they first appear in the answer.
    valid: tuple[str, ...] = ()
    #: Citation-shaped identifiers that were not in the evidence.
    fabricated: tuple[str, ...] = ()
    #: True when the answer may be presented as grounded.
    is_grounded: bool = False
    reasons: tuple[str, ...] = field(default=())

    @property
    def has_fabrications(self) -> bool:
        """True when the model cited something it was never shown."""
        return bool(self.fabricated)


def extract_citations(
    text: str, known_prefixes: Sequence[str] = ()
) -> tuple[str, ...]:
    """Pull every citation-shaped identifier out of generated text.

    Args:
        text: The model's answer.
        known_prefixes: Prefixes to accept outside square brackets, in
            addition to the retrieval layer's own. Normally the prefixes of
            the evidence actually supplied.

    Returns:
        tuple[str, ...]: Identifiers in order of first appearance, without
        duplicates. Trailing punctuation is not included — a citation at the
        end of a sentence is the same citation.
    """
    prefixes = KNOWN_PREFIXES | {prefix.lower() for prefix in known_prefixes}
    found: list[str] = []

    def remember(prefix: str, body: str) -> None:
        """Record one identifier, keeping first-appearance order."""
        identifier = f"{prefix}:{body}".rstrip(".,;:)")
        if identifier not in found:
            found.append(identifier)

    for match in BRACKETED_PATTERN.finditer(text):
        remember(match.group("prefix"), match.group("body"))

    for match in BARE_PATTERN.finditer(text):
        if match.group("prefix").lower() in prefixes:
            remember(match.group("prefix"), match.group("body"))

    return tuple(found)


def validate_citations(
    text: str, context: EvidenceContext
) -> GroundingReport:
    """Check an answer's citations against the evidence it was given.

    Args:
        text: The model's answer.
        context: The evidence that was supplied, whose
            :attr:`~llm.context.EvidenceContext.allowed_citations` is the
            complete set of permitted identifiers.

    Returns:
        GroundingReport: Which citations were real, which were invented, and
        whether the answer may be called grounded.
    """
    allowed = set(context.allowed_citations)
    prefixes = [citation.split(":", 1)[0] for citation in allowed if ":" in citation]
    produced = extract_citations(text, known_prefixes=prefixes)

    valid = tuple(item for item in produced if item in allowed)
    fabricated = tuple(item for item in produced if item not in allowed)

    reasons: list[str] = []
    if fabricated:
        reasons.append(
            "The answer cited "
            + ", ".join(f"'{item}'" for item in fabricated)
            + ", which "
            + ("was" if len(fabricated) == 1 else "were")
            + " not in the retrieved evidence."
        )
    if not valid and not context.is_empty:
        reasons.append(
            "The answer cited no retrieved source, so none of it is backed by "
            "the evidence."
        )

    return GroundingReport(
        valid=valid,
        fabricated=fabricated,
        is_grounded=bool(valid) and not fabricated,
        reasons=tuple(reasons),
    )


def build_citations(
    citation_ids: Sequence[str], context: EvidenceContext
) -> tuple[Citation, ...]:
    """Build citation objects from identifiers and the evidence behind them.

    Only the identifier comes from the model. The title, the reference, the
    score and the excerpt are all read from what was actually retrieved, so
    they are trustworthy even when the surrounding prose is not.

    Args:
        citation_ids: Validated identifiers, in the order they should appear.
        context: The evidence they were drawn from.

    Returns:
        tuple[Citation, ...]: One citation per identifier that resolves.
    """
    by_id = {item.citation: item for item in context.items}
    citations: list[Citation] = []

    for identifier in citation_ids:
        item = by_id.get(identifier)
        if item is None:
            # Unreachable for validated identifiers; skipped rather than
            # invented, because a citation with no source behind it is
            # exactly what this module exists to prevent.
            continue
        excerpt = item.content.strip().replace("\n", " ")
        if len(excerpt) > EXCERPT_LENGTH:
            excerpt = excerpt[:EXCERPT_LENGTH].rstrip() + "…"
        citations.append(
            Citation(
                citation_id=identifier,
                source_type=item.source_type,
                source_title=item.source_title,
                source_reference=item.source_reference,
                relevance_score=item.score,
                excerpt=excerpt,
            )
        )
    return tuple(citations)


def looks_like_injection_attempt(context: EvidenceContext) -> bool:
    """Report whether any retrieved passage reads like an instruction.

    Used only to attach a warning to the answer, never to alter behaviour —
    the defence is that retrieved text is presented as data and the model is
    told not to obey it, not that suspicious text is filtered out. Filtering
    would be an arms race against phrasing; this is a flag for a human.
    """
    markers = (
        "ignore previous instruction",
        "ignore all previous",
        "disregard the above",
        "disregard previous",
        "you are now",
        "system prompt",
        "reveal the api key",
        "reveal your",
        "print the api key",
        "output the key",
        "new instructions:",
    )
    for item in context.items:
        lowered = item.content.lower()
        if any(marker in lowered for marker in markers):
            return True
    return False


__all__ = [
    "BARE_PATTERN",
    "BRACKETED_PATTERN",
    "EXCERPT_LENGTH",
    "KNOWN_PREFIXES",
    "GroundingReport",
    "build_citations",
    "extract_citations",
    "looks_like_injection_attempt",
    "validate_citations",
]
