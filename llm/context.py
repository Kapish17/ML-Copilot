"""Turning retrieved evidence into the block a model reads.

Two jobs, both of which matter more than they look.

**Choosing what fits.** A retrieval can return more text than a model's
context window holds, and passing all of it and hoping is how requests fail in
production. Selection is therefore explicit, bounded by configuration and
**deterministic**: evidence is taken in rank order until a limit is reached,
and what happens at the boundary is a documented rule rather than whatever the
data happened to do. Nothing is dropped silently — the result reports how much
was retrieved, how much was used, and whether anything was cut.

**Making it unmistakably data.** Retrieved documents can contain text that
looks like an instruction, because anyone who can get text into the index can
put one there. The evidence therefore travels inside an explicit
``<retrieved_evidence>`` block, each passage labelled with its citation and
its score, and the system prompt tells the model that everything inside that
block is untrusted data to be quoted, never instructions to be followed. Any
literal delimiter appearing inside a passage is neutralised, so a passage
cannot close the block and continue as if it were the prompt.

The selection rule, in full:

1. Discard evidence scoring below ``min_evidence_score`` — a weak match is not
   evidence, and answering from the least-bad chunk in an index that holds
   nothing relevant is the failure this layer exists to prevent.
2. Take the remaining evidence in rank order, best first.
3. Stop at ``max_context_chunks``.
4. Stop when the next passage would exceed ``max_context_chars``. If at least
   ``min_chunk_chars`` of it would fit, include that much and mark it
   truncated; otherwise leave it out whole.

Rank order is meaning order, so this keeps the best evidence and loses the
weakest — never a reshuffle, never a sample.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from llm.config import CHARS_PER_TOKEN, LLMConfig
from rag.retrieval import RetrievalResult

#: Delimiters of the evidence block. Named constants because the system prompt
#: and the tests both refer to them, and they must not drift apart.
EVIDENCE_OPEN = "<retrieved_evidence>"
EVIDENCE_CLOSE = "</retrieved_evidence>"
#: Marks where a passage was cut, so the model can see it is not the whole
#: thing and does not report a truncated list as complete.
TRUNCATION_MARKER = "\n[... passage truncated ...]"

#: Anything that could be read as an evidence delimiter is defanged before the
#: passage goes into the block. A passage that could close the block would be
#: able to continue as if it were prompt text.
_DELIMITER_PATTERN = re.compile(
    r"</?\s*retrieved_evidence\s*>|\[\s*SOURCE\s+\d+\s*\]", re.IGNORECASE
)
_DELIMITER_REPLACEMENT = "(delimiter removed)"


@dataclass(frozen=True)
class ContextItem:
    """One passage as it will be shown to the model."""

    index: int
    citation: str
    score: float
    content: str
    source_type: str
    source_title: str
    source_reference: str
    truncated: bool = False

    def render(self) -> str:
        """Render this passage as a labelled block."""
        header = (
            f"[SOURCE {self.index}]\n"
            f"citation: {self.citation}\n"
            f"source_type: {self.source_type}\n"
            f"source_title: {self.source_title}\n"
            f"score: {self.score:.3f}\n"
        )
        if self.truncated:
            header += "note: this passage was truncated to fit the context limit\n"
        return f"{header}content:\n{self.content}"

    def as_dict(self) -> dict[str, Any]:
        """Render the item as plain JSON-safe values, without its text."""
        return {
            "index": self.index,
            "citation": self.citation,
            "score": self.score,
            "source_type": self.source_type,
            "source_title": self.source_title,
            "source_reference": self.source_reference,
            "truncated": self.truncated,
            "character_count": len(self.content),
        }


@dataclass(frozen=True)
class EvidenceContext:
    """The evidence chosen for one question, and what it cost."""

    items: tuple[ContextItem, ...] = ()
    retrieved_count: int = 0
    #: Evidence dropped for scoring below the threshold.
    below_threshold_count: int = 0
    #: True when any passage was shortened or any passage was left out.
    truncated: bool = False
    #: Passages left out entirely because the budget was spent.
    omitted_count: int = 0

    def __len__(self) -> int:
        """How many passages will be shown to the model."""
        return len(self.items)

    def __iter__(self):
        """Iterate the passages, best first."""
        return iter(self.items)

    @property
    def is_empty(self) -> bool:
        """True when there is nothing to ground an answer in."""
        return not self.items

    @property
    def context_count(self) -> int:
        """How many passages are in the prompt."""
        return len(self.items)

    @property
    def character_count(self) -> int:
        """Total characters of evidence in the prompt."""
        return sum(len(item.content) for item in self.items)

    @property
    def approximate_tokens(self) -> int:
        """The character count expressed as an approximate token count.

        A rough English heuristic, not a tokeniser. Reported so a caller can
        see roughly what a request costs without this package taking on a
        tokeniser dependency per model family.
        """
        return self.character_count // CHARS_PER_TOKEN

    @property
    def allowed_citations(self) -> tuple[str, ...]:
        """Exactly the citations the model is permitted to use.

        The grounding validator checks generated citations against this set,
        so a citation the model was never shown cannot survive.
        """
        return tuple(dict.fromkeys(item.citation for item in self.items))

    def render(self) -> str:
        """Render the whole evidence block.

        Empty evidence still produces the block, with a line saying so —
        omitting it would leave the model to guess whether retrieval ran and
        found nothing, or never ran.
        """
        if not self.items:
            body = "(no evidence was retrieved for this question)"
        else:
            body = "\n\n".join(item.render() for item in self.items)
        return f"{EVIDENCE_OPEN}\n{body}\n{EVIDENCE_CLOSE}"

    def as_dict(self) -> dict[str, Any]:
        """Render the selection as plain JSON-safe values, without the text."""
        return {
            "retrieved_count": self.retrieved_count,
            "context_count": self.context_count,
            "below_threshold_count": self.below_threshold_count,
            "omitted_count": self.omitted_count,
            "context_truncated": self.truncated,
            "character_count": self.character_count,
            "approximate_tokens": self.approximate_tokens,
            "allowed_citations": list(self.allowed_citations),
            "items": [item.as_dict() for item in self.items],
        }


def neutralise_delimiters(text: str) -> str:
    """Remove anything in a passage that could pass for a block delimiter.

    Indexed content is untrusted: a passage containing
    ``</retrieved_evidence>`` could otherwise appear to close the evidence
    block, and everything after it would read as prompt rather than data. This
    is the structural half of the injection defence; the instructions in the
    system prompt are the other half, and neither is relied on alone.
    """
    return _DELIMITER_PATTERN.sub(_DELIMITER_REPLACEMENT, text)


def build_context(
    results: Sequence[RetrievalResult], config: LLMConfig
) -> EvidenceContext:
    """Choose the evidence for one question, within the configured limits.

    Args:
        results: Retrieved passages, best first.
        config: Supplies every limit and the evidence threshold.

    Returns:
        EvidenceContext: The chosen passages and an account of what was left
        out — nothing is dropped without being counted.
    """
    ranked = list(results)
    retrieved_count = len(ranked)

    usable = [
        result for result in ranked if result.score >= config.min_evidence_score
    ]
    below_threshold = retrieved_count - len(usable)

    items: list[ContextItem] = []
    used_chars = 0
    truncated = False
    omitted = 0

    for result in usable:
        if len(items) >= config.max_context_chunks:
            omitted += 1
            continue

        content = neutralise_delimiters(result.content)
        remaining = config.max_context_chars - used_chars

        if len(content) > remaining:
            # The marker is part of what gets stored, so it comes out of the
            # same budget — otherwise a "truncated" passage would overshoot
            # the limit it was truncated to respect.
            usable = remaining - len(TRUNCATION_MARKER)
            if usable < config.min_chunk_chars:
                # Too little room left to say anything useful; leave the
                # passage out whole rather than reduce it to a stub.
                omitted += 1
                truncated = True
                continue
            content = content[:usable].rstrip() + TRUNCATION_MARKER
            truncated = True
            was_truncated = True
        else:
            was_truncated = False

        items.append(
            ContextItem(
                index=len(items) + 1,
                citation=result.citation,
                score=float(result.score),
                content=content,
                source_type=result.source_type,
                source_title=result.source_title,
                source_reference=result.source_reference,
                truncated=was_truncated,
            )
        )
        used_chars += len(content)

    return EvidenceContext(
        items=tuple(items),
        retrieved_count=retrieved_count,
        below_threshold_count=below_threshold,
        truncated=truncated or omitted > 0,
        omitted_count=omitted,
    )


__all__ = [
    "EVIDENCE_CLOSE",
    "EVIDENCE_OPEN",
    "TRUNCATION_MARKER",
    "ContextItem",
    "EvidenceContext",
    "build_context",
    "neutralise_delimiters",
]
