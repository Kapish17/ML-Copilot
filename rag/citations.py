"""Stable references that point a claim back at its evidence.

Every retrieved chunk carries a citation, and a citation is enough on its own
to find the passage again: it names the kind of source, which source, and
which part of it.

::

    docs:ml-readme#leakage-prevention        a section of ml/README.md
    experiment:exp_84a8d53a1f5f_20260828T134457Z_e420
    experiment:exp_84a8d53a1f5f_...#explainability

The point is not the syntax. It is that when a future answer says "according
to experiment exp_84a8…", the reader can resolve that string: an experiment
citation is the id the experiment store already uses, and a documentation
citation is a file and a heading anchor. **No answer text is generated here** —
this module produces references, and a later commit will decide how a model is
asked to use them.
"""

from __future__ import annotations

from rag.documents import SourceType, slugify

#: The short prefix each source type gets in a citation.
CITATION_PREFIXES: dict[str, str] = {
    SourceType.PROJECT_DOCUMENTATION.value: "docs",
    SourceType.EXPERIMENT.value: "experiment",
    SourceType.ML_REFERENCE.value: "ml",
}
#: Used for a source type with no registered prefix, so a new ingestion
#: adapter still produces resolvable citations.
FALLBACK_PREFIX = "source"

#: Separates the source from the part of it being cited.
FRAGMENT_SEPARATOR = "#"


def citation_prefix(source_type: str) -> str:
    """Return the citation prefix for a source type."""
    return CITATION_PREFIXES.get(str(source_type), FALLBACK_PREFIX)


def document_slug(source_reference: str) -> str:
    """Reduce a source reference to the slug used in citations.

    A repository-relative path such as ``ml/README.md`` becomes
    ``ml-readme``: readable, stable, and free of separators that would make a
    citation ambiguous.
    """
    reference = str(source_reference)
    if reference.lower().endswith(".md"):
        reference = reference[: -len(".md")]
    elif reference.lower().endswith(".markdown"):
        reference = reference[: -len(".markdown")]
    return slugify(reference)


def build_citation(
    *,
    source_type: str,
    source_reference: str,
    fragment: str | None = None,
) -> str:
    """Build the citation for a source, optionally down to one part of it.

    Args:
        source_type: The source vocabulary term.
        source_reference: What identifies the source — a path, an experiment
            id. Experiment ids are kept verbatim, because the id *is* the
            reference a reader would look up.
        fragment: The section or record part being cited.

    Returns:
        str: A reference such as ``docs:root-readme#error-contract``.
    """
    prefix = citation_prefix(source_type)
    if str(source_type) == SourceType.EXPERIMENT.value:
        # An experiment id is already stable, unique and the thing a person
        # would paste into the API; slugifying it would break that.
        body = str(source_reference)
    else:
        body = document_slug(source_reference)

    citation = f"{prefix}:{body}"
    if fragment:
        citation = f"{citation}{FRAGMENT_SEPARATOR}{slugify(fragment)}"
    return citation


def citation_for_heading_path(
    *,
    source_type: str,
    source_reference: str,
    heading_path: tuple[str, ...],
) -> str:
    """Build a citation whose fragment is the innermost heading.

    The innermost heading is the most specific true statement about where a
    passage lives. The full path is kept on the chunk itself for display.
    """
    fragment = heading_path[-1] if heading_path else None
    return build_citation(
        source_type=source_type, source_reference=source_reference, fragment=fragment
    )


def parse_citation(citation: str) -> tuple[str, str, str | None]:
    """Split a citation back into its parts.

    Args:
        citation: A reference produced by :func:`build_citation`.

    Returns:
        tuple: ``(prefix, body, fragment)``, with ``fragment`` ``None`` when
        the citation names a whole source.
    """
    prefix, _, rest = citation.partition(":")
    body, separator, fragment = rest.partition(FRAGMENT_SEPARATOR)
    return prefix, body, fragment if separator else None


__all__ = [
    "CITATION_PREFIXES",
    "build_citation",
    "citation_for_heading_path",
    "citation_prefix",
    "document_slug",
    "parse_citation",
]
