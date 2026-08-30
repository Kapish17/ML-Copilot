"""Turning a request's filter fields into the retrieval layer's own filter.

One translation, in one place. The API exposes named fields because a typed
request is easier to document and validate than a free-form mapping, but they
are converted straight into
:func:`~rag.retrieval.service.build_metadata_filter` and handed to the
retrieval service — **no filtering happens in the API layer.**

That matters for a reason beyond tidiness. The retrieval layer applies a
filter *before* ranking, so asking for the five best classification
experiments searches classification experiments. A filter applied afterwards
in a route would rank everything and then discard most of it, silently
returning two results when five were asked for. Re-implementing it here would
reintroduce exactly that bug.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from rag.documents import SourceType
from rag.errors import ConfigurationError
from rag.retrieval import build_metadata_filter
from rag.stores import MetadataFilter

#: The metadata a caller may filter on, in the order they are documented.
#: Everything here is a key an ingestion adapter puts on a chunk, so the
#: retrieval layer already knows how to match it.
FILTERABLE_FIELDS: tuple[str, ...] = (
    "task_type",
    "dataset_fingerprint",
    "target_column",
    "selected_model",
    "primary_metric",
    "experiment_id",
)

#: Source types a caller may restrict to. An open vocabulary in ``rag/``, but
#: the API validates against the ones that exist so a typo is an error rather
#: than a silently empty result.
KNOWN_SOURCE_TYPES: tuple[str, ...] = tuple(item.value for item in SourceType)


def validate_source_types(source_types: Sequence[str]) -> tuple[str, ...]:
    """Check requested source types against the vocabulary that exists.

    Args:
        source_types: What the caller asked for.

    Returns:
        tuple[str, ...]: The trimmed, de-duplicated values.

    Raises:
        ConfigurationError: If one is not a known source type. Answering an
            empty result to a misspelled filter would look like "nothing
            matched" and send the caller looking in the wrong place.
    """
    cleaned = tuple(dict.fromkeys(value.strip() for value in source_types if value.strip()))
    unknown = sorted(set(cleaned) - set(KNOWN_SOURCE_TYPES))
    if unknown:
        raise ConfigurationError(
            "Unknown source type(s): "
            + ", ".join(unknown)
            + ". Available: "
            + ", ".join(KNOWN_SOURCE_TYPES)
            + ".",
            details={"source_types": unknown, "available": list(KNOWN_SOURCE_TYPES)},
        )
    return cleaned


def build_filter(
    *,
    source_types: Sequence[str] = (),
    metadata: Mapping[str, Any] | None = None,
) -> MetadataFilter:
    """Build the retrieval layer's filter from a request's fields.

    Args:
        source_types: Restrict to these kinds of source.
        metadata: Named metadata fields; ``None`` values are ignored, so an
            unset field is not a filter on ``None``.

    Returns:
        MetadataFilter: The retrieval layer's own filter object.

    Raises:
        ConfigurationError: If a source type is unknown.
    """
    values = {
        key: value
        for key, value in (metadata or {}).items()
        if key in FILTERABLE_FIELDS and value is not None and str(value).strip()
    }
    return build_metadata_filter(
        source_types=validate_source_types(source_types), equals=values
    )


__all__ = [
    "FILTERABLE_FIELDS",
    "KNOWN_SOURCE_TYPES",
    "build_filter",
    "validate_source_types",
]
