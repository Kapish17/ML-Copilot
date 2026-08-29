"""Turning sources into indexable documents.

``documentation``  project Markdown, from an explicit allowlist
``experiments``    stored ``ExperimentRun`` records, rendered as facts

Both produce :class:`~rag.documents.Document` objects and nothing else, so the
indexer downstream does not know or care where a document came from. Adding a
source — an ML reference text, a notebook export — means writing another
loader here.

The dependency runs one way: ingestion reads the experiment store, and nothing
in ``ml/`` knows this package exists.
"""

from rag.ingestion.documentation import (
    discover_documentation_paths,
    load_document,
    load_documentation,
)
from rag.ingestion.experiments import (
    ExperimentStoreLike,
    experiment_metadata,
    experiment_to_document,
    load_experiments,
    render_experiment,
)

__all__ = [
    "ExperimentStoreLike",
    "discover_documentation_paths",
    "experiment_metadata",
    "experiment_to_document",
    "load_document",
    "load_documentation",
    "load_experiments",
    "render_experiment",
]
