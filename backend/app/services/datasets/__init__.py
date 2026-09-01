"""Dataset ingestion and profiling.

The workflow is split so each step can be tested on its own:

``validation``  upload-level checks (extension, size, emptiness)
``ingestion``   format detection and the per-format adapters (CSV, Excel, JSON)
``loader``      decoding and parsing CSV bytes into a DataFrame
``profiler``    dataset- and column-level statistics
``quality``     heuristic data-quality findings
``target``      analysis of an optional target column
``service``     orchestration used by the API layer
"""

from app.services.datasets.service import DatasetProfilingService, LoadedDataset

__all__ = ["DatasetProfilingService", "LoadedDataset"]
