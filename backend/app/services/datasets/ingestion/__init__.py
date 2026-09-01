"""Dataset ingestion: turning an uploaded file of some format into a DataFrame.

::

    file format
        |
        v
    format detection      detection.py   — which adapter should try these bytes
        |
        v
    format adapter        *_adapter.py   — the one format-specific parse
        |
        v
    standardised DataFrame               — where formats stop existing
        |
        v
    DatasetProfilingService / ExperimentRunner / agent / SHAP

This package is the *only* place in the application that knows about file
formats. Above it, a route hands over bytes and a filename. Below it,
everything works on a pandas DataFrame and cannot tell CSV from a spreadsheet
from JSON — which is what makes the ML layer, the agent, the retrieval layer
and the explainability layer format-agnostic by construction rather than by
convention.

Three formats are implemented: ``.csv``, ``.xlsx`` and ``.json``. Parquet, SQL,
databases, Google Sheets, S3 and URL ingestion are **not implemented**.
"""

from app.services.datasets.ingestion.base import (
    BaseDatasetAdapter,
    DatasetAdapter,
    DatasetMetadata,
    IngestedDataset,
)
from app.services.datasets.ingestion.csv_adapter import CSVAdapter
from app.services.datasets.ingestion.detection import detect_format, supported_formats
from app.services.datasets.ingestion.excel_adapter import ExcelAdapter
from app.services.datasets.ingestion.formats import (
    EXTENSIONS,
    MEDIA_TYPES,
    DatasetFormat,
)
from app.services.datasets.ingestion.json_adapter import JSONAdapter
from app.services.datasets.ingestion.registry import (
    DatasetAdapterRegistry,
    default_registry,
)

__all__ = [
    "BaseDatasetAdapter",
    "CSVAdapter",
    "DatasetAdapter",
    "DatasetAdapterRegistry",
    "DatasetFormat",
    "DatasetMetadata",
    "EXTENSIONS",
    "ExcelAdapter",
    "IngestedDataset",
    "JSONAdapter",
    "MEDIA_TYPES",
    "default_registry",
    "detect_format",
    "supported_formats",
]
