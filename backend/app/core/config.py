"""Application configuration.

Settings are read from environment variables (see ``.env.example`` at the
repository root). The implementation deliberately relies on the standard
library only; a richer configuration layer will be introduced when the
application actually needs external services.

Every limit and heuristic threshold used by the dataset service lives here so
that behaviour can be tuned without touching the code that depends on it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from app import __version__

BYTES_PER_MB = 1024 * 1024

# Upload limits -------------------------------------------------------------
DEFAULT_MAX_UPLOAD_MB = 25
DEFAULT_MAX_DATASET_ROWS = 1_000_000
DEFAULT_MAX_DATASET_COLUMNS = 1_000
SUPPORTED_DATASET_EXTENSIONS = (".csv",)

# Profiling / heuristic thresholds ------------------------------------------
DEFAULT_PROFILE_TOP_VALUES = 10
DEFAULT_HIGH_MISSING_RATIO = 0.40
DEFAULT_HIGH_CARDINALITY_RATIO = 0.50
DEFAULT_ID_UNIQUENESS_RATIO = 0.99
DEFAULT_CATEGORICAL_MAX_UNIQUE_RATIO = 0.50
DEFAULT_MAX_CATEGORICAL_DISTINCT = 50
DEFAULT_MAX_CLASSIFICATION_CLASSES = 20
DEFAULT_IMBALANCE_RATIO = 0.80


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    """Read a positive integer from the environment.

    Args:
        name: Environment variable name.
        default: Value used when the variable is unset or blank.
        minimum: Smallest accepted value.

    Returns:
        int: The parsed value.

    Raises:
        ValueError: If the variable is set to something unusable. Failing at
            startup is preferred over silently running with a wrong limit.
    """
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


@dataclass(frozen=True)
class Settings:
    """Immutable runtime settings for the backend service."""

    # Application
    app_name: str = "ML Copilot API"
    app_version: str = __version__
    app_env: str = "development"
    log_level: str = "INFO"

    # Dataset upload limits
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_MB * BYTES_PER_MB
    max_dataset_rows: int = DEFAULT_MAX_DATASET_ROWS
    max_dataset_columns: int = DEFAULT_MAX_DATASET_COLUMNS
    supported_dataset_extensions: tuple[str, ...] = SUPPORTED_DATASET_EXTENSIONS

    # Profiling behaviour
    profile_top_values: int = DEFAULT_PROFILE_TOP_VALUES
    categorical_max_unique_ratio: float = DEFAULT_CATEGORICAL_MAX_UNIQUE_RATIO
    max_categorical_distinct: int = DEFAULT_MAX_CATEGORICAL_DISTINCT

    # Data-quality heuristics
    high_missing_ratio: float = DEFAULT_HIGH_MISSING_RATIO
    high_cardinality_ratio: float = DEFAULT_HIGH_CARDINALITY_RATIO
    id_uniqueness_ratio: float = DEFAULT_ID_UNIQUENESS_RATIO

    # Target analysis heuristics
    max_classification_classes: int = DEFAULT_MAX_CLASSIFICATION_CLASSES
    imbalance_ratio: float = DEFAULT_IMBALANCE_RATIO

    @property
    def max_upload_mb(self) -> float:
        """Upload limit expressed in megabytes, for user-facing messages."""
        return self.max_upload_bytes / BYTES_PER_MB


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings.

    Returns:
        Settings: Configuration resolved from the process environment,
        falling back to development-friendly defaults.
    """
    return Settings(
        app_env=os.getenv("APP_ENV", "development"),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        max_upload_bytes=_env_int("MAX_UPLOAD_MB", DEFAULT_MAX_UPLOAD_MB) * BYTES_PER_MB,
        max_dataset_rows=_env_int("MAX_DATASET_ROWS", DEFAULT_MAX_DATASET_ROWS),
        max_dataset_columns=_env_int("MAX_DATASET_COLUMNS", DEFAULT_MAX_DATASET_COLUMNS),
    )
