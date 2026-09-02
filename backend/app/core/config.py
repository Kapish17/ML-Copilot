"""Application configuration.

Settings are read from environment variables (see ``.env.example`` at the
repository root). The implementation deliberately relies on the standard
library only; a richer configuration layer will be introduced when the
application actually needs external services.

Every limit and heuristic threshold used by the dataset and experiment services
lives here so that behaviour can be tuned without touching the code that
depends on it. No route file hard-codes a limit of its own.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from app import __version__

BYTES_PER_MB = 1024 * 1024
#: Repository root, derived from this file rather than the working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Upload limits -------------------------------------------------------------
DEFAULT_MAX_UPLOAD_MB = 25
DEFAULT_MAX_DATASET_ROWS = 1_000_000
DEFAULT_MAX_DATASET_COLUMNS = 1_000
#: The dataset formats the API accepts. Each has an adapter in
#: ``app.services.datasets.ingestion``; adding an extension here without an
#: adapter changes nothing, because the registry is the real allowlist.
#: Parquet, SQL, databases, cloud storage and URL ingestion are not implemented.
SUPPORTED_DATASET_EXTENSIONS = (".csv", ".xlsx", ".json")

# Browser access -----------------------------------------------------------
#: Origins the dashboard may be served from. The frontend runs as a separate
#: service, so its requests are cross-origin and a browser blocks them without
#: an explicit allowance. An explicit list, never a wildcard; empty disables
#: the middleware entirely.
DEFAULT_CORS_ALLOW_ORIGINS = ("http://localhost:3000", "http://127.0.0.1:3000")

# Profiling / heuristic thresholds ------------------------------------------
DEFAULT_PROFILE_TOP_VALUES = 10
DEFAULT_HIGH_MISSING_RATIO = 0.40
DEFAULT_HIGH_CARDINALITY_RATIO = 0.50
DEFAULT_ID_UNIQUENESS_RATIO = 0.99
DEFAULT_CATEGORICAL_MAX_UNIQUE_RATIO = 0.50
DEFAULT_MAX_CATEGORICAL_DISTINCT = 50
DEFAULT_MAX_CLASSIFICATION_CLASSES = 20
DEFAULT_IMBALANCE_RATIO = 0.80

# Experiment execution ------------------------------------------------------
#: Where experiment records are stored. Local JSON files; MLflow and any
#: database are not implemented.
DEFAULT_EXPERIMENT_STORE_DIR = PROJECT_ROOT / "ml" / "experiments" / "runs"
#: Experiment execution is synchronous in this commit, so every limit below
#: exists to keep one HTTP request from running unboundedly long.
DEFAULT_MIN_CV_FOLDS = 2
DEFAULT_MAX_CV_FOLDS = 10
DEFAULT_CV_FOLDS = 5
DEFAULT_MAX_CANDIDATE_MODELS = 6
DEFAULT_MAX_EXPERIMENT_ROWS = 200_000
DEFAULT_MAX_EXPERIMENT_FEATURE_COLUMNS = 200
DEFAULT_MAX_EXPERIMENT_TAGS = 10
#: SHAP limits, mirroring ``ml.explainability.config`` defaults.
DEFAULT_EXPLANATION_REFERENCE_ROWS = 200
DEFAULT_EXPLANATION_ROWS = 500
DEFAULT_EXPLANATION_TOP_FEATURES = 50
#: History listing.
DEFAULT_EXPERIMENT_PAGE_LIMIT = 50
DEFAULT_MAX_EXPERIMENT_PAGE_LIMIT = 200
DEFAULT_MAX_COMPARISON_EXPERIMENTS = 25


def _env_origins(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    """Read a comma-separated origin allowlist from the environment.

    Args:
        name: Environment variable name.
        default: Value used when the variable is unset.

    Returns:
        tuple[str, ...]: The configured origins, blanks removed. An explicitly
        empty value means "no cross-origin access", which is a meaningful
        setting rather than a mistake: it is what a deployment serving the
        dashboard from this same origin wants.
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    return tuple(origin.strip() for origin in raw.split(",") if origin.strip())


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

    # Browser access
    cors_allow_origins: tuple[str, ...] = DEFAULT_CORS_ALLOW_ORIGINS

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

    # Experiment execution
    experiment_store_dir: Path = DEFAULT_EXPERIMENT_STORE_DIR
    min_cv_folds: int = DEFAULT_MIN_CV_FOLDS
    max_cv_folds: int = DEFAULT_MAX_CV_FOLDS
    default_cv_folds: int = DEFAULT_CV_FOLDS
    max_candidate_models: int = DEFAULT_MAX_CANDIDATE_MODELS
    max_experiment_rows: int = DEFAULT_MAX_EXPERIMENT_ROWS
    max_experiment_feature_columns: int = DEFAULT_MAX_EXPERIMENT_FEATURE_COLUMNS
    max_experiment_tags: int = DEFAULT_MAX_EXPERIMENT_TAGS

    # Explanation limits
    explanation_reference_rows: int = DEFAULT_EXPLANATION_REFERENCE_ROWS
    explanation_rows: int = DEFAULT_EXPLANATION_ROWS
    explanation_top_features: int = DEFAULT_EXPLANATION_TOP_FEATURES

    # Experiment history
    experiment_page_limit: int = DEFAULT_EXPERIMENT_PAGE_LIMIT
    max_experiment_page_limit: int = DEFAULT_MAX_EXPERIMENT_PAGE_LIMIT
    max_comparison_experiments: int = DEFAULT_MAX_COMPARISON_EXPERIMENTS

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
    store_dir = os.getenv("EXPERIMENT_STORE_DIR", "").strip()
    return Settings(
        app_env=os.getenv("APP_ENV", "development"),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        max_upload_bytes=_env_int("MAX_UPLOAD_MB", DEFAULT_MAX_UPLOAD_MB) * BYTES_PER_MB,
        cors_allow_origins=_env_origins(
            "CORS_ALLOW_ORIGINS", DEFAULT_CORS_ALLOW_ORIGINS
        ),
        max_dataset_rows=_env_int("MAX_DATASET_ROWS", DEFAULT_MAX_DATASET_ROWS),
        max_dataset_columns=_env_int("MAX_DATASET_COLUMNS", DEFAULT_MAX_DATASET_COLUMNS),
        experiment_store_dir=(
            Path(store_dir) if store_dir else DEFAULT_EXPERIMENT_STORE_DIR
        ),
        max_cv_folds=_env_int("MAX_CV_FOLDS", DEFAULT_MAX_CV_FOLDS, minimum=2),
        max_candidate_models=_env_int(
            "MAX_CANDIDATE_MODELS", DEFAULT_MAX_CANDIDATE_MODELS
        ),
        max_experiment_rows=_env_int(
            "MAX_EXPERIMENT_ROWS", DEFAULT_MAX_EXPERIMENT_ROWS
        ),
        explanation_rows=_env_int("EXPLANATION_ROWS", DEFAULT_EXPLANATION_ROWS),
    )
