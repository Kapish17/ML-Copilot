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

# API authentication --------------------------------------------------------
#: **Off by default, and that is a deliberate choice rather than an oversight.**
#: `docker compose up --build` has to bring up a working system with no secret
#: to configure, because that is what makes this project demonstrable. Turning
#: authentication on is one environment variable; leaving it off is a local
#: tool listening on loopback, which is what the Compose file publishes.
DEFAULT_API_AUTH_ENABLED = False
#: Shortest key accepted when authentication is enabled. Not a strength
#: estimate — a long key can still be guessable — but it refuses the failure
#: mode that actually happens: someone types `API_AUTH_KEY=test` to get past a
#: start-up error and leaves it there. 32 characters is what
#: `secrets.token_urlsafe(24)` produces, which is what the documentation tells
#: people to run.
MIN_API_AUTH_KEY_LENGTH = 32

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
#: Where the winning fitted model of each run is written, so it can be
#: predicted from later. Separate from the run records because it holds a
#: different kind of thing: binary pickles rather than readable JSON, larger,
#: and — because unpickling executes code — a directory that must be treated as
#: executable and written to by nothing but this application. See the trust
#: boundary at the top of `ml/artifacts/store.py`.
DEFAULT_MODEL_ARTIFACT_DIR = PROJECT_ROOT / "ml" / "experiments" / "models"
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
#: Most records one prediction request may carry. Prediction is far cheaper
#: than training, but it is still synchronous and still holds a worker, so it
#: is bounded like every other path that does real work.
DEFAULT_MAX_PREDICTION_RECORDS = 500
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


#: What counts as "on". Spelled out rather than "anything but empty", so
#: `API_AUTH_ENABLED=false` cannot switch authentication on, and a typo cannot
#: switch it *off* silently either — an unrecognised value is an error.
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off", ""})


def _env_bool(name: str, default: bool) -> bool:
    """Read a boolean environment variable.

    Args:
        name: Variable to read.
        default: Value used when the variable is unset or blank.

    Returns:
        bool: The parsed value.

    Raises:
        ValueError: If the variable holds something that is neither true nor
            false. A security switch must never be decided by a guess: with
            `API_AUTH_ENABLED=ture`, refusing to start is the safe outcome and
            quietly serving unauthenticated traffic is not.
    """
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    if raw in _TRUE_VALUES:
        return True
    if raw in _FALSE_VALUES:
        return False
    raise ValueError(
        f"{name} must be one of {sorted(_TRUE_VALUES | _FALSE_VALUES - {''})}, "
        f"got {raw!r}"
    )


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

    # API authentication
    api_auth_enabled: bool = DEFAULT_API_AUTH_ENABLED
    #: The expected bearer token. Empty unless authentication is enabled, and
    #: **read from the environment only** — there is no default, none is
    #: generated, and nothing writes it anywhere. It never reaches a response
    #: body, a log line, an experiment record, a RAG document or the OpenAPI
    #: schema; `backend/tests/test_authentication.py` asserts each of those.
    api_auth_key: str = ""

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
    model_artifact_dir: Path = DEFAULT_MODEL_ARTIFACT_DIR
    max_prediction_records: int = DEFAULT_MAX_PREDICTION_RECORDS
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

    def __post_init__(self) -> None:
        """Refuse a configuration that would run unprotected while claiming not to.

        The dangerous state is ``API_AUTH_ENABLED=true`` with no key. Whatever
        the intent, the outcome would be an operator who believes the service
        is protected. Two ways to resolve that are both wrong: generating a key
        would produce a secret nobody knows and every restart would change it,
        and falling back to a built-in default would ship a password that is in
        the source of a public repository.

        So this fails, here, before the application is built — which means
        `uvicorn` exits with the reason on stderr rather than serving traffic.
        The check lives on the dataclass rather than in :func:`get_settings`
        so it holds for every ``Settings`` ever constructed, including the ones
        tests build by hand.

        Raises:
            ValueError: If authentication is enabled without a usable key.
        """
        if not self.api_auth_enabled:
            return
        key = self.api_auth_key
        if not key.strip():
            raise ValueError(
                "API_AUTH_ENABLED is true but API_AUTH_KEY is empty. Set a key, "
                "or set API_AUTH_ENABLED=false. No key is generated and there "
                "is no default."
            )
        if len(key) < MIN_API_AUTH_KEY_LENGTH:
            raise ValueError(
                f"API_AUTH_KEY must be at least {MIN_API_AUTH_KEY_LENGTH} "
                "characters. Generate one with: "
                "python -c \"import secrets; print(secrets.token_urlsafe(24))\""
            )

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
    artifact_dir = os.getenv("MODEL_ARTIFACT_DIR", "").strip()
    return Settings(
        app_env=os.getenv("APP_ENV", "development"),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        api_auth_enabled=_env_bool("API_AUTH_ENABLED", DEFAULT_API_AUTH_ENABLED),
        # Stripped, because a `.env` file routinely leaves a trailing newline
        # or a stray space and a credential that fails to match for that reason
        # is an afternoon nobody gets back.
        api_auth_key=os.getenv("API_AUTH_KEY", "").strip(),
        max_upload_bytes=_env_int("MAX_UPLOAD_MB", DEFAULT_MAX_UPLOAD_MB) * BYTES_PER_MB,
        cors_allow_origins=_env_origins(
            "CORS_ALLOW_ORIGINS", DEFAULT_CORS_ALLOW_ORIGINS
        ),
        max_dataset_rows=_env_int("MAX_DATASET_ROWS", DEFAULT_MAX_DATASET_ROWS),
        max_dataset_columns=_env_int("MAX_DATASET_COLUMNS", DEFAULT_MAX_DATASET_COLUMNS),
        experiment_store_dir=(
            Path(store_dir) if store_dir else DEFAULT_EXPERIMENT_STORE_DIR
        ),
        model_artifact_dir=(
            Path(artifact_dir) if artifact_dir else DEFAULT_MODEL_ARTIFACT_DIR
        ),
        max_prediction_records=_env_int(
            "MAX_PREDICTION_RECORDS", DEFAULT_MAX_PREDICTION_RECORDS
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
