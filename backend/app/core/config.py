"""Application configuration.

Settings are read from environment variables (see ``.env.example`` at the
repository root). The implementation deliberately relies on the standard
library only; a richer configuration layer will be introduced when the
application actually needs external services.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from app import __version__


@dataclass(frozen=True)
class Settings:
    """Immutable runtime settings for the backend service."""

    app_name: str = "ML Copilot API"
    app_version: str = __version__
    app_env: str = "development"
    log_level: str = "INFO"


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
    )
