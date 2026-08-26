"""Shared FastAPI dependencies.

Services are built per request from the active settings. Resolving settings
through a dependency keeps them overridable in tests without touching the
process environment.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from app.core.config import Settings, get_settings
from app.services.datasets import DatasetProfilingService

SettingsDep = Annotated[Settings, Depends(get_settings)]


def get_dataset_service(settings: SettingsDep) -> DatasetProfilingService:
    """Provide a dataset profiling service bound to the active settings."""
    return DatasetProfilingService(settings)


DatasetServiceDep = Annotated[DatasetProfilingService, Depends(get_dataset_service)]
