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
from app.services.experiments import ExperimentHistoryService, ExperimentRunner
from ml.experiments import LocalExperimentStore
from ml.experiments.store import ExperimentStore

SettingsDep = Annotated[Settings, Depends(get_settings)]


def get_dataset_service(settings: SettingsDep) -> DatasetProfilingService:
    """Provide a dataset profiling service bound to the active settings."""
    return DatasetProfilingService(settings)


DatasetServiceDep = Annotated[DatasetProfilingService, Depends(get_dataset_service)]


def get_experiment_store(settings: SettingsDep) -> ExperimentStore:
    """Provide the experiment store the service layer should use.

    The return type is the storage *interface*, so a future MLflow or database
    backend is a change to this one function. **Only the local JSON store is
    implemented.** Constructing it has no side effect; the directory is created
    on the first write.
    """
    return LocalExperimentStore(settings.experiment_store_dir)


ExperimentStoreDep = Annotated[ExperimentStore, Depends(get_experiment_store)]


def get_experiment_runner(
    settings: SettingsDep,
    store: ExperimentStoreDep,
    datasets: DatasetServiceDep,
) -> ExperimentRunner:
    """Provide an experiment runner wired to its collaborators."""
    return ExperimentRunner(settings, store, datasets)


ExperimentRunnerDep = Annotated[ExperimentRunner, Depends(get_experiment_runner)]


def get_experiment_history(
    settings: SettingsDep, store: ExperimentStoreDep
) -> ExperimentHistoryService:
    """Provide the experiment history service for the active store."""
    return ExperimentHistoryService(settings, store)


ExperimentHistoryDep = Annotated[
    ExperimentHistoryService, Depends(get_experiment_history)
]
