"""Tests for the experiment service layer, independent of HTTP.

Two things are checked here that a request-level test cannot show.

**The runner is a library.** It takes a DataFrame and a plain options object
and returns a structured result, so the same operation an HTTP route performs
can be performed by a script or, later, by an agent tool — without FastAPI,
without a file, and without knowing where records are kept.

**The dependency direction holds.** ``ml/`` must not import FastAPI, and the
experiment store must not either. That is what makes the API an adapter around
the ML engine rather than a layer tangled into it.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from app.core.config import Settings
from app.services.datasets import DatasetProfilingService
from app.services.experiments import (
    ExperimentHistoryService,
    ExperimentOptions,
    ExperimentRunner,
    run_experiment,
)
from ml.errors import ConfigurationError, InvalidFoldCountError
from ml.experiments import LocalExperimentStore
from tests.factories import learnable_classification_csv, regression_csv

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings whose experiment store is a fresh temporary directory."""
    return Settings(experiment_store_dir=tmp_path / "runs")


@pytest.fixture
def runner(settings: Settings) -> ExperimentRunner:
    """A runner wired to a temporary store and the dataset service."""
    return ExperimentRunner(
        settings,
        LocalExperimentStore(settings.experiment_store_dir),
        DatasetProfilingService(settings),
    )


def frame_from(content: bytes) -> pd.DataFrame:
    """Parse test CSV bytes into the standardised representation."""
    from io import BytesIO

    return pd.read_csv(BytesIO(content))


# --------------------------------------------------------------------------
# Options validation
# --------------------------------------------------------------------------


def test_blank_values_are_treated_as_absent(settings: Settings) -> None:
    """A form sends empty strings; they are not choices."""
    options = ExperimentOptions(
        target_column="  ", name="", tags=("", " baseline ")
    ).validated(settings)

    assert options.target_column is None
    assert options.name is None
    assert options.tags == ("baseline",)


def test_duplicate_models_collapse(settings: Settings) -> None:
    """Asking for one model twice is asking for it once."""
    options = ExperimentOptions(
        models=("logistic_regression", "logistic_regression")
    ).validated(settings)

    assert options.models == ("logistic_regression",)


@pytest.mark.parametrize(
    "options",
    [
        ExperimentOptions(strategy="bootstrap"),
        ExperimentOptions(test_size=0.99),
        ExperimentOptions(random_state=-1),
        ExperimentOptions(max_categorical_cardinality=0),
        ExperimentOptions(excluded_columns=("a",), identifier_columns=("a",)),
        ExperimentOptions(tags=tuple(f"tag{index}" for index in range(50))),
    ],
)
def test_unusable_options_are_refused(
    settings: Settings, options: ExperimentOptions
) -> None:
    """Everything checkable without the data is checked before any work."""
    with pytest.raises(ConfigurationError):
        options.validated(settings)


def test_a_fold_count_outside_the_configured_range_is_refused(
    settings: Settings,
) -> None:
    """The limit comes from settings, not from a constant in a route."""
    with pytest.raises(InvalidFoldCountError) as exc_info:
        ExperimentOptions(folds=settings.max_cv_folds + 1).validated(settings)

    assert exc_info.value.details["max_folds"] == settings.max_cv_folds


def test_only_explicit_preprocessing_choices_become_overrides(
    settings: Settings,
) -> None:
    """Anything the caller left alone stays as the profile inferred it."""
    options = ExperimentOptions(scaling_strategy="none", random_state=3).validated(
        settings
    )

    assert options.preprocessing_overrides == {
        "scaling_strategy": "none",
        "random_state": 3,
    }


def test_a_false_flag_is_an_override_not_an_absence(settings: Settings) -> None:
    """``False`` is a decision; only ``None`` means "not specified"."""
    options = ExperimentOptions(add_missing_indicators=False).validated(settings)

    assert options.preprocessing_overrides == {"add_missing_indicators": False}


# --------------------------------------------------------------------------
# The runner as a library
# --------------------------------------------------------------------------


def test_the_runner_consumes_a_dataframe_not_a_file(
    runner: ExperimentRunner,
) -> None:
    """The pipeline is entered with data, never with a path or a format.

    This is what keeps a future Excel, Parquet or SQL adapter a change to
    ingestion alone. CSV is still the only implemented input format.
    """
    frame = frame_from(learnable_classification_csv())
    result = runner.run_frame(
        frame,
        ExperimentOptions(
            target_column="renewed", models=("logistic_regression",), folds=3
        ),
        source_format="parquet",
    )

    assert result.record.dataset.source_format == "parquet"
    assert result.record.dataset.row_count == len(frame)
    assert result.record.selected_model == "logistic_regression"
    assert result.stored is True


def test_the_convenience_function_runs_one_experiment(
    settings: Settings,
) -> None:
    """``run_experiment`` is the shape a future agent tool would call.

    No agent, LLM or RAG integration is implemented; this only asserts that
    the operation is callable without FastAPI, without a file and without
    knowing where records live.
    """
    result = run_experiment(
        frame_from(regression_csv()),
        settings=settings,
        store=LocalExperimentStore(settings.experiment_store_dir),
        dataset_service=DatasetProfilingService(settings),
        target_column="price",
        models=["linear_regression"],
        folds=3,
        name="library call",
    )

    assert result.record.task_type == "regression"
    assert result.record.evaluation.primary_metric == "rmse"
    payload = result.as_dict()
    assert payload["execution"]["mode"] == "synchronous"


def test_a_run_is_readable_through_the_history_service(
    settings: Settings, runner: ExperimentRunner
) -> None:
    """What the runner saved is what the history service reads back."""
    store = LocalExperimentStore(settings.experiment_store_dir)
    result = runner.run_frame(
        frame_from(learnable_classification_csv()),
        ExperimentOptions(
            target_column="renewed", models=("logistic_regression",), folds=3
        ),
    )
    history = ExperimentHistoryService(settings, store)

    assert history.get(result.record.experiment_id).to_dict() == result.record.to_dict()
    assert len(history.list(task_type="classification")) == 1
    assert history.list(task_type="regression") == ()


def test_comparing_fewer_than_two_runs_is_refused(settings: Settings) -> None:
    """The limit lives in the service, not in the route."""
    history = ExperimentHistoryService(
        settings, LocalExperimentStore(settings.experiment_store_dir)
    )

    with pytest.raises(ConfigurationError):
        history.compare(["exp_only_20260101T000000Z_0000"])


def test_the_store_directory_comes_from_settings(settings: Settings) -> None:
    """Nothing hard-codes where records are written."""
    store = LocalExperimentStore(settings.experiment_store_dir)

    assert store.root == settings.experiment_store_dir


# --------------------------------------------------------------------------
# Architecture
# --------------------------------------------------------------------------


def _imported_modules(path: Path) -> set[str]:
    """Return the top-level module names a Python file imports."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


def test_the_ml_layer_does_not_import_the_web_framework() -> None:
    """``ml/`` is a library. It must not know FastAPI or the backend exists.

    The API is an adapter around the ML engine; if this ever fails, the
    dependency has been inverted and the engine can no longer be used, tested
    or replaced on its own.

    ``ml/tests`` is excluded: one integration test deliberately profiles a
    dataset with the real backend service to prove the structural contract
    between the two layers holds. That is a test importing both, not the
    library depending on the framework.
    """
    offenders = {
        str(path.relative_to(REPOSITORY_ROOT)): sorted(
            _imported_modules(path) & {"fastapi", "starlette", "app", "pydantic"}
        )
        for path in (REPOSITORY_ROOT / "ml").rglob("*.py")
        if "tests" not in path.parts
    }
    assert not {path: names for path, names in offenders.items() if names}


def test_the_ml_layer_imports_without_the_backend_installed() -> None:
    """A fresh interpreter can import the whole ML layer on its own."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import ml, ml.experiments, ml.explainability, ml.models.selection; "
            "print('ok')",
        ],
        capture_output=True,
        text=True,
        cwd=REPOSITORY_ROOT,
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_the_experiment_services_do_not_import_the_web_framework() -> None:
    """The application services are drivable from anywhere, not just a route.

    They may use the backend's configuration and errors; they may not depend
    on FastAPI, which is what would tie experiment execution to one transport.
    """
    service_dir = REPOSITORY_ROOT / "backend" / "app" / "services"
    offenders = {
        str(path.relative_to(REPOSITORY_ROOT)): sorted(
            _imported_modules(path) & {"fastapi", "starlette"}
        )
        for path in service_dir.rglob("*.py")
    }
    assert not {path: names for path, names in offenders.items() if names}


def test_the_routes_contain_no_machine_learning() -> None:
    """Route modules orchestrate; they do not compute.

    They may name the registry and the metric tables to *describe* what is
    available, but nothing that fits, splits, scores or explains belongs here.
    """
    api_dir = REPOSITORY_ROOT / "backend" / "app" / "api"
    forbidden = {"sklearn", "shap", "numpy", "pandas"}
    offenders = {
        str(path.relative_to(REPOSITORY_ROOT)): sorted(
            _imported_modules(path) & forbidden
        )
        for path in api_dir.rglob("*.py")
    }
    assert not {path: names for path, names in offenders.items() if names}
