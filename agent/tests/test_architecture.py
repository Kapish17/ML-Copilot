"""The dependency rules, enforced by reading the imports.

The agent orchestrates capabilities it does not own. That is only true while
it stays unable to compute anything itself, and the cheapest way to keep it
true is to fail the build when it grows an import it should not have.

Two directions are checked. The agent must not reach *down* into the
implementations — no web framework, no SDK, no pandas, no scikit-learn, no
SHAP — because a layer that can build a model itself will eventually build one
instead of orchestrating. And the engines must not reach *up* into the agent,
because ``ml/``, ``rag/`` and ``llm/`` are usable and testable without it, and
that is worth keeping.

The collaborators reach the agent through protocols instead. A test here
checks that the real services satisfy them, so "structural typing" is a
verified claim rather than a hopeful one.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
AGENT_ROOT = REPOSITORY_ROOT / "agent"

#: Packages the agent layer must never import directly. The first two would
#: make it a web component or bind it to one vendor; the rest are the
#: computation it exists to delegate.
FORBIDDEN_IN_AGENT: frozenset[str] = frozenset(
    {"fastapi", "starlette", "openai", "pandas", "numpy", "sklearn", "shap"}
)


def module_imports(path: pathlib.Path) -> set[str]:
    """Every top-level package one module imports."""
    tree = ast.parse(path.read_text())
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module.split(".")[0])
    return found


def package_modules(package: str) -> list[pathlib.Path]:
    """Every non-test module of a top-level package."""
    return [
        path
        for path in (REPOSITORY_ROOT / package).rglob("*.py")
        if "tests" not in path.parts and "__pycache__" not in path.parts
    ]


def test_the_agent_layer_is_present() -> None:
    """A guard: an empty sweep would pass every test below vacuously."""
    assert len(package_modules("agent")) >= 12


@pytest.mark.parametrize("path", package_modules("agent"), ids=lambda p: p.name)
def test_the_agent_imports_no_web_sdk_or_ml_package(path: pathlib.Path) -> None:
    """The core rule, checked module by module."""
    offending = module_imports(path) & FORBIDDEN_IN_AGENT

    assert not offending, f"{path.relative_to(REPOSITORY_ROOT)} imports {sorted(offending)}"


def test_the_agent_does_not_import_the_backend() -> None:
    """So exposing it over HTTP later cannot close a loop.

    The services are reached through protocols. The wiring — which real
    service goes where — happens at the call site, which is the backend's job
    and not this package's.
    """
    for path in package_modules("agent"):
        assert "app" not in module_imports(path), path.relative_to(REPOSITORY_ROOT)


def test_the_agent_reaches_generation_only_through_the_provider_abstraction() -> None:
    """It may import ``llm``; it may not import a vendor's SDK."""
    importers = {
        path.name for path in package_modules("agent") if "llm" in module_imports(path)
    }

    assert importers  # it does use the abstraction
    for path in package_modules("agent"):
        assert "openai" not in module_imports(path)


@pytest.mark.parametrize("package", ["ml", "rag", "llm"])
def test_the_engines_do_not_import_the_agent(package: str) -> None:
    """Each stays usable, and testable, with no agent involved."""
    for path in package_modules(package):
        assert "agent" not in module_imports(path), path.relative_to(REPOSITORY_ROOT)


def test_only_the_backend_edge_imports_the_agent() -> None:
    """The backend may use the agent — in a small, named set of places.

    The dependency is one-directional and this test says how far it reaches.
    Wiring belongs in ``api/dependencies.py``, which is the one module that
    knows all five packages exist; the factory takes the two overrides a test
    needs; error translation belongs in ``core/agent_errors.py``, the sibling
    of the ML and knowledge mappings; and the application service holds the
    orchestrator and the budget policy.

    Everything else — the route and the schemas above all — talks to the agent
    through those. The route in particular must not import ``agent``: it
    depends on the application service and on nothing deeper, which is what
    keeps it thin and what would make a change of orchestrator a change to
    these four files.
    """
    importers = {
        path.relative_to(REPOSITORY_ROOT).as_posix()
        for path in package_modules("backend/app")
        if "agent" in module_imports(path)
    }

    assert importers == {
        "backend/app/api/dependencies.py",
        "backend/app/core/agent_errors.py",
        "backend/app/main.py",
        "backend/app/services/agent/budgets.py",
        "backend/app/services/agent/datasets.py",
        "backend/app/services/agent/service.py",
    }, importers


def test_rag_still_does_not_import_the_llm_layer() -> None:
    """Commit 9's rule, re-checked: retrieval works with no model involved."""
    for path in package_modules("rag"):
        assert "llm" not in module_imports(path), path.relative_to(REPOSITORY_ROOT)


def test_the_agent_can_be_imported_without_the_web_layer() -> None:
    """A fresh interpreter, importing only the agent."""
    import subprocess  # noqa: PLC0415 - test-only, never in the package itself
    import sys

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import agent; "
            "import sys; "
            "assert 'fastapi' not in sys.modules; "
            "assert 'openai' not in sys.modules; "
            "assert 'sklearn' not in sys.modules; "
            "assert 'pandas' not in sys.modules; "
            "print(sorted(agent.build_default_registry().names()))",
        ],
        capture_output=True,
        text=True,
        cwd=REPOSITORY_ROOT,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "[]"


def test_the_real_services_satisfy_the_agent_protocols() -> None:
    """Structural typing, verified rather than assumed.

    This is the test that would fail if a service's signature drifted away
    from what the tools expect — which is the risk of using protocols instead
    of imports, and the reason to check it explicitly.
    """
    from app.core.config import Settings
    from app.services.datasets import DatasetProfilingService
    from ml.experiments.local_store import LocalExperimentStore
    from rag.config import RagConfig
    from rag.retrieval import RetrievalService
    from rag.stores import LocalVectorStore

    from agent.tools.base import (
        ExperimentLookup,
        ProfilingService,
        RetrievalLike,
    )

    settings = Settings()
    assert isinstance(DatasetProfilingService(settings), ProfilingService)

    config = RagConfig()
    retrieval = RetrievalService(config, store=LocalVectorStore(config.index_dir))
    assert isinstance(retrieval, RetrievalLike)

    assert isinstance(LocalExperimentStore(settings.experiment_store_dir), ExperimentLookup)


def test_the_experiment_runner_matches_the_executor_the_tool_calls() -> None:
    """The one callable the experiment tool needs, checked by signature."""
    import inspect

    from app.services.experiments.runner import run_experiment

    parameters = inspect.signature(run_experiment).parameters

    for expected in (
        "frame",
        "settings",
        "store",
        "dataset_service",
        "target_column",
        "models",
        "dataset_label",
        "retain_artifacts",
    ):
        assert expected in parameters, expected
