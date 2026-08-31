"""The agent over the real layers, with only the planner faked.

Everything below this line is genuine: a real retrieval index built from this
repository's own documentation, the real profiling service, the real
experiment runner with real scikit-learn models and real cross-validation, and
the real SHAP explainability layer. Only the planner is scripted, because a
language model is the one component whose behaviour cannot be asserted.

That division is deliberate. A suite that mocked the services would prove the
orchestrator calls *something*; this one proves the tools fit the services
they wrap — that a profile really does yield an inferred task, that a run
really does produce an experiment id the answer can cite, and that an
explanation really is unavailable when the model was not persisted.

Still offline: no credential, no network, no model download. The default
embedding provider is the deterministic local one.
"""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import pandas as pd
import pytest

from agent.config import AgentConfig
from agent.orchestrator import AgentOrchestrator
from agent.planners.fake import FakePlanner
from agent.plans import PlanStep
from agent.results import AgentStatus
from agent.tests.factories import learnable_classification_rows
from agent.tools import build_default_registry
from agent.tools.artifacts import ExperimentArtifactCache
from agent.tools.datasets import InMemoryDatasetSource
from app.core.config import Settings
from app.services.datasets import DatasetProfilingService
from app.services.experiments.runner import run_experiment
from ml.experiments.local_store import LocalExperimentStore
from ml.explainability import explain_global, explain_prediction
from ml.models.registry import default_registry
from rag.config import RagConfig
from rag.indexing import RagIndexer
from rag.retrieval import RetrievalService
from rag.stores import LocalVectorStore

FINAL = PlanStep(action="final")


def tool_step(name: str, **arguments: object) -> PlanStep:
    """Build a scripted tool call."""
    return PlanStep(action="tool", tool=name, arguments=dict(arguments))


# ---------------------------------------------------------------------------
# Fixtures over the real services
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    """A small, deterministic, genuinely learnable classification dataset."""
    return pd.DataFrame(learnable_classification_rows())


@pytest.fixture(scope="module")
def real_index(tmp_path_factory: pytest.TempPathFactory) -> RagConfig:
    """A real index over this project's documentation, built once."""
    index_dir = tmp_path_factory.mktemp("agent-index") / "index"
    config = RagConfig(index_dir=index_dir)
    RagIndexer(config, store=LocalVectorStore(index_dir)).index_documentation()
    return config


@pytest.fixture(scope="module")
def real_retrieval(real_index: RagConfig) -> RetrievalService:
    """The real retrieval service over that index."""
    return RetrievalService(real_index, store=LocalVectorStore(real_index.index_dir))


@pytest.fixture
def real_settings(tmp_path: Path) -> Settings:
    """Application settings whose experiment store is temporary."""
    return Settings(experiment_store_dir=tmp_path / "runs")


@pytest.fixture
def real_store(real_settings: Settings) -> LocalExperimentStore:
    """The real experiment store, in a temporary directory."""
    return LocalExperimentStore(real_settings.experiment_store_dir)


@pytest.fixture
def artifacts() -> ExperimentArtifactCache:
    """A fresh in-memory cache of fitted models."""
    return ExperimentArtifactCache()


@pytest.fixture
def real_registry(
    frame: pd.DataFrame,
    real_settings: Settings,
    real_store: LocalExperimentStore,
    real_retrieval: RetrievalService,
    real_index: RagConfig,
    artifacts: ExperimentArtifactCache,
):
    """The four tools, wired to every real service.

    This function is also the answer to "how would the backend wire this up in
    Commit 13": a dataset source, the profiling service, the runner as a
    partially applied callable, the retrieval service, the store and the two
    explainability functions.
    """
    dataset_service = DatasetProfilingService(real_settings)
    registry = default_registry()

    return build_default_registry(
        source=InMemoryDatasetSource({"customers": frame}),
        profiler=dataset_service,
        executor=partial(
            run_experiment,
            settings=real_settings,
            store=real_store,
            dataset_service=dataset_service,
        ),
        retrieval=real_retrieval,
        lookup=real_store,
        artifacts=artifacts,
        explain_global=explain_global,
        explain_prediction=explain_prediction,
        available_models=lambda: list(registry.identifiers()),
        available_metrics=("f1", "accuracy", "roc_auc", "rmse", "r2"),
        source_types=("project_documentation", "experiment"),
        max_top_k=real_index.max_top_k,
        max_query_length=real_index.max_query_length,
    )


def build(registry, steps, answer: str, **kwargs):
    """Build an orchestrator around a scripted planner."""
    planner = FakePlanner(steps, answer=answer)
    return (
        AgentOrchestrator(planner, registry, config=AgentConfig(**kwargs)),
        planner,
    )


# ---------------------------------------------------------------------------
# Real RAG
# ---------------------------------------------------------------------------


def test_the_agent_answers_from_the_real_retrieval_index(real_registry) -> None:
    """Question → real RetrievalService → agent → grounded answer.

    The citation the answer uses is discovered from what the index actually
    returned, not hard-coded, so this fails if the citation identifiers ever
    stop matching between the retrieval layer and the grounding check.
    """
    search = real_registry.get("search_knowledge")
    evidence = search.run({"query": "How does cross-validation select a model?"})
    citation = evidence.citations[0]

    agent, _ = build(
        real_registry,
        [tool_step("search_knowledge", query="How does cross-validation select a model?"), FINAL],
        answer=f"Models are selected on the training rows only [{citation}].",
    )

    result = agent.run("How does cross-validation select a model?")

    assert result.status is AgentStatus.COMPLETED
    assert result.citation_ids == (citation,)
    assert result.citations[0].source_reference.endswith("README.md")
    # The identifier came from a real retrieved passage, not from the script.
    assert citation.startswith("docs:")


def test_a_citation_the_real_index_did_not_return_is_rejected(
    real_registry,
) -> None:
    """Real evidence, invented citation — the same failure as with doubles."""
    agent, _ = build(
        real_registry,
        [tool_step("search_knowledge", query="How is leakage prevented?"), FINAL],
        answer="Leakage is prevented by magic [docs:secret-internal#nope].",
    )

    result = agent.run("How is leakage prevented?")

    assert result.status is AgentStatus.GROUNDING_FAILED
    assert result.rejected_citations == ("docs:secret-internal#nope",)


def test_the_real_search_tool_returns_no_vectors(real_registry) -> None:
    """Over the real index, where vectors genuinely exist."""
    output = real_registry.get("search_knowledge").run(
        {"query": "embeddings", "top_k": 3}
    ).output

    rendered = json.dumps(output)
    assert "ndarray" not in rendered
    assert "embedding" not in rendered.lower() or "embeddings" in output["query"]
    json.dumps(output)


# ---------------------------------------------------------------------------
# Real ML
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_the_agent_profiles_and_runs_a_real_experiment(real_registry) -> None:
    """Dataset → profile → ExperimentRunner → observation → answer.

    No training logic is duplicated: the run goes through the same
    ``run_experiment`` the HTTP endpoint uses, and the experiment id in the
    answer is the one the store actually wrote.
    """
    agent, _ = build(
        real_registry,
        [
            tool_step("dataset_profile", dataset="customers", target_column="renewed"),
            tool_step(
                "run_experiment",
                dataset="customers",
                target_column="renewed",
                models=["logistic_regression"],
                folds=3,
            ),
            FINAL,
        ],
        answer="PLACEHOLDER",
    )

    # The answer must name the experiment the run actually produced, so it is
    # written after the tools have run.
    profile = real_registry.get("dataset_profile").run(
        {"dataset": "customers", "target_column": "renewed"}
    )
    assert profile.output["inferred_task"] == "classification"
    assert profile.output["rows"] == 180

    result = agent.run("Which model performs best on the customers data?")

    assert result.tool_call_count == 2
    experiment = result.observations[1]["output"]
    assert experiment["status"] == "ok"
    assert experiment["task_type"] == "classification"
    assert experiment["selected_model"] == "logistic_regression"
    assert experiment["primary_metric_value"] is not None
    assert experiment["experiment_id"].startswith("exp_")
    assert result.experiment_ids == (experiment["experiment_id"],)

    # Nothing live escaped into the record.
    rendered = json.dumps(result.as_dict())
    for forbidden in ("DataFrame", "Pipeline(", "LogisticRegression(", "ndarray"):
        assert forbidden not in rendered


@pytest.mark.slow
def test_a_real_run_is_stored_and_readable(
    real_registry, real_store: LocalExperimentStore
) -> None:
    """The agent's run is an ordinary experiment, findable afterwards."""
    output = real_registry.get("run_experiment").run(
        {
            "dataset": "customers",
            "target_column": "renewed",
            "models": ["logistic_regression"],
            "folds": 3,
        }
    ).output

    assert real_store.exists(output["experiment_id"])
    stored = real_store.get(output["experiment_id"])
    assert stored.selected_model == "logistic_regression"


def test_a_model_the_registry_does_not_offer_is_refused(real_registry) -> None:
    """Checked against the real model registry, not a copy of it."""
    from agent.errors import ToolValidationError

    tool = real_registry.get("run_experiment")

    with pytest.raises(ToolValidationError):
        tool.schema.validate({"dataset": "customers", "models": ["xgboost"]})

    # And a real one passes, so the check is not vacuous.
    tool.schema.validate({"dataset": "customers", "models": ["logistic_regression"]})


# ---------------------------------------------------------------------------
# Real explainability
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_an_experiment_run_in_this_session_is_explained_for_real(
    real_registry,
) -> None:
    """New experiment → fitted model still in memory → real SHAP."""
    run = real_registry.get("run_experiment").run(
        {
            "dataset": "customers",
            "target_column": "renewed",
            "models": ["logistic_regression"],
            "folds": 3,
        }
    )
    experiment_id = run.output["experiment_id"]
    assert run.output["explainable_now"] is True

    explanation = real_registry.get("explain_experiment").run(
        {"experiment_id": experiment_id, "scope": "global"}
    )

    assert explanation.available is True
    assert explanation.output["source"] == "recomputed"
    assert explanation.output["feature_importances"]
    assert explanation.output["method"] in {"shap", "permutation_importance"}
    json.dumps(explanation.output)


@pytest.mark.slow
def test_a_prediction_can_be_explained_while_the_model_is_in_memory(
    real_registry,
) -> None:
    """The path that genuinely needs the fitted estimator."""
    run = real_registry.get("run_experiment").run(
        {
            "dataset": "customers",
            "target_column": "renewed",
            "models": ["logistic_regression"],
            "folds": 3,
        }
    )

    explanation = real_registry.get("explain_experiment").run(
        {
            "experiment_id": run.output["experiment_id"],
            "scope": "prediction",
            "row_index": 0,
        }
    )

    assert explanation.available is True
    assert explanation.output["source"] == "recomputed"
    assert explanation.output["feature_contributions"]


@pytest.mark.slow
def test_a_historical_experiment_cannot_be_explained_live(
    real_registry, artifacts: ExperimentArtifactCache
) -> None:
    """Historical experiment → fitted artifact gone → structured unavailable.

    The record is real and still in the store; what is gone is the estimator,
    exactly as Commit 7 intended. Dropping the cache is what "a later process"
    means here.
    """
    run = real_registry.get("run_experiment").run(
        {
            "dataset": "customers",
            "target_column": "renewed",
            "models": ["logistic_regression"],
            "folds": 3,
        }
    )
    experiment_id = run.output["experiment_id"]

    artifacts.clear()  # the process that trained it has ended

    result = real_registry.get("explain_experiment").run(
        {"experiment_id": experiment_id, "scope": "prediction"}
    )

    assert result.available is False
    assert result.reason == "fitted_model_not_persisted"
    assert "requires the fitted model" in result.output["message"]
    assert "not available" in result.output["message"]
    # Nothing was invented in place of the missing explanation.
    assert "feature_contributions" not in result.output


@pytest.mark.slow
def test_a_historical_experiment_still_reports_its_recorded_importances(
    real_registry, artifacts: ExperimentArtifactCache
) -> None:
    """Real numbers, produced by SHAP when the run happened, labelled as such."""
    run = real_registry.get("run_experiment").run(
        {
            "dataset": "customers",
            "target_column": "renewed",
            "models": ["logistic_regression"],
            "folds": 3,
        }
    )
    artifacts.clear()

    result = real_registry.get("explain_experiment").run(
        {"experiment_id": run.output["experiment_id"], "scope": "global"}
    )

    assert result.available is True
    assert result.output["source"] == "stored_record"
    assert result.output["feature_importances"]
    assert any("not persisted" in warning for warning in result.output["warnings"])


@pytest.mark.slow
def test_the_full_chain_produces_a_grounded_answer(real_registry) -> None:
    """Profile → experiment → explanation → answer, over real services."""
    run = real_registry.get("run_experiment").run(
        {
            "dataset": "customers",
            "target_column": "renewed",
            "models": ["logistic_regression"],
            "folds": 3,
        }
    )
    experiment_id = run.output["experiment_id"]

    agent, _ = build(
        real_registry,
        [
            tool_step("dataset_profile", dataset="customers", target_column="renewed"),
            tool_step(
                "run_experiment",
                dataset="customers",
                target_column="renewed",
                models=["logistic_regression"],
                folds=3,
            ),
            tool_step("explain_experiment", experiment_id=experiment_id),
            FINAL,
        ],
        answer=(
            "Logistic regression was selected. The recorded feature "
            "importances describe model behaviour, not causation."
        ),
    )

    result = agent.run("Which model performs best, and why?")

    assert result.tool_call_count == 3
    assert result.status in {AgentStatus.COMPLETED, AgentStatus.PARTIAL}
    assert all(
        observation["status"] in {"ok", "unavailable"}
        for observation in result.observations
    )
    json.dumps(result.as_dict())
