"""The four tools: what they return, and what they refuse.

Each tool is tested against a double for the service it wraps, so a failure
here is a failure of the tool rather than of profiling, training or ranking.
The integration tests exercise the real services separately.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from agent.errors import ToolValidationError
from agent.observations import ensure_json_safe
from agent.tests.factories import (
    FakeArtifacts,
    FakeExecutor,
    FakeProfiler,
    FakeRetrieval,
    FakeRetrievalResult,
    FakeStore,
    experiment_payload,
)
from agent.tools.artifacts import ExperimentArtifactCache
from agent.tools.datasets import DatasetProfileTool, InMemoryDatasetSource
from agent.tools.experiments import RunExperimentTool
from agent.tools.explainability import (
    REASON_NOT_PERSISTED,
    REASON_NOT_RECORDED,
    REASON_UNKNOWN_EXPERIMENT,
    SOURCE_RECOMPUTED,
    SOURCE_STORED_RECORD,
    ExplainExperimentTool,
)
from agent.tools.knowledge import SearchKnowledgeTool

MODELS = ("logistic_regression", "random_forest_classifier")
METRICS = ("f1", "accuracy")


# ---------------------------------------------------------------------------
# dataset_profile
# ---------------------------------------------------------------------------


@pytest.fixture
def profile_tool(dataset_source: InMemoryDatasetSource) -> DatasetProfileTool:
    """The profiling tool over a fixed profile."""
    return DatasetProfileTool(dataset_source, FakeProfiler())


def test_the_profile_tool_returns_shape_task_and_quality(
    profile_tool: DatasetProfileTool,
) -> None:
    """The fields a planner needs to decide whether to run an experiment."""
    output = profile_tool.run({"dataset": "sales", "target_column": "churned"}).output

    assert output["rows"] == 240
    assert output["columns"] == 4
    assert output["inferred_task"] == "classification"
    assert output["target"]["name"] == "churned"
    assert output["quality_issue_count"] == 1
    assert output["features"][0]["name"] == "income"


def test_the_profile_tool_says_when_no_task_could_be_inferred(
    dataset_source: InMemoryDatasetSource,
) -> None:
    """"No target given" and "task unknown" are different things."""
    payload = FakeProfiler().profile_frame(None).model_dump()
    payload["target"] = None
    tool = DatasetProfileTool(dataset_source, FakeProfiler(payload))

    output = tool.run({"dataset": "sales"}).output

    assert output["target"] is None
    assert output["inferred_task"] is None
    assert "no target column" in output["inferred_task_reason"].lower()


def test_the_profile_tool_returns_no_dataframe(
    profile_tool: DatasetProfileTool,
) -> None:
    """The output is plain values, serialisable as it stands."""
    output = profile_tool.run({"dataset": "sales"}).output

    json.dumps(output)
    assert "DataFrame" not in json.dumps(output)


@pytest.mark.parametrize(
    "requested",
    [
        "../../etc/passwd",
        "/etc/shadow",
        "C:\\Users\\me\\.env",
        "~/.aws/credentials",
        "file:///etc/passwd",
        "http://169.254.169.254/latest/meta-data/",
        "sales.csv",
        "SALES",
    ],
)
def test_a_dataset_can_only_be_named_never_located(
    profile_tool: DatasetProfileTool, requested: str
) -> None:
    """A path is not a name, so there is no path handling to get wrong.

    Every one of these fails validation for the same dull reason: it is not a
    registered dataset name.
    """
    with pytest.raises(ToolValidationError):
        profile_tool.schema.validate({"dataset": requested})


def test_the_profile_tool_passes_the_target_to_the_service(
    dataset_source: InMemoryDatasetSource,
) -> None:
    """And the dataset name as a display label, never as a path."""
    profiler = FakeProfiler()
    tool = DatasetProfileTool(dataset_source, profiler)

    tool.run({"dataset": "sales", "target_column": "churned"})

    assert profiler.calls == [{"filename": "sales", "target_column": "churned"}]


def test_a_dataset_removed_after_validation_is_reported_unavailable() -> None:
    """A structured result, not an exception."""
    source = InMemoryDatasetSource({"sales": object()})
    tool = DatasetProfileTool(source, FakeProfiler())
    source._datasets.clear()  # noqa: SLF001 - simulating a race, deliberately

    result = tool.run({"dataset": "sales"})

    assert result.available is False
    assert result.reason == "unknown_dataset"


# ---------------------------------------------------------------------------
# run_experiment
# ---------------------------------------------------------------------------


@pytest.fixture
def experiment_tool(
    dataset_source: InMemoryDatasetSource, executor: FakeExecutor
) -> RunExperimentTool:
    """The experiment tool over a fake runner."""
    return RunExperimentTool(
        dataset_source,
        executor,
        available_models=MODELS,
        available_metrics=METRICS,
    )


def test_the_experiment_tool_summarises_the_stored_record(
    experiment_tool: RunExperimentTool,
) -> None:
    """Identity, winner, scores and headline importances."""
    output = experiment_tool.run(
        {"dataset": "sales", "target_column": "churned"}
    ).output

    assert output["experiment_id"] == "exp_20260101T000000Z_abc123"
    assert output["selected_model"] == "random_forest_classifier"
    assert output["primary_metric"] == "f1"
    assert output["primary_metric_value"] == 0.86
    assert output["top_features"][0]["feature"] == "income"
    assert len(output["candidates"]) == 2


def test_the_experiment_tool_passes_only_declared_options(
    dataset_source: InMemoryDatasetSource,
) -> None:
    """What the planner chose reaches the runner, and nothing else does."""
    executor = FakeExecutor()
    tool = RunExperimentTool(
        dataset_source, executor, available_models=MODELS, available_metrics=METRICS
    )

    tool.run(
        {
            "dataset": "sales",
            "target_column": "churned",
            "models": ["logistic_regression"],
            "primary_metric": "f1",
            "folds": 5,
            "name": "baseline",
        }
    )

    assert executor.calls == [
        {
            "dataset_label": "sales",
            "target_column": "churned",
            "primary_metric": "f1",
            "folds": 5,
            "name": "baseline",
            "models": ("logistic_regression",),
        }
    ]


@pytest.mark.parametrize(
    "models",
    [
        ["sklearn.ensemble.RandomForestClassifier"],
        ["__import__('os').system"],
        ["my_custom_estimator"],
        ["xgboost"],
        ["lightgbm"],
    ],
)
def test_only_models_the_system_already_supports_can_be_requested(
    experiment_tool: RunExperimentTool, models: list[str]
) -> None:
    """A dotted path, a custom class and an uninstalled library all fail.

    The allowed values come from the model registry, so "what the agent may
    train" and "what the system supports" cannot drift apart.
    """
    with pytest.raises(ToolValidationError):
        experiment_tool.schema.validate({"dataset": "sales", "models": models})


@pytest.mark.parametrize(
    "arguments",
    [
        {"dataset": "sales", "folds": 1},
        {"dataset": "sales", "folds": 500},
        {"dataset": "sales", "primary_metric": "made_up_metric"},
        {"dataset": "sales", "models": ["logistic_regression"] * 9},
        {"dataset": "sales", "random_state": 0},
        {"dataset": "sales", "estimator": "RandomForestClassifier()"},
        {"dataset": "sales", "hyperparameters": {"n_estimators": 500}},
        {"dataset": "sales", "preprocessing": "custom"},
    ],
)
def test_unsafe_or_undeclared_experiment_configuration_is_refused(
    experiment_tool: RunExperimentTool, arguments: dict[str, Any]
) -> None:
    """The configuration surface is short, and everything outside it fails."""
    with pytest.raises(ToolValidationError):
        experiment_tool.schema.validate(arguments)


def test_the_experiment_tool_retains_artifacts_only_when_asked(
    dataset_source: InMemoryDatasetSource,
) -> None:
    """With a cache the fitted objects are kept; without one they are not."""
    cache = ExperimentArtifactCache()
    artifacts = FakeArtifacts()
    executor = FakeExecutor(artifacts=artifacts)

    with_cache = RunExperimentTool(
        dataset_source, executor, available_models=MODELS, artifacts=cache
    )
    output = with_cache.run({"dataset": "sales"}).output

    assert executor.calls[-1]["retain_artifacts"] is True
    assert output["explainable_now"] is True
    assert cache.get("exp_20260101T000000Z_abc123") is artifacts

    without_cache = RunExperimentTool(
        dataset_source, FakeExecutor(), available_models=MODELS
    )
    assert without_cache.run({"dataset": "sales"}).output["explainable_now"] is False


def test_the_experiment_output_holds_no_live_objects(
    experiment_tool: RunExperimentTool,
) -> None:
    """A record summary, serialisable as it stands."""
    output = experiment_tool.run({"dataset": "sales"}).output

    json.dumps(ensure_json_safe(output))


# ---------------------------------------------------------------------------
# search_knowledge
# ---------------------------------------------------------------------------


@pytest.fixture
def search_tool(retrieval: FakeRetrieval) -> SearchKnowledgeTool:
    """The search tool over two fixed passages."""
    return SearchKnowledgeTool(
        retrieval,
        max_top_k=10,
        max_query_length=2_000,
        source_types=("project_documentation", "experiment"),
    )


def test_the_search_tool_returns_evidence_with_citation_ids(
    search_tool: SearchKnowledgeTool,
) -> None:
    """The identifiers are the point; they bound what an answer may cite."""
    result = search_tool.run({"query": "How is leakage prevented?"})

    assert result.output["result_count"] == 2
    assert result.citations == (
        "docs:ml-readme#cross-validation",
        "docs:ml-readme#leakage",
    )
    assert result.output["results"][0]["source_reference"] == "ml/README.md"


def test_the_search_tool_returns_no_embeddings(
    search_tool: SearchKnowledgeTool,
) -> None:
    """Text, scores and attribution — never a vector."""
    payload = json.dumps(search_tool.run({"query": "anything"}).output)

    for forbidden in ("embedding", "vector", "ndarray", "array("):
        assert forbidden not in payload.lower()


def test_an_empty_search_is_a_result_not_a_failure(
    dataset_source: InMemoryDatasetSource,
) -> None:
    """"Nothing matched" is a truthful answer the planner must be able to see."""
    tool = SearchKnowledgeTool(FakeRetrieval([]))

    result = tool.run({"query": "something nobody wrote about"})

    assert result.available is True
    assert result.output["status"] == "no_results"
    assert result.citations == ()


def test_a_long_passage_is_truncated_rather_than_carried_whole() -> None:
    """One document must not be able to spend the whole context budget."""
    tool = SearchKnowledgeTool(
        FakeRetrieval([FakeRetrievalResult("docs:x#y", content="a" * 10_000)]),
        max_passage_chars=500,
    )

    passage = tool.run({"query": "x"}).output["results"][0]

    assert passage["truncated"] is True
    assert len(passage["content"]) <= 500


def test_search_limits_come_from_the_retrieval_configuration(
    retrieval: FakeRetrieval,
) -> None:
    """So a planner and an HTTP client are held to the same rules."""
    tool = SearchKnowledgeTool(retrieval, max_top_k=5, max_query_length=50)

    with pytest.raises(ToolValidationError):
        tool.schema.validate({"query": "x", "top_k": 6})
    with pytest.raises(ToolValidationError):
        tool.schema.validate({"query": "x" * 51})


def test_the_search_tool_cannot_be_asked_to_modify_anything(
    search_tool: SearchKnowledgeTool,
) -> None:
    """There is no argument that names a write, because there is no write."""
    for arguments in (
        {"query": "x", "index": True},
        {"query": "x", "delete": "docs:ml-readme#leakage"},
        {"query": "x", "path": "/etc/passwd"},
        {"query": "x", "document": "new content"},
    ):
        with pytest.raises(ToolValidationError):
            search_tool.schema.validate(arguments)

    assert search_tool.schema.field_names() == ("query", "top_k", "source_types")


# ---------------------------------------------------------------------------
# explain_experiment
# ---------------------------------------------------------------------------


def test_an_experiment_from_this_session_is_explained_live() -> None:
    """The fitted model is still in memory, so SHAP runs for real."""
    cache = ExperimentArtifactCache()
    cache.put("exp_1", FakeArtifacts())
    captured: dict[str, Any] = {}

    def explain_global(model: Any, X: Any, y: Any, **kwargs: Any) -> Any:
        captured["called"] = True
        return _Explanation(
            {
                "status": "available",
                "method": "shap",
                "model_name": "random_forest_classifier",
                "task_type": "classification",
                "sample_count": 180,
                "feature_importances": [
                    {"feature": "income", "importance": 0.4, "rank": 1}
                ],
            }
        )

    tool = ExplainExperimentTool(artifacts=cache, explain_global=explain_global)
    output = tool.run({"experiment_id": "exp_1", "scope": "global"}).output

    assert captured["called"] is True
    assert output["source"] == SOURCE_RECOMPUTED
    assert output["feature_importances"][0]["feature"] == "income"
    assert "not causation" in output["interpretation_note"]


def test_a_historical_experiment_reports_its_recorded_importances() -> None:
    """Real numbers, produced when the run happened, labelled as such."""
    payload = experiment_payload()
    tool = ExplainExperimentTool(lookup=FakeStore({payload["experiment_id"]: payload}))

    output = tool.run({"experiment_id": payload["experiment_id"]}).output

    assert output["source"] == SOURCE_STORED_RECORD
    assert output["feature_importances"][0]["feature"] == "income"
    assert any("not persisted" in warning for warning in output["warnings"])


def test_a_historical_prediction_explanation_is_unavailable() -> None:
    """It needs the estimator, and the estimator was never written down."""
    payload = experiment_payload()
    tool = ExplainExperimentTool(
        lookup=FakeStore({payload["experiment_id"]: payload}),
        artifacts=ExperimentArtifactCache(),
    )

    result = tool.run(
        {"experiment_id": payload["experiment_id"], "scope": "prediction"}
    )

    assert result.available is False
    assert result.reason == REASON_NOT_PERSISTED
    assert "requires the fitted model" in result.output["message"]
    # Nothing was invented in place of the missing explanation.
    assert "feature_contributions" not in result.output


def test_an_experiment_that_recorded_no_explanation_is_unavailable() -> None:
    """No stored summary and no model means nothing honest to return."""
    payload = experiment_payload()
    payload["explainability"] = {}
    tool = ExplainExperimentTool(lookup=FakeStore({payload["experiment_id"]: payload}))

    result = tool.run({"experiment_id": payload["experiment_id"]})

    assert result.available is False
    assert result.reason == REASON_NOT_RECORDED


def test_an_unknown_experiment_is_reported_as_such() -> None:
    """Not confused with "the model is gone"."""
    tool = ExplainExperimentTool(lookup=FakeStore({}))

    result = tool.run({"experiment_id": "exp_does_not_exist"})

    assert result.available is False
    assert result.reason == REASON_UNKNOWN_EXPERIMENT


def test_an_unexplainable_model_is_passed_through_as_unavailable() -> None:
    """The explainability layer's own structured refusal is not overwritten."""
    cache = ExperimentArtifactCache()
    cache.put("exp_1", FakeArtifacts())

    tool = ExplainExperimentTool(
        artifacts=cache,
        explain_global=lambda *args, **kwargs: _Explanation(
            {"status": "unavailable", "reason": "unsupported_estimator"}
        ),
    )
    result = tool.run({"experiment_id": "exp_1"})

    assert result.available is False
    assert result.reason == "unsupported_estimator"


class _Explanation:
    """An explanation object shaped like the explainability layer's own."""

    def __init__(self, payload: dict[str, Any]) -> None:
        """Hold the payload."""
        self._payload = payload

    def as_dict(self) -> dict[str, Any]:
        """Render the payload."""
        return self._payload


# ---------------------------------------------------------------------------
# The artifact cache
# ---------------------------------------------------------------------------


def test_the_artifact_cache_evicts_the_oldest_entry() -> None:
    """A long-lived orchestrator cannot accumulate fitted models."""
    cache = ExperimentArtifactCache(max_entries=2)
    cache.put("a", FakeArtifacts())
    cache.put("b", FakeArtifacts())
    cache.put("c", FakeArtifacts())

    assert cache.experiment_ids() == ("b", "c")
    assert cache.get("a") is None


def test_the_artifact_cache_answers_none_for_anything_it_never_held() -> None:
    """Which is what makes the historical case honest rather than a crash."""
    cache = ExperimentArtifactCache()

    assert cache.get("exp_from_last_week") is None
    assert "exp_from_last_week" not in cache
