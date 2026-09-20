"""Tests for dataset/experiment knowledge: indexing, retrieval, isolation.

A finished experiment is now added to the same retrieval index the Knowledge
Assistant already searched for project documentation — this is what makes
"which features mattered in my experiment?" answerable at all. These tests
read like ``test_api_knowledge.py``'s (a real index, a scripted fake model)
because that is exactly the path being extended, not replaced.

Three things matter more than the rest:

**Isolation.** Two experiments run on two different datasets must never
answer for each other. ``experiment_id`` and ``dataset_fingerprint`` are the
mechanism — both are metadata on every chunk a run produces, and the
retrieval layer applies a metadata filter *before* ranking. If isolation ever
regressed, the sharpest failure would be silent: a question about dataset A
answered, convincingly, from dataset B.

**No raw row ever indexed.** The architecture requirement behind this feature
is "structured knowledge, not a CSV chat". A sentinel value planted in one raw
cell and never repeated anywhere in the derived record — not a column name,
not a metric, not a decision — must not be retrievable, because if it were,
something upstream started indexing the dataset itself rather than the
record ``ml/experiments`` already produces about it.

**Indexing failure does not fail a run.** An experiment is complete and
already saved by the time it would be indexed; if indexing raises, the run
must still be reported as successful, with a warning, exactly like every
other best-effort step in this pipeline (SHAP, the model artifact).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.services.datasets import DatasetProfilingService
from app.services.experiments.options import ExperimentOptions
from app.services.experiments.runner import ExperimentRunner
from llm.config import LLMConfig
from llm.providers.fake import FakeLLMProvider
from ml.experiments import LocalExperimentStore
from rag.config import RagConfig
from rag.indexing import RagIndexer
from rag.retrieval import RetrievalService
from rag.stores import LocalVectorStore
from tests.factories import build_csv, experiment_form, upload_payload

RUN_URL = "/api/v1/experiments/run"
SEARCH_URL = "/api/v1/search"
ASK_URL = "/api/v1/ask"

#: Planted as one raw *numeric* cell value in the "customers" dataset.
#: ``render_experiment`` never writes a raw cell value into the document —
#: only column names, dtypes, decisions, metrics and rounded scores — so this
#: exact number appearing in indexed content would mean something started
#: embedding the dataset's own values rather than the derived record.
#:
#: Deliberately not a categorical value: a categorical column's *levels*
#: legitimately appear in the record, one-hot-encoded into a feature name
#: such as ``plan_gold`` — that is expected feature-engineering output, not
#: raw-row leakage, and testing against a categorical level would conflate
#: the two.
CUSTOMER_SENTINEL = "86675309"


def customers_csv() -> bytes:
    """A small, learnable classification dataset with a planted sentinel."""
    header = ["income", "tenure_months", "plan", "renewed"]
    body: list[list[Any]] = []
    for index in range(120):
        income = int(CUSTOMER_SENTINEL) if index == 0 else 30_000 + (index % 12) * 4_000
        tenure = 1 + (index % 48)
        plan = "gold" if index % 2 else "silver"
        renewed = int((income > 45_000 and tenure > 12) or index % 11 == 0)
        body.append([income, tenure, plan, renewed])
    return build_csv(header, body)


def houses_csv() -> bytes:
    """A small, unrelated regression dataset — a different fingerprint."""
    header = ["size_sqm", "rooms", "district", "price"]
    body: list[list[Any]] = []
    for index in range(120):
        size = 40.0 + (index % 50) * 2.0
        rooms = 1 + (index % 5)
        district = ["north", "south", "centre"][index % 3]
        price = round(1_400 * size + 8_500 * rooms + 20 * index, 2)
        body.append([size, rooms, district, price])
    return build_csv(header, body)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def knowledge_index(tmp_path_factory: pytest.TempPathFactory) -> RagConfig:
    """A real index over the project documentation, isolated to a temp dir.

    Experiments are added to *this* index by the tests below, so it must not
    be the repository's own ``rag/index`` — the same reason
    ``experiment_store_dir`` in ``conftest.py`` isolates the experiment store.
    """
    index_dir = tmp_path_factory.mktemp("dataset-knowledge-index") / "index"
    config = RagConfig(index_dir=index_dir)
    RagIndexer(config, store=LocalVectorStore(index_dir)).index_documentation()
    return config


@pytest.fixture(scope="module")
def experiment_settings(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    """An isolated experiment store, shared by every test in this module."""
    store_dir = tmp_path_factory.mktemp("dataset-knowledge-experiments")
    return Settings(experiment_store_dir=store_dir)


@pytest.fixture(scope="module")
def client(
    knowledge_index: RagConfig, experiment_settings: Settings
) -> Iterator[TestClient]:
    """A client wired to the isolated index and store, with no credential.

    No language-model provider is configured, so ``/api/v1/ask`` is
    unavailable here — this client is for the indexing and search tests,
    which need none. ``ask_client`` below adds a scripted provider.
    """
    application = create_app(experiment_settings, rag_config=knowledge_index)
    with TestClient(application) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def customer_experiment(client: TestClient) -> dict[str, Any]:
    """One completed experiment on the customers dataset."""
    response = client.post(
        RUN_URL,
        files=upload_payload(customers_csv(), "customers.csv"),
        data=experiment_form(
            target_column="renewed",
            models=["logistic_regression"],
            folds=3,
            name="customer renewal",
        ),
    )
    assert response.status_code == 200, response.text
    return response.json()


@pytest.fixture(scope="module")
def house_experiment(client: TestClient) -> dict[str, Any]:
    """One completed experiment on a different, unrelated dataset."""
    response = client.post(
        RUN_URL,
        files=upload_payload(houses_csv(), "houses.csv"),
        data=experiment_form(
            target_column="price",
            models=["linear_regression"],
            folds=3,
            name="house price",
        ),
    )
    assert response.status_code == 200, response.text
    return response.json()


#: Top-level request fields, as opposed to ``filters.*`` metadata.
_REQUEST_FIELDS = {"top_k", "similarity_threshold"}


def search(
    client: TestClient,
    query: str,
    *,
    source_types: list[str] | None = None,
    **fields: Any,
) -> dict[str, Any]:
    """POST a filtered search, returning the parsed response."""
    body: dict[str, Any] = {"query": query, "top_k": 10}
    filters: dict[str, Any] = {}
    for key, value in fields.items():
        (body if key in _REQUEST_FIELDS else filters)[key] = value
    if source_types is not None:
        filters["source_types"] = source_types
    if filters:
        body["filters"] = filters
    response = client.post(SEARCH_URL, json=body)
    assert response.status_code == 200, response.text
    return response.json()


# ---------------------------------------------------------------------------
# A finished experiment becomes searchable
# ---------------------------------------------------------------------------


def test_a_finished_experiment_is_indexed_automatically(
    client: TestClient, customer_experiment: dict[str, Any]
) -> None:
    """No separate indexing call — running the experiment is enough."""
    experiment_id = customer_experiment["experiment_id"]

    response = search(
        client,
        "renewal experiment",
        source_types=["experiment"],
        experiment_id=experiment_id,
    )

    assert response["result_count"] > 0
    assert all(
        result["source_reference"] == experiment_id for result in response["results"]
    )


def test_the_indexed_knowledge_covers_what_was_asked_for(
    client: TestClient, customer_experiment: dict[str, Any]
) -> None:
    """Dataset shape, model, and evaluation are all in the retrievable text.

    This is the "useful representation" requirement made concrete: a reader
    who retrieves this experiment's chunks can answer every example question
    from the feature request — columns, model, score, cross-validation — from
    the text alone.
    """
    experiment_id = customer_experiment["experiment_id"]
    response = search(
        client, "columns model score", experiment_id=experiment_id, top_k=10
    )
    content = "\n".join(result["content"] for result in response["results"])

    for expected in (
        "income",
        "tenure_months",
        "logistic_regression",
        "Selection score",
        customer_experiment["dataset"]["fingerprint"],
    ):
        assert expected in content, f"expected {expected!r} in indexed content"


def test_no_raw_dataset_value_is_ever_indexed(
    client: TestClient, customer_experiment: dict[str, Any]
) -> None:
    """The planted sentinel cell value must not be retrievable from anywhere.

    Column *names* and dtypes are part of the structured record and are
    expected to appear (covered above); a raw cell value is not part of that
    record at all, so its presence here would mean something started
    embedding the dataset itself rather than the derived knowledge document.
    """
    experiment_id = customer_experiment["experiment_id"]
    # A broad query at the lowest possible threshold, so this reads every
    # chunk this experiment produced rather than only whatever happens to
    # rank near the sentinel itself.
    response = search(
        client,
        "experiment dataset model",
        experiment_id=experiment_id,
        similarity_threshold=-1.0,
        top_k=50,
    )

    assert response["result_count"] > 0
    for result in response["results"]:
        assert CUSTOMER_SENTINEL not in result["content"]


# ---------------------------------------------------------------------------
# Isolation between experiments and datasets
# ---------------------------------------------------------------------------


def test_filtering_by_experiment_id_excludes_every_other_experiment(
    client: TestClient,
    customer_experiment: dict[str, Any],
    house_experiment: dict[str, Any],
) -> None:
    """Session/experiment isolation: A's filter never returns B's content."""
    customer_id = customer_experiment["experiment_id"]
    house_id = house_experiment["experiment_id"]
    assert customer_id != house_id

    response = search(
        client,
        "model performance",
        source_types=["experiment"],
        experiment_id=customer_id,
    )

    assert response["result_count"] > 0
    for result in response["results"]:
        assert result["source_reference"] == customer_id
        assert house_id not in result["content"]
        assert "size_sqm" not in result["content"]
        assert "district" not in result["content"]


def test_filtering_by_dataset_fingerprint_isolates_the_same_way(
    client: TestClient,
    customer_experiment: dict[str, Any],
    house_experiment: dict[str, Any],
) -> None:
    """Broader than ``experiment_id``: every run on the same data, no other."""
    customer_fingerprint = customer_experiment["dataset"]["fingerprint"]
    house_fingerprint = house_experiment["dataset"]["fingerprint"]
    assert customer_fingerprint != house_fingerprint

    response = search(
        client,
        "dataset profile",
        source_types=["experiment"],
        dataset_fingerprint=house_fingerprint,
    )

    assert response["result_count"] > 0
    for result in response["results"]:
        assert result["metadata"]["dataset_fingerprint"] == house_fingerprint
        assert "income" not in result["content"]
        assert "tenure_months" not in result["content"]


def test_documentation_search_is_unaffected_by_experiment_indexing(
    client: TestClient,
    customer_experiment: dict[str, Any],
    house_experiment: dict[str, Any],
) -> None:
    """Regression: the original project-documentation RAG still works.

    Indexing two experiments must not change what a documentation-scoped
    search returns, and no experiment's content leaks into it.
    """
    response = search(
        client,
        "How does the project prevent data leakage?",
        source_types=["project_documentation"],
    )

    assert response["result_count"] > 0
    for result in response["results"]:
        assert result["source_type"] == "project_documentation"
        assert "logistic_regression" not in result["content"]
        assert customer_experiment["experiment_id"] not in result["content"]


# ---------------------------------------------------------------------------
# Grounded, isolated answers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def customer_citation(
    knowledge_index: RagConfig, customer_experiment: dict[str, Any]
) -> str:
    """A real citation for the customer experiment, to script the fake model."""
    retriever = RetrievalService(
        knowledge_index, store=LocalVectorStore(knowledge_index.index_dir)
    )
    response = retriever.search(
        "which model was selected",
        top_k=5,
        equals={"experiment_id": customer_experiment["experiment_id"]},
    )
    assert response.results, "the customer experiment must be indexed by now"
    return response.results[0].citation


def test_a_question_scoped_to_one_experiment_is_answered_from_it(
    experiment_settings: Settings,
    knowledge_index: RagConfig,
    customer_experiment: dict[str, Any],
    customer_citation: str,
) -> None:
    """A grounded answer, citing only the experiment it was scoped to."""
    provider = FakeLLMProvider(
        responses=(
            "Logistic regression was selected for the renewal experiment "
            f"[{customer_citation}]."
        )
    )
    application = create_app(
        experiment_settings,
        rag_config=knowledge_index,
        llm_config=LLMConfig(provider="fake", model="fake-model", temperature=0.0),
        llm_provider=provider,
    )
    with TestClient(application) as ask_client:
        response = ask_client.post(
            ASK_URL,
            json={
                "question": "Which model was selected for my experiment?",
                "filters": {"experiment_id": customer_experiment["experiment_id"]},
            },
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "grounded"
    assert payload["is_grounded"] is True
    assert all(
        citation["source_reference"] == customer_experiment["experiment_id"]
        for citation in payload["citations"]
    )


def test_a_question_scoped_to_a_nonexistent_experiment_is_not_hallucinated(
    experiment_settings: Settings,
    knowledge_index: RagConfig,
    customer_experiment: dict[str, Any],
) -> None:
    """Grounding over isolation: no evidence in scope, no invented answer.

    The model is scripted to answer as if it *did* have evidence — proving
    that a fabricated-sounding response is refused because nothing was
    retrieved for this filter, not because the fake happened to decline.
    """
    provider = FakeLLMProvider(
        responses=("The model achieved 99% accuracy on this experiment.",)
    )
    application = create_app(
        experiment_settings,
        rag_config=knowledge_index,
        llm_config=LLMConfig(provider="fake", model="fake-model", temperature=0.0),
        llm_provider=provider,
    )
    with TestClient(application) as ask_client:
        response = ask_client.post(
            ASK_URL,
            json={
                "question": "How well did the model perform?",
                "filters": {"experiment_id": "exp_does_not_exist_00000000T000000Z"},
            },
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "insufficient_evidence"
    assert payload["is_grounded"] is False


# ---------------------------------------------------------------------------
# Indexing failure does not fail the run
# ---------------------------------------------------------------------------


class _BrokenIndexer:
    """Something wired as a knowledge indexer that always fails."""

    def index_documents(self, documents: Any, *, force: bool = False) -> None:
        raise RuntimeError("the index volume is not writable in this test")


def test_a_failing_indexer_does_not_fail_the_experiment(tmp_path: Path) -> None:
    """The run is saved and returned even when indexing raises.

    Mirrors how a failed SHAP explanation is handled: the caller gets a
    successful result plus a warning, never a failed run for a step that is
    not on the critical path of "did the experiment complete".
    """
    settings = Settings(experiment_store_dir=tmp_path / "runs")
    runner = ExperimentRunner(
        settings,
        LocalExperimentStore(settings.experiment_store_dir),
        DatasetProfilingService(settings),
        knowledge_indexer=_BrokenIndexer(),
    )
    header = ["income", "tenure_months", "renewed"]
    rows = [[30_000 + i * 100, i % 36, i % 3 == 0] for i in range(60)]

    result = runner.run_content(
        "customers.csv",
        build_csv(header, rows),
        ExperimentOptions(
            target_column="renewed", models=("logistic_regression",)
        ).validated(settings),
    )

    assert result.stored is True
    assert any("Knowledge Assistant" in warning for warning in result.warnings)


def test_no_indexer_means_no_indexing_and_no_error(tmp_path: Path) -> None:
    """The default before this feature: a run with nothing wired in at all."""
    settings = Settings(experiment_store_dir=tmp_path / "runs")
    runner = ExperimentRunner(
        settings,
        LocalExperimentStore(settings.experiment_store_dir),
        DatasetProfilingService(settings),
    )
    header = ["income", "tenure_months", "renewed"]
    rows = [[30_000 + i * 100, i % 36, i % 3 == 0] for i in range(60)]

    result = runner.run_content(
        "customers.csv",
        build_csv(header, rows),
        ExperimentOptions(
            target_column="renewed", models=("logistic_regression",)
        ).validated(settings),
    )

    assert result.stored is True
    assert not any("Knowledge Assistant" in warning for warning in result.warnings)
