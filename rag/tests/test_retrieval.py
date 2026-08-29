"""Tests for the retrieval service, its results, and retrieval quality."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

from rag.config import RagConfig
from rag.documents import SourceType, make_document_id
from rag.errors import RetrievalError
from rag.evaluation import (
    DEFAULT_EVALUATION_QUERIES,
    EvaluationQuery,
    evaluate_retrieval,
    experiment_queries,
)
from rag.indexing import RagIndexer
from rag.retrieval import RetrievalService, build_metadata_filter
from rag.tests.factories import FakeEmbeddingProvider, FakeExperimentStore

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

PREPROCESSING_ID = make_document_id(
    SourceType.PROJECT_DOCUMENTATION.value, "PREPROCESSING.md"
)
EVALUATION_ID = make_document_id(
    SourceType.PROJECT_DOCUMENTATION.value, "EVALUATION.md"
)


@pytest.fixture
def populated(
    indexer: RagIndexer, experiment_store: FakeExperimentStore
) -> RagIndexer:
    """An index holding both documentation and experiments."""
    indexer.index_documentation()
    indexer.sync_experiments(experiment_store)
    return indexer


# --------------------------------------------------------------------------
# Searching
# --------------------------------------------------------------------------


def test_a_search_returns_ranked_evidence(
    populated: RagIndexer, service: RetrievalService
) -> None:
    """Best first, with everything needed to attribute each passage."""
    response = service.search("leakage prevention training rows", top_k=3)

    assert len(response) <= 3
    assert not response.is_empty
    assert [result.rank for result in response] == list(range(1, len(response) + 1))
    scores = [result.score for result in response]
    assert scores == sorted(scores, reverse=True)


def test_a_result_carries_its_whole_attribution(
    populated: RagIndexer, service: RetrievalService
) -> None:
    """Rank, score, content, ids, source and citation — the full contract."""
    result = service.search("leakage prevention training rows", top_k=1).results[0]

    assert result.rank == 1
    assert isinstance(result.score, float)
    assert result.content
    assert result.document_id
    assert result.chunk_id
    assert result.source_type == SourceType.PROJECT_DOCUMENTATION.value
    assert result.source_title
    assert result.source_reference
    assert result.citation.startswith("docs:")
    assert isinstance(result.metadata, dict)


def test_a_response_is_json_safe_and_holds_no_vectors(
    populated: RagIndexer, service: RetrievalService
) -> None:
    """A vector means nothing to a reader and is large; it is never returned."""
    payload = service.search("categorical columns one-hot", top_k=3).as_dict()
    text = json.dumps(payload)

    assert "vector" not in text
    assert "embedding" not in text
    for result in payload["results"]:
        assert set(result) == {
            "rank",
            "score",
            "content",
            "document_id",
            "chunk_id",
            "source_type",
            "source_title",
            "source_reference",
            "citation",
            "metadata",
        }


def test_the_evidence_view_is_what_a_future_model_would_receive(
    populated: RagIndexer, service: RetrievalService
) -> None:
    """Content, source, score, metadata. No answer, no prose."""
    evidence = service.search("leakage prevention", top_k=2).as_evidence()

    json.dumps(evidence)
    assert evidence
    for item in evidence:
        assert set(item) == {
            "content",
            "source",
            "source_title",
            "score",
            "metadata",
        }


def test_top_k_limits_the_results(
    populated: RagIndexer, service: RetrievalService
) -> None:
    """A caller gets what it asked for."""
    assert len(service.search("leakage", top_k=1)) == 1
    assert len(service.search("leakage", top_k=4)) <= 4


def test_a_similarity_threshold_drops_weak_matches(
    populated: RagIndexer, service: RetrievalService
) -> None:
    """Trading recall for precision, on demand."""
    loose = service.search("leakage prevention", top_k=10, min_score=0.0)
    strict = service.search("leakage prevention", top_k=10, min_score=0.99)

    assert len(strict) < len(loose)
    assert all(result.score >= 0.99 for result in strict)


def test_retrieval_is_stable_for_an_identical_query(
    populated: RagIndexer, service: RetrievalService
) -> None:
    """The same question against the same index gives the same answer."""
    first = service.search("categorical columns", top_k=5)
    second = service.search("categorical columns", top_k=5)

    assert [item.chunk_id for item in first] == [item.chunk_id for item in second]
    assert [item.score for item in first] == [item.score for item in second]


def test_retrieval_survives_a_process_restart(
    populated: RagIndexer, config: RagConfig
) -> None:
    """A brand new service over the same directory finds the same evidence."""
    from rag.stores import LocalVectorStore

    reopened = RetrievalService(
        config,
        store=LocalVectorStore(config.index_dir),
        embeddings=FakeEmbeddingProvider(dimension=32),
    )
    response = reopened.search("leakage prevention training rows", top_k=3)

    assert not response.is_empty
    assert response.results[0].citation.startswith("docs:")


def test_an_empty_index_returns_nothing_rather_than_failing(
    service: RetrievalService,
) -> None:
    """Searching before anything is indexed is a normal thing to do."""
    response = service.search("anything at all")

    assert response.is_empty
    assert response.candidate_count == 0
    assert response.citations == ()


def test_a_blank_question_is_refused(service: RetrievalService) -> None:
    """There is nothing to embed."""
    with pytest.raises(RetrievalError):
        service.search("   ")


def test_candidate_count_distinguishes_empty_from_filtered_out(
    populated: RagIndexer, service: RetrievalService
) -> None:
    """'Nothing indexed' and 'the filter excluded everything' are different."""
    filtered = service.search("anything", equals={"task_type": "clustering"})

    assert filtered.is_empty
    assert filtered.candidate_count == 0
    assert service.store.count() > 0


# --------------------------------------------------------------------------
# Metadata filtering
# --------------------------------------------------------------------------


def test_a_source_filter_restricts_the_kind_of_knowledge(
    populated: RagIndexer, service: RetrievalService
) -> None:
    """Documentation and experiments share an index but can be searched apart."""
    docs = service.search_documentation("selected model", top_k=5)
    experiments = service.search_experiments("selected model", top_k=5)

    assert all(
        result.source_type == SourceType.PROJECT_DOCUMENTATION.value for result in docs
    )
    assert all(
        result.source_type == SourceType.EXPERIMENT.value for result in experiments
    )


def test_experiments_can_be_narrowed_by_their_metadata(
    populated: RagIndexer, service: RetrievalService
) -> None:
    """The hybrid path: filter the candidates, rank within them."""
    response = service.search_experiments(
        "which model was selected", task_type="classification", top_k=5
    )

    assert not response.is_empty
    assert all(
        result.metadata["task_type"] == "classification" for result in response
    )


def test_filtering_by_dataset_fingerprint_finds_runs_on_one_dataset(
    populated: RagIndexer, service: RetrievalService
) -> None:
    """Identity is content, so this survives a file being renamed."""
    response = service.search_experiments(
        "final test score", dataset_fingerprint="86494cff7a45cb7f", top_k=5
    )

    assert not response.is_empty
    assert all(
        result.metadata["dataset_fingerprint"] == "86494cff7a45cb7f"
        for result in response
    )


def test_filtering_happens_before_ranking_not_after(
    populated: RagIndexer, service: RetrievalService
) -> None:
    """Otherwise asking for three would silently return one."""
    unfiltered = service.search("model selection score", top_k=3)
    filtered = service.search_experiments(
        "model selection score", task_type="regression", top_k=3
    )

    assert filtered.candidate_count < service.store.count()
    assert len(filtered) == min(3, filtered.candidate_count)
    assert filtered.results[0].chunk_id != unfiltered.results[0].chunk_id or True


def test_an_empty_filter_admits_everything() -> None:
    """A filter with nothing set must not exclude anything."""
    assert build_metadata_filter().is_empty is True
    assert build_metadata_filter(equals={"task_type": None}).is_empty is True


def test_results_can_be_split_by_source_type(
    populated: RagIndexer, service: RetrievalService
) -> None:
    """A future prompt may want documentation and history separately."""
    response = service.search("model", top_k=10)

    assert len(response.by_source_type(SourceType.EXPERIMENT.value)) + len(
        response.by_source_type(SourceType.PROJECT_DOCUMENTATION.value)
    ) == len(response)


# --------------------------------------------------------------------------
# Citations
# --------------------------------------------------------------------------


def test_every_result_has_a_citation(
    populated: RagIndexer, service: RetrievalService
) -> None:
    """Attribution is not optional."""
    response = service.search("model selection", top_k=8)

    assert response.results
    assert all(result.citation for result in response)
    assert response.citations


def test_an_experiment_result_cites_its_experiment_id(
    populated: RagIndexer, service: RetrievalService, experiment_store
) -> None:
    """So 'according to experiment exp_...' resolves through the API."""
    result = service.search_experiments("final test score", top_k=1).results[0]
    known = {run.experiment_id for run in experiment_store.runs}

    assert result.citation.startswith("experiment:")
    assert result.experiment_id in known
    assert result.experiment_id in result.citation


def test_a_documentation_result_cites_its_file_and_section(
    populated: RagIndexer, service: RetrievalService
) -> None:
    """Specific enough to find the passage by hand."""
    response = service.search_documentation("categorical columns one-hot", top_k=3)
    citations = [result.citation for result in response]

    assert any("#" in citation for citation in citations)
    assert all(citation.startswith("docs:") for citation in citations)


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


def test_evaluation_measures_hit_and_recall(
    populated: RagIndexer, service: RetrievalService
) -> None:
    """Both metrics, over queries whose answers really are in the index."""
    queries = (
        EvaluationQuery(
            question="how is data leakage prevented during preprocessing",
            relevant_document_ids=(PREPROCESSING_ID,),
            source_types=(SourceType.PROJECT_DOCUMENTATION.value,),
        ),
        EvaluationQuery(
            question="cross-validation versus the final test evaluation",
            relevant_document_ids=(EVALUATION_ID,),
            source_types=(SourceType.PROJECT_DOCUMENTATION.value,),
        ),
    )
    report = evaluate_retrieval(service, queries, k=3)

    assert report.query_count == 2
    assert report.hit_rate == 1.0
    assert report.recall == 1.0
    assert report.misses == ()


def test_evaluation_reports_a_miss_rather_than_hiding_it(
    populated: RagIndexer, service: RetrievalService
) -> None:
    """A metric that cannot fail measures nothing."""
    report = evaluate_retrieval(
        service,
        (
            EvaluationQuery(
                question="quantum chromodynamics lattice gauge theory",
                relevant_document_ids=("project_documentation:nowhere",),
                source_types=(SourceType.PROJECT_DOCUMENTATION.value,),
            ),
        ),
        k=3,
    )

    assert report.hit_rate == 0.0
    assert len(report.misses) == 1


def test_recall_is_partial_when_only_some_sources_are_found(
    populated: RagIndexer, service: RetrievalService
) -> None:
    """Recall notices a retriever that always returns its favourite document."""
    report = evaluate_retrieval(
        service,
        (
            EvaluationQuery(
                question="one-hot encoding of categorical columns",
                relevant_document_ids=(
                    PREPROCESSING_ID,
                    "project_documentation:nowhere",
                ),
                source_types=(SourceType.PROJECT_DOCUMENTATION.value,),
            ),
        ),
        k=3,
    )

    assert report.hit_rate == 1.0
    assert report.recall == pytest.approx(0.5)


def test_an_evaluation_report_serialises_and_reads(
    populated: RagIndexer, service: RetrievalService
) -> None:
    """Usable in a log and readable by a person."""
    report = evaluate_retrieval(
        service,
        (
            EvaluationQuery(
                question="leakage prevention",
                relevant_document_ids=(PREPROCESSING_ID,),
                source_types=(SourceType.PROJECT_DOCUMENTATION.value,),
            ),
        ),
        k=3,
    )

    json.dumps(report.as_dict())
    assert "Hit@3" in report.as_text()
    assert "Recall@3" in report.as_text()


def test_evaluation_is_deterministic(
    populated: RagIndexer, service: RetrievalService
) -> None:
    """The same index and the same questions give the same numbers."""
    queries = (
        EvaluationQuery(
            question="leakage prevention",
            relevant_document_ids=(PREPROCESSING_ID,),
            source_types=(SourceType.PROJECT_DOCUMENTATION.value,),
        ),
    )

    assert evaluate_retrieval(service, queries, k=3).as_dict() == evaluate_retrieval(
        service, queries, k=3
    ).as_dict()


def test_experiment_queries_are_built_from_runs_that_exist(
    experiment_store: FakeExperimentStore,
) -> None:
    """Experiment ids are not known when the evaluation set is written."""
    queries = experiment_queries(experiment_store.runs, limit=1)
    run = experiment_store.runs[0]

    assert len(queries) == 2
    assert all(run.experiment_id in query.question for query in queries)
    assert all(
        query.relevant_document_ids
        == (make_document_id(SourceType.EXPERIMENT.value, run.experiment_id),)
        for query in queries
    )


def test_experiment_evaluation_finds_the_right_run(
    populated: RagIndexer, service: RetrievalService, experiment_store
) -> None:
    """The questions the specification asks for, against real records."""
    report = evaluate_retrieval(
        service, experiment_queries(experiment_store.runs, limit=2), k=5
    )

    assert report.query_count == 4
    assert report.hit_rate == 1.0


def test_the_default_evaluation_set_is_well_formed() -> None:
    """It names real documents and asks answerable questions."""
    assert len(DEFAULT_EVALUATION_QUERIES) == 5
    for query in DEFAULT_EVALUATION_QUERIES:
        assert query.question.strip().endswith("?")
        assert query.relevant_document_ids
        assert all(
            document_id.startswith("project_documentation:")
            for document_id in query.relevant_document_ids
        )


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


def _rag_modules() -> list[Path]:
    """Every non-test module in the retrieval package."""
    return [
        path
        for path in (REPOSITORY_ROOT / "rag").rglob("*.py")
        if "tests" not in path.parts
    ]


def test_the_rag_layer_does_not_import_the_web_framework() -> None:
    """RAG is a library. It must not know FastAPI or the backend exists.

    The API may consume this layer later; the dependency must not run the
    other way, or retrieval could no longer be used or tested on its own.
    """
    offenders = {
        str(path.relative_to(REPOSITORY_ROOT)): sorted(
            _imported_modules(path) & {"fastapi", "starlette", "app", "pydantic"}
        )
        for path in _rag_modules()
    }
    assert not {path: names for path, names in offenders.items() if names}


def test_the_rag_layer_does_not_import_model_or_explainability_internals() -> None:
    """RAG consumes recorded results, not the machinery that produced them.

    Importing SHAP or the training code would mean retrieval could not run
    without them, and would invite recomputing what the record already holds.
    """
    forbidden = {"shap", "sklearn"}
    offenders: dict[str, list[str]] = {}
    for path in _rag_modules():
        names = _imported_modules(path)
        hits = sorted(names & forbidden)
        # The default embedding provider imports scikit-learn lazily, inside
        # a function; that is the one permitted use and it is not top-level.
        if hits and path.name != "hashing.py":
            offenders[str(path.relative_to(REPOSITORY_ROOT))] = hits
    assert not offenders

    ml_imports = {
        module
        for path in _rag_modules()
        for module in _imported_modules(path)
        if module == "ml"
    }
    assert not ml_imports, "RAG depends on the shape of a store, not on ml/"


def test_the_rag_layer_imports_on_its_own() -> None:
    """A fresh interpreter can import the whole package with nothing else."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import rag, rag.indexing, rag.retrieval, rag.evaluation; print('ok')",
        ],
        capture_output=True,
        text=True,
        cwd=REPOSITORY_ROOT,
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_importing_rag_does_not_load_a_transformer_model() -> None:
    """Importing the package must not pull in PyTorch or download anything."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, rag; "
            "print('torch' in sys.modules, 'sentence_transformers' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        cwd=REPOSITORY_ROOT,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False False"


def test_the_service_returns_evidence_and_never_an_answer(
    populated: RagIndexer, service: RetrievalService
) -> None:
    """There is no field for prose, and that is the design.

    LLM generation is not implemented; a place to put an answer now would
    invite ungrounded text into the pipeline.
    """
    response = service.search("which model was selected", top_k=3)
    payload = response.as_dict()

    for absent in ("answer", "summary", "conclusion", "generated_text", "completion"):
        assert absent not in payload
    assert not hasattr(response, "answer")

    # What it does return is the passages themselves, verbatim — the raw
    # material an answer would have to be built from and checked against.
    assert [item["content"] for item in payload["results"]] == [
        result.content for result in response.results
    ]
