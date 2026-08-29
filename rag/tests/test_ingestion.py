"""Tests for turning documentation and experiments into indexable documents."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag.config import RagConfig
from rag.documents import SourceType, make_document_id
from rag.errors import SourceNotFoundError, UnsafeSourceError
from rag.ingestion.documentation import (
    discover_documentation_paths,
    extract_title,
    is_forbidden_name,
    load_document,
    load_documentation,
    relative_reference,
)
from rag.ingestion.experiments import (
    experiment_metadata,
    experiment_to_document,
    load_experiments,
    render_experiment,
)
from rag.tests.factories import FakeExperimentRun, FakeExperimentStore, LEAKAGE_DOC


# --------------------------------------------------------------------------
# Documentation
# --------------------------------------------------------------------------


def test_only_the_configured_files_are_discovered(config: RagConfig) -> None:
    """An allowlist, not a crawl."""
    names = [path.name for path in discover_documentation_paths(config)]

    assert names == ["PREPROCESSING.md", "EVALUATION.md"]


def test_source_code_is_never_a_candidate(config: RagConfig, docs_root: Path) -> None:
    """A Python file in the project root is not documentation."""
    (docs_root / "service.py").write_text("SECRET = 'hunter2'\n", encoding="utf-8")
    names = [path.name for path in discover_documentation_paths(config)]

    assert "service.py" not in names


def test_a_missing_configured_file_is_skipped_not_fatal(
    config: RagConfig,
) -> None:
    """A repository that has not grown docs/ yet is not broken."""
    extended = config.with_overrides(
        documentation_files=("PREPROCESSING.md", "NOT_THERE.md")
    )
    names = [path.name for path in discover_documentation_paths(extended)]

    assert names == ["PREPROCESSING.md"]


def test_a_documentation_directory_is_walked_for_markdown(
    config: RagConfig, docs_root: Path
) -> None:
    """The one configurable directory, and only its Markdown."""
    docs = docs_root / "docs"
    docs.mkdir()
    (docs / "architecture.md").write_text("# Architecture\n\nNotes.\n", encoding="utf-8")
    (docs / "diagram.png").write_bytes(b"\x89PNG")

    names = [
        path.name
        for path in discover_documentation_paths(
            config.with_overrides(documentation_dir=docs)
        )
    ]

    assert "architecture.md" in names
    assert "diagram.png" not in names


@pytest.mark.parametrize(
    "name",
    [
        ".env",
        ".env.local",
        ".env.example",
        "secrets.json",
        "api_key.md",
        "service-credentials.md",
        "server.pem",
        "id_rsa",
        "deploy_token.md",
    ],
)
def test_credential_files_are_never_indexed(name: str) -> None:
    """A secret that reaches the index is a secret in every future answer."""
    assert is_forbidden_name(name, RagConfig()) is True


def test_a_secret_file_is_refused_even_when_configured(
    config: RagConfig, docs_root: Path
) -> None:
    """The rule is not overridable by configuration."""
    (docs_root / ".env").write_text("LLM_API_KEY=sk-real-secret\n", encoding="utf-8")
    extended = config.with_overrides(
        documentation_files=("PREPROCESSING.md", ".env")
    )

    names = [path.name for path in discover_documentation_paths(extended)]
    assert ".env" not in names

    with pytest.raises(UnsafeSourceError, match="credential"):
        load_document(Path(".env"), extended)


def test_a_secret_file_in_the_documentation_directory_is_refused(
    config: RagConfig, docs_root: Path
) -> None:
    """The directory walk applies the same rule."""
    docs = docs_root / "docs"
    docs.mkdir()
    (docs / "notes.md").write_text("# Notes\n\nfine\n", encoding="utf-8")
    (docs / "secrets.md").write_text("# Keys\n\nsk-abc\n", encoding="utf-8")

    names = [
        path.name
        for path in discover_documentation_paths(
            config.with_overrides(documentation_dir=docs)
        )
    ]

    assert names == ["PREPROCESSING.md", "EVALUATION.md", "notes.md"]


def test_a_path_outside_the_project_root_is_refused(config: RagConfig) -> None:
    """Nothing outside the repository is indexable, however it is spelled."""
    with pytest.raises(UnsafeSourceError, match="project root"):
        load_document(Path("../../etc/passwd"), config)


def test_a_directory_that_holds_data_is_never_descended(
    config: RagConfig, docs_root: Path
) -> None:
    """Raw datasets are not knowledge."""
    data = docs_root / "data"
    data.mkdir()
    (data / "notes.md").write_text("# Rows\n\n1,2,3\n", encoding="utf-8")

    names = [
        path.name
        for path in discover_documentation_paths(
            config.with_overrides(documentation_dir=data)
        )
    ]

    assert "notes.md" not in names


def test_an_oversized_document_is_refused(config: RagConfig, docs_root: Path) -> None:
    """A guard against pointing the indexer at something enormous."""
    (docs_root / "BIG.md").write_text("# Big\n\n" + "x" * 5_000, encoding="utf-8")
    small = config.with_overrides(max_document_bytes=100)

    with pytest.raises(UnsafeSourceError, match="limit"):
        load_document(Path("BIG.md"), small)


def test_a_missing_file_is_reported(config: RagConfig) -> None:
    """Asking for a specific file that is not there is an error."""
    with pytest.raises(SourceNotFoundError):
        load_document(Path("ABSENT.md"), config)


def test_a_loaded_document_carries_its_title_and_reference(
    config: RagConfig,
) -> None:
    """A chunk should be labelled by what it is, not by a file path."""
    document = load_document(Path("PREPROCESSING.md"), config)

    assert document.source_type == SourceType.PROJECT_DOCUMENTATION.value
    assert document.source_title == "Preprocessing Guide"
    assert document.source_reference == "PREPROCESSING.md"
    assert document.document_id == make_document_id(
        SourceType.PROJECT_DOCUMENTATION.value, "PREPROCESSING.md"
    )
    assert document.metadata["path"] == "PREPROCESSING.md"


def test_a_reference_is_posix_on_every_platform(config: RagConfig, docs_root: Path) -> None:
    """So an index built on Windows and one on Linux agree on ids."""
    nested = docs_root / "sub"
    nested.mkdir()
    path = nested / "GUIDE.md"
    path.write_text("# Guide\n\ntext\n", encoding="utf-8")

    assert relative_reference(path, config) == "sub/GUIDE.md"


def test_a_document_without_a_heading_falls_back_to_its_reference() -> None:
    """Every document needs a title, even one that has no heading."""
    assert extract_title("no heading here", fallback="NOTES.md") == "NOTES.md"
    assert extract_title(LEAKAGE_DOC, fallback="x") == "Preprocessing Guide"


def test_loading_documentation_yields_every_allowed_file(config: RagConfig) -> None:
    """The iterator the indexer consumes."""
    documents = list(load_documentation(config))

    assert [document.source_reference for document in documents] == [
        "PREPROCESSING.md",
        "EVALUATION.md",
    ]


# --------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------


def test_an_experiment_renders_as_readable_structured_facts(
    experiment_run: FakeExperimentRun,
) -> None:
    """Every line is a value from the record."""
    text = render_experiment(experiment_run)

    assert f"Experiment ID: {experiment_run.experiment_id}" in text
    assert "Task: classification" in text
    assert "Dataset fingerprint: 86494cff7a45cb7f" in text
    assert "Selected model: random_forest_classifier" in text
    assert "Selection score: 0.8700 ± 0.0200" in text
    assert "Final test score: 0.8500" in text
    assert "Baseline: most_frequent" in text
    assert "1. tenure_months: 0.3100" in text
    assert "2. income: 0.2700" in text


def test_the_rendered_experiment_has_a_section_per_subject(
    experiment_run: FakeExperimentRun,
) -> None:
    """Headings are what the chunker splits on, so each part is retrievable."""
    text = render_experiment(experiment_run)

    for heading in (
        "## Overview",
        "## Dataset",
        "## Preprocessing",
        "## Model selection",
        "## Final evaluation",
        "## Explainability",
        "## Environment",
    ):
        assert heading in text


def test_the_rendered_experiment_invents_no_conclusion(
    experiment_run: FakeExperimentRun,
) -> None:
    """Ungrounded prose in the index would be retrieved and cited as fact."""
    text = render_experiment(experiment_run).lower()

    for phrase in (
        "performed well",
        "the best model",
        "we recommend",
        "suggests that",
        "clearly shows",
        "therefore",
        "in conclusion",
    ):
        assert phrase not in text


def test_an_experiment_without_an_explanation_says_so(
) -> None:
    """Absence is recorded as absence, not omitted."""
    text = render_experiment(FakeExperimentRun(with_explanation=False))

    assert "No explanation was recorded" in text


def test_experiment_metadata_carries_what_filtering_needs(
    experiment_run: FakeExperimentRun,
) -> None:
    """These are the keys a caller narrows a search by."""
    metadata = experiment_metadata(experiment_run)

    assert metadata["source_type"] == SourceType.EXPERIMENT.value
    assert metadata["experiment_id"] == experiment_run.experiment_id
    assert metadata["dataset_fingerprint"] == "86494cff7a45cb7f"
    assert metadata["task_type"] == "classification"
    assert metadata["target_column"] == "renewed"
    assert metadata["selected_model"] == "random_forest_classifier"
    assert metadata["primary_metric"] == "f1"
    assert metadata["test_score"] == 0.85


def test_experiment_metadata_is_json_safe(experiment_run: FakeExperimentRun) -> None:
    """It is written to disk and returned over an API."""
    json.dumps(experiment_to_document(experiment_run).metadata)


def test_an_experiment_document_is_cited_by_its_id(
    experiment_run: FakeExperimentRun,
) -> None:
    """'According to experiment exp_...' has to resolve."""
    document = experiment_to_document(experiment_run)

    assert document.source_type == SourceType.EXPERIMENT.value
    assert document.source_reference == experiment_run.experiment_id
    assert experiment_run.experiment_id in document.source_title


def test_two_conversions_of_one_experiment_are_identical(
    experiment_run: FakeExperimentRun,
) -> None:
    """Determinism, so re-indexing an unchanged run is a no-op."""
    first = experiment_to_document(experiment_run)
    second = experiment_to_document(experiment_run)

    assert first.document_id == second.document_id
    assert first.source_hash == second.source_hash


def test_a_changed_experiment_changes_its_hash() -> None:
    """A re-run under the same id must be re-indexed, not skipped."""
    first = experiment_to_document(FakeExperimentRun(test_score=0.85))
    second = experiment_to_document(FakeExperimentRun(test_score=0.91))

    assert first.document_id == second.document_id
    assert first.source_hash != second.source_hash


def test_loading_experiments_reads_the_store(
    experiment_store: FakeExperimentStore,
) -> None:
    """The dependency runs one way: RAG reads the store, never the reverse."""
    documents = list(load_experiments(experiment_store))

    assert len(documents) == 2
    assert experiment_store.list_calls == 1
    assert {document.metadata["task_type"] for document in documents} == {
        "classification",
        "regression",
    }


def test_one_unreadable_record_does_not_hide_the_rest(
    experiment_run: FakeExperimentRun,
) -> None:
    """A corrupt record must not make the whole history unsearchable."""

    class Broken:
        """A record that raises when its sections are read."""

        experiment_id = "exp_broken"

        def __getattr__(self, name: str):
            raise ValueError("unreadable")

    store = FakeExperimentStore([Broken(), experiment_run])
    documents = list(load_experiments(store))

    assert len(documents) == 1
    assert documents[0].source_reference == experiment_run.experiment_id
