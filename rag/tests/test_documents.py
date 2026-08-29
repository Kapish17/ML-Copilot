"""Tests for the document model, identity, hashing, chunking and citations."""

from __future__ import annotations

import json

import pytest

from rag.chunking import (
    HEADING_SEPARATOR,
    chunk_document,
    chunk_markdown,
    pack_paragraphs,
    split_paragraphs,
    split_sections,
)
from rag.citations import build_citation, document_slug, parse_citation
from rag.config import RagConfig
from rag.documents import (
    Chunk,
    Document,
    SourceType,
    content_hash,
    jsonable_metadata,
    make_chunk_id,
    make_document_id,
    slugify,
)
from rag.errors import ConfigurationError
from rag.tests.factories import EVALUATION_DOC, LEAKAGE_DOC, TINY_DOC


def make_document(content: str = LEAKAGE_DOC, **overrides) -> Document:
    """Build a documentation document for a test."""
    fields = {
        "source_type": SourceType.PROJECT_DOCUMENTATION.value,
        "source_title": "Preprocessing Guide",
        "source_reference": "PREPROCESSING.md",
        "content": content,
    }
    fields.update(overrides)
    return Document(**fields)


# --------------------------------------------------------------------------
# Identity
# --------------------------------------------------------------------------


def test_a_document_derives_its_own_identifier() -> None:
    """Identity comes from what the document is, not from a generator.

    The id keeps the file's extension because it names the *source*; the
    citation drops it because it names the document a reader would look up.
    """
    document = make_document()

    assert document.document_id == "project_documentation:preprocessing.md"
    assert make_document().document_id == document.document_id


def test_the_same_source_always_gets_the_same_identifier() -> None:
    """Two processes indexing one file agree on its id."""
    first = make_document_id("experiment", "exp_abc123_20260101T000000Z_0001")
    second = make_document_id("experiment", "exp_abc123_20260101T000000Z_0001")

    assert first == second
    assert "exp_abc123" in first


def test_a_different_source_gets_a_different_identifier() -> None:
    """Two files are two documents, even under the same source type."""
    assert make_document_id("project_documentation", "README.md") != make_document_id(
        "project_documentation", "ml/README.md"
    )


def test_an_awkward_reference_still_produces_a_legal_identifier() -> None:
    """A long or exotic reference is hashed rather than truncated blindly."""
    document_id = make_document_id("project_documentation", "a/" * 200 + "deep.md")

    assert document_id.startswith("project_documentation:")
    assert len(document_id) < 80
    assert "/" not in document_id


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("ml/README.md", "ml-readme.md"),
        ("  Leakage Prevention  ", "leakage-prevention"),
        ("Cross-Validation & Selection", "cross-validation-selection"),
        ("", "untitled"),
        ("///", "untitled"),
    ],
)
def test_slugs_are_predictable(value: str, expected: str) -> None:
    """Slugs are what ids and citations are built from, so they are stable."""
    assert slugify(value) == expected


def test_a_chunk_identifier_combines_document_position_and_content() -> None:
    """Position alone would collide across rewrites; content alone across copies."""
    first = make_chunk_id("docs:readme", 0, "some text")
    same = make_chunk_id("docs:readme", 0, "some text")
    moved = make_chunk_id("docs:readme", 1, "some text")
    edited = make_chunk_id("docs:readme", 0, "other text")

    assert first == same
    assert first != moved
    assert first != edited
    assert first.startswith("docs:readme#0000-")


def test_no_chunk_identifier_is_random() -> None:
    """Two independent chunkings of one document produce identical ids."""
    config = RagConfig(chunk_size=400, chunk_overlap=60, min_chunk_size=80)
    first = [chunk.chunk_id for chunk in chunk_document(make_document(), config)]
    second = [chunk.chunk_id for chunk in chunk_document(make_document(), config)]

    assert first == second
    assert len(set(first)) == len(first), "chunk ids must be unique within a document"


# --------------------------------------------------------------------------
# Hashing
# --------------------------------------------------------------------------


def test_identical_text_hashes_identically() -> None:
    """Change detection depends on this and nothing else."""
    assert content_hash("hello world") == content_hash("hello world")


def test_changed_text_changes_the_hash() -> None:
    """Including whitespace: it changes how the text chunks."""
    assert content_hash("hello world") != content_hash("hello  world")
    assert content_hash("hello world") != content_hash("hello world!")


def test_the_source_hash_covers_metadata_as_well_as_content() -> None:
    """A tag change with no text change is still a change worth re-indexing."""
    plain = make_document(metadata={"tags": ["a"]})
    tagged = make_document(metadata={"tags": ["b"]})

    assert plain.content_hash == tagged.content_hash
    assert plain.metadata_hash != tagged.metadata_hash
    assert plain.source_hash != tagged.source_hash


def test_metadata_is_reduced_to_json_safe_values() -> None:
    """Metadata is written to disk and returned over an API."""
    payload = jsonable_metadata(
        {
            "count": 3,
            "score": float("nan"),
            "kind": SourceType.EXPERIMENT,
            "tags": ("a", "b"),
            "nested": {"path": None},
            7: "int key",
        }
    )

    json.dumps(payload)
    assert payload["score"] is None
    assert payload["kind"] == "experiment"
    assert payload["tags"] == ["a", "b"]
    assert payload["7"] == "int key"


def test_a_chunk_round_trips_through_its_stored_form() -> None:
    """What is written to the record file is what comes back."""
    chunk = chunk_document(make_document(), RagConfig(chunk_size=400, chunk_overlap=60))[0]
    restored = Chunk.from_dict(json.loads(json.dumps(chunk.as_dict())))

    assert restored == chunk


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------


def test_sections_follow_the_heading_structure() -> None:
    """A heading marks where the subject changes."""
    sections = split_sections(LEAKAGE_DOC)
    titles = [section.heading_path for section in sections]

    assert ("Preprocessing Guide",) in titles
    assert ("Preprocessing Guide", "Leakage prevention") in titles
    assert ("Preprocessing Guide", "Categorical columns") in titles


def test_a_nested_heading_keeps_its_parent() -> None:
    """The path is what tells a passage read alone where it belongs."""
    text = "# Top\n\nintro\n\n## Middle\n\nbody\n\n### Deep\n\ndetail\n\n## Other\n\nmore\n"
    paths = [section.heading_path for section in split_sections(text)]

    assert ("Top", "Middle", "Deep") in paths
    assert ("Top", "Other") in paths, "a sibling heading pops the deeper level"


def test_a_heading_inside_a_code_block_is_not_a_heading() -> None:
    """A shell comment is not a section."""
    text = "# Real\n\n```bash\n# not a heading\necho hi\n```\n\nafter\n"
    sections = split_sections(text)

    assert len(sections) == 1
    assert sections[0].heading_path == ("Real",)
    assert "echo hi" in sections[0].content


def test_a_fenced_code_block_is_never_split() -> None:
    """Fence markers have to stay with their content."""
    code = "\n".join(f"line {index}" for index in range(40))
    text = f"# Doc\n\nintro\n\n```python\n{code}\n```\n"
    paragraphs = split_paragraphs(text.split("\n\n", 1)[1])
    fenced = [item for item in paragraphs if item.startswith("```")]

    assert len(fenced) == 1
    assert fenced[0].count("```") == 2


def test_paragraphs_are_packed_not_cut_mid_sentence() -> None:
    """Packing stops before overflowing rather than slicing a paragraph."""
    paragraphs = [f"Paragraph number {index}." for index in range(20)]
    chunks = pack_paragraphs(paragraphs, chunk_size=120, overlap=0)

    assert len(chunks) > 1
    assert all(len(chunk) <= 120 for chunk in chunks)
    assert all(chunk.endswith(".") for chunk in chunks)


def test_overlap_repeats_the_tail_of_the_previous_chunk() -> None:
    """A sentence across a boundary stays findable from either side."""
    paragraphs = [f"Sentence {index} about leakage." for index in range(12)]
    with_overlap = pack_paragraphs(paragraphs, chunk_size=150, overlap=50)
    without = pack_paragraphs(paragraphs, chunk_size=150, overlap=0)

    assert len(with_overlap) >= len(without)
    assert any(
        with_overlap[index][:20] in with_overlap[index - 1]
        for index in range(1, len(with_overlap))
    )


def test_a_very_long_paragraph_is_split_on_a_line_boundary() -> None:
    """Last resort, and still not mid-word where a newline is available."""
    long_paragraph = "\n".join(f"row {index} of a wide table" for index in range(60))
    chunks = pack_paragraphs([long_paragraph], chunk_size=200, overlap=0)

    assert len(chunks) > 1
    assert all(len(chunk) <= 220 for chunk in chunks)


def test_tiny_fragments_are_merged_away() -> None:
    """A heading with one line under it is not worth retrieving on its own."""
    config = RagConfig(chunk_size=600, chunk_overlap=50, min_chunk_size=200)
    chunks = chunk_document(make_document(EVALUATION_DOC), config)

    assert chunks
    assert all(len(chunk.content) >= 100 for chunk in chunks)


def test_a_chunk_carries_its_heading_in_the_text_it_will_be_read_as() -> None:
    """The context has to survive being read without the document around it."""
    config = RagConfig(chunk_size=400, chunk_overlap=60, min_chunk_size=80)
    chunks = chunk_document(make_document(), config)
    leakage = [chunk for chunk in chunks if chunk.heading == "Leakage prevention"]

    assert leakage, "expected a chunk under the leakage heading"
    assert leakage[0].content.startswith(
        f"Preprocessing Guide{HEADING_SEPARATOR}Leakage prevention"
    )
    assert leakage[0].heading_path == ("Preprocessing Guide", "Leakage prevention")
    assert leakage[0].metadata["heading"] == "Leakage prevention"


def test_a_document_with_no_headings_still_chunks() -> None:
    """Not every source is structured; none may vanish."""
    config = RagConfig(chunk_size=200, chunk_overlap=20, min_chunk_size=0)
    chunks = chunk_document(make_document("just some prose, no headings at all"), config)

    assert len(chunks) == 1
    assert chunks[0].heading_path == ()


def test_an_empty_document_produces_no_chunks() -> None:
    """Nothing to say, nothing to index."""
    assert chunk_markdown("   \n\n  ", RagConfig()) == []


def test_a_tiny_document_still_produces_one_chunk() -> None:
    """The minimum size merges fragments; it does not delete a whole document."""
    config = RagConfig(chunk_size=400, chunk_overlap=40, min_chunk_size=200)
    chunks = chunk_document(make_document(TINY_DOC), config)

    assert len(chunks) == 1
    assert "One line." in chunks[0].content


def test_chunk_positions_are_reading_order() -> None:
    """Retrieved chunks can be put back in the order they were written."""
    config = RagConfig(chunk_size=300, chunk_overlap=40, min_chunk_size=50)
    chunks = chunk_document(make_document(), config)

    assert [chunk.position for chunk in chunks] == list(range(len(chunks)))


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"chunk_size": 0},
        {"chunk_overlap": -1},
        {"chunk_size": 100, "chunk_overlap": 100},
        {"min_chunk_size": 5_000},
        {"embedding_dimension": 1},
        {"top_k": 0},
        {"similarity_threshold": 2.0},
        {"embedding_batch_size": 0},
    ],
)
def test_unusable_configuration_is_refused(overrides: dict) -> None:
    """An overlap as large as the chunk size would never advance."""
    with pytest.raises(ConfigurationError):
        RagConfig(**overrides)


def test_an_unknown_configuration_field_is_refused() -> None:
    """A typo in an override is an error, not a silently ignored setting."""
    with pytest.raises(ConfigurationError, match="chunk_sze"):
        RagConfig().with_overrides(chunk_sze=100)


def test_top_k_is_capped() -> None:
    """One query cannot ask for the whole index."""
    config = RagConfig(top_k=5, max_top_k=10)

    assert config.resolve_top_k(None) == 5
    assert config.resolve_top_k(7) == 7
    with pytest.raises(ConfigurationError):
        config.resolve_top_k(11)


# --------------------------------------------------------------------------
# Citations
# --------------------------------------------------------------------------


def test_a_documentation_citation_names_the_file_and_the_section() -> None:
    """Enough to find the passage again by hand."""
    citation = build_citation(
        source_type=SourceType.PROJECT_DOCUMENTATION.value,
        source_reference="ml/README.md",
        fragment="Leakage prevention",
    )

    assert citation == "docs:ml-readme#leakage-prevention"


def test_an_experiment_citation_is_the_experiment_id() -> None:
    """So 'according to experiment exp_...' resolves through the existing API."""
    citation = build_citation(
        source_type=SourceType.EXPERIMENT.value,
        source_reference="exp_84a8d53a1f5f_20260828T134457Z_e420",
    )

    assert citation == "experiment:exp_84a8d53a1f5f_20260828T134457Z_e420"


def test_a_citation_parses_back_into_its_parts() -> None:
    """A reference is only useful if it can be resolved."""
    assert parse_citation("docs:ml-readme#leakage-prevention") == (
        "docs",
        "ml-readme",
        "leakage-prevention",
    )
    assert parse_citation("experiment:exp_1") == ("experiment", "exp_1", None)


def test_an_unknown_source_type_still_gets_a_resolvable_citation() -> None:
    """A future ingestion adapter is not required to edit this module."""
    citation = build_citation(source_type="notebook", source_reference="tour.ipynb")

    assert citation.startswith("source:")


def test_the_document_slug_drops_the_markdown_extension() -> None:
    """A citation names a document, not a filename."""
    assert document_slug("backend/README.md") == "backend-readme"


def test_every_chunk_of_a_document_carries_a_citation() -> None:
    """Attribution is not optional."""
    config = RagConfig(chunk_size=300, chunk_overlap=40, min_chunk_size=50)
    chunks = chunk_document(make_document(), config)

    assert all(chunk.citation.startswith("docs:preprocessing") for chunk in chunks)
    assert len({chunk.citation for chunk in chunks}) > 1, "sections differ"
