"""Configuration for the retrieval layer.

Every threshold, size, path and provider name the RAG package uses lives here.
Nothing downstream hard-codes a number: a module that needs a chunk size takes
a :class:`RagConfig`, so behaviour can be changed in one place and a test can
run against different settings without touching the environment.

The defaults are chosen so the whole layer runs **offline, with no API key and
no model download**. See ``rag/README.md`` for what that costs in retrieval
quality and how to trade it for a sentence-transformer model.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

from rag.errors import ConfigurationError

#: Repository root, derived from this file rather than the working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# -- Storage ----------------------------------------------------------------
#: Where the vector index, its records and the manifest are written.
DEFAULT_INDEX_DIR = PROJECT_ROOT / "rag" / "index"

# -- Embeddings -------------------------------------------------------------
#: The offline default. See ``rag/embeddings/hashing.py``.
DEFAULT_EMBEDDING_PROVIDER = "hashing"
#: Dimension of the default provider's vectors. Large enough that hash
#: collisions are rare for a corpus of this size, small enough to stay fast.
DEFAULT_EMBEDDING_DIMENSION = 512
#: Documents embedded per call. Batching matters for the optional neural
#: provider; the default provider vectorises the whole batch either way.
DEFAULT_EMBEDDING_BATCH_SIZE = 64

# -- Chunking ---------------------------------------------------------------
#: Target upper bound on a chunk, in characters. Sections longer than this are
#: split on paragraph boundaries.
DEFAULT_CHUNK_SIZE = 1_200
#: Characters of the previous chunk repeated at the start of the next, so a
#: sentence split across a boundary is still retrievable from either side.
DEFAULT_CHUNK_OVERLAP = 150
#: Below this, a fragment is merged into its neighbour instead of becoming a
#: chunk of its own. A heading with two words under it retrieves nothing
#: useful and dilutes the ranking.
DEFAULT_MIN_CHUNK_SIZE = 120

# -- Retrieval --------------------------------------------------------------
#: Chunks returned when the caller does not say.
DEFAULT_TOP_K = 5
#: Hard cap on ``top_k``, so one query cannot ask for the whole index.
DEFAULT_MAX_TOP_K = 50
#: Minimum cosine similarity for a chunk to be returned at all. Zero means
#: "return the best matches whatever they score"; raise it to trade recall for
#: precision.
DEFAULT_SIMILARITY_THRESHOLD = 0.0
#: Longest query accepted. A question is a question; anything much longer is a
#: document being pasted in, which embeds poorly and is a cheap way to make a
#: server do pointless work. The limit lives here rather than in a caller so
#: that a library user and an HTTP client are held to the same rule.
DEFAULT_MAX_QUERY_LENGTH = 2_000

# -- Documentation ingestion ------------------------------------------------
#: The documentation indexed by default. An explicit allowlist, not a
#: recursive walk: source code, datasets, model artefacts, virtual
#: environments, ``.git`` and secrets are never candidates in the first place.
DEFAULT_DOCUMENTATION_FILES: tuple[str, ...] = (
    "README.md",
    "ml/README.md",
    "backend/README.md",
    "rag/README.md",
    "llm/README.md",
)
#: An additional directory whose Markdown is indexed, when it exists.
DEFAULT_DOCUMENTATION_DIR = PROJECT_ROOT / "docs"
#: Extensions accepted from that directory.
DEFAULT_DOCUMENTATION_EXTENSIONS: tuple[str, ...] = (".md", ".markdown")
#: Largest documentation file that will be read, in bytes. A guard against
#: accidentally pointing the indexer at something enormous.
DEFAULT_MAX_DOCUMENT_BYTES = 2 * 1024 * 1024

#: Directory names never descended into, whatever the configuration says.
FORBIDDEN_DIRECTORY_NAMES: frozenset[str] = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "env",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "site-packages",
        "data",
        "datasets",
        "mlruns",
        "mlartifacts",
        "runs",
        "index",
        ".idea",
        ".vscode",
    }
)

#: Filenames never indexed, whatever the configuration says. Credentials are
#: the point of this list; a secret that reaches the index is a secret in
#: every future answer.
FORBIDDEN_FILE_NAMES: frozenset[str] = frozenset(
    {
        ".env",
        ".env.local",
        ".env.example",
        ".env.production",
        ".env.development",
        "credentials",
        "credentials.json",
        "secrets.json",
        "secrets.yaml",
        "secrets.yml",
        "id_rsa",
        "id_ed25519",
        ".npmrc",
        ".pypirc",
        ".netrc",
        ".htpasswd",
    }
)

#: Filename fragments that mark a file as unsafe to index, matched
#: case-insensitively against the whole name.
FORBIDDEN_NAME_FRAGMENTS: tuple[str, ...] = (
    ".env",
    "secret",
    "credential",
    "password",
    "apikey",
    "api_key",
    "token",
    ".pem",
    ".key",
    ".pfx",
    ".p12",
)


@dataclass(frozen=True)
class RagConfig:
    """Every knob the retrieval layer has, in one immutable object."""

    # Storage
    index_dir: Path = DEFAULT_INDEX_DIR

    # Embeddings
    embedding_provider: str = DEFAULT_EMBEDDING_PROVIDER
    embedding_dimension: int = DEFAULT_EMBEDDING_DIMENSION
    embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE
    #: Model identifier for providers that have one; ignored by the default.
    embedding_model: str | None = None

    # Chunking
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    min_chunk_size: int = DEFAULT_MIN_CHUNK_SIZE

    # Retrieval
    top_k: int = DEFAULT_TOP_K
    max_top_k: int = DEFAULT_MAX_TOP_K
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    max_query_length: int = DEFAULT_MAX_QUERY_LENGTH

    # Documentation ingestion
    project_root: Path = PROJECT_ROOT
    documentation_files: tuple[str, ...] = DEFAULT_DOCUMENTATION_FILES
    documentation_dir: Path | None = DEFAULT_DOCUMENTATION_DIR
    documentation_extensions: tuple[str, ...] = DEFAULT_DOCUMENTATION_EXTENSIONS
    max_document_bytes: int = DEFAULT_MAX_DOCUMENT_BYTES
    forbidden_directory_names: frozenset[str] = FORBIDDEN_DIRECTORY_NAMES
    forbidden_file_names: frozenset[str] = FORBIDDEN_FILE_NAMES
    forbidden_name_fragments: tuple[str, ...] = field(
        default=FORBIDDEN_NAME_FRAGMENTS
    )

    def __post_init__(self) -> None:
        """Validate the combination, not just the individual values."""
        if self.chunk_size < 1:
            raise ConfigurationError(
                "chunk_size must be at least 1.",
                details={"chunk_size": self.chunk_size},
            )
        if self.chunk_overlap < 0:
            raise ConfigurationError(
                "chunk_overlap cannot be negative.",
                details={"chunk_overlap": self.chunk_overlap},
            )
        if self.chunk_overlap >= self.chunk_size:
            # Otherwise a chunk repeats everything the last one held and the
            # splitter never advances.
            raise ConfigurationError(
                "chunk_overlap must be smaller than chunk_size.",
                details={
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                },
            )
        if self.min_chunk_size < 0 or self.min_chunk_size > self.chunk_size:
            raise ConfigurationError(
                "min_chunk_size must be between 0 and chunk_size.",
                details={
                    "min_chunk_size": self.min_chunk_size,
                    "chunk_size": self.chunk_size,
                },
            )
        if self.embedding_dimension < 2:
            raise ConfigurationError(
                "embedding_dimension must be at least 2.",
                details={"embedding_dimension": self.embedding_dimension},
            )
        if self.embedding_batch_size < 1:
            raise ConfigurationError(
                "embedding_batch_size must be at least 1.",
                details={"embedding_batch_size": self.embedding_batch_size},
            )
        if not 1 <= self.top_k <= self.max_top_k:
            raise ConfigurationError(
                f"top_k must be between 1 and max_top_k ({self.max_top_k}).",
                details={"top_k": self.top_k, "max_top_k": self.max_top_k},
            )
        if not -1.0 <= self.similarity_threshold <= 1.0:
            raise ConfigurationError(
                "similarity_threshold must be a cosine similarity in [-1, 1].",
                details={"similarity_threshold": self.similarity_threshold},
            )
        if self.max_query_length < 1:
            raise ConfigurationError(
                "max_query_length must be at least 1.",
                details={"max_query_length": self.max_query_length},
            )

    def with_overrides(self, **overrides: object) -> RagConfig:
        """Return a validated copy with some fields replaced.

        Raises:
            ConfigurationError: If a field name is unknown or a value invalid.
        """
        unknown = sorted(set(overrides) - set(self.__dataclass_fields__))
        if unknown:
            raise ConfigurationError(
                "Unknown configuration field(s): " + ", ".join(unknown) + ".",
                details={"unknown_fields": unknown},
            )
        return replace(self, **overrides)

    def resolve_top_k(self, top_k: int | None) -> int:
        """Return the number of chunks a query should return.

        Args:
            top_k: What the caller asked for, or ``None`` for the default.

        Raises:
            ConfigurationError: If the request is below one or above the cap.
        """
        if top_k is None:
            return self.top_k
        if not 1 <= top_k <= self.max_top_k:
            raise ConfigurationError(
                f"top_k must be between 1 and {self.max_top_k}, got {top_k}.",
                details={"top_k": top_k, "max_top_k": self.max_top_k},
            )
        return top_k

    def resolve_query(self, query: str | None) -> str:
        """Return a usable query, or explain why one is not.

        The same rule for every caller: a library user and an HTTP client are
        held to one definition of "too long" and one of "empty", rather than
        each entry point inventing its own.

        Args:
            query: What the caller asked, in their own words.

        Returns:
            str: The trimmed query.

        Raises:
            ConfigurationError: If it is blank or longer than
                ``max_query_length``.
        """
        text = (query or "").strip()
        if not text:
            raise ConfigurationError(
                "A query is required.", details={"query_length": 0}
            )
        if len(text) > self.max_query_length:
            raise ConfigurationError(
                f"A query may be at most {self.max_query_length} characters, "
                f"got {len(text)}.",
                details={
                    "query_length": len(text),
                    "max_query_length": self.max_query_length,
                },
            )
        return text

    def resolve_threshold(self, threshold: float | None) -> float:
        """Return the minimum similarity a result must reach."""
        if threshold is None:
            return self.similarity_threshold
        if not -1.0 <= threshold <= 1.0:
            raise ConfigurationError(
                "similarity_threshold must be a cosine similarity in [-1, 1].",
                details={"similarity_threshold": threshold},
            )
        return threshold


def config_from_env(**overrides: object) -> RagConfig:
    """Build a configuration from environment variables, then overrides.

    Only the handful of settings worth changing per deployment are read from
    the environment; the rest are code-level defaults. **No secret is read
    here** — the default embedding provider needs no key, and none is
    supported in this commit.
    """
    values: dict[str, object] = {}

    index_dir = os.getenv("RAG_INDEX_DIR", "").strip()
    if index_dir:
        values["index_dir"] = Path(index_dir)

    provider = os.getenv("RAG_EMBEDDING_PROVIDER", "").strip()
    if provider:
        values["embedding_provider"] = provider

    model = os.getenv("RAG_EMBEDDING_MODEL", "").strip()
    if model:
        values["embedding_model"] = model

    for name, key in (
        ("embedding_dimension", "RAG_EMBEDDING_DIMENSION"),
        ("chunk_size", "RAG_CHUNK_SIZE"),
        ("chunk_overlap", "RAG_CHUNK_OVERLAP"),
        ("top_k", "RAG_TOP_K"),
    ):
        raw = os.getenv(key, "").strip()
        if not raw:
            continue
        try:
            values[name] = int(raw)
        except ValueError as exc:
            raise ConfigurationError(
                f"{key} must be an integer, got {raw!r}.", details={key: raw}
            ) from exc

    max_query = os.getenv("RAG_MAX_QUERY_LENGTH", "").strip()
    if max_query:
        try:
            values["max_query_length"] = int(max_query)
        except ValueError as exc:
            raise ConfigurationError(
                f"RAG_MAX_QUERY_LENGTH must be an integer, got {max_query!r}.",
                details={"RAG_MAX_QUERY_LENGTH": max_query},
            ) from exc

    threshold = os.getenv("RAG_SIMILARITY_THRESHOLD", "").strip()
    if threshold:
        try:
            values["similarity_threshold"] = float(threshold)
        except ValueError as exc:
            raise ConfigurationError(
                f"RAG_SIMILARITY_THRESHOLD must be a number, got {threshold!r}.",
                details={"RAG_SIMILARITY_THRESHOLD": threshold},
            ) from exc

    values.update(overrides)
    return RagConfig(**values)  # type: ignore[arg-type]
