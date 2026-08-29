"""Turning project documentation into indexable documents.

This is an **allowlist, not a crawl**. The configuration names the files to
index — the four READMEs by default — plus one optional documentation
directory. Nothing else in the repository is a candidate, so source code,
datasets, model artefacts, virtual environments, ``.git`` and the experiment
store are not merely skipped: they are never looked at.

Three checks stand between a path and the index, and they are enforced here
rather than trusted to the caller:

**Containment.** Every path is resolved and must lie inside the configured
project root. A configured entry of ``../../etc/passwd`` is refused.

**Forbidden names.** ``.env`` and anything whose name suggests a credential —
*secret*, *token*, *password*, ``.pem``, ``.key`` — is refused even when
explicitly listed. A key that reaches the index is a key in every future
answer, so the rule is not overridable by configuration.

**Extension and size.** Only Markdown is read, and only up to a configured
size.

The title comes from the document's first heading when it has one, so a chunk
of ``ml/README.md`` is labelled "ML Copilot — ML Layer" rather than a path.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from pathlib import Path

from rag.chunking import HEADING_PATTERN
from rag.config import RagConfig
from rag.documents import Document, SourceType
from rag.errors import SourceNotFoundError, UnsafeSourceError

logger = logging.getLogger(__name__)


def is_forbidden_name(name: str, config: RagConfig) -> bool:
    """Return whether a filename must never be indexed.

    Args:
        name: The bare filename, no directory part.
        config: Supplies the forbidden names and fragments.
    """
    lowered = name.lower()
    if lowered in config.forbidden_file_names:
        return True
    return any(fragment in lowered for fragment in config.forbidden_name_fragments)


def is_forbidden_path(path: Path, config: RagConfig) -> bool:
    """Return whether any part of a path makes it unindexable."""
    if any(part in config.forbidden_directory_names for part in path.parts):
        return True
    return is_forbidden_name(path.name, config)


def _resolve(reference: str | Path, config: RagConfig) -> Path:
    """Resolve a configured reference to a path inside the project root.

    Raises:
        UnsafeSourceError: If it resolves outside the root, or names a file
            that must never be indexed.
    """
    root = config.project_root.resolve()
    candidate = Path(reference)
    resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()

    if resolved != root and root not in resolved.parents:
        raise UnsafeSourceError(
            "Documentation sources must live inside the project root.",
            details={"reference": str(reference)},
        )
    if is_forbidden_name(resolved.name, config):
        raise UnsafeSourceError(
            f"'{resolved.name}' is never indexed: its name marks it as a "
            "credential or environment file.",
            details={"reference": str(reference), "filename": resolved.name},
        )
    return resolved


def relative_reference(path: Path, config: RagConfig) -> str:
    """Return the repository-relative reference used to identify a file.

    Always POSIX-style, so an index built on Windows and one built on Linux
    agree on document ids and citations.
    """
    root = config.project_root.resolve()
    resolved = path.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:  # pragma: no cover - containment is checked earlier
        return resolved.name


def extract_title(text: str, fallback: str) -> str:
    """Return the document's first heading, or a fallback.

    Args:
        text: The Markdown content.
        fallback: Used when the document has no heading.
    """
    for line in text.splitlines():
        match = HEADING_PATTERN.match(line)
        if match:
            title = match.group(2).strip()
            if title:
                return title
    return fallback


def discover_documentation_paths(config: RagConfig) -> list[Path]:
    """Return every documentation file that should be indexed.

    The configured files come first, in order, followed by any Markdown in
    the documentation directory. Missing configured files are skipped with a
    warning rather than failing the whole index — a repository that has not
    grown ``docs/`` yet is not broken.

    Returns:
        list[pathlib.Path]: Existing, readable, allowed files, de-duplicated.
    """
    paths: list[Path] = []
    seen: set[Path] = set()

    def consider(path: Path) -> None:
        """Add a path if it is a real, allowed, unseen file."""
        if path in seen:
            return
        seen.add(path)
        reference = relative_reference(path, config)
        if not path.is_file():
            logger.info("Documentation source is not present, skipping: %s", reference)
            return
        if is_forbidden_path(Path(reference), config):
            logger.warning("Refusing to index %s", reference)
            return
        paths.append(path)

    for reference in config.documentation_files:
        try:
            consider(_resolve(reference, config))
        except UnsafeSourceError:
            logger.warning("Refusing unsafe documentation source: %s", reference)

    directory = config.documentation_dir
    if directory is not None:
        resolved_dir = _resolve(directory, config)
        if resolved_dir.is_dir():
            for path in sorted(resolved_dir.rglob("*")):
                if not path.is_file():
                    continue
                if path.suffix.lower() not in config.documentation_extensions:
                    continue
                relative = path.resolve().relative_to(config.project_root.resolve())
                if is_forbidden_path(relative, config):
                    logger.warning("Refusing to index %s", path.name)
                    continue
                consider(path.resolve())

    return paths


def load_document(path: Path, config: RagConfig) -> Document:
    """Read one documentation file into a :class:`~rag.documents.Document`.

    Args:
        path: The file to read.
        config: Supplies the size limit and the project root.

    Returns:
        Document: With its title taken from the first heading and its
        reference the repository-relative path.

    Raises:
        SourceNotFoundError: If the file does not exist.
        UnsafeSourceError: If the file is forbidden or too large.
    """
    resolved = _resolve(path, config)
    if not resolved.is_file():
        raise SourceNotFoundError(
            f"Documentation source not found: {resolved.name}",
            details={"reference": relative_reference(resolved, config)},
        )

    size = resolved.stat().st_size
    if size > config.max_document_bytes:
        raise UnsafeSourceError(
            f"'{resolved.name}' is larger than the {config.max_document_bytes} "
            "byte documentation limit.",
            details={"size_bytes": size, "limit": config.max_document_bytes},
        )

    reference = relative_reference(resolved, config)
    text = resolved.read_text(encoding="utf-8", errors="replace")
    return Document(
        source_type=SourceType.PROJECT_DOCUMENTATION.value,
        source_title=extract_title(text, fallback=reference),
        source_reference=reference,
        content=text,
        metadata={
            "source_type": SourceType.PROJECT_DOCUMENTATION.value,
            "path": reference,
            "filename": resolved.name,
            "byte_size": size,
        },
    )


def load_documentation(config: RagConfig) -> Iterator[Document]:
    """Yield a document for every configured documentation file."""
    for path in discover_documentation_paths(config):
        try:
            yield load_document(path, config)
        except (SourceNotFoundError, UnsafeSourceError) as exc:
            logger.warning("Skipping documentation source: %s", exc.message)


def load_documents_from(
    references: Iterable[str | Path], config: RagConfig
) -> list[Document]:
    """Load a specific set of documentation files.

    Used by the indexer when re-indexing a named subset rather than
    everything.
    """
    documents: list[Document] = []
    for reference in references:
        documents.append(load_document(Path(reference), config))
    return documents


__all__ = [
    "discover_documentation_paths",
    "extract_title",
    "is_forbidden_name",
    "is_forbidden_path",
    "load_document",
    "load_documentation",
    "load_documents_from",
    "relative_reference",
]
