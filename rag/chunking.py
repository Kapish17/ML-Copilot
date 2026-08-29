"""Splitting a document into passages worth retrieving.

Cutting every N characters is the obvious approach and the wrong one: it
severs sentences, separates a table from its header, and produces passages
that no longer say what they are about. A retrieved chunk has to stand alone,
because a future model will see it without the document around it.

So the splitter follows the document's own structure:

1. **Sections.** Markdown headings mark where a subject changes. A section
   becomes a chunk when it fits, and its heading path is carried on every
   chunk it produces — a passage about one-hot encoding stays labelled
   "Preprocessing › Feature groups" even when read alone.
2. **Paragraphs.** A section too long for one chunk is split on blank lines,
   never mid-paragraph, and paragraphs are packed until the next would
   overflow.
3. **Hard wrapping, last.** A single paragraph longer than the limit — a long
   table, a wide code block — is cut on a line boundary where possible, and
   only then at a character offset.

Two further rules earn their keep. **Fenced code blocks are never split**: the
fence markers have to stay with their content or the passage becomes nonsense.
And **tiny fragments are merged** into their neighbour, because a heading with
one line under it retrieves nothing useful and dilutes the ranking.

Overlap repeats the tail of one chunk at the head of the next, so a sentence
that falls across a boundary is still findable from either side.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag.citations import citation_for_heading_path
from rag.config import RagConfig
from rag.documents import Chunk, Document, make_chunk_id

#: An ATX heading: one to six hashes, a space, then the text.
HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
#: A fenced code block delimiter, backticks or tildes.
FENCE_PATTERN = re.compile(r"^\s*(```|~~~)")
#: A blank line, i.e. a paragraph boundary.
BLANK_LINE = re.compile(r"\n\s*\n")

#: Joins a heading path for display: "Preprocessing › Feature groups".
HEADING_SEPARATOR = " › "


@dataclass(frozen=True)
class Section:
    """A run of text under one heading path."""

    heading_path: tuple[str, ...]
    content: str

    @property
    def title(self) -> str:
        """The heading path rendered for a human."""
        return HEADING_SEPARATOR.join(self.heading_path)


def _strip_heading_marks(line: str) -> tuple[int, str] | None:
    """Return ``(level, text)`` when a line is a heading, else ``None``."""
    match = HEADING_PATTERN.match(line)
    if match is None:
        return None
    return len(match.group(1)), match.group(2).strip()


def split_sections(text: str) -> list[Section]:
    """Split Markdown into sections, each labelled with its heading path.

    Headings inside fenced code blocks are ignored — a ``# comment`` in a
    shell example is not a section.

    Args:
        text: The document's Markdown.

    Returns:
        list[Section]: Sections in reading order. Text before the first
        heading becomes a section with an empty path.
    """
    sections: list[Section] = []
    path: list[str] = []
    buffer: list[str] = []
    in_fence = False
    fence_marker = ""

    def flush() -> None:
        """Close the current buffer into a section, if it holds anything."""
        content = "\n".join(buffer).strip()
        if content:
            sections.append(Section(heading_path=tuple(path), content=content))
        buffer.clear()

    for line in text.splitlines():
        fence = FENCE_PATTERN.match(line)
        if fence:
            marker = fence.group(1)
            if not in_fence:
                in_fence, fence_marker = True, marker
            elif marker == fence_marker:
                in_fence, fence_marker = False, ""
            buffer.append(line)
            continue

        heading = None if in_fence else _strip_heading_marks(line)
        if heading is None:
            buffer.append(line)
            continue

        level, title = heading
        flush()
        # A level-3 heading sits under the nearest level-2, and replaces any
        # previous level-3. Truncating to `level - 1` does both.
        del path[level - 1 :]
        while len(path) < level - 1:
            # A document that jumps from H1 to H3 leaves a gap; fill it so the
            # path length keeps meaning "depth".
            path.append("")
        path.append(title)

    flush()
    return sections


def split_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs, keeping fenced code blocks whole."""
    paragraphs: list[str] = []
    buffer: list[str] = []
    in_fence = False
    fence_marker = ""

    def flush() -> None:
        """Close the buffer into a paragraph, if it holds anything."""
        content = "\n".join(buffer).strip()
        if content:
            paragraphs.append(content)
        buffer.clear()

    for line in text.splitlines():
        fence = FENCE_PATTERN.match(line)
        if fence:
            marker = fence.group(1)
            if not in_fence:
                # A fence starts: the paragraph before it ends here.
                flush()
                in_fence, fence_marker = True, marker
                buffer.append(line)
                continue
            buffer.append(line)
            if marker == fence_marker:
                in_fence, fence_marker = False, ""
                flush()
            continue

        if not in_fence and not line.strip():
            flush()
            continue
        buffer.append(line)

    flush()
    return paragraphs


def _hard_split(text: str, limit: int) -> list[str]:
    """Split one oversized paragraph, preferring line boundaries."""
    if len(text) <= limit:
        return [text]

    pieces: list[str] = []
    remaining = text
    while len(remaining) > limit:
        window = remaining[:limit]
        cut = window.rfind("\n")
        if cut < limit // 2:
            # No usable line boundary in the back half; cut at the limit
            # rather than emit a chunk that is mostly empty.
            cut = limit
        pieces.append(remaining[:cut].strip())
        remaining = remaining[cut:].lstrip("\n")
    if remaining.strip():
        pieces.append(remaining.strip())
    return [piece for piece in pieces if piece]


def _overlap_tail(text: str, overlap: int) -> str:
    """Return the last ``overlap`` characters, starting at a word boundary."""
    if overlap <= 0 or len(text) <= overlap:
        return ""
    tail = text[-overlap:]
    space = tail.find(" ")
    return tail[space + 1 :] if space != -1 else tail


def pack_paragraphs(
    paragraphs: list[str], *, chunk_size: int, overlap: int
) -> list[str]:
    """Group paragraphs into chunks no larger than ``chunk_size``.

    Paragraphs are added until the next would overflow; the next chunk then
    begins with the tail of the previous one, so a boundary never hides a
    sentence from both sides.
    """
    chunks: list[str] = []
    current: list[str] = []
    length = 0

    def flush() -> None:
        """Close the current chunk."""
        nonlocal length
        if current:
            chunks.append("\n\n".join(current).strip())
            current.clear()
            length = 0

    for paragraph in paragraphs:
        for piece in _hard_split(paragraph, chunk_size):
            addition = len(piece) + (2 if current else 0)
            if current and length + addition > chunk_size:
                previous = "\n\n".join(current)
                flush()
                tail = _overlap_tail(previous, overlap)
                if tail:
                    current.append(tail)
                    length = len(tail)
            current.append(piece)
            length += len(piece) + (2 if length else 0)

    flush()
    return [chunk for chunk in chunks if chunk]


def _merge_small(pieces: list[str], *, minimum: int, chunk_size: int) -> list[str]:
    """Fold undersized pieces into a neighbour where there is room."""
    if minimum <= 0:
        return pieces

    merged: list[str] = []
    for piece in pieces:
        if (
            merged
            and len(piece) < minimum
            and len(merged[-1]) + len(piece) + 2 <= chunk_size
        ):
            merged[-1] = f"{merged[-1]}\n\n{piece}"
            continue
        merged.append(piece)

    # A single leading fragment has no earlier neighbour; give it the next one.
    if len(merged) > 1 and len(merged[0]) < minimum:
        if len(merged[0]) + len(merged[1]) + 2 <= chunk_size:
            merged[1] = f"{merged[0]}\n\n{merged[1]}"
            merged.pop(0)
    return merged


def chunk_markdown(text: str, config: RagConfig) -> list[tuple[tuple[str, ...], str]]:
    """Split Markdown into ``(heading_path, content)`` passages.

    Args:
        text: The document's Markdown.
        config: Supplies the chunk size, overlap and minimum size.

    Returns:
        list: One entry per chunk, in reading order.
    """
    passages: list[tuple[tuple[str, ...], str]] = []
    for section in split_sections(text):
        pieces = pack_paragraphs(
            split_paragraphs(section.content),
            chunk_size=config.chunk_size,
            overlap=config.chunk_overlap,
        )
        pieces = _merge_small(
            pieces, minimum=config.min_chunk_size, chunk_size=config.chunk_size
        )
        passages.extend((section.heading_path, piece) for piece in pieces)

    if not passages:
        stripped = text.strip()
        if stripped:
            passages.append(((), stripped))
    return passages


def _with_heading_context(heading_path: tuple[str, ...], content: str) -> str:
    """Prefix a passage with its heading path, unless it already starts there.

    The heading is the strongest short description of what a passage is about,
    and a retrieved chunk is read without the document around it — so the
    context travels with the text, not only in the metadata.
    """
    labelled = [part for part in heading_path if part]
    if not labelled:
        return content
    header = HEADING_SEPARATOR.join(labelled)
    if content.lstrip().startswith(header):
        return content
    return f"{header}\n\n{content}"


def chunk_document(document: Document, config: RagConfig) -> list[Chunk]:
    """Split a document into retrievable, citable chunks.

    Args:
        document: The source to split.
        config: Chunking settings.

    Returns:
        list[Chunk]: Chunks in reading order, each carrying its document's
        identity, its heading path and a citation that resolves to it.
    """
    chunks: list[Chunk] = []
    for position, (heading_path, content) in enumerate(
        chunk_markdown(document.content, config)
    ):
        text = _with_heading_context(heading_path, content)
        metadata = dict(document.metadata)
        metadata["position"] = position
        if heading_path:
            metadata["heading"] = heading_path[-1]
            metadata["heading_path"] = [part for part in heading_path if part]
        chunks.append(
            Chunk(
                document_id=document.document_id,
                chunk_id=make_chunk_id(document.document_id, position, text),
                content=text,
                source_type=document.source_type,
                source_title=document.source_title,
                source_reference=document.source_reference,
                position=position,
                heading_path=heading_path,
                citation=citation_for_heading_path(
                    source_type=document.source_type,
                    source_reference=document.source_reference,
                    heading_path=heading_path,
                ),
                metadata=metadata,
            )
        )
    return chunks


__all__ = [
    "HEADING_SEPARATOR",
    "Section",
    "chunk_document",
    "chunk_markdown",
    "pack_paragraphs",
    "split_paragraphs",
    "split_sections",
]
