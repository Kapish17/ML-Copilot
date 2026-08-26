"""Validation of incoming dataset uploads.

This module answers a single question: is it safe to try parsing these bytes?
It never touches the filesystem and never trusts the client-supplied filename
for anything other than its extension.
"""

from __future__ import annotations

from pathlib import PurePosixPath, PureWindowsPath
from typing import Protocol

from app.core.config import Settings
from app.core.errors import (
    EmptyFileError,
    FileTooLargeError,
    UnsupportedFileTypeError,
)

UPLOAD_CHUNK_SIZE = 1024 * 1024


class AsyncReadable(Protocol):
    """Minimal interface of the parts of ``UploadFile`` this module uses."""

    async def read(self, size: int = -1) -> bytes:  # pragma: no cover - protocol
        ...


def safe_filename(filename: str | None) -> str:
    """Reduce a client-supplied filename to a bare, path-free name.

    Directory components are stripped so a crafted name such as
    ``../../etc/passwd`` cannot influence anything downstream. Both POSIX and
    Windows separators are handled, since the client may run on either.

    Args:
        filename: The raw filename reported by the client, if any.

    Returns:
        str: A filename with no directory component, or ``"upload"`` when the
        client sent nothing usable.
    """
    if not filename:
        return "upload"
    name = PureWindowsPath(PurePosixPath(filename).name).name.strip()
    return name or "upload"


def validate_extension(filename: str, settings: Settings) -> str:
    """Check the file extension against the supported dataset formats.

    Args:
        filename: A path-free filename.
        settings: Active application settings.

    Returns:
        str: The lowercase extension, including the leading dot.

    Raises:
        UnsupportedFileTypeError: If the extension is missing or unsupported.
    """
    extension = PurePosixPath(filename).suffix.lower()
    if extension not in settings.supported_dataset_extensions:
        supported = ", ".join(settings.supported_dataset_extensions)
        raise UnsupportedFileTypeError(
            f"Unsupported file type {extension or '(none)'}. Supported types: {supported}.",
            details={
                "filename": filename,
                "extension": extension,
                "supported_extensions": list(settings.supported_dataset_extensions),
            },
        )
    return extension


def validate_size(size_bytes: int, settings: Settings) -> None:
    """Reject uploads that are empty or larger than the configured limit.

    Args:
        size_bytes: Number of bytes received.
        settings: Active application settings.

    Raises:
        EmptyFileError: If the upload contains no bytes.
        FileTooLargeError: If the upload exceeds ``max_upload_bytes``.
    """
    if size_bytes <= 0:
        raise EmptyFileError("The uploaded file is empty.")
    if size_bytes > settings.max_upload_bytes:
        raise FileTooLargeError(
            f"File is larger than the {settings.max_upload_mb:.1f} MB upload limit.",
            details={
                "size_bytes": size_bytes,
                "max_upload_bytes": settings.max_upload_bytes,
            },
        )


async def read_upload(upload: AsyncReadable, settings: Settings) -> bytes:
    """Read an upload into memory, stopping as soon as the limit is exceeded.

    Reading in chunks means an oversized upload is rejected after roughly one
    chunk more than the limit, rather than being fully buffered first.

    Args:
        upload: The incoming file, typically a FastAPI ``UploadFile``.
        settings: Active application settings.

    Returns:
        bytes: The complete file content.

    Raises:
        EmptyFileError: If the upload contains no bytes.
        FileTooLargeError: If the upload exceeds ``max_upload_bytes``.
    """
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(UPLOAD_CHUNK_SIZE)
        if not chunk:
            break
        total += len(chunk)
        if total > settings.max_upload_bytes:
            raise FileTooLargeError(
                f"File is larger than the {settings.max_upload_mb:.1f} MB upload limit.",
                details={"max_upload_bytes": settings.max_upload_bytes},
            )
        chunks.append(chunk)

    if total == 0:
        raise EmptyFileError("The uploaded file is empty.")
    return b"".join(chunks)
