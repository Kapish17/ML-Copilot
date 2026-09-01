"""Deciding which format an upload claims to be.

Detection answers one narrow question — *which adapter should try these
bytes?* — and it deliberately cannot answer a second one, *are these bytes
really that format?* Only the adapter's own parse can answer that, so
detection is never the security boundary:

* the **extension** is the primary signal, because it is the one part of a
  filename that describes content rather than location;
* the **media type** is a fallback used only when the filename carries no
  usable extension, and it is supplied by the same untrusted client;
* neither is trusted about the *content*. ``report.xlsx`` holding CSV text and
  ``data.csv`` holding a zip both reach an adapter that rejects them, because
  every adapter validates the bytes it is handed before parsing them.

That split is why "never trust the extension alone" is structural here. The
extension only chooses who does the checking.
"""

from __future__ import annotations

from pathlib import PurePosixPath

from app.core.config import Settings
from app.core.errors import UnsupportedFileTypeError
from app.services.datasets.ingestion.formats import (
    EXTENSIONS,
    MEDIA_TYPES,
    DatasetFormat,
)


def supported_formats(settings: Settings) -> tuple[DatasetFormat, ...]:
    """The formats the configured extension allowlist permits.

    The allowlist lives in :class:`~app.core.config.Settings` — the existing
    configuration object — so enabling or disabling a format is a settings
    change and not a code change, and no second configuration system exists.

    Args:
        settings: Active application settings.

    Returns:
        tuple[DatasetFormat, ...]: The permitted formats, in a stable order.
    """
    allowed = {ext.lower() for ext in settings.supported_dataset_extensions}
    return tuple(
        dataset_format
        for extension, dataset_format in EXTENSIONS.items()
        if extension in allowed
    )


def _media_type(content_type: str | None) -> str:
    """Reduce a Content-Type header to its bare media type, lowercased."""
    if not content_type:
        return ""
    return content_type.split(";", 1)[0].strip().lower()


def detect_format(
    filename: str, settings: Settings, content_type: str | None = None
) -> DatasetFormat:
    """Decide which format an upload should be parsed as.

    Args:
        filename: A path-free filename, already reduced by
            :func:`~app.services.datasets.validation.safe_filename`.
        settings: Active application settings, holding the allowlist.
        content_type: The client's declared media type, if any. Consulted
            only when the filename has no usable extension.

    Returns:
        DatasetFormat: The format whose adapter should read the bytes.

    Raises:
        UnsupportedFileTypeError: If neither signal names a permitted format.
            This is the same error, code and status the CSV-only version
            raised, so existing clients see no change.
    """
    permitted = supported_formats(settings)
    extension = PurePosixPath(filename).suffix.lower()

    detected = EXTENSIONS.get(extension)
    if detected is None and not extension:
        # No extension at all: the media type is the only signal left. It is
        # never allowed to *override* an extension, only to stand in for a
        # missing one, so a mislabelled `.csv` is still read as CSV — and
        # still rejected by the CSV adapter if it is not CSV.
        detected = MEDIA_TYPES.get(_media_type(content_type))

    if detected is None or detected not in permitted:
        supported = ", ".join(fmt.extension for fmt in permitted)
        raise UnsupportedFileTypeError(
            f"Unsupported file type {extension or '(none)'}. "
            f"Supported types: {supported}.",
            details={
                "filename": filename,
                "extension": extension,
                "supported_extensions": [fmt.extension for fmt in permitted],
                "supported_formats": [fmt.value for fmt in permitted],
            },
        )
    return detected


__all__ = ["detect_format", "supported_formats"]
