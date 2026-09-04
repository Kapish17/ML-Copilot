"""Domain errors raised by the service layer.

Every error carries a stable machine-readable ``code`` and the HTTP status the
API should answer with. Route handlers never build error responses by hand:
they let these propagate and a single exception handler turns them into the
documented error envelope. Messages are written for a human reading a frontend,
never containing stack traces or internal paths.
"""

from __future__ import annotations

from typing import Any


class MLCopilotError(Exception):
    """Base class for expected, client-facing application errors."""

    code = "internal_error"
    status_code = 500
    #: Response headers this error requires. Almost every error needs none;
    #: a 401 needs ``WWW-Authenticate``, which HTTP requires on that status
    #: and which tells a client *how* to authenticate rather than only that
    #: it must. Declared on the class so the handler needs no special case.
    headers: dict[str, str] | None = None

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        """Store the user-facing message and optional structured context.

        Args:
            message: Explanation safe to show to an API consumer.
            details: Extra machine-readable context (limits, column names).
        """
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = details or {}


class AuthenticationError(MLCopilotError):
    """Base class for a refused credential.

    Both subclasses answer **401** and carry ``WWW-Authenticate: Bearer``.
    Neither ever quotes what was supplied: echoing a credential back puts it
    into the caller's logs, into any proxy in between, and into a browser's
    network panel, and it tells an attacker their guess was received intact.
    """

    code = "authentication_required"
    status_code = 401
    headers = {"WWW-Authenticate": "Bearer"}


class AuthenticationRequiredError(AuthenticationError):
    """No usable credential was presented at all.

    Distinct from :class:`InvalidCredentialsError` on purpose: "you sent
    nothing" and "what you sent is wrong" are different problems for the
    person debugging, and the distinction is not an oracle — it says nothing
    about the key itself, only about the shape of the request.
    """

    code = "authentication_required"


class InvalidCredentialsError(AuthenticationError):
    """A credential was presented and it is not the configured one.

    The message is identical whatever was wrong with it — wrong value, wrong
    length, one character out. Nothing in the response distinguishes a near
    miss from a random guess.
    """

    code = "invalid_credentials"


class DatasetError(MLCopilotError):
    """Base class for dataset ingestion and profiling failures."""

    code = "dataset_error"
    status_code = 400


class UnsupportedFileTypeError(DatasetError):
    """The uploaded file is not one of the supported dataset formats."""

    code = "unsupported_file_type"
    status_code = 415


class FileTooLargeError(DatasetError):
    """The upload exceeds the configured size limit."""

    code = "file_too_large"
    status_code = 413


class EmptyFileError(DatasetError):
    """The upload contains no bytes at all."""

    code = "empty_file"
    status_code = 400


class InvalidDatasetContentError(DatasetError):
    """The bytes are not valid content for the format they were sent as.

    The common parent of every "this file is not what it claims to be"
    failure. Each format keeps its own code beneath it so a client can tell
    a broken spreadsheet from broken JSON, while a caller that only wants to
    know "the content was unusable" can catch one class.
    """

    code = "invalid_dataset_content"
    status_code = 422


class MalformedCSVError(InvalidDatasetContentError):
    """The file could not be parsed as CSV."""

    code = "malformed_csv"
    status_code = 422


class InvalidExcelError(InvalidDatasetContentError):
    """The file could not be read as an ``.xlsx`` workbook.

    Raised for bytes that are not a workbook at all, for a workbook with no
    readable worksheet, and for one the reader rejects. The message describes
    the problem in the caller's terms; nothing from the underlying reader —
    no traceback, no internal object, no path — reaches it.
    """

    code = "invalid_excel"
    status_code = 422


class InvalidJSONError(InvalidDatasetContentError):
    """The file is not JSON, or is JSON that cannot become a table."""

    code = "invalid_json"
    status_code = 422


class MissingHeaderError(DatasetError):
    """The CSV does not start with a usable header row."""

    code = "missing_header"
    status_code = 422


class DuplicateColumnsError(DatasetError):
    """The CSV header repeats one or more column names."""

    code = "duplicate_columns"
    status_code = 422


class EmptyDatasetError(DatasetError):
    """The file parsed successfully but contains no rows or no columns."""

    code = "empty_dataset"
    status_code = 422


class DatasetTooLargeError(DatasetError):
    """The parsed dataset exceeds the configured row or column limits."""

    code = "dataset_too_large"
    status_code = 413


class TargetColumnNotFoundError(DatasetError):
    """The requested target column is not present in the dataset."""

    code = "target_column_not_found"
    status_code = 422
