"""Dataset endpoints.

The handler stays deliberately thin: it accepts the upload, hands it to the
dataset service and returns the result. Validation, parsing and analysis all
live in the service layer, and failures propagate as typed domain errors that
the application's exception handlers turn into the standard error envelope.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, Form, UploadFile, status

from app.api.dependencies import DatasetServiceDep
from app.api.security import UNAUTHORIZED_RESPONSE, Protected
from app.schemas.dataset import DatasetProfileResponse
from app.schemas.errors import ErrorResponse

router = APIRouter(prefix="/datasets", tags=["datasets"])

_ERROR_RESPONSES: dict[int | str, dict[str, object]] = {
    # Every route below is protected, so each can be refused before it
    # runs. Documented here rather than left for a reader to discover.
    **UNAUTHORIZED_RESPONSE,
    status.HTTP_413_CONTENT_TOO_LARGE: {
        "model": ErrorResponse,
        "description": "The upload or the parsed dataset exceeds a configured limit.",
    },
    status.HTTP_415_UNSUPPORTED_MEDIA_TYPE: {
        "model": ErrorResponse,
        "description": (
            "The file is not a supported dataset format. CSV, Excel (.xlsx) "
            "and JSON are supported."
        ),
    },
    status.HTTP_422_UNPROCESSABLE_CONTENT: {
        "model": ErrorResponse,
        "description": (
            "The request could not be processed, or the file's content is "
            "not valid for the format it was sent as."
        ),
    },
    status.HTTP_400_BAD_REQUEST: {
        "model": ErrorResponse,
        "description": "The upload is empty or otherwise unusable.",
    },
}


@router.post(
    "/profile",
    dependencies=[Protected],
    response_model=DatasetProfileResponse,
    responses=_ERROR_RESPONSES,
    summary="Profile a CSV, Excel or JSON dataset",
)
async def profile_dataset(
    service: DatasetServiceDep,
    file: Annotated[
        UploadFile,
        File(
            description=(
                "Dataset to profile. CSV, Excel (.xlsx — first worksheet) or "
                "JSON (an array of objects, or an object holding one such "
                "array)."
            )
        ),
    ],
    target_column: Annotated[
        str | None,
        Form(description="Optional name of the column to analyse as the target."),
    ] = None,
) -> DatasetProfileResponse:
    """Validate an uploaded dataset and return its profile.

    CSV, Excel (``.xlsx``) and JSON are accepted. The file's format decides
    only which adapter reads it: every format becomes the same standardised
    table first, and the profile is computed by one implementation that cannot
    tell them apart. The response reports the format under ``source_format``
    as context.

    Excel workbooks are read from their **first worksheet**. JSON must be an
    array of objects, one per row, or an object holding one such array.

    The file is processed in memory and is not stored.
    """
    return await service.profile_upload(
        file,
        filename=file.filename,
        target_column=target_column,
        content_type=file.content_type,
    )
