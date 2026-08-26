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
from app.schemas.dataset import DatasetProfileResponse
from app.schemas.errors import ErrorResponse

router = APIRouter(prefix="/datasets", tags=["datasets"])

_ERROR_RESPONSES: dict[int | str, dict[str, object]] = {
    status.HTTP_413_CONTENT_TOO_LARGE: {
        "model": ErrorResponse,
        "description": "The upload or the parsed dataset exceeds a configured limit.",
    },
    status.HTTP_415_UNSUPPORTED_MEDIA_TYPE: {
        "model": ErrorResponse,
        "description": "The file is not a supported dataset format.",
    },
    status.HTTP_422_UNPROCESSABLE_CONTENT: {
        "model": ErrorResponse,
        "description": "The request or the CSV content could not be processed.",
    },
    status.HTTP_400_BAD_REQUEST: {
        "model": ErrorResponse,
        "description": "The upload is empty or otherwise unusable.",
    },
}


@router.post(
    "/profile",
    response_model=DatasetProfileResponse,
    responses=_ERROR_RESPONSES,
    summary="Profile a CSV dataset",
)
async def profile_dataset(
    service: DatasetServiceDep,
    file: Annotated[UploadFile, File(description="CSV dataset to profile.")],
    target_column: Annotated[
        str | None,
        Form(description="Optional name of the column to analyse as the target."),
    ] = None,
) -> DatasetProfileResponse:
    """Validate an uploaded CSV file and return its profile.

    The file is processed in memory and is not stored.
    """
    return await service.profile_upload(
        file, filename=file.filename, target_column=target_column
    )
