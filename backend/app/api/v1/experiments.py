"""Experiment endpoints.

Each handler does the same four things and nothing else: take the request, hand
it to a service, validate the structured result against a response schema,
return it. The orchestration lives in
:mod:`app.services.experiments.runner`, the querying in
:mod:`app.services.experiments.history`, and the machine learning in ``ml/`` —
no route here computes a statistic, fits a model or reads a file.

Failures propagate. Dataset errors, preprocessing errors, model errors and
experiment errors are all turned into the one documented envelope by the
application's exception handlers, so no handler builds an error response by
hand.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, Path, Query, UploadFile, status

from app.api.dependencies import (
    ExperimentHistoryDep,
    ExperimentRunnerDep,
    PredictionServiceDep,
    SettingsDep,
)
from app.api.security import UNAUTHORIZED_RESPONSE, Protected
from app.api.v1.experiment_form import ExperimentOptionsDep
from app.schemas.errors import ErrorResponse
from app.schemas.experiment import (
    ExperimentCapabilitiesResponse,
    ExperimentComparisonResponse,
    ExperimentCompareRequest,
    ExperimentHeadline,
    ExperimentListResponse,
    ExperimentRecord,
    ExperimentRunResponse,
    ModelInfo,
)
from app.schemas.prediction import (
    ModelAvailability,
    PredictionRequest,
    PredictionResponse,
)
from app.services.experiments.history import SORT_KEYS, SORT_ORDERS
from app.services.experiments.options import SELECTION_STRATEGIES
from ml.evaluation.metrics import CLASSIFICATION_METRICS, REGRESSION_METRICS
from ml.models.registry import list_available_models

router = APIRouter(prefix="/experiments", tags=["experiments"])

_RUN_ERRORS: dict[int | str, dict[str, object]] = {
    # Every route below is protected, so each can be refused before it
    # runs. Documented here rather than left for a reader to discover.
    **UNAUTHORIZED_RESPONSE,
    status.HTTP_400_BAD_REQUEST: {
        "model": ErrorResponse,
        "description": "The configuration is invalid — unknown model, metric, strategy or fold count.",
    },
    status.HTTP_409_CONFLICT: {
        "model": ErrorResponse,
        "description": "The request conflicts with the data, e.g. a classifier for a regression target.",
    },
    status.HTTP_413_CONTENT_TOO_LARGE: {
        "model": ErrorResponse,
        "description": "The upload or the dataset exceeds a configured limit.",
    },
    status.HTTP_415_UNSUPPORTED_MEDIA_TYPE: {
        "model": ErrorResponse,
        "description": (
            "The file is not a supported dataset format. CSV, Excel (.xlsx) "
            "and JSON are implemented; Parquet, SQL and remote sources are not."
        ),
    },
    status.HTTP_422_UNPROCESSABLE_CONTENT: {
        "model": ErrorResponse,
        "description": "The request or the dataset content could not be processed.",
    },
}

_LOOKUP_ERRORS: dict[int | str, dict[str, object]] = {
    # Every route below is protected, so each can be refused before it
    # runs. Documented here rather than left for a reader to discover.
    **UNAUTHORIZED_RESPONSE,
    status.HTTP_400_BAD_REQUEST: {
        "model": ErrorResponse,
        "description": "The experiment id is malformed.",
    },
    status.HTTP_404_NOT_FOUND: {
        "model": ErrorResponse,
        "description": "No experiment is stored under that id.",
    },
    status.HTTP_500_INTERNAL_SERVER_ERROR: {
        "model": ErrorResponse,
        "description": "The stored record could not be read.",
    },
}

_COMPARE_ERRORS: dict[int | str, dict[str, object]] = {
    # Every route below is protected, so each can be refused before it
    # runs. Documented here rather than left for a reader to discover.
    **UNAUTHORIZED_RESPONSE,
    status.HTTP_400_BAD_REQUEST: {
        "model": ErrorResponse,
        "description": "Fewer than two ids, too many ids, or a malformed id.",
    },
    status.HTTP_404_NOT_FOUND: {
        "model": ErrorResponse,
        "description": "One of the experiments is not stored.",
    },
    status.HTTP_409_CONFLICT: {
        "model": ErrorResponse,
        "description": "The experiments do not share a task and metric, so they cannot be ranked.",
    },
}


@router.post(
    "/run",
    dependencies=[Protected],
    response_model=ExperimentRunResponse,
    responses=_RUN_ERRORS,
    summary="Run a complete experiment on an uploaded dataset",
)
async def run_experiment_endpoint(
    runner: ExperimentRunnerDep,
    file: Annotated[
        UploadFile,
        File(
            description=(
                "Dataset to run on. CSV, Excel (.xlsx — first worksheet) or "
                "JSON (an array of objects, or an object holding one such "
                "array)."
            )
        ),
    ],
    options: ExperimentOptionsDep,
) -> ExperimentRunResponse:
    """Profile, prepare, cross-validate, select, evaluate, explain and record.

    The dataset is validated and parsed in memory, profiled, turned into a
    leakage-safe train/test split, and every candidate model is
    cross-validated on the **training rows only**. The winner is retrained on
    the full training data and measured **once** on the untouched test set,
    then explained with SHAP. The whole run is stored as an experiment record
    and returned here.

    CSV, Excel and JSON uploads all run the identical pipeline: the ingestion
    adapter turns each into the same standardised table, and nothing after
    that point knows which format it was. The record notes the format under
    ``dataset.source_format``, while the run's identity is the content
    fingerprint — so the same data uploaded as CSV and as JSON produces the
    same fingerprint and is recognisably the same dataset.

    The upload is never written to disk. What is stored is the record: the
    dataset's content fingerprint, shape and column types, the preprocessing
    decisions, the scores and the explanation — never the data, the fitted
    pipeline or the explainer.

    Execution is synchronous: the response arrives when the run has finished.
    """
    result = await runner.run_upload(
        file,
        filename=file.filename,
        options=options,
        content_type=file.content_type,
    )
    return ExperimentRunResponse.model_validate(result.as_dict())


@router.get(
    "",
    dependencies=[Protected],
    response_model=ExperimentListResponse,
    responses={
        **UNAUTHORIZED_RESPONSE,
        status.HTTP_400_BAD_REQUEST: {
            "model": ErrorResponse,
            "description": "A filter, sort key, order or limit is invalid.",
        }
    },
    summary="List stored experiments",
)
def list_experiments(
    history: ExperimentHistoryDep,
    settings: SettingsDep,
    dataset_fingerprint: Annotated[
        str | None,
        Query(description="Only runs on this dataset, identified by content."),
    ] = None,
    target_column: Annotated[
        str | None, Query(description="Only runs predicting this column.")
    ] = None,
    task_type: Annotated[
        str | None, Query(description="'classification' or 'regression'.")
    ] = None,
    model_name: Annotated[
        str | None, Query(description="Only runs whose winner was this model.")
    ] = None,
    strategy: Annotated[
        str | None,
        Query(description="'cross_validation' or 'holdout'.", examples=list(SELECTION_STRATEGIES)),
    ] = None,
    primary_metric: Annotated[
        str | None, Query(description="Only runs judged by this metric.")
    ] = None,
    tags: Annotated[
        list[str] | None,
        Query(description="Only runs carrying every one of these tags."),
    ] = None,
    sort_by: Annotated[
        str | None,
        Query(description="Sort key.", examples=list(SORT_KEYS)),
    ] = None,
    order: Annotated[
        str | None,
        Query(
            description=(
                "'desc' (default) is best or newest first; sorting by "
                "primary_metric reads the metric's own direction, so the "
                "largest F1 leads but the smallest RMSE does."
            ),
            examples=list(SORT_ORDERS),
        ),
    ] = None,
    limit: Annotated[
        int | None, Query(description="Maximum number of experiments to return.")
    ] = None,
) -> ExperimentListResponse:
    """Return the stored experiments matching every filter given.

    All filters are optional and combine with "and". Records live in local
    JSON files; no database and no MLflow tracking server is involved.
    """
    runs = history.list(
        dataset_fingerprint=dataset_fingerprint,
        target_column=target_column,
        task_type=task_type,
        model_name=model_name,
        selection_strategy=strategy,
        primary_metric=primary_metric,
        tags=tags or (),
        sort_by=sort_by,
        order=order,
        limit=limit,
    )
    return ExperimentListResponse(
        count=len(runs),
        limit=limit if limit is not None else settings.experiment_page_limit,
        experiments=[
            ExperimentHeadline.model_validate(run.headline()) for run in runs
        ],
    )


@router.get(
    "/capabilities",
    response_model=ExperimentCapabilitiesResponse,
    summary="List the models, metrics and limits an experiment may use",
)
def experiment_capabilities(settings: SettingsDep) -> ExperimentCapabilitiesResponse:
    """Describe what a valid experiment request may contain.

    Returned from the model registry and the configured limits, so a client
    never has to hard-code an identifier the server already knows. No model is
    trained by this call.
    """
    return ExperimentCapabilitiesResponse(
        models=[ModelInfo.model_validate(item) for item in list_available_models()],
        metrics={
            "classification": [item.key for item in CLASSIFICATION_METRICS],
            "regression": [item.key for item in REGRESSION_METRICS],
        },
        strategies=list(SELECTION_STRATEGIES),
        sort_keys=list(SORT_KEYS),
        limits={
            "max_upload_bytes": settings.max_upload_bytes,
            "max_experiment_rows": settings.max_experiment_rows,
            "max_experiment_feature_columns": settings.max_experiment_feature_columns,
            "min_cv_folds": settings.min_cv_folds,
            "max_cv_folds": settings.max_cv_folds,
            "default_cv_folds": settings.default_cv_folds,
            "max_candidate_models": settings.max_candidate_models,
            "max_comparison_experiments": settings.max_comparison_experiments,
            "max_experiment_page_limit": settings.max_experiment_page_limit,
            "explanation_rows": settings.explanation_rows,
            "explanation_top_features": settings.explanation_top_features,
        },
        supported_dataset_extensions=list(settings.supported_dataset_extensions),
    )


@router.post(
    "/compare",
    dependencies=[Protected],
    response_model=ExperimentComparisonResponse,
    responses=_COMPARE_ERRORS,
    summary="Rank several stored experiments against each other",
)
def compare_experiments_endpoint(
    history: ExperimentHistoryDep, request: ExperimentCompareRequest
) -> ExperimentComparisonResponse:
    """Compare stored experiments on the one metric they share.

    Runs judged by different metrics, or solving different tasks, are refused
    with a 409 rather than ranked: an RMSE and an F1 do not belong in the same
    column. Ordering reads the shared metric's own declared direction, and a
    run with no score sorts last rather than winning by accident.
    """
    comparison = history.compare(request.experiment_ids)
    payload = comparison.summary()
    payload["higher_is_better"] = comparison.higher_is_better
    payload["table"] = comparison.as_text()
    return ExperimentComparisonResponse.model_validate(payload)


@router.get(
    "/{experiment_id}",
    dependencies=[Protected],
    response_model=ExperimentRecord,
    responses=_LOOKUP_ERRORS,
    summary="Fetch one stored experiment",
)
def get_experiment(
    history: ExperimentHistoryDep,
    experiment_id: Annotated[
        str, Path(description="Identifier returned when the experiment ran.")
    ],
) -> ExperimentRecord:
    """Return one stored experiment record.

    The record is exactly what the run returned, minus the execution metadata:
    identity, dataset fingerprint, preprocessing, model selection, the final
    evaluation, the explanation and the environment.
    """
    return ExperimentRecord.model_validate(history.get(experiment_id).to_dict())


_MODEL_ERRORS: dict[int | str, dict[str, object]] = {
    **UNAUTHORIZED_RESPONSE,
    status.HTTP_400_BAD_REQUEST: {
        "model": ErrorResponse,
        "description": "The experiment id is malformed.",
    },
    status.HTTP_404_NOT_FOUND: {
        "model": ErrorResponse,
        "description": "No experiment is stored under that id.",
    },
}

_PREDICT_ERRORS: dict[int | str, dict[str, object]] = {
    **_MODEL_ERRORS,
    status.HTTP_409_CONFLICT: {
        "model": ErrorResponse,
        "description": (
            "`model_not_available` — the experiment exists but has no stored "
            "model. A run recorded before model persistence existed, or one "
            "whose artifact has been removed. Re-run the experiment to make "
            "one. Deliberately not a 404: the run itself is fine."
        ),
    },
    status.HTTP_422_UNPROCESSABLE_CONTENT: {
        "model": ErrorResponse,
        "description": (
            "`invalid_prediction_input` — a record is missing a feature the "
            "model needs, carries one it was not trained on, or holds a value "
            "that is not the kind of thing the column expects. The details "
            "name the record and the columns."
        ),
    },
    status.HTTP_500_INTERNAL_SERVER_ERROR: {
        "model": ErrorResponse,
        "description": "The stored model could not be read.",
    },
}


@router.get(
    "/{experiment_id}/model",
    dependencies=[Protected],
    response_model=ModelAvailability,
    responses=_MODEL_ERRORS,
    summary="Report whether an experiment can be predicted from",
)
def experiment_model(
    predictions: PredictionServiceDep,
    experiment_id: Annotated[
        str, Path(description="Identifier returned when the experiment ran.")
    ],
) -> ModelAvailability:
    """Describe the stored model, or say why there is none.

    Answered from the **artifact store**, not from the stored record. The
    record notes that a model was written when the run finished; this says
    whether one is there now, which is a different question once an artifact
    can be deleted or a volume wiped.

    When a model is present the response carries the exact feature schema a
    prediction must satisfy — each column's name, the branch of the fitted
    preprocessing that handles it, and the dtype it had at training time — so
    a client builds its form from what the model actually wants rather than
    from a guess. Only the manifest is read; the model itself is not
    deserialised, so asking is cheap.

    **No filesystem location appears in the response**, and none is available
    to appear: nothing in this path handles one.
    """
    return ModelAvailability.model_validate(predictions.describe(experiment_id))


@router.post(
    "/{experiment_id}/predict",
    dependencies=[Protected],
    response_model=PredictionResponse,
    responses=_PREDICT_ERRORS,
    summary="Predict from the model this experiment trained",
)
def predict_from_experiment(
    predictions: PredictionServiceDep,
    experiment_id: Annotated[
        str, Path(description="Identifier returned when the experiment ran.")
    ],
    request: PredictionRequest,
) -> PredictionResponse:
    """Predict for one or more records, using the model this run produced.

    The model is chosen by the id in the URL and by nothing else. **No path,
    filename or artifact reference is accepted from a request**, so there is
    nothing for a caller to point at another file; the id is validated before
    it addresses either a record or a directory.

    The prediction runs through the **same fitted pipeline the experiment
    produced** — the preprocessing that learned its statistics from the
    training rows, with the winning estimator behind it. Nothing is refitted,
    which is what makes a prediction made today comparable to the held-out
    score that run reported.

    One record or a thousand take the same shape: `records` is always a list,
    and `predictions` comes back in the same order and the same length, with
    each item carrying the index of the record that produced it. A classifier
    whose estimator supplies them also returns a probability per class.

    A feature the model was not trained on is **refused**, not ignored. Such a
    column is dropped rather than used, so a misspelt name would otherwise
    produce a confident prediction made without the value the caller believed
    they supplied.
    """
    return PredictionResponse.model_validate(
        predictions.predict(experiment_id, request.records)
    )
