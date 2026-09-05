"""Request and response contracts for predicting from a stored model.

One shape for one record and for a thousand: a request always carries a
``records`` list, and a response always carries a ``predictions`` list of the
same length in the same order. A single prediction is a batch of one, which
means a client writes one code path rather than two and a caller never has to
discover that the response shape changed under them at n = 2.

Every field a caller sends is a **feature value**. There is no path, no
filename, no model reference, no serialisation format and no code — the model
is chosen by the experiment id in the URL, which the store validates before it
addresses anything.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, JsonValue

#: Ceiling on what the schema itself will accept, independent of the server's
#: configured limit. A caller sending a million records should be refused by
#: request validation before a list of a million dictionaries is built.
MAX_RECORDS_HARD_LIMIT = 1000


class PredictionRequest(BaseModel):
    """The records to predict on."""

    model_config = ConfigDict(extra="forbid")

    records: list[dict[str, JsonValue]] = Field(
        ...,
        min_length=1,
        max_length=MAX_RECORDS_HARD_LIMIT,
        description=(
            "One object per row, each mapping a feature name to its value. "
            "Every feature the model was trained on must be present; a feature "
            "it was not trained on is refused rather than ignored, because a "
            "column the model does not know is dropped rather than used and a "
            "misspelt name would otherwise produce a confident prediction made "
            "without it. `null` is accepted for a missing value — the model's "
            "imputation was fitted for exactly that. Ask "
            "`GET /api/v1/experiments/{experiment_id}/model` for the expected "
            "feature names and kinds."
        ),
        examples=[
            [
                {
                    "tenure_months": 30,
                    "monthly_spend": 34.89,
                    "support_tickets": 2,
                    "satisfaction_score": 6.7,
                }
            ]
        ],
    )


class PredictedFeature(BaseModel):
    """One column the model expects, and how it is treated."""

    name: str = Field(..., description="The column's name, exactly as trained on.")
    kind: str = Field(
        ...,
        description=(
            "Which branch of the fitted preprocessing handles it: `numeric`, "
            "`categorical`, `boolean` or `datetime`. What a value is checked "
            "against before it reaches the model."
        ),
    )
    dtype: str = Field(
        ..., description="The pandas dtype the column had at training time."
    )


class PredictedModel(BaseModel):
    """Which model produced a prediction, and what it was trained on.

    Everything here is about the model. **No filesystem location appears**, and
    none is available to appear: the service never handles one.
    """

    experiment_id: str
    created_at: str = Field(..., description="When the model artifact was written.")
    model_name: str = Field(..., description="Registry identifier of the estimator.")
    display_name: str
    task_type: str = Field(..., description="`classification` or `regression`.")
    target_column: str = Field(..., description="The column this model predicts.")
    classes: list[JsonValue] = Field(
        default_factory=list,
        description=(
            "The class labels a classifier can return, in the order its "
            "probabilities are given. Empty for regression."
        ),
    )
    features: list[PredictedFeature] = Field(default_factory=list)
    train_row_count: int
    test_row_count: int | None = Field(
        None,
        description=(
            "How many held-out rows the metric below was measured on. A score "
            "without its sample size overstates what is known about it."
        ),
    )
    primary_metric: str
    primary_metric_value: float | None = Field(
        None,
        description=(
            "The winner's score on the **held-out test set**, carried so a "
            "caller can see how much to trust the model. It is a measurement "
            "of the model over many rows, **not a confidence in this "
            "prediction** — nothing here scores an individual answer."
        ),
    )
    supports_probabilities: bool = Field(
        False,
        description=(
            "Whether this model can report a probability per class. False for "
            "regression, and for a classifier whose estimator does not provide "
            "them — in which case `probabilities` is null on every item."
        ),
    )
    artifact_schema_version: str | None = Field(
        None, description="Which manifest shape the stored artifact uses."
    )


class PredictionItem(BaseModel):
    """One record's result."""

    index: int = Field(
        ...,
        description=(
            "Position of the record in the submitted batch. Results are "
            "returned in submission order, and this makes that checkable "
            "rather than assumed."
        ),
    )
    prediction: JsonValue = Field(
        ...,
        description=(
            "The predicted class label for a classifier, or the predicted "
            "number for a regressor."
        ),
    )
    probabilities: dict[str, float] | None = Field(
        None,
        description=(
            "Probability per class label, for a classifier whose estimator "
            "provides them. `null` for regression, and for a classifier that "
            "does not. **These are the model's own outputs**, produced by the "
            "same fitted estimator that chose the label — useful for seeing "
            "how close a decision was, and not a calibrated statement about "
            "how often the model is right in the world."
        ),
    )


class PredictionResponse(BaseModel):
    """Predictions for a batch, and the model that made them."""

    predictions: list[PredictionItem]
    prediction_count: int
    model: PredictedModel


class ModelAvailability(BaseModel):
    """Whether an experiment can be predicted from, and with what.

    Answered from the **artifact store**, not from the experiment record: the
    record says a model was written when the run finished, and this says
    whether one is usable now.

    When no model is usable every descriptive field is null and ``features`` is
    empty, rather than a placeholder schema — so a client that renders a form
    from ``features`` renders nothing, instead of an empty form whose every
    submission would fail.
    """

    # Every key the service produces is declared below, so `forbid` is a real
    # check rather than a formality: a field added to the manifest's public
    # summary and not to this model fails loudly instead of leaking through.
    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    status: str = Field(
        ...,
        description=(
            "The model's lifecycle state: `available` — a usable model is "
            "stored; `not_available` — this run has no artifact, which is "
            "normal for one recorded before model persistence existed or one "
            "whose artifact was removed; `corrupted` — an artifact is stored "
            "and does not check out. All three are 200 responses: whether a "
            "run can be predicted from is an answer, not a failure. Only an "
            "unknown experiment is a 404."
        ),
        examples=["available"],
    )
    available: bool = Field(
        ...,
        description=(
            "Whether a prediction can be attempted — `status == \"available\"`, "
            "kept as a boolean for a client that only needs to branch."
        ),
    )
    reason_code: str | None = Field(
        None,
        description=(
            "Why no model is usable, as a stable code to branch on: "
            "`no_artifact`, `manifest_unreadable`, `manifest_invalid`, "
            "`unsupported_schema_version`, `model_file_missing`, "
            "`model_file_truncated` or `model_too_large`. `null` when the "
            "model is available. It describes the artifact's condition and "
            "never where it is kept."
        ),
    )
    reason: str | None = Field(
        None,
        description=(
            "The same thing as a sentence for a person, ending in what to do "
            "about it. `null` when the model is available."
        ),
    )
    max_records: int = Field(
        ..., description="Most records one prediction request may carry."
    )
    model_name: str | None = None
    display_name: str | None = None
    task_type: str | None = None
    target_column: str | None = None
    classes: list[JsonValue] = Field(default_factory=list)
    features: list[PredictedFeature] = Field(default_factory=list)
    created_at: str | None = Field(
        None, description="When the model artifact was written."
    )
    train_row_count: int | None = None
    test_row_count: int | None = Field(
        None,
        description=(
            "How many held-out rows `primary_metric_value` was measured on. "
            "Carried beside the metric because a score without its sample "
            "size overstates what is known."
        ),
    )
    primary_metric: str | None = None
    primary_metric_value: float | None = Field(
        None,
        description=(
            "The winner's score on the **held-out test set**. A measurement of "
            "the model, not a confidence in any prediction it goes on to make."
        ),
    )
    supports_probabilities: bool = Field(
        False,
        description=(
            "Whether a prediction from this model can carry a probability per "
            "class. False for regression, and for a classifier whose estimator "
            "does not provide them."
        ),
    )
    artifact_schema_version: str | None = Field(
        None,
        description=(
            "Which manifest shape the stored artifact uses. Present when a "
            "model is available."
        ),
    )


__all__ = [
    "MAX_RECORDS_HARD_LIMIT",
    "ModelAvailability",
    "PredictedFeature",
    "PredictedModel",
    "PredictionItem",
    "PredictionRequest",
    "PredictionResponse",
]
