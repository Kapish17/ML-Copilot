"""The multipart contract of an experiment request.

A dataset is a file, and a file cannot travel inside a JSON body, so the run
endpoint takes ``multipart/form-data``: the dataset as an upload and the
configuration as form fields. FastAPI cannot flatten a Pydantic model into a
form when the same request also carries a file, so the configuration is
declared as the dependency below — one place, fully described, and each field
appears in the generated OpenAPI schema.

The dependency's only job is to turn form fields into
:class:`~app.services.experiments.options.ExperimentOptions`, which is the
FastAPI-free request object the service layer actually works with. Range checks
that need nothing but the value itself are declared here; everything that
depends on configured limits or on the dataset is checked by ``ExperimentOptions``
and by the ML layer's own validators.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Form

from app.services.experiments import ExperimentOptions
from app.services.experiments.options import (
    DEFAULT_STRATEGY,
    MAX_TEST_SIZE,
    MIN_TEST_SIZE,
    SELECTION_STRATEGIES,
)


def experiment_options(
    target_column: Annotated[
        str | None,
        Form(
            description=(
                "Column to predict. When omitted the last column is used by "
                "convention and the response says so in its warnings; "
                "automatic target detection is not implemented."
            ),
            examples=["renewed"],
        ),
    ] = None,
    models: Annotated[
        list[str] | None,
        Form(
            description=(
                "Registry identifiers of the models to consider, repeated once "
                "per model. When omitted, every registered model for the "
                "detected task is a candidate. GET /api/v1/experiments/"
                "capabilities lists them."
            ),
            examples=[["logistic_regression", "random_forest_classifier"]],
        ),
    ] = None,
    primary_metric: Annotated[
        str | None,
        Form(
            description=(
                "Metric the winner is chosen by. Defaults to the task's own "
                "default metric."
            ),
            examples=["f1"],
        ),
    ] = None,
    strategy: Annotated[
        str,
        Form(
            description=(
                "'cross_validation' scores candidates over folds of the "
                "training rows and never reads the test set, so the final "
                "measurement is unbiased. 'holdout' scores them on the test "
                "set, which makes the reported result optimistic."
            ),
            examples=list(SELECTION_STRATEGIES),
        ),
    ] = DEFAULT_STRATEGY,
    folds: Annotated[
        int | None,
        Form(description="Number of cross-validation folds.", ge=2, examples=[5]),
    ] = None,
    test_size: Annotated[
        float | None,
        Form(
            description="Share of rows held out for the final evaluation.",
            ge=MIN_TEST_SIZE,
            le=MAX_TEST_SIZE,
        ),
    ] = None,
    random_state: Annotated[
        int | None,
        Form(
            description="Seed for the split and every model that accepts one.",
            ge=0,
        ),
    ] = None,
    excluded_columns: Annotated[
        list[str] | None,
        Form(description="Columns to keep out of the feature set."),
    ] = None,
    identifier_columns: Annotated[
        list[str] | None,
        Form(description="Columns the caller knows to be identifiers."),
    ] = None,
    scaling_strategy: Annotated[
        str | None,
        Form(
            description="Numeric scaling.",
            examples=["standard", "minmax", "none"],
        ),
    ] = None,
    numeric_imputation: Annotated[
        str | None,
        Form(
            description="How missing numeric values are filled.",
            examples=["median", "mean", "constant"],
        ),
    ] = None,
    categorical_imputation: Annotated[
        str | None,
        Form(
            description="How missing categorical values are filled.",
            examples=["most_frequent", "constant"],
        ),
    ] = None,
    add_missing_indicators: Annotated[
        bool | None,
        Form(description="Whether a was-missing flag is added per column."),
    ] = None,
    max_categorical_cardinality: Annotated[
        int | None,
        Form(
            description=(
                "Above this many distinct values a categorical column is "
                "excluded rather than one-hot encoded."
            ),
            ge=1,
        ),
    ] = None,
    explain: Annotated[
        bool,
        Form(description="Whether to explain the winning model with SHAP."),
    ] = True,
    name: Annotated[
        str | None, Form(description="Short label for the run.")
    ] = None,
    description: Annotated[
        str | None, Form(description="Longer free text about the run.")
    ] = None,
    tags: Annotated[
        list[str] | None,
        Form(description="Labels for grouping and later retrieval."),
    ] = None,
) -> ExperimentOptions:
    """Assemble the experiment configuration from the submitted form."""
    return ExperimentOptions(
        target_column=target_column,
        models=tuple(models or ()),
        primary_metric=primary_metric,
        strategy=strategy,
        folds=folds,
        test_size=test_size,
        random_state=random_state,
        excluded_columns=tuple(excluded_columns or ()),
        identifier_columns=tuple(identifier_columns or ()),
        scaling_strategy=scaling_strategy,
        numeric_imputation=numeric_imputation,
        categorical_imputation=categorical_imputation,
        add_missing_indicators=add_missing_indicators,
        max_categorical_cardinality=max_categorical_cardinality,
        explain=explain,
        name=name,
        description=description,
        tags=tuple(tags or ()),
    )


ExperimentOptionsDep = Annotated[ExperimentOptions, Depends(experiment_options)]
