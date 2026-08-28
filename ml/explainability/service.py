"""The public explainability operations.

Two questions, two functions:

``explain_global``      what generally drives this model's predictions
``explain_prediction``  why did *this* row get *this* answer

Both take an already-trained model and read it only. Nothing is refitted: the
preprocessing step inside the pipeline is used exactly as it was fitted during
training, the estimator is used exactly as trained, and no target value ever
reaches either of them. Explaining a model cannot change it.

The pipeline boundary matters here. A trained model is
``Pipeline(preprocessing, estimator)``, and SHAP needs the numbers the
*estimator* saw, not the raw columns. So raw rows are pushed through the
already-fitted preprocessing, and the transformed frame — carrying Commit 3's
feature names, ``contract_Month-to-month`` rather than ``x2`` — is what the
explainer works on. The original pipeline is left untouched and still usable.

Results are structured, never prose. A future agent receives
``feature=monthly_charges, contribution=+0.31, direction=increases_prediction``
and writes the sentence itself.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import is_classifier
from sklearn.utils.validation import check_is_fitted

from ml.errors import (
    EmptyExplanationDataError,
    ExplainabilityError,
    InvalidExplanationRowError,
    InvalidTrainedModelError,
    MissingFeatureColumnsError,
)
from ml.explainability.config import ExplanationConfig, limit_rows
from ml.explainability.permutation import (
    PERMUTATION_AGGREGATION,
    permutation_global_importance,
)
from ml.explainability.results import (
    FeatureImportance,
    GlobalExplanation,
    LocalExplanation,
    rank_contributions,
    rank_importances,
    to_float,
)
from ml.explainability.shap_backend import (
    ShapUnavailable,
    base_value_for,
    build_explainer,
    compute_shap_values,
    mean_absolute_importance,
    row_contributions,
    select_output,
)
from ml.explainability.strategy import ExplainerKind, select_explainer
from ml.explainability.types import ExplanationMethod, ExplanationStatus
from ml.models.result import TrainedModel

SHAP_AGGREGATION = "mean absolute SHAP value per feature"


def _validate_model(trained_model: Any) -> TrainedModel:
    """Check that the object is a usable, fitted trained model.

    Raises:
        InvalidTrainedModelError: If it is not, or if its pipeline is unfitted.
    """
    if not isinstance(trained_model, TrainedModel):
        raise InvalidTrainedModelError(
            "Explanations need a TrainedModel produced by the training layer, "
            f"not {type(trained_model).__name__}.",
            details={"received_type": type(trained_model).__name__},
        )
    try:
        check_is_fitted(trained_model.preprocessor)
        check_is_fitted(trained_model.estimator)
    except Exception as exc:  # noqa: BLE001 - sklearn raises NotFittedError
        raise InvalidTrainedModelError(
            f"The trained model is not fitted: {exc}"
        ) from exc
    return trained_model


def _as_frame(data: pd.DataFrame | pd.Series) -> pd.DataFrame:
    """Accept a single row as a Series and treat it as a one-row frame."""
    if isinstance(data, pd.Series):
        return data.to_frame().T
    return data


def _align_features(trained_model: TrainedModel, data: Any) -> pd.DataFrame:
    """Select exactly the columns the preprocessing was fitted on, in order.

    Extra columns are ignored and column order does not matter, but a missing
    one is an error naming what is absent.

    Raises:
        ExplainabilityError: If the data is not a DataFrame or Series.
        EmptyExplanationDataError: If there are no rows.
        MissingFeatureColumnsError: If a fitted feature column is absent.
    """
    if not isinstance(data, (pd.DataFrame, pd.Series)):
        raise ExplainabilityError(
            "Explanation data must be a pandas DataFrame or Series, not "
            f"{type(data).__name__}.",
            details={"received_type": type(data).__name__},
        )

    frame = _as_frame(data)
    if frame.shape[0] == 0:
        raise EmptyExplanationDataError("There are no rows to explain.")

    required = [str(name) for name in trained_model.preprocessor.feature_names_in_]
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise MissingFeatureColumnsError(
            "The data to explain is missing column(s) the model was fitted "
            "on: " + ", ".join(missing) + ".",
            details={"missing_columns": missing, "required_columns": required},
        )
    return frame.loc[:, required]


def _transform(trained_model: TrainedModel, frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the already-fitted preprocessing and keep the feature names.

    ``transform`` never learns anything, so this cannot change the model.
    """
    transformed = trained_model.preprocessor.transform(frame)
    if isinstance(transformed, pd.DataFrame):
        return transformed
    return pd.DataFrame(  # pragma: no cover - pipelines are configured for pandas
        transformed, columns=list(trained_model.feature_names), index=frame.index
    )


def _classes_of(trained_model: TrainedModel) -> list[Any] | None:
    """Return the estimator's classes, in its own order, for a classifier."""
    if not is_classifier(trained_model.estimator):
        return None
    return list(getattr(trained_model.estimator, "classes_", []))


def _unavailable_global(
    trained_model: TrainedModel, reason: str, warnings: tuple[str, ...] = ()
) -> GlobalExplanation:
    """Build the result used when no global importance could be produced."""
    return GlobalExplanation(
        status=ExplanationStatus.UNAVAILABLE,
        method=ExplanationMethod.NONE,
        model_name=trained_model.model_name,
        task_type=trained_model.task_type.value,
        feature_count=len(trained_model.feature_names),
        reason=reason,
        warnings=warnings,
    )


def explain_global(
    trained_model: TrainedModel,
    X_reference: pd.DataFrame,
    y_reference: pd.Series | None = None,
    *,
    config: ExplanationConfig | None = None,
    top_n: int | None = None,
    background: pd.DataFrame | None = None,
) -> GlobalExplanation:
    """Rank the features that generally drive a model's predictions.

    SHAP values are computed over the reference rows and reduced to one number
    per feature: the mean absolute SHAP value, which is how far that feature
    moved the model's output on average, in either direction.

    **Which rows to pass.** The training features are the natural choice — they
    are what the model learned from, so they describe the behaviour it actually
    acquired, and using them keeps the held-out test set genuinely untouched.
    Passing test features is also sound and answers a slightly different
    question ("how does the model behave on unseen rows"); no target is
    involved and nothing is fitted, so neither choice leaks. What the reference
    data does is define the distribution SHAP compares each row against, which
    is why it must be representative.

    Args:
        trained_model: An already-trained model. It is read, never modified.
        X_reference: Raw feature rows to summarise the model over.
        y_reference: True values for those rows. Only used if SHAP cannot
            explain the model and the permutation fallback is needed.
        config: Row limits and seed.
        top_n: Keep only the most important ``top_n`` features.
        background: Raw rows to use as the reference distribution for
            explainers that need one. Defaults to ``X_reference``.

    Returns:
        GlobalExplanation: Ranked importances, or a structured reason why not.

    Raises:
        InvalidTrainedModelError: The model is not a fitted trained model.
        EmptyExplanationDataError: There are no reference rows.
        MissingFeatureColumnsError: A fitted feature column is absent.
    """
    model = _validate_model(trained_model)
    settings = config or ExplanationConfig()

    aligned = _align_features(model, X_reference)
    capped, sampled = limit_rows(
        aligned, settings.max_explanation_rows, random_state=settings.random_state
    )
    notes: list[str] = []
    if sampled:
        notes.append(
            f"Sampled {capped.shape[0]} of {aligned.shape[0]} reference rows "
            f"(max_explanation_rows={settings.max_explanation_rows})."
        )

    transformed = _transform(model, capped)
    names = list(transformed.columns)
    plan = select_explainer(model.estimator)
    classes = _classes_of(model)
    is_classification = classes is not None

    if plan.supported:
        try:
            reference = transformed
            if plan.kind is ExplainerKind.LINEAR:
                source = _align_features(model, background) if background is not None else aligned
                limited, _ = limit_rows(
                    source,
                    settings.max_reference_rows,
                    random_state=settings.random_state,
                )
                reference = _transform(model, limited)

            explainer = build_explainer(model.estimator, plan, reference)
            values = compute_shap_values(explainer, transformed)
            importances, explained_output, shap_notes = mean_absolute_importance(
                values, is_classification=is_classification, classes=classes
            )
            ranked = rank_importances(
                names, [float(value) for value in importances]
            )
            return GlobalExplanation(
                status=ExplanationStatus.AVAILABLE,
                method=ExplanationMethod.SHAP,
                model_name=model.model_name,
                task_type=model.task_type.value,
                feature_importances=ranked[:top_n] if top_n else ranked,
                sample_count=values.row_count,
                feature_count=len(names),
                explainer=plan.explainer_name,
                aggregation=SHAP_AGGREGATION,
                explained_output=explained_output,
                warnings=(*notes, *shap_notes),
            )
        except ShapUnavailable as exc:
            notes.append(f"SHAP was unavailable, so the fallback was used: {exc}")
    else:
        notes.append(f"SHAP was unavailable, so the fallback was used: {plan.reason}")

    if y_reference is None:
        return _unavailable_global(
            model,
            "SHAP could not explain this model and permutation importance "
            "needs the reference targets, which were not supplied. Pass "
            "y_reference to use the fallback.",
            tuple(notes),
        )

    target = pd.Series(y_reference).loc[capped.index]
    importances = permutation_global_importance(
        model.estimator, transformed, target, config=settings
    )
    ranked = rank_importances(names, [float(value) for value in importances])
    return GlobalExplanation(
        status=ExplanationStatus.AVAILABLE,
        method=ExplanationMethod.PERMUTATION_IMPORTANCE,
        model_name=model.model_name,
        task_type=model.task_type.value,
        feature_importances=ranked[:top_n] if top_n else ranked,
        sample_count=int(transformed.shape[0]),
        feature_count=len(names),
        explainer="permutation_importance",
        aggregation=PERMUTATION_AGGREGATION,
        explained_output="the model's score",
        warnings=tuple(notes),
    )


def _unavailable_local(
    model: TrainedModel,
    reason: str,
    *,
    prediction: Any,
    probability: float | None,
    probabilities: dict[str, float] | None,
    predicted_class: str | None,
    positive_class: str | None,
    warnings: tuple[str, ...] = (),
) -> LocalExplanation:
    """Build the result used when no local explanation could be produced.

    The prediction is still reported — it is a fact about the model — but no
    contributions are invented to go with it.
    """
    return LocalExplanation(
        status=ExplanationStatus.UNAVAILABLE,
        method=ExplanationMethod.NONE,
        model_name=model.model_name,
        task_type=model.task_type.value,
        prediction=prediction,
        probability=probability,
        probabilities=probabilities,
        predicted_class=predicted_class,
        positive_class=positive_class,
        sample_count=1,
        feature_count=len(model.feature_names),
        reason=reason,
        warnings=warnings,
    )


def explain_prediction(
    trained_model: TrainedModel,
    row: pd.DataFrame | pd.Series,
    *,
    background: pd.DataFrame | None = None,
    config: ExplanationConfig | None = None,
    top_n: int | None = None,
    target_class: Any | None = None,
) -> LocalExplanation:
    """Explain why one row received the prediction it did.

    Each feature gets a signed contribution: how far it moved the model's
    output away from the base value, which is the model's average output over
    the background rows. Contributions are on the model's own output scale, not
    in probability units.

    For a classifier the explained class is the predicted one unless
    ``target_class`` says otherwise, and the result names the predicted class,
    the explained class and the positive class separately so none has to be
    inferred. The positive class follows the convention established in Commit
    4 — the last of the estimator's sorted classes — which is what
    ``estimator.classes_[-1]`` gives.

    Args:
        trained_model: An already-trained model. It is read, never modified.
        row: One row of raw features, as a one-row DataFrame or a Series.
        background: Raw rows giving the reference distribution. Required for
            linear models, which have no other way to know what "average"
            means; tree models do not need it.
        config: Row limits and seed.
        top_n: Keep only the ``top_n`` largest contributions.
        target_class: Explain this class instead of the predicted one.

    Returns:
        LocalExplanation: Contributions, or a structured reason why not. No
        local contributions are ever invented from the global fallback.

    Raises:
        InvalidTrainedModelError: The model is not a fitted trained model.
        InvalidExplanationRowError: More or fewer than one row was given.
        MissingFeatureColumnsError: A fitted feature column is absent.
        ExplainabilityError: A linear model was given no background rows, or
            ``target_class`` is not one of the model's classes.
    """
    model = _validate_model(trained_model)
    settings = config or ExplanationConfig()

    aligned = _align_features(model, row)
    if aligned.shape[0] != 1:
        raise InvalidExplanationRowError(
            "A local explanation covers exactly one row; "
            f"{aligned.shape[0]} were given. Use explain_global to summarise "
            "many rows.",
            details={"row_count": int(aligned.shape[0])},
        )

    transformed = _transform(model, aligned)
    names = list(transformed.columns)
    classes = _classes_of(model)
    is_classification = classes is not None

    if target_class is not None and (not is_classification or target_class not in classes):
        raise ExplainabilityError(
            f"'{target_class}' is not one of the model's classes.",
            details={"available_classes": [str(item) for item in (classes or [])]},
        )

    prediction = model.estimator.predict(transformed)[0]
    predicted_class = str(prediction) if is_classification else None
    positive_class = str(classes[-1]) if is_classification and classes else None

    probabilities: dict[str, float] | None = None
    if is_classification and hasattr(model.estimator, "predict_proba"):
        scores = model.estimator.predict_proba(transformed)[0]
        probabilities = {
            str(label): float(score)
            for label, score in zip(classes, scores, strict=True)
        }

    explained = target_class if target_class is not None else prediction
    explained_class = str(explained) if is_classification else None
    probability = (
        probabilities.get(explained_class) if probabilities and explained_class else None
    )

    plan = select_explainer(model.estimator)
    if not plan.supported:
        return _unavailable_local(
            model,
            f"{plan.reason} Permutation importance is a global measure and "
            "cannot explain a single prediction, so no contributions are "
            "reported.",
            prediction=prediction,
            probability=probability,
            probabilities=probabilities,
            predicted_class=predicted_class,
            positive_class=positive_class,
        )

    reference = transformed
    if plan.kind is ExplainerKind.LINEAR:
        if background is None:
            raise ExplainabilityError(
                "Explaining a linear model needs background rows to compare "
                "against: pass background=<the training features>. Tree "
                "models do not need one.",
                details={"model_name": model.model_name},
            )
        limited, _ = limit_rows(
            _align_features(model, background),
            settings.max_reference_rows,
            random_state=settings.random_state,
        )
        reference = _transform(model, limited)

    try:
        explainer = build_explainer(model.estimator, plan, reference)
        values = compute_shap_values(explainer, transformed)
        selection = select_output(
            values,
            is_classification=is_classification,
            classes=classes,
            target_class=explained if is_classification else None,
        )
        contributions = row_contributions(values, row=0, selection=selection)
        base_value = base_value_for(values, selection)
    except ShapUnavailable as exc:
        return _unavailable_local(
            model,
            str(exc),
            prediction=prediction,
            probability=probability,
            probabilities=probabilities,
            predicted_class=predicted_class,
            positive_class=positive_class,
        )

    raw_row = aligned.iloc[0]
    ranked = rank_contributions(
        names,
        [float(value) for value in contributions],
        [to_float(value) for value in np.asarray(transformed.iloc[0])],
        [raw_row.get(name, None) for name in names],
    )
    notes = (selection.note,) if selection.note else ()

    return LocalExplanation(
        status=ExplanationStatus.AVAILABLE,
        method=ExplanationMethod.SHAP,
        model_name=model.model_name,
        task_type=model.task_type.value,
        prediction=prediction,
        probability=probability,
        probabilities=probabilities,
        predicted_class=predicted_class,
        explained_class=selection.explained_class or explained_class,
        positive_class=positive_class,
        base_value=base_value,
        feature_contributions=ranked[:top_n] if top_n else ranked,
        sample_count=1,
        feature_count=len(names),
        explainer=plan.explainer_name,
        warnings=notes,
    )


def get_feature_importance(
    trained_model: TrainedModel,
    X_reference: pd.DataFrame,
    y_reference: pd.Series | None = None,
    *,
    top_n: int | None = None,
    config: ExplanationConfig | None = None,
) -> tuple[dict[str, Any], ...]:
    """Return ranked feature importances as plain records.

    A convenience over :func:`explain_global`, shaped for a caller that wants
    the facts and nothing else — the form a future agent would receive as a
    tool result. An unavailable explanation returns an empty tuple; call
    :func:`explain_global` when the reason matters.

    Args:
        trained_model: An already-trained model.
        X_reference: Raw feature rows to summarise the model over.
        y_reference: True values, needed only by the permutation fallback.
        top_n: Keep only the most important ``top_n`` features.
        config: Row limits and seed.

    Returns:
        tuple[dict, ...]: ``{"feature", "importance", "rank"}`` records,
        ordered most important first.
    """
    explanation = explain_global(
        trained_model, X_reference, y_reference, config=config, top_n=top_n
    )
    if not explanation.available:
        return ()
    return tuple(entry.as_dict() for entry in explanation.feature_importances)


__all__ = [
    "ExplanationConfig",
    "FeatureImportance",
    "GlobalExplanation",
    "LocalExplanation",
    "explain_global",
    "explain_prediction",
    "get_feature_importance",
]
