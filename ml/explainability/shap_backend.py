"""Running SHAP and making sense of what it returns.

SHAP's output shape depends on the explainer and the model. A regressor gives
one value per feature per row; a random forest on a binary problem gives one
per class as well; a histogram-boosting classifier gives a single output for
the positive class; a multiclass model gives one per class. Older versions
return a list of arrays where newer ones return a three-dimensional array.

Rather than spread those cases through the rest of the package, everything is
normalised here into one shape — ``(rows, features, outputs)`` with a base
value per output — and a single function decides which output answers the
question being asked.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from ml.explainability.strategy import ExplainerPlan
from ml.explainability.types import ExplainerKind

#: Description used when a model has one output rather than one per class.
SINGLE_OUTPUT = "the model's single output"


@dataclass(frozen=True)
class ShapValues:
    """SHAP values in one predictable shape.

    Attributes:
        values: ``(rows, features, outputs)``. ``outputs`` is 1 for a
            regressor or a classifier that emits a single margin.
        base_values: One expected value per output.
    """

    values: np.ndarray
    base_values: np.ndarray

    @property
    def output_count(self) -> int:
        """How many outputs the model produced values for."""
        return int(self.values.shape[2])

    @property
    def row_count(self) -> int:
        """How many rows were explained."""
        return int(self.values.shape[0])

    @property
    def feature_count(self) -> int:
        """How many features each row was explained over."""
        return int(self.values.shape[1])


@dataclass(frozen=True)
class OutputSelection:
    """Which SHAP output answers the question, and how to read it."""

    index: int
    negated: bool = False
    explained_class: str | None = None
    note: str | None = None


class ShapUnavailable(RuntimeError):
    """SHAP could not produce values for this model and data.

    Raised inside this module and turned into a structured result by the
    service layer; it never reaches a caller.
    """


def build_explainer(
    estimator: BaseEstimator, plan: ExplainerPlan, background: pd.DataFrame
):
    """Construct the SHAP explainer a plan calls for.

    The tree explainer walks the fitted trees and needs no background data.
    The linear explainer needs one: SHAP values are always relative to some
    reference distribution, and for a linear model that means the mean and
    covariance of the features it is being compared against.

    Args:
        estimator: The fitted estimator from inside the trained pipeline.
        plan: The explainer choice made by the strategy module.
        background: Reference rows, already transformed and capped.

    Returns:
        A SHAP explainer.

    Raises:
        ShapUnavailable: If SHAP rejects the estimator.
    """
    import shap

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if plan.kind is ExplainerKind.TREE:
                return shap.TreeExplainer(estimator)
            if plan.kind is ExplainerKind.LINEAR:
                return shap.LinearExplainer(estimator, background)
    except Exception as exc:  # noqa: BLE001 - any SHAP refusal is reported alike
        raise ShapUnavailable(
            f"SHAP could not build an explainer for "
            f"{type(estimator).__name__}: {exc}"
        ) from exc

    raise ShapUnavailable(  # pragma: no cover - guarded by the strategy module
        f"No SHAP explainer is configured for {type(estimator).__name__}."
    )


def normalise_values(raw: Any, expected: Any, *, feature_count: int) -> ShapValues:
    """Reshape whatever SHAP returned into ``(rows, features, outputs)``.

    Args:
        raw: The explainer's values — a list of per-class arrays, a 2D array,
            or a 3D array, depending on version and model.
        expected: The explainer's expected value — a scalar or an array.
        feature_count: How many features were passed in, used to check shape.

    Returns:
        ShapValues: The values and one base value per output.

    Raises:
        ShapUnavailable: If the values cannot be interpreted.
    """
    if isinstance(raw, list):
        stacked = np.stack([np.asarray(item, dtype="float64") for item in raw], axis=-1)
    else:
        stacked = np.asarray(raw, dtype="float64")

    if stacked.ndim == 2:
        stacked = stacked[:, :, np.newaxis]
    if stacked.ndim != 3:
        raise ShapUnavailable(
            f"SHAP returned values with an unexpected shape {stacked.shape}."
        )
    if stacked.shape[1] != feature_count:
        raise ShapUnavailable(
            f"SHAP returned {stacked.shape[1]} feature values for "
            f"{feature_count} features."
        )

    base = np.atleast_1d(np.asarray(expected, dtype="float64")).ravel()
    outputs = stacked.shape[2]
    if base.size == 1 and outputs > 1:
        base = np.repeat(base, outputs)
    if base.size != outputs:
        base = np.resize(base, outputs)

    return ShapValues(values=stacked, base_values=base)


def compute_shap_values(explainer, features: pd.DataFrame) -> ShapValues:
    """Run an explainer over a transformed feature frame.

    Args:
        explainer: A SHAP explainer built by :func:`build_explainer`.
        features: Transformed features, with Commit 3's column names.

    Returns:
        ShapValues: Normalised values and base values.

    Raises:
        ShapUnavailable: If SHAP fails on this data.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = explainer.shap_values(features)
    except Exception as exc:  # noqa: BLE001 - any SHAP failure is reported alike
        raise ShapUnavailable(f"SHAP failed to compute values: {exc}") from exc

    return normalise_values(
        raw, getattr(explainer, "expected_value", 0.0), feature_count=features.shape[1]
    )


def select_output(
    shap_values: ShapValues,
    *,
    is_classification: bool,
    classes: list[Any] | None,
    target_class: Any | None,
) -> OutputSelection:
    """Pick the SHAP output that explains the class in question.

    Three arrangements occur in practice:

    * a regressor, or any model with one output — index 0;
    * one output per class — index of the requested class;
    * a binary classifier with a single output, which is the margin for the
      positive class. Explaining the other class is then the negation of those
      values, because the two margins are mirror images.

    Args:
        shap_values: Normalised values.
        is_classification: Whether the model predicts classes.
        classes: The estimator's classes, in its own order.
        target_class: The class to explain.

    Returns:
        OutputSelection: The output index, whether to negate it, and a note.

    Raises:
        ShapUnavailable: If the outputs cannot be matched to the classes.
    """
    outputs = shap_values.output_count

    if not is_classification or classes is None:
        if outputs != 1:
            raise ShapUnavailable(
                f"A regression model produced {outputs} SHAP outputs, which "
                "cannot be interpreted as a single prediction."
            )
        return OutputSelection(index=0)

    labels = list(classes)
    positive = labels[-1]
    wanted = target_class if target_class is not None else positive

    if outputs == len(labels):
        return OutputSelection(
            index=labels.index(wanted), explained_class=str(wanted)
        )

    if outputs == 1 and len(labels) == 2:
        negated = wanted != positive
        note = (
            f"This model produces a single SHAP output, for the positive class "
            f"'{positive}'. Contributions for '{wanted}' are the negation of "
            "those values."
            if negated
            else None
        )
        return OutputSelection(
            index=0, negated=negated, explained_class=str(wanted), note=note
        )

    raise ShapUnavailable(
        f"SHAP returned {outputs} output(s) for a model with {len(labels)} "
        "classes, so a per-class explanation cannot be produced."
    )


def row_contributions(
    shap_values: ShapValues, *, row: int, selection: OutputSelection
) -> np.ndarray:
    """Return one row's contributions for the selected output."""
    values = shap_values.values[row, :, selection.index]
    return -values if selection.negated else values


def base_value_for(shap_values: ShapValues, selection: OutputSelection) -> float:
    """Return the expected value for the selected output."""
    value = float(shap_values.base_values[selection.index])
    return -value if selection.negated else value


def mean_absolute_importance(
    shap_values: ShapValues,
    *,
    is_classification: bool,
    classes: list[Any] | None,
) -> tuple[np.ndarray, str, tuple[str, ...]]:
    """Reduce SHAP values to one importance per feature.

    The mean absolute SHAP value answers "how far did this feature move the
    output, on average, in either direction". Signs are dropped on purpose: a
    feature that pushes strongly both ways is influential, and averaging the
    signed values would hide that.

    Multiclass models get one set of values per class; those are averaged, so
    the result describes overall influence rather than influence on any one
    class.

    Args:
        shap_values: Normalised values.
        is_classification: Whether the model predicts classes.
        classes: The estimator's classes, in its own order.

    Returns:
        tuple: importances per feature, a description of which output was
        summarised, and any warnings the caller should surface.
    """
    magnitudes = np.abs(shap_values.values)
    outputs = shap_values.output_count
    notes: list[str] = []

    if outputs == 1:
        return magnitudes[:, :, 0].mean(axis=0), SINGLE_OUTPUT, ()

    if is_classification and classes is not None and outputs == len(classes) == 2:
        positive = str(list(classes)[-1])
        return (
            magnitudes[:, :, -1].mean(axis=0),
            f"the positive class '{positive}'",
            (
                f"Binary model: importances are the mean absolute SHAP value "
                f"for the positive class '{positive}'.",
            ),
        )

    if is_classification and classes is not None and outputs == len(classes):
        notes.append(
            "Multiclass model: importances are averaged over the "
            f"{outputs} classes, so they describe overall influence rather "
            "than influence on one class."
        )
        return (
            magnitudes.mean(axis=2).mean(axis=0),
            f"all {outputs} classes, averaged",
            tuple(notes),
        )

    notes.append(
        f"The model produced {outputs} SHAP outputs; importances are averaged "
        "over them."
    )
    return magnitudes.mean(axis=2).mean(axis=0), f"{outputs} outputs, averaged", tuple(notes)
