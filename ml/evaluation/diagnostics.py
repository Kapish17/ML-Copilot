"""Things about a finished run that are worth a second look.

A run can complete successfully and still be worth questioning. The winner may
score far worse on held-out rows than it did in cross-validation; the folds may
disagree with each other enough that the mean means little; the test split may
be sixteen rows; one class may hold ninety-five per cent of the data. None of
those is an error — the pipeline did exactly what it was asked — and none of
them should be discovered by a reader squinting at four numbers on a page.

So this module reads a finished run's own recorded numbers and returns
:class:`Diagnostic` objects: a stable code, a severity, and a sentence.

---------------------------------------------------------------------------
What these are, and what they are not
---------------------------------------------------------------------------
**They are signals, not verdicts.** A gap between cross-validated and held-out
performance is consistent with overfitting, and also with an unlucky split, a
small test set, or a distribution that shifts across the data's natural order.
This module cannot tell those apart and does not try, so it says *"potential
overfitting signal: held-out performance is materially below cross-validation
performance"* and never *"the model is overfit"*. Every message here is written
to that standard, and a test asserts the verdict words are absent.

**They prove nothing about leakage.** Leakage is prevented structurally, by
fitting on training rows only — see `ml/pipelines/preparation.py`. A diagnostic
is a prompt to look, not evidence of a fault.

**They are warnings, not failures.** Nothing here fails a run or changes a
score. A run with four diagnostics is a completed run with four things worth
reading; the pipeline's own errors are raised where they happen, in the layers
that can raise them.

**They compute nothing new.** Every threshold is applied to a number the run
already recorded — the CV mean and spread, the held-out score, the class
distribution, the split sizes. This module fits nothing, reads no dataset and
sees no cell value.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ml.evaluation.metrics import MetricDirection

# ---------------------------------------------------------------------------
# Thresholds
#
# Round numbers, chosen to be defensible rather than tuned. Each is the point
# at which a careful reader would want to look again — not a point at which
# something is wrong. They are module constants so a caller can see them, a
# test can name them, and the documentation cannot drift from the code.
# ---------------------------------------------------------------------------

#: A held-out score this much worse than the cross-validated one, as a share of
#: the cross-validated score, is worth mentioning. Fifteen per cent is a wide
#: margin on purpose: fold-to-fold noise alone moves a small dataset's score by
#: several per cent, and a diagnostic that fires on noise is one people learn to
#: ignore.
GENERALISATION_GAP_RATIO = 0.15

#: Cross-validation spread this large relative to the mean says the folds
#: disagreed enough that the mean is a weak summary of them.
HIGH_CV_VARIABILITY_RATIO = 0.25

#: Fewer rows than this and every score on the page is a small-sample estimate.
SMALL_DATASET_ROWS = 200
#: Fewer held-out rows than this and the final measurement is one too.
SMALL_TEST_ROWS = 50

#: A majority class holding more than this share makes accuracy flattering and
#: a minority class hard to learn.
CLASS_IMBALANCE_RATIO = 0.80
#: And this much is severe enough to say so more strongly.
SEVERE_CLASS_IMBALANCE_RATIO = 0.95


class Severity(str, Enum):
    """How much attention a diagnostic asks for."""

    #: Worth knowing when reading the numbers.
    INFO = "info"
    #: Worth looking into before trusting the result.
    WARNING = "warning"

    @property
    def is_warning(self) -> bool:
        """Whether this is more than a note."""
        return self is Severity.WARNING


#: Stable codes. A client branches on these; the sentence beside them is for a
#: person and may be reworded without breaking anything.
GENERALISATION_GAP = "generalisation_gap"
HIGH_CV_VARIABILITY = "high_cv_variability"
SMALL_DATASET = "small_dataset"
SMALL_TEST_SET = "small_test_set"
CLASS_IMBALANCE = "class_imbalance"
MISSING_CLASS_IN_TEST = "missing_class_in_test"
UNDEFINED_METRIC = "undefined_metric"
BASELINE_NOT_BEATEN = "baseline_not_beaten"
SELECTION_USED_TEST_DATA = "selection_used_test_data"


@dataclass(frozen=True)
class Diagnostic:
    """One thing about a run that is worth a second look."""

    code: str
    severity: Severity
    #: A sentence for a person. Hedged where the evidence is circumstantial —
    #: see the module docstring on why that is not timidity.
    message: str
    #: The numbers the message is about, so a reader can check it.
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Render the diagnostic as plain, JSON-friendly values."""
        return {
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Diagnostic:
        """Rebuild a diagnostic from a stored record.

        An unrecognised severity reads back as ``info`` rather than raising: a
        record written by a later version should still be readable, and a
        diagnostic is a note about a run rather than part of its result.
        """
        try:
            severity = Severity(str(payload.get("severity", "info")))
        except ValueError:  # pragma: no cover - defensive
            severity = Severity.INFO
        return cls(
            code=str(payload.get("code", "")),
            severity=severity,
            message=str(payload.get("message", "")),
            details=dict(payload.get("details") or {}),
        )


def _finite(value: Any) -> float | None:
    """Return a usable float, or ``None``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _relative_shortfall(
    selection: float, held_out: float, *, higher_is_better: bool
) -> float | None:
    """How much worse the held-out score is, as a share of the CV score.

    Positive means the held-out measurement is the *worse* of the two, whichever
    direction the metric runs in. ``None`` when the comparison would divide by
    something close to zero, which is where a ratio stops meaning anything —
    an RMSE of 0.0001 against 0.0002 is a hundred per cent worse and nobody
    should be told about it.
    """
    scale = abs(selection)
    if scale < 1e-9:
        return None
    difference = (selection - held_out) if higher_is_better else (held_out - selection)
    return difference / scale


def generalisation_diagnostics(
    *,
    selection_score: Any,
    held_out_score: Any,
    direction: MetricDirection | str | None,
    metric: str,
) -> list[Diagnostic]:
    """Compare what chose the model against what measured it.

    These are the two numbers this project is most careful to keep apart, and
    the gap between them is the single most informative thing about a run that
    a reader might otherwise miss.

    Args:
        selection_score: The cross-validated score the winner was chosen by.
        held_out_score: The one measurement on the untouched test set.
        direction: Which way the metric runs.
        metric: The metric's key, for the message.

    Returns:
        list[Diagnostic]: Empty when the two agree, or when either is missing.
    """
    selection = _finite(selection_score)
    held_out = _finite(held_out_score)
    if selection is None or held_out is None:
        return []

    higher_is_better = _higher_is_better(direction)
    shortfall = _relative_shortfall(
        selection, held_out, higher_is_better=higher_is_better
    )
    if shortfall is None or shortfall < GENERALISATION_GAP_RATIO:
        return []

    return [
        Diagnostic(
            code=GENERALISATION_GAP,
            severity=Severity.WARNING,
            message=(
                "Potential overfitting signal: held-out performance is "
                f"materially below cross-validation performance. The winner "
                f"scored {selection:.4f} {metric} across the training folds "
                f"and {held_out:.4f} on the held-out test set, a shortfall of "
                f"{shortfall:.0%}. That is consistent with overfitting, and "
                "also with an unlucky split or a small test set — it is worth "
                "checking, not a conclusion."
            ),
            details={
                "metric": metric,
                "cross_validation_score": selection,
                "held_out_score": held_out,
                "relative_shortfall": round(shortfall, 4),
                "threshold": GENERALISATION_GAP_RATIO,
            },
        )
    ]


def variability_diagnostics(
    *, selection_score: Any, selection_score_std: Any, metric: str, folds: Any = None
) -> list[Diagnostic]:
    """Say when the folds disagreed enough that their mean is a weak summary.

    Deliberately **not** described as a confidence interval. The standard
    deviation of five fold scores is a spread over five correlated,
    overlapping training sets; treating it as a standard error would overstate
    what it supports, and this project says so wherever the number appears.
    """
    mean = _finite(selection_score)
    spread = _finite(selection_score_std)
    if mean is None or spread is None or abs(mean) < 1e-9 or spread <= 0:
        return []

    ratio = spread / abs(mean)
    if ratio < HIGH_CV_VARIABILITY_RATIO:
        return []

    fold_note = f" across {folds} folds" if isinstance(folds, int) and folds else ""
    return [
        Diagnostic(
            code=HIGH_CV_VARIABILITY,
            severity=Severity.WARNING,
            message=(
                f"Cross-validation scores varied widely{fold_note}: "
                f"{mean:.4f} ± {spread:.4f} {metric}, a spread of {ratio:.0%} "
                "of the mean. The folds disagreed enough that the mean is a "
                "weak summary of them, so treat the ranking between close "
                "candidates as provisional. The spread is a fold-to-fold "
                "range, not a confidence interval."
            ),
            details={
                "metric": metric,
                "mean": mean,
                "standard_deviation": spread,
                "relative_spread": round(ratio, 4),
                "threshold": HIGH_CV_VARIABILITY_RATIO,
                "folds": folds if isinstance(folds, int) else None,
            },
        )
    ]


def size_diagnostics(
    *, row_count: Any, train_row_count: Any, test_row_count: Any
) -> list[Diagnostic]:
    """Say when there was not much data to measure anything with."""
    found: list[Diagnostic] = []

    rows = _finite(row_count)
    if rows is not None and rows < SMALL_DATASET_ROWS:
        found.append(
            Diagnostic(
                code=SMALL_DATASET,
                severity=Severity.WARNING,
                message=(
                    f"Small dataset: {int(rows)} rows in total. Every score on "
                    "this run is a small-sample estimate, and the difference "
                    "between two close models is unlikely to be meaningful."
                ),
                details={
                    "row_count": int(rows),
                    "train_row_count": int(_finite(train_row_count) or 0),
                    "threshold": SMALL_DATASET_ROWS,
                },
            )
        )

    test_rows = _finite(test_row_count)
    if test_rows is not None and 0 < test_rows < SMALL_TEST_ROWS:
        found.append(
            Diagnostic(
                code=SMALL_TEST_SET,
                severity=Severity.WARNING,
                message=(
                    f"Small held-out set: the final measurement was taken on "
                    f"{int(test_rows)} rows. One measurement on that many rows "
                    "carries a wide margin of error, whatever it says."
                ),
                details={
                    "test_row_count": int(test_rows),
                    "threshold": SMALL_TEST_ROWS,
                },
            )
        )
    return found


def class_diagnostics(
    details: Mapping[str, Any] | None, *, train_class_labels: Sequence[Any] = ()
) -> list[Diagnostic]:
    """Read the class distribution the evaluation already recorded.

    Args:
        details: The ``classification_details`` block — class labels, counts and
            the confusion matrix. ``None`` for regression, which produces
            nothing.
        train_class_labels: Labels the model was trained to predict, when known.
            A label the model knows and the test split does not contain is
            worth saying, because every metric for that class is then computed
            from no rows at all.
    """
    if not details:
        return []

    distribution = details.get("class_distribution") or {}
    if not isinstance(distribution, Mapping) or not distribution:
        return []

    counts = {
        str(label): int(count)
        for label, count in distribution.items()
        if isinstance(count, (int, float)) and not isinstance(count, bool)
    }
    total = sum(counts.values())
    found: list[Diagnostic] = []

    if total > 0 and counts:
        majority_label, majority_count = max(counts.items(), key=lambda item: item[1])
        share = majority_count / total
        if share >= CLASS_IMBALANCE_RATIO:
            severe = share >= SEVERE_CLASS_IMBALANCE_RATIO
            found.append(
                Diagnostic(
                    code=CLASS_IMBALANCE,
                    severity=Severity.WARNING,
                    message=(
                        f"{'Extreme' if severe else 'Notable'} class imbalance: "
                        f"'{majority_label}' is {share:.0%} of the evaluated "
                        "rows. Accuracy flatters a model on data like this — a "
                        "model that always predicts the majority class would "
                        f"score about {share:.0%} — so read the per-class "
                        "numbers and the confusion matrix rather than the "
                        "headline."
                    ),
                    details={
                        "majority_class": majority_label,
                        "majority_share": round(share, 4),
                        "class_distribution": counts,
                        "threshold": CLASS_IMBALANCE_RATIO,
                    },
                )
            )

    known = [str(label) for label in train_class_labels if str(label)]
    missing = [label for label in known if label not in counts]
    if known and missing:
        found.append(
            Diagnostic(
                code=MISSING_CLASS_IN_TEST,
                severity=Severity.WARNING,
                message=(
                    "A class the model can predict does not appear in the "
                    f"held-out rows: {', '.join(sorted(missing)[:5])}. Every "
                    "per-class number for it was computed from no rows, so it "
                    "says nothing about how the model handles that class."
                ),
                details={
                    "missing_classes": sorted(missing),
                    "evaluated_classes": sorted(counts),
                },
            )
        )
    return found


def metric_diagnostics(unavailable: Mapping[str, Any] | None) -> list[Diagnostic]:
    """Report metrics the evaluation could not compute, and why.

    One diagnostic covering all of them rather than one each: they usually
    share a cause — an estimator with no probabilities makes ROC-AUC
    undefined — and a list of five identical warnings is noise.
    """
    if not unavailable:
        return []
    reasons = {str(key): str(value) for key, value in unavailable.items()}
    if not reasons:
        return []

    names = ", ".join(sorted(reasons))
    return [
        Diagnostic(
            code=UNDEFINED_METRIC,
            severity=Severity.INFO,
            message=(
                f"Some metrics could not be computed for this run: {names}. "
                "They are reported as unavailable with a reason rather than "
                "filled in with a substitute."
            ),
            details={"unavailable": reasons},
        )
    ]


def baseline_diagnostics(comparison: Mapping[str, Any] | None) -> list[Diagnostic]:
    """Say when the model did not beat the naive baseline.

    The most useful warning in the module, and the plainest: a model that does
    not beat "always predict the majority class" has not learned anything from
    the features, whatever its headline score looks like.
    """
    if not comparison:
        return []
    beats = comparison.get("beats_baseline")
    if beats is None or bool(beats):
        return []

    model_value = _finite(comparison.get("model_value"))
    baseline_value = _finite(comparison.get("baseline_value"))
    metric = str(comparison.get("metric") or "the primary metric")
    scores = (
        f" ({model_value:.4f} against {baseline_value:.4f})"
        if model_value is not None and baseline_value is not None
        else ""
    )
    return [
        Diagnostic(
            code=BASELINE_NOT_BEATEN,
            severity=Severity.WARNING,
            message=(
                f"The selected model did not beat the naive baseline on "
                f"{metric}{scores}. A model that does not improve on always "
                "predicting the majority class — or the mean — has not learned "
                "much from the features, whatever the headline number looks "
                "like."
            ),
            details=dict(comparison),
        )
    ]


def selection_diagnostics(
    *, uses_test_data: Any, strategy: Any = None
) -> list[Diagnostic]:
    """Say when the reported test score also chose the model.

    Under the holdout strategy, selection and final evaluation are the same
    measurement, so the reported score is the best of several draws rather than
    an unbiased estimate. The record has always carried ``is_unbiased``; this
    puts it where a reader will see it.
    """
    if not bool(uses_test_data):
        return []
    return [
        Diagnostic(
            code=SELECTION_USED_TEST_DATA,
            severity=Severity.WARNING,
            message=(
                "The held-out score also chose the model. Under the "
                f"'{strategy or 'holdout'}' strategy the winner is picked by "
                "its test-set score, so that score is the best of several "
                "draws and is optimistic as an estimate of future performance. "
                "Cross-validated selection keeps the two apart."
            ),
            details={"strategy": str(strategy) if strategy else None},
        )
    ]


def _higher_is_better(direction: MetricDirection | str | None) -> bool:
    """Read a metric direction from either an enum or its stored string."""
    if isinstance(direction, MetricDirection):
        return direction is MetricDirection.HIGHER_IS_BETTER
    return str(direction or MetricDirection.HIGHER_IS_BETTER.value) == (
        MetricDirection.HIGHER_IS_BETTER.value
    )


def diagnose_run(
    *,
    metric: str,
    direction: MetricDirection | str | None,
    selection_score: Any = None,
    selection_score_std: Any = None,
    held_out_score: Any = None,
    folds: Any = None,
    uses_test_data: Any = False,
    strategy: Any = None,
    row_count: Any = None,
    train_row_count: Any = None,
    test_row_count: Any = None,
    classification_details: Mapping[str, Any] | None = None,
    train_class_labels: Sequence[Any] = (),
    unavailable_metrics: Mapping[str, Any] | None = None,
    baseline_comparison: Mapping[str, Any] | None = None,
) -> tuple[Diagnostic, ...]:
    """Run every check over one finished experiment's recorded numbers.

    Every argument is a value the run already stored. Nothing is fitted, no
    dataset is read, and no cell value is seen — which is what makes this
    callable from a stored record long after the data is gone.

    Returns:
        tuple[Diagnostic, ...]: In a fixed order, warnings before notes, so two
        runs of the same experiment produce the same list.
    """
    found: list[Diagnostic] = []
    found.extend(selection_diagnostics(uses_test_data=uses_test_data, strategy=strategy))
    found.extend(
        generalisation_diagnostics(
            selection_score=selection_score,
            held_out_score=held_out_score,
            direction=direction,
            metric=metric,
        )
    )
    found.extend(
        variability_diagnostics(
            selection_score=selection_score,
            selection_score_std=selection_score_std,
            metric=metric,
            folds=folds,
        )
    )
    found.extend(baseline_diagnostics(baseline_comparison))
    found.extend(
        class_diagnostics(classification_details, train_class_labels=train_class_labels)
    )
    found.extend(
        size_diagnostics(
            row_count=row_count,
            train_row_count=train_row_count,
            test_row_count=test_row_count,
        )
    )
    found.extend(metric_diagnostics(unavailable_metrics))

    # Warnings first, and otherwise in the order the checks ran. Stable, so a
    # rendered list does not reshuffle between two views of the same run.
    return tuple(
        sorted(found, key=lambda item: 0 if item.severity.is_warning else 1)
    )


def summarise_diagnostics(diagnostics: Sequence[Diagnostic]) -> dict[str, Any]:
    """Count what was found, for a listing that has no room for the sentences."""
    warnings = [item for item in diagnostics if item.severity.is_warning]
    return {
        "count": len(diagnostics),
        "warning_count": len(warnings),
        "codes": [item.code for item in diagnostics],
    }


__all__ = [
    "BASELINE_NOT_BEATEN",
    "CLASS_IMBALANCE",
    "CLASS_IMBALANCE_RATIO",
    "GENERALISATION_GAP",
    "GENERALISATION_GAP_RATIO",
    "HIGH_CV_VARIABILITY",
    "HIGH_CV_VARIABILITY_RATIO",
    "MISSING_CLASS_IN_TEST",
    "SELECTION_USED_TEST_DATA",
    "SEVERE_CLASS_IMBALANCE_RATIO",
    "SMALL_DATASET",
    "SMALL_DATASET_ROWS",
    "SMALL_TEST_ROWS",
    "SMALL_TEST_SET",
    "UNDEFINED_METRIC",
    "Diagnostic",
    "Severity",
    "baseline_diagnostics",
    "class_diagnostics",
    "diagnose_run",
    "generalisation_diagnostics",
    "metric_diagnostics",
    "selection_diagnostics",
    "size_diagnostics",
    "summarise_diagnostics",
    "variability_diagnostics",
]
