"""Tests for train/test splitting and target-distribution reporting."""

from __future__ import annotations

import pandas as pd
import pytest

from ml.evaluation.splitting import (
    describe_target,
    looks_discrete,
    resolve_stratification,
    resolve_task_type,
    split_dataset,
)
from ml.features.types import TaskType

MAX_CLASSES = 20


def _features(rows: int) -> pd.DataFrame:
    """A trivial feature frame of the requested length."""
    return pd.DataFrame({"x": range(rows)})


def _balanced_target(rows: int) -> pd.Series:
    """A two-class target with an even split."""
    return pd.Series(["a" if index % 2 else "b" for index in range(rows)])


def test_split_respects_the_test_size() -> None:
    """The requested fraction of rows is held out."""
    split = split_dataset(
        _features(100),
        _balanced_target(100),
        test_size=0.25,
        random_state=1,
        task_type=TaskType.CLASSIFICATION,
    )
    assert split.X_train.shape[0] == 75
    assert split.X_test.shape[0] == 25


def test_split_keeps_features_and_target_aligned() -> None:
    """Each split half keeps matching feature and target rows."""
    features, target = _features(40), _balanced_target(40)
    split = split_dataset(
        features, target, test_size=0.25, random_state=1, task_type=TaskType.CLASSIFICATION
    )

    assert list(split.X_train.index) == list(split.y_train.index)
    assert list(split.X_test.index) == list(split.y_test.index)
    assert set(split.X_train.index).isdisjoint(split.X_test.index)


def test_split_is_reproducible_for_a_given_seed() -> None:
    """The same seed always produces the same rows on each side."""
    arguments = {
        "test_size": 0.25,
        "random_state": 7,
        "task_type": TaskType.CLASSIFICATION,
    }
    first = split_dataset(_features(60), _balanced_target(60), **arguments)
    second = split_dataset(_features(60), _balanced_target(60), **arguments)

    assert list(first.X_test.index) == list(second.X_test.index)


def test_a_different_seed_produces_a_different_split() -> None:
    """The seed genuinely drives the shuffle."""
    first = split_dataset(
        _features(60),
        _balanced_target(60),
        test_size=0.25,
        random_state=1,
        task_type=TaskType.CLASSIFICATION,
    )
    second = split_dataset(
        _features(60),
        _balanced_target(60),
        test_size=0.25,
        random_state=2,
        task_type=TaskType.CLASSIFICATION,
    )
    assert list(first.X_test.index) != list(second.X_test.index)


def test_classification_split_is_stratified() -> None:
    """Class proportions are preserved on both sides."""
    target = pd.Series(["a"] * 80 + ["b"] * 20)
    split = split_dataset(
        _features(100),
        target,
        test_size=0.2,
        random_state=3,
        task_type=TaskType.CLASSIFICATION,
    )

    assert split.stratified is True
    assert split.y_train.value_counts(normalize=True)["b"] == pytest.approx(0.2)
    assert split.y_test.value_counts(normalize=True)["b"] == pytest.approx(0.2)


def test_regression_split_is_not_stratified() -> None:
    """A continuous target is never stratified, and the reason is reported."""
    target = pd.Series([float(index) for index in range(50)])
    split = split_dataset(
        _features(50), target, test_size=0.2, random_state=3, task_type=TaskType.REGRESSION
    )

    assert split.stratified is False
    assert "Regression" in (split.stratification_note or "")


def test_single_class_target_is_not_stratified() -> None:
    """Nothing to stratify when every row has the same label."""
    stratify, note = resolve_stratification(
        pd.Series(["a"] * 20), task_type=TaskType.CLASSIFICATION, test_size=0.2
    )
    assert stratify is False
    assert "fewer than two classes" in (note or "")


def test_tiny_class_disables_stratification_gracefully() -> None:
    """A class with one member cannot appear on both sides, so we fall back."""
    target = pd.Series(["a"] * 19 + ["rare"])
    stratify, note = resolve_stratification(
        target, task_type=TaskType.CLASSIFICATION, test_size=0.2
    )

    assert stratify is False
    assert "rare" in (note or "")


def test_stratification_falls_back_when_the_test_set_is_too_small() -> None:
    """More classes than test rows means stratification cannot be honoured."""
    target = pd.Series([f"class_{index % 6}" for index in range(12)])
    stratify, note = resolve_stratification(
        target, task_type=TaskType.CLASSIFICATION, test_size=0.1
    )

    assert stratify is False
    assert "classes" in (note or "")


def test_split_still_works_when_stratification_is_impossible() -> None:
    """Falling back must produce a usable split, not an error."""
    target = pd.Series(["a"] * 19 + ["rare"])
    split = split_dataset(
        _features(20), target, test_size=0.2, random_state=1, task_type=TaskType.CLASSIFICATION
    )

    assert split.stratified is False
    assert split.X_train.shape[0] + split.X_test.shape[0] == 20


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (["a", "b", "a"], True),
        ([True, False, True], True),
        ([0, 1, 1, 0], True),
        ([float(index) for index in range(50)], False),
    ],
)
def test_looks_discrete(values: list, expected: bool) -> None:
    """Only a small set of distinct labels counts as discrete."""
    assert looks_discrete(pd.Series(values), MAX_CLASSES) is expected


def test_many_distinct_integers_are_continuous() -> None:
    """An integer column with many values is a regression target."""
    assert looks_discrete(pd.Series(range(200)), MAX_CLASSES) is False


def test_declared_task_type_wins_over_inspection() -> None:
    """An explicit task from the profile is never second-guessed."""
    target = pd.Series([float(index) for index in range(50)])
    assert (
        resolve_task_type(target, TaskType.CLASSIFICATION, MAX_CLASSES)
        is TaskType.CLASSIFICATION
    )


def test_auto_task_type_uses_the_data() -> None:
    """Without a declared task, the target's shape decides."""
    assert (
        resolve_task_type(pd.Series(["a", "b"]), TaskType.AUTO, MAX_CLASSES)
        is TaskType.CLASSIFICATION
    )
    assert (
        resolve_task_type(
            pd.Series([float(index) for index in range(50)]), TaskType.AUTO, MAX_CLASSES
        )
        is TaskType.REGRESSION
    )


def test_describe_classification_target() -> None:
    """Class counts, shares and the imbalance ratio are reported."""
    distribution = describe_target(
        pd.Series(["a"] * 80 + ["b"] * 20), TaskType.CLASSIFICATION
    )

    assert distribution.class_counts == {"a": 80, "b": 20}
    assert distribution.class_percentages == {"a": 80.0, "b": 20.0}
    assert distribution.majority_class == "a"
    assert distribution.minority_class == "b"
    assert distribution.imbalance_ratio == pytest.approx(4.0)


def test_describe_regression_target() -> None:
    """A continuous target is summarised with statistics, not classes."""
    distribution = describe_target(
        pd.Series([1.0, 2.0, 3.0, 4.0]), TaskType.REGRESSION
    )

    assert distribution.class_counts is None
    assert distribution.numeric_summary is not None
    assert distribution.numeric_summary["mean"] == pytest.approx(2.5)
    assert distribution.numeric_summary["minimum"] == pytest.approx(1.0)


def test_distribution_is_serialisable() -> None:
    """The distribution renders as plain values for later reporting."""
    payload = describe_target(pd.Series(["a", "b"]), TaskType.CLASSIFICATION).as_dict()
    assert payload["task_type"] == "classification"
    assert payload["class_counts"] == {"a": 1, "b": 1}
