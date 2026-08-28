"""Synthetic datasets and profile stand-ins used by the ML test suite.

Everything is generated in memory with a fixed seed. No test reads an external
dataset or touches the network.

The ``Fake*`` classes below implement only the attributes the ML layer reads
from a dataset profile. They exist to prove that the profiling adapter depends
on the shape of a profile rather than on a particular class — the real
profiling output is exercised separately in ``test_profile_integration.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

SEED = 20240826


@dataclass
class FakeProfiledColumn:
    """A profiled column, carrying only what the ML layer reads."""

    name: str
    inferred_type: Any
    is_constant: bool = False
    unique_count: int = 10
    non_null_count: int = 100


@dataclass
class FakeQualityIssue:
    """A quality finding, carrying only what the ML layer reads."""

    code: str
    columns: list[str] = field(default_factory=list)


@dataclass
class FakeQualityReport:
    """A collection of quality findings."""

    issues: list[FakeQualityIssue] = field(default_factory=list)


@dataclass
class FakeTargetProfile:
    """The target section of a profile."""

    name: str
    task_suggestion: str


@dataclass
class FakeProfile:
    """A dataset profile stand-in."""

    columns: list[FakeProfiledColumn]
    quality: FakeQualityReport = field(default_factory=FakeQualityReport)
    target: FakeTargetProfile | None = None


def churn_profile() -> FakeProfile:
    """A profile covering every branch of the inference rules."""
    return FakeProfile(
        columns=[
            FakeProfiledColumn("customer_id", "integer", unique_count=100),
            FakeProfiledColumn("age", "float"),
            FakeProfiledColumn("monthly_charges", "float"),
            FakeProfiledColumn("contract", "categorical", unique_count=3),
            FakeProfiledColumn("is_active", "boolean", unique_count=2),
            FakeProfiledColumn("signup_date", "datetime", unique_count=90),
            FakeProfiledColumn("notes", "text", unique_count=100),
            FakeProfiledColumn("region_code", "categorical", unique_count=400),
            FakeProfiledColumn("plan", "categorical", is_constant=True, unique_count=1),
            FakeProfiledColumn("blank", "empty", non_null_count=0, unique_count=0),
            FakeProfiledColumn("churn", "categorical", unique_count=2),
        ],
        quality=FakeQualityReport(
            issues=[
                FakeQualityIssue("possible_id_column", ["customer_id"]),
                FakeQualityIssue("missing_values", ["age"]),
            ]
        ),
        target=FakeTargetProfile("churn", "classification"),
    )


def classification_frame(rows: int = 120) -> pd.DataFrame:
    """A mixed-type dataset with missing values and a binary target."""
    rng = np.random.default_rng(SEED)
    frame = pd.DataFrame(
        {
            "customer_id": range(1, rows + 1),
            "age": rng.integers(18, 80, rows).astype("float64"),
            "monthly_charges": rng.normal(70, 20, rows).round(2),
            "contract": rng.choice(
                ["Month-to-month", "One-year", "Two-year"], rows
            ),
            "payment_method": rng.choice(
                ["Electronic check", "Mailed check", "Credit card"], rows
            ),
            "is_active": rng.choice([True, False], rows),
            "signup_date": pd.to_datetime("2022-01-01")
            + pd.to_timedelta(rng.integers(0, 700, rows), unit="D"),
            "notes": [f"free text note {index}" for index in range(rows)],
            "plan": ["basic"] * rows,
            "churn": rng.choice(["yes", "no"], rows, p=[0.3, 0.7]),
        }
    )
    frame.loc[frame.index[:10], "age"] = np.nan
    frame.loc[frame.index[5:12], "contract"] = np.nan
    return frame


def regression_frame(rows: int = 100) -> pd.DataFrame:
    """A numeric dataset with a continuous target."""
    rng = np.random.default_rng(SEED)
    size = rng.normal(120, 40, rows).round(1)
    rooms = rng.integers(1, 6, rows)
    return pd.DataFrame(
        {
            "size_sqm": size,
            "rooms": rooms.astype("float64"),
            "district": rng.choice(["north", "south", "east", "west"], rows),
            "price": (size * 3_000 + rooms * 10_000 + rng.normal(0, 5_000, rows)).round(
                2
            ),
        }
    )


def learnable_classification_frame(rows: int = 300) -> pd.DataFrame:
    """A binary dataset where the label really does follow the features.

    The churn fixture above is deliberately random, which is right for testing
    preprocessing but useless for testing that a model beats a baseline. Here
    the label is a thresholded function of two numeric columns plus a
    categorical effect, with a little noise so the problem is not trivially
    separable.
    """
    rng = np.random.default_rng(SEED)
    income = rng.normal(50_000, 15_000, rows).round(2)
    tenure = rng.integers(0, 120, rows).astype("float64")
    segment = rng.choice(["retail", "business", "public"], rows)
    segment_effect = pd.Series(segment).map(
        {"retail": 0.0, "business": 1.2, "public": -0.8}
    )

    signal = (
        (income - income.mean()) / income.std()
        + (tenure - tenure.mean()) / tenure.std()
        + segment_effect.to_numpy()
        + rng.normal(0, 0.4, rows)
    )
    return pd.DataFrame(
        {
            "income": income,
            "tenure_months": tenure,
            "segment": segment,
            "renewed": np.where(signal > 0, "yes", "no"),
        }
    )


def imbalanced_classification_frame(
    rows: int = 200, minority_count: int = 30
) -> pd.DataFrame:
    """A binary dataset with a small but workable minority class.

    Used to check that stratified folds keep the rare class present in every
    validation fold, which plain K-fold cannot guarantee.
    """
    rng = np.random.default_rng(SEED)
    labels = np.array(["majority"] * (rows - minority_count) + ["minority"] * minority_count)
    rng.shuffle(labels)
    signal = np.where(labels == "minority", 1.5, -0.5) + rng.normal(0, 0.7, rows)
    return pd.DataFrame(
        {
            "measure": signal.round(4),
            "noise": rng.normal(0, 1, rows).round(4),
            "outcome": labels,
        }
    )


def rare_class_frame(rows: int = 40, rare_count: int = 3) -> pd.DataFrame:
    """A dataset whose rare class has too few rows for a five-fold split."""
    rng = np.random.default_rng(SEED)
    labels = np.array(["common"] * (rows - rare_count) + ["rare"] * rare_count)
    return pd.DataFrame(
        {
            "measure": rng.normal(0, 1, rows).round(4),
            "noise": rng.normal(0, 1, rows).round(4),
            "outcome": labels,
        }
    )


def multiclass_frame(rows: int = 300) -> pd.DataFrame:
    """A three-class dataset with a learnable signal."""
    rng = np.random.default_rng(SEED)
    first = rng.normal(0, 1, rows)
    second = rng.normal(0, 1, rows)
    score = first + second + rng.normal(0, 0.3, rows)
    grade = np.where(score < -0.8, "low", np.where(score < 0.8, "medium", "high"))
    return pd.DataFrame(
        {
            "first_measure": first.round(4),
            "second_measure": second.round(4),
            "grade": grade,
        }
    )


def constant_target_classification_frame(rows: int = 40) -> pd.DataFrame:
    """A dataset whose label never varies, so ROC-AUC cannot be computed."""
    rng = np.random.default_rng(SEED)
    return pd.DataFrame(
        {
            "value": rng.normal(0, 1, rows).round(4),
            "other": rng.normal(5, 2, rows).round(4),
            "label": ["only_class"] * rows,
        }
    )


def constant_target_regression_frame(rows: int = 40) -> pd.DataFrame:
    """A dataset whose numeric target never varies, so R² is undefined."""
    rng = np.random.default_rng(SEED)
    return pd.DataFrame(
        {
            "value": rng.normal(0, 1, rows).round(4),
            "other": rng.normal(5, 2, rows).round(4),
            "amount": [7.5] * rows,
        }
    )


def numeric_frame() -> pd.DataFrame:
    """A tiny numeric dataset with one missing value and a binary target."""
    return pd.DataFrame(
        {
            "value": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0],
            "other": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
            "label": ["a", "b", "a", "b", "a", "b", "a", "b"],
        }
    )


def categorical_frame() -> pd.DataFrame:
    """A tiny categorical dataset with one missing value and a binary target."""
    return pd.DataFrame(
        {
            "contract": [
                "Month-to-month",
                "One-year",
                "Two-year",
                None,
                "Month-to-month",
                "One-year",
                "Two-year",
                "Month-to-month",
            ],
            "label": ["a", "b", "a", "b", "a", "b", "a", "b"],
        }
    )


def experiment_run(
    *,
    experiment_id: str = "exp_0123456789ab_20260101T000000Z_0001",
    configuration_hash: str = "0123456789ab",
    name: str = "test run",
    fingerprint: str = "abcdef0123456789",
    target_column: str = "renewed",
    task_type: str = "classification",
    primary_metric: str = "f1",
    model_name: str = "logistic_regression",
    strategy: str = "cross_validation",
    selection_score: float | None = 0.81,
    selection_score_std: float | None = 0.02,
    test_score: float | None = 0.80,
    baseline_score: float | None = 0.71,
    created_at: datetime | None = None,
    tags: tuple[str, ...] = (),
):
    """Build a small ExperimentRun without running the whole pipeline.

    The store and comparison tests care about the record's shape and its
    metadata, not about how a real model scored, so they use this rather than
    training anything.
    """
    from ml.experiments.run import (
        DatasetSection,
        EnvironmentSection,
        EvaluationSection,
        ExperimentRun,
        ExplainabilitySection,
        PreprocessingSection,
        SelectionSection,
    )

    improvement = (
        None
        if test_score is None or baseline_score is None
        else round(test_score - baseline_score, 6)
    )
    return ExperimentRun(
        experiment_id=experiment_id,
        configuration_hash=configuration_hash,
        created_at=created_at or datetime(2026, 1, 1, tzinfo=timezone.utc),
        name=name,
        tags=tags,
        dataset=DatasetSection(
            fingerprint=fingerprint,
            row_count=300,
            column_count=4,
            target_column=target_column,
            task_type=task_type,
            columns=("income", "tenure_months", "segment", target_column),
            dtypes={"income": "float64"},
            source_format="csv",
        ),
        preprocessing=PreprocessingSection(
            config={"scaling_strategy": "standard"},
            feature_groups={"numeric": ["income", "tenure_months"]},
            selected_columns=("income", "tenure_months"),
            transformed_feature_names=("income", "tenure_months"),
            train_row_count=240,
            test_row_count=60,
            test_size=0.2,
            random_state=42,
            stratified=True,
        ),
        selection=SelectionSection(
            strategy=strategy,
            folds=5,
            primary_metric=primary_metric,
            primary_metric_direction=(
                "higher_is_better" if primary_metric != "rmse" else "lower_is_better"
            ),
            candidate_models=(model_name,),
            candidates=({"model_name": model_name, "status": "succeeded"},),
            selected_model=model_name,
            selection_score=selection_score,
            selection_score_std=selection_score_std,
            scored_on="training_folds",
        ),
        evaluation=EvaluationSection(
            primary_metric=primary_metric,
            primary_metric_value=test_score,
            metrics={primary_metric: test_score} if test_score is not None else {},
            baseline_identifier="majority_class_baseline",
            baseline_metrics=(
                {primary_metric: baseline_score} if baseline_score is not None else {}
            ),
            baseline_comparison={"absolute_improvement": improvement},
            test_row_count=60,
            is_unbiased=True,
        ),
        explainability=ExplainabilitySection(
            status="available",
            method="shap",
            explainer="LinearExplainer",
            feature_importances=({"feature": "income", "importance": 0.5, "rank": 1},),
            sample_count=240,
            feature_count=2,
        ),
        environment=EnvironmentSection(
            python_version="3.11.0",
            platform="Linux-x86_64",
            packages={"pandas": "3.0.5"},
            random_state=42,
        ),
    )
