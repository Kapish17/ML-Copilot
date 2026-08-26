"""Integration test across the profiling and preprocessing layers.

Everywhere else the ML layer is tested against profile stand-ins, which proves
it does not depend on a particular profiling class. This module does the
opposite: it runs the real profiler from the backend and feeds its output
straight into the ML layer, proving the two halves actually fit together and
that profiling information is reused rather than recomputed.

The backend is imported lazily so the ML suite still runs on its own.
"""

from __future__ import annotations

import pytest

pytest.importorskip("app", reason="the backend package is not on the path")

from app.core.config import Settings  # noqa: E402
from app.services.datasets import DatasetProfilingService  # noqa: E402

from ml.features.inference import infer_configuration  # noqa: E402
from ml.features.types import ColumnRole, ExclusionReason, TaskType  # noqa: E402
from ml.pipelines.preparation import prepare_dataset  # noqa: E402
from ml.tests.factories import classification_frame  # noqa: E402


@pytest.fixture(scope="module")
def profiled():
    """Profile the churn dataset with the real profiling service."""
    frame = classification_frame()
    content = frame.to_csv(index=False).encode("utf-8")
    profile = DatasetProfilingService(Settings()).profile_content(
        "churn.csv", content, target_column="churn"
    )
    return frame, profile


def test_real_profile_drives_the_feature_groups(profiled) -> None:
    """Each profiled type lands in the matching pipeline branch."""
    _, profile = profiled
    config = infer_configuration(profile, target_column="churn").config

    assert set(config.numeric_columns) == {"age", "monthly_charges"}
    assert set(config.categorical_columns) == {"contract", "payment_method"}
    assert config.boolean_columns == ("is_active",)
    assert config.datetime_columns == ("signup_date",)


def test_profiling_findings_exclude_columns(profiled) -> None:
    """The identifier, the free-text column and the constant column are dropped."""
    _, profile = profiled
    inferred = infer_configuration(profile, target_column="churn")
    decisions = {item.column: item for item in inferred.decisions}

    assert decisions["customer_id"].role is ColumnRole.IDENTIFIER
    assert decisions["customer_id"].reason_code is ExclusionReason.PROFILE_POSSIBLE_ID
    assert decisions["notes"].reason_code is ExclusionReason.FREE_TEXT
    assert decisions["plan"].reason_code is ExclusionReason.CONSTANT_COLUMN


def test_task_type_is_reused_from_the_profile(profiled) -> None:
    """The profiler already decided the task; the ML layer does not redo it."""
    _, profile = profiled
    config = infer_configuration(profile, target_column="churn").config

    assert profile.target is not None
    assert profile.target.task_suggestion.value == "classification"
    assert config.task_type is TaskType.CLASSIFICATION


def test_profile_to_prepared_dataset(profiled) -> None:
    """The full path from a profiled dataset to model-ready arrays works."""
    frame, profile = profiled
    inferred = infer_configuration(profile, target_column="churn")
    prepared = prepare_dataset(frame, inferred.config, decisions=inferred.decisions)

    assert prepared.train_row_count + prepared.test_row_count == len(frame)
    assert prepared.X_train.isna().to_numpy().sum() == 0
    assert "contract_Month-to-month" in prepared.feature_names
    assert "churn" not in prepared.X_train.columns
    assert len(prepared.column_decisions) == len(frame.columns)


def test_profiling_reasons_survive_into_the_result(profiled) -> None:
    """The explanation produced during profiling reaches the final result."""
    frame, profile = profiled
    inferred = infer_configuration(profile, target_column="churn")
    prepared = prepare_dataset(frame, inferred.config, decisions=inferred.decisions)

    summary = prepared.summary()
    reasons = {item["column"]: item["reason_code"] for item in summary["column_decisions"]}

    assert reasons["customer_id"] == ExclusionReason.PROFILE_POSSIBLE_ID.value
    assert reasons["notes"] == ExclusionReason.FREE_TEXT.value
