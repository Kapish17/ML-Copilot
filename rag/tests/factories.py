"""Deterministic builders for the retrieval test suite.

Everything the tests need is generated here: small Markdown documents, a
synthetic experiment record, and a fake embedding provider.

The fake provider matters most. Almost every test needs *an* embedding, and
none of them should need PyTorch or a model download to get one — so the
default provider under test is a deterministic bag-of-words projection with a
tiny fixed vocabulary. It behaves like a real provider in the ways the rest of
the layer depends on (fixed dimension, unit length, stable across processes)
and it is legible enough that a test can reason about which document *should*
win a query.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

import numpy as np

from rag.embeddings.base import VECTOR_DTYPE, normalise

CSV_ENCODING = "utf-8"


# --------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------


class FakeEmbeddingProvider:
    """A deterministic embedding provider with no dependencies.

    Each token is hashed to a dimension and its count added there, then the
    vector is normalised. Two texts sharing words point in similar directions;
    two texts sharing nothing are orthogonal. That is enough structure for the
    store, the indexer and the retrieval service to be tested honestly, and it
    needs nothing beyond numpy.

    ``load_count`` records how often the vocabulary was built, so a test can
    assert that nothing is loaded until something is actually embedded.
    """

    def __init__(self, dimension: int = 32, *, name: str = "fake") -> None:
        """Configure the provider without building anything."""
        self._dimension = int(dimension)
        self._name = name
        self._ready = False
        self.load_count = 0
        self.embed_document_calls = 0
        self.embed_query_calls = 0

    @property
    def identifier(self) -> str:
        """Stable name of this provider and its dimension."""
        return f"{self._name}-{self._dimension}"

    @property
    def dimension(self) -> int:
        """Length of every vector this provider returns."""
        return self._dimension

    @property
    def is_loaded(self) -> bool:
        """Whether the provider has done its (pretend) loading."""
        return self._ready

    def _ensure_ready(self) -> None:
        """Do the deferred work exactly once, so laziness is observable."""
        if self._ready:
            return
        self._ready = True
        self.load_count += 1

    def _vector(self, text: str) -> np.ndarray:
        """Project one text into the vector space."""
        vector = np.zeros(self._dimension, dtype=VECTOR_DTYPE)
        for token in text.lower().split():
            word = "".join(character for character in token if character.isalnum())
            if not word:
                continue
            digest = hashlib.sha256(word.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self._dimension
            vector[index] += 1.0
        return vector

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Embed passages for indexing."""
        self._ensure_ready()
        self.embed_document_calls += 1
        items = list(texts)
        if not items:
            return np.zeros((0, self._dimension), dtype=VECTOR_DTYPE)
        return normalise(np.vstack([self._vector(text) for text in items]))

    def embed_query(self, text: str) -> np.ndarray:
        """Embed one question."""
        self._ensure_ready()
        self.embed_query_calls += 1
        return normalise(self._vector(text))[0]


# --------------------------------------------------------------------------
# Documentation
# --------------------------------------------------------------------------


LEAKAGE_DOC = """# Preprocessing Guide

Everything about turning a dataset into model-ready features.

## Leakage prevention

Every transformer is fitted on the training rows alone. The median that fills a
missing value, the categories a one-hot encoder knows and the mean a scaler
subtracts are all computed from training data and then applied to the test set
unchanged. Nothing the test set contains can influence how a feature is built.

## Categorical columns

Categorical columns are one-hot encoded. Unknown categories seen at prediction
time are ignored rather than raising, and a column with more distinct values
than the cardinality limit is excluded instead of encoded.

## Datetime columns

Datetime columns are expanded into year, month, day and weekday components.
"""

EVALUATION_DOC = """# Evaluation Guide

## Cross-validation versus the final test

Cross-validation scores every candidate model over folds of the training rows.
The winner is retrained on the full training data and measured exactly once on
the untouched test set, which is what makes that final number unbiased.

## Baselines

Every result is reported against a naive baseline, because a score without a
reference point says nothing.
"""

TINY_DOC = """# Tiny

One line.
"""


def documentation_files() -> dict[str, str]:
    """The Markdown files a documentation test should write to disk."""
    return {
        "PREPROCESSING.md": LEAKAGE_DOC,
        "EVALUATION.md": EVALUATION_DOC,
    }


def write_documentation(root, files: dict[str, str] | None = None) -> list[str]:
    """Write test documentation into a directory and return the references."""
    written = []
    for name, text in (files or documentation_files()).items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding=CSV_ENCODING)
        written.append(name)
    return written


# --------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------


class FakeSection:
    """A record section built from keyword arguments."""

    def __init__(self, **fields: Any) -> None:
        """Set every given field as an attribute."""
        for key, value in fields.items():
            setattr(self, key, value)


class FakeExperimentRun:
    """A stand-in for ``ExperimentRun`` with the fields ingestion reads.

    Structural rather than a real record, so the retrieval tests stay
    independent of the experiment schema's internals while still exercising
    exactly the attributes the adapter touches.
    """

    def __init__(
        self,
        *,
        experiment_id: str = "exp_abc123def456_20260101T120000Z_0001",
        name: str = "renewal baseline",
        task_type: str = "classification",
        selected_model: str = "random_forest_classifier",
        primary_metric: str = "f1",
        fingerprint: str = "86494cff7a45cb7f",
        target_column: str = "renewed",
        test_score: float = 0.85,
        selection_score: float = 0.87,
        tags: tuple[str, ...] = ("baseline",),
        with_explanation: bool = True,
    ) -> None:
        """Build a complete synthetic record."""
        self.experiment_id = experiment_id
        self.configuration_hash = experiment_id.split("_")[1]
        self.created_at = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
        self.name = name
        self.description = "a synthetic run for the retrieval tests"
        self.tags = tags
        self.schema_version = "1.0"

        self.dataset = FakeSection(
            fingerprint=fingerprint,
            fingerprint_algorithm="sha256",
            row_count=240,
            column_count=4,
            target_column=target_column,
            task_type=task_type,
            columns=("income", "tenure_months", "segment", target_column),
            dtypes={"income": "int64", "tenure_months": "int64", "segment": "str"},
            source_format="csv",
            data_quality_issues=(
                {"code": "possible_id_column", "columns": ["customer_id"]},
            ),
        )
        self.preprocessing = FakeSection(
            config={"scaling_strategy": "standard", "numeric_imputation": "median"},
            feature_groups={"numeric": ["income", "tenure_months"], "categorical": ["segment"]},
            selected_columns=("income", "tenure_months", "segment"),
            excluded_columns=("customer_id",),
            identifier_columns=(),
            transformed_feature_names=("income", "tenure_months", "segment_business"),
            column_decisions=(
                {
                    "column": "income",
                    "role": "feature",
                    "reason": "Profiled as integer; handled by the numeric branch.",
                },
            ),
            train_row_count=192,
            test_row_count=48,
            test_size=0.2,
            random_state=42,
            stratified=True,
            stratification_note=None,
            rows_dropped_missing_target=0,
        )
        self.selection = FakeSection(
            strategy="cross_validation",
            folds=5,
            primary_metric=primary_metric,
            primary_metric_direction="higher_is_better",
            candidate_models=("logistic_regression", selected_model),
            candidates=(
                {
                    "model_name": "logistic_regression",
                    "status": "succeeded",
                    "score": 0.81,
                    "score_std": 0.03,
                    "error": None,
                },
                {
                    "model_name": selected_model,
                    "status": "succeeded",
                    "score": selection_score,
                    "score_std": 0.02,
                    "error": None,
                },
            ),
            selected_model=selected_model,
            selection_score=selection_score,
            selection_score_std=0.02,
            scored_on="training_folds",
            uses_test_data=False,
        )
        self.evaluation = FakeSection(
            primary_metric=primary_metric,
            primary_metric_value=test_score,
            metrics={"accuracy": 0.86, primary_metric: test_score},
            unavailable_metrics={},
            baseline_identifier="most_frequent",
            baseline_metrics={primary_metric: 0.71},
            baseline_comparison={
                "metric": primary_metric,
                "direction": "higher_is_better",
                "model_value": test_score,
                "baseline_value": 0.71,
                "absolute_improvement": round(test_score - 0.71, 4),
                "relative_improvement": None,
                "beats_baseline": True,
            },
            classification_details={
                "class_count": 2,
                "class_labels": ["no", "yes"],
                "averaging": "binary",
                "positive_label": "yes",
            },
            test_row_count=48,
            is_unbiased=True,
        )
        self.explainability = (
            FakeSection(
                status="available",
                method="shap",
                explainer="TreeExplainer",
                aggregation="mean_absolute",
                explained_output="probability",
                feature_importances=(
                    {"feature": "tenure_months", "importance": 0.31, "rank": 1},
                    {"feature": "income", "importance": 0.27, "rank": 2},
                    {"feature": "segment_business", "importance": 0.04, "rank": 3},
                ),
                sample_count=192,
                feature_count=3,
                reason=None,
                warnings=(),
            )
            if with_explanation
            else None
        )
        self.environment = FakeSection(
            python_version="3.11.9",
            platform="Linux-6.1.0-x86_64",
            packages={"pandas": "3.0.5", "scikit-learn": "1.9.0"},
            random_state=42,
        )

    @property
    def task_type(self) -> str:
        """The dataset's task."""
        return self.dataset.task_type

    @property
    def selected_model(self) -> str:
        """The winning model's identifier."""
        return self.selection.selected_model

    @property
    def primary_metric(self) -> str:
        """The metric the winner was chosen by."""
        return self.selection.primary_metric


class FakeExperimentStore:
    """A stand-in for ``ExperimentStore`` holding runs in memory.

    Satisfies the structural protocol the ingestion adapter depends on, which
    is the point: RAG never imports a concrete store.
    """

    def __init__(self, runs: Sequence[Any] = ()) -> None:
        """Hold the given runs."""
        self.runs = list(runs)
        self.list_calls = 0

    def list(self, query: Any = None) -> list[Any]:
        """Return every stored run."""
        self.list_calls += 1
        return list(self.runs)

    def add(self, run: Any) -> None:
        """Add one run."""
        self.runs.append(run)

    def remove(self, experiment_id: str) -> None:
        """Drop one run by id."""
        self.runs = [run for run in self.runs if run.experiment_id != experiment_id]


__all__ = [
    "EVALUATION_DOC",
    "LEAKAGE_DOC",
    "TINY_DOC",
    "FakeEmbeddingProvider",
    "FakeExperimentRun",
    "FakeExperimentStore",
    "FakeSection",
    "documentation_files",
    "write_documentation",
]
