"""Turning stored experiments into searchable documents.

An :class:`~ml.experiments.run.ExperimentRun` is a nested record built for
machines. This module renders it as structured Markdown built for retrieval:
one document per experiment, with a heading per section, so the chunker splits
it where the subject changes and a question about feature importance finds the
importance section rather than the whole run.

**Only stored facts are written.** Every line is a value taken from the
record — a metric, a model name, a decision, a warning. Nothing here judges,
summarises or concludes: this module never writes "the model performed well"
or "the forest was the better choice", because neither is in the record. That
reading is a job for a future model with the evidence in front of it, and
inventing it here would put ungrounded prose into the index where it would be
retrieved and cited as fact. **No LLM generation is implemented.**

The metadata carried on every chunk — experiment id, dataset fingerprint, task,
target, selected model, primary metric — is what makes filtered retrieval
possible: "the best classification runs on this dataset" is a metadata filter
plus a semantic query, not a full scan.

The dependency runs one way. This module reads the experiment store; nothing
in ``ml/experiments`` knows the retrieval layer exists, so experiments can be
recorded with no index present and the index can be rebuilt from them at any
time.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from rag.config import RagConfig
from rag.documents import Document, SourceType

logger = logging.getLogger(__name__)

#: Decimal places used when rendering a metric.
SCORE_PRECISION = 4
#: Most ranked features written into a document. The record itself may hold
#: more; a passage listing two hundred features retrieves poorly.
MAX_RENDERED_FEATURES = 25
#: Most per-column preprocessing decisions written out in full.
MAX_RENDERED_DECISIONS = 40


@runtime_checkable
class ExperimentStoreLike(Protocol):
    """The part of an experiment store this module uses.

    Structural, so the retrieval layer depends on the *shape* of a store
    rather than importing one. Any implementation of Commit 7's
    ``ExperimentStore`` satisfies it, and so does a stand-in in a test.
    """

    def list(self, query: Any = None) -> Sequence[Any]:
        """Return the stored runs."""
        ...  # pragma: no cover - protocol


def _number(value: Any, precision: int = SCORE_PRECISION) -> str:
    """Render a number for reading, or ``n/a`` when there is none."""
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        if isinstance(value, float):
            if value != value:  # NaN
                return "n/a"
            return f"{value:.{precision}f}"
        return str(value)
    return str(value)


def _score_with_spread(value: Any, spread: Any) -> str:
    """Render a score and its standard deviation, when there is one."""
    if value is None:
        return "n/a"
    if spread is None:
        return _number(value)
    return f"{_number(value)} ± {_number(spread)}"


def _metric_lines(metrics: Mapping[str, Any]) -> list[str]:
    """Render a metric mapping as sorted ``- key: value`` lines."""
    return [f"- {key}: {_number(value)}" for key, value in sorted(metrics.items())]


def _overview_section(run: Any) -> list[str]:
    """Identity and labels: what this run was and when it happened."""
    lines = [
        f"Experiment ID: {run.experiment_id}",
        f"Name: {run.name}",
        f"Created at: {run.created_at.isoformat()}",
        f"Configuration hash: {run.configuration_hash}",
        f"Task: {run.task_type}",
        f"Selected model: {run.selected_model}",
        f"Primary metric: {run.primary_metric}",
    ]
    if run.description:
        lines.append(f"Description: {run.description}")
    if run.tags:
        lines.append(f"Tags: {', '.join(run.tags)}")
    return lines


def _dataset_section(run: Any) -> list[str]:
    """What data the run used, identified by content rather than filename."""
    dataset = run.dataset
    lines = [
        f"Dataset fingerprint: {dataset.fingerprint}",
        f"Fingerprint algorithm: {dataset.fingerprint_algorithm}",
        f"Rows: {dataset.row_count}",
        f"Columns: {dataset.column_count}",
        f"Target column: {dataset.target_column}",
        f"Task type: {dataset.task_type}",
    ]
    if dataset.source_format:
        lines.append(f"Source format: {dataset.source_format}")
    if dataset.columns:
        lines.append(f"Column names: {', '.join(dataset.columns)}")
    if dataset.dtypes:
        lines.append("Column types:")
        lines.extend(
            f"- {name}: {dtype}" for name, dtype in sorted(dataset.dtypes.items())
        )
    if dataset.data_quality_issues:
        lines.append("Data quality findings recorded at profiling time:")
        for issue in dataset.data_quality_issues:
            code = issue.get("code", "unknown")
            columns = ", ".join(str(item) for item in issue.get("columns", ()) or ())
            lines.append(f"- {code}" + (f" ({columns})" if columns else ""))
    return lines


def _preprocessing_section(run: Any) -> list[str]:
    """How the data was prepared, and what was left out of the features."""
    preprocessing = run.preprocessing
    lines = [
        f"Training rows: {preprocessing.train_row_count}",
        f"Test rows: {preprocessing.test_row_count}",
        f"Test size: {_number(preprocessing.test_size, 2)}",
        f"Random state: {preprocessing.random_state}",
        f"Stratified split: {_number(preprocessing.stratified)}",
    ]
    if preprocessing.stratification_note:
        lines.append(f"Stratification note: {preprocessing.stratification_note}")
    if preprocessing.rows_dropped_missing_target:
        lines.append(
            "Rows dropped for a missing target: "
            f"{preprocessing.rows_dropped_missing_target}"
        )

    config = preprocessing.config or {}
    for key in (
        "scaling_strategy",
        "numeric_imputation",
        "categorical_imputation",
        "add_missing_indicators",
        "max_categorical_cardinality",
    ):
        if key in config:
            lines.append(f"{key.replace('_', ' ').capitalize()}: {_number(config[key])}")

    if preprocessing.feature_groups:
        lines.append("Feature groups:")
        for group, columns in sorted(preprocessing.feature_groups.items()):
            if columns:
                lines.append(f"- {group}: {', '.join(columns)}")
    if preprocessing.selected_columns:
        lines.append(f"Selected columns: {', '.join(preprocessing.selected_columns)}")
    if preprocessing.excluded_columns:
        lines.append(f"Excluded columns: {', '.join(preprocessing.excluded_columns)}")
    if preprocessing.identifier_columns:
        lines.append(
            f"Identifier columns: {', '.join(preprocessing.identifier_columns)}"
        )
    if preprocessing.transformed_feature_names:
        names = preprocessing.transformed_feature_names
        lines.append(f"Transformed feature count: {len(names)}")
        lines.append(f"Transformed feature names: {', '.join(names)}")

    decisions = preprocessing.column_decisions or ()
    if decisions:
        lines.append("Per-column decisions:")
        for decision in decisions[:MAX_RENDERED_DECISIONS]:
            column = decision.get("column", "?")
            role = decision.get("role", "?")
            reason = decision.get("reason") or ""
            lines.append(f"- {column}: {role}" + (f" — {reason}" if reason else ""))
        if len(decisions) > MAX_RENDERED_DECISIONS:
            lines.append(
                f"- (+{len(decisions) - MAX_RENDERED_DECISIONS} further columns)"
            )
    return lines


def _selection_section(run: Any) -> list[str]:
    """Which models were considered and how the winner was chosen."""
    selection = run.selection
    lines = [
        f"Selection strategy: {selection.strategy}",
        f"Primary metric: {selection.primary_metric}",
        f"Metric direction: {selection.primary_metric_direction or 'n/a'}",
        f"Selected model: {selection.selected_model}",
        "Selection score: "
        + _score_with_spread(selection.selection_score, selection.selection_score_std),
        f"Scored on: {selection.scored_on or 'n/a'}",
        f"Used test data for selection: {_number(selection.uses_test_data)}",
        # Rendered so a retrieved passage can answer "why this model?" with the
        # sentence the ML layer composed, rather than leaving an answering
        # model to reconstruct the reason from loose numbers.
        f"Why this model won: {selection.selection_rationale}",
    ]
    if selection.folds:
        lines.append(f"Cross-validation folds: {selection.folds}")
    if selection.candidate_models:
        lines.append(f"Candidate models: {', '.join(selection.candidate_models)}")
    if selection.candidates:
        lines.append("Candidate results:")
        for candidate in selection.candidates:
            name = candidate.get("model_name", "?")
            status = candidate.get("status", "?")
            score = _score_with_spread(
                candidate.get("score"), candidate.get("score_std")
            )
            line = f"- {name}: {score} ({status})"
            error = candidate.get("error")
            if error:
                line += f" — {error}"
            lines.append(line)
    return lines


def _evaluation_section(run: Any) -> list[str]:
    """The single measurement on data the model had never seen."""
    evaluation = run.evaluation
    lines = [
        f"Primary metric: {evaluation.primary_metric}",
        f"Final test score: {_number(evaluation.primary_metric_value)}",
        f"Test rows: {evaluation.test_row_count}",
        f"Unbiased evaluation: {_number(evaluation.is_unbiased)}",
    ]
    if evaluation.metrics:
        lines.append("Test metrics:")
        lines.extend(_metric_lines(evaluation.metrics))
    if evaluation.baseline_identifier:
        lines.append(f"Baseline: {evaluation.baseline_identifier}")
    if evaluation.baseline_metrics:
        lines.append("Baseline metrics:")
        lines.extend(_metric_lines(evaluation.baseline_metrics))

    comparison = evaluation.baseline_comparison or {}
    if comparison:
        lines.append("Comparison against the baseline:")
        for key in (
            "metric",
            "direction",
            "model_value",
            "baseline_value",
            "absolute_improvement",
            "relative_improvement",
            "beats_baseline",
        ):
            if key in comparison:
                lines.append(f"- {key}: {_number(comparison[key])}")

    details = evaluation.classification_details or {}
    if details:
        lines.append("Classification details:")
        for key in ("class_count", "averaging", "positive_label"):
            if key in details:
                lines.append(f"- {key}: {_number(details[key])}")
        labels = details.get("class_labels")
        if labels:
            lines.append(f"- class labels: {', '.join(str(item) for item in labels)}")

    if evaluation.unavailable_metrics:
        lines.append("Metrics that could not be computed:")
        lines.extend(
            f"- {key}: {reason}"
            for key, reason in sorted(evaluation.unavailable_metrics.items())
        )

    diagnostics = evaluation.diagnostics or ()
    if diagnostics:
        # Rendered verbatim, wording included. The messages are deliberately
        # written as signals rather than verdicts, and paraphrasing them into
        # a retrievable document is how "potential overfitting signal" becomes
        # "the model is overfit" three hops later.
        lines.append(
            "Diagnostics — signals worth a second look, not verdicts and not "
            "failures:"
        )
        for item in diagnostics:
            severity = item.get("severity", "info")
            lines.append(f"- [{severity}] {item.get('code', '?')}: "
                         f"{item.get('message', '')}")
    else:
        lines.append("Diagnostics: none were raised for this run.")
    return lines


def _explainability_section(run: Any) -> list[str]:
    """What the explanation found, or why there is none."""
    explainability = run.explainability
    if explainability is None:
        return ["No explanation was recorded for this experiment."]

    lines = [
        f"Explanation status: {explainability.status}",
        f"Explanation method: {explainability.method}",
    ]
    if explainability.explainer:
        lines.append(f"Explainer: {explainability.explainer}")
    if explainability.aggregation:
        lines.append(f"Aggregation: {explainability.aggregation}")
    if explainability.explained_output:
        lines.append(f"Explained output: {explainability.explained_output}")
    if explainability.sample_count:
        lines.append(f"Rows explained: {explainability.sample_count}")
    if explainability.feature_count:
        lines.append(f"Features explained: {explainability.feature_count}")
    if explainability.reason:
        lines.append(f"Reason: {explainability.reason}")

    importances = explainability.feature_importances or ()
    if importances:
        lines.append("Top features by importance:")
        for entry in importances[:MAX_RENDERED_FEATURES]:
            feature = entry.get("feature", "?")
            value = _number(entry.get("importance"))
            rank = entry.get("rank")
            prefix = f"{rank}. " if rank is not None else "- "
            lines.append(f"{prefix}{feature}: {value}")
        if len(importances) > MAX_RENDERED_FEATURES:
            lines.append(
                f"- (+{len(importances) - MAX_RENDERED_FEATURES} further features)"
            )
        lines.append(
            "Importance describes model behaviour and association, not "
            "causation."
        )
    if explainability.warnings:
        lines.append("Explanation warnings:")
        lines.extend(f"- {warning}" for warning in explainability.warnings)
    return lines


def _environment_section(run: Any) -> list[str]:
    """What a reproduction attempt would need."""
    environment = run.environment
    lines = [
        f"Python version: {environment.python_version}",
        f"Platform: {environment.platform}",
        f"Random state: {environment.random_state}",
    ]
    if environment.packages:
        lines.append("Package versions:")
        lines.extend(
            f"- {name}: {version}"
            for name, version in sorted(environment.packages.items())
        )
    return lines


#: The document's sections, in order. Each is a heading and a renderer.
SECTION_RENDERERS: tuple[tuple[str, Any], ...] = (
    ("Overview", _overview_section),
    ("Dataset", _dataset_section),
    ("Preprocessing", _preprocessing_section),
    ("Model selection", _selection_section),
    ("Final evaluation", _evaluation_section),
    ("Explainability", _explainability_section),
    ("Environment", _environment_section),
)


def render_experiment(run: Any) -> str:
    """Render one experiment record as structured Markdown.

    The headings are what the chunker splits on, so each section becomes its
    own retrievable passage with its own citation fragment.

    Args:
        run: An ``ExperimentRun``.

    Returns:
        str: The record as readable text. Facts only.
    """
    parts = [f"# Experiment {run.experiment_id}", ""]
    for heading, renderer in SECTION_RENDERERS:
        lines = renderer(run)
        if not lines:
            continue
        parts.append(f"## {heading}")
        parts.append("")
        parts.extend(lines)
        parts.append("")
    return "\n".join(parts).strip() + "\n"


def experiment_metadata(run: Any) -> dict[str, Any]:
    """Return the searchable metadata carried on every chunk of a run.

    These are the keys a caller filters on before the semantic search runs,
    which is what makes "the best classification experiment on this dataset"
    answerable without scanning everything.
    """
    return {
        "source_type": SourceType.EXPERIMENT.value,
        "experiment_id": run.experiment_id,
        "configuration_hash": run.configuration_hash,
        "dataset_fingerprint": run.dataset.fingerprint,
        "task_type": run.task_type,
        "target_column": run.dataset.target_column,
        "selected_model": run.selected_model,
        "primary_metric": run.primary_metric,
        "selection_strategy": run.selection.strategy,
        "selection_score": run.selection.selection_score,
        "test_score": run.evaluation.primary_metric_value,
        "is_unbiased": run.evaluation.is_unbiased,
        "created_at": run.created_at.isoformat(),
        "tags": list(run.tags),
        "row_count": run.dataset.row_count,
        "column_count": run.dataset.column_count,
        "train_row_count": run.preprocessing.train_row_count,
        "feature_count": run.feature_count,
        #: A count, not the sentences: enough to filter on "runs with nothing
        #: flagged" without putting a paragraph in every chunk's metadata.
        "warning_count": run.evaluation.warning_count,
    }


def experiment_to_document(run: Any) -> Document:
    """Convert one experiment record into an indexable document.

    Args:
        run: An ``ExperimentRun``.

    Returns:
        Document: Whose reference is the experiment id, so its citation is
        ``experiment:<experiment_id>`` and resolves through the existing API.
    """
    return Document(
        source_type=SourceType.EXPERIMENT.value,
        source_title=f"Experiment {run.experiment_id} — {run.name}",
        source_reference=run.experiment_id,
        content=render_experiment(run),
        metadata=experiment_metadata(run),
    )


def load_experiments(
    store: ExperimentStoreLike, config: RagConfig | None = None
) -> Iterator[Document]:
    """Yield a document for every experiment the store holds.

    A record the store cannot read is skipped with a warning rather than
    failing the whole index — one corrupt file must not make the rest of the
    history unsearchable.

    Args:
        store: Anything satisfying :class:`ExperimentStoreLike`.
        config: Unused today; accepted so the signature matches the
            documentation loader and can grow limits later.
    """
    for run in store.list():
        try:
            yield experiment_to_document(run)
        except Exception as exc:  # noqa: BLE001 - one bad record, not all
            logger.warning(
                "Skipping unreadable experiment record: %s", type(exc).__name__
            )


def documents_from_runs(runs: Iterable[Any]) -> list[Document]:
    """Convert a specific set of runs, for callers that already have them."""
    return [experiment_to_document(run) for run in runs]


__all__ = [
    "MAX_RENDERED_FEATURES",
    "ExperimentStoreLike",
    "documents_from_runs",
    "experiment_metadata",
    "experiment_to_document",
    "load_experiments",
    "render_experiment",
]
