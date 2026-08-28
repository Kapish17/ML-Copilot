"""Experiment tracking: remembering what was run, and what came of it.

Without this layer every result vanishes when the process exits. With it, each
complete pass through the pipeline leaves a record — which data, prepared how,
which models were considered, which won, how it scored on data it had never
seen, and what the explanations said.

``fingerprint``   identifying a dataset by its content, not its filename
``identity``      configuration hashes and safe experiment identifiers
``serialization`` explicit JSON conversion, refusing model artefacts
``run``           the versioned ``ExperimentRun`` record
``store``         the storage contract and how history is queried
``local_store``   JSON files on disk, written atomically
``builder``       composing a record from results the pipeline already made
``comparison``    ranking past runs that are actually comparable

Storage is behind an interface on purpose. The only implementation today writes
JSON to a directory; a database or a tracking service can replace it without
touching anything above. **MLflow, PostgreSQL and any vector database are not
implemented.**
"""

from ml.experiments.builder import create_experiment_run
from ml.experiments.comparison import (
    ComparisonRow,
    ExperimentComparison,
    compare_experiments,
)
from ml.experiments.fingerprint import DatasetFingerprint, fingerprint_dataset
from ml.experiments.identity import (
    configuration_hash,
    generate_experiment_id,
    validate_experiment_id,
)
from ml.experiments.local_store import LocalExperimentStore
from ml.experiments.run import (
    EXPERIMENT_SCHEMA_VERSION,
    DatasetSection,
    EnvironmentSection,
    EvaluationSection,
    ExperimentRun,
    ExplainabilitySection,
    PreprocessingSection,
    SelectionSection,
)
from ml.experiments.store import (
    ExperimentQuery,
    ExperimentSortKey,
    ExperimentStore,
)

__all__ = [
    "EXPERIMENT_SCHEMA_VERSION",
    "ComparisonRow",
    "DatasetFingerprint",
    "DatasetSection",
    "EnvironmentSection",
    "EvaluationSection",
    "ExperimentComparison",
    "ExperimentQuery",
    "ExperimentRun",
    "ExperimentSortKey",
    "ExperimentStore",
    "ExplainabilitySection",
    "LocalExperimentStore",
    "PreprocessingSection",
    "SelectionSection",
    "compare_experiments",
    "configuration_hash",
    "create_experiment_run",
    "fingerprint_dataset",
    "generate_experiment_id",
    "validate_experiment_id",
]
