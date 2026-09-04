"""Predicting from a stored model.

The service the two prediction routes are adapters around. It owns the order of
operations and nothing else — validation lives in :mod:`ml.artifacts.prediction`
and loading in :mod:`ml.artifacts.store` — but the order is the point:

    experiment exists?  ->  model stored?  ->  records valid?  ->  predict

Each step answers a different question and fails differently, which is what
lets a caller act on the answer. An unknown id is a 404 and means the run is
gone. A known id with no artifact is a **409** and means the run is fine but
predates model persistence, or its artifact was removed — a completely
different situation with a completely different fix. Only after both is the
submitted data looked at.

**No path, filename or location ever comes from the request.** A route hands
this service an experiment id and a list of records. The id is validated by
the experiment store before it addresses a record and again by the artifact
store before it addresses a directory; the records never touch the filesystem
at all. There is no argument anywhere in this module that could name a file.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Mapping

from app.core.config import Settings
from app.services.experiments.history import ExperimentHistoryService
from ml.artifacts import ModelArtifactStore, build_frame, predict
from ml.artifacts.schema import ModelArtifactMetadata

logger = logging.getLogger(__name__)


class PredictionService:
    """Answer prediction requests from models this application persisted."""

    def __init__(
        self,
        settings: Settings,
        history: ExperimentHistoryService,
        artifacts: ModelArtifactStore,
    ) -> None:
        """Wire the service to its collaborators.

        Args:
            settings: Active settings, supplying the batch ceiling.
            history: Used to confirm the experiment exists before its model is
                looked for, so "no such run" and "no model for this run" stay
                distinguishable.
            artifacts: Where persisted models live.
        """
        self._settings = settings
        self._history = history
        self._artifacts = artifacts

    def describe(self, experiment_id: str) -> dict[str, Any]:
        """Report whether this experiment can be predicted from, and with what.

        **The store is the authority, not the record.** A run's stored record
        notes that a model was written when it finished; whether one is there
        *now* is a different question, and an artifact can be deleted or a
        volume wiped. So this asks the store, and a client that builds a form
        from the answer is building it from something that exists.

        Reading the manifest costs a small JSON read — the model itself is not
        deserialised, so asking the question is cheap enough to do on every
        page load.

        Args:
            experiment_id: The run to ask about.

        Returns:
            dict: ``available`` plus, when there is a model, the schema a
            prediction must satisfy. No filesystem location, in either case.

        Raises:
            ExperimentNotFoundError: If no such experiment is stored.
            InvalidExperimentIdError: If the identifier is malformed.
        """
        record = self._history.get(experiment_id)

        if not self._artifacts.exists(experiment_id):
            return {
                "experiment_id": record.experiment_id,
                "available": False,
                "reason": (
                    "This experiment has no stored model. Runs recorded before "
                    "model persistence was added, and runs whose artifact has "
                    "been removed, cannot be predicted from — re-run the "
                    "experiment to create one."
                ),
                "max_records": self._settings.max_prediction_records,
            }

        metadata = self._artifacts.metadata_for(experiment_id)
        return {
            "experiment_id": record.experiment_id,
            "available": True,
            "reason": None,
            "max_records": self._settings.max_prediction_records,
            **metadata.public_summary(),
        }

    def predict(
        self, experiment_id: str, records: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        """Predict for one or more records from a stored model.

        The pipeline is used exactly as it was loaded. **Its preprocessing was
        fitted once, during the experiment, on the training rows alone** — it
        is applied here and refitted nowhere, which is what makes a prediction
        made today comparable to the score that run reported.

        Args:
            experiment_id: Which run's model to use.
            records: One mapping of feature name to value per row.

        Returns:
            dict: One result per record, plus a description of the model.

        Raises:
            ExperimentNotFoundError: If no such experiment is stored.
            ModelArtifactNotFoundError: If it has no stored model.
            ModelArtifactUnreadableError: If the artifact fails its checks.
            PredictionInputError: If the records do not match the schema.
        """
        record = self._history.get(experiment_id)
        model = self._artifacts.load(record.experiment_id)

        frame = build_frame(
            records,
            model.metadata,
            max_records=self._settings.max_prediction_records,
        )
        result = predict(model, frame)

        # Counts and names only. **No feature value and no prediction** — the
        # submitted records are somebody's data, exactly like an uploaded
        # dataset, and the reasons that keep dataset rows out of the log keep
        # these out too.
        logger.info(
            "Predicted %d record(s) from %s using %s",
            len(result.predictions),
            record.experiment_id,
            model.metadata.model_name,
        )
        return result.as_dict()


__all__ = ["PredictionService"]
