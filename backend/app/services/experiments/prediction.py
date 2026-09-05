"""Predicting from a stored model.

The service the two prediction routes are adapters around. It owns the order of
operations and nothing else — validation lives in :mod:`ml.artifacts.prediction`
and loading in :mod:`ml.artifacts.store` — but the order is the point:

    experiment exists?  ->  model usable?  ->  records valid?  ->  predict

Each step answers a different question and fails differently, which is what
lets a caller act on the answer:

===========================  ======  =========================================
Situation                    Status  What the person does about it
===========================  ======  =========================================
No such experiment           404     Check the id; the run is gone.
Run exists, no artifact      409     Re-run the experiment to create a model.
Run exists, artifact broken  500     Nothing — it is this service's own file.
Records do not match         422     Fix the record and send it again.
===========================  ======  =========================================

The middle two are the pair worth being careful about. Both mean "there is no
model to use", and collapsing them would be a mistake: the first is normal and
expected for every run recorded before model persistence existed, and the
second is a fault in something this service wrote. Only after all of them is
the submitted data looked at.

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
from ml.artifacts.store import (
    REASON_MANIFEST_INVALID,
    REASON_MANIFEST_UNREADABLE,
    REASON_MODEL_FILE_MISSING,
    REASON_MODEL_FILE_TRUNCATED,
    REASON_MODEL_TOO_LARGE,
    REASON_NO_ARTIFACT,
    REASON_UNSUPPORTED_SCHEMA_VERSION,
)
from ml.errors import (
    MLError,
    ModelArtifactNotFoundError,
    ModelArtifactUnreadableError,
)

logger = logging.getLogger(__name__)

#: A sentence per stable reason code from the artifact store.
#:
#: The store decides *what* is wrong and says so in a code a client can branch
#: on; the sentence lives here, in the layer that knows it is talking to a
#: person. Each one ends with what to do about it, because the difference
#: between these states is entirely a difference in the fix: a run from before
#: persistence existed needs re-running, a manifest from a newer ML Copilot
#: needs an upgrade, and a half-deleted directory needs neither guessed at.
#:
#: **None of them names a file, a directory or an exception type.** What went
#: wrong with the stored bytes is the operator's problem, and the log line the
#: store wrote is where it belongs.
_REASONS: dict[str, str] = {
    REASON_NO_ARTIFACT: (
        "This experiment has no stored model. Runs recorded before model "
        "persistence was added, and runs whose artifact has been removed, "
        "cannot be predicted from — re-run the experiment to create one."
    ),
    REASON_MANIFEST_UNREADABLE: (
        "This experiment's stored model is damaged and cannot be used. "
        "Re-run the experiment to create a new one."
    ),
    REASON_MANIFEST_INVALID: (
        "This experiment's stored model is damaged and cannot be used. "
        "Re-run the experiment to create a new one."
    ),
    REASON_UNSUPPORTED_SCHEMA_VERSION: (
        "This model was written by a newer version of ML Copilot than the one "
        "running here, so it cannot be read. Upgrade the service, or re-run "
        "the experiment on this one."
    ),
    REASON_MODEL_FILE_MISSING: (
        "This experiment's model file is missing, although its description is "
        "still stored. Re-run the experiment to create a new one."
    ),
    REASON_MODEL_FILE_TRUNCATED: (
        "This experiment's model file does not match the size recorded when it "
        "was written, so it is not safe to use. Re-run the experiment to "
        "create a new one."
    ),
    REASON_MODEL_TOO_LARGE: (
        "This experiment's model file is larger than this service will load."
    ),
}

#: What to say when the store reports a code this service has never heard of —
#: a state added later and not mapped here. Better a true, vague sentence than
#: a `KeyError` in a status endpoint.
_UNKNOWN_REASON = (
    "This experiment's stored model cannot be used. Re-run the experiment to "
    "create a new one."
)


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
        notes that a model was written when it finished; whether one is usable
        *now* is a different question, and an artifact can be deleted, a volume
        wiped, or a file left half-written by a full disk. So this asks the
        store, and a client that builds a form from the answer is building it
        from something that exists.

        The answer comes from a **single** :meth:`~ml.artifacts.store.
        LocalModelArtifactStore.status` call, which both decides the state and
        hands back the manifest it read on the way. Two calls — "does it
        exist?" then "describe it" — could disagree if the directory changed
        between them, and would read the manifest twice to find out.

        Three states, and the difference between the last two is what a person
        does next:

        ``available``
            There is a model, and the response carries the schema a prediction
            must satisfy.
        ``not_available``
            There is no artifact. Normal for a run recorded before persistence
            existed. The fix is to re-run the experiment.
        ``corrupted``
            There is an artifact and it does not check out. The fix depends on
            *why*, which is what ``reason_code`` is for.

        None of the three is an error: "this run cannot be predicted from" is
        an answer, and a client asking the question needs it as a 200 so it can
        render the right thing. Only an unknown *experiment* is a 404.

        Args:
            experiment_id: The run to ask about.

        Returns:
            dict: The state, a stable reason code when there is one, and — when
            the manifest could be read — the model's description and the schema
            a prediction must satisfy. **No filesystem location, in any case.**

        Raises:
            ExperimentNotFoundError: If no such experiment is stored.
            InvalidExperimentIdError: If the identifier is malformed.
        """
        record = self._history.get(experiment_id)
        status = self._artifacts.status(record.experiment_id)

        summary: dict[str, Any] = {
            "experiment_id": record.experiment_id,
            "status": status.state,
            "available": status.is_available,
            "reason_code": status.reason_code,
            "reason": (
                None
                if status.is_available
                else _REASONS.get(status.reason_code or "", _UNKNOWN_REASON)
            ),
            "max_records": self._settings.max_prediction_records,
        }

        # The manifest is described only when the artifact is usable. A
        # corrupted one may well have a readable manifest — the model file
        # beside it is what failed — but publishing a feature schema for a
        # model that cannot answer would invite a client to build a form whose
        # every submission fails.
        if status.is_available and status.metadata is not None:
            summary.update(status.metadata.public_summary())
        return summary

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

        # The same status check the model endpoint answers from, so the two
        # cannot disagree about whether a prediction is possible. It also
        # means a truncated or unreadable artifact is refused **before**
        # anything is deserialised: unpickling a file already known to be
        # wrong is work done on behalf of a request that cannot succeed.
        status = self._artifacts.status(record.experiment_id)
        if not status.is_available:
            raise self._unusable(record.experiment_id, status.reason_code)

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

    @staticmethod
    def _unusable(experiment_id: str, reason_code: str | None) -> MLError:
        """The right exception for an artifact that cannot be predicted from.

        The split is between *whose problem it is*, and it is why these two do
        not share a status code:

        **No artifact** is the caller's to act on and nobody's fault — the run
        predates persistence, or its model was deleted. It is a **409**, whose
        message says to re-run the experiment.

        **A corrupted artifact** is this service's own broken file. Nothing the
        caller changes will help, and an operator should see it in the log, so
        it is a **500** — answered with a generic message and no details, like
        every other 5xx here. The specific reason went to the log and to the
        model endpoint, which reports it as a state rather than as a failure.
        """
        if reason_code in (None, REASON_NO_ARTIFACT):
            return ModelArtifactNotFoundError(
                f"No model is stored for experiment '{experiment_id}'.",
                details={"experiment_id": experiment_id},
            )
        logger.warning(
            "Refusing to predict from the artifact for %s: %s",
            experiment_id,
            reason_code,
        )
        return ModelArtifactUnreadableError(
            "The stored model for this experiment could not be used.",
            details={"experiment_id": experiment_id, "reason_code": reason_code},
        )


__all__ = ["PredictionService"]
