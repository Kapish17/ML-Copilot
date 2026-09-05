"""Where a trained model is kept, and the rules for getting it back.

One directory per experiment under a single root this application owns::

    <root>/<experiment_id>/model.joblib     the fitted Pipeline
    <root>/<experiment_id>/artifact.json    the manifest

---------------------------------------------------------------------------
The trust boundary
---------------------------------------------------------------------------
``joblib`` is pickle underneath, and **unpickling executes code in the file**.
That is not a flaw to work around here; it is the reason this module is written
the way it is. The rule is simple and absolute:

    **This store loads only files this application wrote, in the one directory
    it owns, addressed by a validated experiment id.**

Four things enforce it, and each is a separate barrier:

1. **No path ever comes from a request.** The public API takes an experiment
   id. There is no method that takes a path, a filename, a directory or a URL,
   so there is nothing for a caller to point somewhere else.
2. **The id is validated before it becomes a path.** `validate_experiment_id`
   admits letters, digits, underscores and hyphens only — no separator, no dot
   segment, no drive letter — so `../../etc/passwd` is refused as an id rather
   than sanitised into something that looks acceptable.
3. **The resolved path is re-checked.** After resolution it must still be
   inside the root. Either check alone would do; a symlink planted in the root
   is the case that needs the second.
4. **The filename is a constant.** The manifest records the model file's name
   for a human reading the directory, and this module never reads it back:
   opening a name that came out of a file would put the choice of what to
   deserialise back into data.

**What this does not defend against.** Anyone who can write into the artifact
root can replace a model with a malicious pickle, and can update the manifest's
digest to match. The digest catches corruption and accidental substitution, not
an attacker who already has write access to the volume — at which point they
can also replace the application's own code. The boundary is the directory: it
must be treated as executable code, and nothing but this application may write
to it. **Third-party model files are not supported** and there is no endpoint,
argument or configuration that would load one.

---------------------------------------------------------------------------
Three states, and one function that decides between them
---------------------------------------------------------------------------
An artifact is `available`, `not_available`, or `corrupted`, and
:meth:`LocalModelArtifactStore.status` is the only code that decides which.
Everything else — :meth:`exists`, the model endpoint, the reason a dashboard
shows — reads that one answer, so "can this be predicted from?" cannot be
answered two different ways by two callers.

The check is deliberately **cheap and shallow**: the manifest is parsed and
validated, and the model file is checked for presence and for the size the
manifest recorded. Nothing is unpickled, so asking costs a small JSON read and
a ``stat`` and can be done on every page load. The **deep** checks — the
SHA-256 digest, and that the object really is a ``Pipeline`` — need the file
itself and therefore happen in :meth:`load`. An artifact that passes the
shallow check and fails the deep one is possible, rare, and reported honestly
by the endpoint that meets it rather than papered over by a status call that
pretends to have looked.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

from sklearn.pipeline import Pipeline

from ml.artifacts.schema import ModelArtifactMetadata
from ml.errors import ModelArtifactNotFoundError, ModelArtifactUnreadableError
from ml.experiments.identity import validate_experiment_id

logger = logging.getLogger(__name__)

#: The two filenames in an artifact directory. **Constants, never read from a
#: manifest**: the name of the file to deserialise is a decision this module
#: makes, not data it accepts.
MODEL_FILENAME = "model.joblib"
MANIFEST_FILENAME = "artifact.json"

#: Refuse to load a model file larger than this. A guard against a corrupted or
#: substituted file exhausting memory before anything has had a chance to check
#: it. Generous: a six-model forest on a wide dataset is a few tens of MB.
MAX_MODEL_BYTES = 512 * 1024 * 1024


#: The three states an experiment's model can be in, as far as predicting from
#: it is concerned. Stable strings: they are part of the API contract and a
#: client branches on them.
STATE_AVAILABLE = "available"
STATE_NOT_AVAILABLE = "not_available"
STATE_CORRUPTED = "corrupted"

#: Why an artifact is not usable. Stable codes, so a client can distinguish
#: "this run never had a model" from "this run's model is broken" without
#: reading a sentence written for a human.
REASON_NO_ARTIFACT = "no_artifact"
REASON_MANIFEST_UNREADABLE = "manifest_unreadable"
REASON_MANIFEST_INVALID = "manifest_invalid"
REASON_UNSUPPORTED_SCHEMA_VERSION = "unsupported_schema_version"
REASON_MODEL_FILE_MISSING = "model_file_missing"
REASON_MODEL_FILE_TRUNCATED = "model_file_truncated"
REASON_MODEL_TOO_LARGE = "model_too_large"


@dataclass(frozen=True)
class ArtifactStatus:
    """Whether one experiment's model can be predicted from, and why not.

    The metadata is carried along when there is any, so a caller that asks the
    status and then wants the feature schema does not read the manifest twice.
    """

    experiment_id: str
    #: One of :data:`STATE_AVAILABLE`, :data:`STATE_NOT_AVAILABLE`,
    #: :data:`STATE_CORRUPTED`.
    state: str
    #: A stable ``REASON_*`` code when the state is not ``available``.
    reason_code: str | None = None
    #: The manifest, when it could be read. Present for ``available``, and for
    #: a ``corrupted`` artifact whose manifest parsed but whose model file did
    #: not check out — the schema is still true even when the model is gone.
    metadata: ModelArtifactMetadata | None = None

    @property
    def is_available(self) -> bool:
        """Whether a prediction could be attempted."""
        return self.state == STATE_AVAILABLE


@dataclass(frozen=True)
class LoadedModel:
    """A fitted pipeline and the manifest that describes it."""

    #: The **exact** pipeline the experiment fitted: the preprocessing that was
    #: fitted on the training rows, with the winning estimator behind it. It is
    #: used as loaded and is never refitted.
    pipeline: Pipeline
    metadata: ModelArtifactMetadata


@runtime_checkable
class ModelArtifactStore(Protocol):
    """Where fitted models live.

    A protocol so the storage decision stays replaceable, in the same way
    `ExperimentStore` does for run records. **Only the local implementation
    below exists**; no model registry, object store or database is involved.
    """

    def save(
        self, experiment_id: str, pipeline: Pipeline, metadata: ModelArtifactMetadata
    ) -> ModelArtifactMetadata:
        """Persist one fitted pipeline and its manifest."""

    def status(self, experiment_id: str) -> ArtifactStatus:
        """Report whether this experiment's model is usable, and why not."""

    def exists(self, experiment_id: str) -> bool:
        """Whether a usable artifact is stored for this experiment."""

    def load(self, experiment_id: str) -> LoadedModel:
        """Return the fitted pipeline and its manifest."""

    def metadata_for(self, experiment_id: str) -> ModelArtifactMetadata:
        """Return the manifest without loading the model."""

    def delete(self, experiment_id: str) -> bool:
        """Remove an artifact. Returns whether anything was there."""


class LocalModelArtifactStore:
    """The filesystem implementation of :class:`ModelArtifactStore`."""

    def __init__(self, root: Path | str) -> None:
        """Point the store at a directory.

        The directory is created on the first write, so constructing a store
        has no side effect and an application starts fine with no models.
        """
        self._root = Path(root)

    @property
    def root(self) -> Path:
        """The directory this store owns."""
        return self._root

    # -- Addressing --------------------------------------------------------

    def directory_for(self, experiment_id: str) -> Path:
        """Return one experiment's artifact directory, refusing unsafe ids.

        The identifier is validated first — it may hold no separators and no
        dot segments — and the resolved path is then checked to be inside the
        store root. This is the only place an outside string becomes a path in
        this module, and nothing downstream repeats the check because nothing
        downstream builds a path.

        Args:
            experiment_id: The run's identifier.

        Returns:
            pathlib.Path: The directory, which may not exist yet.

        Raises:
            InvalidExperimentIdError: If the identifier is malformed, or would
                resolve outside the store root.
        """
        validate_experiment_id(experiment_id)
        root = self._root.resolve()
        candidate = (root / experiment_id).resolve()
        if candidate != root and root not in candidate.parents:
            raise ModelArtifactUnreadableError(
                f"Experiment id '{experiment_id}' resolves outside the model "
                "store directory.",
                details={"experiment_id": experiment_id},
            )
        return self._root / experiment_id

    # -- Writing -----------------------------------------------------------

    def save(
        self, experiment_id: str, pipeline: Pipeline, metadata: ModelArtifactMetadata
    ) -> ModelArtifactMetadata:
        """Persist a fitted pipeline and its manifest.

        Written model-first, then manifest, each through a temporary file and
        an atomic ``os.replace``. The order matters: the manifest is what
        :meth:`exists` looks for, so a crash between the two leaves a directory
        that reports *no* artifact rather than one that reports an artifact
        whose model is half-written.

        Args:
            experiment_id: The run this model belongs to.
            pipeline: The fitted ``Pipeline(preprocessing, estimator)``.
            metadata: The manifest describing it.

        Returns:
            ModelArtifactMetadata: The manifest as stored.

        Raises:
            ModelArtifactUnreadableError: If the model cannot be serialised.
        """
        import joblib  # Imported here so `ml.artifacts` imports without it.

        directory = self.directory_for(experiment_id)
        directory.mkdir(parents=True, exist_ok=True)

        model_path = directory / MODEL_FILENAME
        temporary = directory / f".{MODEL_FILENAME}.{secrets.token_hex(4)}.tmp"
        try:
            try:
                joblib.dump(pipeline, temporary, compress=3)
            except Exception as exc:  # noqa: BLE001 - any failure is the same
                raise ModelArtifactUnreadableError(
                    "The trained model could not be serialised.",
                    details={"reason": type(exc).__name__},
                ) from exc
            os.replace(temporary, model_path)
        finally:
            temporary.unlink(missing_ok=True)

        payload = metadata.as_dict()
        # Recorded for a human reading the directory. **Never read back to
        # decide what to open** — see the module docstring.
        payload["model_file"] = {
            "name": MODEL_FILENAME,
            "format": "joblib",
            "bytes": model_path.stat().st_size,
            "sha256": _digest_of(model_path),
        }
        _write_json(directory / MANIFEST_FILENAME, payload)

        logger.info(
            "Persisted the model for %s: %s, %d bytes",
            experiment_id,
            metadata.model_name,
            payload["model_file"]["bytes"],
        )
        return metadata

    # -- Reading -----------------------------------------------------------

    def status(self, experiment_id: str) -> ArtifactStatus:
        """Report whether this experiment's model can be predicted from.

        The one place that decides between the three states, so no caller has
        to reimplement "is there a usable model here" and no two callers can
        reach different conclusions. Nothing is unpickled: the manifest is read
        and validated, and the model file is checked for presence and for the
        size the manifest recorded. See the module docstring on why the digest
        and the type check belong to :meth:`load` instead.

        This **never raises**. A caller asking about a model wants an answer
        about the model, and "the id is malformed" is, from here, just another
        way of having no artifact — the endpoints that must reject a bad id do
        so before they get this far.

        Args:
            experiment_id: The run to ask about.

        Returns:
            ArtifactStatus: The state, a stable reason code when it is not
            ``available``, and the manifest when one could be read.
        """
        try:
            directory = self.directory_for(experiment_id)
        except Exception:  # noqa: BLE001 - an unusable id has no artifact
            return ArtifactStatus(
                experiment_id, STATE_NOT_AVAILABLE, REASON_NO_ARTIFACT
            )

        manifest = directory / MANIFEST_FILENAME
        if not manifest.is_file():
            # Nothing was ever written here, or the whole directory is gone.
            # Not a fault: it is the state every run recorded before model
            # persistence existed is in.
            return ArtifactStatus(
                experiment_id, STATE_NOT_AVAILABLE, REASON_NO_ARTIFACT
            )

        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return ArtifactStatus(
                experiment_id, STATE_CORRUPTED, REASON_MANIFEST_UNREADABLE
            )

        try:
            metadata = ModelArtifactMetadata.from_dict(payload)
        except ModelArtifactUnreadableError as exc:
            # An artifact written by a future version is a different problem
            # from one written by a broken one, and a client that wants to say
            # "upgrade" rather than "re-run" needs to tell them apart.
            unsupported = "supported" in exc.details
            return ArtifactStatus(
                experiment_id,
                STATE_CORRUPTED,
                (
                    REASON_UNSUPPORTED_SCHEMA_VERSION
                    if unsupported
                    else REASON_MANIFEST_INVALID
                ),
            )

        model_path = directory / MODEL_FILENAME
        if not model_path.is_file():
            # The manifest is written second, so this is a directory somebody
            # emptied rather than a half-finished save.
            return ArtifactStatus(
                experiment_id,
                STATE_CORRUPTED,
                REASON_MODEL_FILE_MISSING,
                metadata,
            )

        size = model_path.stat().st_size
        recorded = _recorded_size(payload)
        if recorded is not None and size != recorded:
            return ArtifactStatus(
                experiment_id,
                STATE_CORRUPTED,
                REASON_MODEL_FILE_TRUNCATED,
                metadata,
            )
        if size > MAX_MODEL_BYTES:
            return ArtifactStatus(
                experiment_id, STATE_CORRUPTED, REASON_MODEL_TOO_LARGE, metadata
            )

        return ArtifactStatus(experiment_id, STATE_AVAILABLE, None, metadata)

    def exists(self, experiment_id: str) -> bool:
        """Whether a usable artifact is stored for this experiment.

        A thin reading of :meth:`status`, so "usable" means one thing in this
        codebase. A corrupted artifact answers **False** here: a caller asking
        this question is deciding whether to offer a prediction, and an
        artifact whose manifest cannot be read is not one to offer.
        """
        return self.status(experiment_id).is_available

    def metadata_for(self, experiment_id: str) -> ModelArtifactMetadata:
        """Return the manifest without deserialising the model.

        This is what an endpoint answering "what features does this model
        want?" calls, so asking the question costs a small JSON read rather
        than unpickling a forest.

        Raises:
            ModelArtifactNotFoundError: If no artifact is stored.
            ModelArtifactUnreadableError: If the manifest cannot be read.
        """
        directory = self.directory_for(experiment_id)
        manifest = directory / MANIFEST_FILENAME
        if not manifest.is_file():
            raise ModelArtifactNotFoundError(
                f"No model is stored for experiment '{experiment_id}'.",
                details={"experiment_id": experiment_id},
            )
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ModelArtifactUnreadableError(
                "The model manifest could not be read.",
                details={"experiment_id": experiment_id, "reason": type(exc).__name__},
            ) from exc
        return ModelArtifactMetadata.from_dict(payload)

    def load(self, experiment_id: str) -> LoadedModel:
        """Return the fitted pipeline and its manifest.

        The manifest is read and validated **before** the model is
        deserialised, and the model file's digest is checked against the one
        recorded at save time. The digest is an integrity check, not an
        authenticity check: it catches a truncated write or a file swapped by
        accident, and it does not stop anyone who can write to this directory —
        see the module docstring on the trust boundary.

        Raises:
            ModelArtifactNotFoundError: If no artifact is stored.
            ModelArtifactUnreadableError: If anything about it fails to check
                out, or the pickle cannot be read by this interpreter.
        """
        import joblib

        directory = self.directory_for(experiment_id)
        metadata = self.metadata_for(experiment_id)

        # A constant, so the file opened is decided by this code and not by
        # anything the manifest happens to contain.
        model_path = directory / MODEL_FILENAME
        if not model_path.is_file():
            raise ModelArtifactNotFoundError(
                f"No model file is stored for experiment '{experiment_id}'.",
                details={"experiment_id": experiment_id},
            )

        size = model_path.stat().st_size
        if size > MAX_MODEL_BYTES:
            raise ModelArtifactUnreadableError(
                "The stored model is larger than this service will load.",
                details={"experiment_id": experiment_id, "bytes": size},
            )

        expected = _recorded_digest(directory / MANIFEST_FILENAME)
        if expected is not None and _digest_of(model_path) != expected:
            raise ModelArtifactUnreadableError(
                "The stored model does not match the digest recorded when it "
                "was written.",
                details={"experiment_id": experiment_id},
            )

        try:
            pipeline = joblib.load(model_path)
        except Exception as exc:  # noqa: BLE001 - unpickling fails many ways
            raise ModelArtifactUnreadableError(
                "The stored model could not be loaded by this service.",
                details={"experiment_id": experiment_id, "reason": type(exc).__name__},
            ) from exc

        if not isinstance(pipeline, Pipeline):
            raise ModelArtifactUnreadableError(
                "The stored artifact is not a model pipeline.",
                details={
                    "experiment_id": experiment_id,
                    "found_type": type(pipeline).__name__,
                },
            )

        return LoadedModel(pipeline=pipeline, metadata=metadata)

    # -- Removal -----------------------------------------------------------

    def delete(self, experiment_id: str) -> bool:
        """Remove one experiment's artifact directory.

        Returns:
            bool: Whether anything was removed.
        """
        directory = self.directory_for(experiment_id)
        if not directory.is_dir():
            return False
        shutil.rmtree(directory)
        return True

    def stored_ids(self) -> list[str]:
        """Every experiment id with a stored artifact, sorted."""
        if not self._root.is_dir():
            return []
        return sorted(
            entry.name
            for entry in self._root.iterdir()
            if entry.is_dir() and (entry / MANIFEST_FILENAME).is_file()
        )


def _digest_of(path: Path) -> str:
    """Return the SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _recorded_size(payload: Any) -> int | None:
    """Return the model size a manifest recorded, or ``None`` if it has none.

    The cheap half of the integrity check: comparing this against the file on
    disk costs a ``stat`` and catches the common corruption — a truncated or
    replaced file — without reading a byte of the model. The digest in
    :func:`_recorded_digest` is the thorough half, and it is only worth paying
    for at load time.
    """
    if not isinstance(payload, Mapping):
        return None
    recorded = (payload.get("model_file") or {}).get("bytes")
    return recorded if isinstance(recorded, int) and not isinstance(
        recorded, bool
    ) else None


def _recorded_digest(manifest: Path) -> str | None:
    """Return the model digest a manifest recorded, or ``None`` if it has none."""
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):  # pragma: no cover - checked by the caller
        return None
    recorded = (payload.get("model_file") or {}).get("sha256")
    return str(recorded) if isinstance(recorded, str) else None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically, so a crash never leaves half a manifest."""
    temporary = path.parent / f".{path.name}.{secrets.token_hex(4)}.tmp"
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "MANIFEST_FILENAME",
    "MAX_MODEL_BYTES",
    "MODEL_FILENAME",
    "REASON_MANIFEST_INVALID",
    "REASON_MANIFEST_UNREADABLE",
    "REASON_MODEL_FILE_MISSING",
    "REASON_MODEL_FILE_TRUNCATED",
    "REASON_MODEL_TOO_LARGE",
    "REASON_NO_ARTIFACT",
    "REASON_UNSUPPORTED_SCHEMA_VERSION",
    "STATE_AVAILABLE",
    "STATE_CORRUPTED",
    "STATE_NOT_AVAILABLE",
    "ArtifactStatus",
    "LoadedModel",
    "LocalModelArtifactStore",
    "ModelArtifactStore",
]
