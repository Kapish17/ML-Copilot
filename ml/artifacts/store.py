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
from typing import Any, Protocol, runtime_checkable

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

    def exists(self, experiment_id: str) -> bool:
        """Whether both halves of a usable artifact are present.

        An unsafe identifier is not an error here — it is simply an experiment
        with no model, which is what a caller asking "can I predict?" needs to
        know.
        """
        try:
            directory = self.directory_for(experiment_id)
        except Exception:  # noqa: BLE001 - an unusable id has no artifact
            return False
        return (directory / MANIFEST_FILENAME).is_file() and (
            directory / MODEL_FILENAME
        ).is_file()

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
    "LoadedModel",
    "LocalModelArtifactStore",
    "ModelArtifactStore",
]
