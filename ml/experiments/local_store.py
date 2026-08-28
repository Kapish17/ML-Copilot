"""Experiment history on the local filesystem.

One directory per run, one readable JSON file inside it:

    runs/
      exp_4f2a91c0d8e3_20260826T134500Z_9f3a/
        experiment.json

Plain JSON is a deliberate intermediate step, not a final answer. It has no
server to run, no schema migration to manage and no dependency to install, and
a record can be read with an editor when something looks wrong. When the
project outgrows it, the replacement implements :class:`ExperimentStore` and
nothing above it changes. **MLflow and PostgreSQL are not implemented.**

Two failure modes get explicit attention.

**A half-written file.** A process killed mid-write must not leave a record
that parses as neither the old nor the new version. Writes go to a temporary
file in the same directory, are flushed to disk, and are then moved into place
with an atomic rename — so a reader sees either the previous record or the
complete new one, never a fragment.

**A corrupted record.** ``get`` raises: a caller who names a run deserves to
know it is broken. ``list`` skips it with a warning, so one bad file cannot
hide an entire history, and :meth:`LocalExperimentStore.verify` reports exactly
which records are unreadable and why.
"""

from __future__ import annotations

import logging
import os
import secrets
import shutil
from pathlib import Path

from ml.errors import ExperimentError, ExperimentNotFoundError, InvalidExperimentIdError
from ml.experiments.identity import validate_experiment_id
from ml.experiments.run import ExperimentRun
from ml.experiments.serialization import json_dumps, json_loads
from ml.experiments.store import ExperimentQuery, apply_query

logger = logging.getLogger(__name__)

#: Where runs are kept when the caller does not choose. Relative to this
#: package, so it does not depend on the working directory.
DEFAULT_RUNS_DIRECTORY = Path(__file__).resolve().parent / "runs"
#: The single file each run directory contains.
RUN_FILENAME = "experiment.json"


class LocalExperimentStore:
    """An :class:`~ml.experiments.store.ExperimentStore` backed by JSON files."""

    def __init__(self, root: Path | str | None = None) -> None:
        """Point the store at a directory.

        The directory is created on first write, so constructing a store never
        has a side effect.

        Args:
            root: Where runs live. Defaults to ``ml/experiments/runs``.
        """
        self._root = Path(root) if root is not None else DEFAULT_RUNS_DIRECTORY

    @property
    def root(self) -> Path:
        """The directory this store reads and writes."""
        return self._root

    def directory_for(self, experiment_id: str) -> Path:
        """Return the directory holding one run, refusing unsafe identifiers.

        The identifier is validated first — it may hold no separators and no
        dot segments — and the resolved path is then checked to be inside the
        store root. Either check alone would do; both are cheap, and this is
        the one place where an outside string becomes a filesystem path.

        Args:
            experiment_id: The run's identifier.

        Returns:
            pathlib.Path: The run's directory, which may not exist yet.

        Raises:
            InvalidExperimentIdError: If the identifier is malformed, or would
                resolve outside the store root.
        """
        validate_experiment_id(experiment_id)
        root = self._root.resolve()
        candidate = (root / experiment_id).resolve()
        if candidate != root and root not in candidate.parents:
            raise InvalidExperimentIdError(
                f"Experiment id '{experiment_id}' resolves outside the store "
                "directory.",
                details={"experiment_id": experiment_id},
            )
        return self._root / experiment_id

    def path_for(self, experiment_id: str) -> Path:
        """Return the JSON file holding one run."""
        return self.directory_for(experiment_id) / RUN_FILENAME

    def save(self, run: ExperimentRun) -> str:
        """Persist a run, replacing any record already under its identifier.

        Args:
            run: The run to store.

        Returns:
            str: The run's identifier.

        Raises:
            InvalidExperimentIdError: If the run's identifier is unsafe.
            SerializationError: If the run cannot be written as valid JSON.
        """
        directory = self.directory_for(run.experiment_id)
        # Serialise before touching the filesystem: a record that cannot be
        # written leaves nothing behind, not even an empty directory.
        payload = json_dumps(run.to_dict())
        directory.mkdir(parents=True, exist_ok=True)

        target = directory / RUN_FILENAME
        temporary = directory / f".{RUN_FILENAME}.{secrets.token_hex(4)}.tmp"
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return run.experiment_id

    def exists(self, experiment_id: str) -> bool:
        """Return whether a readable file is stored under this identifier."""
        try:
            return self.path_for(experiment_id).is_file()
        except InvalidExperimentIdError:
            return False

    def get(self, experiment_id: str) -> ExperimentRun:
        """Load one run.

        Args:
            experiment_id: The run's identifier.

        Returns:
            ExperimentRun: The stored run.

        Raises:
            InvalidExperimentIdError: If the identifier is unsafe.
            ExperimentNotFoundError: If nothing is stored under it.
            MalformedExperimentError: If the file is not valid JSON.
            UnsupportedSchemaVersionError: If it was written under a schema
                version this code cannot read.
            InvalidExperimentRecordError: If a required field is missing.
        """
        path = self.path_for(experiment_id)
        if not path.is_file():
            raise ExperimentNotFoundError(
                f"No experiment is stored under '{experiment_id}'.",
                details={"experiment_id": experiment_id, "root": str(self._root)},
            )
        return ExperimentRun.from_dict(json_loads(path.read_text(encoding="utf-8")))

    def _stored_ids(self) -> list[str]:
        """Return the identifiers of every directory holding a run file."""
        if not self._root.is_dir():
            return []
        return sorted(
            entry.name
            for entry in self._root.iterdir()
            if entry.is_dir() and (entry / RUN_FILENAME).is_file()
        )

    def list(self, query: ExperimentQuery | None = None) -> tuple[ExperimentRun, ...]:
        """Return the stored runs matching a query.

        Unreadable records are skipped with a warning rather than failing the
        whole listing; :meth:`verify` reports them. For this local backend
        every matching record is read from disk, which is fine for the
        directory sizes a project accumulates by hand.
        """
        runs: list[ExperimentRun] = []
        for experiment_id in self._stored_ids():
            try:
                runs.append(self.get(experiment_id))
            except ExperimentError as exc:
                logger.warning(
                    "Skipping unreadable experiment %s: %s", experiment_id, exc
                )
        return apply_query(runs, query)

    def verify(self) -> tuple[tuple[str, str], ...]:
        """Report every stored record that cannot be read, and why.

        Returns:
            tuple: ``(experiment_id, problem)`` pairs; empty when the store is
            healthy.
        """
        problems: list[tuple[str, str]] = []
        for experiment_id in self._stored_ids():
            try:
                self.get(experiment_id)
            except ExperimentError as exc:
                problems.append((experiment_id, str(exc)))
        return tuple(problems)

    def delete(self, experiment_id: str) -> bool:
        """Remove a run and its directory.

        Args:
            experiment_id: The run's identifier.

        Returns:
            bool: True when something was removed.

        Raises:
            InvalidExperimentIdError: If the identifier is unsafe.
        """
        directory = self.directory_for(experiment_id)
        if not directory.is_dir():
            return False
        shutil.rmtree(directory)
        return True
