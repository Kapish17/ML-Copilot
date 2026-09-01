"""The dataset one request lends the agent, and the reasons it is lent rather than kept.

**Uploaded datasets are processed in memory for the request and are never
persisted as raw data by the agent.** Everything in this module exists to make
that sentence true rather than aspirational.

An agent that can profile and train needs data, and the only way data reaches
this system is an upload. That is a genuinely awkward combination: a
long-running orchestration and a file that must not outlive the request. So the
dataset is modelled as a *loan*. It arrives, is validated and parsed by the
ingestion path that already exists, is registered under a fixed name for the
length of one call, and the reference is dropped when the call returns.

Three properties follow, and each is worth stating because each is a thing that
could easily have gone the other way.

**The name is a constant, not the filename.** The agent addresses the dataset
as ``uploaded_dataset``, always. A client's filename never becomes an
identifier the planner can name, which is what makes ``../../secret.csv``,
``C:\\secret.csv`` and ``/etc/passwd`` uninteresting: they are not names the
tool's schema accepts, and they are not names anything looks up. The
sanitised filename survives only as display text on the response.

**Identity is the content fingerprint.** Commit 7 already identifies a dataset
by what is in it. Two uploads of the same data are the same dataset however
they were named, and an experiment record points at a fingerprint rather than
at a file that no longer exists.

**Nothing is written.** The bytes are parsed in memory and never touch a disk.
The experiment store keeps a record — fingerprint, shape, decisions, scores —
and no rows. The retrieval index is not told the upload happened.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent.tools.datasets import InMemoryDatasetSource
from app.services.datasets import DatasetProfilingService
from app.services.datasets.validation import AsyncReadable
from ml.experiments.fingerprint import fingerprint_dataset

#: The one name the agent may address an uploaded dataset by.
#:
#: A constant on purpose. If this were the filename, a client would choose a
#: string that reaches the tool schema, the planner prompt and the experiment
#: record — and the interesting question would become "what does the system do
#: with a name like ``../../etc/passwd``". With a constant there is no such
#: question: that string is not a registered name, so it is refused exactly as
#: a typo would be, and nothing anywhere resolves a name to a location.
UPLOADED_DATASET_NAME = "uploaded_dataset"


@dataclass(frozen=True)
class RequestDataset:
    """One dataset, on loan to one request.

    Holds the parsed frame and the few facts about it that are safe to report:
    its content fingerprint, its shape, its column names and the display
    filename. **No rows.** Everything a caller or a planner learns about the
    contents comes from the profiling tool's structured summary, never from
    this object and never from the raw data.
    """

    #: The standardised in-memory dataset. Dropped when the request ends.
    frame: Any
    #: Content fingerprint — the canonical identity, from Commit 7.
    fingerprint: str
    #: Path-free filename, for display only. Never used to reach anything.
    filename: str
    row_count: int
    column_count: int
    columns: tuple[str, ...]

    @property
    def name(self) -> str:
        """The name the agent addresses this dataset by."""
        return UPLOADED_DATASET_NAME

    def source(self) -> InMemoryDatasetSource:
        """Build the request-scoped source the tools resolve names through."""
        return InMemoryDatasetSource({UPLOADED_DATASET_NAME: self.frame})

    def planner_context(self) -> dict[str, Any]:
        """What the planner is told about the dataset.

        Deliberately four facts and no data. The planner needs to know that a
        dataset exists and what to call it; the shape helps it judge whether an
        experiment is worth running. What it must not get is content — a cell
        that reads "ignore previous instructions" has to arrive as a profiled
        value inside a tool observation, where it is already treated as
        untrusted, and never as prompt text.
        """
        return {
            "dataset_available": True,
            "dataset_name": self.name,
            "dataset_rows": self.row_count,
            "dataset_columns": self.column_count,
        }

    def as_dict(self) -> dict[str, Any]:
        """Render the safe facts about the dataset for a response."""
        return {
            "name": self.name,
            "filename": self.filename,
            "fingerprint": self.fingerprint,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "columns": list(self.columns),
            "persisted": False,
        }


async def load_request_dataset(
    service: DatasetProfilingService, upload: AsyncReadable, filename: str | None
) -> RequestDataset:
    """Turn an upload into a dataset the agent may use for one request.

    The whole of the ingestion is the dataset service's: the extension is
    checked before any bytes are read, the size is bounded by the configured
    limit, the filename is reduced to a bare name, and the content is parsed
    into a standardised frame. Nothing here re-implements any of that, and no
    second set of limits is introduced.

    Args:
        service: The existing profiling service, which owns ingestion.
        upload: The incoming file.
        filename: What the client called it. Used for its extension, and kept
            as display text — never as a location.

    Returns:
        RequestDataset: The frame and the safe facts about it.

    Raises:
        DatasetError: If the upload or its content fails validation — an
            unsupported extension, an oversized file, malformed CSV, or a
            dataset with no usable rows or columns.
    """
    loaded = await service.load_upload(upload, filename)
    frame = loaded.frame
    rows, columns = frame.shape

    return RequestDataset(
        frame=frame,
        fingerprint=fingerprint_dataset(frame).value,
        filename=loaded.filename,
        row_count=int(rows),
        column_count=int(columns),
        columns=tuple(str(name) for name in frame.columns),
    )


__all__ = ["UPLOADED_DATASET_NAME", "RequestDataset", "load_request_dataset"]
