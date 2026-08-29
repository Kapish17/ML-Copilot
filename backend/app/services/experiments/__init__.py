"""Experiment execution and history, as application services.

``options``  the validated description of one experiment request
``runner``   running the pipeline end to end and recording the result
``history``  reading, filtering and comparing what has been stored

Nothing in this package imports FastAPI. The API layer is an adapter around
these services, not the other way round, which is what lets the same
:class:`~app.services.experiments.runner.ExperimentRunner` be driven by an HTTP
route today and by a background worker or an agent tool later. **No agent, LLM
or background execution is implemented.**
"""

from app.services.experiments.history import ExperimentHistoryService
from app.services.experiments.options import ExperimentOptions
from app.services.experiments.runner import (
    ExperimentRunner,
    ExperimentRunResult,
    run_experiment,
)

__all__ = [
    "ExperimentHistoryService",
    "ExperimentOptions",
    "ExperimentRunResult",
    "ExperimentRunner",
    "run_experiment",
]
