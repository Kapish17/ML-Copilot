"""Limits and seeds for explanation runs.

SHAP is not free. Computing values over a very large frame can take longer than
training the model did, so the number of rows used is capped and the excess is
sampled — deterministically, and never silently: every result reports how many
rows it actually used and warns when sampling happened.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

#: Rows used as the background (reference) distribution an explainer compares
#: against. A few hundred is enough to estimate feature means and covariance.
DEFAULT_MAX_REFERENCE_ROWS = 200
#: Rows SHAP values are computed over when summarising a model globally.
DEFAULT_MAX_EXPLANATION_ROWS = 500
#: Times each feature is shuffled by the permutation fallback.
DEFAULT_PERMUTATION_REPEATS = 10
#: Seed for every sampling and shuffling decision in this package.
DEFAULT_RANDOM_STATE = 42


@dataclass(frozen=True)
class ExplanationConfig:
    """How much data an explanation may use, and with which seed."""

    max_reference_rows: int = DEFAULT_MAX_REFERENCE_ROWS
    max_explanation_rows: int = DEFAULT_MAX_EXPLANATION_ROWS
    permutation_repeats: int = DEFAULT_PERMUTATION_REPEATS
    random_state: int = DEFAULT_RANDOM_STATE

    def as_dict(self) -> dict[str, int]:
        """Render the configuration as plain values."""
        return {
            "max_reference_rows": self.max_reference_rows,
            "max_explanation_rows": self.max_explanation_rows,
            "permutation_repeats": self.permutation_repeats,
            "random_state": self.random_state,
        }


def limit_rows(
    frame: pd.DataFrame, limit: int, *, random_state: int
) -> tuple[pd.DataFrame, bool]:
    """Cap a frame at ``limit`` rows, sampling deterministically if needed.

    The sample keeps the original row order, so two runs with the same seed
    produce not just the same rows but the same arrangement of them.

    Args:
        frame: Rows to cap.
        limit: Largest number of rows allowed.
        random_state: Seed for the sample.

    Returns:
        tuple[pandas.DataFrame, bool]: The rows to use, and whether sampling
        was needed — which the caller must report rather than hide.
    """
    if limit <= 0 or frame.shape[0] <= limit:
        return frame, False
    sampled = frame.sample(n=limit, random_state=random_state)
    return sampled.sort_index(), True
