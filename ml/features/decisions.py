"""Per-column decisions, so feature selection is never a black box.

Every column of the dataset ends up with exactly one decision record saying
what happened to it and why. Nothing is dropped silently: a column excluded
because it looks like an identifier carries that reason all the way through to
the result object.
"""

from __future__ import annotations

from dataclasses import dataclass

from ml.features.types import ColumnRole, ExclusionReason, FeatureType


@dataclass(frozen=True)
class ColumnDecision:
    """What the configuration does with one column, and the reason for it."""

    column: str
    role: ColumnRole
    reason: str
    feature_type: FeatureType | None = None
    reason_code: ExclusionReason | None = None

    @property
    def is_feature(self) -> bool:
        """True when the column is used as a model feature."""
        return self.role is ColumnRole.FEATURE

    def as_dict(self) -> dict[str, str | None]:
        """Render the decision as plain, JSON-friendly values."""
        return {
            "column": self.column,
            "role": self.role.value,
            "feature_type": self.feature_type.value if self.feature_type else None,
            "reason": self.reason,
            "reason_code": self.reason_code.value if self.reason_code else None,
        }
