"""The validated description of one experiment request.

:class:`ExperimentOptions` is the boundary between "what a caller asked for"
and "what the ML layer is told to do". It is a plain frozen dataclass with no
FastAPI and no pandas in sight, which is what lets the same runner serve an
HTTP route today and a future agent tool call tomorrow — neither needs to know
how the other phrased the request.

Validation here is limited to what can be checked without the data: shapes,
ranges and configured limits. Anything that depends on the dataset — whether a
model suits the detected task, whether a metric exists for it — is checked by
the runner once the task is known, using the ML layer's own validators rather
than a second copy of its rules.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from app.core.config import Settings
from ml.errors import ConfigurationError, InvalidFoldCountError

#: Selection strategies the API accepts, mirroring ``ml.models.comparison``.
SELECTION_STRATEGIES = ("cross_validation", "holdout")
DEFAULT_STRATEGY = "cross_validation"

#: Preprocessing fields a caller may override. Deliberately a small, explicit
#: set: these are the choices worth exposing, and an allowlist means a typo is
#: an error rather than a silently ignored field.
OVERRIDE_FIELDS = (
    "scaling_strategy",
    "numeric_imputation",
    "categorical_imputation",
    "add_missing_indicators",
    "max_categorical_cardinality",
    "test_size",
    "random_state",
)

MIN_TEST_SIZE = 0.05
MAX_TEST_SIZE = 0.5
MAX_NAME_LENGTH = 120
MAX_DESCRIPTION_LENGTH = 2_000
MAX_TAG_LENGTH = 40


def _clean(value: str | None) -> str | None:
    """Trim a string and treat a blank one as absent."""
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _clean_names(values: Sequence[str] | None) -> tuple[str, ...]:
    """Trim a sequence of names, dropping blanks and preserving order."""
    if not values:
        return ()
    seen: list[str] = []
    for value in values:
        name = str(value).strip()
        if name and name not in seen:
            seen.append(name)
    return tuple(seen)


@dataclass(frozen=True)
class ExperimentOptions:
    """Everything about an experiment except the data itself."""

    target_column: str | None = None
    models: tuple[str, ...] = ()
    primary_metric: str | None = None
    strategy: str = DEFAULT_STRATEGY
    folds: int | None = None
    test_size: float | None = None
    random_state: int | None = None
    excluded_columns: tuple[str, ...] = ()
    identifier_columns: tuple[str, ...] = ()
    scaling_strategy: str | None = None
    numeric_imputation: str | None = None
    categorical_imputation: str | None = None
    add_missing_indicators: bool | None = None
    max_categorical_cardinality: int | None = None
    explain: bool = True
    name: str | None = None
    description: str | None = None
    tags: tuple[str, ...] = field(default=())

    @property
    def preprocessing_overrides(self) -> dict[str, Any]:
        """The explicit preprocessing values, ready for ``with_overrides``.

        Only fields the caller actually set appear, so anything inferred from
        the dataset profile survives untouched.
        """
        chosen = {name: getattr(self, name) for name in OVERRIDE_FIELDS}
        return {name: value for name, value in chosen.items() if value is not None}

    def resolved_name(self, fallback: str) -> str:
        """The run's label, falling back to something recognisable."""
        return self.name or fallback

    def validated(self, settings: Settings) -> ExperimentOptions:
        """Return a normalised copy, or raise if the request cannot be run.

        Args:
            settings: Active application settings, supplying every limit.

        Returns:
            ExperimentOptions: The same request with strings trimmed and
            blanks turned into ``None``.

        Raises:
            ConfigurationError: If a value is outside its accepted range or a
                configured limit.
            InvalidFoldCountError: If the fold count is unusable.
        """
        strategy = (_clean(self.strategy) or DEFAULT_STRATEGY).lower()
        if strategy not in SELECTION_STRATEGIES:
            raise ConfigurationError(
                f"Unknown selection strategy '{strategy}'. Available: "
                + ", ".join(SELECTION_STRATEGIES)
                + ".",
                details={
                    "strategy": strategy,
                    "available_strategies": list(SELECTION_STRATEGIES),
                },
            )

        if self.folds is not None and not (
            settings.min_cv_folds <= self.folds <= settings.max_cv_folds
        ):
            raise InvalidFoldCountError(
                f"folds must be between {settings.min_cv_folds} and "
                f"{settings.max_cv_folds}, got {self.folds}.",
                details={
                    "folds": self.folds,
                    "min_folds": settings.min_cv_folds,
                    "max_folds": settings.max_cv_folds,
                },
            )

        models = _clean_names(self.models)
        if len(models) > settings.max_candidate_models:
            raise ConfigurationError(
                f"At most {settings.max_candidate_models} candidate models may "
                f"be requested, got {len(models)}.",
                details={
                    "requested": len(models),
                    "max_candidate_models": settings.max_candidate_models,
                },
            )

        if self.test_size is not None and not (
            MIN_TEST_SIZE <= self.test_size <= MAX_TEST_SIZE
        ):
            raise ConfigurationError(
                f"test_size must be between {MIN_TEST_SIZE} and {MAX_TEST_SIZE}, "
                f"got {self.test_size}.",
                details={
                    "test_size": self.test_size,
                    "minimum": MIN_TEST_SIZE,
                    "maximum": MAX_TEST_SIZE,
                },
            )

        if self.random_state is not None and self.random_state < 0:
            raise ConfigurationError(
                "random_state must be zero or greater.",
                details={"random_state": self.random_state},
            )

        if (
            self.max_categorical_cardinality is not None
            and self.max_categorical_cardinality < 1
        ):
            raise ConfigurationError(
                "max_categorical_cardinality must be at least 1.",
                details={
                    "max_categorical_cardinality": self.max_categorical_cardinality
                },
            )

        tags = _clean_names(self.tags)
        if len(tags) > settings.max_experiment_tags:
            raise ConfigurationError(
                f"At most {settings.max_experiment_tags} tags are allowed, got "
                f"{len(tags)}.",
                details={
                    "tag_count": len(tags),
                    "max_tags": settings.max_experiment_tags,
                },
            )
        if any(len(tag) > MAX_TAG_LENGTH for tag in tags):
            raise ConfigurationError(
                f"Each tag must be at most {MAX_TAG_LENGTH} characters.",
                details={"max_tag_length": MAX_TAG_LENGTH},
            )

        name = _clean(self.name)
        if name is not None and len(name) > MAX_NAME_LENGTH:
            raise ConfigurationError(
                f"name must be at most {MAX_NAME_LENGTH} characters.",
                details={"max_name_length": MAX_NAME_LENGTH},
            )
        description = _clean(self.description)
        if description is not None and len(description) > MAX_DESCRIPTION_LENGTH:
            raise ConfigurationError(
                f"description must be at most {MAX_DESCRIPTION_LENGTH} characters.",
                details={"max_description_length": MAX_DESCRIPTION_LENGTH},
            )

        overlap = sorted(set(self.excluded_columns) & set(self.identifier_columns))
        if overlap:
            raise ConfigurationError(
                "A column cannot be both excluded and an identifier: "
                + ", ".join(overlap)
                + ".",
                details={"columns": overlap},
            )

        return replace(
            self,
            target_column=_clean(self.target_column),
            models=models,
            primary_metric=_clean(self.primary_metric),
            strategy=strategy,
            excluded_columns=_clean_names(self.excluded_columns),
            identifier_columns=_clean_names(self.identifier_columns),
            scaling_strategy=_clean(self.scaling_strategy),
            numeric_imputation=_clean(self.numeric_imputation),
            categorical_imputation=_clean(self.categorical_imputation),
            name=name,
            description=description,
            tags=tags,
        )
