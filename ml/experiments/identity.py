"""Naming experiment runs.

Two identities are recorded, because they answer different questions.

**The configuration hash** is derived only from the inputs: which data, which
preprocessing, which candidate models, which selection strategy, which seed.
Re-run the same setup tomorrow and it hashes the same, which is how repeated
runs of one configuration are found — and how a claim of reproducibility can be
checked rather than trusted.

**The experiment id** identifies one *execution*. It carries the configuration
hash so the two stay linked, plus the moment it ran and a short random suffix,
so deliberately repeating a run produces a separate record instead of silently
overwriting the first. A timestamp alone would not do: two runs in the same
second would collide, and the id would say nothing about what was run.

    exp_<configuration hash>_<UTC timestamp>_<random suffix>
    exp_4f2a91c0d8e3_20260826T134500Z_9f3a

Ids double as directory names, so they are restricted to characters that
cannot escape a storage directory. Nothing derived from user input reaches the
filesystem without passing :func:`validate_experiment_id`.
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
from datetime import datetime, timezone
from typing import Any

from ml.errors import InvalidExperimentIdError

#: Prefix every experiment id carries, so a stray value is obvious on sight.
EXPERIMENT_ID_PREFIX = "exp"
#: Characters of the configuration digest kept in the id.
CONFIGURATION_HASH_LENGTH = 12
#: Bytes of randomness distinguishing two executions of the same configuration
#: started within the same second.
RUN_SUFFIX_BYTES = 2
#: Ids are used as directory names: letters, digits, underscore and hyphen
#: only, starting with an alphanumeric. No dots, no separators, no traversal.
EXPERIMENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")

TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"


def canonical_json(payload: Any) -> str:
    """Render a value as JSON in a form that hashes reproducibly.

    Keys are sorted and whitespace removed, so two equal configurations
    produce byte-identical text regardless of how their dictionaries were
    built.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def configuration_hash(components: dict[str, Any]) -> str:
    """Hash the inputs that define an experiment's configuration.

    Args:
        components: The configuration to hash. Every value must already be
            JSON-friendly; the caller decides what counts as configuration.

    Returns:
        str: A short hex digest, stable across processes and machines.
    """
    digest = hashlib.sha256(canonical_json(components).encode("utf-8"))
    return digest.hexdigest()[:CONFIGURATION_HASH_LENGTH]


def generate_experiment_id(
    config_hash: str, *, created_at: datetime | None = None
) -> str:
    """Build the identifier for one execution.

    Args:
        config_hash: The configuration hash this run belongs to.
        created_at: When the run happened; now, in UTC, when omitted.

    Returns:
        str: A validated experiment id.
    """
    moment = created_at or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    stamp = moment.astimezone(timezone.utc).strftime(TIMESTAMP_FORMAT)
    suffix = secrets.token_hex(RUN_SUFFIX_BYTES)
    return validate_experiment_id(
        f"{EXPERIMENT_ID_PREFIX}_{config_hash}_{stamp}_{suffix}"
    )


def validate_experiment_id(experiment_id: Any) -> str:
    """Check that an identifier is safe to use as a storage key.

    An id becomes a directory name, so anything that could climb out of the
    storage root — a separator, a dot segment, an absolute path — is rejected
    here rather than being sanitised into something that looks acceptable.

    Args:
        experiment_id: The identifier to check.

    Returns:
        str: The identifier, unchanged.

    Raises:
        InvalidExperimentIdError: If it is not a safe, well-formed id.
    """
    if not isinstance(experiment_id, str):
        raise InvalidExperimentIdError(
            "An experiment id must be a string, not "
            f"{type(experiment_id).__name__}.",
            details={"received_type": type(experiment_id).__name__},
        )
    if not EXPERIMENT_ID_PATTERN.match(experiment_id):
        raise InvalidExperimentIdError(
            f"'{experiment_id}' is not a valid experiment id. Ids may contain "
            "letters, digits, underscores and hyphens only, must start with a "
            "letter or digit, and must be at most 128 characters.",
            details={"experiment_id": experiment_id},
        )
    return experiment_id
