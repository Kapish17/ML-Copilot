"""Configuration for the language-model layer.

Every provider name, model, limit and timeout the layer uses lives here.
Nothing downstream hard-codes a model name or a token budget: a module that
needs one takes an :class:`LLMConfig`.

**The configuration holds the *name* of the environment variable that carries
the API key — never the key itself.** That is not a stylistic choice. A
configuration object gets logged, repr'd into a traceback, serialised into a
debug endpoint and passed around; a key inside one leaks by accident sooner or
later. The provider reads the environment at the moment it needs to
authenticate, uses the value, and never stores it on an attribute.

The defaults describe a system that is **configured but not credentialed**:
importing this package, running the tests and using retrieval all work with no
key present. Only an actual generation request fails, and it fails with a
clear configuration error rather than a stack trace from an SDK.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace

from llm.errors import LLMConfigurationError

# -- Providers --------------------------------------------------------------
#: The OpenAI-compatible chat-completions provider. Works against OpenAI
#: itself and against anything that speaks the same API — Azure OpenAI, vLLM,
#: Ollama, LM Studio, OpenRouter — by pointing ``base_url`` at it.
PROVIDER_OPENAI = "openai"
#: The deterministic in-process provider used by the tests. Never contacts
#: anything.
PROVIDER_FAKE = "fake"
AVAILABLE_PROVIDERS = (PROVIDER_OPENAI, PROVIDER_FAKE)

DEFAULT_PROVIDER = PROVIDER_OPENAI

# -- Model ------------------------------------------------------------------
#: The default model identifier. It is a *default*, not a promise: no model is
#: available until a key and an endpoint are configured, and the provider says
#: so rather than pretending.
DEFAULT_MODEL = "gpt-4o-mini"
#: Environment variable the API key is read from, at generation time.
DEFAULT_API_KEY_ENV = "LLM_API_KEY"
#: Zero, because a grounded answer should not vary between identical runs.
DEFAULT_TEMPERATURE = 0.0
#: Upper bound on the generated answer.
DEFAULT_MAX_OUTPUT_TOKENS = 900
#: Seconds to wait for a provider before giving up.
DEFAULT_TIMEOUT_SECONDS = 30.0
#: Retries after a *transient* failure only. Small and bounded: a request that
#: is failing for a reason retrying cannot fix should fail quickly.
DEFAULT_MAX_RETRIES = 2

# -- Evidence and context ---------------------------------------------------
#: Chunks retrieved before any context trimming. Equal to the context limit by
#: default, so the ordinary path drops nothing and ``context_truncated`` means
#: what it says rather than being true on every answer. Raise it above
#: ``max_context_chunks`` to over-fetch and let the evidence threshold prune.
DEFAULT_MAX_RETRIEVED_CHUNKS = 6
#: Chunks actually placed in the prompt.
DEFAULT_MAX_CONTEXT_CHUNKS = 6
#: Characters of evidence allowed in the prompt.
DEFAULT_MAX_CONTEXT_CHARS = 12_000
#: Smallest useful fragment of a chunk. A chunk that cannot contribute at
#: least this much is left out entirely rather than reduced to a stub.
DEFAULT_MIN_CHUNK_CHARS = 400
#: Similarity below which a chunk is not treated as evidence at all. Above
#: zero on purpose: an answer built from the least-bad match in an index that
#: has nothing relevant is exactly the failure this layer exists to prevent.
DEFAULT_MIN_EVIDENCE_SCORE = 0.05
#: Characters per token, for the approximate budget reported alongside the
#: character count. A rough English heuristic, and labelled as one — no
#: tokeniser is bundled.
CHARS_PER_TOKEN = 4


@dataclass(frozen=True)
class LLMConfig:
    """Every knob the language-model layer has, in one immutable object."""

    provider: str = DEFAULT_PROVIDER
    model: str = DEFAULT_MODEL
    #: Name of the environment variable holding the key. Not the key.
    api_key_env: str = DEFAULT_API_KEY_ENV
    #: Optional endpoint override, for an OpenAI-compatible service that is
    #: not OpenAI. ``None`` uses the SDK's default.
    base_url: str | None = None

    temperature: float = DEFAULT_TEMPERATURE
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_MAX_RETRIES

    max_retrieved_chunks: int = DEFAULT_MAX_RETRIEVED_CHUNKS
    max_context_chunks: int = DEFAULT_MAX_CONTEXT_CHUNKS
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS
    min_chunk_chars: int = DEFAULT_MIN_CHUNK_CHARS
    min_evidence_score: float = DEFAULT_MIN_EVIDENCE_SCORE

    def __post_init__(self) -> None:
        """Validate the combination, not just the individual values."""
        if self.provider not in AVAILABLE_PROVIDERS:
            raise LLMConfigurationError(
                f"Unknown LLM provider '{self.provider}'. Available: "
                + ", ".join(AVAILABLE_PROVIDERS)
                + ".",
                details={
                    "provider": self.provider,
                    "available": list(AVAILABLE_PROVIDERS),
                },
            )
        if not self.model.strip():
            raise LLMConfigurationError(
                "A model identifier is required.", details={"model": self.model}
            )
        if not 0.0 <= self.temperature <= 2.0:
            raise LLMConfigurationError(
                "temperature must be between 0.0 and 2.0.",
                details={"temperature": self.temperature},
            )
        if self.max_output_tokens < 1:
            raise LLMConfigurationError(
                "max_output_tokens must be at least 1.",
                details={"max_output_tokens": self.max_output_tokens},
            )
        if self.timeout_seconds <= 0:
            raise LLMConfigurationError(
                "timeout_seconds must be greater than zero.",
                details={"timeout_seconds": self.timeout_seconds},
            )
        if self.max_retries < 0:
            raise LLMConfigurationError(
                "max_retries cannot be negative.",
                details={"max_retries": self.max_retries},
            )
        if self.max_context_chunks < 1 or self.max_retrieved_chunks < 1:
            raise LLMConfigurationError(
                "chunk limits must be at least 1.",
                details={
                    "max_context_chunks": self.max_context_chunks,
                    "max_retrieved_chunks": self.max_retrieved_chunks,
                },
            )
        if self.max_context_chunks > self.max_retrieved_chunks:
            # Otherwise the context limit is unreachable and misleading.
            raise LLMConfigurationError(
                "max_context_chunks cannot exceed max_retrieved_chunks.",
                details={
                    "max_context_chunks": self.max_context_chunks,
                    "max_retrieved_chunks": self.max_retrieved_chunks,
                },
            )
        if self.max_context_chars < 1:
            raise LLMConfigurationError(
                "max_context_chars must be at least 1.",
                details={"max_context_chars": self.max_context_chars},
            )
        if not 0 <= self.min_chunk_chars <= self.max_context_chars:
            raise LLMConfigurationError(
                "min_chunk_chars must be between 0 and max_context_chars.",
                details={
                    "min_chunk_chars": self.min_chunk_chars,
                    "max_context_chars": self.max_context_chars,
                },
            )
        if not -1.0 <= self.min_evidence_score <= 1.0:
            raise LLMConfigurationError(
                "min_evidence_score must be a similarity in [-1, 1].",
                details={"min_evidence_score": self.min_evidence_score},
            )

    # -- Credentials -------------------------------------------------------

    @property
    def has_api_key(self) -> bool:
        """Whether a key is present in the environment right now.

        Reads the variable but does not return, store or log its value — only
        whether there is one. Safe to call from anywhere, including a health
        check or a status endpoint.
        """
        return bool(os.getenv(self.api_key_env, "").strip())

    def resolve_api_key(self) -> str:
        """Read the API key from the environment.

        Called by a provider at the moment it authenticates, never at import
        and never at construction. The value is used and dropped; nothing in
        this layer keeps it on an attribute.

        Raises:
            LLMConfigurationError: If the variable is unset or blank. The
                message names the *variable*, which is not a secret, and never
                any part of a value.
        """
        key = os.getenv(self.api_key_env, "").strip()
        if not key:
            raise LLMConfigurationError(
                f"No API key found. Set the {self.api_key_env} environment "
                "variable to use the "
                f"'{self.provider}' provider. Retrieval works without one; "
                "only answer generation needs a key.",
                details={
                    "provider": self.provider,
                    "api_key_env": self.api_key_env,
                    "model": self.model,
                },
            )
        return key

    # -- Derived -----------------------------------------------------------

    @property
    def approximate_context_tokens(self) -> int:
        """The character budget expressed as an approximate token budget."""
        return self.max_context_chars // CHARS_PER_TOKEN

    def with_overrides(self, **overrides: object) -> LLMConfig:
        """Return a validated copy with some fields replaced.

        Raises:
            LLMConfigurationError: If a field name is unknown.
        """
        unknown = sorted(set(overrides) - set(self.__dataclass_fields__))
        if unknown:
            raise LLMConfigurationError(
                "Unknown configuration field(s): " + ", ".join(unknown) + ".",
                details={"unknown_fields": unknown},
            )
        return replace(self, **overrides)

    def describe(self) -> dict[str, object]:
        """Render the configuration for a log or a status response.

        Reports whether a key is configured, never the key. Everything here is
        safe to show a caller.
        """
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "api_key_configured": self.has_api_key,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "max_retrieved_chunks": self.max_retrieved_chunks,
            "max_context_chunks": self.max_context_chunks,
            "max_context_chars": self.max_context_chars,
            "approximate_context_tokens": self.approximate_context_tokens,
            "min_evidence_score": self.min_evidence_score,
        }


def _env_float(name: str, default: float) -> float:
    """Read a float from the environment, failing loudly on nonsense."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise LLMConfigurationError(
            f"{name} must be a number, got {raw!r}.", details={name: raw}
        ) from exc


def _env_int(name: str, default: int) -> int:
    """Read an integer from the environment, failing loudly on nonsense."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise LLMConfigurationError(
            f"{name} must be an integer, got {raw!r}.", details={name: raw}
        ) from exc


def config_from_env(**overrides: object) -> LLMConfig:
    """Build a configuration from environment variables, then overrides.

    Reads settings only. The API key is **not** read here — the configuration
    records which variable holds it, and the provider reads that variable when
    it authenticates.
    """
    values: dict[str, object] = {
        "provider": os.getenv("LLM_PROVIDER", "").strip() or DEFAULT_PROVIDER,
        "model": os.getenv("LLM_MODEL", "").strip() or DEFAULT_MODEL,
        "api_key_env": os.getenv("LLM_API_KEY_ENV", "").strip() or DEFAULT_API_KEY_ENV,
        "base_url": os.getenv("LLM_BASE_URL", "").strip() or None,
        "temperature": _env_float("LLM_TEMPERATURE", DEFAULT_TEMPERATURE),
        "max_output_tokens": _env_int(
            "LLM_MAX_OUTPUT_TOKENS", DEFAULT_MAX_OUTPUT_TOKENS
        ),
        "timeout_seconds": _env_float("LLM_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS),
        "max_retries": _env_int("LLM_MAX_RETRIES", DEFAULT_MAX_RETRIES),
        "max_context_chars": _env_int(
            "LLM_MAX_CONTEXT_CHARS", DEFAULT_MAX_CONTEXT_CHARS
        ),
        "max_context_chunks": _env_int(
            "LLM_MAX_CONTEXT_CHUNKS", DEFAULT_MAX_CONTEXT_CHUNKS
        ),
        "max_retrieved_chunks": _env_int(
            "LLM_MAX_RETRIEVED_CHUNKS", DEFAULT_MAX_RETRIEVED_CHUNKS
        ),
        "min_evidence_score": _env_float(
            "LLM_MIN_EVIDENCE_SCORE", DEFAULT_MIN_EVIDENCE_SCORE
        ),
    }
    values.update(overrides)
    return LLMConfig(**values)  # type: ignore[arg-type]


__all__ = [
    "AVAILABLE_PROVIDERS",
    "CHARS_PER_TOKEN",
    "PROVIDER_FAKE",
    "PROVIDER_OPENAI",
    "LLMConfig",
    "config_from_env",
]
