"""Embedding providers, and how one is chosen.

``base``                  the :class:`EmbeddingProvider` contract
``hashing``               the offline default; no download, no key, no network
``sentence_transformer``  optional neural embeddings, lazily loaded

Providers are resolved by name from a :class:`~rag.config.RagConfig`, so the
choice lives in configuration and nothing downstream names a model. Adding a
provider — a hosted API, a different local model — means writing a class that
satisfies the protocol and registering it here.

The optional provider is imported only when it is asked for, so this package
imports without PyTorch installed.
"""

from __future__ import annotations

from rag.config import RagConfig
from rag.embeddings.base import EmbeddingProvider, batched, normalise
from rag.embeddings.hashing import HashingEmbeddingProvider
from rag.errors import ConfigurationError

#: Provider names accepted in configuration.
HASHING = "hashing"
SENTENCE_TRANSFORMER = "sentence_transformer"
AVAILABLE_PROVIDERS = (HASHING, SENTENCE_TRANSFORMER)


def build_embedding_provider(config: RagConfig) -> EmbeddingProvider:
    """Construct the embedding provider a configuration asks for.

    Nothing is loaded or downloaded here: every provider defers that to its
    first use.

    Args:
        config: Names the provider, its dimension and, where relevant, the
            model.

    Returns:
        EmbeddingProvider: A ready-to-use provider.

    Raises:
        ConfigurationError: If the provider name is unknown.
    """
    name = config.embedding_provider.strip().lower()

    if name == HASHING:
        return HashingEmbeddingProvider(dimension=config.embedding_dimension)

    if name == SENTENCE_TRANSFORMER:
        # Imported here so the default path never touches PyTorch.
        from rag.embeddings.sentence_transformer import (
            DEFAULT_MODEL,
            SentenceTransformerEmbeddingProvider,
        )

        return SentenceTransformerEmbeddingProvider(
            model_name=config.embedding_model or DEFAULT_MODEL,
            dimension=None,
            batch_size=config.embedding_batch_size,
        )

    raise ConfigurationError(
        f"Unknown embedding provider '{config.embedding_provider}'. Available: "
        + ", ".join(AVAILABLE_PROVIDERS)
        + ".",
        details={
            "embedding_provider": config.embedding_provider,
            "available": list(AVAILABLE_PROVIDERS),
        },
    )


__all__ = [
    "AVAILABLE_PROVIDERS",
    "HASHING",
    "SENTENCE_TRANSFORMER",
    "EmbeddingProvider",
    "HashingEmbeddingProvider",
    "batched",
    "build_embedding_provider",
    "normalise",
]
