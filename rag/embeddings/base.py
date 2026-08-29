"""The embedding contract.

Retrieval depends on this interface, never on a particular model or vendor.
Swapping the offline default for a sentence-transformer, or later for a hosted
API, is a change to which provider is constructed — the chunker, the vector
store and the retrieval service do not know the difference.

Two operations, deliberately separate. ``embed_documents`` embeds passages
being indexed; ``embed_query`` embeds a question being asked. They are the
same operation for the providers here, but several real embedding models use
different instructions or prefixes for the two, and a provider that needs to
must be able to without the callers changing.

Every provider must satisfy three properties, because the rest of the layer
relies on them:

- **Fixed dimension.** ``dimension`` is what every vector has, always.
- **Unit length.** Vectors are L2-normalised, so a dot product *is* the cosine
  similarity and the store never has to normalise again.
- **Determinism.** The same text embeds to the same vector, in this process
  and the next. An index built yesterday must stay comparable to a query
  asked today.

``identifier`` names the provider and its model. It goes in the manifest, and
it is how the indexer notices that an index was built in a different embedding
space and must be rebuilt rather than queried.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np

#: The dtype every provider returns. Half the memory of float64 and more than
#: enough precision for a cosine ranking.
VECTOR_DTYPE = np.float32


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Turns text into unit-length vectors of a fixed dimension."""

    @property
    def identifier(self) -> str:
        """Stable name of this provider and model, e.g. ``hashing-512``.

        Recorded in the index manifest. Two indexes with different
        identifiers are not comparable.
        """
        ...  # pragma: no cover - protocol

    @property
    def dimension(self) -> int:
        """Length of every vector this provider returns."""
        ...  # pragma: no cover - protocol

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Embed passages for indexing.

        Args:
            texts: The passages, in order.

        Returns:
            numpy.ndarray: ``(len(texts), dimension)`` float32, L2-normalised,
            in the same order as the input.
        """
        ...  # pragma: no cover - protocol

    def embed_query(self, text: str) -> np.ndarray:
        """Embed one question.

        Returns:
            numpy.ndarray: ``(dimension,)`` float32, L2-normalised.
        """
        ...  # pragma: no cover - protocol


def normalise(vectors: np.ndarray) -> np.ndarray:
    """Scale each row to unit length, leaving all-zero rows alone.

    A zero row means the text shared nothing with the vocabulary — an empty
    string, or a passage of pure punctuation. Dividing by its zero norm would
    produce ``NaN`` and poison every later comparison, so it stays zero and
    simply scores zero against everything.

    Args:
        vectors: A ``(rows, dimension)`` array.

    Returns:
        numpy.ndarray: float32, same shape, each non-zero row L2-normalised.
    """
    array = np.asarray(vectors, dtype=VECTOR_DTYPE)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    safe = np.where(norms == 0, 1.0, norms)
    return (array / safe).astype(VECTOR_DTYPE, copy=False)


def batched(texts: Sequence[str], size: int) -> list[Sequence[str]]:
    """Split a sequence into batches of at most ``size``."""
    if size < 1:
        raise ValueError("batch size must be at least 1")
    return [texts[start : start + size] for start in range(0, len(texts), size)]


__all__ = ["VECTOR_DTYPE", "EmbeddingProvider", "batched", "normalise"]
