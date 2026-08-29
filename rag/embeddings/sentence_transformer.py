"""The optional neural embedding provider.

This is the provider to select when term overlap is not enough — when a
question should find a passage that means the same thing in different words.
It is **optional and not installed by default**, because it is expensive in a
way the rest of this project is not:

======================  ====================================================
Model                   ``sentence-transformers/all-MiniLM-L6-v2``
Embedding dimension     384
Package footprint       ``sentence-transformers`` pulls in PyTorch and
                        ``transformers``: roughly 4 GB installed, against
                        about 1 GB for the whole project without it.
Model download          ~90 MB, fetched from the Hugging Face hub on first
                        use and cached under the user's cache directory.
                        **This is the one network access in the RAG layer**,
                        and it happens only if this provider is selected.
Runtime                 CPU is fine. A few hundred short chunks embed in a
                        few seconds; no GPU is required.
======================  ====================================================

Everything here is lazy. The module imports without ``sentence_transformers``
present, constructing a provider downloads nothing, and the model is loaded on
the first call to embed — so importing the RAG package, or running a test
suite that uses the default provider, never touches PyTorch.

**No document or experiment content leaves the machine.** The download is the
model's own weights, and inference is local. There is no API key and no hosted
embedding service in this commit.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from rag.embeddings.base import VECTOR_DTYPE, batched, normalise
from rag.errors import EmbeddingProviderUnavailableError

#: Name recorded in the index manifest.
PROVIDER_NAME = "sentence_transformer"
#: Small, widely used, and good enough for short passages.
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
#: The dimension that model produces. Verified against the loaded model.
DEFAULT_DIMENSION = 384

INSTALL_HINT = (
    "The sentence-transformer provider needs the optional dependency: "
    "pip install sentence-transformers. The default 'hashing' provider "
    "requires no installation and no download."
)


class SentenceTransformerEmbeddingProvider:
    """Embeddings from a locally-run sentence-transformer model."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        dimension: int | None = None,
        batch_size: int = 32,
    ) -> None:
        """Configure the provider without loading anything.

        Args:
            model_name: Hub identifier of the model to use.
            dimension: Expected embedding dimension. Checked against the
                model once it loads; leave unset to take the model's own.
            batch_size: Passages encoded per forward pass.
        """
        self._model_name = model_name
        self._declared_dimension = dimension
        self._batch_size = max(1, int(batch_size))
        self._model = None

    @property
    def identifier(self) -> str:
        """Provider name and model, e.g. ``sentence_transformer-all-MiniLM-L6-v2``."""
        return f"{PROVIDER_NAME}-{self._model_name.rsplit('/', 1)[-1]}"

    @property
    def dimension(self) -> int:
        """Length of every vector this provider returns.

        Taken from the declared dimension when one was given, so that the
        value is available without loading the model; otherwise the model is
        loaded to ask it.
        """
        if self._declared_dimension is not None:
            return self._declared_dimension
        return int(self._load().get_sentence_embedding_dimension())

    @property
    def is_loaded(self) -> bool:
        """Whether the model has actually been loaded yet."""
        return self._model is not None

    def _load(self):
        """Load the model on first use.

        Raises:
            EmbeddingProviderUnavailableError: If the optional dependency is
                missing, or the model cannot be loaded — offline with nothing
                cached, for instance. The message says how to proceed rather
                than surfacing an import traceback.
        """
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingProviderUnavailableError(
                f"sentence-transformers is not installed. {INSTALL_HINT}",
                details={"provider": PROVIDER_NAME, "model": self._model_name},
            ) from exc

        try:
            model = SentenceTransformer(self._model_name)
        except Exception as exc:  # noqa: BLE001 - hub, network and cache errors
            raise EmbeddingProviderUnavailableError(
                f"Could not load embedding model '{self._model_name}'. It is "
                "downloaded on first use and cached afterwards, so this "
                "usually means no network access and no cached copy. "
                f"{INSTALL_HINT}",
                details={"provider": PROVIDER_NAME, "model": self._model_name},
            ) from exc

        actual = int(model.get_sentence_embedding_dimension())
        if self._declared_dimension is not None and actual != self._declared_dimension:
            raise EmbeddingProviderUnavailableError(
                f"Model '{self._model_name}' produces {actual}-dimensional "
                f"vectors, but {self._declared_dimension} was configured.",
                details={
                    "provider": PROVIDER_NAME,
                    "model": self._model_name,
                    "model_dimension": actual,
                    "configured_dimension": self._declared_dimension,
                },
            )
        self._declared_dimension = actual
        self._model = model
        return model

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        """Encode a sequence of texts in batches."""
        model = self._load()
        if not texts:
            return np.zeros((0, self.dimension), dtype=VECTOR_DTYPE)
        parts = [
            np.asarray(
                model.encode(
                    list(batch),
                    batch_size=self._batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
            )
            for batch in batched(list(texts), self._batch_size)
        ]
        return normalise(np.vstack(parts))

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Embed passages for indexing."""
        return self._encode(list(texts))

    def embed_query(self, text: str) -> np.ndarray:
        """Embed one question."""
        return self._encode([text])[0]


__all__ = [
    "DEFAULT_DIMENSION",
    "DEFAULT_MODEL",
    "PROVIDER_NAME",
    "SentenceTransformerEmbeddingProvider",
]
