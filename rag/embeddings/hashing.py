"""The offline default embedding provider.

Two hashed n-gram channels, concatenated and L2-normalised:

- **words**, unigrams and bigrams, which match on shared terminology;
- **character n-grams within word boundaries**, three to five characters,
  which survive the morphology that trips a word-only match — *leak*,
  *leaks*, *leakage*, *leakage-safe* all share substrings.

``sklearn.feature_extraction.text.HashingVectorizer`` does the work. It is
**stateless**: there is no vocabulary to fit, so nothing has to be persisted
alongside the index, adding a document never invalidates the vectors already
stored, and the same text embeds identically in this process and the next.
That property is what makes the index reproducible, and it is the reason this
provider is the default rather than a fitted TF-IDF.

**What this is, honestly.** These are *term-overlap* embeddings. They match a
question to a passage by the words and word-fragments they share. They do
**not** capture meaning: a question about "avoiding target leakage" will not
find a passage that only ever says "keeping the test set untouched", because
the two share almost no substrings. True semantic matching needs a trained
model — see :mod:`rag.embeddings.sentence_transformer`, which is supported and
optional.

**What it buys.** No download, no API key, no network, no gigabytes of
dependencies, and identical behaviour on every machine. For a corpus of
project documentation and experiment records — where questions and documents
use the same vocabulary because both are about this project — it retrieves
well enough to be useful, and the evaluation in :mod:`rag.evaluation` reports
exactly how well rather than asserting it.

Resource requirements: none beyond scikit-learn and numpy, both already
required by the ML layer. Embedding a few hundred chunks takes well under a
second, and nothing is loaded at import time.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from rag.embeddings.base import VECTOR_DTYPE, normalise

#: Name recorded in the index manifest.
PROVIDER_NAME = "hashing"
#: Word n-gram range for the first channel.
WORD_NGRAM_RANGE = (1, 2)
#: Character n-gram range for the second channel, within word boundaries.
CHAR_NGRAM_RANGE = (3, 5)
#: Fraction of the total dimension given to the word channel. The character
#: channel gets the rest. Words carry more signal per feature; characters are
#: there to catch what the words miss.
WORD_CHANNEL_SHARE = 0.5


class HashingEmbeddingProvider:
    """Deterministic, offline embeddings from hashed n-grams."""

    def __init__(self, dimension: int = 512) -> None:
        """Configure the provider.

        Nothing is built here. The vectorisers are created on first use, so
        importing this module — or constructing a provider that is never
        used — costs nothing.

        Args:
            dimension: Total vector length, split across the two channels.

        Raises:
            ValueError: If the dimension is too small to split.
        """
        if dimension < 4:
            raise ValueError("dimension must be at least 4")
        self._dimension = int(dimension)
        self._word_features = max(2, int(self._dimension * WORD_CHANNEL_SHARE))
        self._char_features = self._dimension - self._word_features
        self._word_vectorizer = None
        self._char_vectorizer = None

    @property
    def identifier(self) -> str:
        """Provider name and dimension, e.g. ``hashing-512``."""
        return f"{PROVIDER_NAME}-{self._dimension}"

    @property
    def dimension(self) -> int:
        """Length of every vector this provider returns."""
        return self._dimension

    def _ensure_ready(self) -> None:
        """Build the vectorisers on first use.

        Imported here rather than at module scope so that importing the RAG
        package does not pull in scikit-learn until something actually
        embeds.
        """
        if self._word_vectorizer is not None:
            return
        from sklearn.feature_extraction.text import HashingVectorizer

        # alternate_sign keeps the hashed features roughly unbiased, which
        # matters because collisions are certain at this dimension.
        # norm=None because normalisation happens once, after concatenation.
        self._word_vectorizer = HashingVectorizer(
            n_features=self._word_features,
            analyzer="word",
            ngram_range=WORD_NGRAM_RANGE,
            lowercase=True,
            norm=None,
            alternate_sign=True,
        )
        self._char_vectorizer = HashingVectorizer(
            n_features=self._char_features,
            analyzer="char_wb",
            ngram_range=CHAR_NGRAM_RANGE,
            lowercase=True,
            norm=None,
            alternate_sign=True,
        )

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        """Embed a batch of texts into unit-length vectors."""
        self._ensure_ready()
        if not texts:
            return np.zeros((0, self._dimension), dtype=VECTOR_DTYPE)

        prepared = [text if text and text.strip() else " " for text in texts]
        # Each channel is normalised before concatenation so that a very long
        # passage cannot let one channel dominate the other purely by length.
        words = normalise(self._word_vectorizer.transform(prepared).toarray())
        chars = normalise(self._char_vectorizer.transform(prepared).toarray())
        return normalise(np.hstack([words, chars]))

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Embed passages for indexing.

        Returns:
            numpy.ndarray: ``(len(texts), dimension)`` float32, L2-normalised.
        """
        return self._embed(list(texts))

    def embed_query(self, text: str) -> np.ndarray:
        """Embed one question.

        Documents and queries go through the same transformation, because
        this provider has no notion of instruction prefixes — both are just
        text, and both must land in the same space.
        """
        return self._embed([text])[0]


__all__ = ["PROVIDER_NAME", "HashingEmbeddingProvider"]
