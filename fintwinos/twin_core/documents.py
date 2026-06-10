"""Pure-numpy TF-IDF document store with cosine-similarity search.

Holds policy documents, case narratives, filings and any other text artefact
in the twin. Retrieval is classic lexical TF-IDF: lowercase alphanumeric
tokenisation, smoothed inverse document frequency, L2-normalised vectors and
a cosine-similarity ranking — all implemented directly in numpy with no ML
dependencies, so it is fast, deterministic and fully offline.

Implements the :class:`fintwinos.core.interfaces.DocumentStore` protocol.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenise(text: str) -> list[str]:
    """Lowercase alphanumeric tokens of ``text`` (the corpus tokenisation rule)."""
    return _TOKEN_RE.findall(text.lower())


class TfidfDocumentStore:
    """Incremental TF-IDF index over a tokenised corpus.

    Documents can be added (or replaced by id) at any time; the TF-IDF matrix
    is rebuilt lazily on the next :meth:`search`, so bulk ingestion costs one
    vectorisation pass rather than one per document.

    Vector weights use term frequency normalised by document length times the
    smoothed idf ``log((1 + N) / (1 + df)) + 1``; rows are L2-normalised so
    the matrix-vector product against a normalised query vector is exactly
    cosine similarity.
    """

    def __init__(self) -> None:
        self._docs: dict[str, dict[str, Any]] = {}
        self._order: list[str] = []
        self._dirty = True
        self._ids: list[str] = []
        self._vocab: dict[str, int] = {}
        self._idf: np.ndarray = np.zeros(0)
        self._matrix: np.ndarray = np.zeros((0, 0))

    # -- write ---------------------------------------------------------------------

    def add(self, doc_id: str, text: str, metadata: dict[str, Any] | None = None) -> None:
        """Add a document (or replace it if ``doc_id`` already exists)."""
        if doc_id not in self._docs:
            self._order.append(doc_id)
        self._docs[doc_id] = {
            "text": text,
            "metadata": dict(metadata or {}),
            "tokens": tokenise(text),
        }
        self._dirty = True

    # -- read ----------------------------------------------------------------------

    def get(self, doc_id: str) -> dict[str, Any] | None:
        """The stored document as ``{"doc_id", "text", "metadata"}``, or None."""
        doc = self._docs.get(doc_id)
        if doc is None:
            return None
        return {"doc_id": doc_id, "text": doc["text"], "metadata": dict(doc["metadata"])}

    def count(self) -> int:
        """Number of documents currently in the corpus."""
        return len(self._docs)

    def doc_ids(self) -> list[str]:
        """Document ids in insertion order."""
        return list(self._order)

    # -- search --------------------------------------------------------------------

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        """Top-``k`` documents by cosine similarity to ``query``.

        Returns ``(doc_id, score)`` pairs sorted by descending score (ties
        broken by doc id for determinism); documents with zero similarity are
        omitted, so fewer than ``k`` results may be returned.
        """
        if not self._docs or k <= 0:
            return []
        if self._dirty:
            self._rebuild()
        query_counts = np.zeros(len(self._vocab))
        for token in tokenise(query):
            idx = self._vocab.get(token)
            if idx is not None:
                query_counts[idx] += 1.0
        total = query_counts.sum()
        if total == 0:
            return []
        query_vector = (query_counts / total) * self._idf
        norm = float(np.linalg.norm(query_vector))
        if norm == 0:
            return []
        scores = self._matrix @ (query_vector / norm)
        ranked = sorted(range(len(self._ids)), key=lambda i: (-scores[i], self._ids[i]))
        return [
            (self._ids[i], float(scores[i]))
            for i in ranked[:k]
            if scores[i] > 0.0
        ]

    # -- internal -------------------------------------------------------------------

    def _rebuild(self) -> None:
        """Recompute vocabulary, idf vector and the normalised TF-IDF matrix."""
        ids = list(self._order)
        vocab_terms = sorted({t for doc_id in ids for t in self._docs[doc_id]["tokens"]})
        vocab = {term: i for i, term in enumerate(vocab_terms)}
        n_docs, n_terms = len(ids), len(vocab_terms)
        counts = np.zeros((n_docs, n_terms))
        for row, doc_id in enumerate(ids):
            for token in self._docs[doc_id]["tokens"]:
                counts[row, vocab[token]] += 1.0
        document_frequency = (counts > 0).sum(axis=0)
        idf = np.log((1.0 + n_docs) / (1.0 + document_frequency)) + 1.0
        lengths = counts.sum(axis=1, keepdims=True)
        lengths[lengths == 0] = 1.0
        tfidf = (counts / lengths) * idf
        norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._ids = ids
        self._vocab = vocab
        self._idf = idf
        self._matrix = tfidf / norms
        self._dirty = False
