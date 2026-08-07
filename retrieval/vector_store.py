"""
Vector index over chunk embeddings.

Uses FAISS (`IndexFlatIP` over L2-normalised vectors == cosine similarity)
when available. `faiss-cpu` is optional at import time - if it isn't
installed, `NumpyFlatIndex` does the identical brute-force search in NumPy.
For a knowledge base this small (a few dozen chunks) the two are
performance-equivalent; FAISS is used because the assignment explicitly
asks for it and because it's the correct choice once the corpus grows.
"""
from __future__ import annotations

from typing import List, Protocol

import numpy as np


class VectorIndex(Protocol):
    def search(self, query_vector: np.ndarray, top_k: int) -> tuple[List[int], List[float]]: ...


class FaissFlatIndex:
    def __init__(self, vectors: np.ndarray):
        import faiss  # lazy import

        self.dim = vectors.shape[1]
        self._index = faiss.IndexFlatIP(self.dim)
        self._index.add(vectors)

    def search(self, query_vector: np.ndarray, top_k: int) -> tuple[List[int], List[float]]:
        scores, indices = self._index.search(query_vector.reshape(1, -1), top_k)
        return indices[0].tolist(), scores[0].tolist()


class NumpyFlatIndex:
    """Brute-force cosine search; drop-in replacement when faiss is unavailable."""

    def __init__(self, vectors: np.ndarray):
        self._vectors = vectors

    def search(self, query_vector: np.ndarray, top_k: int) -> tuple[List[int], List[float]]:
        scores = self._vectors @ query_vector
        top_k = min(top_k, len(scores))
        indices = np.argsort(-scores)[:top_k]
        return indices.tolist(), scores[indices].tolist()


def build_vector_index(vectors: np.ndarray) -> VectorIndex:
    try:
        return FaissFlatIndex(vectors)
    except Exception as exc:  # noqa: BLE001 - deliberate broad fallback
        print(f"[vector_store] Falling back to NumpyFlatIndex: {exc}")
        return NumpyFlatIndex(vectors)
