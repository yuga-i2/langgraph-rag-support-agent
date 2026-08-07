"""
Embedding model wrapper.

Production path: sentence-transformers loads BAAI/bge-small-en-v1.5 locally
(~130MB, CPU-friendly, strong retrieval quality for its size).

`transformers`/`sentence-transformers`/`torch` are imported lazily, inside
`_load()`, rather than at module level. Two reasons:
  1. The graph-routing and node logic can be unit-tested on a machine that
     hasn't downloaded any model weights yet (see tests/).
  2. `app.py --offline-demo` can run the whole workflow with a lightweight
     hashing embedder when no GPU/large-RAM box is available, which is
     useful for quick smoke-testing the graph wiring itself.

Both embedders expose the same `.encode(texts) -> np.ndarray[float32]`
interface (L2-normalised rows), so the retriever code never needs to know
which one is active.
"""
from __future__ import annotations

import hashlib
from typing import List, Protocol

import numpy as np

from config import MODEL_CONFIG


class Embedder(Protocol):
    def encode(self, texts: List[str]) -> np.ndarray: ...


class HuggingFaceEmbedder:
    """Real embedder used in production. Requires sentence-transformers + torch."""

    def __init__(
        self,
        model_name: str = MODEL_CONFIG.embedding_model_name,
        revision: str = MODEL_CONFIG.embedding_model_revision,
    ):
        self.model_name = model_name
        self.revision = revision
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # lazy import

            self._model = SentenceTransformer(self.model_name, revision=self.revision)
        return self._model

    def encode(self, texts: List[str]) -> np.ndarray:
        model = self._load()
        # bge models are trained with a query instruction prefix for asymmetric
        # search; we keep passages unprefixed and let callers add the query
        # instruction explicitly (see retriever.embed_query).
        vectors = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return vectors.astype(np.float32)


class HashingEmbedder:
    """
    Deterministic, dependency-free fallback embedder.

    Not semantically meaningful beyond crude token overlap - this exists only
    so the LangGraph plumbing, retrieval scoring/reranking, and verification
    logic can be exercised in tests/CI without a multi-hundred-megabyte model
    download. It is never used when a real model is available.
    """

    def __init__(self, dims: int = 384):
        self.dims = dims

    def encode(self, texts: List[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dims), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in text.lower().split():
                h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
                vectors[i, h % self.dims] += 1.0
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms


def build_embedder(offline: bool = False) -> Embedder:
    if offline:
        return HashingEmbedder()
    try:
        embedder = HuggingFaceEmbedder()
        embedder._load()  # fail fast if weights aren't available
        return embedder
    except Exception as exc:  # noqa: BLE001 - deliberate broad fallback
        print(f"[embedding_model] Falling back to HashingEmbedder: {exc}")
        return HashingEmbedder()
