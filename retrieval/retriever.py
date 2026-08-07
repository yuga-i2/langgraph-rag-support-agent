"""
Hybrid retriever (stand-out feature #1: Hybrid Retrieval).

Combines dense semantic search (FAISS over BAAI/bge-small-en-v1.5
embeddings) with sparse keyword search (BM25). Pure vector search alone
under-weights exact tokens that matter a lot in a support KB - error codes
like `render_failed`, role names like `Viewer`, document IDs like `KB-004`.
Pure BM25 alone misses paraphrases ("sync" vs "connection refresh"). The
two scores are min-max normalised per query and combined with a configurable
weight, then the top passages are returned as the reranked result.

The index is built once per process (`Retriever.build`) and reused across
questions; embeddings are computed once at build time, not per-query.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from config import RETRIEVAL_CONFIG, KB_DIR, RESOLVED_CASES_PATH
from graph.state import RetrievedDoc
from models.embedding_model import Embedder, build_embedder
from retrieval.chunker import Chunk, chunk_documents
from retrieval.keyword_search import KeywordIndex
from retrieval.loader import load_all_documents
from retrieval.vector_store import VectorIndex, build_vector_index

# bge models expect this instruction prefix on the *query* side only.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


@dataclass
class ScoredChunk:
    chunk: Chunk
    vector_score: float
    keyword_score: float
    combined_score: float


def _min_max_normalise(values: List[float]) -> List[float]:
    if not values:
        return values
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [0.5 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


class Retriever:
    def __init__(self, chunks: List[Chunk], embedder: Embedder, vector_index: VectorIndex,
                 keyword_index: KeywordIndex, chunk_vectors: np.ndarray):
        self.chunks = chunks
        self.embedder = embedder
        self.vector_index = vector_index
        self.keyword_index = keyword_index
        self.chunk_vectors = chunk_vectors

    @classmethod
    def build(
        cls,
        kb_dir: Path = KB_DIR,
        resolved_cases_path: Path = RESOLVED_CASES_PATH,
        offline: bool = False,
    ) -> "Retriever":
        documents = load_all_documents(kb_dir, resolved_cases_path)
        chunks = chunk_documents(
            documents,
            chunk_size=RETRIEVAL_CONFIG.chunk_size_chars,
            overlap=RETRIEVAL_CONFIG.chunk_overlap_chars,
        )
        embedder = build_embedder(offline=offline)
        chunk_vectors = embedder.encode([c.text for c in chunks])
        vector_index = build_vector_index(chunk_vectors)
        keyword_index = KeywordIndex([c.text for c in chunks])
        return cls(chunks, embedder, vector_index, keyword_index, chunk_vectors)

    def search(self, query: str, top_k: int | None = None) -> List[RetrievedDoc]:
        top_k = top_k or RETRIEVAL_CONFIG.top_k_final

        query_vector = self.embedder.encode([BGE_QUERY_INSTRUCTION + query])[0]
        vec_idx, vec_scores = self.vector_index.search(query_vector, RETRIEVAL_CONFIG.top_k_vector)
        kw_idx, kw_scores = self.keyword_index.search(query, RETRIEVAL_CONFIG.top_k_keyword)

        vec_scores_norm = dict(zip(vec_idx, _min_max_normalise(vec_scores)))
        kw_scores_norm = dict(zip(kw_idx, _min_max_normalise(kw_scores)))

        candidate_indices = set(vec_idx) | set(kw_idx)
        scored: List[ScoredChunk] = []
        for idx in candidate_indices:
            v_score = vec_scores_norm.get(idx, 0.0)
            k_score = kw_scores_norm.get(idx, 0.0)
            combined = (
                RETRIEVAL_CONFIG.vector_weight * v_score
                + RETRIEVAL_CONFIG.keyword_weight * k_score
            )
            scored.append(ScoredChunk(self.chunks[idx], v_score, k_score, combined))

        scored.sort(key=lambda s: s.combined_score, reverse=True)
        top = scored[:top_k]

        return [
            RetrievedDoc(
                source_id=s.chunk.source_id,
                title=s.chunk.title,
                passage=s.chunk.text,
                score=round(s.combined_score, 4),
                vector_score=round(s.vector_score, 4),
                keyword_score=round(s.keyword_score, 4),
                is_superseded=s.chunk.is_superseded,
            )
            for s in top
        ]

    def top_score(self, query: str) -> float:
        """Used for confidence-based clarification routing."""
        results = self.search(query, top_k=1)
        return results[0]["score"] if results else 0.0
