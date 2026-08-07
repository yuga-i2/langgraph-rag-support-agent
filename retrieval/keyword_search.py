"""
BM25 keyword search - the keyword half of the hybrid-retrieval stand-out
feature (see retrieval/retriever.py for how it's blended with vector
search). Catches exact-token matches (error codes like `render_failed`,
role names, document IDs) that a dense embedding can under-weight.
"""
from __future__ import annotations

import re
from typing import List

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class KeywordIndex:
    def __init__(self, corpus_texts: List[str]):
        self._tokenized_corpus = [_tokenize(t) for t in corpus_texts]
        self._bm25 = BM25Okapi(self._tokenized_corpus)

    def search(self, query: str, top_k: int) -> tuple[List[int], List[float]]:
        scores = self._bm25.get_scores(_tokenize(query))
        top_k = min(top_k, len(scores))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return order, [float(scores[i]) for i in order]
