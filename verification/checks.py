"""
Pure verification functions (hallucination guard + schema + evidence checks).

Kept dependency-free (no LangGraph, no model calls) so they can be unit
tested directly against hand-written answer/evidence pairs, independent of
whatever the local model actually produced that run.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Protocol

import numpy as np

from graph.state import RetrievedDoc
from verification.schema import SupportResponse

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "and", "or",
    "in", "on", "for", "with", "this", "that", "it", "if", "not", "can",
    "be", "as", "by", "at", "from",
}


class SupportsEncode(Protocol):
    def encode(self, texts: List[str]) -> np.ndarray: ...


def _tokenize(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS}


def split_sentences(answer: str) -> List[str]:
    cleaned = answer.strip()
    if not cleaned:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(cleaned) if s.strip()]


@dataclass
class EvidenceCheck:
    evidence_overlap: float
    unsupported_sentences: List[str]
    lexical_scores: List[float]
    semantic_scores: List[float]


def check_evidence_support(
    answer: str,
    retrieved_docs: List[RetrievedDoc],
    min_overlap: float = 0.25,
    embedder: Optional[SupportsEncode] = None,
    semantic_threshold: float = 0.55,
) -> EvidenceCheck:
    """
    Hallucination guard, in two layers:

    1. Lexical overlap (always on): a sentence is "lexically supported" if
       it shares >= `min_overlap` of its content words with some retrieved
       passage. Cheap, dependency-free, catches verbatim/near-verbatim
       hallucinations reliably.

    2. Semantic similarity (on when `embedder` is passed): a sentence is
       "semantically supported" if its embedding's cosine similarity to
       some passage embedding is >= `semantic_threshold`. This catches a
       *correct paraphrase* that lexical overlap alone would wrongly flag
       (e.g. "Analysts can't see secrets" vs. "reveal stored secrets" -
       low word overlap, same meaning), and reuses the embedder that's
       already loaded for retrieval, so it adds no new dependency or
       extra model load.

    A sentence only needs to pass ONE of the two layers to count as
    supported - they catch different failure modes, so requiring both
    would make the guard stricter than either alone and increase false
    positives on legitimate paraphrases.

    A sentence that is *only* a citation marker or filler ("Let me know if
    that helps.") is skipped rather than flagged.
    """
    sentences = split_sentences(answer)
    if not sentences:
        return EvidenceCheck(evidence_overlap=0.0, unsupported_sentences=[], lexical_scores=[], semantic_scores=[])

    evidence_tokens = [_tokenize(d["passage"]) for d in retrieved_docs]

    checkable_sentences = []
    lexical_scores: List[float] = []
    for sentence in sentences:
        sentence_tokens = _tokenize(sentence)
        if len(sentence_tokens) < 3:  # too short to meaningfully judge (e.g. "[KB-004]")
            continue
        best_overlap = 0.0
        for ev_tokens in evidence_tokens:
            if not ev_tokens:
                continue
            overlap = len(sentence_tokens & ev_tokens) / len(sentence_tokens)
            best_overlap = max(best_overlap, overlap)
        checkable_sentences.append(sentence)
        lexical_scores.append(best_overlap)

    if not checkable_sentences:
        return EvidenceCheck(evidence_overlap=1.0, unsupported_sentences=[], lexical_scores=[], semantic_scores=[])

    semantic_scores: List[float] = [0.0] * len(checkable_sentences)
    if embedder is not None and retrieved_docs:
        try:
            passage_texts = [d["passage"] for d in retrieved_docs]
            sentence_vecs = embedder.encode(checkable_sentences)
            passage_vecs = embedder.encode(passage_texts)
            # Vectors are already L2-normalised by both embedder implementations,
            # so the dot product IS the cosine similarity.
            similarity_matrix = sentence_vecs @ passage_vecs.T
            semantic_scores = similarity_matrix.max(axis=1).tolist()
        except Exception:  # noqa: BLE001 - semantic layer is best-effort, never fatal
            semantic_scores = [0.0] * len(checkable_sentences)

    unsupported: List[str] = []
    for sentence, lex_score, sem_score in zip(checkable_sentences, lexical_scores, semantic_scores):
        lexically_supported = lex_score >= min_overlap
        semantically_supported = sem_score >= semantic_threshold
        if not (lexically_supported or semantically_supported):
            unsupported.append(sentence)

    supported = len(checkable_sentences) - len(unsupported)
    overlap_ratio = supported / len(checkable_sentences)
    return EvidenceCheck(
        evidence_overlap=round(overlap_ratio, 3),
        unsupported_sentences=unsupported,
        lexical_scores=[round(s, 3) for s in lexical_scores],
        semantic_scores=[round(s, 3) for s in semantic_scores],
    )


def check_schema(payload: dict) -> tuple[bool, list[str]]:
    try:
        SupportResponse.model_validate(payload)
        return True, []
    except Exception as exc:  # noqa: BLE001 - we want the message, not the type
        return False, [str(exc)]


def has_citations(answer: str) -> bool:
    return bool(re.search(r"\[[A-Za-z0-9\-]+\]", answer))
