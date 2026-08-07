"""
Shared, typed state that flows through every LangGraph node.

Every node reads what it needs from AgentState and returns a partial dict
of updates (the LangGraph reducer pattern). Nothing is stored on instance
attributes or globals, so the graph stays stateless and re-entrant.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class RetrievedDoc(TypedDict):
    source_id: str        # e.g. "KB-004" or "CASE-1041"
    title: str
    passage: str           # the chunk text used as evidence
    score: float            # combined hybrid retrieval score, 0-1
    vector_score: float
    keyword_score: float
    is_superseded: bool     # True for resolved cases marked "superseded"


class VerificationResult(TypedDict):
    passed: bool
    schema_valid: bool
    has_sources: bool
    evidence_overlap: float
    unsupported_sentences: List[str]
    failure_reasons: List[str]


class AgentState(TypedDict, total=False):
    # --- input ---
    question: str
    previous_question: Optional[str]      # last question in this session, for follow-up detection

    # --- triage ---
    classification: str                  # answerable | requires_clarification | requires_escalation | out_of_scope
    triage_reason: str
    rewritten_query: str                  # query-rewriting stand-out feature

    # --- retrieval ---
    retrieved_docs: List[RetrievedDoc]
    retrieval_confidence: float

    # --- generation ---
    answer: str
    sources: List[Dict[str, str]]
    confidence: float
    generation_attempts: int

    # --- verification / retry ---
    verification_result: VerificationResult
    retry_count: int
    requires_human: bool
    reason: str
    clarification_question: Optional[str]
    warnings: List[str]

    # --- final structured output ---
    final_response: Dict[str, Any]

    # --- observability ---
    execution_log: List[str]
    node_trace: List[str]
    timings_ms: Dict[str, float]
