"""
Verification node.

Runs three independent checks against the generated answer:
  1. Schema validity  - does the assembled payload satisfy SupportResponse?
  2. Source presence   - at least one retrieved source was cited.
  3. Evidence support  - hallucination guard: what fraction of answer
     sentences are grounded in the retrieved passages?

Evidence support (`check_evidence_support`) runs a lexical token-overlap
check unconditionally, and additionally runs a semantic (embedding cosine
similarity) check when an embedder is available - a sentence only needs to
pass ONE of the two to count as grounded. The factory pattern
(`make_verification_node`) lets `build_graph` pass in the retriever's
already-loaded embedder, so the semantic layer costs no extra model load.

All three checks must pass for `verification_result.passed = True`.
Failure reasons are recorded verbatim so the retry/safe-failure decision in
graph/workflow.py is fully explainable.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from config import VERIFICATION_CONFIG
from graph.state import AgentState
from utils.logging import timed_step
from verification.checks import check_evidence_support, check_schema, has_citations


def _build_candidate_payload(state: AgentState) -> dict:
    return {
        "classification": state["classification"],
        "answer": state.get("answer", ""),
        "sources": state.get("sources", []),
        "confidence": state.get("confidence", 0.0),
        "requires_human": state["classification"] == "requires_escalation",
        "reason": state.get("triage_reason", "Generated from retrieved OrbitDesk documentation."),
        "clarification_question": None,
        "warnings": state.get("warnings", []),
    }


def make_verification_node(embedder: Optional[Any] = None):
    def verification_node(state: AgentState) -> Dict[str, Any]:
        execution_log = state.get("execution_log", [])
        node_trace = state.get("node_trace", [])
        timings_ms = state.get("timings_ms", {})

        with timed_step(execution_log, node_trace, timings_ms, "verification", "Running Verification"):
            payload = _build_candidate_payload(state)
            schema_valid, schema_errors = check_schema(payload)

            docs = state.get("retrieved_docs", [])
            answer = state.get("answer", "")
            cited = has_citations(answer)

            evidence = check_evidence_support(
                answer,
                docs,
                min_overlap=0.25,
                embedder=embedder,
                semantic_threshold=VERIFICATION_CONFIG.min_semantic_similarity,
            )
            guard_label = "lexical+semantic" if embedder is not None else "lexical-only"

            failure_reasons: list[str] = []
            if not schema_valid:
                failure_reasons.append(f"Schema validation failed: {schema_errors}")
            if not cited or len(payload["sources"]) == 0:
                failure_reasons.append("Answer does not cite any retrieved source.")
            if evidence.evidence_overlap < VERIFICATION_CONFIG.min_evidence_overlap:
                failure_reasons.append(
                    f"Only {evidence.evidence_overlap:.0%} of answer sentences are grounded "
                    f"in retrieved evidence per the {guard_label} guard "
                    f"(need >= {VERIFICATION_CONFIG.min_evidence_overlap:.0%})."
                )
            if state.get("confidence", 0.0) < VERIFICATION_CONFIG.min_answer_confidence:
                failure_reasons.append(
                    f"Confidence {state.get('confidence', 0.0):.2f} below threshold "
                    f"{VERIFICATION_CONFIG.min_answer_confidence:.2f}."
                )

            passed = len(failure_reasons) == 0

            verification_result = {
                "passed": passed,
                "schema_valid": schema_valid,
                "has_sources": len(payload["sources"]) > 0,
                "evidence_overlap": evidence.evidence_overlap,
                "unsupported_sentences": evidence.unsupported_sentences,
                "failure_reasons": failure_reasons,
                "guard_type": guard_label,
            }

            status = "Verification Passed" if passed else f"Verification Failed: {failure_reasons}"
            execution_log.append(status)
            print(status)

        return {
            "verification_result": verification_result,
            "execution_log": execution_log,
            "node_trace": node_trace,
            "timings_ms": timings_ms,
        }

    return verification_node
