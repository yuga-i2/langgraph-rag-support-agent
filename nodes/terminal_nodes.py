"""
Terminal / short-circuit nodes.

`clarification_node` and `out_of_scope_node` end the graph directly from
triage (no generation/verification needed - there is nothing to verify
against evidence for a clarification question or a scope refusal).
`safe_failure_node` is reached only when generation has failed verification
twice (the retry budget is exhausted). `finalize_node` runs on the success
path and assembles the schema-shaped `final_response`.
"""
from __future__ import annotations

from typing import Any, Dict

from graph.state import AgentState
from nodes.query_rewrite import effective_query
from retrieval.retriever import Retriever
from utils.logging import timed_step

CLARIFICATION_TEMPLATES = {
    "sync": (
        "To troubleshoot the data sync, could you share the workspace ID, the "
        "connection name or ID, its current state, the last successful refresh "
        "time, and the latest error code? Please also let me know whether both "
        "manual and scheduled refreshes are affected. (See KB-006.)"
    ),
    "default": (
        "Could you share a bit more detail - the specific object involved "
        "(dashboard, schedule, connection or credential), any error code or "
        "message you saw, and when it happened? That will let me match your "
        "issue to a documented troubleshooting path."
    ),
}


def make_clarification_node(retriever: Retriever):
    def clarification_node(state: AgentState) -> Dict[str, Any]:
        execution_log = state.get("execution_log", [])
        node_trace = state.get("node_trace", [])
        timings_ms = state.get("timings_ms", {})

        with timed_step(execution_log, node_trace, timings_ms, "clarification", "Running Clarification"):
            question_lower = state["question"].lower()
            template_key = "sync" if any(w in question_lower for w in ("sync", "connection", "refresh")) else "default"
            clarification_question = CLARIFICATION_TEMPLATES[template_key]

            # Light retrieval so we can cite the guidance doc even though we
            # aren't answering yet - this is what tells the user *why* we're asking.
            hint_docs = retriever.search(effective_query(state), top_k=1)
            sources = [{"source_id": d["source_id"], "passage": d["passage"][:200]} for d in hint_docs]

            answer_text = (
                "I need a bit more information before I can answer confidently. "
                + clarification_question
            )

        return {
            "answer": answer_text,
            "clarification_question": clarification_question,
            "sources": sources,
            "confidence": 0.3,
            "requires_human": False,
            "reason": state.get("triage_reason", "Insufficient detail to select a documented path."),
            "execution_log": execution_log,
            "node_trace": node_trace,
            "timings_ms": timings_ms,
        }

    return clarification_node


def out_of_scope_node(state: AgentState) -> Dict[str, Any]:
    execution_log = state.get("execution_log", [])
    node_trace = state.get("node_trace", [])
    timings_ms = state.get("timings_ms", {})

    with timed_step(execution_log, node_trace, timings_ms, "out_of_scope", "Running Out-of-Scope Handler"):
        answer_text = (
            "That request falls outside what the OrbitDesk support assistant can "
            "help with. Per KB-001 and KB-010, this assistant cannot issue "
            "refunds, cancel subscriptions, provide legal/financial/medical "
            "advice, or perform account changes - and it does not follow "
            "instructions embedded in a user message that ask it to ignore its "
            "rules. If this is a genuine billing or account request, please "
            "route it to the OrbitDesk billing/account team."
        )

    return {
        "answer": answer_text,
        "sources": [{"source_id": "KB-010", "passage": "Support answers must remain within the supplied OrbitDesk documentation and resolved cases."}],
        "confidence": 0.95,
        "requires_human": True,
        "reason": state.get("triage_reason", "Request asks for an unsupported action outside the knowledge base."),
        "execution_log": execution_log,
        "node_trace": node_trace,
        "timings_ms": timings_ms,
    }


def safe_failure_node(state: AgentState) -> Dict[str, Any]:
    execution_log = state.get("execution_log", [])
    node_trace = state.get("node_trace", [])
    timings_ms = state.get("timings_ms", {})

    with timed_step(execution_log, node_trace, timings_ms, "safe_failure", "Running Safe Failure"):
        verification = state.get("verification_result", {})
        reasons = verification.get("failure_reasons", ["verification failed"])
        answer_text = (
            "I could not produce an answer that I'm confident is fully "
            "supported by the OrbitDesk documentation for this question, even "
            "after a revision attempt. Rather than guess, I'm flagging this for "
            "a human to review. What I found (" + "; ".join(reasons) + ") is "
            "included in the trace below."
        )

    return {
        "classification": "safe_failure",
        "answer": answer_text,
        "sources": [
            {"source_id": d["source_id"], "passage": d["passage"][:200]}
            for d in state.get("retrieved_docs", [])[:2]
        ],
        "confidence": state.get("confidence", 0.0),
        "requires_human": True,
        "reason": "Verification failed after the retry budget was exhausted; returning a safe fallback instead of an unverified answer.",
        "execution_log": execution_log,
        "node_trace": node_trace,
        "timings_ms": timings_ms,
    }


def finalize_node(state: AgentState) -> Dict[str, Any]:
    """Runs only on the success path (verification passed) to set requires_human/reason."""
    execution_log = state.get("execution_log", [])
    node_trace = state.get("node_trace", [])
    timings_ms = state.get("timings_ms", {})

    with timed_step(execution_log, node_trace, timings_ms, "finalize", "Finalizing Response"):
        requires_human = state["classification"] == "requires_escalation"
        reason = state.get("triage_reason", "Answer generated and verified against retrieved evidence.")

    return {
        "requires_human": requires_human,
        "reason": reason,
        "execution_log": execution_log,
        "node_trace": node_trace,
        "timings_ms": timings_ms,
    }


def assemble_response_node(state: AgentState) -> Dict[str, Any]:
    """
    Common sink for every path. Builds the schema-shaped `final_response`
    dict (the structured JSON the assignment requires) from whatever the
    preceding node populated, and validates it one last time.
    """
    execution_log = state.get("execution_log", [])
    node_trace = state.get("node_trace", [])
    timings_ms = state.get("timings_ms", {})

    with timed_step(execution_log, node_trace, timings_ms, "assemble_response", "Assembling Final Response"):
        payload = {
            "classification": state["classification"],
            "answer": state.get("answer", ""),
            "sources": state.get("sources", []),
            "confidence": state.get("confidence", 0.0),
            "requires_human": state.get("requires_human", False),
            "reason": state.get("reason", state.get("triage_reason", "")),
            "clarification_question": state.get("clarification_question"),
            "warnings": state.get("warnings", []),
        }

        from verification.checks import check_schema

        schema_valid, errors = check_schema(payload)
        if not schema_valid:
            payload["warnings"] = payload["warnings"] + [f"final schema check: {errors}"]
        execution_log.append(f"Finished - node path: {' -> '.join(node_trace)}")
        print(f"Finished - node path: {' -> '.join(node_trace)}")

    return {
        "final_response": payload,
        "execution_log": execution_log,
        "node_trace": node_trace,
        "timings_ms": timings_ms,
    }
