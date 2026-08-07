"""
Confidence gate (stand-out feature #3: Confidence-Based Routing).

Sits between Retrieval and Generation. Triage can mark a question
"answerable" based on its phrasing, but if the hybrid retriever comes back
with a weak top score, the knowledge base likely doesn't actually cover it
well - generating an answer anyway is how support bots hallucinate. This
node downgrades that case to `requires_clarification` *before* generation
ever runs, which is cheaper and safer than generating-then-verifying-then-
retrying-then-failing.

`requires_escalation` questions are exempt: they're expected to have
partial/limited retrieval support by nature (the KB documents the
escalation *procedure*, not a fix), so low retrieval confidence there is
normal, not a sign of a bad match.
"""
from __future__ import annotations

from typing import Any, Dict

from config import RETRIEVAL_CONFIG
from graph.state import AgentState
from utils.logging import log_step


def confidence_gate_node(state: AgentState) -> Dict[str, Any]:
    execution_log = state.get("execution_log", [])
    node_trace = state.get("node_trace", [])
    node_trace.append("confidence_gate")

    classification = state["classification"]
    confidence = state.get("retrieval_confidence", 0.0)

    if classification == "answerable" and confidence < RETRIEVAL_CONFIG.low_confidence_threshold:
        log_step(
            execution_log,
            f"Confidence gate: retrieval confidence {confidence:.2f} is below "
            f"{RETRIEVAL_CONFIG.low_confidence_threshold:.2f}; downgrading to requires_clarification.",
        )
        return {"classification": "requires_clarification", "execution_log": execution_log, "node_trace": node_trace}

    log_step(execution_log, f"Confidence gate: retrieval confidence {confidence:.2f} is sufficient; proceeding to generation.")
    return {"execution_log": execution_log, "node_trace": node_trace}
