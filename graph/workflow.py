"""
Builds the LangGraph StateGraph for the OrbitDesk support agent.

    START
      |
  query_rewrite (conversation-memory follow-up detection)
      |
   triage --------------------------------------------------------+
      |-- out_of_scope -----------------------> out_of_scope       |
      |-- requires_clarification -------------> clarification      |
      |-- answerable / requires_escalation --> retrieval            |
                                                   |                |
                                             confidence_gate         |
                                                   |-- low conf --> clarification
                                                   |-- ok  --------> generation <-+
                                                                        |         |
                                                                   verification    |
                                                                        |-- fail, retries left -> prepare_retry -+
                                                                        |-- fail, no retries left -> safe_failure
                                                                        |-- pass -----------------> finalize
      clarification / out_of_scope / safe_failure / finalize --> assemble_response --> END

Loop protection: `prepare_retry` is only reachable while
`retry_count < VERIFICATION_CONFIG.max_retries` (checked in
`route_after_verification`), and it always increments retry_count before
looping back, so the generation<->verification cycle runs at most
`max_retries + 1` times regardless of what the model outputs.
"""
from __future__ import annotations

from typing import Any, Dict

from langgraph.graph import END, StateGraph

from config import VERIFICATION_CONFIG
from graph.state import AgentState
from models.generation_model import build_generation_model
from nodes.confidence_gate import confidence_gate_node
from nodes.generation_node import make_generation_node
from nodes.query_rewrite import query_rewrite_node
from nodes.retrieval_node import make_retrieval_node
from nodes.terminal_nodes import (
    assemble_response_node,
    finalize_node,
    make_clarification_node,
    out_of_scope_node,
    safe_failure_node,
)
from nodes.triage import triage_node
from nodes.verification_node import make_verification_node
from retrieval.retriever import Retriever
from utils.logging import log_step


def prepare_retry_node(state: AgentState) -> Dict[str, Any]:
    execution_log = state.get("execution_log", [])
    node_trace = state.get("node_trace", [])
    node_trace.append("prepare_retry")
    retry_count = state.get("retry_count", 0) + 1
    log_step(execution_log, f"Verification failed - retrying generation (attempt {retry_count + 1}).")
    return {"retry_count": retry_count, "execution_log": execution_log, "node_trace": node_trace}


def route_after_triage(state: AgentState) -> str:
    classification = state["classification"]
    if classification == "out_of_scope":
        return "out_of_scope"
    if classification == "requires_clarification":
        return "clarification"
    return "retrieval"  # answerable | requires_escalation


def route_after_confidence_gate(state: AgentState) -> str:
    return "clarification" if state["classification"] == "requires_clarification" else "generation"


def route_after_verification(state: AgentState) -> str:
    if state["verification_result"]["passed"]:
        return "finalize"
    if state.get("retry_count", 0) < VERIFICATION_CONFIG.max_retries:
        return "retry"
    return "safe_failure"


def build_graph(retriever: Retriever, offline: bool = False, generation_model=None):
    """
    `generation_model` can be injected directly (used by tests to force
    specific verification outcomes). When omitted, one is built normally
    based on `offline`.
    """
    if generation_model is None:
        generation_model = build_generation_model(offline=offline)

    graph = StateGraph(AgentState)

    graph.add_node("query_rewrite", query_rewrite_node)
    graph.add_node("triage", triage_node)
    graph.add_node("retrieval", make_retrieval_node(retriever))
    graph.add_node("confidence_gate", confidence_gate_node)
    graph.add_node("clarification", make_clarification_node(retriever))
    graph.add_node("out_of_scope", out_of_scope_node)
    graph.add_node("generation", make_generation_node(generation_model))
    graph.add_node("verification", make_verification_node(retriever.embedder))
    graph.add_node("prepare_retry", prepare_retry_node)
    graph.add_node("safe_failure", safe_failure_node)
    graph.add_node("finalize", finalize_node)
    graph.add_node("assemble_response", assemble_response_node)

    graph.set_entry_point("query_rewrite")
    graph.add_edge("query_rewrite", "triage")

    graph.add_conditional_edges(
        "triage",
        route_after_triage,
        {"out_of_scope": "out_of_scope", "clarification": "clarification", "retrieval": "retrieval"},
    )
    graph.add_edge("retrieval", "confidence_gate")
    graph.add_conditional_edges(
        "confidence_gate",
        route_after_confidence_gate,
        {"clarification": "clarification", "generation": "generation"},
    )
    graph.add_edge("generation", "verification")
    graph.add_conditional_edges(
        "verification",
        route_after_verification,
        {"finalize": "finalize", "retry": "prepare_retry", "safe_failure": "safe_failure"},
    )
    graph.add_edge("prepare_retry", "generation")

    for terminal in ("clarification", "out_of_scope", "safe_failure", "finalize"):
        graph.add_edge(terminal, "assemble_response")
    graph.add_edge("assemble_response", END)

    return graph.compile(), generation_model.model_name if hasattr(generation_model, "model_name") else "unknown"


def initial_state(question: str, previous_question: str | None = None) -> AgentState:
    return AgentState(
        question=question,
        previous_question=previous_question,
        retry_count=0,
        generation_attempts=0,
        warnings=[],
        execution_log=[],
        node_trace=[],
        timings_ms={},
    )
