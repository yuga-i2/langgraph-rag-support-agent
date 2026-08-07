"""
Query rewrite node (stand-out feature: conversation memory / query
rewriting).

Runs first, before Triage. If the current question looks like a follow-up
("What about admins?", "And the API too?") AND a previous question was
supplied by the caller (app.py / the Streamlit UI track this per session),
it prepends the previous question so Triage and Retrieval both see enough
context to work with. If it doesn't look like a follow-up, or there is no
previous question, the node is a no-op.

This is intentionally heuristic, not LLM-based: a query rewrite that
consumes a full model call would roughly double the cost of every request
just to handle the minority of turns that are actually follow-ups. The
trade-off is documented in the README - it will miss some follow-ups
phrased as complete sentences, but it costs nothing and never makes a
clear, self-contained question worse.

`state["question"]` is left untouched (it's the record of what the user
actually typed); everything downstream reads `effective_query(state)`.
"""
from __future__ import annotations

from typing import Any, Dict

from graph.state import AgentState
from utils.logging import log_step

FOLLOWUP_STARTERS = (
    "what about", "how about", "and ", "also ", "same for", "what if",
    "and what about", "does that", "is that", "can they", "can it",
)
MAX_FOLLOWUP_WORDS = 7


def is_probable_followup(question: str) -> bool:
    text = question.strip().lower()
    if not text:
        return False
    if text.startswith(FOLLOWUP_STARTERS):
        return True
    return len(text.split()) <= MAX_FOLLOWUP_WORDS


def effective_query(state: AgentState) -> str:
    """The text every downstream node should actually search/classify on."""
    return state.get("rewritten_query") or state["question"]


def query_rewrite_node(state: AgentState) -> Dict[str, Any]:
    execution_log = state.get("execution_log", [])
    node_trace = state.get("node_trace", [])
    node_trace.append("query_rewrite")

    question = state["question"]
    previous_question = state.get("previous_question")

    if previous_question and is_probable_followup(question):
        rewritten = f"{previous_question.rstrip('.?!')}. Follow-up: {question}"
        log_step(execution_log, f"Query rewrite: treating as a follow-up to the previous question.")
        return {"rewritten_query": rewritten, "execution_log": execution_log, "node_trace": node_trace}

    log_step(execution_log, "Query rewrite: question is self-contained; no rewrite applied.")
    return {"execution_log": execution_log, "node_trace": node_trace}
