"""
Retrieval node.

Runs the hybrid retriever, filters out chunks whose combined score is
negligible, and records a `retrieval_confidence` (the top chunk's score).
That confidence is used by `graph/workflow.route_after_retrieval` to
implement stand-out feature #3 (Confidence-Based Routing): even a question
that *looks* answerable at triage time gets rerouted to clarification if
the knowledge base genuinely has no strong match, instead of letting the
generator guess.
"""
from __future__ import annotations

from typing import Any, Dict

from config import RETRIEVAL_CONFIG
from graph.state import AgentState
from nodes.query_rewrite import effective_query
from retrieval.retriever import Retriever
from utils.logging import timed_step


def make_retrieval_node(retriever: Retriever):
    def retrieval_node(state: AgentState) -> Dict[str, Any]:
        execution_log = state.get("execution_log", [])
        node_trace = state.get("node_trace", [])
        timings_ms = state.get("timings_ms", {})

        with timed_step(execution_log, node_trace, timings_ms, "retrieval", "Running Retrieval"):
            docs = retriever.search(effective_query(state), top_k=RETRIEVAL_CONFIG.top_k_final)
            confidence = docs[0]["score"] if docs else 0.0
            log_line = f"Retrieved {len(docs)} chunks (top score={confidence:.2f})"
            execution_log.append(log_line)
            print(log_line)
            for d in docs:
                execution_log.append(
                    f"  - {d['source_id']} ({d['title']}) score={d['score']:.2f}"
                    + (" [superseded]" if d["is_superseded"] else "")
                )

        return {
            "retrieved_docs": docs,
            "retrieval_confidence": confidence,
            "execution_log": execution_log,
            "node_trace": node_trace,
            "timings_ms": timings_ms,
        }

    return retrieval_node
