"""
Response Generation node.

Calls the local generation model with a prompt that contains *only* the
retrieved passages, and asks for inline [source_id] citations. Confidence
here is a simple, explainable blend of retrieval confidence and citation
coverage (fraction of retrieved sources actually cited) rather than a
number invented by the model - small local instruct models are not
well-calibrated confidence estimators, so we compute it deterministically
and let the model focus on writing the answer.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

from graph.state import AgentState, RetrievedDoc
from models.generation_model import GenerationModel
from nodes.query_rewrite import effective_query
from utils.logging import timed_step

CITATION_RE = re.compile(r"\[([A-Za-z0-9\-]+)\]")


def _extract_cited_source_ids(answer_text: str) -> List[str]:
    return list(dict.fromkeys(CITATION_RE.findall(answer_text)))


def _compute_confidence(retrieval_confidence: float, docs: List[RetrievedDoc], cited_ids: List[str]) -> float:
    if not docs:
        return 0.0
    doc_ids = {d["source_id"] for d in docs}
    cited_and_retrieved = doc_ids & set(cited_ids)
    citation_coverage = len(cited_and_retrieved) / max(len(doc_ids), 1)
    # Weighted blend: retrieval quality matters most, citation discipline second.
    confidence = 0.7 * retrieval_confidence + 0.3 * citation_coverage
    return round(min(confidence, 1.0), 3)


def make_generation_node(generation_model: GenerationModel):
    def generation_node(state: AgentState) -> Dict[str, Any]:
        execution_log = state.get("execution_log", [])
        node_trace = state.get("node_trace", [])
        timings_ms = state.get("timings_ms", {})
        attempts = state.get("generation_attempts", 0) + 1

        with timed_step(execution_log, node_trace, timings_ms, "generation", "Generating Response"):
            docs = state.get("retrieved_docs", [])
            result = generation_model.generate(effective_query(state), docs)
            cited_ids = _extract_cited_source_ids(result.text)
            confidence = _compute_confidence(state.get("retrieval_confidence", 0.0), docs, cited_ids)

            sources = [
                {"source_id": d["source_id"], "passage": d["passage"][:280]}
                for d in docs
                if d["source_id"] in cited_ids or not cited_ids  # fall back to all retrieved docs
            ]

            execution_log.append(f"Generated answer with {result.model_name} (attempt {attempts})")
            print(f"Generated answer with {result.model_name} (attempt {attempts})")

        return {
            "answer": result.text,
            "sources": sources,
            "confidence": confidence,
            "generation_attempts": attempts,
            "execution_log": execution_log,
            "node_trace": node_trace,
            "timings_ms": timings_ms,
        }

    return generation_node
