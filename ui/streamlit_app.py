"""
Streamlit UI - Explainability Panel (stand-out feature).

Run with:
    streamlit run ui/streamlit_app.py -- --offline-demo   (no model download)
    streamlit run ui/streamlit_app.py                      (real local models)

Shows, for every question asked:
  - which LangGraph nodes executed and in what order
  - the retrieved passages with their hybrid retrieval scores
  - the generated answer and its structured JSON
  - verification status (pass/fail, evidence overlap, failure reasons)
  - per-node latency and total latency
This is deliberately a thin presentation layer over app.run_question - all
actual logic lives in graph/ and nodes/, not here.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import time

import streamlit as st

from config import QUERY_CACHE_PATH
from graph.workflow import build_graph, initial_state
from retrieval.retriever import Retriever
from utils.cache import QueryCache

st.set_page_config(page_title="OrbitDesk Support Agent", layout="wide")


@st.cache_resource(show_spinner="Loading retriever and local model...")
def load_app(offline: bool):
    retriever = Retriever.build(offline=offline)
    graph_app, model_name = build_graph(retriever, offline=offline)
    return graph_app, model_name


def main():
    st.title("OrbitDesk Support Agent")
    st.caption("Local-first RAG support agent - LangGraph + Hugging Face, fully offline after model download.")

    offline = st.sidebar.checkbox(
        "Offline demo mode (hashing embedder + mock generator, no model download)",
        value="--offline-demo" in sys.argv,
    )
    use_cache = st.sidebar.checkbox("Use smart cache", value=True)

    graph_app, model_name = load_app(offline)
    st.sidebar.markdown(f"**Generation model:** `{model_name}`")

    sample_questions = json.load(open("data/sample_questions.json"))["questions"]
    sample_labels = {q["question_id"]: q["question"] for q in sample_questions}
    chosen_sample = st.sidebar.selectbox(
        "Load a sample question", ["(none)"] + list(sample_labels.keys())
    )

    default_text = sample_labels.get(chosen_sample, "") if chosen_sample != "(none)" else ""
    question = st.text_area("Ask a question", value=default_text, height=90)

    if "previous_question" not in st.session_state:
        st.session_state.previous_question = None

    treat_as_followup = st.checkbox(
        "Treat as a follow-up to my last question",
        value=False,
        help="Prepends your previous question so short follow-ups like "
             "'what about admins?' keep the right context.",
    )
    if st.session_state.previous_question:
        st.caption(f"Previous question in this session: \u201c{st.session_state.previous_question}\u201d")

    if st.button("Run", type="primary") and question.strip():
        previous_question = st.session_state.previous_question if treat_as_followup else None
        cache = QueryCache(QUERY_CACHE_PATH)
        cached = cache.get(question) if (use_cache and previous_question is None) else None

        if cached:
            st.info("Served from smart cache (identical question asked before).")
            response = cached
            node_trace = response.get("_debug", {}).get("node_trace", [])
            timings = response.get("_debug", {}).get("timings_ms", {})
            retrieved = response.get("_debug", {}).get("retrieved_docs", [])
            latency = response.get("_debug", {}).get("total_latency_ms", 0)
        else:
            start = time.perf_counter()
            result = graph_app.invoke(initial_state(question, previous_question=previous_question))
            latency = round((time.perf_counter() - start) * 1000, 1)
            response = result["final_response"]
            node_trace = result["node_trace"]
            timings = result["timings_ms"]
            retrieved = result.get("retrieved_docs", [])
            if result.get("rewritten_query"):
                st.caption(f"Rewritten query used for retrieval: \u201c{result['rewritten_query']}\u201d")
            if use_cache and previous_question is None:
                debug_payload = dict(response)
                debug_payload["_debug"] = {
                    "node_trace": node_trace, "timings_ms": timings,
                    "total_latency_ms": latency,
                    "retrieved_docs": [{"source_id": d["source_id"], "score": d["score"]} for d in retrieved],
                }
                cache.set(question, debug_payload)

        st.session_state.previous_question = question

        col_answer, col_meta = st.columns([2, 1])

        with col_answer:
            st.subheader("Answer")
            st.write(response["answer"])
            st.subheader("Structured JSON")
            st.json({k: v for k, v in response.items() if k != "_debug"})

        with col_meta:
            st.subheader("Classification")
            st.metric("Route", response["classification"])
            st.metric("Confidence", f"{response['confidence']:.2f}")
            st.metric("Requires human", str(response["requires_human"]))
            st.metric("Total latency", f"{latency:.0f} ms")

            st.subheader("Node execution trace")
            st.code(" -> ".join(node_trace))
            if timings:
                st.table({"node": list(timings.keys()), "latency_ms": list(timings.values())})

        st.subheader("Retrieved evidence")
        if retrieved:
            for d in retrieved:
                title = d.get("title", "")
                score = d.get("score", 0)
                passage = d.get("passage", "")
                superseded = d.get("is_superseded", False)
                label = f"{d['source_id']} - {title} (score {score:.2f})" + (" [SUPERSEDED]" if superseded else "")
                with st.expander(label):
                    st.write(passage)
        else:
            st.write("No documents were retrieved for this question (clarification / out-of-scope path).")

        if response.get("warnings"):
            st.subheader("Warnings")
            for w in response["warnings"]:
                st.warning(w)


if __name__ == "__main__":
    main()
