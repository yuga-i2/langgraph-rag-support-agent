"""
CLI entry point for the OrbitDesk Support Agent.

Usage:
    python app.py                              interactive REPL
    python app.py -q "Can a Viewer create an API credential?"
    python app.py --samples                    run all data/sample_questions.json
    python app.py --offline-demo                use the offline hashing embedder +
                                                  mock generator (no model download,
                                                  useful for smoke-testing the graph)
"""
from __future__ import annotations

import argparse
import json
import time

from config import QUERY_CACHE_PATH
from graph.workflow import build_graph, initial_state
from retrieval.retriever import Retriever
from utils.cache import QueryCache


def run_question(app, cache: QueryCache, question: str, use_cache: bool = True, previous_question: str | None = None) -> dict:
    if use_cache and previous_question is None:
        cached = cache.get(question)
        if cached is not None:
            print(f"[cache] Reusing cached response for: {question[:60]}...")
            return cached

    start = time.perf_counter()
    result = app.invoke(initial_state(question, previous_question=previous_question))
    elapsed_ms = (time.perf_counter() - start) * 1000

    response = result["final_response"]
    response["_debug"] = {
        "node_trace": result["node_trace"],
        "timings_ms": result["timings_ms"],
        "total_latency_ms": round(elapsed_ms, 1),
        "retry_count": result.get("retry_count", 0),
        "rewritten_query": result.get("rewritten_query"),
        "retrieved_docs": [
            {"source_id": d["source_id"], "score": d["score"]} for d in result.get("retrieved_docs", [])
        ],
    }

    if use_cache and previous_question is None:
        cache.set(question, response)
    return response


def print_response(question: str, response: dict) -> None:
    print("\n" + "-" * 78)
    print(f"Q: {question}")
    print("-" * 78)
    print(response["answer"])
    print("\nStructured JSON:")
    printable = {k: v for k, v in response.items() if k != "_debug"}
    print(json.dumps(printable, indent=2))
    print(f"\n[debug] node path: {' -> '.join(response['_debug']['node_trace'])}")
    print(f"[debug] total latency: {response['_debug']['total_latency_ms']} ms")


def main():
    parser = argparse.ArgumentParser(description="OrbitDesk Local Support Agent")
    parser.add_argument("-q", "--question", type=str, help="Ask a single question and exit.")
    parser.add_argument("--samples", action="store_true", help="Run every question in data/sample_questions.json")
    parser.add_argument("--offline-demo", action="store_true", help="Use hashing embedder + mock generator (no model download).")
    parser.add_argument("--no-cache", action="store_true", help="Disable the query cache for this run.")
    args = parser.parse_args()

    print("Building retriever (loading knowledge base + embedding model)...")
    load_start = time.perf_counter()
    retriever = Retriever.build(offline=args.offline_demo)
    print(f"Retriever ready in {(time.perf_counter() - load_start) * 1000:.0f} ms.")

    print("Loading generation model...")
    gen_load_start = time.perf_counter()
    app, model_name = build_graph(retriever, offline=args.offline_demo)
    print(f"Generation model '{model_name}' ready in {(time.perf_counter() - gen_load_start) * 1000:.0f} ms.")

    cache = QueryCache(QUERY_CACHE_PATH)
    use_cache = not args.no_cache

    if args.question:
        response = run_question(app, cache, args.question, use_cache)
        print_response(args.question, response)
        return

    if args.samples:
        samples = json.load(open("data/sample_questions.json"))["questions"]
        for item in samples:
            response = run_question(app, cache, item["question"], use_cache)
            print_response(item["question"], response)
        return

    print("\nOrbitDesk Support Agent - interactive mode. Type 'exit' to quit.\n")
    previous_question = None
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question or question.lower() in {"exit", "quit"}:
            break
        response = run_question(app, cache, question, use_cache, previous_question=previous_question)
        print_response(question, response)
        previous_question = question


if __name__ == "__main__":
    main()
