"""
Small logging helper shared by all nodes.

Prints a human-readable trace to stdout (matching the format requested in
the assignment: "Running Triage", "Retrieved 4 chunks", ...) AND appends the
same lines to state["execution_log"] so the Streamlit UI / JSON output can
show the full trace for explainability, without re-parsing stdout.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict, Iterator, List


def log_step(execution_log: List[str], message: str) -> None:
    print(message)
    execution_log.append(message)


@contextmanager
def timed_step(
    execution_log: List[str],
    node_trace: List[str],
    timings_ms: Dict[str, float],
    node_name: str,
    start_message: str,
) -> Iterator[None]:
    """Logs a node's start/end and records its wall-clock latency in state."""
    log_step(execution_log, start_message)
    node_trace.append(node_name)
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        timings_ms[node_name] = round(elapsed_ms, 2)
        log_step(execution_log, f"Finished {node_name} ({elapsed_ms:.0f} ms)")
