"""
Renders diagram/graph_diagram.png - a static picture of the LangGraph
workflow, used in the README and as the PNG/JPG submission requirement.

Uses matplotlib with plain rectangles/arrows (no graphviz dependency,
so it renders identically on any machine with the requirements.txt
installed).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as MplPath

NODE_STYLE = dict(boxstyle="round,pad=0.35", linewidth=1.4)
COLORS = {
    "entry": "#e8eef9",
    "deterministic": "#dff0d8",
    "model": "#fdebd0",
    "terminal": "#f5d5d5",
    "sink": "#e6e6e6",
}

# (id, label, x, y, w, h, kind)
NODES = [
    ("query_rewrite", "Query Rewrite\n(follow-up detection)", 0, 12.4, 3.4, 1.1, "deterministic"),
    ("triage", "Triage\n(rule-based classifier)", 0, 10.6, 3.4, 1.1, "deterministic"),
    ("out_of_scope", "Out of Scope\nHandler", -5.6, 8.0, 3.0, 1.0, "terminal"),
    ("clarification", "Clarification\nHandler", -1.9, 8.0, 3.0, 1.0, "terminal"),
    ("retrieval", "Retrieval\n(FAISS + BM25 hybrid)", 3.0, 8.0, 3.4, 1.0, "deterministic"),
    ("confidence_gate", "Confidence Gate", 3.0, 5.9, 3.4, 1.0, "deterministic"),
    ("generation", "Response Generation\n(local HF LLM)", 3.0, 3.8, 3.4, 1.0, "model"),
    ("verification", "Verification\n(schema + evidence + confidence)", 3.0, 1.7, 3.6, 1.0, "deterministic"),
    ("prepare_retry", "Prepare Retry\n(retry_count += 1, max 1)", 8.0, 3.8, 3.2, 1.0, "deterministic"),
    ("safe_failure", "Safe Failure", 0.4, -0.9, 2.6, 0.9, "terminal"),
    ("finalize", "Finalize", 3.4, -0.9, 2.6, 0.9, "terminal"),
    ("assemble", "Assemble Final\nResponse (JSON)", -2.0, -3.2, 3.4, 1.0, "sink"),
]

EDGES = [
    ("query_rewrite", "triage", None),
    ("triage", "out_of_scope", "out_of_scope"),
    ("triage", "clarification", "requires_clarification"),
    ("triage", "retrieval", "answerable /\nrequires_escalation"),
    ("retrieval", "confidence_gate", None),
    ("confidence_gate", "clarification", "low confidence"),
    ("confidence_gate", "generation", "sufficient confidence"),
    ("generation", "verification", None),
    ("verification", "finalize", "passed"),
    ("verification", "prepare_retry", "failed, retries left"),
    ("verification", "safe_failure", "failed, no retries left"),
    ("prepare_retry", "generation", "loop back"),
    ("out_of_scope", "assemble", None),
    ("clarification", "assemble", None),
    ("safe_failure", "assemble", None),
    ("finalize", "assemble", None),
]

pos = {n[0]: (n[2] + n[4] / 2, n[3] + n[5] / 2) for n in NODES}
box = {n[0]: n for n in NODES}


def draw():
    fig, ax = plt.subplots(figsize=(14, 15.5))
    ax.set_xlim(-7.5, 12.5)
    ax.set_ylim(-4.5, 14.0)
    ax.axis("off")
    ax.set_title(
        "OrbitDesk Support Agent - LangGraph Workflow",
        fontsize=16, fontweight="bold", pad=14,
    )

    for node_id, label, x, y, w, h, kind in NODES:
        fb = FancyBboxPatch(
            (x, y), w, h, **NODE_STYLE,
            facecolor=COLORS[kind], edgecolor="#333333", zorder=3,
        )
        ax.add_patch(fb)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                 fontsize=9.5, zorder=4, wrap=True)

    for src, dst, label in EDGES:
        _, _, sx, sy, sw, sh, _ = box[src]
        _, _, dx, dy, dw, dh, _ = box[dst]
        start = (sx + sw / 2, sy)
        end = (dx + dw / 2, dy + dh)
        if sy < dy:  # generation <- prepare_retry loop-back goes upward on the right
            start = (sx + sw, sy + sh / 2)
            end = (dx + dw, dy + dh / 2)
        arrow = FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=14,
            connectionstyle="arc3,rad=0.08", color="#555555", linewidth=1.2, zorder=2,
        )
        ax.add_patch(arrow)
        if label:
            mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
            ax.text(mx, my, label, fontsize=7.6, color="#444444",
                     ha="center", va="center",
                     bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85), zorder=5)

    legend_items = [
        ("deterministic", "Deterministic code (rules / retrieval / checks)"),
        ("model", "Local HF model call"),
        ("terminal", "Terminal handler (short-circuit)"),
        ("sink", "Final response assembly"),
    ]
    for i, (kind, text) in enumerate(legend_items):
        ly = -4.3
        lx = -7.5 + i * 5.2
        ax.add_patch(FancyBboxPatch((lx, ly), 0.35, 0.28, boxstyle="round,pad=0.05",
                                     facecolor=COLORS[kind], edgecolor="#333333"))
        ax.text(lx + 0.5, ly + 0.14, text, fontsize=8, va="center")

    fig.tight_layout()
    fig.savefig("diagram/graph_diagram.png", dpi=200, bbox_inches="tight")
    print("Saved diagram/graph_diagram.png")


if __name__ == "__main__":
    draw()
