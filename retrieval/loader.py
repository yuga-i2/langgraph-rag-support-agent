"""
Loads the two source materials the assignment provides:

1. Markdown knowledge-base documents (with YAML front matter: document_id,
   title, status, ...).
2. resolved_cases.json (previously resolved / superseded support cases).

Both are normalised into a single `SourceDocument` shape so the chunker and
retriever don't need to know where a piece of text came from.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

FRONT_MATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)


@dataclass
class SourceDocument:
    source_id: str          # "KB-003" or "CASE-1041"
    title: str
    text: str                # plain text body used for chunking
    doc_type: str            # "knowledge_base" | "resolved_case"
    is_superseded: bool = False
    metadata: Dict[str, str] = field(default_factory=dict)


def _parse_front_matter(raw: str) -> tuple[Dict[str, str], str]:
    match = FRONT_MATTER_RE.match(raw)
    if not match:
        return {}, raw
    header_block, body = match.groups()
    meta: Dict[str, str] = {}
    for line in header_block.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip().strip("[]").strip()
    return meta, body.strip()


def load_knowledge_base(kb_dir: Path) -> List[SourceDocument]:
    docs: List[SourceDocument] = []
    for md_path in sorted(kb_dir.glob("*.md")):
        raw = md_path.read_text(encoding="utf-8")
        meta, body = _parse_front_matter(raw)
        docs.append(
            SourceDocument(
                source_id=meta.get("document_id", md_path.stem),
                title=meta.get("title", md_path.stem),
                text=body,
                doc_type="knowledge_base",
                is_superseded=meta.get("status", "current") == "superseded",
                metadata=meta,
            )
        )
    return docs


def _case_to_text(case: dict) -> str:
    lines = [f"Title: {case['title']}"]
    if case.get("symptoms"):
        lines.append("Symptoms: " + "; ".join(case["symptoms"]))
    if case.get("resolution"):
        lines.append("Resolution steps: " + "; ".join(case["resolution"]))
    if case.get("important_limit"):
        lines.append("Important limit: " + case["important_limit"])
    if case.get("superseded_reason"):
        lines.append("Superseded because: " + case["superseded_reason"])
    lines.append(f"Status: {case['status']}")
    return "\n".join(lines)


def load_resolved_cases(path: Path) -> List[SourceDocument]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    docs: List[SourceDocument] = []
    for case in payload.get("cases", []):
        docs.append(
            SourceDocument(
                source_id=case["case_id"],
                title=case["title"],
                text=_case_to_text(case),
                doc_type="resolved_case",
                is_superseded=case.get("status") == "superseded",
                metadata={
                    "status": case.get("status", ""),
                    "source_documents": ",".join(case.get("source_documents", [])),
                },
            )
        )
    return docs


def load_all_documents(kb_dir: Path, resolved_cases_path: Path) -> List[SourceDocument]:
    return load_knowledge_base(kb_dir) + load_resolved_cases(resolved_cases_path)
