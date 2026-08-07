"""
Chunks SourceDocuments for retrieval.

Strategy: split on Markdown headings first (so a chunk never straddles two
unrelated sections), then hard-wrap any section that is still longer than
`chunk_size_chars` with a character overlap. This keeps chunks topically
coherent (important for the "multi-document retrieval" test case, where we
want KB-003's timezone section and KB-004's troubleshooting section to come
back as two distinct, clean chunks) while still bounding chunk size for the
embedding model's context window.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from retrieval.loader import SourceDocument

HEADING_RE = re.compile(r"\n(?=#{1,3} )")


@dataclass
class Chunk:
    chunk_id: str
    source_id: str
    title: str
    doc_type: str
    is_superseded: bool
    text: str


def _split_section(section: str, size: int, overlap: int) -> list[str]:
    section = section.strip()
    if len(section) <= size:
        return [section] if section else []
    pieces = []
    start = 0
    while start < len(section):
        end = min(start + size, len(section))
        pieces.append(section[start:end].strip())
        if end == len(section):
            break
        start = end - overlap
    return [p for p in pieces if p]


def chunk_document(doc: SourceDocument, chunk_size: int, overlap: int) -> list[Chunk]:
    sections = HEADING_RE.split(doc.text) if doc.doc_type == "knowledge_base" else [doc.text]
    chunks: list[Chunk] = []
    for section_idx, section in enumerate(sections):
        for piece_idx, piece in enumerate(_split_section(section, chunk_size, overlap)):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.source_id}::{section_idx}.{piece_idx}",
                    source_id=doc.source_id,
                    title=doc.title,
                    doc_type=doc.doc_type,
                    is_superseded=doc.is_superseded,
                    text=piece,
                )
            )
    return chunks


def chunk_documents(docs: list[SourceDocument], chunk_size: int, overlap: int) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc, chunk_size, overlap))
    return all_chunks
