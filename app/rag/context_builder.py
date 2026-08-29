"""
Phase 6 — Compact evidence context builder.

Turns a list of SearchResult chunks (already retrieved AND deduplicated by
vector_store.search_policy -- see Phase 4) into the short POLICY EVIDENCE
block sent to the LLM. This is the token-efficiency layer: only a handful
of short, labeled clauses ever reach the model -- never whole sections,
never the raw PDF, never full conversation history.
"""

from __future__ import annotations

from typing import List

from .. import config
from .vector_store import SearchResult


def build_evidence_context(
    results: List[SearchResult],
    max_chunks: int = config.DEFAULT_CONTEXT_CHUNKS,
    max_chars_per_chunk: int = config.MAX_EVIDENCE_CHARS_PER_CHUNK,
) -> str:
    """
    Format up to `max_chunks` retrieved clauses into a compact, labeled
    block, one clause per line: "[Clause X, Page Y] <text>".

    No further deduplication happens here -- search_policy already removes
    exact-duplicate chunk text before this function ever sees the results,
    so re-deduping here would just be repeating Phase 4's logic. Each
    clause is truncated defensively at max_chars_per_chunk; Phase 3's
    clause-aware chunking should already keep clauses well under this, so
    truncation should rarely trigger in practice.
    """
    if not results:
        return ""

    lines = []
    for r in results[:max_chunks]:
        clause_label = r.clause_number or "General"
        text = r.text
        if len(text) > max_chars_per_chunk:
            text = text[:max_chars_per_chunk].rstrip() + "..."
        lines.append(f"[Clause {clause_label}, Page {r.page_number}] {text}")
    return "\n".join(lines)
