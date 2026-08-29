"""
Phase 3 — Clause-aware chunking.

Turns the page-level text from extractor.py into structured chunks that
respect the document's real shape: Section -> Clause -> Sub-clause. This is
deliberately NOT fixed-size chunking — a clause about dental exclusions
should stay one chunk regardless of whether it's 40 characters or 400,
because that's the unit a citation ("Clause 4.2(c), Page 17") refers to.

Nothing here touches embeddings, FAISS, or the LLM — see app/rag/ (later
phases) for that. This module's only job is: extracted pages in, clause
chunks with full metadata out.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from .extractor import DocumentData

# A clause is split into multiple parts if its flattened text exceeds this
# many characters. Chosen to comfortably fit the 256-token context of
# all-MiniLM-L6-v2 (Phase 4) with room to spare, while staying well above
# the length of a typical single clause.
MAX_CHUNK_CHARS = 800

# Simple, explainable keyword signals — not authoritative on their own,
# but useful metadata for Phase 6 retrieval ("also look for related
# exclusions/exceptions").
_EXCLUSION_SECTION_RE = re.compile(r"exclu|exception", re.IGNORECASE)
_EXCLUSION_KEYWORDS = (
    "excluded", "exclusion", "not covered", "shall not", "will not be",
    "does not cover", "no claim shall", "not payable", "not applicable",
    "except", "unless",
)

# --------------------------------------------------------------------------
# Structural line patterns
# --------------------------------------------------------------------------

# Explicit "SECTION 4: EXCLUSIONS" style headings.
_SECTION_RE = re.compile(r"^SECTION\s+\d+\b\s*[:\-]?\s*(.*)$", re.IGNORECASE)

# A short, mostly-uppercase standalone line — catches bare headings like
# "EXCLUSIONS" that don't use the word "SECTION". Deliberately conservative
# (short, letters+digits+basic punctuation only) to avoid mistaking a
# defined term or a shouty sentence for a heading.
_ALLCAPS_HEADING_RE = re.compile(r"^[A-Z0-9][A-Z0-9 ,&/\-]{2,79}$")

# Clause numbers REQUIRE a dot (3.1, 4.2.1, ...). Bare integers are
# deliberately excluded — insurance text is full of non-clause numbers
# ("48 months", "90 days") that would otherwise be misread as new clauses.
# An optional sub-id in parens directly after the number handles clauses
# written as "4.2(c) ..." in a single line.
_CLAUSE_RE = re.compile(
    r"^(\d+\.\d+(?:\.\d+)*)\s*(?:\(([a-zA-Z0-9]{1,3})\))?(?:\s+(.*))?$"
)

# A bare sub-clause marker on its own line, e.g. "(a) Routine dental
# check-ups are excluded." — attached to whichever numbered clause came
# before it (see `current_main_clause` in build_clauses).
_SUBCLAUSE_RE = re.compile(r"^\(([a-zA-Z0-9]{1,3})\)\s+(.*)$")


def _is_section_heading(line: str) -> bool:
    if _SECTION_RE.match(line):
        return True
    if _CLAUSE_RE.match(line) or _SUBCLAUSE_RE.match(line):
        return False
    if not any(ch.isalpha() for ch in line):
        return False
    word_count = len(line.split())
    if not (1 <= word_count <= 8):
        return False
    return bool(_ALLCAPS_HEADING_RE.match(line)) and line == line.upper()


# --------------------------------------------------------------------------
# Data structure
# --------------------------------------------------------------------------

@dataclass
class Clause:
    """One clause-aware chunk, ready for embedding (Phase 4) and citation."""

    chunk_id: str
    document_id: str
    policy_name: str
    policy_version: str

    section: str              # heading text in effect when this chunk started, "" if none yet
    clause_number: str        # e.g. "4.2(c)"; "" if no clause number detected (preamble text)
    page_number: int          # primary page shown in citations = min(pages)
    pages: List[int]          # every page this chunk's text was drawn from

    text: str
    char_count: int

    is_exclusion_section: bool          # heading text suggests an exclusions/exceptions section
    contains_exclusion_language: bool   # chunk text itself contains exclusion-style wording

    part_index: int           # 1-based; >1 only when a clause was split for length
    part_total: int


def _normalize_clause_text(text: str) -> str:
    """Flatten PDF line-wrapping into one clean sentence-flow string."""
    return re.sub(r"\s+", " ", text).strip()


def split_oversized_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> List[str]:
    """
    Split long clause text on sentence boundaries, never mid-word, so a
    citation always points at whole sentences of policy language.
    """
    if len(text) <= max_chars:
        return [text]

    sentences = re.split(r"(?<=[.;])\s+", text)
    parts: List[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if current and len(candidate) > max_chars:
            parts.append(current.strip())
            current = sentence
        else:
            current = candidate
    if current:
        parts.append(current.strip())

    # Safety net: a single "sentence" longer than max_chars (no punctuation
    # at all) still gets hard-split so nothing downstream ever sees a chunk
    # larger than max_chars.
    final_parts: List[str] = []
    for part in parts:
        if len(part) <= max_chars:
            final_parts.append(part)
        else:
            final_parts.extend(
                part[i : i + max_chars].strip() for i in range(0, len(part), max_chars)
            )
    return final_parts


# --------------------------------------------------------------------------
# Main chunking
# --------------------------------------------------------------------------

def build_clauses(
    document: DocumentData,
    policy_name: str,
    policy_version: str,
) -> List[Clause]:
    """
    Walk the document's pages in order and emit clause-aware chunks.

    Pages with extraction_status != "ok" are skipped (there's no reliable
    text to parse), but this does NOT reset any clause currently being
    accumulated — if a clause genuinely continues past a scanned/blank
    page, its later continuation is still attached correctly.
    """
    clauses: List[Clause] = []
    seq = 0

    current_section = ""
    current_main_clause = ""  # resolves bare "(a)" lines to e.g. "4.2(a)"

    buffer_lines: List[str] = []
    buffer_clause_number = ""
    buffer_pages: List[int] = []

    def flush() -> None:
        nonlocal buffer_lines, buffer_clause_number, buffer_pages, seq

        if not buffer_lines:
            buffer_clause_number = ""
            buffer_pages = []
            return

        full_text = _normalize_clause_text(" ".join(buffer_lines))
        if not full_text:
            buffer_lines = []
            buffer_clause_number = ""
            buffer_pages = []
            return

        pages_sorted = sorted(set(buffer_pages))
        parts = split_oversized_text(full_text)
        part_total = len(parts)

        for part_index, part_text in enumerate(parts, start=1):
            seq += 1
            clauses.append(
                Clause(
                    chunk_id=f"{document.document_id}-{seq:04d}",
                    document_id=document.document_id,
                    policy_name=policy_name,
                    policy_version=policy_version,
                    section=current_section,
                    clause_number=buffer_clause_number,
                    page_number=pages_sorted[0] if pages_sorted else 0,
                    pages=pages_sorted,
                    text=part_text,
                    char_count=len(part_text),
                    is_exclusion_section=bool(_EXCLUSION_SECTION_RE.search(current_section)),
                    contains_exclusion_language=any(
                        kw in part_text.lower() for kw in _EXCLUSION_KEYWORDS
                    ),
                    part_index=part_index,
                    part_total=part_total,
                )
            )

        buffer_lines = []
        buffer_clause_number = ""
        buffer_pages = []

    for page in document.pages:
        if page.extraction_status != "ok":
            continue

        for raw_line in page.text.split("\n"):
            line = raw_line.strip()
            if not line:
                continue

            if _is_section_heading(line):
                flush()
                current_section = line
                continue

            clause_match = _CLAUSE_RE.match(line)
            if clause_match:
                flush()
                number, sub, rest_text = clause_match.groups()
                current_main_clause = number
                buffer_clause_number = f"{number}({sub})" if sub else number
                buffer_pages.append(page.page_number)
                if rest_text:
                    buffer_lines.append(rest_text)
                continue

            subclause_match = _SUBCLAUSE_RE.match(line)
            if subclause_match:
                flush()
                sub, rest_text = subclause_match.groups()
                buffer_clause_number = (
                    f"{current_main_clause}({sub})" if current_main_clause else f"({sub})"
                )
                buffer_pages.append(page.page_number)
                if rest_text:
                    buffer_lines.append(rest_text)
                continue

            # Plain continuation line -> belongs to whatever clause/preamble
            # is currently open.
            buffer_pages.append(page.page_number)
            buffer_lines.append(line)

    flush()  # final clause in the document

    return clauses


# --------------------------------------------------------------------------
# Reporting helper
# --------------------------------------------------------------------------

def get_clause_summary(clauses: List[Clause]) -> dict:
    """Aggregate stats used by tests, logs, or a future admin view."""
    numbered = [c for c in clauses if c.clause_number]
    preamble = [c for c in clauses if not c.clause_number]
    exclusion_flagged = [c for c in clauses if c.contains_exclusion_language]
    split_clauses = [c for c in clauses if c.part_total > 1]

    return {
        "total_chunks": len(clauses),
        "numbered_clauses": len(numbered),
        "preamble_chunks": len(preamble),
        "chunks_with_exclusion_language": len(exclusion_flagged),
        "clauses_split_for_length": len({c.clause_number for c in split_clauses}),
        "sections_detected": sorted({c.section for c in clauses if c.section}),
    }
