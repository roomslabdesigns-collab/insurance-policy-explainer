"""
Phase 10 — Locating and highlighting verified evidence in the original PDF.

Nothing here is LLM-controlled: every highlight starts from the page
number and exact clause text already resolved by Phase 8's
citation_validator (itself sourced from Phase 2/3's extraction). This
module's only job is finding WHERE that already-verified text sits on the
page, using PyMuPDF's own text search -- never semantic similarity, never
an LLM guess, and never a fabricated box when nothing is actually found.

The original PDF is NEVER modified: every function here opens a fresh
fitz.Document from the source path, reads (and, for rendering, draws
highlight annotations onto) that in-memory copy, then closes it -- nothing
is ever written back to the file at `pdf_path`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import fitz  # PyMuPDF

# Match levels, in the order they're attempted -- see locate_evidence_on_page.
MATCH_EXACT = "exact"
MATCH_SENTENCE = "sentence"
MATCH_EXCERPT = "excerpt"
MATCH_NOT_FOUND = "not_found"
MATCH_SOURCE_UNAVAILABLE = "source_unavailable"
MATCH_INVALID_PAGE = "invalid_page"

MIN_EXCERPT_CHARS = 15
EXCERPT_LENGTH = 60


@dataclass
class HighlightResult:
    """
    Everything the UI needs to render (or honestly fail to render) a
    highlighted page. `found=False` always means exactly that -- no
    fabricated box is ever returned; the caller must show a fallback.
    """

    found: bool
    match_level: str
    page_number: int
    matched_text: str
    quads: List["fitz.Quad"] = field(default_factory=list)
    error_message: str = ""


def _split_sentences(text: str) -> List[str]:
    """Split on sentence-ending punctuation, tolerant of however much
    whitespace (including line breaks) separates them in the stored text."""
    parts = re.split(r"(?<=[.;:])\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) >= MIN_EXCERPT_CHARS]


def locate_evidence_on_page(pdf_path: str, page_number: int, evidence_text: str) -> HighlightResult:
    """
    Tiered, deterministic search over ONE page:
      1. Exact match of the full evidence text.
      2. Sentence-level match -- handles a clause that doesn't match as
         one block (the PDF's own line/paragraph wrapping can put subtly
         different whitespace between sentences than our stored, flattened
         clause text), but whose individual sentences are still exact
         substrings of the page's text.
      3. A short excerpt from the start of the evidence -- still real,
         unmodified source text, just shorter and more likely to survive
         minor formatting differences than the full clause.
    Never fuzzy-matches, and never returns found=True without a real
    PyMuPDF-located quad backing it.
    """
    if not pdf_path or not Path(pdf_path).exists():
        return HighlightResult(
            False, MATCH_SOURCE_UNAVAILABLE, page_number, "",
            error_message="The original PDF file is no longer available at its saved location.",
        )

    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:  # PyMuPDF can raise several exception types for a bad file
        return HighlightResult(
            False, MATCH_SOURCE_UNAVAILABLE, page_number, "",
            error_message=f"Could not open the source PDF: {exc}",
        )

    try:
        if not (1 <= page_number <= doc.page_count):
            return HighlightResult(
                False, MATCH_INVALID_PAGE, page_number, "",
                error_message=f"Page {page_number} is outside this document's {doc.page_count} pages.",
            )

        page = doc.load_page(page_number - 1)  # fitz pages are 0-based internally

        # Tier 1: exact.
        quads = page.search_for(evidence_text, quads=True)
        if quads:
            return HighlightResult(True, MATCH_EXACT, page_number, evidence_text, quads=list(quads))

        # Tier 2: sentence-level.
        matched_quads: List["fitz.Quad"] = []
        matched_sentences: List[str] = []
        for sentence in _split_sentences(evidence_text):
            found = page.search_for(sentence, quads=True)
            if found:
                matched_quads.extend(found)
                matched_sentences.append(sentence)
        if matched_quads:
            return HighlightResult(
                True, MATCH_SENTENCE, page_number, " / ".join(matched_sentences), quads=matched_quads
            )

        # Tier 3: short excerpt from the start of the evidence.
        excerpt = evidence_text.strip()[:EXCERPT_LENGTH].rsplit(" ", 1)[0]
        if len(excerpt) >= MIN_EXCERPT_CHARS:
            found = page.search_for(excerpt, quads=True)
            if found:
                return HighlightResult(True, MATCH_EXCERPT, page_number, excerpt, quads=list(found))

        return HighlightResult(
            False, MATCH_NOT_FOUND, page_number, "",
            error_message="The evidence text could not be located precisely on this page.",
        )
    finally:
        doc.close()


def render_page_image(
    pdf_path: str, page_number: int, quads: Optional[List["fitz.Quad"]] = None, zoom: float = 2.0
) -> bytes:
    """
    Renders one page as PNG bytes, with highlight annotations (if any)
    drawn on a freshly-opened, in-memory copy of the page. `pdf_path` is
    only ever read here, never written to -- closing without saving means
    the user's original file is guaranteed unchanged regardless of how
    many times this is called.
    """
    doc = fitz.open(pdf_path)
    try:
        if not (1 <= page_number <= doc.page_count):
            raise ValueError(f"Page {page_number} is outside this document's {doc.page_count} pages.")
        page = doc.load_page(page_number - 1)
        for quad in quads or []:
            annot = page.add_highlight_annot(quad)
            annot.update()
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        return pix.tobytes("png")
    finally:
        doc.close()
