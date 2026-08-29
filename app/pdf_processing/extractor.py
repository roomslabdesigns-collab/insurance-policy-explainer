"""
Phase 2 — PDF text extraction with page-level traceability.

This module's only responsibility is: given a policy PDF, produce faithful,
page-numbered text plus enough metadata to relocate and highlight that text
in the original file later (Phase 10). It does NOT chunk by clause, embed,
or talk to an LLM — see app/rag/ (later phases) for that.

Text fidelity rule: we only ever normalize whitespace. We never rewrite,
summarize, or strip characters from policy wording, because every later
answer's trustworthiness depends on the stored text matching the source
document exactly.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Union

import fitz  # PyMuPDF

PathLike = Union[str, Path]

# Fewer than this many non-whitespace characters on a page is treated as
# "probably scanned/image-based" rather than "successfully extracted".
# This is a simple, explainable heuristic — not OCR detection.
LOW_TEXT_CHAR_THRESHOLD = 20


# --------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------

class PDFProcessingError(Exception):
    """Base class for all PDF processing failures in this module."""


class InvalidPDFError(PDFProcessingError):
    """
    Raised when a file is missing, not a PDF, empty, password-protected,
    or corrupted. Callers should catch this and show the message to the
    user instead of letting a raw PyMuPDF/OS exception surface.
    """


# --------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------

@dataclass
class PageData:
    """Everything we know about one extracted page."""

    page_number: int        # 1-based — always show this to users, never the
                             # internal 0-based fitz page index
    text: str                # safely cleaned text (whitespace-normalized)
    raw_text: str             # untouched, exactly as PyMuPDF extracted it —
                              # kept for later exact-text search/highlighting
    char_count: int          # len(raw_text.strip()) — used for status checks
    extraction_status: str   # "ok" | "low_text" | "empty"


@dataclass
class DocumentData:
    """Container for one processed PDF and all of its pages."""

    document_id: str                          # stable content hash (sha256[:16])
    filename: str
    file_path: str                            # absolute path — needed to reopen
                                               # the PDF later for highlighting
    total_pages: int
    pages: List[PageData] = field(default_factory=list)


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def validate_pdf(file_path: PathLike) -> Path:
    """
    Confirm that `file_path` points to a real, readable, non-empty,
    non-encrypted PDF with at least one page.

    Raises InvalidPDFError with a human-readable message on any problem.
    Never lets a raw PyMuPDF/OS exception escape to the caller — the whole
    point of this function is that upstream code (Streamlit, tests, the
    RAG pipeline) can show `str(exc)` directly to a user without a crash.
    """
    path = Path(file_path)

    if not path.exists():
        raise InvalidPDFError(f"File not found: {path}")
    if not path.is_file():
        raise InvalidPDFError(f"Path is not a file: {path}")
    if path.suffix.lower() != ".pdf":
        raise InvalidPDFError(
            f"File does not appear to be a PDF (extension '{path.suffix}'): {path}"
        )
    if path.stat().st_size == 0:
        raise InvalidPDFError(f"File is empty (0 bytes): {path}")

    # Actually try to open it — this is what catches corrupted files,
    # truncated downloads, and non-PDF content wearing a .pdf extension.
    try:
        doc = fitz.open(path)
    except Exception as exc:  # PyMuPDF can raise several exception types
        raise InvalidPDFError(
            f"File could not be opened as a PDF ({exc}): {path}"
        ) from exc

    try:
        if doc.needs_pass:
            raise InvalidPDFError(
                f"PDF is password-protected and cannot be processed: {path}"
            )
        if doc.page_count == 0:
            raise InvalidPDFError(f"PDF has no pages: {path}")
    finally:
        doc.close()

    return path


# --------------------------------------------------------------------------
# Text cleaning
# --------------------------------------------------------------------------

_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_MULTI_BLANK_LINE_RE = re.compile(r"\n{3,}")


def clean_text(raw_text: str) -> str:
    """
    Minimal, safe whitespace cleanup ONLY.

    Allowed:  collapsing repeated spaces/tabs, trimming trailing whitespace
              per line, collapsing 3+ blank lines down to one blank line.
    Never:    rewording, reordering, removing punctuation, or touching
              digits/clause numbers. The result must remain a faithful
              (just tidier) copy of the source text.
    """
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [_MULTI_SPACE_RE.sub(" ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = _MULTI_BLANK_LINE_RE.sub("\n\n", text)
    return text.strip()


def determine_extraction_status(
    raw_text: str, low_text_threshold: int = LOW_TEXT_CHAR_THRESHOLD
) -> str:
    """
    Classify a page's extraction result so downstream code never silently
    treats a scanned/image page as if it were successfully read.

    "empty"     — no extractable text at all (e.g. a pure image page)
    "low_text"  — a little text, but too little to trust (e.g. a page that
                  is mostly a scanned image with one caption line)
    "ok"        — normal text page
    """
    stripped = raw_text.strip()
    if len(stripped) == 0:
        return "empty"
    if len(stripped) < low_text_threshold:
        return "low_text"
    return "ok"


# --------------------------------------------------------------------------
# Document ID
# --------------------------------------------------------------------------

def compute_document_id(file_path: PathLike) -> str:
    """
    Content-based document ID: sha256 of the file's bytes, truncated to 16
    hex characters.

    Hashing the *content* (not the filename) means the same policy file
    always gets the same ID even if it's renamed or re-uploaded, while two
    genuinely different files never collide. Read in 64KB chunks so we
    never hold the whole file in memory just to fingerprint it.
    """
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()[:16]


# --------------------------------------------------------------------------
# Main extraction
# --------------------------------------------------------------------------

def extract_pdf(file_path: PathLike) -> DocumentData:
    """
    Validate and extract a policy PDF page by page.

    Memory behavior: pages are processed one at a time and the fitz
    Document is opened via a `with` block, which guarantees `close()` is
    called even if extraction raises partway through — important on an
    8 GB machine when someone uploads a large policy.
    """
    path = validate_pdf(file_path)
    document_id = compute_document_id(path)

    pages: List[PageData] = []
    with fitz.open(path) as doc:
        total_pages = doc.page_count
        for index in range(total_pages):
            page = doc.load_page(index)
            raw_text = page.get_text("text")
            pages.append(
                PageData(
                    page_number=index + 1,  # 1-based for humans; fitz's own
                                             # index above is 0-based
                    text=clean_text(raw_text),
                    raw_text=raw_text,
                    char_count=len(raw_text.strip()),
                    extraction_status=determine_extraction_status(raw_text),
                )
            )
            page = None  # drop the reference before moving to the next page

    return DocumentData(
        document_id=document_id,
        filename=path.name,
        file_path=str(path.resolve()),
        total_pages=total_pages,
        pages=pages,
    )


# --------------------------------------------------------------------------
# Reporting helpers
# --------------------------------------------------------------------------

def get_extraction_summary(document: DocumentData) -> dict:
    """Aggregate per-page statuses into a summary usable by tests, logs, or the UI."""
    ok_pages = [p for p in document.pages if p.extraction_status == "ok"]
    low_pages = [p for p in document.pages if p.extraction_status == "low_text"]
    empty_pages = [p for p in document.pages if p.extraction_status == "empty"]
    problem_pages = sorted(p.page_number for p in low_pages + empty_pages)

    return {
        "document_id": document.document_id,
        "filename": document.filename,
        "total_pages": document.total_pages,
        "pages_ok": len(ok_pages),
        "pages_low_text": len(low_pages),
        "pages_empty": len(empty_pages),
        "problem_pages": problem_pages,
    }


def preview_text(text: str, max_chars: int = 280) -> str:
    """Short, safe preview for logs/terminal output — never dumps a full page."""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " …"
