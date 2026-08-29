"""PDF processing: extraction (Phase 2), clause-aware chunking (Phase 3),
and evidence highlighting (Phase 10)."""

from .clause_chunker import (
    Clause,
    MAX_CHUNK_CHARS,
    build_clauses,
    get_clause_summary,
    split_oversized_text,
)
from .extractor import (
    DocumentData,
    InvalidPDFError,
    LOW_TEXT_CHAR_THRESHOLD,
    PageData,
    PDFProcessingError,
    clean_text,
    compute_document_id,
    determine_extraction_status,
    extract_pdf,
    get_extraction_summary,
    preview_text,
    validate_pdf,
)
from .highlighter import HighlightResult, locate_evidence_on_page, render_page_image

__all__ = [
    "Clause",
    "DocumentData",
    "HighlightResult",
    "InvalidPDFError",
    "LOW_TEXT_CHAR_THRESHOLD",
    "MAX_CHUNK_CHARS",
    "PageData",
    "PDFProcessingError",
    "build_clauses",
    "clean_text",
    "compute_document_id",
    "determine_extraction_status",
    "extract_pdf",
    "get_clause_summary",
    "get_extraction_summary",
    "locate_evidence_on_page",
    "preview_text",
    "render_page_image",
    "split_oversized_text",
    "validate_pdf",
]
