"""
Phase 8 — Application-controlled citation resolution and quote validation.

The LLM never authors citation metadata. It only ever picks a short
evidence_id label (E1, E2, ...) from the set actually supplied to it in
the prompt; this module is the ONLY place a Citation object gets built,
always from the real, already-retrieved SearchResult behind that label --
never from anything the LLM wrote.

Kept dependency-free at runtime (SearchResult is only used as a type hint,
guarded by TYPE_CHECKING) so this module can never be part of an import
cycle with app.rag.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from ..rag.vector_store import SearchResult


@dataclass
class Citation:
    """
    Fully application-resolved citation. Every field traces back to
    Phase 2/3's stored metadata or Phase 4's retrieval score -- none of it
    can be LLM-generated.
    """

    chunk_id: str
    document_id: str
    policy_name: str
    policy_version: str
    clause_number: str
    section: str
    page_number: int
    pages: List[int]
    full_text: str              # complete, untouched clause text (for "View Evidence" / Phase 10 highlighting)
    display_excerpt: str        # shorter excerpt shown in the answer card
    is_direct_quote: bool       # True only if display_excerpt was verified against full_text
    quote_match_type: str       # "exact" | "normalized" | "app_selected_excerpt"
    is_exclusion_section: bool
    contains_exclusion_language: bool


@dataclass
class EvidenceIdValidation:
    passed: bool
    reason: str
    result: Optional["SearchResult"]


def validate_evidence_id(
    evidence_id: Optional[str],
    label_map: Dict[str, "SearchResult"],
    expected_document_id: str,
) -> EvidenceIdValidation:
    """
    The evidence_id must: exist, belong to the evidence actually supplied
    to the LLM for THIS question, belong to the currently active policy
    document, and reference non-empty text. Fails closed on any doubt --
    an evidence_id absent entirely is fine (some statuses need none); an
    evidence_id present but unresolvable is not.
    """
    if evidence_id is None:
        return EvidenceIdValidation(True, "No evidence_id was cited (not required for this status).", None)

    result = label_map.get(evidence_id)
    if result is None:
        return EvidenceIdValidation(
            False, f"evidence_id {evidence_id!r} was not among the evidence supplied to the model.", None
        )

    if result.document_id != expected_document_id:
        return EvidenceIdValidation(
            False,
            f"evidence_id {evidence_id!r} belongs to a different policy document than the active one.",
            None,
        )

    if not result.text or not result.text.strip():
        return EvidenceIdValidation(False, f"evidence_id {evidence_id!r} refers to empty source text.", None)

    return EvidenceIdValidation(True, "Evidence ID validated.", result)


# A quoted fragment inside the model's answer, e.g. `"excluded unless..."` --
# at least 15 chars to avoid matching trivial/incidental punctuation.
_QUOTED_FRAGMENT_RE = re.compile(r'["“]([^"”]{15,300})["”]')


def _normalize_for_match(text: str) -> str:
    """Whitespace + common punctuation-variant normalization -- NOT fuzzy
    matching. Two strings must be identical after this to count as a match."""
    text = text.lower()
    text = re.sub(r"[‘’]", "'", text)
    text = re.sub(r"[“”]", '"', text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .,;:\"'")


def verify_quote(answer_text: str, source_text: str, max_excerpt_chars: int = 240) -> Tuple[str, bool, str]:
    """
    Exact match, then normalized match, on any quoted fragment the model
    included in its answer. If nothing verifies -- including the common
    case where the model never attempted a literal quote at all -- falls
    back to an excerpt selected DIRECTLY from the source text, so the
    displayed excerpt is always guaranteed real either way. Deliberately
    not fuzzy: a near-miss does not pass, by design (see module docstring
    in claim_checker.py for why loose matching is avoided throughout).
    """
    match = _QUOTED_FRAGMENT_RE.search(answer_text)
    if match:
        candidate = match.group(1).strip()
        if candidate and candidate in source_text:
            return candidate, True, "exact"
        if candidate and _normalize_for_match(candidate) in _normalize_for_match(source_text):
            return candidate, True, "normalized"

    excerpt = (
        source_text if len(source_text) <= max_excerpt_chars
        else source_text[:max_excerpt_chars].rstrip() + "..."
    )
    return excerpt, False, "app_selected_excerpt"


def build_citation(
    result: "SearchResult", answer_text: str, policy_name: str, policy_version: str
) -> Citation:
    """The single place a Citation is ever constructed."""
    display_excerpt, is_direct_quote, match_type = verify_quote(answer_text, result.text)
    return Citation(
        chunk_id=result.chunk_id,
        document_id=result.document_id,
        policy_name=policy_name,
        policy_version=policy_version,
        clause_number=result.clause_number,
        section=result.section,
        page_number=result.page_number,
        pages=result.pages,
        full_text=result.text,
        display_excerpt=display_excerpt,
        is_direct_quote=is_direct_quote,
        quote_match_type=match_type,
        is_exclusion_section=result.is_exclusion_section,
        contains_exclusion_language=result.contains_exclusion_language,
    )
