"""
Phase 8 — Status-level guardrails and the related-exclusion check.

These rules cross-check the LLM's DECLARED status against Phase 3's
independently-computed clause metadata (is_exclusion_section,
contains_exclusion_language) -- signals the LLM never sees or controls,
which is exactly what makes them useful as a check ON the LLM rather than
just another thing it could talk its way around. This is the concrete fix
for the exact failure Phase 7 documented (a Covered/Excluded mislabeling)
without relying on further prompt tuning, which Phase 7 showed can trade
one failure mode for a worse one.

Deliberately simple, deterministic, and imperfect: these keyword/metadata
signals can miss a real exclusion phrased unusually, or flag a coverage
clause that happens to mention "not covered" in passing. When in doubt,
every rule here resolves toward the safer status (Insufficient Evidence)
rather than trusting the more "confident" one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

from ..utils.response_parser import STATUS_COVERED, STATUS_EXCLUDED, STATUS_INSUFFICIENT
from .citation_validator import Citation

if TYPE_CHECKING:
    from ..rag.vector_store import SearchResult


@dataclass
class GuardrailOutcome:
    status: str          # possibly downgraded from the input status
    note: Optional[str]  # human-readable explanation, or None if nothing changed
    downgraded: bool


def apply_status_guardrails(status: str, citation: Optional[Citation]) -> GuardrailOutcome:
    """
    Rule 2 (Explicit Exclusion requires explicit evidence) and Rule 3
    (Covered requires positive evidence), enforced via metadata computed
    independently back in Phase 3 -- not via trusting the LLM's own
    judgment about what its cited text says.
    """
    if citation is None:
        return GuardrailOutcome(status, None, False)

    has_exclusion_signal = citation.is_exclusion_section or citation.contains_exclusion_language

    if status == STATUS_EXCLUDED and not has_exclusion_signal:
        return GuardrailOutcome(
            STATUS_INSUFFICIENT,
            (
                f"Downgraded from 'Explicitly Excluded': the cited clause "
                f"({citation.clause_number or 'preamble'}) has no exclusion language or "
                "exclusion-section metadata to confirm this."
            ),
            True,
        )

    if status == STATUS_COVERED and citation.is_exclusion_section:
        return GuardrailOutcome(
            STATUS_INSUFFICIENT,
            (
                f"Downgraded from 'Covered': the cited clause ({citation.clause_number or 'preamble'}) "
                "is from an exclusions section, which is not positive evidence of coverage."
            ),
            True,
        )

    return GuardrailOutcome(status, None, False)


def check_related_exclusions(
    status: str, cited_chunk_id: Optional[str], other_results: List["SearchResult"]
) -> Optional[str]:
    """
    Section 8's related-exclusion check: if the final status is Covered but
    another chunk retrieved ALONGSIDE the cited one also carries exclusion
    signals, surface it instead of silently dropping it. Non-blocking: this
    augments the reliability note rather than rejecting the answer, since
    the co-retrieved exclusion may genuinely not apply to this question.
    """
    if status != STATUS_COVERED:
        return None

    flagged = [
        r for r in other_results
        if r.chunk_id != cited_chunk_id and (r.is_exclusion_section or r.contains_exclusion_language)
    ]
    if not flagged:
        return None

    clause_list = ", ".join(r.clause_number or "preamble" for r in flagged)
    return (
        f"Note: related exclusion language was also found in Clause {clause_list} among the "
        "retrieved evidence -- review it before relying on this answer."
    )


def get_confidence_label(status: str, validation_passed: bool, citation: Optional[Citation]) -> str:
    """
    Section 10: evidence-quality labels instead of a fabricated confidence
    percentage. Reflects how much of the pipeline actually verified
    something, not how fluent the LLM's wording is.
    """
    if not validation_passed or citation is None:
        return "Insufficient Evidence"
    if status in (STATUS_COVERED, STATUS_EXCLUDED):
        return "Verified Evidence"
    return "Limited Evidence"
