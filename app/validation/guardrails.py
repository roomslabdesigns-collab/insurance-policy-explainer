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
from typing import TYPE_CHECKING, List, Optional, Set

from ..utils.response_parser import STATUS_COVERED, STATUS_EXCLUDED, STATUS_INSUFFICIENT
from .citation_validator import Citation

if TYPE_CHECKING:
    from ..rag.vector_store import SearchResult

# Phase 12 (evaluation-driven): generic insurance boilerplate words that
# would coincidentally overlap between almost any question and almost any
# clause's exception condition ("claims", "policy", "covered"...), without
# actually indicating the question describes the SAME specific exception
# scenario. Excluded from the overlap check in _mentions_exception_scenario
# below to avoid false-positive downgrades. Stemmed to a 6-char prefix, the
# same crude stemming the comparison itself uses (see that function).
_GENERIC_OVERLAP_STOPWORDS: Set[str] = {
    "claim", "claims", "policy", "covere", "payabl", "insure", "treatm",
    "medica", "benefi", "sectio", "clause", "expens", "requir", "genera",
    "specif", "period", "provid", "applic",
}


@dataclass
class GuardrailOutcome:
    status: str          # possibly downgraded from the input status
    note: Optional[str]  # human-readable explanation, or None if nothing changed
    downgraded: bool


def _stemmed_significant_words(text: str, min_len: int = 5) -> Set[str]:
    """
    Crude stemming (fixed 6-char prefix) so "accident"/"accidental" count
    as the same signal without a real NLP dependency. `min_len` filters out
    short common words before stemming; the stoplist above filters out
    longer but still-generic insurance boilerplate afterward.
    """
    words = set()
    for raw in text.split():
        cleaned = raw.strip(".,;:!?()\"'").lower()
        if len(cleaned) < min_len:
            continue
        stem = cleaned[:6]
        if stem in _GENERIC_OVERLAP_STOPWORDS:
            continue
        words.add(stem)
    return words


def _mentions_exception_scenario(question: str, exception_condition_text: str) -> Set[str]:
    """Returns the shared specific terms (e.g. {'accide'}) if the question's
    own wording overlaps with the clause's exception condition, else empty."""
    return _stemmed_significant_words(question) & _stemmed_significant_words(exception_condition_text)


def apply_status_guardrails(
    status: str, citation: Optional[Citation], question: str = ""
) -> GuardrailOutcome:
    """
    Rule 2 (Explicit Exclusion requires explicit evidence), Rule 3 (Covered
    requires positive evidence), and Rule 4 (a conditional exception the
    question itself describes should not be overridden by the clause's
    general rule) -- all enforced via metadata computed independently back
    in Phase 3, not via trusting the LLM's own judgment about what its
    cited text says.

    Rule 4 is the direct fix for the exact pattern behind every
    Wrong-but-Confident result in the Phase 11 evaluation: a clause stating
    a general rule and an exception in one sentence (e.g. "excluded UNLESS
    required to treat an accidental injury"), where the LLM applied the
    general rule to a question that was specifically describing the
    exception's own scenario.
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

    if status == STATUS_EXCLUDED and citation.exception_condition_text:
        overlap = _mentions_exception_scenario(question, citation.exception_condition_text)
        if overlap:
            return GuardrailOutcome(
                STATUS_INSUFFICIENT,
                (
                    f"Downgraded from 'Explicitly Excluded': the cited clause "
                    f"({citation.clause_number or 'preamble'}) has a conditional exception "
                    f"({citation.exception_condition_text!r}) that the question appears to describe "
                    f"(shared terms: {', '.join(sorted(overlap))}) -- the exception may apply instead "
                    "of the general rule."
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
