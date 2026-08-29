"""
Phase 8 — Trust layer between the LLM's structured response and the UI.

Everything here is deterministic Python logic reusing data already in
memory (Phase 2/3's stored metadata, Phase 4's retrieval results) -- no
additional model, no extra LLM call. See app.rag.answer_generator for how
these pieces are assembled into the full pipeline.
"""

from .citation_validator import (
    Citation,
    EvidenceIdValidation,
    build_citation,
    validate_evidence_id,
    verify_quote,
)
from .claim_checker import ClaimCheckResult, check_claim_support, extract_numbers
from .guardrails import (
    GuardrailOutcome,
    apply_status_guardrails,
    check_related_exclusions,
    get_confidence_label,
)

__all__ = [
    "Citation",
    "ClaimCheckResult",
    "EvidenceIdValidation",
    "GuardrailOutcome",
    "apply_status_guardrails",
    "build_citation",
    "check_claim_support",
    "check_related_exclusions",
    "extract_numbers",
    "get_confidence_label",
    "validate_evidence_id",
    "verify_quote",
]
