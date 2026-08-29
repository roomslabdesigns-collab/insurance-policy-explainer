"""
Phase 7/8 — Structured, evidence-gated, guardrail-validated answer generation.

Pipeline: retrieve (Phase 4/5, unchanged) -> evidence sufficiency gate
(Python logic, run BEFORE any LLM call) -> compact labeled evidence context
-> local LLM (Phase 6's generate_completion, unchanged) -> parse structured
output -> 🛡️ Phase 8 trust layer (evidence-ID validation, quote
verification, status guardrails, claim checking, related-exclusion check)
-> a GroundedResponse the UI can render directly, plus a logged validation
event.

Core rule enforced here in CODE, not just in the prompt: the LLM is never
treated as the source of truth for clause numbers, page numbers, or even
its own STATUS choice. It only ever picks a short evidence label (E1, E2,
...); every other trust decision is made by app.validation using data the
LLM never sees or controls.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .. import config
from ..llm.client import LLMError, generate_completion
from ..storage.validation_logger import RetrievedEvidenceLogEntry, log_validation_event
from ..utils.response_parser import ALLOWED_LLM_STATUSES, STATUS_COVERED, STATUS_EXCLUDED, STATUS_INSUFFICIENT, parse_llm_response
from ..validation.citation_validator import Citation, build_citation, validate_evidence_id
from ..validation.claim_checker import check_claim_support
from ..validation.guardrails import apply_status_guardrails, check_related_exclusions, get_confidence_label
from .retrieval import retrieve
from .vector_store import PolicyIndex, SearchResult

logger = logging.getLogger(__name__)

# Assigned by this module's pre-LLM gate ONLY -- the LLM never sees or
# produces this value, so it can never be confused with the four
# LLM-classified statuses above.
STATUS_NO_EVIDENCE = "No Evidence Found"
ALL_STATUSES = ALLOWED_LLM_STATUSES | {STATUS_NO_EVIDENCE}

# Phase 8, section 9 — safe, application-generated fallback text, always
# kept separate from anything the LLM writes.
NO_EVIDENCE_MESSAGE = "I couldn't find this clearly addressed in the uploaded policy."
INSUFFICIENT_EVIDENCE_MESSAGE = (
    "The policy contains related information, but it doesn't provide enough evidence to "
    "answer this confidently."
)
VALIDATION_FAILED_MESSAGE = (
    "I found relevant policy information, but I couldn't verify the answer reliably. Please "
    "review the cited policy section directly."
)

# Compact grounding prompt: numbered/labeled evidence, an explicit closed
# set of statuses, and a fixed three-line output the parser knows how to
# read. Deliberately short -- every extra sentence here is tokens paid on
# every single question. NOTE: earlier wording that explicitly defined
# what "Covered" means was tried and reverted (see Phase 7 notes) after it
# caused the model to fabricate confident cross-clause reasoning on
# unrelated questions -- Phase 8's guardrails below catch the original,
# more benign mislabeling this simpler wording still produces sometimes.
STRUCTURED_SYSTEM_PROMPT = (
    "You are answering questions about an insurance policy using ONLY the numbered EVIDENCE "
    "below. Never use outside insurance knowledge or guess. Choose STATUS from exactly: "
    "Covered, Explicitly Excluded, Not Mentioned, Insufficient Evidence. Never say Explicitly "
    "Excluded unless the evidence states an exclusion outright -- silence means Not Mentioned. "
    "If evidence is incomplete or unclear, use Insufficient Evidence. Cite EVIDENCE_ID using "
    "only the labels shown (e.g. E1) -- never invent a label, clause number, or page number. "
    "Reply with EXACTLY these three lines, in this order, and nothing else -- all three lines "
    "are mandatory even when EVIDENCE_ID is NONE:\n"
    "STATUS: <one of the four statuses>\n"
    "EVIDENCE_ID: <the single best-supporting label, or NONE>\n"
    "ANSWER: <concise, plain-language answer, 1-3 sentences>"
)


@dataclass
class GroundedResponse:
    question: str
    status: str
    answer_text: str
    citation: Optional[Citation]
    confidence_label: str
    retrieved_results: List[SearchResult]
    validation_passed: bool
    reliability_note: str
    response_time_seconds: Optional[float] = None
    raw_llm_response: Optional[str] = None
    # Populated from the LLM server's own usage report (Phase 6) when a call was made;
    # None when the pre-LLM evidence gate rejected the question -- an honest "0 tokens used".
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None


def check_evidence_sufficiency(
    results: List[SearchResult], min_score: float = config.MIN_EVIDENCE_SCORE
) -> bool:
    """
    Cheap Python-level pre-filter, run BEFORE any LLM call -- see
    config.MIN_EVIDENCE_SCORE for why this threshold is a low floor rather
    than a final trust decision.
    """
    return bool(results) and results[0].score >= min_score


def build_labeled_evidence(
    results: List[SearchResult], max_chunks: int
) -> Tuple[str, Dict[str, SearchResult]]:
    """
    Format retrieved chunks as "[E1] <text>" -- no clause number or page
    number in the prompt at all, so the model has nothing to echo back
    incorrectly. `label_map` is how the app translates E1 back to real
    metadata after generation.
    """
    selected = results[:max_chunks]
    label_map: Dict[str, SearchResult] = {}
    lines = []
    for i, r in enumerate(selected, start=1):
        label = f"E{i}"
        label_map[label] = r
        text = r.text
        if len(text) > config.MAX_EVIDENCE_CHARS_PER_CHUNK:
            text = text[: config.MAX_EVIDENCE_CHARS_PER_CHUNK].rstrip() + "..."
        lines.append(f"[{label}] {text}")
    return "\n".join(lines), label_map


def build_structured_user_prompt(evidence_block: str, question: str) -> str:
    return f"EVIDENCE:\n{evidence_block}\n\nQUESTION:\n{question}"


def _log(
    policy_index: PolicyIndex,
    question: str,
    label_map: Dict[str, SearchResult],
    selected_evidence_id: Optional[str],
    status: str,
    validation_passed: bool,
    reliability_note: str,
    answer_text: str,
    response_time_seconds: Optional[float],
) -> None:
    evidence_entries = [
        RetrievedEvidenceLogEntry(
            label=label, clause_number=r.clause_number, page_number=r.page_number, score=r.score
        )
        for label, r in label_map.items()
    ]
    log_validation_event(
        document_id=policy_index.document_id,
        policy_name=policy_index.policy_name,
        question=question,
        retrieved_evidence=evidence_entries,
        selected_evidence_id=selected_evidence_id,
        status=status,
        validation_passed=validation_passed,
        reliability_note=reliability_note,
        answer_text=answer_text,
        response_time_seconds=response_time_seconds,
    )


def generate_grounded_response(
    policy_index: PolicyIndex,
    question: str,
    max_chunks: int = config.DEFAULT_CONTEXT_CHUNKS,
    min_score: float = config.MIN_EVIDENCE_SCORE,
) -> GroundedResponse:
    """
    The full pipeline for one question. Makes at most ONE LLM call (or
    zero, if the evidence gate rejects the question) -- no chained or
    speculative calls, per the token-efficiency requirement. Every path
    through this function ends in a validation log entry.
    """
    if not question or not question.strip():
        raise ValueError("Question must not be empty.")

    results = retrieve(policy_index, question, max_k=max_chunks)

    # --- Gate 1: evidence sufficiency (before any LLM call) ---
    if not check_evidence_sufficiency(results, min_score=min_score):
        top_score = results[0].score if results else None
        logger.info("Evidence gate rejected question=%r (top score=%s)", question, top_score)
        note = "No LLM call was made -- retrieved evidence did not meet the minimum relevance threshold."
        _log(policy_index, question, {}, None, STATUS_NO_EVIDENCE, True, note, NO_EVIDENCE_MESSAGE, None)
        return GroundedResponse(
            question=question,
            status=STATUS_NO_EVIDENCE,
            answer_text=NO_EVIDENCE_MESSAGE,
            citation=None,
            confidence_label=get_confidence_label(STATUS_NO_EVIDENCE, True, None),
            retrieved_results=results,
            validation_passed=True,
            reliability_note=note,
        )

    evidence_block, label_map = build_labeled_evidence(results, max_chunks=max_chunks)
    user_prompt = build_structured_user_prompt(evidence_block, question)

    try:
        llm_response = generate_completion(STRUCTURED_SYSTEM_PROMPT, user_prompt)
    except LLMError as exc:
        logger.warning("LLM call failed for question=%r: %s", question, exc)
        note = "LLM call failed -- see application logs."
        answer = f"The local LLM is currently unavailable ({exc}). Please try again."
        _log(policy_index, question, label_map, None, STATUS_INSUFFICIENT, False, note, answer, None)
        return GroundedResponse(
            question=question,
            status=STATUS_INSUFFICIENT,
            answer_text=answer,
            citation=None,
            confidence_label=get_confidence_label(STATUS_INSUFFICIENT, False, None),
            retrieved_results=results,
            validation_passed=False,
            reliability_note=note,
        )

    parsed = parse_llm_response(llm_response.text)

    def _reject(note: str) -> GroundedResponse:
        logger.warning(
            "Validation failed for question=%r: %s | raw_response=%r", question, note, llm_response.text
        )
        _log(
            policy_index, question, label_map, parsed.evidence_id, STATUS_INSUFFICIENT, False,
            note, VALIDATION_FAILED_MESSAGE, llm_response.response_time_seconds,
        )
        return GroundedResponse(
            question=question,
            status=STATUS_INSUFFICIENT,
            answer_text=VALIDATION_FAILED_MESSAGE,
            citation=None,
            confidence_label=get_confidence_label(STATUS_INSUFFICIENT, False, None),
            retrieved_results=results,
            validation_passed=False,
            reliability_note=note,
            response_time_seconds=llm_response.response_time_seconds,
            prompt_tokens=llm_response.prompt_tokens,
            completion_tokens=llm_response.completion_tokens,
            raw_llm_response=llm_response.text,
        )

    # --- Gate 2: response format ---
    if not parsed.is_valid_format:
        return _reject("Could not parse a valid status/answer from the model's response.")
    if parsed.status not in ALLOWED_LLM_STATUSES:
        return _reject(f"Model returned an unrecognized status: {parsed.status!r}.")
    if not parsed.answer_text.strip():
        return _reject("Model returned an empty answer.")

    # --- Gate 3: evidence ID validation (🛡️ Phase 8, section 3) ---
    id_check = validate_evidence_id(parsed.evidence_id, label_map, policy_index.document_id)
    if not id_check.passed:
        return _reject(id_check.reason)

    # Substantive Covered/Excluded claims require a citation -- enforced
    # here, not left to the prompt alone.
    if id_check.result is None and parsed.status in (STATUS_COVERED, STATUS_EXCLUDED):
        return _reject(f"Model gave status {parsed.status!r} without citing any evidence_id.")

    notes: List[str] = []
    citation: Optional[Citation] = None
    final_status = parsed.status
    status_was_downgraded = False

    if id_check.result is not None:
        citation = build_citation(id_check.result, parsed.answer_text, policy_index.policy_name, policy_index.policy_version)

        # --- Gate 4: claim-to-evidence check (🛡️ section 6) ---
        claim_check = check_claim_support(parsed.answer_text, citation.full_text)
        if claim_check.is_high_risk:
            note = (
                f"Downgraded: the answer contains number(s) not found in the cited evidence "
                f"({', '.join(claim_check.unsupported_numbers)}) -- possible unsupported claim."
            )
            _log(
                policy_index, question, label_map, parsed.evidence_id, STATUS_INSUFFICIENT, False,
                note, INSUFFICIENT_EVIDENCE_MESSAGE, llm_response.response_time_seconds,
            )
            return GroundedResponse(
                question=question,
                status=STATUS_INSUFFICIENT,
                answer_text=INSUFFICIENT_EVIDENCE_MESSAGE,
                citation=citation,
                confidence_label=get_confidence_label(STATUS_INSUFFICIENT, False, citation),
                retrieved_results=results,
                validation_passed=False,
                reliability_note=note,
                response_time_seconds=llm_response.response_time_seconds,
                prompt_tokens=llm_response.prompt_tokens,
                completion_tokens=llm_response.completion_tokens,
                raw_llm_response=llm_response.text,
            )

        # --- Gate 5: status guardrails (🛡️ section 7, rules 2 & 3) ---
        guardrail = apply_status_guardrails(final_status, citation, question)
        final_status = guardrail.status
        status_was_downgraded = guardrail.downgraded
        if guardrail.downgraded:
            notes.append(guardrail.note)

        # --- Gate 6: related-exclusion check (🛡️ section 8) ---
        related_note = check_related_exclusions(final_status, citation.chunk_id, results)
        if related_note:
            notes.append(related_note)

        if citation.quote_match_type != "app_selected_excerpt":
            notes.append(f"Quote verified against source text ({citation.quote_match_type} match).")

    # Only replace the LLM's own answer text when its underlying STATUS
    # claim was overturned -- an informational note (related exclusion,
    # quote-verification) alongside an UNCHANGED status keeps the original
    # answer, since nothing about it was actually invalidated.
    final_answer = INSUFFICIENT_EVIDENCE_MESSAGE if status_was_downgraded else parsed.answer_text
    reliability_note = " ".join(notes) if notes else "Validated."

    _log(
        policy_index, question, label_map, parsed.evidence_id, final_status, True,
        reliability_note, final_answer, llm_response.response_time_seconds,
    )

    return GroundedResponse(
        question=question,
        status=final_status,
        answer_text=final_answer,
        citation=citation,
        confidence_label=get_confidence_label(final_status, True, citation),
        retrieved_results=results,
        validation_passed=True,
        reliability_note=reliability_note,
        response_time_seconds=llm_response.response_time_seconds,
        prompt_tokens=llm_response.prompt_tokens,
        completion_tokens=llm_response.completion_tokens,
        raw_llm_response=llm_response.text,
    )
