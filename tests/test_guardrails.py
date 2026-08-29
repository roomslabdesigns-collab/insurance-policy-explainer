"""
Phase 8 verification script.

Run (with the llama.cpp server already running for the live tests):
    python tests/test_guardrails.py

Covers the six guardrail scenarios from the Phase 8 spec:
  A. Valid citation                           -> answer passes validation      [live]
  B. Fake evidence ID                         -> rejected                      [unit]
  C. Invented page/clause in the answer text  -> ignored; verified metadata used [unit]
  D. Unsupported number in the answer         -> flagged                       [unit]
  E. Not Mentioned vs Excluded                -> never classified as Excluded  [unit + live]
  F. Conflicting evidence (coverage + exclusion retrieved together) -> surfaced, not hidden [unit + live]

B/C/D/E/F's core logic is tested at the unit level against controlled,
synthetic SearchResult fixtures -- this is deliberate: the live LLM cannot
be reliably made to produce a fake evidence ID or an invented page number
on demand, so the guardrail functions themselves are tested directly and
deterministically. A/E/F are additionally exercised live through the real
pipeline for end-to-end confidence.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config
from app.llm import is_server_available
from app.pdf_processing import build_clauses, extract_pdf
from app.rag import build_or_load_index, generate_grounded_response
from app.rag.retrieval import retrieve
from app.rag.vector_store import SearchResult
from app.utils import STATUS_COVERED, STATUS_EXCLUDED
from app.validation import (
    apply_status_guardrails,
    build_citation,
    check_claim_support,
    check_related_exclusions,
    validate_evidence_id,
)

SAMPLE_POLICY_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "policies" / "sample_health_policy.pdf"
)
POLICY_NAME = "ABC Health Shield Policy"
POLICY_VERSION = "2025"


def check(label: str, condition: bool) -> bool:
    print(f"  [{'OK' if condition else 'FAIL'}] {label}")
    return condition


def make_result(document_id: str, **overrides) -> SearchResult:
    defaults = dict(
        rank=1,
        score=0.9,
        chunk_id="synthetic-test-chunk",
        document_id=document_id,
        clause_number="9.9",
        section="SECTION 9: SYNTHETIC TEST",
        page_number=1,
        pages=[1],
        text="This is a plain statement with no exclusion wording at all.",
        is_exclusion_section=False,
        contains_exclusion_language=False,
    )
    defaults.update(overrides)
    return SearchResult(**defaults)


def run_unit_tests(policy_index) -> bool:
    passed = True
    document_id = policy_index.document_id

    print("=== Test B: Fake evidence ID is rejected ===")
    results = retrieve(policy_index, "Is dental treatment covered?", max_k=3)
    label_map = {f"E{i}": r for i, r in enumerate(results, start=1)}
    fake_check = validate_evidence_id("E99", label_map, document_id)
    passed &= check("Fake evidence_id 'E99' rejected", not fake_check.passed)
    real_check = validate_evidence_id("E1", label_map, document_id)
    passed &= check("Real evidence_id 'E1' accepted", real_check.passed and real_check.result is not None)
    print()

    print("=== Test C: Invented page/clause in answer text is ignored ===")
    real_result = label_map["E1"]
    fake_answer = "This is covered under Clause 99.9, Page 404, according to the policy."
    citation = build_citation(real_result, fake_answer, POLICY_NAME, POLICY_VERSION)
    passed &= check(
        f"Citation page ({citation.page_number}) matches real retrieval, not the fake '404'",
        citation.page_number == real_result.page_number and citation.page_number != 404,
    )
    passed &= check(
        f"Citation clause ({citation.clause_number!r}) matches real retrieval, not the fake '99.9'",
        citation.clause_number == real_result.clause_number and citation.clause_number != "99.9",
    )
    print()

    print("=== Test D: Unsupported number in the answer is flagged ===")
    evidence_text = "A waiting period of 48 months applies to pre-existing conditions."
    bad = check_claim_support("The waiting period for this is 12 months.", evidence_text)
    passed &= check(f"Invented number '12' flagged: {bad.unsupported_numbers}", bad.is_high_risk and "12" in bad.unsupported_numbers)
    good = check_claim_support("The waiting period for this is 48 months.", evidence_text)
    passed &= check("Number actually present in evidence is NOT flagged", not good.is_high_risk)
    print()

    print("=== Test E: Explicitly Excluded requires exclusion evidence (metadata cross-check) ===")
    no_signal_citation = build_citation(
        make_result(document_id, is_exclusion_section=False, contains_exclusion_language=False),
        "irrelevant", POLICY_NAME, POLICY_VERSION,
    )
    outcome = apply_status_guardrails(STATUS_EXCLUDED, no_signal_citation)
    passed &= check(
        f"'Explicitly Excluded' with NO exclusion signal is downgraded (-> {outcome.status})", outcome.downgraded
    )

    real_exclusion_citation = build_citation(
        make_result(document_id, is_exclusion_section=True, contains_exclusion_language=True, clause_number="4.1"),
        "irrelevant", POLICY_NAME, POLICY_VERSION,
    )
    outcome2 = apply_status_guardrails(STATUS_EXCLUDED, real_exclusion_citation)
    passed &= check("'Explicitly Excluded' WITH real exclusion signal is kept", not outcome2.downgraded)

    covered_from_exclusion_section = build_citation(
        make_result(document_id, is_exclusion_section=True, contains_exclusion_language=True, clause_number="4.2(a)"),
        "irrelevant", POLICY_NAME, POLICY_VERSION,
    )
    outcome3 = apply_status_guardrails(STATUS_COVERED, covered_from_exclusion_section)
    passed &= check(
        f"'Covered' citing an exclusion-section clause is downgraded (-> {outcome3.status})", outcome3.downgraded
    )
    print()

    print("=== Test F: Conflicting evidence (exclusion elsewhere) is surfaced, not hidden ===")
    other_results = [
        make_result(document_id, chunk_id="cov-1", clause_number="2.1", is_exclusion_section=False, contains_exclusion_language=False),
        make_result(document_id, chunk_id="exc-1", clause_number="4.2(c)", is_exclusion_section=True, contains_exclusion_language=True),
    ]
    note_with_conflict = check_related_exclusions(STATUS_COVERED, "cov-1", other_results)
    passed &= check(
        f"Related exclusion surfaced when present: {note_with_conflict!r}",
        note_with_conflict is not None and "4.2(c)" in note_with_conflict,
    )
    note_without_conflict = check_related_exclusions(STATUS_COVERED, "cov-1", [other_results[0]])
    passed &= check("No note when no exclusion is among the OTHER retrieved chunks", note_without_conflict is None)
    print()

    return passed


def run_live_tests(policy_index) -> bool:
    passed = True

    print("=== Test A (live): Valid citation passes validation ===")
    r = generate_grounded_response(policy_index, "Is there a waiting period for pre-existing conditions?")
    print(f"  Status: {r.status} | Confidence: {r.confidence_label} | Validated: {r.validation_passed}")
    print(f"  Answer: {r.answer_text}")
    if r.citation:
        print(f"  Source: {r.citation.section} → Clause {r.citation.clause_number} → Page {r.citation.page_number}")
    passed &= check("Validation passed", r.validation_passed)
    passed &= check("Citation resolved", r.citation is not None)
    passed &= check("Confidence label is 'Verified Evidence'", r.confidence_label == "Verified Evidence")
    print()

    print("=== Test E (live regression): silence is never 'Explicitly Excluded' ===")
    for q in ["Does this policy cover space travel?", "Is maternity care covered under this policy?"]:
        r = generate_grounded_response(policy_index, q)
        print(f'  Q: "{q}" -> status={r.status}, confidence={r.confidence_label}')
        passed &= check(f"  Status is NOT 'Explicitly Excluded'", r.status != STATUS_EXCLUDED)
    print()

    print("=== Test F (live, observational): coverage + exclusion retrieved together ===")
    r = generate_grounded_response(policy_index, "Is dental treatment covered?")
    print(f"  Status: {r.status} | Confidence: {r.confidence_label}")
    print(f"  Answer: {r.answer_text}")
    print(f"  Reliability note: {r.reliability_note}")
    print("  (informational only -- whether the related-exclusion note fires depends on the")
    print("   LLM's live status choice; Test F's hard assertions are covered at the unit level above)")
    print()

    print("=== General knowledge / no-evidence regression (Phase 6/7) ===")
    r = generate_grounded_response(policy_index, "What is the capital of France?")
    print(f"  Status: {r.status} | Answer: {r.answer_text}")
    passed &= check("No hallucinated general-knowledge answer", "Paris" not in r.answer_text)
    print()

    return passed


def main() -> None:
    if not SAMPLE_POLICY_PATH.exists():
        print("Sample policy not found — run tests/test_pdf_extraction.py first.")
        sys.exit(1)

    document = extract_pdf(SAMPLE_POLICY_PATH)
    clauses = build_clauses(document, policy_name=POLICY_NAME, policy_version=POLICY_VERSION)
    policy_index = build_or_load_index(document, clauses, POLICY_NAME, POLICY_VERSION)

    all_passed = run_unit_tests(policy_index)

    print(f"Checking local LLM server at {config.LLM_SERVER_URL} ...")
    if is_server_available():
        print("Server is reachable.\n")
        all_passed &= run_live_tests(policy_index)
    else:
        print("Server not running -- skipping live tests (A, E-live, F-live). Unit tests above still count.\n")

    print("=" * 78)
    if all_passed:
        print("Phase 8 environment check: PASS")
    else:
        print("Phase 8 environment check: FAIL (see [FAIL] lines above)")
        sys.exit(1)


if __name__ == "__main__":
    main()
