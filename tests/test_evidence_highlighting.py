"""
Phase 10 verification script.

Run:
    python tests/test_evidence_highlighting.py

Tests the highlighter module directly (pure Python, no Streamlit needed)
against the synthetic sample policy:
  A. Exact evidence         -> found via tier-1 exact match
  B. Formatting difference  -> tier-1 fails, tier-2 sentence match succeeds
  C. Cross-page isolation   -> the same text is searched independently on
                               two different pages, each result correct
  D. Highlight failure      -> genuinely absent text -> honest not_found,
                               with a still-renderable plain page image
  E. Multiple sources       -> several retrieved chunks each independently
                               locatable on their own pages
Also verifies the original PDF file is never modified.
"""

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.pdf_processing import build_clauses, extract_pdf
from app.pdf_processing.highlighter import (
    MATCH_EXACT,
    MATCH_NOT_FOUND,
    MATCH_SENTENCE,
    locate_evidence_on_page,
    render_page_image,
)
from app.rag import build_or_load_index

SAMPLE_POLICY_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "policies" / "sample_health_policy.pdf"
)
POLICY_NAME = "ABC Health Shield Policy"
POLICY_VERSION = "2025"


def check(label: str, condition: bool) -> bool:
    print(f"  [{'OK' if condition else 'FAIL'}] {label}")
    return condition


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_clause(clauses, clause_number):
    return next(c for c in clauses if c.clause_number == clause_number)


def main() -> None:
    if not SAMPLE_POLICY_PATH.exists():
        print("Sample policy not found — run tests/test_pdf_extraction.py first.")
        sys.exit(1)

    original_hash_before = file_hash(SAMPLE_POLICY_PATH)

    document = extract_pdf(SAMPLE_POLICY_PATH)
    clauses = build_clauses(document, policy_name=POLICY_NAME, policy_version=POLICY_VERSION)
    policy_index = build_or_load_index(document, clauses, POLICY_NAME, POLICY_VERSION)
    pdf_path = policy_index.source_pdf_path

    all_passed = True
    all_passed &= check("source_pdf_path was persisted on the PolicyIndex", bool(pdf_path))

    # ------------------------------------------------------------------
    # Test A — Exact evidence
    # ------------------------------------------------------------------
    print("\n=== Test A: Exact evidence ===")
    cosmetic = find_clause(clauses, "4.1")
    result_a = locate_evidence_on_page(pdf_path, cosmetic.page_number, cosmetic.text)
    all_passed &= check(f"Found via exact match (level={result_a.match_level})", result_a.found and result_a.match_level == MATCH_EXACT)
    all_passed &= check("At least one highlight quad returned", len(result_a.quads) >= 1)

    # ------------------------------------------------------------------
    # Test B — Formatting difference (tier-1 fails, tier-2 sentence match succeeds)
    # ------------------------------------------------------------------
    print("\n=== Test B: Formatting difference (multi-sentence clause) ===")
    dental = find_clause(clauses, "2.3")  # two sentences: coverage + what's excluded

    # NOTE: PyMuPDF's search_for() turned out to already normalize whitespace
    # internally -- a version with only extra spaces/newlines injected still
    # matched at tier 1 (a pleasant robustness discovery, logged in the Phase
    # 10 report). To genuinely exercise tier 2, we inject actual extra
    # CONTENT between the two real sentences -- content that doesn't exist
    # on the page at all -- while keeping each original sentence verbatim.
    disrupted_text = dental.text.replace(
        ". Routine", ". [See also the annexure for details] Routine", 1
    )
    result_b_full = locate_evidence_on_page(pdf_path, dental.page_number, disrupted_text)
    all_passed &= check(
        "Full text WITH injected extra content fails tier-1 exact match (as expected)",
        not (result_b_full.found and result_b_full.match_level == MATCH_EXACT),
    )
    all_passed &= check(
        f"Sentence-level tier-2 still locates the real sentences (level={result_b_full.match_level})",
        result_b_full.found and result_b_full.match_level == MATCH_SENTENCE,
    )

    # ------------------------------------------------------------------
    # Test C — Cross-page isolation
    # ------------------------------------------------------------------
    print("\n=== Test C: Cross-page isolation (same text searched on two pages) ===")
    result_c_correct_page = locate_evidence_on_page(pdf_path, dental.page_number, dental.text)
    other_page = 4 if dental.page_number != 4 else 5
    result_c_wrong_page = locate_evidence_on_page(pdf_path, other_page, dental.text)
    all_passed &= check(f"Found on its real page ({dental.page_number})", result_c_correct_page.found)
    all_passed &= check(
        f"NOT found on an unrelated page ({other_page}) -- no cross-page contamination",
        not result_c_wrong_page.found,
    )
    print(
        "  (Note: this sample policy has no clause that genuinely spans two pages; this test "
        "verifies per-page isolation, which is what real cross-page handling relies on.)"
    )

    # ------------------------------------------------------------------
    # Test D — Highlight failure (honest fallback, still renders the page)
    # ------------------------------------------------------------------
    print("\n=== Test D: Highlight failure (text genuinely not in the PDF) ===")
    fake_text = "This sentence about lunar habitat coverage does not exist anywhere in this policy."
    result_d = locate_evidence_on_page(pdf_path, dental.page_number, fake_text)
    all_passed &= check(f"Honestly reports not found (level={result_d.match_level})", not result_d.found and result_d.match_level == MATCH_NOT_FOUND)
    all_passed &= check("No fabricated quads returned", result_d.quads == [])
    try:
        plain_image = render_page_image(pdf_path, dental.page_number, [])
        all_passed &= check("Page still renders as a plain image (no fake highlight)", len(plain_image) > 0)
    except Exception as exc:
        all_passed &= check(f"Page still renders as a plain image (no fake highlight): {exc}", False)

    # ------------------------------------------------------------------
    # Test E — Multiple sources, independently inspectable
    # ------------------------------------------------------------------
    print("\n=== Test E: Multiple evidence sources, each independently located ===")
    multi_clauses = [find_clause(clauses, n) for n in ("2.3", "4.2(c)", "3.1")]
    for c in multi_clauses:
        r = locate_evidence_on_page(pdf_path, c.page_number, c.text)
        all_passed &= check(f"Clause {c.clause_number} (page {c.page_number}) located independently: {r.match_level}", r.found)

    # ------------------------------------------------------------------
    # Original PDF integrity
    # ------------------------------------------------------------------
    print("\n=== Original PDF integrity ===")
    original_hash_after = file_hash(SAMPLE_POLICY_PATH)
    all_passed &= check(
        "Original PDF file is byte-for-byte unchanged after all rendering",
        original_hash_before == original_hash_after,
    )

    print()
    print("Phase 10 environment check:", "PASS" if all_passed else "FAIL")
    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
