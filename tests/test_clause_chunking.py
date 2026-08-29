"""
Phase 3 verification script.

Run:
    python tests/test_clause_chunking.py

Checks that clause-aware chunking correctly finds specific known clauses in
the synthetic sample policy (built in Phase 2), attaches the right section,
page number, and exclusion flags, and that the length-based splitter works.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.pdf_processing import (
    build_clauses,
    extract_pdf,
    get_clause_summary,
    split_oversized_text,
)

SAMPLE_POLICY_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "policies" / "sample_health_policy.pdf"
)

POLICY_NAME = "ABC Health Shield Policy"
POLICY_VERSION = "2025"


def check(label: str, condition: bool) -> bool:
    status = "OK" if condition else "FAIL"
    print(f"  [{status}] {label}")
    return condition


def run_known_clause_checks(clauses) -> bool:
    by_number = {}
    for c in clauses:
        by_number.setdefault(c.clause_number, []).append(c)

    all_passed = True

    print("=== Known-clause checks against the synthetic policy ===")

    dental = by_number.get("2.3", [])
    all_passed &= check(
        "Clause 2.3 detected with 'Dental Treatment' text",
        bool(dental) and "Dental Treatment" in dental[0].text,
    )
    all_passed &= check(
        "Clause 2.3 is on page 3",
        bool(dental) and dental[0].page_number == 3,
    )
    all_passed &= check(
        "Clause 2.3 is NOT flagged as an exclusion section",
        bool(dental) and not dental[0].is_exclusion_section,
    )

    waiting = by_number.get("3.1", [])
    all_passed &= check(
        "Clause 3.1 detected and mentions '48'",
        bool(waiting) and "48" in waiting[0].text,
    )

    dental_exclusion = by_number.get("4.2(c)", [])
    all_passed &= check(
        "Sub-clause 4.2(c) detected via bare '(c)' line",
        bool(dental_exclusion),
    )
    all_passed &= check(
        "Clause 4.2(c) correctly flagged is_exclusion_section",
        bool(dental_exclusion) and dental_exclusion[0].is_exclusion_section,
    )
    all_passed &= check(
        "Clause 4.2(c) correctly flagged contains_exclusion_language",
        bool(dental_exclusion) and dental_exclusion[0].contains_exclusion_language,
    )
    all_passed &= check(
        "Clause 4.2(c) is on page 5",
        bool(dental_exclusion) and dental_exclusion[0].page_number == 5,
    )

    first_year = by_number.get("4.3", [])
    all_passed &= check(
        "Clause 4.3 (first-year claims) detected and mentions '90 days'",
        bool(first_year) and "90 days" in first_year[0].text,
    )

    # Regression check for the false-positive risk called out in the design
    # notes: a bare duration number ("48 months...") must NOT be misread as
    # a new clause "48".
    all_passed &= check(
        "No false-positive clause '48' created from '48 months' wording",
        "48" not in by_number,
    )

    return all_passed


def run_splitter_check() -> bool:
    print("\n=== Length-based splitter check ===")
    long_text = "This is a test sentence about coverage. " * 40  # ~1640 chars
    parts = split_oversized_text(long_text, max_chars=800)
    ok = len(parts) > 1 and all(len(p) <= 800 for p in parts)
    check(f"Oversized text split into {len(parts)} parts, each <= 800 chars", ok)
    return ok


def main() -> None:
    if not SAMPLE_POLICY_PATH.exists():
        print("Sample policy not found — run tests/test_pdf_extraction.py first.")
        sys.exit(1)

    document = extract_pdf(SAMPLE_POLICY_PATH)
    clauses = build_clauses(document, policy_name=POLICY_NAME, policy_version=POLICY_VERSION)
    summary = get_clause_summary(clauses)

    print(f"Policy           : {POLICY_NAME} ({POLICY_VERSION})")
    print(f"Total chunks      : {summary['total_chunks']}")
    print(f"Numbered clauses  : {summary['numbered_clauses']}")
    print(f"Preamble chunks   : {summary['preamble_chunks']}")
    print(f"Exclusion-language chunks : {summary['chunks_with_exclusion_language']}")
    print(f"Sections detected : {summary['sections_detected']}")

    print("\n--- All detected clauses ---")
    for c in clauses:
        label = c.clause_number if c.clause_number else "(preamble)"
        flag = " [EXCLUSION]" if c.contains_exclusion_language else ""
        print(f"  Page {c.page_number:>2} | {label:<10} | {c.text[:70]}{'...' if len(c.text) > 70 else ''}{flag}")

    clause_checks_passed = run_known_clause_checks(clauses)
    splitter_passed = run_splitter_check()

    print()
    if clause_checks_passed and splitter_passed:
        print("Phase 3 environment check: PASS")
    else:
        print("Phase 3 environment check: FAIL (see [FAIL] lines above)")
        sys.exit(1)


if __name__ == "__main__":
    main()
