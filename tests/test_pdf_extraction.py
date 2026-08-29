"""
Phase 2 verification script.

Run:
    python tests/test_pdf_extraction.py                       # uses the generated sample policy
    python tests/test_pdf_extraction.py path\\to\\real_policy.pdf  # uses a real policy PDF
"""

import sys
from pathlib import Path

# Make the `app` package importable when this script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.pdf_processing import (
    InvalidPDFError,
    extract_pdf,
    get_extraction_summary,
    preview_text,
)

SAMPLE_POLICY_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "policies" / "sample_health_policy.pdf"
)


def demonstrate_error_handling() -> None:
    """Prove that bad inputs produce a clean message instead of a crash."""
    print("=== Error-handling checks (these are SUPPOSED to fail) ===")
    checks = [
        ("does_not_exist.pdf", "missing file"),
        (str(Path(__file__)), "non-PDF file (this .py script)"),
    ]
    for bad_path, label in checks:
        try:
            extract_pdf(bad_path)
            print(f"  [UNEXPECTED] {label}: no error was raised")
        except InvalidPDFError as exc:
            print(f"  [OK] {label} correctly rejected -> {exc}")
    print()


def run_extraction_report(pdf_path: Path) -> None:
    print(f"=== Extracting: {pdf_path.name} ===")
    document = extract_pdf(pdf_path)
    summary = get_extraction_summary(document)

    print(f"Filename        : {summary['filename']}")
    print(f"Document ID     : {summary['document_id']}")
    print(f"Total pages     : {summary['total_pages']}")
    print(f"Pages OK        : {summary['pages_ok']}")
    print(f"Pages low-text  : {summary['pages_low_text']}")
    print(f"Pages empty     : {summary['pages_empty']}")
    if summary["problem_pages"]:
        print(
            f"Problem pages   : {summary['problem_pages']}  "
            "(may be scanned/image-based — OCR not implemented in this phase)"
        )
    else:
        print("Problem pages   : none")

    print("\n--- Preview of extracted text (first 3 successfully-extracted pages) ---")
    shown = 0
    for page in document.pages:
        if page.extraction_status != "ok":
            continue
        print(f"\n[Page {page.page_number}] ({page.char_count} chars, status={page.extraction_status})")
        print(preview_text(page.text, max_chars=220))
        shown += 1
        if shown >= 3:
            break

    print("\nPhase 2 environment check: PASS")


def main() -> None:
    if len(sys.argv) > 1:
        pdf_path = Path(sys.argv[1])
    else:
        if not SAMPLE_POLICY_PATH.exists():
            print("No sample policy found — generating one for testing...\n")
            from make_sample_policy import build_sample_policy

            build_sample_policy(SAMPLE_POLICY_PATH)
        pdf_path = SAMPLE_POLICY_PATH

    demonstrate_error_handling()
    run_extraction_report(pdf_path)


if __name__ == "__main__":
    main()
