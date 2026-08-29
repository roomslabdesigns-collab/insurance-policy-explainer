"""
Phase 7 verification script.

Run (with the llama.cpp server already running):
    python tests/test_structured_answers.py

Tests the structured, evidence-gated pipeline against clearly supported,
explicitly excluded, ambiguous, unsupported, and insufficient-evidence
questions. Three of these are deliberate REGRESSIONS against Phase 6's
documented hallucinations -- this script explicitly checks whether Phase
7's stricter status classification and pre-LLM gate fixed them.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.llm import LLMError, is_server_available
from app import config
from app.pdf_processing import build_clauses, extract_pdf
from app.rag import ALL_STATUSES, generate_grounded_response
from app.rag import build_or_load_index
from app.utils import STATUS_COVERED, STATUS_EXCLUDED

SAMPLE_POLICY_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "policies" / "sample_health_policy.pdf"
)
POLICY_NAME = "ABC Health Shield Policy"
POLICY_VERSION = "2025"

TEST_CASES = [
    {
        "label": "Clearly Supported",
        "question": "Is dental treatment covered?",
    },
    {
        "label": "Clearly Supported (waiting period)",
        "question": "Is there a waiting period for pre-existing conditions?",
        "expect_status": STATUS_COVERED,
    },
    {
        "label": "Explicit Exclusion",
        "question": "Is cosmetic surgery excluded?",
        "expect_status": STATUS_EXCLUDED,
    },
    {
        "label": "Ambiguous (multiple valid clauses)",
        "question": "What dental treatments are not covered?",
        "expect_status": STATUS_EXCLUDED,
    },
    {
        "label": "REGRESSION vs Phase 6: silence != exclusion",
        "question": "Does this policy cover space travel?",
        "must_not_be": [STATUS_EXCLUDED],
    },
    {
        "label": "REGRESSION vs Phase 6: silence != exclusion",
        "question": "Is maternity care covered under this policy?",
        "must_not_be": [STATUS_EXCLUDED],
    },
    {
        "label": "REGRESSION vs Phase 6: general-knowledge answer",
        "question": "What is the capital of France?",
        "expect_status": "No Evidence Found",
    },
    {
        "label": "Insufficient Evidence (no monetary limit stated)",
        "question": "What is the maximum amount payable for hospitalisation expenses?",
        "must_not_be": [STATUS_COVERED, STATUS_EXCLUDED],
    },
]


def check(label: str, condition: bool) -> bool:
    print(f"  [{'OK' if condition else 'FAIL'}] {label}")
    return condition


def preview(text: str, max_chars: int = 90) -> str:
    text = text.strip()
    return text if len(text) <= max_chars else text[:max_chars].rstrip() + "…"


def main() -> None:
    print(f"Checking local LLM server at {config.LLM_SERVER_URL} ...")
    if not is_server_available():
        print(
            f"\nERROR: No LLM server responding at {config.LLM_SERVER_URL}.\n"
            "Start it first:\n"
            f'  .\\venv\\Scripts\\python.exe -m llama_cpp.server --model "{config.LLM_MODEL_PATH}" '
            f"--n_ctx {config.LLM_CONTEXT_SIZE} --host 127.0.0.1 --port 8000\n"
        )
        sys.exit(1)
    print("Server is reachable.\n")

    if not SAMPLE_POLICY_PATH.exists():
        print("Sample policy not found — run tests/test_pdf_extraction.py first.")
        sys.exit(1)

    document = extract_pdf(SAMPLE_POLICY_PATH)
    clauses = build_clauses(document, policy_name=POLICY_NAME, policy_version=POLICY_VERSION)
    policy_index = build_or_load_index(document, clauses, POLICY_NAME, POLICY_VERSION)

    all_checks_passed = True

    for case in TEST_CASES:
        print("=" * 78)
        print(f"[{case['label']}]")
        print(f'Q: "{case["question"]}"')

        try:
            response = generate_grounded_response(policy_index, case["question"])
        except LLMError as exc:
            print(f"  LLM ERROR: {exc}")
            all_checks_passed = False
            continue

        evidence_ids = ", ".join(
            f"E{i}[{r.clause_number or '(preamble)'}]" for i, r in enumerate(response.retrieved_results, start=1)
        )
        print(f"\nRetrieved evidence IDs: {evidence_ids or '(none)'}")
        print(f"Status    : {response.status}")
        print(f"Answer    : {response.answer_text}")
        if response.citation:
            print(
                f"Citation  : Clause {response.citation.clause_number or '(preamble)'}, "
                f"Page {response.citation.page_number} — {preview(response.citation.evidence_text)}"
            )
        else:
            print("Citation  : (none)")
        print(f"Validation: {'PASSED' if response.validation_passed else 'FAILED'} — {response.reliability_note}")
        if response.response_time_seconds is not None:
            print(f"Time      : {response.response_time_seconds:.2f}s")

        print()
        all_checks_passed &= check("Status is one of the allowed values", response.status in ALL_STATUSES)
        all_checks_passed &= check("Answer text is non-empty", bool(response.answer_text.strip()))

        if response.status in (STATUS_COVERED, STATUS_EXCLUDED):
            all_checks_passed &= check(
                f"{response.status} status has a resolved citation", response.citation is not None
            )

        if "expect_status" in case:
            all_checks_passed &= check(
                f"Status is exactly {case['expect_status']!r}", response.status == case["expect_status"]
            )
        if "must_not_be" in case:
            all_checks_passed &= check(
                f"Status is NOT any of {case['must_not_be']}", response.status not in case["must_not_be"]
            )
        print()

    print("=" * 78)
    if all_checks_passed:
        print("Phase 7 environment check: PASS")
    else:
        print("Phase 7 environment check: FAIL (see [FAIL] lines above)")
        sys.exit(1)


if __name__ == "__main__":
    main()
