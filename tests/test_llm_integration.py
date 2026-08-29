"""
Phase 6 verification script.

Run (with the llama.cpp server already running -- see README / phase
notes for the exact start command):
    python tests/test_llm_integration.py

Sends a handful of test questions through the full pipeline (retrieve ->
compact evidence context -> local LLM), covering: clear relevant evidence,
relevant evidence with an overriding exclusion, incomplete evidence,
unrelated evidence, and unsupported questions. Prints the retrieved
evidence, the LLM's answer, response time, and a simple heuristic flag for
numbers in the answer that don't appear anywhere in the evidence.
"""

import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.llm import LLMError, answer_question, is_server_available
from app import config
from app.pdf_processing import build_clauses, extract_pdf
from app.rag import build_or_load_index

SAMPLE_POLICY_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "policies" / "sample_health_policy.pdf"
)
POLICY_NAME = "ABC Health Shield Policy"
POLICY_VERSION = "2025"

TEST_CASES = [
    {
        "label": "Relevant evidence + overriding exclusion",
        "question": "Is dental treatment fully covered with no conditions?",
    },
    {
        "label": "Relevant evidence (clear coverage)",
        "question": "Is dental treatment covered?",
    },
    {
        "label": "Relevant evidence (waiting period)",
        "question": "Is there a waiting period for pre-existing conditions?",
    },
    {
        "label": "Incomplete evidence (policy never states a monetary limit)",
        "question": "What is the maximum amount payable for hospitalisation expenses?",
    },
    {
        "label": "Unrelated evidence (nonsense query for this document)",
        "question": "What is the capital of France?",
    },
    {
        "label": "Unsupported (topic not in policy)",
        "question": "Does this policy cover space travel?",
    },
    {
        "label": "Unsupported (topic not in policy)",
        "question": "Is maternity care covered under this policy?",
    },
]


def extract_numbers(text: str) -> set:
    return set(re.findall(r"\d+(?:\.\d+)?", text))


def find_unsupported_numbers(answer: str, evidence: str) -> list:
    """
    Heuristic only: numbers in the answer that don't appear anywhere in the
    retrieved evidence. Can false-positive (e.g. a number that's correct
    but drawn from a DIFFERENT retrieved clause than expected) and can
    false-negative (an invented number that happens to coincide with one
    elsewhere in the evidence) -- flagged for human review, not a verdict.
    """
    return sorted(extract_numbers(answer) - extract_numbers(evidence))


def preview(text: str, max_chars: int = 100) -> str:
    text = text.strip()
    return text if len(text) <= max_chars else text[:max_chars].rstrip() + "…"


def main() -> None:
    print(f"Checking local LLM server at {config.LLM_SERVER_URL} ...")
    if not is_server_available():
        print(
            f"\nERROR: No LLM server responding at {config.LLM_SERVER_URL}.\n"
            "Start it first in another terminal:\n"
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

    response_times = []
    suite_start = time.time()

    for case in TEST_CASES:
        print("=" * 78)
        print(f"[{case['label']}]")
        print(f'Q: "{case["question"]}"')

        try:
            result = answer_question(policy_index, case["question"])
        except LLMError as exc:
            print(f"  LLM ERROR: {exc}")
            continue

        response_times.append(result.response_time_seconds)

        print("\nRetrieved evidence:")
        if result.retrieved_results:
            for r in result.retrieved_results:
                clause_label = r.clause_number or "(preamble)"
                print(f"  - Clause {clause_label}, Page {r.page_number}, score={r.score:.3f}: {preview(r.text, 90)}")
        else:
            print("  (none retrieved)")

        print(f"\nLLM answer:\n  {result.answer_text}")

        print(
            f"\nResponse time: {result.response_time_seconds:.2f}s   "
            f"prompt_tokens={result.prompt_tokens}  completion_tokens={result.completion_tokens}"
        )

        suspicious_numbers = find_unsupported_numbers(result.answer_text, result.evidence_context)
        if suspicious_numbers:
            print(f"\n  [REVIEW] Numbers in answer not found in evidence (heuristic, may be a false alarm): {suspicious_numbers}")
        print()

    total_elapsed = time.time() - suite_start
    print("=" * 78)
    print("=== Performance summary ===")
    if response_times:
        print(f"  Questions answered : {len(response_times)}/{len(TEST_CASES)}")
        print(f"  Avg response time  : {sum(response_times)/len(response_times):.2f}s")
        print(f"  Min / Max          : {min(response_times):.2f}s / {max(response_times):.2f}s")
    print(f"  Total suite time   : {total_elapsed:.2f}s")
    print("\nPhase 6 integration test complete. Review each answer above for hallucinations manually.")


if __name__ == "__main__":
    main()
