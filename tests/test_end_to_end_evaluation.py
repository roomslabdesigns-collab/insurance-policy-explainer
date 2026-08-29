"""
Phase 11 verification / evaluation script.

Run (with the llama.cpp server already running):
    python tests/test_end_to_end_evaluation.py

Runs the ACTUAL user-facing pipeline (retrieval -> evidence selection ->
local LLM -> Phase 8 guardrails) against the 40-question end-to-end golden
dataset, classifies every result, saves a timestamped run under
data/evaluation/e2e_runs/<run_id>/, prints the full report, and compares
against the previous run if one exists (regression tracking).
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config
from app.evaluation import (
    compare_runs,
    evaluate_all,
    generate_text_report,
    get_previous_run_id,
    load_golden_e2e_dataset,
    save_run,
)
from app.llm import is_server_available
from app.pdf_processing import build_clauses, extract_pdf
from app.rag import build_or_load_index

SAMPLE_POLICY_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "policies" / "sample_health_policy.pdf"
)
POLICY_NAME = "ABC Health Shield Policy"
POLICY_VERSION = "2025"

pd.set_option("display.max_colwidth", 60)
pd.set_option("display.width", 160)


def main() -> None:
    print(f"Checking local LLM server at {config.LLM_SERVER_URL} ...")
    if not is_server_available():
        print(
            f"\nERROR: No LLM server responding at {config.LLM_SERVER_URL}.\n"
            "This evaluation needs the real pipeline, including real LLM calls. Start it first:\n"
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

    dataset = load_golden_e2e_dataset()
    print(f"Golden end-to-end dataset: {len(dataset)} questions loaded.")
    print("Running the full pipeline for each question (this calls the real local LLM — expect ~2-15s per question)...\n")

    results_df = evaluate_all(policy_index, dataset)

    run_id = save_run(results_df, POLICY_NAME, policy_index.document_id)
    print(f"Run saved as: {run_id}\n")

    print(generate_text_report(run_id))

    print("\n=== Per-question classification ===")
    print(
        results_df[["question", "category", "expected_status", "actual_status", "classification", "failure_stage"]]
        .to_string(index=False)
    )

    non_correct = results_df[results_df["classification"] != "Correct"]
    if not non_correct.empty:
        print("\n=== Non-correct results in detail ===")
        print(
            non_correct[
                ["question", "expected_status", "actual_status", "classification", "failure_stage", "answer_text"]
            ].to_string(index=False)
        )

    previous_run_id = get_previous_run_id(run_id)
    if previous_run_id:
        print(f"\n=== Regression comparison: {previous_run_id} -> {run_id} ===")
        print(compare_runs(previous_run_id, run_id).to_string(index=False))
    else:
        print("\n(No previous run found — this is the baseline for future regression comparisons.)")


if __name__ == "__main__":
    main()
