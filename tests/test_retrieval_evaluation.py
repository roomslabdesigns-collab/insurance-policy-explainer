"""
Phase 5 verification script.

Run:
    python tests/test_retrieval_evaluation.py

Evaluates the Phase 4 retrieval system against the hand-verified golden
dataset (app/evaluation/golden_retrieval_dataset.csv) and prints a full
report: Recall@1/@3/@5, MRR, per-category breakdown, an unsupported-question
threshold analysis, a retrieval-diversity check, and a failure list.
No LLM is loaded anywhere in this script.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.evaluation import (
    GOLDEN_DATASET_PATH,
    analyze_diversity,
    analyze_threshold_candidates,
    compute_category_metrics,
    compute_overall_metrics,
    evaluate_retrieval,
    get_failures,
    load_golden_dataset,
    recommend_threshold,
)
from app.pdf_processing import build_clauses, extract_pdf
from app.rag import build_or_load_index

SAMPLE_POLICY_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "policies" / "sample_health_policy.pdf"
)
POLICY_NAME = "ABC Health Shield Policy"
POLICY_VERSION = "2025"

RESULTS_OUTPUT_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "evaluation" / "retrieval_eval_results_latest.csv"
)

pd.set_option("display.max_colwidth", 60)
pd.set_option("display.width", 160)


def main() -> None:
    if not SAMPLE_POLICY_PATH.exists():
        print("Sample policy not found — run tests/test_pdf_extraction.py first.")
        sys.exit(1)

    document = extract_pdf(SAMPLE_POLICY_PATH)
    clauses = build_clauses(document, policy_name=POLICY_NAME, policy_version=POLICY_VERSION)
    policy_index = build_or_load_index(document, clauses, POLICY_NAME, POLICY_VERSION)

    dataset = load_golden_dataset()
    print(f"Golden dataset: {len(dataset)} questions loaded from {GOLDEN_DATASET_PATH.name}")

    results_df = evaluate_retrieval(policy_index, dataset)

    RESULTS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_OUTPUT_PATH, index=False)
    print(f"Full per-question results saved to: {RESULTS_OUTPUT_PATH}")

    threshold_df = analyze_threshold_candidates(results_df)
    recommended = recommend_threshold(threshold_df)
    overall = compute_overall_metrics(results_df, threshold=recommended)

    print("\n=== Overall Retrieval Metrics (supported questions only) ===")
    print(
        f"  Questions evaluated : {overall['num_supported_questions']} supported, "
        f"{overall['num_unsupported_questions']} unsupported"
    )
    print(f"  Recall@1            : {overall['recall_at_1']:.1%}")
    print(f"  Recall@3            : {overall['recall_at_3']:.1%}")
    print(f"  Recall@5            : {overall['recall_at_5']:.1%}")
    print(f"  MRR                 : {overall['mrr']:.3f}")

    print("\n=== Per-Category Metrics ===")
    category_df = compute_category_metrics(results_df)
    print(
        category_df.to_string(
            index=False,
            formatters={
                "recall_at_1": "{:.0%}".format,
                "recall_at_3": "{:.0%}".format,
                "recall_at_5": "{:.0%}".format,
                "mrr": "{:.3f}".format,
            },
        )
    )

    print("\n=== Threshold Candidate Analysis (unsupported-question safety) ===")
    print(
        threshold_df.to_string(
            index=False,
            formatters={
                "unsupported_false_positive_rate": "{:.0%}".format,
                "correct_top1_retained_rate": "{:.0%}".format,
            },
        )
    )
    if recommended is not None:
        print(
            f"\n  Recommended candidate threshold: {recommended:.2f} "
            "(zero false positives on unsupported questions, while keeping every correct top-1 answer)"
        )
    else:
        print(
            "\n  No candidate threshold achieves zero false positives while retaining every "
            "correct top-1 answer — positive/negative scores overlap somewhere in this range. "
            "Phase 8 will need more than a single hard cutoff (e.g. a score margin check, or "
            "requiring agreement across top-2 results)."
        )

    print("\n=== Retrieval Diversity Check ===")
    diversity = analyze_diversity(results_df)
    print(f"  Avg. unique clauses among top-5 results  : {diversity['avg_unique_clauses_in_top5']:.2f} / 5")
    print(f"  Avg. unique sections among top-5 results : {diversity['avg_unique_sections_in_top5']:.2f} / 5")
    print(f"  Questions with a repeated clause in top-5 : {diversity['questions_with_repeated_clause_in_top5']}")
    for q in diversity["repetitive_questions"]:
        print(f"    - {q}")

    print("\n=== Failure / Attention List ===")
    failures = get_failures(results_df)
    if failures.empty:
        print(
            "  None — every supported question found its evidence within top-5, and no "
            "unsupported question scored suspiciously high."
        )
    else:
        print(failures.to_string(index=False))

    print("\nPhase 5 evaluation complete. (No LLM was loaded during this run.)")


if __name__ == "__main__":
    main()
