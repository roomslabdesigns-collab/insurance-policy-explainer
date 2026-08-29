"""
Phase 4 verification script.

Run:
    python tests/test_semantic_search.py

Builds (or reuses) a FAISS index over the synthetic sample policy's clauses,
then runs realistic insurance questions -- both ones the policy should
answer and ones it clearly does not cover -- to see how semantic search
behaves BEFORE any LLM is involved.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.pdf_processing import build_clauses, extract_pdf
from app.rag import build_or_load_index, search_policy

SAMPLE_POLICY_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "policies" / "sample_health_policy.pdf"
)
POLICY_NAME = "ABC Health Shield Policy"
POLICY_VERSION = "2025"

POSITIVE_QUERIES = [
    "Is dental treatment covered?",
    "What are the exclusions in this policy?",
    "Is there a waiting period for pre-existing conditions?",
    "What happens if I make a claim during the first year?",
    "What is covered under day care treatment?",
]

# Topics this synthetic policy never mentions at all.
NEGATIVE_QUERIES = [
    "Does this policy cover international travel emergencies?",
    "Is there a maternity benefit waiting period?",
    "Does this policy cover pet insurance?",
]


def check(label: str, condition: bool) -> bool:
    status = "OK" if condition else "FAIL"
    print(f"  [{status}] {label}")
    return condition


def preview(text: str, max_chars: int = 140) -> str:
    text = text.strip()
    return text if len(text) <= max_chars else text[:max_chars].rstrip() + " …"


def run_query_batch(policy_index, label: str, queries) -> list:
    print(f"\n=== {label} ===")
    top1_scores = []
    for query in queries:
        results = search_policy(policy_index, query, top_k=3)
        print(f'\nQ: "{query}"')
        if not results:
            print("  (no results)")
            continue
        top1_scores.append(results[0].score)
        for r in results:
            clause_label = r.clause_number or "(preamble)"
            print(
                f"  #{r.rank}  score={r.score:.3f}  "
                f"section='{r.section or '-'}'  clause={clause_label}  page={r.page_number}"
            )
            print(f"       {preview(r.text)}")
    return top1_scores


def main() -> None:
    if not SAMPLE_POLICY_PATH.exists():
        print("Sample policy not found — run tests/test_pdf_extraction.py first.")
        sys.exit(1)

    document = extract_pdf(SAMPLE_POLICY_PATH)
    clauses = build_clauses(document, policy_name=POLICY_NAME, policy_version=POLICY_VERSION)

    print("=== Building index (first run embeds from scratch) ===")
    t0 = time.time()
    policy_index = build_or_load_index(document, clauses, POLICY_NAME, POLICY_VERSION)
    print(f"  Ready in {time.time() - t0:.2f}s — {policy_index.chunk_count} chunks indexed.")

    print("\n=== Re-requesting the same index (should reuse the saved one) ===")
    t0 = time.time()
    policy_index_2 = build_or_load_index(document, clauses, POLICY_NAME, POLICY_VERSION)
    print(f"  Loaded in {time.time() - t0:.2f}s — {policy_index_2.chunk_count} chunks.")

    positive_scores = run_query_batch(
        policy_index, "Positive queries (policy should have evidence)", POSITIVE_QUERIES
    )
    negative_scores = run_query_batch(
        policy_index, "Negative queries (policy should NOT have evidence)", NEGATIVE_QUERIES
    )

    print("\n=== Score summary (for Phase 8 threshold-tuning later) ===")
    if positive_scores:
        print(
            f"  Positive queries — avg top-1 score: {sum(positive_scores)/len(positive_scores):.3f}, "
            f"min: {min(positive_scores):.3f}, max: {max(positive_scores):.3f}"
        )
    if negative_scores:
        print(
            f"  Negative queries — avg top-1 score: {sum(negative_scores)/len(negative_scores):.3f}, "
            f"min: {min(negative_scores):.3f}, max: {max(negative_scores):.3f}"
        )
    print(
        "  NOTE: FAISS always returns its nearest neighbors, even for topics the\n"
        "  policy never mentions — a low score is the only signal the match is\n"
        "  weak. An actual cutoff is deferred to Phase 8, once we've also seen\n"
        "  this distribution on a real policy."
    )

    print("\n=== Validation checks ===")
    all_passed = True

    all_passed &= check(
        "Vector count matches chunk count",
        policy_index.index.ntotal == policy_index.chunk_count == len(clauses),
    )

    all_query_results = [
        r for q in POSITIVE_QUERIES + NEGATIVE_QUERIES for r in search_policy(policy_index, q, top_k=3)
    ]
    all_passed &= check(
        "Every search result belongs to the correct document_id",
        all(r.document_id == document.document_id for r in all_query_results),
    )

    dedup_ok = True
    for q in POSITIVE_QUERIES:
        texts = [r.text for r in search_policy(policy_index, q, top_k=5)]
        if len(set(texts)) != len(texts):
            dedup_ok = False
    all_passed &= check("No exact-duplicate chunk text within any single query's results", dedup_ok)

    try:
        search_policy(policy_index, "   ", top_k=3)
        all_passed &= check("Empty query raises a clear error", False)
    except ValueError as exc:
        all_passed &= check(f"Empty query raises a clear error ({exc})", True)

    all_passed &= check(
        "Reloaded index matches the freshly built one (chunk count)",
        policy_index_2.chunk_count == policy_index.chunk_count,
    )

    print()
    if all_passed:
        print("Phase 4 environment check: PASS")
    else:
        print("Phase 4 environment check: FAIL (see [FAIL] lines above)")
        sys.exit(1)


if __name__ == "__main__":
    main()
