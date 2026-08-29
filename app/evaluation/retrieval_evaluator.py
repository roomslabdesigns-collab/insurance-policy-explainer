"""
Phase 5 — Retrieval evaluation against a manually verified golden dataset.

No LLM is used anywhere in this module, including to judge correctness.
Every expected answer in golden_retrieval_dataset.csv was written by hand
by reading the actual policy text -- that's what makes Recall@K and MRR
computed here trustworthy numbers, rather than one AI grading another.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd

from ..rag.retrieval import rank_of_first_match, retrieve
from ..rag.vector_store import PolicyIndex

GOLDEN_DATASET_PATH = Path(__file__).resolve().parent / "golden_retrieval_dataset.csv"

UNSUPPORTED_SENTINEL = "NONE"
# Sentinel for "this chunk has no clause number" (the title/preamble text) --
# distinct from a genuinely blank/malformed CSV cell, so the two are never
# confused when parsing expected_clause_number.
PREAMBLE_SENTINEL = "PREAMBLE"

MAX_K = 5
DEFAULT_THRESHOLD_CANDIDATES = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]

REQUIRED_COLUMNS = {
    "question",
    "expected_clause_number",
    "expected_page_number",
    "expected_chunk_id",
    "category",
    "notes",
}


def load_golden_dataset(path: Path = GOLDEN_DATASET_PATH) -> pd.DataFrame:
    """
    Load the golden dataset. `expected_chunk_id` is optional per-row (most
    real-world editors of this file will know clause numbers from reading
    the PDF, not internal chunk IDs) -- `expected_clause_number` plus
    `expected_page_number` is the primary, human-verifiable match key.
    """
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Golden dataset is missing required columns: {sorted(missing)}")
    return df


def _expected_clause_set(raw: str) -> List[str]:
    """Split a possibly '|'-separated list of acceptable clause numbers."""
    parts = [c.strip() for c in raw.split("|")]
    return ["" if p.upper() == PREAMBLE_SENTINEL else p for p in parts if p != ""]


def evaluate_retrieval(
    policy_index: PolicyIndex, dataset: pd.DataFrame, max_k: int = MAX_K
) -> pd.DataFrame:
    """
    Run every golden-dataset question through retrieval ONCE (at
    top_k=max_k) and record enough detail to compute Recall@1/@3/@5, MRR,
    and the unsupported-question false-positive analysis from a single
    dataframe -- no re-querying per K value.
    """
    rows = []
    for _, item in dataset.iterrows():
        question = item["question"]
        is_unsupported = item["expected_clause_number"].strip().upper() == UNSUPPORTED_SENTINEL
        expected_clauses = [] if is_unsupported else _expected_clause_set(item["expected_clause_number"])

        results = retrieve(policy_index, question, max_k=max_k)

        rank = None if is_unsupported else rank_of_first_match(results, expected_clauses)
        reciprocal_rank = 0.0 if rank is None else 1.0 / rank
        top1 = results[0] if results else None

        rows.append(
            {
                "question": question,
                "category": item["category"],
                "expected_clause_number": item["expected_clause_number"],
                "is_unsupported": is_unsupported,
                "rank": rank,
                "reciprocal_rank": reciprocal_rank,
                "hit_top1": rank == 1,
                "hit_top3": rank is not None and rank <= 3,
                "hit_top5": rank is not None and rank <= 5,
                "top1_clause": (top1.clause_number or "(preamble)") if top1 else "",
                "top1_section": top1.section if top1 else "",
                "top1_page": top1.page_number if top1 else None,
                "top1_score": top1.score if top1 else None,
                "top3_clauses": ", ".join(r.clause_number or "(preamble)" for r in results[:3]),
                "top5_unique_clauses": len({r.clause_number for r in results[:5]}),
                "top5_unique_sections": len({r.section for r in results[:5]}),
                "notes": item.get("notes", ""),
            }
        )

    return pd.DataFrame(rows)


def compute_overall_metrics(results_df: pd.DataFrame, threshold: Optional[float] = None) -> dict:
    """
    Recall@1/@3/@5 and MRR over SUPPORTED questions only -- recall isn't a
    meaningful concept for a question the policy was never expected to
    answer. Unsupported-question behavior is reported separately.
    """
    supported = results_df[~results_df["is_unsupported"]]
    unsupported = results_df[results_df["is_unsupported"]]

    metrics = {
        "num_supported_questions": len(supported),
        "num_unsupported_questions": len(unsupported),
        "recall_at_1": supported["hit_top1"].mean() if len(supported) else float("nan"),
        "recall_at_3": supported["hit_top3"].mean() if len(supported) else float("nan"),
        "recall_at_5": supported["hit_top5"].mean() if len(supported) else float("nan"),
        "mrr": supported["reciprocal_rank"].mean() if len(supported) else float("nan"),
    }
    if threshold is not None and len(unsupported):
        metrics["threshold_used"] = threshold
        metrics["unsupported_false_positive_rate_at_threshold"] = (
            unsupported["top1_score"] >= threshold
        ).mean()
    return metrics


def compute_category_metrics(results_df: pd.DataFrame) -> pd.DataFrame:
    """Recall@K and MRR broken down by category (supported questions only)."""
    supported = results_df[~results_df["is_unsupported"]]
    return (
        supported.groupby("category")
        .agg(
            n=("question", "count"),
            recall_at_1=("hit_top1", "mean"),
            recall_at_3=("hit_top3", "mean"),
            recall_at_5=("hit_top5", "mean"),
            mrr=("reciprocal_rank", "mean"),
        )
        .reset_index()
    )


def analyze_threshold_candidates(
    results_df: pd.DataFrame, thresholds: Optional[List[float]] = None
) -> pd.DataFrame:
    """
    For each candidate threshold, report:
      - unsupported_false_positive_rate: fraction of UNSUPPORTED questions
        whose top-1 score would incorrectly clear the threshold.
      - correct_top1_retained_rate: fraction of correctly-answered SUPPORTED
        questions (hit_top1 already True) whose score would still clear the
        threshold -- i.e. not get wrongly discarded.
    This is the standard trade-off view used to pick a real threshold in
    Phase 8; it is analysis, not a decision made here.
    """
    thresholds = thresholds or DEFAULT_THRESHOLD_CANDIDATES
    supported_hits = results_df[(~results_df["is_unsupported"]) & (results_df["hit_top1"])]
    unsupported = results_df[results_df["is_unsupported"]]

    rows = []
    for t in thresholds:
        fp_rate = (unsupported["top1_score"] >= t).mean() if len(unsupported) else float("nan")
        retained = (supported_hits["top1_score"] >= t).mean() if len(supported_hits) else float("nan")
        rows.append(
            {
                "threshold": t,
                "unsupported_false_positive_rate": fp_rate,
                "correct_top1_retained_rate": retained,
            }
        )
    return pd.DataFrame(rows)


def recommend_threshold(threshold_df: pd.DataFrame) -> Optional[float]:
    """
    Simple, explainable recommendation: the lowest candidate threshold that
    drives the unsupported false-positive rate to zero while still
    retaining every correctly-answered supported question. Returns None if
    no candidate achieves both -- meaning positive/negative scores overlap
    and Phase 8 will need more than a single hard cutoff.
    """
    clean = threshold_df[
        (threshold_df["unsupported_false_positive_rate"] == 0.0)
        & (threshold_df["correct_top1_retained_rate"] == 1.0)
    ]
    return float(clean["threshold"].min()) if not clean.empty else None


def analyze_diversity(results_df: pd.DataFrame) -> dict:
    """
    Lightweight check (no reranking) for whether top-5 results tend to be
    repeats of the same clause/section rather than a genuinely diverse set
    -- important because a coverage clause's overriding exclusion usually
    lives in a *different* section, and low diversity means it may never
    surface even at top_k=5.
    """
    supported = results_df[~results_df["is_unsupported"]]
    repetitive = supported[supported["top5_unique_clauses"] < 5]
    return {
        "avg_unique_clauses_in_top5": supported["top5_unique_clauses"].mean() if len(supported) else float("nan"),
        "avg_unique_sections_in_top5": supported["top5_unique_sections"].mean() if len(supported) else float("nan"),
        "questions_with_repeated_clause_in_top5": len(repetitive),
        "repetitive_questions": repetitive["question"].tolist(),
    }


def get_failures(results_df: pd.DataFrame, suspicious_score_threshold: float = 0.5) -> pd.DataFrame:
    """
    Rows worth a human's attention:
      - supported questions that missed Recall@5 entirely
      - unsupported questions whose top-1 score looks suspiciously high
        (a candidate false positive)
    A Recall@1 miss that still succeeds within top-3/top-5 is informative
    but not a failure -- it's visible directly in the per-question results
    table (`rank` column) instead of being duplicated here.
    """
    supported_misses = results_df[(~results_df["is_unsupported"]) & (~results_df["hit_top5"])].copy()
    supported_misses["failure_type"] = "not_found_in_top5"

    suspicious_unsupported = results_df[
        results_df["is_unsupported"] & (results_df["top1_score"] >= suspicious_score_threshold)
    ].copy()
    suspicious_unsupported["failure_type"] = "unsupported_question_scored_high"

    failures = pd.concat([supported_misses, suspicious_unsupported], ignore_index=True)
    cols = [
        "question", "category", "expected_clause_number",
        "top1_clause", "top1_section", "top1_page", "top1_score",
        "failure_type", "notes",
    ]
    return failures[cols] if not failures.empty else failures
