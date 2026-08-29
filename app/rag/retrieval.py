"""
Phase 5 — Thin retrieval helpers for evaluation (and later, Phase 6/8
sufficiency checks).

Deliberately does NOT reimplement embedding or FAISS logic -- everything
here calls straight through to vector_store.search_policy. The only reason
this module exists is to retrieve once at the largest k a caller needs and
let it slice that single result list for smaller k values, instead of
re-embedding the same query multiple times (e.g. once each for Recall@1,
Recall@3, Recall@5).
"""

from __future__ import annotations

from typing import Iterable, List, Optional

from .vector_store import PolicyIndex, SearchResult, search_policy


def retrieve(policy_index: PolicyIndex, query: str, max_k: int = 5) -> List[SearchResult]:
    """
    Retrieve up to `max_k` results in a single embedding + FAISS call.
    Slice the returned list (e.g. results[:1], results[:3]) to evaluate
    Recall@1/@3/@5 without re-running the query multiple times.
    """
    return search_policy(policy_index, query, top_k=max_k)


def clause_in_results(results: Iterable[SearchResult], expected_clause_numbers: Iterable[str]) -> bool:
    """True if any of the expected clause numbers appears anywhere in `results`."""
    expected = {c.strip() for c in expected_clause_numbers}
    return any(r.clause_number in expected for r in results)


def rank_of_first_match(
    results: List[SearchResult], expected_clause_numbers: Iterable[str]
) -> Optional[int]:
    """1-based rank of the first result matching any expected clause number, or None."""
    expected = {c.strip() for c in expected_clause_numbers}
    for i, r in enumerate(results, start=1):
        if r.clause_number in expected:
            return i
    return None
