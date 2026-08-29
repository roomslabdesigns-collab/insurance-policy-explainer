"""
Phase 11 — End-to-end pipeline evaluation.

Calls the ACTUAL user-facing pipeline (app.rag.generate_grounded_response)
for every golden-dataset question -- not a mocked or partial version of it.
Retrieval, evidence selection, the local LLM, and Phase 8's guardrails all
run exactly as they do for a real user; this module only measures and
classifies what comes out the other end.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List

import pandas as pd

from ..rag.answer_generator import generate_grounded_response
from ..rag.retrieval import rank_of_first_match, retrieve
from ..rag.vector_store import PolicyIndex
from .metrics import classify, determine_failure_stage
from .retrieval_evaluator import PREAMBLE_SENTINEL, UNSUPPORTED_SENTINEL  # noqa: F401 (re-exported for callers)
from .retrieval_evaluator import _expected_clause_set  # reuse Phase 5's dataset-parsing convention verbatim

GOLDEN_E2E_DATASET_PATH = Path(__file__).resolve().parent / "golden_end_to_end_dataset.csv"

BROAD_RETRIEVAL_K = 10  # evaluation-only instrumentation call, larger than what's actually sent to the LLM

REQUIRED_COLUMNS = {
    "question", "expected_status", "expected_clause_number", "expected_page_number",
    "expected_key_facts", "should_abstain", "category", "notes",
}


def load_golden_e2e_dataset(path: Path = GOLDEN_E2E_DATASET_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Golden dataset is missing required columns: {sorted(missing)}")
    df["should_abstain"] = df["should_abstain"].str.strip().str.upper() == "TRUE"
    return df


def _check_key_facts(answer_text: str, raw_key_facts: str) -> bool:
    """Case-insensitive substring presence check for each '|'-separated
    expected fact. An empty spec always passes -- not every question needs one."""
    facts = [f.strip() for f in raw_key_facts.split("|") if f.strip()]
    if not facts:
        return True
    lowered = answer_text.lower()
    return all(fact.lower() in lowered for fact in facts)


def _is_unsupported_marker(expected_raw: str) -> bool:
    """A blank cell is treated the same as the explicit NONE sentinel --
    forgiving of a dataset maintainer leaving it empty for an abstain-
    expected row, rather than silently mis-scoring it as a citation miss."""
    return expected_raw.strip().upper() in (UNSUPPORTED_SENTINEL, "")


def _check_citation(actual_clause_number, expected_raw: str) -> bool:
    if _is_unsupported_marker(expected_raw):
        return actual_clause_number is None
    expected_set = _expected_clause_set(expected_raw)
    if actual_clause_number is None:
        return False
    return actual_clause_number in expected_set


def evaluate_one(policy_index: PolicyIndex, item: pd.Series) -> dict:
    question = item["question"]
    expected_status = item["expected_status"]
    expected_clause_raw = item["expected_clause_number"]
    should_abstain = bool(item["should_abstain"])

    # Instrumentation-only broad retrieval, to separate "never retrieved at
    # all" from "retrieved, but ranked outside what actually got sent to
    # the LLM" -- generate_grounded_response's own retrieval already only
    # returns the top few chunks that WERE sent.
    t_retrieval_start = time.time()
    broad_results = retrieve(policy_index, question, max_k=BROAD_RETRIEVAL_K)
    retrieval_time = time.time() - t_retrieval_start

    is_unsupported = _is_unsupported_marker(expected_clause_raw)
    expected_set = [] if is_unsupported else _expected_clause_set(expected_clause_raw)
    retrieval_correct = True if is_unsupported else rank_of_first_match(broad_results, expected_set) is not None

    t_total_start = time.time()
    response = generate_grounded_response(policy_index, question)
    total_time = time.time() - t_total_start

    evidence_selection_correct = (
        True if is_unsupported
        else rank_of_first_match(response.retrieved_results, expected_set) is not None
    )

    actual_clause_number = response.citation.clause_number if response.citation else None
    status_correct = response.status == expected_status
    citation_correct = _check_citation(actual_clause_number, expected_clause_raw)
    key_facts_ok = _check_key_facts(response.answer_text, item["expected_key_facts"])

    classification = classify(
        actual_status=response.status,
        validation_passed=response.validation_passed,
        should_abstain=should_abstain,
        status_correct=status_correct,
        citation_correct=citation_correct,
        key_facts_ok=key_facts_ok,
    )
    failure_stage = determine_failure_stage(classification, retrieval_correct, evidence_selection_correct)

    llm_time = response.response_time_seconds or 0.0

    return {
        "question": question,
        "category": item["category"],
        "expected_status": expected_status,
        "expected_clause_number": expected_clause_raw,
        "should_abstain": should_abstain,
        "actual_status": response.status,
        "actual_clause_number": actual_clause_number or "",
        "confidence_label": response.confidence_label,
        "validation_passed": response.validation_passed,
        "status_correct": status_correct,
        "citation_correct": citation_correct,
        "key_facts_ok": key_facts_ok,
        "retrieval_correct": retrieval_correct,
        "evidence_selection_correct": evidence_selection_correct,
        "classification": classification,
        "failure_stage": failure_stage,
        "retrieval_time_seconds": retrieval_time,
        "llm_time_seconds": llm_time,
        "total_time_seconds": total_time,
        "retrieved_chunk_count": len(response.retrieved_results),
        "prompt_tokens": response.prompt_tokens,
        "completion_tokens": response.completion_tokens,
        "answer_char_count": len(response.answer_text),
        "answer_text": response.answer_text,
        "reliability_note": response.reliability_note,
        "notes": item["notes"],
    }


def evaluate_all(policy_index: PolicyIndex, dataset: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = [evaluate_one(policy_index, item) for _, item in dataset.iterrows()]
    return pd.DataFrame(rows)
