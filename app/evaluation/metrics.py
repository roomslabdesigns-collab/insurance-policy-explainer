"""
Phase 11 — Classification and metric computation for the end-to-end evaluation.

Answer correctness is graded WITHOUT an LLM judge, by design: a deterministic
composite of three checks the pipeline already produces --
  1. status_correct    -- does the final status match the expected status?
  2. citation_correct   -- does the displayed citation point to the expected clause?
  3. key_facts_ok        -- do expected keywords/numbers appear in the answer text?
This is the same "simple, deterministic, documented limitations" philosophy
as Phase 8's numeric claim-checker -- not a substitute for human review, but
an honest, explainable, reproducible proxy.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

CLASS_CORRECT = "Correct"
CLASS_INCORRECT = "Incorrect"
CLASS_APPROPRIATE_ABSTENTION = "Appropriate Abstention"
CLASS_INCORRECT_ABSTENTION = "Incorrect Abstention"
CLASS_WRONG_BUT_CONFIDENT = "Wrong-but-Confident"
CLASS_CITATION_FAILURE = "Citation Failure"
CLASS_VALIDATION_FAILURE = "Validation Failure"

ALL_CLASSIFICATIONS = [
    CLASS_CORRECT,
    CLASS_INCORRECT,
    CLASS_APPROPRIATE_ABSTENTION,
    CLASS_INCORRECT_ABSTENTION,
    CLASS_WRONG_BUT_CONFIDENT,
    CLASS_CITATION_FAILURE,
    CLASS_VALIDATION_FAILURE,
]

# Statuses that represent the system declining to assert a specific
# coverage/exclusion determination. "Not Mentioned" IS included: it only
# ever claims absence ("the policy doesn't address this"), never a
# specific Covered/Excluded conclusion, so it is exactly as much of a
# hedge as "Insufficient Evidence" -- it just names a different reason
# (topic absent, vs. evidence ambiguous). An earlier version of this
# constant excluded it, which meant a genuinely correct, safe "Not
# Mentioned" answer to an unsupported question (e.g. "Is there a maternity
# waiting period?" -> "There is no mention of this in the evidence") was
# being scored as Wrong-but-Confident -- the single most safety-critical
# metric in this whole evaluation -- purely from a classification gap, not
# an actual dangerous answer. Found via the 3B-model comparison run.
ABSTENTION_STATUSES = {"No Evidence Found", "Insufficient Evidence", "Not Mentioned"}

STAGE_RETRIEVAL_FAILURE = "Retrieval Failure"
STAGE_EVIDENCE_SELECTION_FAILURE = "Evidence Selection Failure"
STAGE_GENERATION_FAILURE = "Generation Failure"
STAGE_CITATION_FAILURE = "Citation Failure"
STAGE_GUARDRAIL_FAILURE = "Guardrail Failure"


def classify(
    *,
    actual_status: str,
    validation_passed: bool,
    should_abstain: bool,
    status_correct: bool,
    citation_correct: bool,
    key_facts_ok: bool,
) -> str:
    """
    Priority order matters: a pipeline-level validation rejection is
    reported as such regardless of where the fallback status happened to
    land; only then do we check whether the (possibly-fallback) status
    represents a correct abstention, a correct substantive answer, or a
    wrong one.
    """
    if not validation_passed:
        return CLASS_VALIDATION_FAILURE

    if actual_status in ABSTENTION_STATUSES:
        return CLASS_APPROPRIATE_ABSTENTION if should_abstain else CLASS_INCORRECT_ABSTENTION

    if status_correct and citation_correct and key_facts_ok:
        return CLASS_CORRECT
    if status_correct and not citation_correct:
        return CLASS_CITATION_FAILURE
    if status_correct and citation_correct and not key_facts_ok:
        return CLASS_INCORRECT

    return CLASS_WRONG_BUT_CONFIDENT


def determine_failure_stage(
    classification: str, retrieval_correct: bool, evidence_selection_correct: bool
) -> str:
    """
    Section 6: attribute a non-Correct result to the pipeline stage most
    responsible, so different failures point at different fixes.

    Appropriate Abstention is a GOOD outcome (correctly declining to
    guess), not a failure -- it gets no stage, same as Correct.

    Guardrail Failure is reserved for exactly what the spec defines it as:
    "an unsafe or unsupported answer passed validation" -- i.e. a
    Wrong-but-Confident result, since by construction it has
    validation_passed=True (anything the guardrails actually caught became
    Validation Failure instead, landing it in Generation Failure below,
    since the underlying cause is the same LLM misjudgment, just one that
    was safely contained rather than one that slipped through).
    """
    if classification in (CLASS_CORRECT, CLASS_APPROPRIATE_ABSTENTION):
        return ""
    if not retrieval_correct:
        return STAGE_RETRIEVAL_FAILURE
    if not evidence_selection_correct:
        return STAGE_EVIDENCE_SELECTION_FAILURE
    if classification == CLASS_WRONG_BUT_CONFIDENT:
        return STAGE_GUARDRAIL_FAILURE
    if classification == CLASS_CITATION_FAILURE:
        return STAGE_CITATION_FAILURE
    if classification in (CLASS_INCORRECT, CLASS_INCORRECT_ABSTENTION, CLASS_VALIDATION_FAILURE):
        return STAGE_GENERATION_FAILURE
    return ""


def compute_summary_metrics(df: pd.DataFrame) -> dict:
    """Section 4's headline metrics, computed over one evaluation run's results."""
    n = len(df)
    if n == 0:
        return {"total_questions": 0}

    should_abstain_n = int(df["should_abstain"].sum())
    should_answer_n = n - should_abstain_n
    has_expected_citation = df["expected_clause_number"].astype(str).str.upper() != "NONE"

    def safe_mean(series: pd.Series) -> Optional[float]:
        return float(series.mean()) if len(series) else float("nan")

    return {
        "total_questions": n,
        "answer_accuracy": safe_mean(df["classification"] == CLASS_CORRECT),
        "status_accuracy": safe_mean(df["status_correct"]),
        "citation_accuracy": safe_mean(df.loc[has_expected_citation, "citation_correct"]),
        "retrieval_success_rate": safe_mean(df["retrieval_correct"]),
        "evidence_selection_success_rate": safe_mean(df["evidence_selection_correct"]),
        "appropriate_abstention_rate": (
            (df["classification"] == CLASS_APPROPRIATE_ABSTENTION).sum() / should_abstain_n
            if should_abstain_n else float("nan")
        ),
        "incorrect_abstention_rate": (
            (df["classification"] == CLASS_INCORRECT_ABSTENTION).sum() / should_answer_n
            if should_answer_n else float("nan")
        ),
        "wrong_but_confident_rate": safe_mean(df["classification"] == CLASS_WRONG_BUT_CONFIDENT),
        "citation_failure_rate": safe_mean(df["classification"] == CLASS_CITATION_FAILURE),
        "validation_failure_rate": safe_mean(df["classification"] == CLASS_VALIDATION_FAILURE),
        "avg_response_time_seconds": safe_mean(df["total_time_seconds"]),
        "median_response_time_seconds": float(df["total_time_seconds"].median()) if n else float("nan"),
        "max_response_time_seconds": float(df["total_time_seconds"].max()) if n else float("nan"),
        "avg_prompt_tokens": safe_mean(df["prompt_tokens"].dropna()),
        "avg_completion_tokens": safe_mean(df["completion_tokens"].dropna()),
        "avg_chunks_retrieved": safe_mean(df["retrieved_chunk_count"]),
        "questions_answered_with_zero_llm_calls": int((df["prompt_tokens"].isna()).sum()),
    }
