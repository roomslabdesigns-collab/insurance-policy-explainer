"""
Phase 11 — Run persistence, regression comparison, and reporting.

Every evaluation run is saved under a timestamped run_id, never
overwriting a previous run -- so "did this change actually help?" is
always answerable by comparing two real runs, not memory or guesswork.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from .metrics import compute_summary_metrics

RUNS_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "evaluation" / "e2e_runs"

COMPARISON_METRICS = [
    "answer_accuracy",
    "status_accuracy",
    "citation_accuracy",
    "retrieval_success_rate",
    "appropriate_abstention_rate",
    "incorrect_abstention_rate",
    "wrong_but_confident_rate",
    "validation_failure_rate",
    "avg_response_time_seconds",
]


def save_run(results_df: pd.DataFrame, policy_name: str, document_id: str) -> str:
    """Persists one run's full results + summary. Returns the new run_id."""
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(run_dir / "results.csv", index=False)

    summary_record = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "policy_name": policy_name,
        "document_id": document_id,
        "metrics": compute_summary_metrics(results_df),
        "failure_stage_counts": (
            results_df.loc[results_df["failure_stage"] != "", "failure_stage"].value_counts().to_dict()
        ),
        "classification_counts": results_df["classification"].value_counts().to_dict(),
    }
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_record, f, indent=2, ensure_ascii=False)

    with open(RUNS_ROOT / "latest_run_id.txt", "w", encoding="utf-8") as f:
        f.write(run_id)

    return run_id


def list_runs() -> list:
    """Run IDs, oldest to newest (timestamp-based names sort naturally)."""
    if not RUNS_ROOT.exists():
        return []
    return sorted(p.name for p in RUNS_ROOT.iterdir() if p.is_dir())


def get_latest_run_id() -> Optional[str]:
    runs = list_runs()
    return runs[-1] if runs else None


def get_previous_run_id(run_id: str) -> Optional[str]:
    runs = list_runs()
    if run_id not in runs:
        return None
    idx = runs.index(run_id)
    return runs[idx - 1] if idx > 0 else None


def load_run_summary(run_id: str) -> dict:
    with open(RUNS_ROOT / run_id / "summary.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_run_results(run_id: str) -> pd.DataFrame:
    return pd.read_csv(RUNS_ROOT / run_id / "results.csv")


def compare_runs(previous_run_id: str, current_run_id: str) -> pd.DataFrame:
    """Section 13: lightweight, explicit before/after metric comparison."""
    prev = load_run_summary(previous_run_id)["metrics"]
    curr = load_run_summary(current_run_id)["metrics"]
    rows = []
    for key in COMPARISON_METRICS:
        p, c = prev.get(key), curr.get(key)
        numeric = isinstance(p, (int, float)) and isinstance(c, (int, float)) and p == p and c == c
        rows.append({"metric": key, "previous": p, "current": c, "delta": (c - p) if numeric else None})
    return pd.DataFrame(rows)


def _fmt_pct(value) -> str:
    return f"{value:.1%}" if isinstance(value, (int, float)) and value == value else "n/a"


def generate_text_report(run_id: str) -> str:
    """
    Section 11: a readable summary generated ENTIRELY from this run's own
    numbers -- every claim traces back to a specific metric, never an
    independent assertion.
    """
    summary = load_run_summary(run_id)
    m = summary["metrics"]
    stage_counts = summary.get("failure_stage_counts", {})

    lines = [
        f"# End-to-End Evaluation Report — Run {run_id}",
        f"Policy: {summary['policy_name']}  |  Questions evaluated: {m['total_questions']}",
        "",
        "## Overall System Performance",
        f"- Answer Accuracy: {_fmt_pct(m['answer_accuracy'])}",
        f"- Status Accuracy: {_fmt_pct(m['status_accuracy'])}",
        f"- Citation Accuracy: {_fmt_pct(m.get('citation_accuracy'))}",
        f"- Retrieval Success Rate: {_fmt_pct(m['retrieval_success_rate'])}",
        f"- Evidence Selection Success Rate: {_fmt_pct(m['evidence_selection_success_rate'])}",
        f"- Appropriate Abstention Rate: {_fmt_pct(m['appropriate_abstention_rate'])}",
        f"- Incorrect Abstention Rate: {_fmt_pct(m['incorrect_abstention_rate'])}",
        f"- **Wrong-but-Confident Rate: {_fmt_pct(m['wrong_but_confident_rate'])}** (the single most important safety metric)",
        f"- Validation Failure Rate: {_fmt_pct(m['validation_failure_rate'])}",
        (
            f"- Avg response time: {m['avg_response_time_seconds']:.2f}s "
            f"(median {m['median_response_time_seconds']:.2f}s, max {m['max_response_time_seconds']:.2f}s)"
        ),
        f"- Avg chunks retrieved per question: {m['avg_chunks_retrieved']:.1f}",
        f"- Questions answered with ZERO LLM calls (pre-LLM gate): {m['questions_answered_with_zero_llm_calls']}",
        "",
        "## What Works Well",
    ]

    positives = []
    if m["appropriate_abstention_rate"] == m["appropriate_abstention_rate"] and m["appropriate_abstention_rate"] >= 0.8:
        positives.append(f"Appropriate abstention on unsupported questions is strong ({_fmt_pct(m['appropriate_abstention_rate'])}).")
    citation_acc = m.get("citation_accuracy")
    if citation_acc == citation_acc and citation_acc >= 0.8:
        positives.append(f"Citation accuracy is strong ({_fmt_pct(citation_acc)}) -- displayed sources reliably point to the right clause.")
    if m["wrong_but_confident_rate"] <= 0.05:
        positives.append(f"Wrong-but-confident rate is low ({_fmt_pct(m['wrong_but_confident_rate'])}) -- the guardrails are catching most bad answers.")
    if not positives:
        positives.append("No metric cleared the bar for a positive callout this run — see failure modes below.")
    lines.extend(f"- {p}" for p in positives)

    lines.append("")
    lines.append("## Main Failure Modes")
    if stage_counts:
        for stage, count in sorted(stage_counts.items(), key=lambda x: -x[1]):
            lines.append(f"- {stage}: {count} question(s)")
    else:
        lines.append("- None — every question was classified Correct or an appropriate abstention.")

    lines.append("")
    lines.append("## Recommended Improvements (priority order)")
    recs = []
    total = max(1, m["total_questions"])
    if m["wrong_but_confident_rate"] > 0.05:
        recs.append("Wrong-but-confident rate is above 5% — prioritize this before anything else; it's the most dangerous failure mode.")
    if stage_counts.get("Generation Failure", 0) >= max(2, total * 0.1):
        recs.append("The LLM misinterprets available evidence on several questions — review the prompt wording, or evaluate the 3B model.")
    if stage_counts.get("Retrieval Failure", 0) > 0:
        recs.append("Some expected evidence was never retrieved at all — revisit chunking/embedding for the affected categories.")
    if stage_counts.get("Evidence Selection Failure", 0) > 0:
        recs.append("Expected evidence was retrieved but ranked outside the chunks actually sent to the LLM — consider raising DEFAULT_CONTEXT_CHUNKS.")
    if not recs:
        recs.append("No high-priority issues surfaced by this run's failures.")
    lines.extend(f"{i + 1}. {r}" for i, r in enumerate(recs))

    return "\n".join(lines)
