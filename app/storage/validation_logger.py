"""
Phase 8 — Lightweight validation event logging (JSONL).

Appends one JSON object per generated answer. Deliberately NOT a database:
this is a debugging/evaluation trail, not a queryable store -- Phase 9's
user-facing feedback storage and Phase 11's full evaluation framework will
use a real SQLite schema in app/database/. Kept append-only and
best-effort: a logging failure must never break answer generation for the
user.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

DEFAULT_LOG_PATH = (
    Path(__file__).resolve().parent.parent.parent / "data" / "logs" / "validation_log.jsonl"
)


@dataclass
class RetrievedEvidenceLogEntry:
    label: str
    clause_number: str
    page_number: int
    score: float


def log_validation_event(
    *,
    document_id: str,
    policy_name: str,
    question: str,
    retrieved_evidence: List[RetrievedEvidenceLogEntry],
    selected_evidence_id: Optional[str],
    status: str,
    validation_passed: bool,
    reliability_note: str,
    answer_text: str,
    response_time_seconds: Optional[float],
    log_path: Path = DEFAULT_LOG_PATH,
) -> None:
    """
    Records what a human reviewing this system later would want to know:
    what was asked, what was retrieved, what the model picked, and whether
    the trust layer accepted it -- without storing full evidence text
    (already recoverable from document_id + clause_number if needed).
    """
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "document_id": document_id,
        "policy_name": policy_name,
        "question": question,
        "retrieved_evidence": [asdict(e) for e in retrieved_evidence],
        "selected_evidence_id": selected_evidence_id,
        "status": status,
        "validation_passed": validation_passed,
        "reliability_note": reliability_note,
        "answer_text": answer_text,
        "answer_char_count": len(answer_text),
        "response_time_seconds": response_time_seconds,
    }
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.warning("Failed to write validation log entry: %s", exc)
