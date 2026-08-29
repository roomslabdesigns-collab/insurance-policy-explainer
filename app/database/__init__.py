"""Local SQLite storage: user feedback and lightweight analytics events
(Phase 9/11). The full evaluation-run schema lives under data/evaluation/
(JSON/CSV, see app/evaluation) rather than here -- kept queryable-by-SQL
data separate from timestamped evaluation-run archives."""

from .db import (
    DB_PATH,
    count_evidence_views,
    get_all_feedback,
    get_feedback_summary,
    init_db,
    log_evidence_view,
    save_feedback,
)

__all__ = [
    "DB_PATH",
    "count_evidence_views",
    "get_all_feedback",
    "get_feedback_summary",
    "init_db",
    "log_evidence_view",
    "save_feedback",
]
