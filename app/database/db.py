"""
Phase 9/11 — SQLite storage for user feedback and lightweight product
analytics (evidence-view events).

Kept intentionally minimal: real per-question/answer history stays in
Phase 8's JSONL validation log; this file only stores what a dashboard
needs to aggregate cheaply with plain SQL.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "app.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    document_id TEXT NOT NULL,
    policy_name TEXT NOT NULL,
    question TEXT NOT NULL,
    status TEXT NOT NULL,
    feedback_type TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS evidence_views (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    document_id TEXT NOT NULL,
    question TEXT NOT NULL
);
"""


def _get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(_SCHEMA)
    return conn


def init_db() -> None:
    """Create the database file and schema if they don't exist yet."""
    _get_connection().close()


def save_feedback(
    document_id: str, policy_name: str, question: str, status: str, feedback_type: str
) -> None:
    """
    Stores only what's needed to evaluate answer quality later -- no
    personal information, no IP address, no user identifier.
    """
    conn = _get_connection()
    try:
        conn.execute(
            "INSERT INTO feedback (timestamp, document_id, policy_name, question, status, feedback_type) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                datetime.now(timezone.utc).isoformat(),
                document_id,
                policy_name,
                question,
                status,
                feedback_type,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_feedback_summary(document_id: Optional[str] = None) -> List[Tuple[str, int]]:
    """Feedback-type counts, optionally scoped to one policy."""
    conn = _get_connection()
    try:
        query = "SELECT feedback_type, COUNT(*) FROM feedback"
        params: Tuple = ()
        if document_id:
            query += " WHERE document_id = ?"
            params = (document_id,)
        query += " GROUP BY feedback_type"
        return conn.execute(query, params).fetchall()
    finally:
        conn.close()


def get_all_feedback(document_id: Optional[str] = None) -> List[Tuple]:
    """Raw feedback rows (timestamp, document_id, policy_name, question,
    status, feedback_type), newest first -- for the analytics dashboard."""
    conn = _get_connection()
    try:
        query = (
            "SELECT timestamp, document_id, policy_name, question, status, feedback_type "
            "FROM feedback"
        )
        params: Tuple = ()
        if document_id:
            query += " WHERE document_id = ?"
            params = (document_id,)
        query += " ORDER BY timestamp DESC"
        return conn.execute(query, params).fetchall()
    finally:
        conn.close()


def log_evidence_view(document_id: str, question: str) -> None:
    """Records that a user clicked 'View Evidence in Policy' -- lets the
    dashboard report an honest evidence-view rate instead of omitting it."""
    conn = _get_connection()
    try:
        conn.execute(
            "INSERT INTO evidence_views (timestamp, document_id, question) VALUES (?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(), document_id, question),
        )
        conn.commit()
    finally:
        conn.close()


def count_evidence_views(document_id: Optional[str] = None) -> int:
    conn = _get_connection()
    try:
        query = "SELECT COUNT(*) FROM evidence_views"
        params: Tuple = ()
        if document_id:
            query += " WHERE document_id = ?"
            params = (document_id,)
        return conn.execute(query, params).fetchone()[0]
    finally:
        conn.close()
