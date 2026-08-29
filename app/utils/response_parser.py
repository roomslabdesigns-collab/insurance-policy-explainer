"""
Phase 7 — Parses the local LLM's structured response into a typed result.

Written defensively: a 1.5B local model will not always follow formatting
instructions perfectly. This parser is deliberately hybrid -- it scans for
the requested tagged fields (STATUS:/ANSWER:/EVIDENCE_ID:) first, and also
tries to pull the same fields out of a JSON object if the model wraps its
answer in one anyway (small models are sometimes RLHF'd toward JSON despite
being asked for something else). It never raises on malformed input --
callers get a ParsedAnswer with is_valid_format=False instead, so a
malformed response degrades to a safe fallback rather than a crash.

Deliberately has ZERO dependency on app.rag or app.llm -- this module only
knows about text in, a typed record out. Keeping it dependency-free is what
lets app.rag.answer_generator depend on it without any risk of a cycle.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

STATUS_COVERED = "Covered"
STATUS_EXCLUDED = "Explicitly Excluded"
STATUS_NOT_MENTIONED = "Not Mentioned"
STATUS_INSUFFICIENT = "Insufficient Evidence"

ALLOWED_LLM_STATUSES = {STATUS_COVERED, STATUS_EXCLUDED, STATUS_NOT_MENTIONED, STATUS_INSUFFICIENT}

# Tolerates near-miss phrasing the model might produce despite instructions,
# normalized back to the exact allowed status string.
_STATUS_ALIASES = {
    "covered": STATUS_COVERED,
    "explicitly excluded": STATUS_EXCLUDED,
    "excluded": STATUS_EXCLUDED,
    "not mentioned": STATUS_NOT_MENTIONED,
    "not addressed": STATUS_NOT_MENTIONED,
    "insufficient evidence": STATUS_INSUFFICIENT,
    "insufficient": STATUS_INSUFFICIENT,
    "unclear": STATUS_INSUFFICIENT,
}


@dataclass
class ParsedAnswer:
    """Result of parsing one raw LLM response string."""

    status: Optional[str]        # normalized to one of ALLOWED_LLM_STATUSES, or None if unparseable
    answer_text: str
    evidence_id: Optional[str]   # raw label as the model wrote it (e.g. "E1"), or None
    is_valid_format: bool        # True only if BOTH status and a non-empty answer were extracted
    raw_response: str            # untouched LLM text, kept for logging/debugging


def _normalize_status(raw: str) -> Optional[str]:
    cleaned = raw.strip().strip('"').strip("*").lower()
    if cleaned in _STATUS_ALIASES:
        return _STATUS_ALIASES[cleaned]
    for key, value in _STATUS_ALIASES.items():
        if key in cleaned:
            return value
    return None


def _is_bare_status_echo(answer_text: str) -> bool:
    """
    Catches a reproducible glitch seen twice across evaluation runs: the
    model sometimes emits a bare status word/phrase as the ANSWER itself
    (answer_text literally "Insufficient Evidence", with no explanation)
    instead of an actual sentence. A one-to-three-word echo of a status
    label is never a real answer, so this is treated as an unparseable
    response -- degrading safely to a validation-failure fallback --
    rather than trusted and shown to the user at face value.
    """
    cleaned = answer_text.strip().strip(".").lower()
    return cleaned in _STATUS_ALIASES


def _try_parse_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    candidate = match.group(0)
    for attempt in (candidate, re.sub(r",\s*([}\]])", r"\1", candidate)):
        try:
            data = json.loads(attempt)
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, ValueError):
            continue
    return {}


_FIELD_RE = re.compile(
    r"(STATUS|ANSWER|EVIDENCE_ID)\s*:\s*(.*?)(?=\n\s*(?:STATUS|ANSWER|EVIDENCE_ID)\s*:|\Z)",
    re.IGNORECASE | re.DOTALL,
)


def _try_parse_tagged(text: str) -> dict:
    """
    Order-agnostic: scans for STATUS:/ANSWER:/EVIDENCE_ID: labels wherever
    they appear and in whatever order the model produced them, stopping
    each field's capture at the next recognized label (or end of text). A
    small local model won't always emit fields in the exact requested
    order or always include all three -- this tolerates that instead of
    silently losing a field the way a rigid fixed-order regex would.
    """
    fields: dict = {}
    for match in _FIELD_RE.finditer(text):
        key = match.group(1).upper()
        value = match.group(2).strip()
        if key == "STATUS":
            fields["status"] = value.splitlines()[0].strip()
        elif key == "ANSWER":
            fields["answer"] = value.strip()
        elif key == "EVIDENCE_ID":
            fields["evidence_id"] = value.splitlines()[0].strip()
    return fields


def parse_llm_response(raw_response: str) -> ParsedAnswer:
    """
    Extract status/answer/evidence_id from a raw LLM response, trying the
    tagged format first (what the prompt asks for) and falling back to a
    permissive JSON scan if tagged fields are missing.
    """
    fields = _try_parse_tagged(raw_response)
    if not fields.get("status") or not fields.get("answer"):
        fields = {**_try_parse_json(raw_response), **fields}

    status_raw = fields.get("status")
    answer_text = str(fields.get("answer") or "").strip()
    evidence_id = fields.get("evidence_id")

    if isinstance(evidence_id, str):
        evidence_id = evidence_id.strip().strip('"').rstrip(".,")
        if evidence_id.upper() in ("NONE", "N/A", ""):
            evidence_id = None

    status = _normalize_status(status_raw) if status_raw else None
    is_valid_format = (
        status is not None and bool(answer_text) and not _is_bare_status_echo(answer_text)
    )

    return ParsedAnswer(
        status=status,
        answer_text=answer_text,
        evidence_id=evidence_id,
        is_valid_format=is_valid_format,
        raw_response=raw_response,
    )
