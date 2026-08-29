"""
Phase 8 — Lightweight, deterministic claim-to-evidence checking.

Explicitly NOT a semantic fact-checker. This catches one narrow, high-value
signal: a number in the answer (a waiting period, a monetary figure, a
day/month count) that does not appear anywhere in the cited evidence text.
It is a heuristic, not proof of a hallucination:

- False positives happen: a number can be legitimately restated in
  different units (e.g. "48 months" vs "4 years") and would be wrongly
  flagged.
- False negatives happen: an invented number that coincidentally matches a
  digit sequence elsewhere in the evidence text would slip through.

Given those limits, a flag here downgrades the answer to a cautious status
rather than being treated as definitive proof of a fabrication -- exactly
the "prefer abstention over a perfect fact-checker" trade-off this phase
calls for.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Set

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def extract_numbers(text: str) -> Set[str]:
    return set(_NUMBER_RE.findall(text))


@dataclass
class ClaimCheckResult:
    unsupported_numbers: List[str]
    is_high_risk: bool


def check_claim_support(answer_text: str, evidence_text: str) -> ClaimCheckResult:
    """Numbers present in the answer but absent from its cited evidence."""
    unsupported = sorted(extract_numbers(answer_text) - extract_numbers(evidence_text))
    return ClaimCheckResult(unsupported_numbers=unsupported, is_high_risk=bool(unsupported))
