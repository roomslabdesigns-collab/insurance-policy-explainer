"""
Phase 6/12 — Centralized, overridable configuration.

Every value here can be overridden with an environment variable of the same
name, either exported in the shell (e.g. `$env:LLM_SERVER_URL = "..."` in
PowerShell) or placed in a `.env` file at the project root (copy
`.env.example` to `.env` and edit it -- `.env` is gitignored, so local
overrides never get committed). Values actually already set in the
environment always win over `.env`, so a real deployment can still
override via the shell without editing files.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env", override=False)


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# --- Local LLM server (llama_cpp.server: an OpenAI-compatible REST API
#     wrapping llama.cpp's inference engine) ---
# Qwen2.5-3B-Instruct is the default as of the Phase 12 model comparison:
# measured (on the same 41-question evaluation, after fixing an unrelated
# evidence-ID parsing gap) to beat the earlier 1.5B default on every
# accuracy metric -- Answer Accuracy 39.0% vs 34.1%, Citation Accuracy
# 73.3% vs 66.7%, Appropriate Abstention 100% vs 85.7% -- while keeping an
# identical, clean 0% Wrong-but-Confident rate. The cost is real: roughly
# 2x the response time (~6.5s vs ~3.1s avg) and ~700MB more RAM. See
# docs/evaluation.md for the full comparison. A lighter 1.5B option remains
# available -- see README's "Sample Data" / setup notes for the download
# command -- for machines where the extra RAM/latency isn't worth it.
LLM_SERVER_URL = _env_str("LLM_SERVER_URL", "http://127.0.0.1:8000")
LLM_MODEL_NAME = _env_str("LLM_MODEL_NAME", "qwen2.5-3b-instruct")
LLM_MODEL_PATH = _env_str(
    "LLM_MODEL_PATH", str(PROJECT_ROOT / "models" / "qwen2.5-3b-instruct-q4_k_m.gguf")
)
LLM_CONTEXT_SIZE = _env_int("LLM_CONTEXT_SIZE", 2048)  # set when the SERVER starts, not per-request
# 90s (not the earlier 60s): the 3B model occasionally needs more than 60s
# on 8GB CPU-only hardware, observed directly during evaluation.
LLM_TIMEOUT_SECONDS = _env_int("LLM_TIMEOUT_SECONDS", 90)
LLM_MAX_TOKENS = _env_int("LLM_MAX_TOKENS", 300)
LLM_TEMPERATURE = _env_float("LLM_TEMPERATURE", 0.1)  # low: this is grounded extraction, not creative writing

# --- Evidence context sent to the LLM (token-efficiency controls) ---
DEFAULT_CONTEXT_CHUNKS = _env_int("DEFAULT_CONTEXT_CHUNKS", 3)
MAX_EVIDENCE_CHARS_PER_CHUNK = _env_int("MAX_EVIDENCE_CHARS_PER_CHUNK", 500)

# --- Evidence sufficiency gate (Phase 7), run in Python BEFORE any LLM call ---
# Phase 5's golden-dataset evaluation found positive and negative query
# top-1 scores overlapping in roughly the 0.37-0.48 range (a correct-but-
# plainly-worded answer scored 0.374; an unsupported question scored
# 0.476) -- no single threshold cleanly separates them. This value is
# deliberately a low FLOOR, not a final trust decision: it only needs to
# catch genuinely unrelated evidence (Phase 6 observed a 0.04-score match
# for a completely off-topic question), leaving the nuanced borderline
# cases for the LLM's own status classification plus the cross-checks in
# app.rag.answer_generator.
MIN_EVIDENCE_SCORE = _env_float("MIN_EVIDENCE_SCORE", 0.30)
