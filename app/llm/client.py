"""
Phase 6 — Reusable client for the local llama.cpp LLM server.

Talks to the OpenAI-compatible /v1/chat/completions endpoint exposed by
`python -m llama_cpp.server`. The model is loaded exactly once, by that
server process, when it starts -- this module never loads a model itself,
it only ever sends short-lived HTTP requests. That keeps the Python
application (and any Streamlit reruns later) lightweight regardless of how
many questions get asked.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import requests

from .. import config

# NOTE: app.rag imports app.llm.client (Phase 7's answer_generator.py calls
# generate_completion). To avoid a top-level circular import (rag -> llm ->
# rag), the rag imports this module needs are deferred to inside
# answer_question() below, where they're only ever needed at call time, long
# after both packages have finished loading. Type hints stay accurate via
# the TYPE_CHECKING guard (harmless at runtime -- never actually imported).
if TYPE_CHECKING:
    from ..rag.vector_store import PolicyIndex, SearchResult


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------

class LLMError(Exception):
    """Base class for all local LLM communication failures."""


class LLMConnectionError(LLMError):
    """The llama.cpp server is not reachable at the configured URL."""


class LLMTimeoutError(LLMError):
    """The server did not respond within the configured timeout."""


class LLMResponseError(LLMError):
    """The server responded, but with an error or an empty/invalid completion."""


def _server_start_hint() -> str:
    return (
        f'  .\\venv\\Scripts\\python.exe -m llama_cpp.server --model "{config.LLM_MODEL_PATH}" '
        f"--n_ctx {config.LLM_CONTEXT_SIZE} --host 127.0.0.1 --port 8000"
    )


# --------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------

@dataclass
class LLMResponse:
    """Raw result of one chat completion call."""

    text: str
    response_time_seconds: float
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None


@dataclass
class GroundedAnswer:
    """
    Full Phase 6 pipeline result: the LLM's answer PLUS the retrieved
    evidence it was based on. Keeping `retrieved_results` alongside the
    generated text -- rather than discarding it once the prompt is built --
    is what lets Phase 7/8 attach verified citations without re-running
    retrieval.
    """

    question: str
    answer_text: str
    evidence_context: str
    retrieved_results: List[SearchResult]
    response_time_seconds: float
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]


# --------------------------------------------------------------------------
# Compact grounding prompt
# --------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You answer questions about an insurance policy using ONLY the POLICY EVIDENCE below. "
    "Do not use outside insurance knowledge and do not guess. If the evidence does not clearly "
    "answer the question, say so plainly. If the policy is simply silent on a topic, say it is "
    "not addressed -- never say it is excluded unless the evidence explicitly excludes it. "
    "Keep your answer concise (2-4 sentences), in plain, simple language."
)


def build_user_prompt(evidence_context: str, question: str) -> str:
    if not evidence_context:
        evidence_context = "(No relevant policy evidence was retrieved.)"
    return f"POLICY EVIDENCE:\n{evidence_context}\n\nQUESTION:\n{question}\n\nANSWER:"


# --------------------------------------------------------------------------
# Low-level HTTP client
# --------------------------------------------------------------------------

def is_server_available(server_url: str = config.LLM_SERVER_URL, timeout: float = 3.0) -> bool:
    """Quick health check -- hits /v1/models, which llama_cpp.server always exposes."""
    try:
        response = requests.get(f"{server_url}/v1/models", timeout=timeout)
        return response.status_code == 200
    except requests.RequestException:
        return False


def generate_completion(
    system_prompt: str,
    user_prompt: str,
    server_url: str = config.LLM_SERVER_URL,
    model_name: str = config.LLM_MODEL_NAME,
    max_tokens: int = config.LLM_MAX_TOKENS,
    temperature: float = config.LLM_TEMPERATURE,
    timeout: float = config.LLM_TIMEOUT_SECONDS,
) -> LLMResponse:
    """
    Send one chat completion request to the local llama.cpp server.

    Raises a specific LLMError subclass with a clear, beginner-friendly
    message on any failure -- callers (including the Streamlit UI in a
    later phase) should catch LLMError and show it to the user instead of
    crashing.
    """
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    start = time.time()
    try:
        response = requests.post(f"{server_url}/v1/chat/completions", json=payload, timeout=timeout)
    except requests.ConnectionError as exc:
        raise LLMConnectionError(
            f"Could not reach the local LLM server at {server_url}.\n"
            f"Is it running? Start it with:\n{_server_start_hint()}"
        ) from exc
    except requests.Timeout as exc:
        raise LLMTimeoutError(
            f"The local LLM server did not respond within {timeout}s. "
            "The model may be overloaded, the machine may be low on memory, "
            "or max_tokens may be set too high for this hardware."
        ) from exc
    except requests.RequestException as exc:
        raise LLMConnectionError(f"Error communicating with the local LLM server: {exc}") from exc

    elapsed = time.time() - start

    if response.status_code != 200:
        raise LLMResponseError(
            f"The local LLM server returned an error (HTTP {response.status_code}): "
            f"{response.text[:300]}"
        )

    try:
        data = response.json()
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
    except (KeyError, IndexError, ValueError) as exc:
        raise LLMResponseError(
            f"The local LLM server returned an unexpected response format: {response.text[:300]}"
        ) from exc

    if not text or not text.strip():
        raise LLMResponseError(
            "The local LLM returned an empty response. Try again, or check the server's own "
            "console output for errors."
        )

    return LLMResponse(
        text=text.strip(),
        response_time_seconds=elapsed,
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
    )


# --------------------------------------------------------------------------
# Orchestration: retrieve -> build context -> call LLM
# --------------------------------------------------------------------------

def answer_question(
    policy_index: "PolicyIndex",
    question: str,
    max_chunks: int = config.DEFAULT_CONTEXT_CHUNKS,
    **llm_kwargs,
) -> GroundedAnswer:
    """
    The full Phase 6 pipeline for one question: retrieve evidence (reusing
    Phase 4/5's retrieval, unchanged), build a compact evidence context,
    call the local LLM, and return both the generated text and the raw
    retrieved chunks together.

    Superseded by app.rag.answer_generator.generate_grounded_response() as
    of Phase 7 (structured status + verified citations) -- kept as-is here
    since it's still a valid, simpler pipeline and existing Phase 6 tests
    depend on it.
    """
    from ..rag.context_builder import build_evidence_context
    from ..rag.retrieval import retrieve

    results = retrieve(policy_index, question, max_k=max_chunks)
    evidence_context = build_evidence_context(results, max_chunks=max_chunks)
    user_prompt = build_user_prompt(evidence_context, question)

    llm_response = generate_completion(SYSTEM_PROMPT, user_prompt, **llm_kwargs)

    return GroundedAnswer(
        question=question,
        answer_text=llm_response.text,
        evidence_context=evidence_context,
        retrieved_results=results,
        response_time_seconds=llm_response.response_time_seconds,
        prompt_tokens=llm_response.prompt_tokens,
        completion_tokens=llm_response.completion_tokens,
    )
