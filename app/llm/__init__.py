"""Local LLM integration (Phase 6): HTTP client for the llama.cpp server,
compact grounding prompt, and the retrieve -> context -> LLM pipeline."""

from .client import (
    SYSTEM_PROMPT,
    GroundedAnswer,
    LLMConnectionError,
    LLMError,
    LLMResponse,
    LLMResponseError,
    LLMTimeoutError,
    answer_question,
    build_user_prompt,
    generate_completion,
    is_server_available,
)

__all__ = [
    "SYSTEM_PROMPT",
    "GroundedAnswer",
    "LLMConnectionError",
    "LLMError",
    "LLMResponse",
    "LLMResponseError",
    "LLMTimeoutError",
    "answer_question",
    "build_user_prompt",
    "generate_completion",
    "is_server_available",
]
