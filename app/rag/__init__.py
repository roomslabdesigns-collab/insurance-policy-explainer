"""RAG pipeline: embeddings + vector search (Phase 4), evidence context
building (Phase 6), structured evidence-gated answer generation (Phase 7)."""

from .answer_generator import (
    ALL_STATUSES,
    STATUS_NO_EVIDENCE,
    Citation,
    GroundedResponse,
    build_labeled_evidence,
    check_evidence_sufficiency,
    generate_grounded_response,
)
from .context_builder import build_evidence_context
from .embeddings import (
    DEFAULT_BATCH_SIZE,
    EMBEDDING_DIM,
    EMBEDDING_MODEL_NAME,
    build_embedding_text,
    embed_clauses,
    embed_texts,
    get_embedding_model,
)
from .retrieval import clause_in_results, rank_of_first_match, retrieve
from .vector_store import (
    DEFAULT_TOP_K,
    PolicyIndex,
    SearchResult,
    VectorStoreError,
    build_index,
    build_or_load_index,
    index_exists,
    list_processed_policies,
    load_index,
    save_index,
    search_policy,
)

__all__ = [
    "ALL_STATUSES",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_TOP_K",
    "EMBEDDING_DIM",
    "EMBEDDING_MODEL_NAME",
    "STATUS_NO_EVIDENCE",
    "Citation",
    "GroundedResponse",
    "PolicyIndex",
    "SearchResult",
    "VectorStoreError",
    "build_embedding_text",
    "build_evidence_context",
    "build_index",
    "build_labeled_evidence",
    "build_or_load_index",
    "check_evidence_sufficiency",
    "clause_in_results",
    "embed_clauses",
    "embed_texts",
    "generate_grounded_response",
    "get_embedding_model",
    "index_exists",
    "list_processed_policies",
    "load_index",
    "rank_of_first_match",
    "retrieve",
    "save_index",
    "search_policy",
]
