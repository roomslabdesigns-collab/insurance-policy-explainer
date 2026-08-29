"""
Phase 4 — FAISS vector index with on-disk persistence, one index per policy.

Each policy document gets its own dedicated index directory
(data/indexes/<document_id>/) rather than one shared index filtered by
metadata. This is a deliberate simplicity/safety choice: it makes it
structurally impossible to accidentally retrieve chunks from the wrong
policy (there's no shared index to mix up), at the cost of not sharing an
index across policies -- which this project never needs to do, since the
UI always searches exactly one "active policy" at a time.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import faiss

from ..pdf_processing import Clause, DocumentData
from .embeddings import EMBEDDING_DIM, EMBEDDING_MODEL_NAME, embed_clauses, embed_texts

INDEX_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "indexes"

DEFAULT_TOP_K = 5


class VectorStoreError(Exception):
    """Raised for any index build/load/search problem -- never fails silently."""


@dataclass
class SearchResult:
    """One retrieved clause, with its similarity score and full citation metadata."""

    rank: int
    score: float
    chunk_id: str
    document_id: str
    clause_number: str
    section: str
    page_number: int
    pages: List[int]
    text: str
    is_exclusion_section: bool
    contains_exclusion_language: bool
    exception_condition_text: str


@dataclass
class PolicyIndex:
    """An in-memory FAISS index plus the parallel list of Clause chunks it was built from."""

    document_id: str
    policy_name: str
    policy_version: str
    embedding_model: str
    source_pdf_path: str  # original PDF location -- Phase 10 needs this to reopen it for highlighting
    clauses: List[Clause]
    index: "faiss.Index"

    @property
    def chunk_count(self) -> int:
        return len(self.clauses)


def _index_dir(document_id: str) -> Path:
    return INDEX_ROOT / document_id


def list_processed_policies() -> List[dict]:
    """
    Lightweight scan of saved manifests for the Streamlit sidebar's policy
    picker -- reads only small JSON files, never loads a FAISS index or
    reconstructs Clause objects, so listing every processed policy costs
    almost nothing regardless of how many exist.
    """
    if not INDEX_ROOT.exists():
        return []
    policies = []
    for manifest_path in sorted(INDEX_ROOT.glob("*/manifest.json")):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                policies.append(json.load(f))
        except (OSError, json.JSONDecodeError):
            continue  # a corrupted/partial manifest just doesn't show up in the list
    return policies


def index_exists(document_id: str) -> bool:
    d = _index_dir(document_id)
    return (
        (d / "manifest.json").exists()
        and (d / "index.faiss").exists()
        and (d / "chunks.json").exists()
    )


def _read_manifest(document_id: str) -> dict:
    with open(_index_dir(document_id) / "manifest.json", "r", encoding="utf-8") as f:
        return json.load(f)


def build_index(
    clauses: List[Clause],
    policy_name: str,
    policy_version: str,
    document_id: str,
    source_pdf_path: str = "",
    batch_size: int = 16,
) -> PolicyIndex:
    """Embed every clause and build a fresh in-memory FAISS index (not saved yet)."""
    if not clauses:
        raise VectorStoreError("Cannot build an index from zero clauses.")

    vectors = embed_clauses(clauses, batch_size=batch_size)  # already L2-normalized

    # Cosine similarity via inner product: for unit-length vectors,
    # dot(a, b) == cos(angle between a and b). IndexFlatIP performs an
    # exact (brute-force) search -- entirely appropriate here since a
    # single policy's clause count is small (tens to low hundreds), so
    # there is no accuracy/speed trade-off to make with an approximate
    # index like IVF or HNSW.
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(vectors)

    if index.ntotal != len(clauses):
        raise VectorStoreError(
            f"Vector count ({index.ntotal}) does not match chunk count ({len(clauses)})."
        )

    return PolicyIndex(
        document_id=document_id,
        policy_name=policy_name,
        policy_version=policy_version,
        embedding_model=EMBEDDING_MODEL_NAME,
        source_pdf_path=source_pdf_path,
        clauses=clauses,
        index=index,
    )


def save_index(policy_index: PolicyIndex) -> Path:
    """Persist the FAISS index, chunk metadata, and a manifest to disk."""
    directory = _index_dir(policy_index.document_id)
    directory.mkdir(parents=True, exist_ok=True)

    faiss.write_index(policy_index.index, str(directory / "index.faiss"))

    with open(directory / "chunks.json", "w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in policy_index.clauses], f, ensure_ascii=False, indent=2)

    manifest = {
        "document_id": policy_index.document_id,
        "policy_name": policy_index.policy_name,
        "policy_version": policy_index.policy_version,
        "embedding_model": policy_index.embedding_model,
        "embedding_dim": EMBEDDING_DIM,
        "chunk_count": policy_index.chunk_count,
        "source_pdf_path": policy_index.source_pdf_path,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(directory / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return directory


def load_index(document_id: str) -> PolicyIndex:
    """Load a previously saved index. Raises VectorStoreError on any inconsistency."""
    directory = _index_dir(document_id)
    if not index_exists(document_id):
        raise VectorStoreError(
            f"No saved index found for document_id={document_id!r} at {directory}"
        )

    manifest = _read_manifest(document_id)
    index = faiss.read_index(str(directory / "index.faiss"))

    with open(directory / "chunks.json", "r", encoding="utf-8") as f:
        raw_clauses = json.load(f)
    clauses = [Clause(**c) for c in raw_clauses]

    if index.ntotal != len(clauses):
        raise VectorStoreError(
            f"Corrupted index for {document_id}: {index.ntotal} vectors but "
            f"{len(clauses)} stored chunks. Rebuild the index."
        )

    return PolicyIndex(
        document_id=manifest["document_id"],
        policy_name=manifest["policy_name"],
        policy_version=manifest["policy_version"],
        embedding_model=manifest["embedding_model"],
        source_pdf_path=manifest.get("source_pdf_path", ""),  # "" for indexes saved before Phase 10
        clauses=clauses,
        index=index,
    )


def build_or_load_index(
    document: DocumentData,
    clauses: List[Clause],
    policy_name: str,
    policy_version: str,
    batch_size: int = 16,
    force_rebuild: bool = False,
) -> PolicyIndex:
    """
    Incremental-processing entry point.

    Reuses a saved index when the document has already been processed with
    the SAME embedding model and the SAME number of chunks. `document_id`
    already comes from Phase 2's content hash, so re-uploading an identical
    PDF automatically hits this cache -- no separate "has this file
    changed?" check is needed on top of that. If the chunker or embedding
    model changed since the saved index was built, this rebuilds rather
    than silently serving stale data. A manifest saved before Phase 10
    (missing source_pdf_path, needed for evidence highlighting) is also
    treated as stale so it gets backfilled automatically.
    """
    document_id = document.document_id

    if not force_rebuild and index_exists(document_id):
        manifest = _read_manifest(document_id)
        if (
            manifest.get("embedding_model") == EMBEDDING_MODEL_NAME
            and manifest.get("chunk_count") == len(clauses)
            and manifest.get("source_pdf_path")
        ):
            return load_index(document_id)
        # Stale -- fall through and rebuild instead of trusting old data.

    policy_index = build_index(
        clauses,
        policy_name=policy_name,
        policy_version=policy_version,
        document_id=document_id,
        source_pdf_path=document.file_path,
        batch_size=batch_size,
    )
    save_index(policy_index)
    return policy_index


def search_policy(
    policy_index: PolicyIndex,
    query: str,
    top_k: int = DEFAULT_TOP_K,
    min_score: Optional[float] = None,
) -> List[SearchResult]:
    """
    Embed `query` and return its top_k most similar clauses from
    `policy_index`, highest similarity first.

    `min_score` is optional and unset by default: this phase's job is to
    measure and report scores honestly, not guess a cutoff before we've
    seen how they behave on real questions. Phase 8's "no evidence, no
    answer" guardrail is what sets a real threshold, informed by Phase 5's
    retrieval testing.
    """
    if not query or not query.strip():
        raise ValueError("Query must not be empty.")
    if policy_index.chunk_count == 0:
        raise VectorStoreError("This policy's index has no chunks to search.")

    # Over-fetch slightly so exact-duplicate chunk text can be filtered out
    # without the result shrinking below top_k when duplicates exist.
    fetch_k = min(top_k * 2, policy_index.chunk_count)

    query_vector = embed_texts([query.strip()])
    scores, indices = policy_index.index.search(query_vector, fetch_k)

    results: List[SearchResult] = []
    seen_texts = set()
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue  # FAISS pads with -1 if fewer than fetch_k vectors exist
        clause = policy_index.clauses[idx]
        if clause.text in seen_texts:
            continue  # skip exact-duplicate chunk text
        seen_texts.add(clause.text)

        if min_score is not None and float(score) < min_score:
            continue

        results.append(
            SearchResult(
                rank=len(results) + 1,
                score=float(score),
                chunk_id=clause.chunk_id,
                document_id=clause.document_id,
                clause_number=clause.clause_number,
                section=clause.section,
                page_number=clause.page_number,
                pages=clause.pages,
                text=clause.text,
                is_exclusion_section=clause.is_exclusion_section,
                contains_exclusion_language=clause.contains_exclusion_language,
                exception_condition_text=clause.exception_condition_text,
            )
        )
        if len(results) >= top_k:
            break

    return results
