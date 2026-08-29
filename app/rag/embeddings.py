"""
Phase 4 — Embedding generation.

An embedding is a fixed-length vector of numbers representing the *meaning*
of a piece of text, produced by a model trained so that texts with similar
meaning end up as nearby vectors regardless of shared vocabulary. That's
what lets a plain-English question like "Is dental covered?" find a clause
written as "Dental treatment is covered only if necessitated by an
accidental bodily injury..." even though the two share almost no words.
Keyword search cannot do this; semantic search can.

Model: sentence-transformers/all-MiniLM-L6-v2 -- 384-dimensional output,
~90MB on disk, fast on CPU. An appropriate choice for an 8GB RAM machine.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from ..pdf_processing import Clause

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2's fixed output size

# Conservative for 8GB RAM. MiniLM is tiny enough that a much larger batch
# would still be safe on its own, but the full app also holds Streamlit,
# FAISS, and (later) a local LLM in memory at the same time -- a modest
# batch size here leaves headroom for everything running alongside it.
DEFAULT_BATCH_SIZE = 16


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """
    Load the embedding model exactly once per process.

    `lru_cache(maxsize=1)` on a no-argument function is a minimal
    singleton: the first call constructs the SentenceTransformer (reading
    ~90MB of weights); every later call, from anywhere in the app --
    including repeated Streamlit reruns in Phase 9 -- returns the same
    in-memory instance instead of reloading it.
    """
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def build_embedding_text(clause: Clause) -> str:
    """
    Compact text fed to the embedding model -- NOT the text shown to users.

    Prepending the section name gives the model useful context (e.g. "this
    is an exclusion clause") without repeating clause numbers, page
    numbers, or policy name, which carry no semantic meaning for a model
    like this and would just add noise. `clause.text` itself is stored and
    displayed completely untouched, separately, for citation purposes.
    """
    if clause.section:
        return f"Section: {clause.section}\nText: {clause.text}"
    return clause.text


def embed_texts(texts: List[str], batch_size: int = DEFAULT_BATCH_SIZE) -> np.ndarray:
    """
    Embed a list of strings in batches, returning L2-normalized float32
    vectors -- normalized so that cosine similarity reduces to a plain dot
    product (see vector_store.py's IndexFlatIP usage).
    """
    if not texts:
        return np.empty((0, EMBEDDING_DIM), dtype="float32")

    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype("float32")


def embed_clauses(clauses: List[Clause], batch_size: int = DEFAULT_BATCH_SIZE) -> np.ndarray:
    """Embed a list of Clause chunks using their compact embedding text."""
    texts = [build_embedding_text(c) for c in clauses]
    return embed_texts(texts, batch_size=batch_size)
