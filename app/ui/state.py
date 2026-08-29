"""
Phase 9 — Streamlit session state helpers and cached resource loading.

Caching design: there is exactly ONE cached loader
(`_load_existing_policy_cached`), keyed by document_id, so a given policy
never has two copies of its FAISS index / clause list resident in memory
at once. Processing a brand-new upload runs once (triggered by an explicit
button click, not by a rerun) and writes the on-disk index via Phase 4's
build_or_load_index; the very next Streamlit rerun (which Streamlit always
performs after a button click) then loads it through the same cached path
everything else uses.

The embedding model itself (app.rag.get_embedding_model) is cached via a
plain lru_cache, which already persists for the life of the Python process
regardless of Streamlit reruns -- no extra work needed for that here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import streamlit as st

from app.pdf_processing import build_clauses, extract_pdf
from app.rag import PolicyIndex, VectorStoreError, build_or_load_index, load_index

POLICIES_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "policies"


def init_session_state() -> None:
    defaults = {
        "active_document_id": None,
        # Display-only history: {"question": str, "response": GroundedResponse}.
        # Never replayed to the LLM -- each question is answered fresh
        # against the current evidence, per the token-efficiency requirement.
        "chat_history": [],
        "feedback_given": {},        # {chat_history index: feedback label} -- prevents duplicate submissions
        "evidence_shown": {},        # {chat_history index: True} -- lazy-reveals the PDF viewer only on click
        "other_evidence_shown": {},  # {widget key: True} -- same, for the "other retrieved evidence" panel
        "debug_mode": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_resource(show_spinner=False)
def _load_existing_policy_cached(document_id: str) -> PolicyIndex:
    """The single canonical in-memory copy of any given policy's index,
    for the life of this Streamlit session."""
    return load_index(document_id)


def process_uploaded_policy(uploaded_file, policy_name: str, policy_version: str) -> str:
    """
    Saves the uploaded file to disk (PyMuPDF needs a real path, not just
    bytes in memory) and runs it through the pipeline ONCE -- this is
    triggered by an explicit "Process Policy" button click, not a rerun,
    so it deliberately isn't wrapped in st.cache_resource itself; the
    resulting index is what get_active_policy_index() will load afterward.

    Returns the resulting document_id. Raises InvalidPDFError /
    VectorStoreError with a clean message on a bad file -- callers should
    catch these and display them, never let them crash the app.
    """
    POLICIES_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = Path(uploaded_file.name).name  # strip any path components from the filename
    dest_path = POLICIES_DIR / safe_name
    dest_path.write_bytes(uploaded_file.getvalue())

    document = extract_pdf(dest_path)
    clauses = build_clauses(document, policy_name=policy_name, policy_version=policy_version)
    policy_index = build_or_load_index(document, clauses, policy_name, policy_version)
    return policy_index.document_id


def get_active_policy_index() -> Optional[PolicyIndex]:
    """Resolves the currently active document_id to a loaded PolicyIndex,
    or None if nothing is active yet / the saved index has gone missing."""
    document_id = st.session_state.get("active_document_id")
    if not document_id:
        return None
    try:
        return _load_existing_policy_cached(document_id)
    except VectorStoreError:
        return None
