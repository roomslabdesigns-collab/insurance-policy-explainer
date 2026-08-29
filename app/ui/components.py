"""
Phase 9 — Reusable Streamlit render functions.

Every function here is presentation only: it reads data already computed
by app.rag / app.validation and renders it. No pipeline logic lives here,
per the "don't duplicate business logic inside UI components" requirement.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import streamlit as st

from app.database import log_evidence_view, save_feedback
from app.pdf_processing import InvalidPDFError
from app.rag import GroundedResponse, VectorStoreError, list_processed_policies
from app.rag.answer_generator import STATUS_NO_EVIDENCE
from app.utils import STATUS_COVERED, STATUS_EXCLUDED, STATUS_INSUFFICIENT, STATUS_NOT_MENTIONED

from .evidence_viewer import render_evidence_viewer, render_other_evidence
from .state import get_active_policy_index, process_uploaded_policy

STATUS_DISPLAY = {
    STATUS_COVERED: ("✅", "Covered"),
    STATUS_EXCLUDED: ("⛔", "Explicitly Excluded"),
    STATUS_NOT_MENTIONED: ("❓", "Not Mentioned"),
    STATUS_INSUFFICIENT: ("⚠️", "Insufficient Evidence"),
    STATUS_NO_EVIDENCE: ("🚫", "No Evidence Found"),
}

CONFIDENCE_DISPLAY = {
    "Verified Evidence": "🟢 Verified Evidence",
    "Limited Evidence": "🟡 Limited Evidence",
    "Insufficient Evidence": "🔴 Insufficient Evidence",
}

_NEUTRAL_NOTES = {"Validated.", "Validated (no citation needed for this status)."}


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

def render_sidebar() -> None:
    st.sidebar.title("📄 Policy Setup")

    st.sidebar.subheader("1. Upload a Policy")
    uploaded_file = st.sidebar.file_uploader("Insurance policy PDF", type=["pdf"])

    default_name = Path(uploaded_file.name).stem if uploaded_file else ""
    policy_name = st.sidebar.text_input("Policy name", value=default_name, key="policy_name_input")
    policy_version = st.sidebar.text_input(
        "Policy version / year", value=str(datetime.now().year), key="policy_version_input"
    )

    process_disabled = uploaded_file is None
    if st.sidebar.button("Process Policy", type="primary", disabled=process_disabled):
        _run_processing(uploaded_file, policy_name or "Untitled Policy", policy_version or "unknown")

    st.sidebar.divider()
    st.sidebar.subheader("2. Active Policy")
    _render_policy_picker()

    policy_index = get_active_policy_index()
    if policy_index is not None:
        st.sidebar.divider()
        st.sidebar.subheader("Policy Information")
        st.sidebar.write(f"**Name:** {policy_index.policy_name}")
        st.sidebar.write(f"**Version:** {policy_index.policy_version}")
        st.sidebar.write(f"**Indexed chunks:** {policy_index.chunk_count}")
        st.sidebar.write("**Status:** ✅ Ready to answer questions")

    st.sidebar.divider()
    st.sidebar.checkbox("🔧 Debug mode (show technical error details)", key="debug_mode")


def _run_processing(uploaded_file, policy_name: str, policy_version: str) -> None:
    try:
        with st.sidebar.status("Processing policy...", expanded=True) as status:
            status.write("📖 Reading policy document...")
            status.write("🔍 Extracting text and detecting clauses...")
            status.write("🧮 Creating embeddings and building the search index...")
            document_id = process_uploaded_policy(uploaded_file, policy_name, policy_version)
            status.update(label="✅ Ready to answer questions", state="complete", expanded=False)

        st.session_state.active_document_id = document_id
        st.session_state.chat_history = []
        st.session_state.feedback_given = {}
        st.sidebar.success(f"'{policy_name}' processed successfully.")
        st.rerun()

    except InvalidPDFError as exc:
        st.sidebar.error(f"Could not process this PDF: {exc}")
    except VectorStoreError as exc:
        st.sidebar.error(f"Could not build a search index for this policy: {exc}")
    except Exception as exc:  # last-resort guard -- the UI must never crash on a bad upload
        st.sidebar.error("An unexpected error occurred while processing the policy.")
        if st.session_state.get("debug_mode"):
            st.sidebar.exception(exc)


def _render_policy_picker() -> None:
    policies = list_processed_policies()
    if not policies:
        st.sidebar.caption("No processed policies yet — upload one above.")
        return

    # Most recently processed first, so a fresh session with no explicit
    # selection defaults to the policy the user most likely just finished
    # working with, rather than an arbitrary one.
    policies = sorted(policies, key=lambda p: p.get("created_at", ""), reverse=True)
    labels = [f"{p['policy_name']} ({p['policy_version']})" for p in policies]
    doc_ids = [p["document_id"] for p in policies]

    current_id = st.session_state.get("active_document_id")
    current_index = doc_ids.index(current_id) if current_id in doc_ids else 0

    selected_label = st.sidebar.selectbox("Active policy", labels, index=current_index)
    selected_doc_id = doc_ids[labels.index(selected_label)]

    if selected_doc_id != current_id:
        st.session_state.active_document_id = selected_doc_id
        st.session_state.chat_history = []
        st.session_state.feedback_given = {}
        st.rerun()


# --------------------------------------------------------------------------
# Answer display
# --------------------------------------------------------------------------

def render_answer_card(
    response: GroundedResponse,
    history_index: int,
    document_id: str,
    policy_name: str,
    source_pdf_path: str = "",
) -> None:
    icon, label = STATUS_DISPLAY.get(response.status, ("", response.status))
    confidence = CONFIDENCE_DISPLAY.get(response.confidence_label, response.confidence_label)

    st.markdown(f"#### {icon} {label}")
    st.caption(f"Evidence quality: {confidence}")

    st.markdown("**Answer** _(AI explanation of the evidence below)_")
    st.write(response.answer_text)

    if response.citation:
        quote_label = "Direct Policy Text" + (" ✓ verified quote" if response.citation.is_direct_quote else " (verified excerpt)")
        st.markdown(f"**{quote_label}**")
        st.info(response.citation.display_excerpt)

        st.markdown("**Source**")
        clause_label = response.citation.clause_number or "General / Preamble"
        st.write(
            f"{response.citation.policy_name} ({response.citation.policy_version}) — "
            f"{response.citation.section or 'N/A'} → Clause {clause_label} → Page {response.citation.page_number}"
        )

        # Lazy by design: opening the PDF and rendering a page image only
        # happens once the user actually clicks this -- not on every rerun
        # (e.g. while they're typing the next question).
        if st.button("🔍 View Evidence in Policy", key=f"view_evidence_{history_index}"):
            st.session_state.evidence_shown[history_index] = True
            log_evidence_view(document_id, response.question)

        if st.session_state.evidence_shown.get(history_index):
            with st.container(border=True):
                render_evidence_viewer(response.citation, source_pdf_path)

        render_other_evidence(
            response.retrieved_results, response.citation.chunk_id, source_pdf_path, history_index
        )
    else:
        st.caption("No specific policy clause applies to this response.")

    if response.reliability_note and response.reliability_note not in _NEUTRAL_NOTES:
        st.caption(f"ℹ️ {response.reliability_note}")

    st.caption(
        "This is an educational explanation of your uploaded policy, not an insurance decision. "
        "Your insurer's official policy wording and determination always govern."
    )

    render_feedback_buttons(response, history_index, document_id, policy_name)


def render_feedback_buttons(
    response: GroundedResponse, history_index: int, document_id: str, policy_name: str
) -> None:
    already = st.session_state.feedback_given.get(history_index)
    if already:
        st.caption(f"Thanks for the feedback — recorded as: {already}")
        return

    col1, col2, col3, _ = st.columns([1, 1, 1, 3])
    if col1.button("👍 Helpful", key=f"fb_helpful_{history_index}"):
        _submit_feedback(response, history_index, document_id, policy_name, "helpful", "👍 Helpful")
    if col2.button("👎 Not Helpful", key=f"fb_not_helpful_{history_index}"):
        _submit_feedback(response, history_index, document_id, policy_name, "not_helpful", "👎 Not Helpful")
    if col3.button("⚠️ Incorrect", key=f"fb_incorrect_{history_index}"):
        _submit_feedback(response, history_index, document_id, policy_name, "incorrect", "⚠️ Incorrect")


def _submit_feedback(
    response: GroundedResponse, history_index: int, document_id: str, policy_name: str, feedback_type: str, label: str
) -> None:
    save_feedback(
        document_id=document_id,
        policy_name=policy_name,
        question=response.question,
        status=response.status,
        feedback_type=feedback_type,
    )
    st.session_state.feedback_given[history_index] = label
    st.rerun()
