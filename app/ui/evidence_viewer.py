"""
Phase 10 — Streamlit "View Evidence in Policy" component.

Presentation only: calls app.pdf_processing.highlighter (deterministic,
non-LLM text location) and renders whatever it honestly finds -- a real
highlighted page image, or a clear "couldn't be highlighted" fallback that
still shows the verified citation and evidence text. Never fakes a match.
"""

from __future__ import annotations

from typing import List

import streamlit as st

from app.pdf_processing.highlighter import MATCH_EXACT, locate_evidence_on_page, render_page_image
from app.rag.vector_store import SearchResult
from app.validation.citation_validator import Citation


def _render_one_page(source_pdf_path: str, page_number: int, evidence_text: str) -> None:
    result = locate_evidence_on_page(source_pdf_path, page_number, evidence_text)

    if result.found:
        try:
            image_bytes = render_page_image(source_pdf_path, page_number, result.quads)
            st.image(image_bytes, caption=f"Page {page_number} — supporting text highlighted", use_column_width=True)
            if result.match_level != MATCH_EXACT:
                st.caption(
                    f"ℹ️ Located via a {result.match_level} match (the clause's line-wrapping in the "
                    "PDF differs slightly from the stored text, so a shorter exact match was used)."
                )
            return
        except Exception:
            pass  # fall through to the honest failure path below

    st.warning(
        "The source page is available, but the exact text could not be highlighted automatically. "
        "Please review the cited policy text above."
    )
    try:
        image_bytes = render_page_image(source_pdf_path, page_number, [])
        st.image(image_bytes, caption=f"Page {page_number} (no highlight available)", use_column_width=True)
    except Exception:
        st.caption(
            "This page's image could not be rendered either — the citation above remains "
            "independently verified against the stored policy text."
        )


def render_evidence_viewer(citation: Citation, source_pdf_path: str) -> None:
    """Full evidence panel for the ONE verified citation an answer relied on."""
    st.markdown(f"**Policy:** {citation.policy_name} ({citation.policy_version})")
    st.markdown(f"**Section:** {citation.section or 'N/A'}")
    st.markdown(f"**Clause:** {citation.clause_number or 'General / Preamble'}")

    pages = citation.pages or [citation.page_number]
    page_label = f"Page {pages[0]}" if len(pages) == 1 else f"Pages {', '.join(map(str, pages))} (clause spans multiple pages)"
    st.markdown(f"**{page_label}**")

    st.markdown("**Direct Policy Text:**")
    st.info(citation.full_text)

    if not source_pdf_path:
        st.warning(
            "The source page is available, but the exact text could not be highlighted automatically. "
            "Please review the cited policy text above."
        )
        return

    for page_number in pages:
        if len(pages) > 1:
            st.markdown(f"##### 📄 Page {page_number}")
        _render_one_page(source_pdf_path, page_number, citation.full_text)


def render_other_evidence(
    retrieved_results: List[SearchResult],
    cited_chunk_id: str,
    source_pdf_path: str,
    history_index: int,
) -> None:
    """
    Section 8/9: lets the user inspect OTHER chunks that were retrieved
    alongside the cited one (e.g. a related exclusion flagged by Phase 8's
    related-exclusion check), each independently -- clearly labeled as
    additional retrieved context, never presented as if it were itself
    highlighted supporting evidence for the answer.
    """
    others = [r for r in retrieved_results if r.chunk_id != cited_chunk_id]
    if not others:
        return

    with st.expander(f"📎 Other retrieved evidence ({len(others)}) — not cited in this answer"):
        st.caption(
            "These were retrieved alongside the cited clause but were NOT the source the answer "
            "relied on. Reviewing them can be useful, e.g. if a related exclusion applies."
        )
        for r in others:
            label = r.clause_number or "(preamble)"
            st.markdown(f"**Clause {label}, Page {r.page_number}** — {r.text[:140]}{'...' if len(r.text) > 140 else ''}")
            key = f"view_other_{history_index}_{r.chunk_id}"
            if st.button(f"View Clause {label} in Policy", key=key):
                st.session_state.other_evidence_shown[key] = True
            if st.session_state.other_evidence_shown.get(key):
                _render_one_page(source_pdf_path, r.page_number, r.text)
            st.markdown("---")
