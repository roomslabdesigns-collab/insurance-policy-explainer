"""
Insurance Policy Explainer — Streamlit application (Phase 9).

Run with:
    streamlit run app.py

This file only wires together UI components and pipeline functions that
already exist under app/ -- no retrieval, LLM, or validation logic lives
here directly.
"""

import streamlit as st

from app import config
from app.llm import LLMError, is_server_available
from app.rag import generate_grounded_response
from app.ui.components import render_answer_card, render_sidebar
from app.ui.state import get_active_policy_index, init_session_state

st.set_page_config(page_title="Insurance Policy Explainer", page_icon="📄", layout="wide")

init_session_state()
render_sidebar()

st.title("📄 Insurance Policy Explainer")
st.caption(
    "Ask questions about your uploaded insurance policy in plain English. "
    "**The policy document is the source of truth** — this tool helps you understand it, "
    "but your insurer's official policy wording and decision always govern."
)

policy_index = get_active_policy_index()

if policy_index is None:
    st.info("👈 Upload and process an insurance policy PDF in the sidebar to get started.")
    st.stop()

if not is_server_available(timeout=1.5):
    st.error(
        "The local AI model server isn't running, so questions can't be answered right now.\n\n"
        "Start it in a terminal:\n\n"
        f'`.\\venv\\Scripts\\python.exe -m llama_cpp.server --model "{config.LLM_MODEL_PATH}" '
        f"--n_ctx {config.LLM_CONTEXT_SIZE} --host 127.0.0.1 --port 8000`"
    )
    st.stop()

st.subheader("Ask a Question")

EXAMPLE_QUESTIONS = [
    "Is dental treatment covered?",
    "What are the waiting periods?",
    "What conditions are excluded?",
]
st.session_state.setdefault("question_input", "")
example_cols = st.columns(len(EXAMPLE_QUESTIONS))
for col, example in zip(example_cols, EXAMPLE_QUESTIONS):
    if col.button(example, key=f"example_{example}"):
        st.session_state["question_input"] = example

question = st.text_input(
    "Your question", key="question_input", placeholder="e.g. Is dental treatment covered?"
)
ask_clicked = st.button("Ask", type="primary")

if ask_clicked and question.strip():
    with st.spinner("Reading the policy and generating a grounded answer..."):
        response = None
        try:
            response = generate_grounded_response(policy_index, question.strip())
        except LLMError as exc:
            st.error(f"The local AI model had a problem answering this question: {exc}")
        except ValueError as exc:
            st.warning(str(exc))
        except Exception as exc:  # last-resort guard -- never show a raw traceback
            st.error("Something went wrong while generating an answer.")
            if st.session_state.get("debug_mode"):
                st.exception(exc)

    if response is not None:
        st.session_state.chat_history.append({"question": question.strip(), "response": response})
        st.rerun()

st.divider()

if not st.session_state.chat_history:
    st.caption("No questions asked yet in this session.")
else:
    st.subheader("Answers")
    for i, entry in reversed(list(enumerate(st.session_state.chat_history))):
        st.markdown(f"**❓ {entry['question']}**")
        render_answer_card(
            entry["response"],
            history_index=i,
            document_id=policy_index.document_id,
            policy_name=policy_index.policy_name,
            source_pdf_path=policy_index.source_pdf_path,
        )
        st.divider()
