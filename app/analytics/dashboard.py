"""
Phase 11 — Analytics dashboard.

Run as its own Streamlit app (separate from the main user-facing app.py,
since this is a developer/product-owner tool, not part of the end-user flow):

    streamlit run app/analytics/dashboard.py --server.port 8502

Two clearly separated sections, because they measure different things:
  1. Golden Dataset Evaluation -- controlled, hand-verified questions,
     run through the real pipeline (app.evaluation.end_to_end_evaluator).
  2. Real User Feedback -- whatever actual usage has been logged so far
     (small, self-selected, and not a substitute for #1).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.database import count_evidence_views, get_all_feedback, get_feedback_summary
from app.evaluation import compare_runs, generate_text_report, list_runs, load_run_results, load_run_summary
from app.storage.validation_logger import DEFAULT_LOG_PATH

st.set_page_config(page_title="Policy Explainer — Analytics", page_icon="📊", layout="wide")

# Deliberate, non-default color mapping -- categorical identity for the
# classification outcomes, with severity treated as a status signal
# (green=good, blue=good-but-cautious, amber=moderate concern, red=critical,
# purple=guardrail-caught) rather than an auto-cycled rainbow palette.
CLASS_COLORS = {
    "Correct": "#2E7D32",
    "Appropriate Abstention": "#1565C0",
    "Incorrect Abstention": "#F9A825",
    "Incorrect": "#F9A825",
    "Citation Failure": "#EF6C00",
    "Wrong-but-Confident": "#C62828",
    "Validation Failure": "#6A1B9A",
}
STAGE_COLORS = {
    "Retrieval Failure": "#C62828",
    "Evidence Selection Failure": "#EF6C00",
    "Generation Failure": "#F9A825",
    "Citation Failure": "#6A1B9A",
    "Guardrail Failure": "#1565C0",
}

st.title("📊 Insurance Policy Explainer — Analytics")
st.caption(
    "Two separate measurements: **golden-dataset evaluation** (controlled, hand-verified questions) "
    "and **real user feedback** (actual usage). Don't conflate them — a good golden-dataset score "
    "doesn't guarantee real users are happy, and small feedback samples can't replace systematic evaluation."
)

# ============================================================================
# Section 1 — Golden Dataset Evaluation
# ============================================================================
st.header("1. Golden Dataset Evaluation")

runs = list_runs()
if not runs:
    st.info(
        "No evaluation runs found yet. Run the evaluator first:\n\n"
        "`.\\venv\\Scripts\\python.exe tests\\test_end_to_end_evaluation.py`"
    )
else:
    selected_run = st.selectbox("Evaluation run", options=list(reversed(runs)), index=0)
    summary = load_run_summary(selected_run)
    results_df = load_run_results(selected_run)
    m = summary["metrics"]

    def pct(value) -> str:
        return f"{value:.0%}" if isinstance(value, (int, float)) and value == value else "n/a"

    st.subheader("Quality")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Answer Accuracy", pct(m.get("answer_accuracy")))
    c2.metric("Status Accuracy", pct(m.get("status_accuracy")))
    c3.metric("Citation Accuracy", pct(m.get("citation_accuracy")))
    c4.metric("Retrieval Success Rate", pct(m.get("retrieval_success_rate")))

    st.subheader("Safety")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Appropriate Abstention", pct(m.get("appropriate_abstention_rate")))
    s2.metric("Incorrect Abstention", pct(m.get("incorrect_abstention_rate")))
    s3.metric("Wrong-but-Confident ⚠️", pct(m.get("wrong_but_confident_rate")))
    s4.metric("Validation Failure", pct(m.get("validation_failure_rate")))
    if m.get("wrong_but_confident_rate", 0) and m["wrong_but_confident_rate"] > 0.05:
        st.error(
            f"Wrong-but-Confident rate is {pct(m['wrong_but_confident_rate'])} — this is the single "
            "most important number on this page. Review the failing questions below before anything else."
        )

    st.subheader("Performance")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Avg response time", f"{m.get('avg_response_time_seconds', 0):.2f}s")
    p2.metric("Median response time", f"{m.get('median_response_time_seconds', 0):.2f}s")
    p3.metric("Slowest response", f"{m.get('max_response_time_seconds', 0):.2f}s")
    p4.metric("Avg chunks retrieved", f"{m.get('avg_chunks_retrieved', 0):.1f}")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Result classification**")
        class_counts = results_df["classification"].value_counts().reset_index()
        class_counts.columns = ["classification", "count"]
        fig = px.bar(
            class_counts, x="count", y="classification", orientation="h",
            color="classification", color_discrete_map=CLASS_COLORS,
        )
        fig.update_layout(showlegend=False, yaxis_title="", xaxis_title="Questions", height=320)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("**Response time distribution**")
        fig = px.histogram(results_df, x="total_time_seconds", nbins=15)
        fig.update_traces(marker_color="#1565C0")
        fig.update_layout(xaxis_title="Seconds", yaxis_title="Questions", height=320, bargap=0.05)
        st.plotly_chart(fig, use_container_width=True)

    stage_counts = results_df.loc[results_df["failure_stage"] != "", "failure_stage"].value_counts()
    if len(stage_counts):
        st.markdown("**Failures by pipeline stage**")
        stage_df = stage_counts.reset_index()
        stage_df.columns = ["stage", "count"]
        fig = px.bar(
            stage_df, x="count", y="stage", orientation="h",
            color="stage", color_discrete_map=STAGE_COLORS,
        )
        fig.update_layout(showlegend=False, yaxis_title="", xaxis_title="Questions", height=280)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("No pipeline failures in this run — every question was Correct or an appropriate abstention.")

    st.subheader("Accuracy by category")
    cat_df = (
        results_df.assign(correct=(results_df["classification"] == "Correct"))
        .groupby("category")
        .agg(n=("question", "count"), accuracy=("correct", "mean"))
        .reset_index()
        .sort_values("accuracy")
    )
    fig = px.bar(cat_df, x="accuracy", y="category", orientation="h", text=cat_df["n"].apply(lambda n: f"n={n}"))
    fig.update_traces(marker_color="#1565C0")
    fig.update_layout(xaxis_title="Answer accuracy", yaxis_title="", xaxis_tickformat=".0%", height=320)
    st.plotly_chart(fig, use_container_width=True)

    previous_runs = [r for r in runs if r < selected_run]
    if previous_runs:
        st.subheader("Regression comparison vs. previous run")
        comparison_df = compare_runs(previous_runs[-1], selected_run)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    else:
        st.caption("This is the earliest run on record — nothing to compare against yet.")

    with st.expander("📄 Full text report"):
        st.text(generate_text_report(selected_run))

    with st.expander(f"🔍 Per-question results ({len(results_df)})"):
        st.dataframe(
            results_df[
                ["question", "category", "expected_status", "actual_status", "classification",
                 "failure_stage", "total_time_seconds", "answer_text"]
            ],
            use_container_width=True, hide_index=True,
        )

st.divider()

# ============================================================================
# Section 2 — Real User Feedback (separate from golden-dataset evaluation)
# ============================================================================
st.header("2. Real User Feedback")
st.caption(
    "From actual Streamlit sessions (Phase 9's feedback buttons + evidence-view clicks). "
    "Small samples here are directional, not statistically conclusive."
)

feedback_rows = get_all_feedback()
total_logged_questions = 0
if DEFAULT_LOG_PATH.exists():
    with open(DEFAULT_LOG_PATH, "r", encoding="utf-8") as f:
        total_logged_questions = sum(1 for _ in f)

total_views = count_evidence_views()

if not feedback_rows and total_logged_questions == 0:
    st.info("No real usage recorded yet — ask questions through the main app (`streamlit run app.py`) to populate this section.")
else:
    feedback_df = pd.DataFrame(
        feedback_rows, columns=["timestamp", "document_id", "policy_name", "question", "status", "feedback_type"]
    )
    counts = Counter(feedback_df["feedback_type"]) if not feedback_df.empty else Counter()
    total_feedback = sum(counts.values())

    f1, f2, f3, f4 = st.columns(4)
    f1.metric("Questions logged (all sessions)", total_logged_questions)
    f2.metric("Feedback submitted", total_feedback)
    f3.metric("Helpful rate", f"{counts.get('helpful', 0) / total_feedback:.0%}" if total_feedback else "n/a")
    f4.metric(
        "Evidence-view rate",
        f"{total_views / total_logged_questions:.0%}" if total_logged_questions else "n/a",
    )

    if total_feedback:
        st.markdown("**Feedback breakdown**")
        fb_df = pd.DataFrame(
            [{"type": k, "count": v} for k, v in counts.items()]
        )
        fig = go.Figure(go.Bar(
            x=fb_df["count"], y=fb_df["type"], orientation="h",
            marker_color=["#2E7D32" if t == "helpful" else "#C62828" if t == "incorrect" else "#F9A825" for t in fb_df["type"]],
        ))
        fig.update_layout(xaxis_title="Count", yaxis_title="", height=220)
        st.plotly_chart(fig, use_container_width=True)

    if not feedback_df.empty:
        with st.expander(f"🔍 Recent feedback ({len(feedback_df)})"):
            st.dataframe(feedback_df.head(50), use_container_width=True, hide_index=True)
