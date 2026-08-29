# Testing

## Why standalone scripts, not pytest

Every script in `tests/` was written as the corresponding development phase was built, and each one **prints what it checked and why**, not just pass/fail — because for a trust-and-safety-focused project, the reasoning behind a check matters as much as the result. Converting them to pytest now would be a testing-framework migration with no functional benefit and real risk of introducing regressions across ten files; `run_tests.bat` runs them all in build order and reports a summary instead.

## What's covered, and why those components specifically

Priority went to the components where a silent failure could produce a *misleading* answer — not raw line coverage.

| Script | Covers | Why it matters |
|---|---|---|
| `test_pdf_extraction.py` | PDF validation, page-by-page text extraction, corrupted/scanned-page detection | Every downstream citation depends on page numbers being right from the start. |
| `test_clause_chunking.py` | Clause/section regex detection, the false-positive regression ("48 months" not misread as clause "48") | Wrong chunk boundaries corrupt every citation and retrieval result built on top. |
| `test_semantic_search.py` | Embedding + FAISS retrieval, score sanity (positive vs. negative queries), dedup | If retrieval finds the wrong clause, a perfectly-behaved LLM still produces a wrong answer. |
| `test_retrieval_evaluation.py` | Recall@1/3/5, MRR, threshold analysis against the golden retrieval dataset | Quantifies retrieval quality *before* the LLM can mask a retrieval problem with fluent prose. |
| `test_llm_direct_binding.py` | The local model loads and generates at all | Fast, no-server smoke test for the environment itself. |
| `test_llm_integration.py` | The llama.cpp server integration, plus documented hallucination checks | Confirms the LLM only ever gets called through the intended HTTP path. |
| `test_structured_answers.py` | STATUS/EVIDENCE_ID/ANSWER parsing, the "silence ≠ exclusion" regression | The single most safety-critical parsing step in the whole pipeline. |
| `test_guardrails.py` | Fake evidence IDs, invented citations, unsupported numbers, exclusion-metadata cross-checks, related-exclusion surfacing | **This is the trust layer.** A bug here means an unverified claim could reach a user labeled as verified. |
| `test_streamlit_app.py` | The full app via Streamlit's `AppTest` — upload errors, a real question end-to-end, LLM-server-down handling | Confirms the wiring, not just the individual pieces, actually works together. |
| `test_evidence_highlighting.py` | Exact/normalized/excerpt text location, cross-page isolation, honest "not found" fallback, original-PDF-never-modified | A fabricated highlight would be actively *more* misleading than no highlight at all. |

## Running everything

```powershell
run_tests.bat
```

Scripts marked `[LLM server required]` in that script's output need `llama_cpp.server` running first (`run_app.bat`, or start it standalone — see `docs/setup_manual.md`).

## Running the end-to-end evaluation separately

The golden-dataset evaluation (Phase 11) is intentionally **not** part of `run_tests.bat` — it takes several minutes (real LLM calls for ~40 questions) and produces a timestamped result set meant to be tracked over time, not a pass/fail gate.

```powershell
run_evaluation.bat
```

See `docs/evaluation.md` for what it measures and the actual results.
