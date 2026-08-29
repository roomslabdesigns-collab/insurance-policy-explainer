# PolicyLens AI — Evidence-Grounded Insurance Policy Explainer

A local, private, retrieval-augmented AI application that answers plain-English questions about an uploaded insurance policy PDF — and never answers unless it can point to the exact clause and page it came from.

> **This is an educational document-explanation tool, not insurance advice.** It does not make coverage decisions. Your insurance policy document and your insurer's official interpretation and decision always govern.

---

## The Problem

Insurance policy documents are long, dense, and written in language most people don't use day to day. A customer trying to answer a simple question — *"Is dental treatment covered?"*, *"What's the waiting period for a pre-existing condition?"*, *"What's excluded?"* — usually has to read a 30-50 page PDF themselves, or call the insurer and wait. The cost of getting it wrong is real: a customer who wrongly assumes something is covered can face a denied claim; one who wrongly assumes something is excluded may not claim something they were entitled to.

A generic chatbot "helpfully" answering from general insurance knowledge makes this *worse*, not better — it will sound confident regardless of whether it actually read your specific policy.

## The Solution

This application uses retrieval-augmented generation (RAG) constrained by one non-negotiable rule:

> ## NO EVIDENCE → NO ANSWER
>
> The uploaded policy document is the only source of truth. If the system can't point to a specific clause that supports an answer, it says so — it does not guess, and it does not fall back on general insurance knowledge.

Every answer a user sees has already passed through an application-controlled trust layer that the LLM cannot override: the LLM never gets to invent a page number, a clause number, or a citation. It only ever picks from evidence the application already retrieved and verified.

## Key Features

All of the following are implemented and tested — not aspirational:

- **Policy-grounded answers** — every substantive answer (`Covered` / `Explicitly Excluded`) requires a citation; the LLM cannot skip this.
- **Evidence-based citations** — Section → Clause → Page, resolved entirely from application metadata, never from LLM output.
- **Safe abstention** — `Not Mentioned` / `Insufficient Evidence` / `No Evidence Found` are first-class outcomes, not failures. Silence in the policy is never presented as an exclusion.
- **Citation validation** — a cited `evidence_id` must exist, belong to the evidence actually shown to the model, and belong to the active policy; a direct quote is verified against the source text (exact → normalized match) before being labeled as one.
- **Guardrails that cross-check the LLM against independent metadata** — e.g. a `Covered` claim citing a clause from an Exclusions section gets automatically downgraded, using clause-structure metadata computed back in the PDF-parsing stage, which the LLM never sees.
- **PDF evidence highlighting** — click "View Evidence in Policy" and see the actual PDF page with the supporting sentence highlighted, or an honest "couldn't auto-highlight" fallback — never a fabricated highlight.
- **100% local / private processing** — PDF parsing, embeddings, vector search, and the LLM all run on your machine. No document content is ever sent to a third-party API.
- **End-to-end evaluation with a dashboard** — a hand-verified 41-question golden dataset run through the real pipeline (not a mock), with results tracked run-over-run for regression testing.

## Architecture

```mermaid
flowchart TD
    U[User] --> UI[Streamlit UI]
    UI --> PDF[PDF Processing<br/>PyMuPDF extraction + clause-aware chunking]
    PDF --> IDX[Embeddings + FAISS<br/>all-MiniLM-L6-v2, cosine similarity]
    IDX --> SEL[Evidence Selection<br/>top-k retrieval + sufficiency gate]
    SEL -->|evidence found| LLM[Local LLM<br/>Qwen2.5-3B via llama.cpp]
    SEL -->|evidence too weak| ABSTAIN1[Abstain — no LLM call made]
    LLM --> TRUST["🛡️ Citation Validation + Guardrails<br/>(app-controlled, not the LLM)"]
    TRUST --> ANSWER[Verified Answer + Source Evidence]
    ABSTAIN1 --> ANSWER
    ANSWER --> UI

    style TRUST fill:#6A1B9A,color:#fff
    style ABSTAIN1 fill:#1565C0,color:#fff
```

**The LLM is deliberately not the source of truth anywhere in this diagram.** It receives short, labeled evidence excerpts (`[E1]`, `[E2]`, `[E3]` — no page/clause numbers in the prompt at all) and picks a label; every citation shown to the user is rebuilt from the application's own stored metadata for that label, never from anything the model wrote.

## Trust & Safety Architecture

### Grounded generation
The LLM only ever sees the top 3 retrieved clauses for a question (~300 tokens on average) — never the full policy, never prior conversation history. It is explicitly instructed not to use outside insurance knowledge.

### No evidence, no answer
Before the LLM is even called, a Python-level relevance check on the retrieval score can reject the question outright (an honest *"I couldn't find this clearly addressed in the uploaded policy"* — zero LLM tokens spent). This threshold (0.30) is a deliberately low floor, chosen from measured score distributions in Phase 5's evaluation, not a guess — see `app/config.py` for the reasoning.

### Application-controlled citations
The LLM selects a short label (`E1`/`E2`/`E3`); the application resolves that label back to the real, stored clause/page/section. There is no code path by which the LLM's own text can become a displayed clause number or page number.

### Citation validation
- An `evidence_id` must exist, belong to the evidence actually supplied for that question, and belong to the active policy document (cosmetic formatting like `[E1]` vs `E1` is normalized — recognizing an equivalent label, never loosening which labels count as valid).
- A claimed direct quote is checked against the source text (exact match → whitespace/punctuation-normalized match); if neither succeeds, the UI shows a verified excerpt taken directly from the stored clause instead — never an unverified "quote."
- Numbers in the answer (waiting periods, day counts) that don't appear anywhere in the cited evidence get flagged and the answer is downgraded to a safe fallback.

### Guardrails
A `Covered`/`Explicitly Excluded` status is cross-checked against clause-level metadata computed independently during PDF processing — signals the LLM never sees:
- `is_exclusion_section` / `contains_exclusion_language` — a mismatch (e.g. `Covered` citing a clause from the Exclusions section) is downgraded to `Insufficient Evidence`.
- `exception_condition_text` — when a clause states a general rule and a specific exception in one sentence (e.g. *"excluded **unless** required to treat an accidental injury"*), and the user's question itself describes that exception's scenario, an `Explicitly Excluded` conclusion is downgraded rather than trusted — this was added after the first evaluation run traced every Wrong-but-Confident case to exactly this pattern (see Evaluation).
- A bare status-label echoed back as the "answer" (e.g. the text literally reading *"Insufficient Evidence"* with no explanation) is treated as a malformed response, not a real answer.

### Honest limitations
**These guardrails reduce, but do not eliminate, wrong answers.** The keyword/metadata heuristics can still miss an exclusion phrased unusually that this dataset didn't happen to test, and a small local LLM can still misjudge evidence in ways no current guardrail checks for — the measured 0% Wrong-but-Confident rate (see Evaluation) reflects this specific 41-question run, not a guarantee against every future case. This system does not claim to have solved hallucination; it claims to have built a pipeline that measures it, catches what it's specifically designed to catch, and fails toward caution rather than confidence on everything else.

## Evaluation

Methodology and full results, including a candid account of two evaluation-code bugs found and fixed along the way: [`docs/evaluation.md`](docs/evaluation.md). Summary of the current, corrected numbers:

A 41-question golden dataset (hand-verified against the actual test policy, not LLM-generated) covering coverage, exclusions, waiting periods, eligibility, ambiguous/broad questions, unsupported topics, and conflict/exception cases was run through the **real, complete pipeline** — real retrieval, real local LLM calls, real guardrails.

| Metric | Original baseline (1.5B) | Current default (3B) |
|---|---|---|
| Answer Accuracy | 36.6% | **39.0%** |
| Status Accuracy | 56.1% | 56.1% |
| Citation Accuracy | 73.3% | **73.3%** |
| Retrieval Success Rate | 100% | 100% |
| Appropriate Abstention Rate | 85.7% | **100%** |
| **Wrong-but-Confident Rate** | **4.9%*** | **0.0%** |
| Avg. response time | 3.3s | 6.5s |

*\*Corrected from an initially-reported 7.3% after a classification bug in the evaluation itself was found and fixed — see below.*

**The path here, briefly (full detail with every traced example in `docs/evaluation.md`):**

1. The first evaluation run found the dominant failure was the LLM being too cautious (safe), but a handful of cases were **Wrong-but-Confident** (dangerous) — traced to clauses stating a general rule and an exception in one sentence (e.g. *"excluded unless required to treat an accidental injury"*). Added a guardrail cross-checking the question against a clause's own exception condition, plus a check catching the LLM echoing a bare status label as its "answer." **Wrong-but-Confident: 4.9% → 0%** (re-verified by tracing each case individually).
2. Testing **Qwen2.5-3B** as a swap-in replacement surfaced a real parsing gap (the 3B model formats citations as `[E1]`, not `E1` — a fix, not a loosening) and — independently — a genuine bug in the *evaluation's own scoring*: it wasn't treating a correct, safe `Not Mentioned` answer as an abstention, so it briefly mis-scored one as Wrong-but-Confident. Fixed both. Recomputing every historical run from stored data (no new LLM calls needed) showed the ORIGINAL 1.5B baseline was actually 4.9%, not the 7.3% first reported — corrected above rather than left standing.
3. **Net result: the 3B model beats the 1.5B model on every accuracy metric while keeping an identical, clean 0% Wrong-but-Confident rate** — at the cost of roughly 2x response time and ~700MB more RAM. Made it the new default.

A visual walkthrough of this whole path — the corrected baseline, the traced before/after examples, and the model comparison — is published here: **[Zero Wrong-but-Confident](https://claude.ai/code/artifact/d1acefcc-d68a-4e4d-8401-7ed6aff89798)**.

**Why accuracy sits at 39%, not 80%+ (also in `docs/evaluation.md`):** 34% of the dataset (14/41 questions) is *deliberately* unsupported or ambiguous — abstaining on those is the correct behavior, not a miss, which caps the realistic ceiling well below 100%. Of the remaining 27 questions, 11 failures are **100% the same safe direction** (the model hedges to `Insufficient Evidence` on a paraphrased or conditional-exception question it can't confidently resolve) — zero are confidently wrong. Pushing accuracy further would mean either a substantially more capable model (7B+ starts to strain 8GB RAM) or making the system guess more often on uncertain cases — which would directly reintroduce the Wrong-but-Confident risk this whole evaluation exists to catch. That trade-off is deliberate, not an oversight.

**This project does not claim production readiness.** A 39% raw answer-accuracy rate on a 41-question dataset is not a production-grade number, and is reported as such — even though the safety-critical metric is clean and the reasons for the accuracy ceiling are understood and documented, not hand-waved.

## Quick Start (Windows)

Requires Python 3.11+ and ~3GB free disk space (mostly the model). No paid API, no cloud account, no GPU required.

```powershell
git clone <this-repo>
cd insurance-policy-explainer

setup.bat            REM one-time: creates venv, installs dependencies
download_model.bat   REM one-time: downloads the local LLM (~2.1GB, from Hugging Face)
run_app.bat          REM starts the LLM server + the Streamlit app
```

Then open **http://localhost:8501**, upload `data/policies/sample_health_policy.pdf` (included — see [Sample Data](#sample-data)) or your own policy PDF, and ask a question.

If you'd rather run each step yourself (or you're not on Windows), see [`docs/setup_manual.md`](docs/setup_manual.md) for the exact commands each `.bat` file wraps.

## Running Tests and Evaluation

This project uses standalone, narrated verification scripts (one per phase of development) rather than pytest — each script explains what it's checking and why as it runs, which matters more here than a green checkmark, since the whole point of this project is *why* an answer can be trusted.

```powershell
run_tests.bat          REM runs every verification script in build order
run_evaluation.bat      REM runs the 41-question golden-dataset evaluation (needs the LLM server running)
run_dashboard.bat       REM launches the analytics dashboard on :8502
```

See [`docs/testing.md`](docs/testing.md) for what each test script actually covers and why those components were prioritized.

## Sample Data

`data/policies/sample_health_policy.pdf` is a **synthetic** policy document written for this project — not a real insurer's document, safe to redistribute, safe to test with. It includes definitions, coverage clauses, waiting periods, and exclusions modeled on real policy structure.

Try asking:
- *"Is dental treatment covered?"*
- *"What is the waiting period for pre-existing conditions?"*
- *"What happens if I make a claim during the first year?"*
- *"Does this policy cover space travel?"* (an intentionally unsupported question — watch it abstain safely)

You can also upload your own real policy PDF — nothing leaves your machine.

## Screenshots

See [`docs/screenshots/`](docs/screenshots/) for the upload flow, a verified answer with citation, the evidence-highlighting view, a safe abstention, and the analytics dashboard.

## Project Structure

```
insurance-policy-explainer/
├── app/
│   ├── pdf_processing/   # extraction, clause-aware chunking, PDF highlighting
│   ├── rag/              # embeddings, FAISS, retrieval, answer generation
│   ├── llm/               # local LLM HTTP client (llama.cpp server)
│   ├── validation/        # citation validation, guardrails, claim checking
│   ├── evaluation/        # golden datasets, evaluator, metrics, reports
│   ├── analytics/         # analytics dashboard (separate Streamlit app)
│   ├── database/          # SQLite: user feedback, evidence-view events
│   ├── storage/           # JSONL validation/audit log
│   ├── ui/                 # Streamlit components (presentation only)
│   └── config.py           # centralized, .env-overridable configuration
├── tests/                  # one narrated verification script per phase
├── data/
│   ├── policies/            # uploaded PDFs (sample_health_policy.pdf is committed)
│   ├── indexes/             # FAISS indexes (regenerated automatically)
│   ├── evaluation/           # golden-dataset run results (data, kept)
│   └── logs/                  # JSONL audit trail (regenerated automatically)
├── models/                    # local GGUF model (downloaded, not committed)
├── docs/                       # architecture/eval/testing notes, case study, screenshots
├── app.py                      # Streamlit entry point
├── requirements.txt
├── .env.example
└── .gitignore
```

*(Deviation from a generic template, noted for transparency: there's no separate top-level `evaluation/` folder — the golden datasets live in `app/evaluation/` next to the code that consumes them, and run results live in `data/evaluation/`, consistent with how the rest of the project separates code from data.)*

## Limitations

- Tested against one synthetic sample policy and a 41-question dataset — not yet validated against a large, diverse set of real-world policy documents and formats.
- Even at 3B, the LLM still can't reliably connect a paraphrased question to a differently-worded clause, or resolve a general-rule-plus-exception sentence — it now fails safely on these (hedges) rather than dangerously (guesses wrong), but doesn't get them *right*. A 7B+ model might help further but starts to strain 8GB RAM.
- Guardrail metadata (exclusion detection) is keyword/structure-based, not a semantic understanding of legal language — it can miss an exclusion phrased unusually.
- Evidence highlighting depends on PyMuPDF's text search; it correctly falls back to "couldn't highlight, here's the verified text" rather than faking a match, but can't highlight a clause whose exact wording doesn't literally appear on the PDF page (e.g. from encoding artifacts in unusual PDFs).
- Single-user, single-machine design — no authentication, no multi-user document isolation, no production-grade scaling.

## Roadmap (not implemented — see [`docs/case_study.md`](docs/case_study.md) for reasoning)

**V2:** multi-policy comparison, policy version tracking, and further conflict/exception detection beyond the specific pattern already fixed (see Evaluation) — e.g. exceptions spanning multiple sentences or clauses rather than one.
**V3:** a larger evaluation dataset across multiple real policies, a human-review workflow for flagged answers, multilingual support.
**A real production version** would additionally need: authentication and per-user document isolation, encrypted document storage, monitoring/alerting on the safety metrics measured here, a scalable vector database, and professional legal/compliance review — none of which are portfolio-appropriate to build without an actual insurer as a partner.

## License / Disclaimer

Educational/portfolio project. Not affiliated with any insurer. Not insurance advice. See the in-app disclaimer shown with every answer.
