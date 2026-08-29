# Resume, LinkedIn, and Interview Pitch

All claims below are drawn directly from `docs/evaluation.md`'s measured results — no invented metrics.

## Resume (2-3 bullets)

- Designed and built an evidence-grounded RAG application (Python, FAISS, local LLM via llama.cpp, Streamlit) that answers questions about insurance policy PDFs and abstains rather than guesses when evidence is insufficient — fully local/private inference, zero paid APIs.
- Built a multi-layer trust system (application-controlled citations, source-quote validation, LLM-blind guardrails) that cross-checks LLM conclusions against independently-computed document metadata, automatically downgrading unsafe answers before they reach the user.
- Ran a 41-question end-to-end evaluation against the live pipeline, measuring a "Wrong-but-Confident" safety rate (7.3%) alongside accuracy — surfacing a specific, prioritized architectural fix rather than generic prompt tuning.

## LinkedIn Project Description

I designed and built PolicyLens AI, a local, evidence-grounded RAG application that helps users understand insurance policy documents without relying on general AI knowledge. The system enforces a strict "no evidence, no answer" architecture: a local LLM (Qwen2.5, via llama.cpp) only ever explains evidence retrieved by FAISS/embeddings, and every citation shown to users is resolved from application-controlled document metadata — the LLM can never invent a page or clause number.

I built a dedicated trust layer that validates citations, checks quotes against source text, flags unsupported numeric claims, and cross-checks the LLM's conclusions against independently-computed clause metadata (catching, for example, a "Covered" answer sourced from a clause in the Exclusions section). I ran a 41-question end-to-end evaluation against the full live pipeline — not a mocked version — and measured concrete safety metrics, including a "Wrong-but-Confident rate," which I treat as the single most important number for a trustworthy AI product. The result: 100% retrieval success, 85.7% appropriate abstention on unsupported questions, and a specific, evidence-backed priority for the next improvement rather than a vague "it works most of the time."

## Interview Pitch (30-60 seconds)

"I built an AI tool that lets you upload your insurance policy and ask questions in plain English — 'is dental covered,' 'what's the waiting period' — and get an answer that's always backed by an exact clause and page number from your actual document, never from general knowledge.

The core product decision was treating 'I don't know' as a feature, not a failure: the system refuses to answer — cheaply, before even calling the LLM — when it can't find strong enough evidence, and every citation it shows is built entirely by my own code from the retrieved text, so the LLM literally never gets to write a page number and can't invent one.

The hardest part was realizing a small local model's mistakes don't disappear with better prompting — they just move. I caught this directly: a prompt fix that corrected one wrong answer immediately caused the model to fabricate confident reasoning on two other, unrelated questions. That pushed me toward guardrails that check the model's output against data it never sees at all — like whether the clause it cited is structurally from an 'exclusions' section — which is a far more reliable way to catch a wrong answer than continuing to tune wording.

I evaluated the whole system end-to-end on 41 real questions and specifically tracked what I call the 'wrong-but-confident' rate, because that's the number that actually reflects user trust and risk — not overall accuracy."
