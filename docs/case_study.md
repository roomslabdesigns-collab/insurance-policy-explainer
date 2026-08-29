# Case Study: PolicyLens AI

*An evidence-grounded RAG system for explaining insurance policy documents, built end-to-end — retrieval, local LLM integration, a trust layer, evidence highlighting, and evaluation — as a portfolio demonstration of AI product judgment, not just implementation.*

## The Problem

Insurance policies are dense, legally-worded documents that ordinary customers rarely read fully, yet the consequences of misunderstanding one are real — a denied claim, or missing a benefit they were entitled to. The naive AI answer to this — "put the policy in a chatbot" — actively makes the problem worse if the chatbot can answer confidently from general insurance knowledge instead of the actual document in front of it. The real product problem isn't "can an LLM answer questions about a PDF" (trivially yes); it's **"can a user trust the answer enough to act on it, and can they verify it themselves in under 10 seconds if they want to?"**

## Product Decision: Why Evidence-Grounded RAG, Not a General Chatbot

Three decisions came directly from that trust requirement, made before any code was written:

1. **NO EVIDENCE → NO ANSWER** as a hard rule, not a soft preference — the system must be structurally incapable of falling back on general knowledge, not just instructed not to.
2. **The LLM is a narrator, not a source of truth.** Every citation a user sees must be traceable to application-controlled metadata, never to something the LLM wrote — because a citation the LLM could invent is a citation that will eventually be wrong in a way nobody catches.
3. **Abstention is a feature to be measured, not a failure to be minimized.** A system that never says "I don't know" is either omniscient or lying. This is why "Wrong-but-Confident Rate" became the single most-tracked metric in the whole project — it's the number that actually corresponds to real-world harm.

## System Design

Twelve phases, each independently tested before the next began: PDF extraction → clause-aware chunking (not fixed-size — regex-based section/clause detection, deliberately requiring a decimal point in clause numbers after a false-positive regression where "48 months" was misread as clause "48") → embeddings + FAISS retrieval → local LLM integration via `llama.cpp`'s server (so the ~1.5GB model loads exactly once, regardless of Streamlit's rerun-on-every-click execution model) → structured, evidence-ID-based generation → a Python-enforced trust layer → a Streamlit UI → PDF evidence highlighting → end-to-end evaluation with a dashboard. Full technical detail is in the main README; this section is about the decisions, not the stack.

## Key Product Decisions (and what changed my mind)

**Prioritizing abstention over guessing.** Early testing (Phase 6) showed the local LLM, when given genuinely irrelevant evidence, would still answer *"The capital of France is Paris"* rather than admit it had nothing relevant. The fix wasn't a better prompt — it was a Python-level relevance floor that rejects the question *before the LLM is ever called*, at zero token cost. Prompt instructions are a request; a code-level gate is a guarantee.

**A prompt fix I tried, then reverted.** When the model mislabeled a clearly-worded exclusion as "Covered," I tried clarifying the prompt's definition of each status word. It fixed that one case — and immediately caused the model to fabricate confident cross-clause reasoning on two *other*, unrelated questions ("this excludes cosmetic surgery, therefore it doesn't cover space travel"). I reverted the change. The lesson that shaped the rest of the project: **a small model's failure modes don't disappear under prompt tuning, they move** — and a code-level guardrail using data the LLM never sees is more reliable than iterating on wording. That single decision is why Phase 8's metadata cross-check (comparing the LLM's claimed status against independently-computed clause metadata) exists at all, and it's what caught the very mislabeling the reverted prompt fix was trying to solve — for free, using data already in memory.

**Application-controlled citations, enforced structurally, not just by convention.** The LLM never sees a page number or clause number in its prompt — only short, meaningless labels (`E1`, `E2`, `E3`). This isn't a validation rule bolted on afterward; it's a constraint on what information physically reaches the model, which means there is no code path by which an invented citation could occur, not just a check that catches it after the fact.

**Evidence highlighting had to fail honestly.** The tiered text-location strategy (exact → sentence-level → short excerpt) will sometimes find nothing on the actual PDF page. The tempting shortcut — highlight the whole page, or show the stored text as if it were confirmed — was explicitly rejected. When highlighting fails, the UI says so and still shows the independently-verified clause text, because a fabricated highlight is *more* misleading than an honest "couldn't auto-locate this."

**Measuring Wrong-but-Confident separately from "accuracy."** A single blended accuracy score would have hidden the most important finding in this project: retrieval is not the bottleneck (100% success), the dominant failure is the system being *too cautious* (safe direction), and the dangerous failures cluster in one specific, nameable pattern (conditional-exception clauses). None of that is visible in a single number.

## Evaluation

41 hand-verified questions run through the real pipeline (not mocked) found: 100% retrieval success, 85.7% appropriate abstention, and a 7.3% Wrong-but-Confident rate — above the 5% threshold I'd set for myself as acceptable. Digging into *why* mattered more than the number itself: all three Wrong-but-Confident cases traced to the same pattern — a clause stating a general rule and an exception in one sentence, where the guardrail's binary "does this clause contain exclusion language" flag can't distinguish "purely an exclusion" from "an exclusion with a carve-out that changes the answer for this specific question." That's a precise, actionable finding a blended accuracy score would never have surfaced. Full methodology and numbers: `docs/evaluation.md`.

I also found and fixed two bugs *in the evaluation code itself* while preparing this case study — a failure-stage mislabeling and a dataset row that silently produced a false "retrieval failure." I'm documenting that here deliberately: an evaluation system that isn't itself scrutinized can be as misleading as the system it's evaluating.

## Iterations

**What I did after the first evaluation.** The item I originally listed here as the top future priority — a metadata-based conditional-exception guardrail — I actually implemented and measured, rather than leaving as a proposal. Result: Wrong-but-Confident dropped from 7.3% to 0%, verified by tracing the specific failing questions individually before and after, not just by reading the topline number. Along the way, re-measuring surfaced a *second*, unrelated bug (the LLM occasionally echoing a bare status label as its "answer") that had actually appeared once in the very first baseline run without being investigated — fixed with a second small, targeted check. Full mechanics and the exact before/after numbers: `docs/evaluation.md`.

The pattern I'd point to as the actual product lesson here: **the fix that worked was found by asking "why" of a bad number, twice** — first tracing the Wrong-but-Confident cases to one clause pattern, then noticing during re-verification that a "fixed" run still had a different bug hiding in what looked like noise. Neither fix would have been found by looking at the accuracy percentage alone.

## What I'd Improve Next

1. **Test against real, diverse policy documents.** Everything here was validated against one synthetic 7-page policy. Real policies vary far more in structure, and clause-detection regex tuned against one document's formatting will need re-validation.
2. **A larger local model (3B) as an A/B comparison** against the current 1.5B, to see whether the "too cautious" (Generation Failure) rate improves, and by how much, against the real cost in response latency on 8GB hardware.
3. **Broaden the conditional-exception guardrail** beyond single-sentence exceptions — it doesn't yet handle an exception stated in a separate clause from the general rule it modifies.
4. **An independent golden-dataset author.** I built the system and wrote its own test — a second reviewer authoring ground truth would be a materially stronger evaluation.
