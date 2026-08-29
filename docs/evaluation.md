# Evaluation Methodology and Results

## Why evaluate the whole pipeline, not just the LLM

A retrieval bug, a chunking gap, or a validation bypass can all produce a wrong answer even with a perfect LLM. This evaluation runs the **actual user-facing function** (`generate_grounded_response`) for every question — real embedding search, real FAISS retrieval, real local LLM calls, real Phase 8 guardrails. Nothing is mocked or short-circuited.

## The golden dataset

`app/evaluation/golden_end_to_end_dataset.csv` — 41 questions, hand-verified against the actual synthetic test policy (never generated or graded by an LLM). Each row specifies:

| Column | Meaning |
|---|---|
| `expected_status` | `Covered` / `Explicitly Excluded` / `Not Mentioned` / `Insufficient Evidence` / `No Evidence Found` |
| `expected_clause_number` | The clause(s) that should be cited (`\|`-separated if multiple are acceptable; `NONE` for unsupported questions) |
| `expected_key_facts` | Optional `\|`-separated keywords/numbers that should appear in the answer (e.g. `48` for a waiting-period answer) |
| `should_abstain` | Whether the correct behavior is some form of non-committal response |
| `category` | Coverage / Exclusion / Waiting Period / Eligibility / Exception / Ambiguous / Unsupported |

Categories deliberately include the hardest cases, not just easy wins: broad/ambiguous questions with multiple valid answers, and **conflict/exception questions** — a coverage clause exists, but an exception elsewhere changes the answer (e.g. *"Does cosmetic surgery after an accident get covered?"*, where the "exclusion" clause itself contains the accident carve-out that makes the true answer "yes").

## Grading without an LLM judge

Grading free-text answers semantically would normally need an LLM judge. This project deliberately avoids that (consistent with never letting an LLM define its own ground truth) and instead uses a **deterministic composite**, the same philosophy as the numeric claim-checker in Phase 8:

- **status_correct** — does the final status match the expected status?
- **citation_correct** — does the displayed citation point to an accepted clause?
- **key_facts_ok** — do the expected keywords/numbers appear in the answer text (case-insensitive substring)?

`Correct` requires all three. This is a proxy, not a certainty — see Limitations below.

## Classification (every question gets exactly one)

| Classification | Meaning |
|---|---|
| Correct | All three checks passed |
| Incorrect | Status + citation correct, but a key fact was missing |
| Appropriate Abstention | Should abstain, and did |
| Incorrect Abstention | Should have answered, but abstained instead |
| **Wrong-but-Confident** | Gave a substantive, non-abstaining status that was wrong |
| Citation Failure | Status correct, but pointed at the wrong clause |
| Validation Failure | Phase 8's guardrails rejected the LLM's raw response outright |

## Pipeline-stage attribution (for non-Correct, non-appropriately-abstained results)

- **Retrieval Failure** — the expected clause was never found even in a wide (top-10) search.
- **Evidence Selection Failure** — found in the wide search, but ranked outside the top-3 chunks actually sent to the LLM.
- **Generation Failure** — the LLM had the right evidence and still reasoned incorrectly, but the result was safely contained (an abstention or a caught validation failure).
- **Citation Failure** — right conclusion, wrong clause cited.
- **Guardrail Failure** — reserved for exactly what it sounds like: an unsafe/wrong answer **passed validation** and reached the user (i.e., every `Wrong-but-Confident` result, by definition, since anything the guardrails actually caught becomes `Validation Failure` / Generation Failure instead).

## Results: baseline, then two measured fixes

| Metric | Baseline (`20260829_075507`) | After both fixes (`20260829_084435`) |
|---|---|---|
| Answer Accuracy | 36.6% | 34.1% |
| Status Accuracy | 56.1% | 53.7% |
| Citation Accuracy | 73.3% | 66.7% |
| Retrieval Success Rate | 100% | 100% |
| Evidence Selection Success Rate | 100% | 100% |
| Appropriate Abstention Rate | 85.7% | 85.7% |
| Incorrect Abstention Rate | 29.6% | 40.7% |
| **Wrong-but-Confident Rate** | **7.3%** (3/41) | **0.0%** (0/41) |
| Validation Failure Rate | 4.9% (2/41) | 7.3% (3/41) |
| Avg / median / max response time | 3.26s / 3.14s / 5.72s | 3.08s / 3.11s / 4.11s |
| Avg prompt / completion tokens | ~299 / ~37 | ~299 / ~37 |

Baseline classification breakdown: **15 Correct, 12 Appropriate Abstention, 8 Incorrect Abstention, 3 Wrong-but-Confident, 2 Validation Failure, 1 Citation Failure.** Failure-stage breakdown: **10 Generation Failure, 3 Guardrail Failure, 1 Citation Failure.**

## What the baseline meant

**Retrieval and evidence selection were not the bottleneck** — both measured 100%. The dominant failure (Generation Failure, 24% of all questions) was the local LLM being *too cautious*: given clear, correctly-retrieved evidence, it still sometimes hedged to "Insufficient Evidence" rather than committing to a definite status — the safer failure direction, given this product's core safety principle.

The smaller but more serious category — 3 Wrong-but-Confident cases (7.3%, above this project's own 5% concern threshold) — all traced to the **same specific pattern**: a clause stating a general rule and a specific exception in one sentence (e.g. *"...excluded unless required to treat an accidental injury"*), where the LLM applied the general rule to a question specifically describing the exception's own scenario.

## The two fixes, and what they actually did

**Fix 1 — conditional-exception guardrail.** Added a clause-metadata field (`exception_condition_text`, extracted via regex for `unless`/`except`/`only if`/`provided that`/`necessitated by`) and a guardrail rule: if a question's own wording overlaps with a cited clause's exception condition (stemmed word comparison with a stoplist for generic insurance terms like "claims"/"policy", to avoid false positives), an `Explicitly Excluded` conclusion sourced from that clause is downgraded to `Insufficient Evidence` rather than trusted.

Traced individually: *"Does cosmetic surgery after an accident get covered?"* and *"What if I need emergency dental surgery after a car accident?"* both flipped from `Explicitly Excluded` (wrong) to `Insufficient Evidence` (safe) — 2 of the original 3 Wrong-but-Confident cases fixed. Wrong-but-Confident: 7.3% → 4.9%.

**Fix 2 — bare status-label echo check.** While re-measuring Fix 1, a *different*, previously-unnoticed bug surfaced: the LLM sometimes returns an answer whose text is just a status label repeated back (e.g. `STATUS: Not Mentioned` with `ANSWER: Insufficient Evidence`) — a self-contradictory, non-explanatory response. This exact glitch had actually appeared once already in the very first baseline run (on *"Are dentures covered?"*), just not investigated at the time. Added a check in the response parser: if the answer text is nothing but a bare echo of a status word/phrase, treat the whole response as malformed (same safe-fallback path as any other validation failure) rather than trust it.

Re-traced *"Are dentures covered?"* after this fix: it now produces a coherent, safe `Insufficient Evidence` answer instead of the garbled echo. Wrong-but-Confident: 4.9% → **0.0%**.

## Reading the accuracy dip correctly

Answer Accuracy fell slightly (36.6% → 34.1%) and Incorrect Abstention rose (29.6% → 40.7%) across these fixes. **This is the expected, intentional trade-off, not a regression to be alarmed by**: both fixes work exclusively by converting a wrong, confidently-stated answer into a safe "I'm not sure" — which by definition can only move a question out of "Correct" or "Wrong-but-Confident" and into "Incorrect Abstention," never the other direction. A system that trades confident wrongness for honest uncertainty is moving in the direction this entire project is designed around, even though it makes the blended accuracy number look slightly worse.

## A note on run-to-run variance

Two identical baseline runs produced **bit-for-bit identical** Answer Accuracy and Status Accuracy (36.6% / 56.1%), while Wrong-but-Confident shifted a couple of points purely from LLM sampling noise (temperature is 0.1, not exactly 0). Treat single-run deltas of a few percentage points as noise, not a real change, **except** for Wrong-but-Confident's move to a clean 0% here, which was verified by tracing the specific previously-failing questions individually, not just by reading the topline number.

## Limitations of this evaluation itself

- 41 questions against one synthetic policy is a useful diagnostic sample, not a statistically powered benchmark. Real-world policy language varies far more.
- The composite correctness check (status + citation + keywords) is a deterministic proxy for "the answer is right," not a substitute for a human reading every response — it can miss subtly wrong phrasing that happens to hit the right status, citation, and keywords.
- The dataset was authored by the same person who built the system. An independent reviewer authoring the golden dataset would be a meaningfully stronger evaluation.
