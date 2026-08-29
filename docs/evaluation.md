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

Categories deliberately include the hardest cases, not just easy wins: broad/ambiguous questions with multiple valid answers, and **conflict/exception questions** — a coverage clause exists, but an exception elsewhere changes the answer.

One dataset row was corrected during this work: *"What dental treatments are not covered?"* originally only accepted clauses 4.2(c)/4.2(a) as citations, but clause 2.3 (nominally a coverage clause) also literally states the same exclusion — the policy repeats it in two places. Citing 2.3 is genuinely correct, not a citation error, so it was added to the accepted set.

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

**"Abstention-family" statuses**: `No Evidence Found`, `Insufficient Evidence`, and `Not Mentioned` are all treated as declining to assert a specific Covered/Excluded conclusion — `Not Mentioned` only ever claims absence ("the policy doesn't address this"), never a specific determination, so it's exactly as much of a hedge as `Insufficient Evidence`, just naming a different reason. **This was a bug fix made during this work** — see "Bug #2" below.

## Pipeline-stage attribution (for non-Correct, non-appropriately-abstained results)

- **Retrieval Failure** — the expected clause was never found even in a wide (top-10) search.
- **Evidence Selection Failure** — found in the wide search, but ranked outside the top-3 chunks actually sent to the LLM.
- **Generation Failure** — the LLM had the right evidence and still reasoned incorrectly, but the result was safely contained (an abstention or a caught validation failure).
- **Citation Failure** — right conclusion, wrong clause cited.
- **Guardrail Failure** — reserved for exactly what it sounds like: an unsafe/wrong answer **passed validation** and reached the user (every `Wrong-but-Confident` result, by definition).

## The full path, run by run

This project's evaluation numbers were revised twice during this work, both times by finding and fixing a real bug rather than accepting a flattering number. Both corrections are documented here rather than quietly folded in.

### Stage 1 — Original 1.5B baseline

| Metric | As first reported | Corrected (see Bug #2) |
|---|---|---|
| Answer Accuracy | 36.6% | 36.6% |
| Wrong-but-Confident | 7.3% (3/41) | **4.9%** (2/41) |

The two genuinely dangerous cases (both confirmed real under the corrected scoring too): *"Does cosmetic surgery after an accident get covered?"* and *"What if I need emergency dental surgery after a car accident?"* — both answered `Explicitly Excluded` when the clause's own accident exception should have made the answer `Covered`.

### Stage 2 — Two guardrail fixes (still on 1.5B)

**Fix 1 — conditional-exception guardrail.** Added a clause-metadata field (`exception_condition_text`, extracted via regex for `unless`/`except`/`only if`/`provided that`/`necessitated by`) and a rule: if a question's own wording overlaps with a cited clause's exception condition (stemmed comparison with a stoplist for generic terms like "claims"/"policy", to avoid false positives), an `Explicitly Excluded` conclusion sourced from that clause is downgraded to `Insufficient Evidence`.

Traced individually, both original dangerous cases flipped from `Explicitly Excluded` (wrong) to `Insufficient Evidence` (safe). **Wrong-but-Confident: 4.9% → 0%**, confirmed by re-tracing, not just reading the topline number.

**Fix 2 — bare status-label echo check.** While re-measuring Fix 1, a different, independent bug surfaced: the LLM sometimes returns an answer whose text is just a status label repeated back (`ANSWER: Insufficient Evidence`). This is a real, worthwhile fix (a genuinely malformed response was being trusted at face value) — but it turned out **not** to be the fix that got Wrong-but-Confident to zero; Fix 1 had already done that. Kept anyway, since it fixes a real bug independent of this specific metric.

### Stage 3 — Testing Qwen2.5-3B as a replacement

Swapping in the 3B model surfaced two more things:

**Bug #1 — evidence-ID bracket format.** The 3B model consistently writes `EVIDENCE_ID: [E1]` with brackets, which the parser didn't recognize, so `validate_evidence_id` correctly-but-wrongly rejected it. This spuriously failed **12 of 41 questions (29%)** whose actual answers were often good. Fixed by stripping `[]` during parsing — recognizing an equivalent label, not loosening which labels are considered valid.

**Bug #2 — the evaluation's own classification gap.** After Bug #1's fix, one case remained flagged Wrong-but-Confident: *"Is there a maternity benefit waiting period?"* → status `Not Mentioned`, answer *"There is no mention of a maternity benefit waiting period in the provided evidence"* — a **correct, safe answer**, misclassified only because `Not Mentioned` wasn't in the set of "abstention-family" statuses the classifier checks (see above). Fixed the classifier, then **recomputed every historical run from its already-stored per-question data — no new LLM calls needed** — to check how far this was skewing the historical record. Result: the original 1.5B baseline was actually 4.9%, not 7.3%; every run after Fix 1 was actually already at 0%, not 4.9% or 2.4%. The numbers in "Stage 1" and "Stage 2" above are already the corrected versions.

## Final results (run `20260829_131412`, Qwen2.5-3B, both fixes, corrected classification)

| Metric | Result |
|---|---|
| Answer Accuracy | 39.0% |
| Status Accuracy | 56.1% |
| Citation Accuracy | 73.3% |
| Retrieval Success Rate | 100% |
| Evidence Selection Success Rate | 100% |
| Appropriate Abstention Rate | **100%** |
| Incorrect Abstention Rate | 33.3% |
| **Wrong-but-Confident Rate** | **0.0%** |
| Validation Failure Rate | 4.9% (2/41) |
| Avg / median / max response time | 6.48s / 6.39s / 9.23s |
| Avg prompt / completion tokens | ~299 / ~37 |

Classification breakdown: **16 Correct, 14 Appropriate Abstention, 9 Incorrect Abstention, 2 Validation Failure, 0 Wrong-but-Confident, 0 Citation Failure.**

## Why accuracy is 39%, not 80%+

Two structural reasons, not one:

**1. 34% of the dataset (14/41) is deliberately unsupported or ambiguous.** Abstaining on these IS correct behavior — they can never contribute to "Correct" in the accuracy sense, and a system that guessed on them to raise this number would be doing exactly the wrong thing.

**2. Of the 27 questions that should get a confident answer, all 11 failures land on the same safe outcome.** Every one of the 9 Incorrect Abstentions and both Validation Failures resolved to `Insufficient Evidence` — **zero** resolved to a confidently wrong `Covered`/`Excluded`. The dominant pattern: paraphrased questions the model doesn't connect to a differently-worded clause (*"Is there a waiting period before any claims can be made?"* → doesn't connect to clause 3.2's "30 days" language), and conditional-exception clauses it still can't confidently resolve even at 3B scale (it now hedges instead of guessing wrong, which is the guardrail's intended effect, but doesn't make it *right*). One of the two Validation Failures also traced to a narrow false positive in the numeric claim-checker (flagging a stray "1" as an unsupported number on a simple definitional question) — a known, documented limitation of that heuristic.

Closing this gap further would require either a materially more capable model (7B+ starts to strain 8GB RAM) or making the system commit to an answer more often when uncertain — which would directly reintroduce Wrong-but-Confident risk. That trade-off is treated as deliberate here, not a bug to be optimized away.

## Model comparison: 1.5B vs 3B

| Metric | 1.5B (both guardrail fixes) | 3B (bracket fix, current default) |
|---|---|---|
| Answer Accuracy | 34.1% | **39.0%** |
| Citation Accuracy | 66.7% | **73.3%** |
| Appropriate Abstention | 85.7% | **100%** |
| Wrong-but-Confident | 0.0% | 0.0% |
| Avg response time | 3.08s | 6.48s |
| Approx. RAM (model + KV cache) | ~1.9-2.1GB | ~2.7GB |

The 3B model won on every accuracy metric while keeping an identical, clean safety record — at roughly double the response time and ~700MB more RAM, both still comfortable on an 8GB machine. Made it the project default.

## A note on run-to-run variance

Two identical 1.5B baseline runs produced **bit-for-bit identical** Answer Accuracy and Status Accuracy, while Wrong-but-Confident shifted a couple of points purely from LLM sampling noise (temperature is 0.1, not exactly 0). Treat single-run deltas of a few percentage points as noise; the 0% Wrong-but-Confident results here were specifically confirmed by tracing individual questions, not by trusting a topline number alone.

## Limitations of this evaluation itself

- 41 questions against one synthetic policy is a useful diagnostic sample, not a statistically powered benchmark. Real-world policy language varies far more.
- The composite correctness check (status + citation + keywords) is a deterministic proxy for "the answer is right," not a substitute for a human reading every response.
- The dataset was authored by the same person who built the system. An independent reviewer authoring the golden dataset would be a meaningfully stronger evaluation.
- This evaluation's own code had two real bugs during this work, both found by looking closely at specific traced examples rather than trusting summary statistics — a reminder that an evaluation harness needs the same scrutiny as the system it measures.
