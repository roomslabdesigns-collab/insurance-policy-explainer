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

## Results (run `20260829_075507`, 41 questions)

| Metric | Result |
|---|---|
| Answer Accuracy | 36.6% |
| Status Accuracy | 56.1% |
| Citation Accuracy | 73.3% |
| Retrieval Success Rate | 100% |
| Evidence Selection Success Rate | 100% |
| Appropriate Abstention Rate | 85.7% |
| Incorrect Abstention Rate | 29.6% |
| **Wrong-but-Confident Rate** | **7.3%** (3/41) |
| Validation Failure Rate | 4.9% (2/41) |
| Avg / median / max response time | 3.26s / 3.14s / 5.72s |
| Avg prompt / completion tokens | ~299 / ~37 |
| Questions answered with zero LLM calls (pre-LLM gate) | 0/41 |

Classification breakdown: **15 Correct, 12 Appropriate Abstention, 8 Incorrect Abstention, 3 Wrong-but-Confident, 2 Validation Failure, 1 Citation Failure.**

Failure-stage breakdown (of the 14 non-correct, non-appropriately-abstained results): **10 Generation Failure, 3 Guardrail Failure, 1 Citation Failure.**

## What this means

**Retrieval and evidence selection are not the bottleneck** — both measured 100% on this dataset. The system's dominant failure (Generation Failure, 24% of all questions) is the local LLM being *too cautious*: given clear, correctly-retrieved evidence, it still sometimes hedges to "Insufficient Evidence" rather than committing to a definite status. Given this product's core safety principle, that is the *safer* failure direction — 27 of 41 questions (66%) landed on either a fully correct answer or an appropriate abstention.

The smaller but more serious category — the 3 Wrong-but-Confident cases (7.3%, above this project's own 5% concern threshold) — all trace to the **same specific pattern**: a clause that states a general rule and a specific exception in one sentence (e.g. *"...excluded unless required to treat an accidental injury"*). The guardrail's exclusion-detection metadata is a binary flag (does this clause contain exclusion language, yes/no); it can't yet tell "this clause is purely an exclusion" apart from "this clause has a conditional carve-out that changes the answer for this specific question." That is the single highest-priority fix identified by this evaluation (see the case study for reasoning on why a metadata-based fix is preferred over further prompt tuning).

## A note on run-to-run variance

Running the identical evaluation twice in a row produced **bit-for-bit identical** Answer Accuracy and Status Accuracy (36.6% / 56.1%), while the Wrong-but-Confident rate shifted from 4.9% to 7.3% purely from LLM sampling noise (temperature is 0.1, not exactly 0). Treat single-run deltas smaller than a few percentage points as noise, not a real regression, when comparing future runs.

## Limitations of this evaluation itself

- 41 questions against one synthetic policy is a useful diagnostic sample, not a statistically powered benchmark. Real-world policy language varies far more.
- The composite correctness check (status + citation + keywords) is a deterministic proxy for "the answer is right," not a substitute for a human reading every response — it can miss subtly wrong phrasing that happens to hit the right status, citation, and keywords.
- The dataset was authored by the same person who built the system. An independent reviewer authoring the golden dataset would be a meaningfully stronger evaluation.
