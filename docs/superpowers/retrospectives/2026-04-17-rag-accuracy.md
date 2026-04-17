# RAG Accuracy Workstream — Retrospective

**Date:** 2026-04-17
**Spec:** `docs/superpowers/specs/2026-04-17-rag-accuracy-hybrid-retrieval-design.md`
**Plan:** `docs/superpowers/plans/2026-04-17-rag-accuracy-hybrid-retrieval.md`
**Branch:** `feat/rag-hybrid-retrieval`
**Tasks:** 11 implemented (0–10), Task 11 = this doc.
**Eval bank:** `scripts/intensive_test.py` (106 queries, HR + contracts).

---

## Outcome

**Spec gate: FAILED.** Faithfulness did not clear the 0.80 bar.

| Metric | Baseline (post-grounding-fix snapshot) | Final | Delta | Target | Result |
|---|---:|---:|---:|---|---|
| `answer_faithfulness` | 0.519 | **0.538** | **+0.019** | ≥0.80 | FAIL |
| `context_recall` | 0.840 | 0.807 | **−0.033** | ≥0.75 | PASS (regressed) |
| `hallucination_rate` | 0.000 | 0.000 | 0 | ≤0.05 | PASS |
| `grounding_bypass_rate` | 0.000 | 0.000 | 0 | ≤0.02 | PASS |
| Latency p50 | not captured pre-change | 12.3s | — | ≤baseline+300ms | UNMEASURED |
| Latency p95 | not captured pre-change | 30.7s | — | — | — |
| Grade distribution | — | A:20 B:63 C:6 D:5 F:12 | — | — | 83/106 passing (78%) |

The RAGAS harness reports `pass: false`. 17/106 queries still graded C/D/F.

Apr 11 reference point (different weights, different grounding code): faith 0.439, recall 0.561. The baseline captured post-model-symlink-repoint and post-grounding-fix is the right comparison.

## What moved what

| Task | Intended effect | Observed effect |
|---|---|---|
| Task 2 (grounding fix) | Restore meaning to the `grounded` signal. Not expected to move RAGAS. | Pre-existing "grounded=false on 12/12" failure mode resolved — the reason Apr 11 RAGAS faith looked artificially low. Cannot be isolated in this A/B because it landed before the baseline. |
| Task 4 (ingestion sparse) | Forward-looking. Every new upload carries SPLADE. | No eval impact (no new uploads during eval). |
| Task 7 (backfill) | Populate `keywords_vector` on 1,636 existing chunks across 4 collections. | Unblocks hybrid retrieval. 2 legacy 1-point test collections rejected the writes because their schema predates the `keywords_vector` slot — out-of-scope to fix. |
| Task 8 (hybrid dense+sparse+RRF) | Primary recall lever. | Did not help. Recall was already 0.84 — there was no recall gap for hybrid to close. RRF added candidates but `_ensure_document_diversity` already saturates the top_k budget from dense alone. |
| Task 9 (KG expansion) | Cross-document recall for entity-linked cases. | No measurable gain on this bank. Likely introduced modest noise: KG-scored chunks at 0.4 are competitive with dense borderline hits, so some rerank candidate swaps happened without a corresponding quality signal to judge "better." |
| Combined hybrid + KG | Expected: recall 0.56 → 0.75+ per the spec's premise. | Actual: recall 0.840 → 0.807. **The premise was wrong — recall was not the bottleneck.** |

## What the spec got right and wrong

**Right:**
- Diagnose-first on the grounding check. Task 1 isolated the exact failing gate (`overlap < 5` absolute floor on the word-gate) against my guess that the number gate was the culprit. Task 2's fix was specific and regression-locked; reviewer caught a follow-up hole (negation-entity false positive) that was fixed inline.
- Decomposition into four workstreams. RAG accuracy as the first workstream gave a fast, measurable bracket: we now know retrieval isn't the bottleneck, so workstream 2 (extraction) and workstream 3 (model training) are the right priorities, not more retrieval work.
- Sparse backfill script architecture (inventory → dry-run → full). Found and fixed two real Qdrant-API gotchas in production (payload-index prerequisite for `IsEmptyCondition`, `IsEmptyCondition` vs `IsNullCondition` semantic). Zero corruption; 99.9% of points backfilled.
- Code review caught a scoring bug in `_rrf_merge` (mutating `chunk.score` with sub-unit RRF values broke `_HIGH_QUALITY_THRESHOLD` and keyword-fallback scale). Fix landed before final eval.

**Wrong:**
- The central premise. The spec assumed the Apr 11 `context_recall=0.561` was a real measurement of DocWain's retrieval. By 2026-04-17 the model symlink had been repointed to the HF-recovered weights, the baseline had implicitly shifted, and recall was already 0.840. I should have run the fresh baseline FIRST and then scoped the workstream — instead the spec was written before baseline capture, and the baseline came in during Task 0 mid-execution. That's a process error: re-measure before re-planning.
- Overconfidence in sparse-recall ceiling. SPLADE sparse was expected to move recall substantially. With dense already hitting most high-relevance chunks and `_ensure_document_diversity` enforcing per-doc coverage, the additional candidates from sparse mostly shuffled rank within an already-complete top-k. The mathematical model "sparse + dense + RRF > dense alone" is true in isolation but not visibly so when the live pipeline already has a diversity guarantee.
- Under-investment in the generation side. Faithfulness 0.519 → 0.538 means 48% of responses are still flagged as not-grounded-enough by the RAGAS harness. That's a prompt/model problem, not a retrieval problem. The spec explicitly deferred this — correctly — but we should have sized the retrieval ceiling first to avoid the 0.42-point gap surprise.

## Recommended next workstream

**Workstream 3 first, not Workstream 2.** Extraction quality (workstream 2) matters for downstream uses but is not the faithfulness-gap driver on this bank. The 17 failing cases break down as:

- Prompt-brittleness: "give brief summary" returns "profile not specified" when retrieval found relevant evidence (model doesn't synthesize; treats the query literally). 6 cases.
- Format-underfilled answers: single-line responses where the bank expected structured tables/lists despite `task_type=extract` mapping. 5 cases.
- Cross-doc aggregation: "average years of experience across all candidates" requires arithmetic the model refuses when context is long. 3 cases.
- Legitimately missing data (bank expects a value the docs don't have). 3 cases.

Of those, 14/17 are model/prompt issues, 3/17 are data issues. That argues for:

1. **Continue V2 training with a focus on multi-doc synthesis and numeric aggregation** (tracks defined in `project_v2_training_status` / `project_docwain_v2`).
2. **Tighten `src/generation/prompts.py` task-type formatting** — specifically `extract` + `aggregate` + `overview` — to force the model into the structured output shape even when evidence is ambiguous.
3. **Skip workstream 2 (extraction) unless the failing case analysis changes.** The current extraction (V2 model on Ollama — the only non-stub) is already providing chunks that retrieval is finding.

## Recall regression — follow-up flag

Recall went **0.840 → 0.807 (−0.033)** despite no query-side regression. Candidate root causes, in rough order of likelihood:

1. `_ensure_document_diversity` now receives KG-sourced chunks (`chunk_type="kg"`, `score=0.4`) alongside real retrieval hits. Its round-robin picks the top-scoring chunk per document — when a KG chunk is the only representative of a document in the candidate pool, it may displace a lower-score-but-more-topical dense hit from a different document that RAGAS's context-recall metric then misses.
2. RRF re-ranking shuffled the input to `_fill_missing_documents`, which uses `seen` document IDs to decide what to scroll for. A slightly different top-k order means different documents are "seen," which changes which fills get added.
3. Rarely: the Qdrant payload index on `sparse_backfilled_at` was added during Task 7. Payload-index creation briefly perturbs query latency on large collections — unlikely but worth noting.

**Proposed follow-up (not in this spec):** narrow KG expansion so it contributes chunks only when dense+sparse didn't find the linked entity, rather than unconditionally for every matched query entity.

## Artefacts

**Commits (14):** `feat/rag-hybrid-retrieval` branch, base `accd959` (plan) … HEAD `a659b20` (Task 10).
- `7386ed5` Task 1 — diagnostic
- `f059386` + `aa42b8c` Task 2 — grounding fix + follow-up
- `7b3366b` Task 0 — baseline capture
- `441812f` + `88d92b0` Task 3 — SparseEncoder on AppState
- `b41af41` Task 4 — ingestion sparse
- `651c56b` Task 5 — backfill skeleton
- `d27b25a` Task 6 — backfill processor
- `fc9deea` — SPLADE model switch (ungated)
- `bffadd3` — payload-index fix for backfill
- `005f0ea` + `d87c2d5` Task 8 — hybrid retrieval + RRF-score fix
- `f9115af` Task 9 — KG expansion
- `a659b20` Task 10 — delete dead code

**Evidence files:**
- `tests/ragas_metrics.baseline.json` — pre-hybrid snapshot
- `tests/ragas_metrics.json` — post-hybrid final
- `/tmp/final_intensive.log` — full eval output
- `/tmp/backfill_all_v2.log` — backfill run log

## Merge recommendation

**Merge the branch anyway.** Reasons:

1. The grounding check fix alone is worth merging — it restores a correctness signal that was reporting `false` on every query.
2. The sparse-retrieval infrastructure is now live. Even if the current bank doesn't reward it, future queries with exact-term signal (legal IDs, product codes) will benefit.
3. The 1,636-chunk backfill already mutated production Qdrant state. Reverting the code without reverting the Qdrant state leaves a mismatch (data has sparse, code doesn't read it).
4. The 3-layer dead retriever deletion is pure cleanup.

Accept the −0.033 recall regression as a follow-up; gate the next merge on closing it.

**Do NOT claim the faith/recall gate.** Label this explicitly as "infrastructure landed, gate not met, pivot to generation workstream" when merging.
