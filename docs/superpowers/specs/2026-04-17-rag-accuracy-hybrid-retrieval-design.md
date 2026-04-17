# RAG Accuracy — Hybrid Retrieval + Grounding Fix

**Date:** 2026-04-17
**Workstream:** 1 of 4 (RAG accuracy). Extraction, model training, and response-speed tuning are separate specs.
**Status:** Approved design, awaiting implementation plan.

## Problem

The last RAGAS run (`tests/ragas_metrics.json`, 2026-04-11, 106 queries) reported:

- `answer_faithfulness: 0.439` (target 0.85)
- `context_recall: 0.561` (target 0.80)
- `hallucination_rate: 0.019` (target ≤0.05)
- `grounding_bypass_rate: 0.009` (target ≤0.02)

Ten HR cases graded F with `faithfulness = 0.25`. `tests/quality_audit_results.json` shows 13/13 cases returning `grounded: false` — including cases that pass content assertions. The grounded signal in logs and UI is currently meaningless.

## Goal

Lift this RAG workstream to **faithfulness ≥ 0.80, context recall ≥ 0.75** on the same 106-query bank, without rewriting the retrieval layer or blocking on the extraction, training, or consolidation workstreams.

## What this spec is not

- Not a model training change.
- Not a fix to the three stub extractors (`structural.py`, `semantic.py`, `vision.py`).
- Not a consolidation of the 25 files in `src/retrieval/`.
- Not a systemd/service rename (`docwain-fast` → `docwain-vllm`).
- Not a reconciliation of `/no_think` vs `Reasoner.use_thinking`.
- Not a visualization, content-generation, or Teams-app change.

Each of those is deferred to its own spec.

## Key insight

The components needed for hybrid retrieval and KG expansion already exist in the codebase — they are simply not wired into the live path:

| Component | File | Used by live path? |
|---|---|---|
| SPLADE sparse encoder | `src/embedding/sparse.py` | No |
| RRF fusion | `src/retrieval/fusion.py` | No |
| BGE-M3 hybrid skeleton | `src/retrieval/bgem3_retriever.py` | No |
| 3-layer retriever (Qdrant+Neo4j+Mongo) | `src/retrieval/unified_retriever.py` | No (unimported) |
| `QueryUnderstanding.entities` | `src/agent/intent.py` | Produced but not forwarded to retriever |
| Qdrant sparse vector slot `keywords_vector` | `src/api/vector_store.py:249` | Provisioned, never populated |

The live path (`core_agent.py` → `src/retrieval/retriever.py::UnifiedRetriever`) is dense-only with a keyword fallback. This spec wires the unused components into it.

## Architecture

```
    Current live path                                 Target path (same files, wired in)
    ─────────────────                                 ──────────────────────────────────
    IntentAnalyzer                                    IntentAnalyzer
       ↓                                                 ↓ (entities forwarded downstream)
    UnifiedRetriever (retriever.py)                   UnifiedRetriever (retriever.py)
       ├─ dense (BGE-large) only                         ├─ dense (BGE-large)
       └─ keyword fallback                               ├─ sparse (SPLADE, embedding/sparse.py)   ← NEW WIRE
                                                        ├─ RRF fuse (fusion.py)                     ← NEW WIRE
                                                        └─ KG expand (Neo4j, via intent entities)   ← NEW WIRE
       ↓                                                 ↓
    rerank_chunks (reranker.py)                       rerank_chunks (reranker.py, unchanged)
       ↓                                                 ↓
    Reasoner                                          Reasoner
       └─ _check_grounding (reports False universally)   └─ _check_grounding (diagnosed and fixed)  ← NEW FIX
```

### Per-query data flow (post-change)

1. `/api/ask` → `execution/router.py` → `CoreAgent.run` (all unchanged).
2. `IntentAnalyzer.analyze(query)` produces `QueryUnderstanding{task_type, output_format, entities, domain_tags, ...}` (unchanged).
3. `UnifiedRetriever.retrieve(query, query_entities=understanding.entities, ...)`:
   - Dense search (BGE-large, Qdrant named-vector `content_vector`).
   - Sparse search (SPLADE via `SparseEncoder`, Qdrant named sparse vector `keywords_vector`).
   - KG expansion: for each of the top-5 query entities, Neo4j 1-hop traversal to pull related `Chunk` IDs (cap 10 per entity).
   - Dense and sparse ID lists fused by `fusion.reciprocal_rank_fusion` with weights `{dense: 0.6, sparse: 0.4}`.
   - KG chunk IDs merged into the fused list while preserving rank order.
   - Hydrate top_k=50 to `List[EvidenceChunk]`.
4. `rerank_chunks` (unchanged) — dense_score + keyword_F1 + cross-encoder (CPU, top-10 pairs).
5. `build_context` (unchanged) — per-task evidence counts.
6. `Reasoner.reason` — same LLM call against vLLM :8100; `_check_grounding` fixed to report truthfully.
7. `compose_response` (unchanged).

### Latency budget (p50, A100)

| Stage | Before | After | Delta |
|---|---|---|---|
| UNDERSTAND | 400ms | 400ms | 0 |
| Dense retrieve | 120ms | 120ms | 0 |
| Sparse retrieve | — | 150ms | **+150ms** (parallel with dense — effective ~0) |
| KG expand | — | 80ms | **+80ms** (parallel — effective ~0) |
| RRF fuse | — | 5ms | +5ms |
| Rerank (CE) | 600ms | 600ms | 0 |
| Context build | 20ms | 20ms | 0 |
| Reason (LLM) | 1.5–7s | 1.5–7s | 0 |
| **Total p50** | **~2.6s** | **~2.8s** worst-case, likely ~+100ms in practice | |

Perf gate: per-query p50 regression must not exceed +300ms on the eval bank. If it does, the spec fails and we revisit.

## Components and interfaces

Four modules change. One new script. One file deleted.

### 1. `src/generation/reasoner.py` — grounding check fix

**Diagnose first, then fix.** Add a diagnostic that runs `_check_grounding` against the 13 fixtures from `tests/quality_audit_results.json` and prints which gate (number-overlap at 20% or word-overlap at 15%) returns False for each. Likely culprit is the 20% ungrounded-numbers gate when answers include percentages or section numbers not matched verbatim in evidence — but we do not edit until diagnostics identify the true failing gate.

**Public surface unchanged:** `Reasoner.reason()` still returns `ReasonerResult(grounded: bool, ...)`. Callers untouched.

**Tests:** `tests/generation/test_grounding_check.py` (NEW) locks 13 fixture cases plus 3 known-ungrounded synthetic cases.

### 2. `src/retrieval/retriever.py::UnifiedRetriever` — hybrid + KG expansion

**Signature change (additive):**

```python
def retrieve(
    self,
    query: str,
    subscription_id: str,
    profile_ids: List[str],
    *,
    document_ids: Optional[List[str]] = None,
    top_k: int = 50,
    correlation_id: Optional[str] = None,
    query_entities: Optional[List[str]] = None,          # NEW
    query_understanding: Optional[dict] = None,          # NEW (future-proofing)
) -> RetrievalResult:
```

**Internal flow (replaces dense + keyword-fallback):**

```python
dense_hits = self._dense_search(query, ...)
sparse_hits = self._sparse_search(query, ...)
fused_ids = reciprocal_rank_fusion(
    {"dense": [c.chunk_id for c in dense_hits],
     "sparse": [c.chunk_id for c in sparse_hits]},
    weights={"dense": 0.6, "sparse": 0.4},
)
kg_chunk_ids = self._kg_expand(query_entities, ...)
candidate_ids = merge_preserving_rank(fused_ids, kg_chunk_ids)
return self._hydrate(candidate_ids[:top_k])
```

**Graceful degradation:**
- `SparseEncoder` absent or raises → warning, dense-only for that query.
- Neo4j driver absent, query raises, or circuit-breaker open → warning, empty KG expansion.
- Chunks with unpopulated sparse slot (pre-backfill) contribute 0 to sparse score but still rank on dense — Qdrant handles natively.

**KG expansion caps:** top-5 query entities, top-10 related chunks per entity. Hard-coded defaults; configurable only if needed.

**Tests:** `tests/retrieval/test_hybrid_retrieve.py` (NEW), `tests/retrieval/test_kg_expansion.py` (NEW).

### 3. `src/agent/core_agent.py` — forward entities

Six call sites: four direct `self._retriever.retrieve(...)` calls and two kwargs-based calls via a `_prefetch_kwargs` dict (lines ~247, ~425, ~435, ~796, ~852, ~856 as of 2026-04-17). Each direct call adds `query_entities=understanding.entities`; each `_prefetch_kwargs` dict gets the same key added. Because `query_entities` is a keyword-only parameter with `None` default on the retriever, call sites that don't have access to `understanding` can omit it — so non-UNDERSTAND paths (prefetch warmup) pass `None` with no behavior change.

### 4. `src/embedding/pipeline/qdrant_ingestion.py` — populate sparse going forward

One-line change at the `sparse_vector=None` site. Encode chunk text with `app_state.sparse_encoder`; use `sparse_to_qdrant()` for the shape. Existing ingestion tests updated so the mock Qdrant client expects a non-None sparse vector.

### 5. `src/api/app_lifespan.py` — one-line init

Add `app_state.sparse_encoder = SparseEncoder()` to the existing parallel init block alongside `embedding_model` and `reranker`. Same `ThreadPoolExecutor` pattern; no new machinery.

### 6. `scripts/backfill_sparse_vectors.py` — one-shot (NEW)

Standalone, no API integration. Arguments:

```
--subscription-id X    # single collection
--all                  # every collection the API can see
--batch-size 64        # SPLADE batch on GPU
--inventory-only       # size check, no writes
--dry-run              # encode but don't upsert (verifies GPU path)
--resume               # skip chunks already marked sparse_backfilled_at
--concurrency 1        # hard-cap while vLLM is running
```

**Flow per collection:**

1. `client.scroll(collection, with_payload=True, with_vectors=False, limit=batch_size, offset=cursor)`.
2. Skip chunks with `payload["sparse_backfilled_at"]` set (resumable, idempotent).
3. Extract `canonical_text` (fall back to `text`).
4. `SparseEncoder.encode_batch(texts)` → list of `{indices, values}` dicts.
5. `client.upsert_points` with `PointStruct(id=chunk_id, vector={"keywords_vector": SparseVector(...)})`, and `set_payload({"sparse_backfilled_at": now_iso()})`.
6. Progress logged per N batches.

**Data safety:** writes only the sparse vector and one payload field. Dense vectors and existing payload untouched. Idempotent — re-running is safe.

**GPU pressure:** SPLADE-v3 (~140M params) with batch=64 fits in the ~6GB headroom on the A100 while vLLM holds the model resident. Monitor KV-cache usage; drop to `--batch-size 16` if vLLM begins evicting. Run during low-traffic window if needed.

**Tests:** `tests/scripts/test_backfill_sparse.py` (NEW) — dry-run path against a mock Qdrant client, resumability, idempotence.

### 7. Deletion — `src/retrieval/unified_retriever.py`

Confirmed unimported by any live code. The only importer is `tests/test_retrieval_integration.py` (8 import sites), which is deleted alongside the module in this spec — those tests only exercise the dead 3-layer retriever and have no live-path coverage value. Prevents future readers from confusing the dead module with the live `retriever.py::UnifiedRetriever`.

## Rollout

Sequential by design so metric movement is attributable.

**Step 0 — Fresh baseline.** Run `python scripts/intensive_test.py` with the API up on port 8000 and vLLM healthy on 8100. This writes `tests/ragas_metrics.json` by invoking `ragas_evaluator.evaluate()` inline. Copy it aside: `cp tests/ragas_metrics.json tests/ragas_metrics.baseline.json`. Commit the baseline so every subsequent step diffs against the same numbers. The Apr 11 metrics predate the model symlink repoint to the HF-recovered weights and should not be treated as baseline.

**Step 1 — Grounding check fix (§1).** Ship, re-run eval. Expect the `grounded` field to start reflecting truth. No recall movement expected.

**Step 2 — `SparseEncoder` init + ingestion update + backfill (§4, §5, §6).** Ship code; kick off backfill. Serving continues on the dense-only path during backfill. Wait for completion.

**Step 3 — Hybrid retrieval wiring (§2 dense+sparse+RRF).** Flip retriever to dense+sparse+RRF. Re-run eval. **Recall should jump.** If it does not, investigate before layering on KG expansion.

**Step 4 — KG entity expansion (§2 KG piece + §3 core_agent wire).** Wire entities through. Re-run eval. Cross-document recall should improve.

**Step 5 — Delete dead code (§7).**

**Step 6 — Gate check.** If `faith ≥0.80 AND recall ≥0.75 AND hallucination ≤0.05 AND latency_p50_delta ≤+300ms`: ship, proceed to next workstream. If not: diagnose failing cluster, decide whether to extend this spec or mark retrieval as done and escalate to the extraction workstream.

**Rollback posture:** no feature flag. Each step is one commit; regressions → `git revert`. The backfill is the only persistent-state mutation, and its only action is populating a pre-provisioned slot — rollback is "stop reading the sparse vector," a single conditional.

## Testing

### Unit tests (NEW)

- `tests/generation/test_grounding_check.py` — 13 audit fixtures + 3 synthetic ungrounded.
- `tests/retrieval/test_hybrid_retrieve.py` — mock Qdrant with known dense+sparse rankings; verify RRF output order and dense-only degradation.
- `tests/retrieval/test_kg_expansion.py` — mock Neo4j driver; verify 1-hop expansion, empty-result silent failure, driver-absent path.
- `tests/scripts/test_backfill_sparse.py` — dry-run, resumability, idempotence.
- `tests/embedding/test_sparse_encoder.py` — create if missing; lock `encode` output shape and `sparse_to_qdrant` conversion.

### Integration test (NEW)

- `tests/intelligence_v2/test_hybrid_e2e.py` — real Qdrant collection seeded with ~20 chunks (dense + sparse populated), real `SparseEncoder` on CPU, real BGE-large, real cross-encoder, stubbed LLM. Three query fixtures exercising dense-biased, sparse-biased, and KG-expansion-biased retrieval.

### Eval gate (authoritative)

`scripts/intensive_test.py` runs the query bank (hardcoded profile IDs for HR resumes + legal contracts) against `/api/ask` at `BASE_URL=http://localhost:8000`, writes `/tmp/intensive_test_results.json`, then invokes `ragas_evaluator.evaluate()` inline and writes `tests/ragas_metrics.json`.

Workflow per eval run:

1. Ensure the API is serving on port 8000 and vLLM is healthy on port 8100.
2. `python scripts/intensive_test.py` (no args).
3. Copy `tests/ragas_metrics.json` to a stable filename for diffing (baseline vs post).
4. Inspect `/tmp/intensive_test_results.json` per-query when a metric regresses.

Baseline capture (Step 0 of rollout): `cp tests/ragas_metrics.json tests/ragas_metrics.baseline.json` before any code change. Each subsequent step produces a new `tests/ragas_metrics.json` that is diffed against the baseline.

**Pass criteria (at Step 6):**

- `answer_faithfulness ≥ 0.80`
- `context_recall ≥ 0.75`
- `hallucination_rate ≤ 0.05`
- `grounding_bypass_rate ≤ 0.02`
- Per-query p50 latency delta ≤ +300ms vs baseline

## Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Both grounding gates are broken, not just one. | Medium | Diagnose-first discipline; do not ship a fix until the failing gate is identified per fixture. |
| SPLADE backfill contends with vLLM for GPU; serving latency regresses. | Medium | Start at `--batch-size 16`; watch GPU KV-cache; back off or schedule for low-traffic window. |
| Wider candidate pool overwhelms the reranker. | Low | Rerank already pre-filters to top-10 by dense+keyword before CE. Wider pool is net positive. |
| KG expansion pulls noise (common entity matches hundreds of chunks). | Medium | Hard caps: top-5 entities × top-10 chunks each. |
| 106-query bank is HR-heavy; gate clears but real-world queries still struggle. | Medium | Acknowledged. Gate-passing here does not close the RAG story; it closes this spec. Real-world telemetry informs the next workstream. |
| Gate clears but gains come entirely from backfill+sparse; KG expansion added no measurable value. | Low | Ship anyway — KG expansion is cheap (~80ms parallel) with upside on cross-document queries under-represented by the current bank. |
| Extraction is the true ceiling; no retrieval fix clears the gate. | Medium | Valid spec outcome: conclude "retrieval is as good as it can be" and escalate to the extraction workstream with diagnostics proving retrieval isn't the bottleneck. |

## Success criteria (restated)

This spec is done when all of the following hold on `main`:

1. `/tmp/ragas_post.json` shows `faith ≥0.80 AND recall ≥0.75 AND hallucination ≤0.05 AND bypass ≤0.02`.
2. Per-query p50 latency delta ≤ +300ms.
3. All new unit and integration tests pass.
4. `grounded: false` in logs means the reasoner could not trace the answer to evidence (not "always false").
5. `src/retrieval/unified_retriever.py` is deleted.
6. Ingestion populates `keywords_vector` for all new chunks.
7. A short retrospective note captures which step moved which metric.

Outcome 7 feeds the scoping of the next workstream (extraction or model training).
