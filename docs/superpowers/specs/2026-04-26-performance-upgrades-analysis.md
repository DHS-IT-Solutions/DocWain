# DocWain Performance Upgrades — In-depth Analysis

**Date:** 2026-04-26
**Branch:** `preprod_v03` (live)
**Scope:** Quantified bottlenecks from live measurements + upgrade options ranked by impact × effort.

---

## Live measurements (this session)

| Component | Measurement | Value |
|---|---|---|
| `/api/ask` end-to-end (non-stream) | wall time, profile with insights | 10–15 s |
| `/api/ask` understand step | LLM call 1 (intent + entities) | **4.7 s** |
| `/api/ask` reason step | LLM call 2 (final answer) | **4.1 s** |
| `/api/ask` retrieve step | Qdrant semantic search | 0–200 ms |
| Researcher v2 per insight type | single LLM call after warm-up | **3.5 s** (cold: 13 s) |
| Researcher v2 per document, 9 types | sequential | ~30–40 s |
| Mongo `insights_index` find | p50 / p95 | 114 / 119 ms |
| Qdrant scroll (filtered) | p50 / p95 | 156 / 476 ms |
| Insight injection on /api/ask | added overhead | ~120 ms |
| GPU utilization (idle) | A100 80 GB single | 0% util, 90% mem reserved by vLLM |
| vLLM prefix cache hit rate | last ~2.6M token queries | **33%** (882K hits / 2.6M queries) |
| Celery worker concurrency | per worker | 4 across 10 queues |

**Where the time goes** in a representative `/api/ask` call:
- 8.8 s LLM (97%)
- 0.12 s Mongo lookup for insight injection (1%)
- 0.20 s Qdrant retrieve (2%)
- everything else: <50 ms

**Conclusion:** The dominant cost is LLM inference. Optimising anything else only matters if you've already moved the LLM number — which is where 80%+ of the achievable gain lives.

---

## Tier 1 — High impact, low-to-moderate effort (do these first)

### 1.1 Parallelise researcher v2 insight-type calls
**Today:** `run_researcher_v2_for_doc` loops over enabled insight types sequentially. 9 types × 3.5 s ≈ 30 s.
**Upgrade:** Submit all 9 LLM calls concurrently (`asyncio` or `ThreadPoolExecutor(8)`). vLLM batches them internally — total time = 1 × LLM call ≈ 4 s.
**Expected gain:** **~7×** on per-doc researcher v2 wall time. Total researcher v2 across 9 types drops from ~30 s to ~4 s.
**Risk:** rate-limit pressure on vLLM-fast — but with one user-facing GPU it can absorb 9-way batch easily. Add a `concurrent_per_doc=4` cap as a safety knob.
**Effort:** half-day. Touches `src/intelligence/researcher_v2/runner.py` only.
**Single-flag revertible:** add `INSIGHTS_RESEARCHER_PARALLEL_ENABLED`.

### 1.2 Stabilise vLLM prefix cache for higher hit rate
**Today:** 33% prefix cache hit rate. Researcher v2 builds a fresh prompt per call with the document text inline; system prompt embeds dynamic guidance per insight type.
**Upgrade:**
- Make the researcher system prompt **identical** across insight types (move per-type guidance to the user message).
- Same trick for `/api/ask` understand and reason: stable system prompts; variable parts at end.
- vLLM auto-caches stable prefixes, so identical leading tokens hit the cache.
**Expected gain:** prefix cache hit rate to **70–85%**, time-to-first-token down ~30–50% on cache hits. For researcher v2 at scale (lots of docs through the same prompts), this compounds with #1.1.
**Risk:** quality regression if system prompt becomes too generic. Mitigate by keeping the user-message prefix stable too.
**Effort:** 2–3 hours. Touches `src/docwain/prompts/researcher_v2_generic.py` and `src/generation/prompts.py`.
**Single-flag revertible:** N/A — pure refactor; keep both prompt builders behind a flag during A/B if cautious.

### 1.3 Wire real embeddings into Qdrant insight writes
**Today:** `insights/store.py` writes `vector = [0.0] * 384` placeholder. Semantic search on insights silently doesn't work.
**Upgrade:** Inject the existing embedder into `QdrantInsightBackend(embedder=...)` in `insights_wiring.py`. The embedding model is already loaded in the app (used for chunks).
**Expected gain:** insight semantic search becomes functional. Enables a future "find similar insights across profiles" surface, and lets `/api/ask` injection match by query embedding instead of just tags. **0 latency cost** (writes are in the researcher v2 task on a background queue).
**Effort:** 1 hour.
**Single-flag revertible:** N/A — fixes a silent gap; behind `INSIGHTS_TYPE_*_ENABLED` already.

### 1.4 Bulk write to Mongo for researcher v2
**Today:** Each insight write = 1 `update_one` call = 1 RTT to CosmosDB (~120 ms). 5 insights × 120 ms = 600 ms per doc just on Mongo.
**Upgrade:** `bulk_write([UpdateOne(...) for each insight])` — single RTT.
**Expected gain:** **5–10× faster Mongo write phase**, ~500 ms saved per doc.
**Effort:** 1 hour. Add a `bulk_upsert` method on `MongoIndexBackend`.
**Single-flag revertible:** N/A — strict speedup, same semantics.

### 1.5 Streaming default for `/api/ask`
**Today:** `stream=False` is the default; the client waits the full 10 s before any byte arrives.
**Upgrade:** Default `stream=True` for `/api/ask`. First token in 500–800 ms; "perceived" latency drops dramatically even though wall time is unchanged.
**Expected gain:** user-perceived `/api/ask` latency from 10 s → <1 s to first token. Same total compute.
**Effort:** the streaming path already exists in `src/main.py`. Need to flip default and verify the proactive injection works in the streaming path (it currently only fires on non-stream).
**Single-flag revertible:** `INSIGHTS_PROACTIVE_INJECTION_STREAMING` — when off, streaming skips injection.

### 1.6 Skip the "understand" LLM call for simple queries
**Today:** Every `/api/ask` invokes a 4.7 s "understand" pass (intent classification + entity extraction). Many queries don't need it.
**Upgrade:**
- Lightweight regex/keyword classifier first (intent, entities) — runs in <5 ms.
- Only call the LLM understand step when classifier confidence is low.
**Expected gain:** ~50% of /api/ask calls skip the 4.7 s step → average wall time drops from 10 s to 5–6 s.
**Risk:** regression on hard queries. Gate via `confidence_threshold` and fall back to LLM understand when uncertain.
**Effort:** 1 day. Touches `src/intelligence/lightweight_intent.py` (already exists) + `src/intelligence/ask_pipeline.py`.
**Single-flag revertible:** `ASK_FAST_INTENT_ENABLED`.

---

## Tier 2 — High impact, higher effort

### 2.1 Speculative decoding on vLLM
**Today:** vLLM 14B alone, no draft model. Token generation latency ~30–40 ms/token.
**Upgrade:** Configure vLLM with a 1.5 B draft model + 14 B verifier (e.g. Qwen3-1.5B or any architecture-compatible small model). vLLM v0.6+ supports `--speculative-config`.
**Expected gain:** **2–3× tokens/sec** on generation-heavy paths (the reason step, researcher emissions). A 4 s reason call becomes 1.5–2 s.
**Risk:** draft model selection matters; mismatched tokenizer = no speedup. Requires GPU memory budget for the draft model (~3 GB extra). A100 80 GB has headroom.
**Effort:** 2–3 days (model selection, vLLM config, evaluation). Pre-requisite: identify a draft model with the same tokenizer as docwain-fast.
**Single-flag revertible:** vLLM config — restart with old config flips it off.

### 2.2 Move chat-side reasoning to a smaller model when context is small
**Today:** `/api/ask` uses 14 B for everything. For short queries with small evidence (<1 K tokens), a 7 B model would be 2× faster with negligible quality loss.
**Upgrade:** Add a `routing` layer: if `evidence_tokens < 1500 AND query_complexity_score < 0.6`, route to a smaller model. Otherwise use 14 B.
**Expected gain:** **~30–40% of `/api/ask` calls** drop to 5–6 s wall time.
**Risk:** quality regression on borderline queries. Need an eval harness to score.
**Effort:** 3–5 days (model serving, routing logic, eval).
**Single-flag revertible:** `ASK_SMALL_MODEL_ROUTING_ENABLED`.

### 2.3 Insight cache layer (Redis read-through)
**Today:** Every `/api/ask` does a fresh Mongo lookup for insights (~120 ms). Insights change rarely (only when researcher v2 runs).
**Upgrade:** Redis cache keyed by `profile_id`, TTL = 5 min, invalidated on researcher v2 write. The existing `redis_intel_cache.py` already has the pattern — extend for insights.
**Expected gain:** insight injection cost from ~120 ms to ~5 ms = 24× faster on the hot path. Total `/api/ask` saves ~115 ms (1% of wall time but free).
**Effort:** 1 day. Touches `insights/store.py` + `insights_wiring.py`.
**Single-flag revertible:** `INSIGHTS_REDIS_CACHE_ENABLED`.

### 2.4 Increase Celery worker concurrency + per-queue concurrency
**Today:** 1 worker × 4 processes across 10 queues. Researcher v2 + extraction + embedding can starve each other.
**Upgrade:**
- Run multiple Celery workers, each pinned to a queue family:
  - `worker-extract` on `extraction_queue, screening_queue` (concurrency=2)
  - `worker-train` on `embedding_queue, kg_queue, researcher_v2_queue, profile_intelligence_queue` (concurrency=4)
  - `worker-refresh` on `researcher_refresh_queue, actions_queue, backfill_queue` (concurrency=2)
- Total = 8 processes vs 4 today, with isolation guarantees.
**Expected gain:** burst throughput **2×**; one slow extraction won't block researcher v2 dispatch.
**Risk:** more memory pressure. With 80GB+ RAM headroom this is fine.
**Effort:** 1 day. Two new systemd unit files; no code change.
**Single-flag revertible:** revert systemd files, restart.

### 2.5 Document-content-hash cache for researcher v2
**Today:** If the same document is re-uploaded (different document_id, same bytes), researcher v2 runs again from scratch.
**Upgrade:** Hash the canonical extracted text. If a previous insight set exists for that hash, copy + retag with new document_id. **No LLM calls.**
**Expected gain:** for any duplicate uploads (real-world common — re-extracting after fixing a metadata issue), researcher v2 cost drops to ~0.
**Effort:** 1 day. New Mongo collection `insights_by_content_hash`; lookup before LLM call.
**Single-flag revertible:** `INSIGHTS_CONTENT_HASH_CACHE_ENABLED`.

---

## Tier 3 — Lower impact OR higher risk (consider after Tier 1+2)

### 3.1 Quantize vLLM model to INT4 (AWQ/GPTQ)
**Today:** docwain-fast is bfloat16. ~74 GB GPU memory for the 14 B model.
**Upgrade:** Quantize to INT4 (AWQ-INT4 with vLLM). Memory footprint drops to ~16 GB. Frees ~58 GB for higher KV cache → larger batches → 2–3× throughput on concurrent requests.
**Expected gain:** **2–3× throughput** under concurrent load. Single-request latency similar (potentially slightly faster from larger batches).
**Risk:** quality regression. AWQ is generally <1% MMLU drop on Qwen-class models, but DocWain has been fine-tuned — needs a quality eval before/after.
**Effort:** 2–3 days (quantization, eval, deployment). Reusable: same artifact for all instances.
**Single-flag revertible:** swap `models/docwain-v2-active` symlink.

### 3.2 Tensor parallelism if more GPUs added
**Today:** `--tensor-parallel-size 1`. Single A100 80 GB. Memory headroom for larger model or higher KV cache.
**Upgrade:** Add a second GPU and run `tensor-parallel-size 2`. Latency per token drops ~40%.
**Expected gain:** **~40% lower per-token latency** on generation-heavy paths.
**Risk:** none code-wise; capex for additional GPU.
**Effort:** infra-bound, not code.

### 3.3 Replace `lightweight_intent` with a small fine-tuned classifier
**Today:** The "understand" step is a full LLM call.
**Upgrade:** Fine-tune a 100 M parameter classifier (DistilBERT-class) on 5 K query→intent pairs. Sub-50ms classification.
**Expected gain:** removes the 4.7 s understand step entirely on classified queries → /api/ask wall time → 5 s.
**Risk:** training data discipline (synthetic only per `feedback_no_customer_data_training.md`); evaluation harness needed.
**Effort:** 1–2 weeks. Crosses into model training territory — gates per `feedback_engineering_first_model_last.md`. Run the engineering options (1.6) first.

### 3.4 Qdrant payload index optimization
**Today:** Qdrant scroll with profile_id filter has p95=476 ms (high variance).
**Upgrade:** Ensure payload index on `profile_id` field for the `insights` collection (the existing `qdrant_indexes.py` bootstraps some indexes — confirm coverage). Add HNSW tuning: `ef_search=64` for higher recall on small collections.
**Expected gain:** Qdrant insight scroll p95 from 476 ms → ~100 ms.
**Risk:** none.
**Effort:** 1 hour. Add to `_ensure_insights_qdrant_collection` in `insights_wiring.py`.
**Single-flag revertible:** N/A.

### 3.5 Compression / vector quantization in Qdrant
**Today:** 384-dim float32 vectors, no quantization.
**Upgrade:** Enable scalar quantization (8-bit) on `insights` collection. 4× smaller on disk + RAM.
**Expected gain:** lower memory footprint as the insights set grows; small recall drop (<1%).
**Risk:** recall regression measurable but small at this scale.
**Effort:** 1 hour. Set on collection create.

### 3.6 Drop ignored Celery results
**Today:** All Celery tasks store results in Redis backend. Most callers `delay()` and never `get()`.
**Upgrade:** `@app.task(ignore_result=True)` on fire-and-forget tasks (researcher_v2 dispatch, refresh).
**Expected gain:** reduces Redis write load; small but measurable under burst.
**Effort:** 1 hour.

### 3.7 Lazy-import in app startup
**Today:** App takes 60–90 s to start (imports torch, transformers, sentence_transformers, vLLM client, etc. eagerly). During restart, the API is down.
**Upgrade:** Defer imports of the embedder / classifier until first use; keep the FastAPI router registration eager. App health responds in <5 s.
**Expected gain:** rolling restarts go from 60–90 s downtime to <10 s.
**Effort:** 1–2 days. Touches multiple `src/api/*.py` modules.
**Single-flag revertible:** N/A — pure refactor.

### 3.8 Move Mongo closer or use a regional read replica
**Today:** CosmosDB cluster is geographically remote → ~120 ms RTT. The Tier 2.3 Redis cache absorbs read latency, but writes remain blocked on it.
**Upgrade:** Either a regional CosmosDB read replica (writes still go to primary, reads local), or migrate to a Mongo deployment in the same region as the app.
**Expected gain:** Mongo find p50 from 114 ms → 5 ms; researcher v2 write phase from 600 ms → 25 ms.
**Risk:** infra change. Multi-region replication has its own consistency considerations.
**Effort:** infra-bound.

---

## Tier 4 — Hygiene wins (low effort, low individual impact, but they add up)

| # | Upgrade | Win | Effort |
|---|---|---|---|
| 4.1 | Connection pool tuning (pymongo `maxPoolSize=100, minPoolSize=10`) | smoother under burst | 30 min |
| 4.2 | Mongo `_id`-suppression on every find — already partially done; sweep all hooks | clean payloads, no surprise serialization errors | 1 hour |
| 4.3 | Add structured timing logs (`stage_ms`) to researcher v2, refresh, /api/ask | observability for the next perf pass | 2 hours |
| 4.4 | Reduce `task_time_limit` from 1800 s to 600 s for researcher v2 | catches stuck LLM calls earlier | 15 min |
| 4.5 | Cache adapter YAMLs in process memory beyond TTL when Blob is unreachable | already implemented; bump TTL to 30 min from 5 | 5 min |
| 4.6 | Truncate document_text in researcher v2 to 8 K chars (currently 16 K) | 50% fewer prompt tokens → 30% faster | 5 min |
| 4.7 | Stop wasting first cold-call latency — warm vLLM at app startup with a dummy prompt | 13 s cold → 3.5 s warm on the first researcher run after restart | 30 min |

---

## Recommended sequencing

**Week 1 (this week, low-risk wins):**
1. 1.4 Bulk Mongo write — instant win, no flag needed
2. 1.3 Wire real embeddings into Qdrant — fixes silent gap
3. 1.1 Parallelise researcher v2 insight types — 7× win on background work
4. 1.2 Stabilise prefix cache prompts — compounds with 1.1
5. 4.6, 4.7 — quick hygiene

**Week 2 (medium-risk, higher impact):**
6. 1.5 Streaming default — biggest user-perceived win
7. 1.6 Fast intent path — removes 4.7 s on simple queries
8. 2.4 Multi-worker Celery — throughput
9. 2.3 Redis insight cache — closes the Mongo network bottleneck for hot reads

**Week 3+ (model-layer):**
10. 2.1 Speculative decoding — 2–3× generation speed
11. 3.1 INT4 quantization — 2–3× concurrent throughput
12. 2.2 Smart routing to smaller model — wall-time win on most queries

---

## Composite expected gains (after Tier 1+2)

If 1.1 + 1.2 + 1.4 + 1.5 + 1.6 + 2.3 + 2.4 are shipped:
- **Researcher v2 per doc:** 30 s → ~5 s (**6×**)
- **`/api/ask` user-perceived latency:** 10 s → <1 s to first token (streaming)
- **`/api/ask` wall time on simple queries:** 10 s → 5–6 s (**~2×**)
- **Insight injection cost:** 120 ms → ~5 ms (**24×**)
- **Burst throughput on Celery:** 2×

If you also ship 2.1 + 2.2 + 3.1 (model layer):
- **`/api/ask` wall time on hard queries:** 10 s → 4–5 s (**2–2.5×**)
- **Concurrent throughput on vLLM:** 3×

---

## What I would NOT recommend without a measurement gate

Per `feedback_measure_before_change.md`, two upgrades that look attractive but should not ship without baselines:
- **3.1 INT4 quantization** — needs a quality regression eval first. Run on the SME eval set (`tests/sme_evalset_v1`) before/after.
- **3.3 Fine-tuned intent classifier** — model training; gated by `feedback_engineering_first_model_last.md`. Run engineering options (1.6) first to see if they suffice.

Both are deferred to the model-layer pass after the engineering layer is squeezed dry.

---

## Risks I'm flagging that don't show in the perf numbers

1. **Single A100 GPU is a SPOF.** vLLM is the bottleneck and the single point of failure. If the GPU dies, the whole product dies. **Recommendation:** at minimum, keep Ollama Cloud as the documented failover (already wired per `project_post_preprod_roadmap.md` item 5; verify the failover path actually works under simulated outage).

2. **Researcher v2 has no rate limiter.** Under a backfill burst (Tier 2.5), it could starve `/api/ask`. **Recommendation:** add a token-bucket on `researcher_v2_queue` so user-facing traffic always has GPU headroom.

3. **Prefix cache invalidation on prompt change.** When we ship 1.2 (prompt stabilisation), the existing 33% hit rate temporarily drops to 0% on the rollout. Expected — not a regression, but worth knowing before measuring "did the change help?" — give it 1 hour of warm traffic before judging.
