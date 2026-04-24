# Unified DocWain Engineering Layer — Design

**Date:** 2026-04-24
**Branch:** `preprod_v02`
**Status:** Approved for implementation (Option A + identity shim path)
**Owner:** Muthu

## 1. Purpose

Deliver a unified DocWain — one model (V2 on vLLM) handling every capability the product requires — via **engineering alone, no new training**. Training remains the long-term plan (see `2026-04-24-docwain-unified-model-analysis.md`), but that path takes 6–12 months. This spec ships the unified experience in 2–3 weeks using prompt engineering, task-specific scaffolding, domain adapters, tool calls, and a Researcher Agent.

## 2. Scope — six components + identity shim

1. **Gateway routed to vLLM local primary.** Cloud 397B becomes emergency fallback only. User directive from 2026-04-23.
2. **Gateway identity shim.** Inject a short `You are DocWain…` system-prompt prefix in the gateway layer so identity is consistent without weight changes (V5 lesson 7). Removable later when identity is trained into weights.
3. **Entity + relation extraction prompt and wiring.** DocWain reads extracted text and emits structured entity/relationship JSON. The KG adapter (`src/tasks/kg.py`) calls this before building `GraphIngestPayload`, closing the gap Plan 3 deferred.
4. **Chart / DOCWAIN_VIZ generation prompt.** System-prompt contract + in-context examples so DocWain emits the JSON viz payloads the frontend renders when queries warrant a chart/comparison.
5. **Domain adapter framework.** Per `feedback_adapter_yaml_blob.md`: loader pulls per-domain YAMLs from Azure Blob, caches with TTL, merges adapter fragments into the active task prompt. Seed adapters: generic, finance, legal, hr, medical, it_support.
6. **Researcher Agent.** Runs in the training stage in parallel with embedding + KG (per `project_researcher_agent_vision.md`). Takes extracted content, emits domain-aware insights via DocWain, writes to Qdrant payload (by document_id) and Neo4j (as insight-typed nodes/edges). Incremental update on new docs; weekend full-set refresh.

## 3. Non-goals

- **No training runs in this workstream.** Zero SFT, DPO, or LoRA adapter training. Pure engineering.
- **No new MongoDB status values.** Existing `pipeline_status`, `stages.*`, `knowledge_graph.status` strands preserved.
- **No changes to HITL gates.** Pipeline flow unchanged.
- **No Cloud 397B removal.** Cloud stays in `src/llm/gateway.py` as fallback; only the primary selection flips.
- **No new query-time compute.** All intelligence precomputed at ingestion (Researcher Agent runs during training stage, not at query time). Consistent with `feedback_intelligence_precompute.md`.
- **No query-time LLM calls beyond the existing Reasoner.** Engineering layer adds prompts and adapters, not new per-query model calls.

## 4. Architecture

```
UPLOAD → EXTRACTION (Plan 1 native / Plan 2 vision) →
         HITL screening → HITL training-approval →
         training-stage parallel tracks:
           ├─ embed_document (Qdrant)
           ├─ build_knowledge_graph (Neo4j)                     ← now calls Entity Extractor before ingest
           └─ run_researcher_agent (Qdrant insights + Neo4j)    ← NEW in this workstream
```

Every task-specific prompt lives in `src/docwain/prompts/` (new package). Each prompt is a Python module exporting:
- A system-prompt string
- A function that builds the user-prompt payload from task inputs
- A response parser + schema validator

Every DocWain model call route:
```
caller → (optional) domain adapter injection →
      task prompt → gateway (vLLM primary → Cloud fallback) →
      DocWain V2 → parsed structured output → caller
```

Identity shim lives at the gateway layer, prepended to every outgoing system prompt. Removable via config flag when training-time identity lands.

## 5. Component details

### 5.1 Gateway routed to vLLM local primary (+ identity shim)

**File:** `src/llm/gateway.py`

- `_init_clients()` re-ordered: primary = `OpenAICompatibleClient` pointing at `Config.VLLM.VLLM_ENDPOINT`; secondary = `OllamaClient` (Cloud 397B); tertiary = `OpenAIClient` (Azure GPT-4o).
- Identity shim string (~100 chars) is prepended to the `system` parameter on every call that goes through `generate_with_metadata` / `generate`. Controlled by a config flag: `Config.Model.IDENTITY_SHIM_ENABLED` (default true).
- Feature flag: `Config.Model.PRIMARY_BACKEND` = `"vllm"` (default) | `"cloud"` (current behavior) | `"azure"`. Flipping back to `"cloud"` restores today's routing.

**Revert:** `export DOCWAIN_MODEL_PRIMARY_BACKEND=cloud` + restart docwain-app. Or env-var default change.

### 5.2 Central prompt registry (lightweight — Wave 2 if time permits)

Not blocking Wave 1. Existing prompts stay where they are; new prompts land in `src/docwain/prompts/` and are referenced by the few places they're used. Full registry refactor is a cleanup task after all prompts are written.

### 5.3 Entity + relation extraction (Wave 1)

**Files:**
- Create `src/docwain/prompts/__init__.py`
- Create `src/docwain/prompts/entity_extraction.py` — system prompt + parse helpers
- Modify `src/tasks/kg.py` — `_canonical_to_graph_payload` optionally enriches payload.entities from a DocWain call before returning

**Behavior:**
- System prompt instructs DocWain to return `{"entities": [{"text", "type", "confidence"}], "relationships": [{"source", "target", "type"}]}` given the extracted text.
- `_canonical_to_graph_payload` takes a new kwarg `extract_entities: bool = True`. When true, it calls DocWain (via gateway) on concatenated page text, parses response, populates `payload.entities` and `payload.typed_relationships`. When false, behaves as today (empty entities).
- Feature flag: `Config.KG.ENTITY_EXTRACTION_ENABLED` (default true). When false, entity extraction is skipped — current Plan 3 behavior preserved.
- Failure: if DocWain call fails or returns malformed JSON, the helper logs the error and falls back to empty entities. KG ingestion completes with just Document + chunk mentions (same as today). Never raises.

**Revert:** `export DOCWAIN_KG_ENTITY_EXTRACTION_ENABLED=false` + restart docwain-celery-worker.

### 5.4 Chart / DOCWAIN_VIZ generation (Wave 2)

Deferred. The Reasoner already emits `<!--DOCWAIN_VIZ ... -->` comments from its system prompt (seen in Plan 2 audit outputs). The model does this today with some reliability. A targeted improvement pass lands in Wave 2.

### 5.5 Domain adapter framework (Wave 2)

Deferred. Per `feedback_adapter_yaml_blob.md`: loader from Azure Blob, TTL cache, subscription override, hot-swap. Substantial work; Wave 2 builds the loader + seed adapters.

### 5.6 Researcher Agent (Wave 1)

**Files:**
- Create `src/docwain/prompts/researcher.py` — system prompt + output schema
- Create `src/tasks/researcher.py` — Celery task `run_researcher_agent` on `researcher_queue`
- Modify `src/celery_app.py` — register `researcher_queue` + routing
- Modify `src/api/pipeline_api.py::trigger_embedding` — dispatch `run_researcher_agent.delay(...)` alongside embedding and KG
- Mongo: new status strand `researcher.status` with values `RESEARCHER_PENDING`, `RESEARCHER_IN_PROGRESS`, `RESEARCHER_COMPLETED`, `RESEARCHER_FAILED`. Added to `src/api/statuses.py`. Independent from `pipeline_status` — same isolation contract as KG (Plan 3).
- Output: per-document `insights` dict persisted to:
  - **Qdrant**: enriched chunk payload with `insights` sub-object (summary, key facts, recommendations, anomalies, questions-to-ask).
  - **Neo4j**: `Insight` node type linked to `Document`, one per insight; relation `Document -[:HAS_INSIGHT]-> Insight`.
- Domain-aware: adapter framework (Wave 2) will inject domain-specific prompt fragments here. Wave 1 ships with the generic adapter inline (baked into the system prompt).
- Refresh cadence: incremental on HITL approval (above). Weekend full-set refresh implemented as a scheduled Celery beat task `researcher_refresh_weekly` — Wave 2 scope.

**Feature flag:** `Config.Researcher.ENABLED` (default true). When false, `trigger_embedding` does NOT dispatch Researcher; nothing changes vs Plan 3.

**Revert:** `export DOCWAIN_RESEARCHER_ENABLED=false` + restart docwain-app + docwain-celery-worker.

**Isolation:** Researcher task writes ONLY to `researcher.*` in MongoDB. Never touches `pipeline_status`, `knowledge_graph.*`, or `stages.*`. Enforced by the same test pattern as Plan 3's `test_kg_task_does_not_touch_pipeline_status`.

## 6. Zero-error discipline

Per `feedback_intelligence_rag_zero_error.md`:

- **Each Wave 1 task ships as its own commit** (not bundled). Rollback = `git revert <sha>`.
- **Each task has mechanically verifiable exit criteria** (unit test + integration test + smoke). No "looks right" exit.
- **Each task adds a feature flag** with default-on behavior but operator can flip off without redeploying code.
- **Document-processing pipeline preservation**: no changes to `src/tasks/extraction.py`, `src/tasks/embedding.py`, or existing extraction / screening behavior. Additions only.
- **MongoDB status contract**: only additions (`researcher.*` strand). No renames, no removals, no transition-point changes.
- **No training data generated**: everything shipped is prompt/code; no dataset prep, no `.jsonl` writes.

## 7. Wave structure

**Wave 1 (this session, 3 tasks + validation):**
- T1: Gateway vLLM primary + identity shim
- T2: Entity extraction module + wire into KG adapter
- T3: Researcher Agent (minimal viable with generic adapter)
- T4: Full-suite + bench + smoke validation

**Wave 2 (next session):**
- Domain adapter framework (Blob loader, TTL cache, seed YAMLs)
- Chart/DOCWAIN_VIZ targeted improvement pass
- Central prompt registry refactor
- Weekend full-set researcher refresh (Celery beat)

**Wave 3 (follow-up, as needed):**
- Per-domain adapter authoring
- Researcher precomputation of task-specific insights (e.g., "what questions should I ask?")
- Identity-in-weights training (begins the 7-phase curriculum at Phase 0 eval harness)

## 8. Success criteria (Wave 1)

- Gateway `_init_clients` lists vLLM as primary; env-var flip restores Cloud primary cleanly.
- Identity shim prefix appears in system prompts on every gateway call when enabled; absent when disabled.
- `_canonical_to_graph_payload` returns payload with non-empty `entities[]` when extraction text contains recognizable entities, empty list on feature-flag disable or DocWain call failure.
- `run_researcher_agent` Celery task exists on `researcher_queue`, processes a test doc end-to-end, writes an Insight node to Neo4j + enriches Qdrant payload.
- `trigger_embedding` dispatches all three tasks (embed, KG, researcher) in parallel. Failure of any one does not affect the others.
- Existing test suite (extraction, KG, API, integration) still passes.
- Bench: 7/7 still at 1.000.
- New unit tests per task pass.

## 9. Risks + mitigations

1. **DocWain vision-grafted V2 may be weak at entity extraction today.** Mitigation: prompt includes few-shot examples; parser tolerates malformed JSON; fallback to empty entities is the safe default. Quality improves over time via training (separate workstream).
2. **Researcher adds latency to the training stage.** Mitigation: Researcher runs in parallel with embedding on its own queue. Pipeline doesn't wait for it; `pipeline_status = TRAINING_COMPLETED` when embedding finishes. Researcher status is independent.
3. **Gateway primary swap could regress response quality** on text-only queries where Cloud 397B was materially better. Mitigation: the 2026-04-23 audit documents the quality gap; user accepted it for latency. Cloud stays as fallback on vLLM failure. If Cloud-quality is needed for specific operations, they can route explicitly via `Config.Model.PRIMARY_BACKEND="cloud"`.
4. **Identity shim conflicts with existing Reasoner system prompts.** Mitigation: the shim prepends, doesn't replace. The 200-line Reasoner system prompt still gets concatenated after the shim. Test: existing response-generation tests still pass.
5. **Researcher Agent hallucinates insights.** Mitigation: insights stored as structured JSON with `confidence` score per field; UI can filter low-confidence. Output parser rejects content that doesn't quote retrieved text for factual claims.

## 10. Rollback plan

If anything regresses in production after Wave 1 ships:

```bash
# Roll back all of Wave 1 simultaneously
export DOCWAIN_MODEL_PRIMARY_BACKEND=cloud
export DOCWAIN_MODEL_IDENTITY_SHIM_ENABLED=false
export DOCWAIN_KG_ENTITY_EXTRACTION_ENABLED=false
export DOCWAIN_RESEARCHER_ENABLED=false
sudo systemctl restart docwain-app docwain-celery-worker
```

All features are env-flag gated with safe defaults. No code change needed for rollback. Git revert is the hard backstop.

## 11. Test contract

- Unit tests per component (see Wave 1 plan).
- Integration test: `trigger_embedding` dispatches all 3 training tasks (embed, KG, researcher).
- Integration test: Researcher failure doesn't affect embedding status; KG failure doesn't affect researcher status.
- Bench: 7/7 extraction bench unchanged at 1.000.
- Smoke: end-to-end on a single doc — dispatch all tasks (with all backends mocked), verify outputs in MongoDB.
