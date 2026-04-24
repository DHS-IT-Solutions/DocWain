# KG as Training-Stage Background Service — Design

**Date:** 2026-04-24
**Branch:** `preprod_v02`
**Status:** Approved for implementation
**Owner:** Muthu (collaboratively brainstormed)

## 1. Purpose

Reintroduce Knowledge Graph (KG) ingestion as a single, centrally-triggered async background service that runs in the training stage, in parallel with embedding but fully isolated from it. Closes a gap left by Plan 1 which correctly removed extraction-stage KG triggers but did not replace them, leaving KG orphaned.

Per `project_post_preprod_roadmap.md` item 4 and `feedback_pipeline_flow.md`: KG belongs in the training stage (HITL-approved), as a backend service, never in extraction or screening.

Per user directive 2026-04-24: **"KG build should be a background process that does not affect the embedding process."**

## 2. Scope

### In scope
- Single consolidated KG dispatch trigger: the training-stage approval endpoint (`POST /{document_id}/embed` in `src/api/pipeline_api.py`), firing `build_knowledge_graph.delay()` alongside the existing `embed_document.delay()`.
- Removal of any stale KG dispatch sites elsewhere (screening task, dataHandler, etc.) to preserve single-source-of-truth for the trigger.
- Update of `_extraction_to_graph_payload` (in `src/tasks/kg.py`) to accept the canonical Plan 1/2 extraction shape (`pages[].blocks[]`, `sheets[].cells`, `slides[].elements[]`, `metadata.doc_intel`).
- Strict isolation: embedding and KG run on different Celery queues (`embedding_queue` vs `kg_queue`); neither task waits on or checks the other; status strands are independent.
- Integration tests verifying both tasks dispatched on approval + KG failure doesn't affect embedding success.

### Non-goals
- **No Researcher Agent work** — that's Plan 4. Plan 3 ingests whatever structure canonical extraction already produces, without enriching it.
- **No new entity-extraction model** — KG gets the extracted text blocks + any entities that happen to be in the extraction output today. When Researcher Agent ships (Plan 4), it populates KG with insights.
- **No Neo4j ontology changes** — reuse existing schema.
- **No changes to query-time `GraphAugmenter`** (continues to read the graph as-is).
- **No new MongoDB status values** — reuse existing `KG_PENDING`/`KG_IN_PROGRESS`/`KG_COMPLETED`/`KG_FAILED` strand per `feedback_mongo_status_stability.md`.

## 3. Architecture

```
                 HITL approval endpoint
                 POST /{document_id}/embed
                        (pipeline_api.py)
                              ↓
                 ┌────────── dispatches ──────────┐
                 ↓                                ↓
       embed_document.delay()          build_knowledge_graph.delay()
       (embedding_queue)                (kg_queue)
                 ↓                                ↓
       reads extraction.json            reads extraction.json
       builds enriched chunks           runs _extraction_to_graph_payload
       upserts Qdrant                   ingest_graph_payload → Neo4j
       updates MongoDB:                 updates MongoDB:
         pipeline_status                  knowledge_graph.status
         stages.embedding.status          (pipeline_status NOT touched)
```

**Isolation guarantees:**
- The two tasks share nothing but the `document_id` in their arguments and the extraction.json on Azure Blob (read-only).
- Embedding never checks KG status. Embedding completes → `PIPELINE_TRAINING_COMPLETED` is set by embedding alone. User sees training complete even if KG is still running or has failed.
- KG never touches `pipeline_status` or `stages.embedding.*`. KG writes exclusively to its own `knowledge_graph.*` subdocument.
- KG failure/retry/DLQ is contained inside the KG worker. Embedding worker is unaware.
- Different Celery queues → different worker pools → true parallelism, no resource contention beyond shared Redis broker.

## 4. Trigger wiring

### 4.1 The one trigger site

`src/api/pipeline_api.py::trigger_embedding` (currently at ~line 152) is the single HITL-approved entry to the training stage. Today it validates `pipeline_status == PIPELINE_SCREENING_COMPLETED` and dispatches `embed_document.delay(document_id, subscription_id, profile_id)`.

Plan 3 adds immediately after that dispatch:

```python
# Dispatch KG build in parallel — fire-and-forget background, fully isolated
# from embedding. KG failure/slowness does not affect pipeline_status.
try:
    from src.tasks.kg import build_knowledge_graph
    build_knowledge_graph.delay(document_id, subscription_id, profile_id)
    logger.info("KG ingestion dispatched for %s", document_id)
except Exception as exc:  # noqa: BLE001
    # Queue-level failure (Redis down, etc.) — log and move on. Do NOT block
    # or affect the embedding dispatch. Operator can backfill KG later.
    logger.warning("KG dispatch failed for %s: %s", document_id, exc)
```

Both dispatches return immediately; the endpoint response tells the operator training has started. Polling the document status will show `embedding.status` and `knowledge_graph.status` evolving independently.

### 4.2 Stale dispatch cleanup

Plan 1 removed KG triggers from extraction_service.py / embedding_service.py / dataHandler.py. Exploration flagged that some KG dispatch may still exist in `src/tasks/screening.py` or similar. Plan 3 grep-searches for every `build_knowledge_graph` or `kg_queue` enqueue reference and:
- Keeps ONLY the one in `pipeline_api.py::trigger_embedding`.
- Removes any other dispatch site, replacing with a comment pointing to spec §4.1.
- The `build_knowledge_graph` task itself stays — it's the worker, not the dispatcher.

### 4.3 Explicit non-dispatch: embed_document does not re-dispatch KG

The current `embed_document` task at `src/tasks/embedding.py:295-298` reportedly dispatches a backfill KG task if KG status is not ready. Plan 3 removes this line — embedding must be completely unaware of KG. If backfill is needed for documents whose KG failed, it's a separate operator-triggered job (out of scope for Plan 3).

## 5. Canonical-shape adapter update

### 5.1 Current state

`src/tasks/kg.py::_extraction_to_graph_payload` reads extraction JSON and expects:
- Top-level `entities[]` (with `{text, type, confidence, evidence, chunk_id}`)
- Top-level `relationships[]`
- Top-level `tables[]`
- Top-level `sections`
- Top-level `metadata{}` with `source_file`, `doc_type`, etc.
- Top-level `temporal_spans[]` (optional)

### 5.2 New canonical shape from Plan 1/2

```json
{
  "doc_id": "...",
  "format": "pdf_native|docx|xlsx|...",
  "path_taken": "native|vision|mixed",
  "pages": [{"page_num": 1, "blocks": [{"text": "...", "block_type": "paragraph"}],
              "tables": [{"rows": [[...], [...]]}], "images": []}],
  "sheets": [{"name": "...", "cells": {...}, "hidden": false, ...}],
  "slides": [{"slide_num": 1, "elements": [...], "notes": "...", ...}],
  "metadata": {
    "doc_intel": {"doc_type_hint": "...", "layout_complexity": "...",
                  "has_handwriting": bool, "routing_confidence": ...},
    "coverage": {"verifier_score": ..., "missed_regions": [],
                 "low_confidence_regions": [], "fallback_invocations": []},
    "extraction_version": "2026-04-XX-..."
  }
}
```

No `entities[]`, `relationships[]`, or `sections` at top level.

### 5.3 Adapter changes

`_extraction_to_graph_payload` gets a backwards-compatible update:

1. **If top-level `entities[]` exists**, keep the existing processing (legacy path for documents extracted before Plan 1).
2. **Otherwise (canonical shape):**
   - Create Document node from `metadata.doc_intel` fields (doc_type_hint → doc_type) plus legacy `metadata.source_file`/`filename` if present.
   - Create chunk entries from `pages[].blocks[]`, `sheets[].cells` (flatten to text), and `slides[].elements[]`. Each chunk becomes a Mention node attached to Document.
   - Do NOT synthesize entities — leave `entities[]`, `typed_relationships[]`, `temporal_spans[]` empty in the GraphIngestPayload. Researcher Agent (Plan 4) will fill these in later.
   - `tables[]` in pages: record as structured data on the page/chunk level, but don't entity-extract from them.
3. **The KG task completes successfully** even with an empty-entities payload. Document + chunk nodes get written. Status flips to `KG_COMPLETED`. Empty is not failure.

### 5.4 Why this matters

Downstream `GraphAugmenter` queries look up entities at retrieval time. Today those queries return empty for canonical-shape documents (because there are no entities yet). That's fine — RAG falls back to vector-only retrieval without entity augmentation until Researcher Agent populates the graph.

## 6. Status contract (unchanged per existing rules)

Per `feedback_mongo_status_stability.md` and `src/api/statuses.py`:

- **`pipeline_status`** (user-facing): `PIPELINE_SCREENING_COMPLETED` → `PIPELINE_EMBEDDING_IN_PROGRESS` → `PIPELINE_TRAINING_COMPLETED` | `PIPELINE_EMBEDDING_FAILED`. **Set by embedding task only. KG never touches this field.**
- **`stages.embedding.status`**: `STAGE_IN_PROGRESS` → `STAGE_COMPLETED` | `STAGE_FAILED`. Set by embedding.
- **`knowledge_graph.status`**: `KG_PENDING` → `KG_IN_PROGRESS` → `KG_COMPLETED` | `KG_FAILED`. Set by KG. Independent strand.

Operators can query both subdocuments to see the full picture. UI reads `pipeline_status` for the primary progress signal; `knowledge_graph.status` is an advanced detail.

## 7. Observability

- KG task logs as it does today (`src/tasks/kg.py` already has logging). No changes.
- KG task writes to Redis observability log in the same pattern Plan 2 established for extraction: key `kg:log:{doc_id}`, TTL 7 days, entry includes `doc_id`, `status`, `nodes_created`, `edges_created`, `timings_ms`, `completed_at`.
- The Plan 2 observability module (`src/extraction/vision/observability.py`) is repurposable — either reuse it directly (rename `ExtractionLogEntry` → generic `StageLogEntry`) or add a sibling `src/kg/observability.py` with a parallel shape.

Recommendation: **sibling `src/kg/observability.py`** — avoids retrofitting the extraction module and keeps KG self-contained. The two logs coexist under different Redis keys.

## 8. Error isolation

- **Redis down at dispatch time:** trigger_embedding's KG dispatch catches the exception, logs, does not raise. Embedding dispatch already happened (or also fails independently — same pattern).
- **KG task fails at runtime** (e.g., Neo4j unreachable): Celery retries per its existing `@app.task(max_retries=3)` decorator; on exhaustion, status → `KG_FAILED` in MongoDB. Embedding is unaffected — `pipeline_status` is already `PIPELINE_TRAINING_COMPLETED` if embedding finished first.
- **Embedding fails:** `pipeline_status` → `PIPELINE_EMBEDDING_FAILED`. KG may still complete; status stays `KG_COMPLETED`. The user-facing pipeline signals failure (embedding is required for search); KG completeness is an independent asset.
- **Both fail:** pipeline fails on embedding; KG fails independently. Operator retries both separately.

No locking. No cross-task wait. No cascading status.

## 9. Testing

- **Unit:** adapter handles canonical shape — produces a GraphIngestPayload with Document node + chunk mentions, empty entities (confirm payload.to_dict() is well-formed).
- **Unit:** adapter handles legacy shape — existing behavior preserved (top-level `entities[]` processed).
- **Unit:** trigger dispatches both tasks on HITL approval.
- **Unit:** KG dispatch failure does not raise out of trigger (embedding dispatch still happens).
- **Unit:** KG task sets `knowledge_graph.status` but never mutates `pipeline_status`.
- **Integration:** fake the HITL trigger, verify Redis receives two enqueued tasks on different queues.
- **Integration smoke:** embedding success + KG failure → `pipeline_status=TRAINING_COMPLETED`, `knowledge_graph.status=KG_FAILED`.

## 10. Phased rollout

Single phase — the change is architectural trigger wiring + a small adapter update. No training workstream. No quality bench (KG content tested via existing `src/kg/` tests and the new unit tests above).

On merge to `preprod_v01`, operator restarts Celery workers to pick up any route changes (the `kg_queue` already exists in `celery_app.py`, so no Celery restart strictly needed for new queue registration — but a restart clears any stale state).

## 11. Risks

1. **Stale KG dispatch site missed** — if a third KG dispatch exists in code not surveyed, both old and new dispatches fire for the same doc. Mitigation: grep-all for `build_knowledge_graph.delay` and `kg_queue` references in the plan's first task.
2. **Canonical-shape adapter produces empty GraphIngestPayload** — acceptable baseline for Plan 3; Researcher Agent (Plan 4) is the content layer. Document this in the adapter docstring so future contributors don't try to add entity extraction back into extraction (wrong place).
3. **KG task retries storm on Neo4j outage** — existing `max_retries=3` limits blast. Not introducing new retry behavior.
4. **Backfill for documents ingested during KG outage** — operator can re-run `trigger_embedding` or add a `trigger_kg_only` endpoint later. Out of scope for Plan 3.

## 12. Success criteria

- `POST /{document_id}/embed` dispatches BOTH `embed_document.delay()` and `build_knowledge_graph.delay()`.
- No other KG dispatch site exists in the codebase (verified by grep).
- `_extraction_to_graph_payload` handles canonical-shape extraction JSON without raising; produces a valid GraphIngestPayload.
- KG failure in tests does NOT flip `pipeline_status` away from `PIPELINE_TRAINING_COMPLETED`.
- All existing tests still pass; new tests added per §9.
- `embed_document` no longer contains a KG-backfill dispatch (removed).
- No Claude/Anthropic/Co-Authored-By in commit messages.
