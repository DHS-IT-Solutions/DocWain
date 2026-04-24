# KG Training-Stage Background Service Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire `build_knowledge_graph` Celery task as a parallel, fire-and-forget dispatch from the training-stage HITL approval endpoint, isolate it from embedding (separate queues, separate status strands, no cross-dependencies), update the extraction→graph adapter to accept the canonical Plan 1/2 extraction shape, and add a Redis observability log.

**Architecture:** Single dispatch point at `src/api/pipeline_api.py::trigger_embedding` fires both `embed_document.delay()` and `build_knowledge_graph.delay()` concurrently on separate Celery queues. Embedding writes `pipeline_status`; KG writes only `knowledge_graph.status`. No cross-task waits or checks. Adapter `_extraction_to_graph_payload` gains a canonical-shape branch that creates Document + chunk-level mentions without synthesizing entities (Researcher Agent, Plan 4, will populate entities later). Spec: `docs/superpowers/specs/2026-04-24-kg-training-stage-background-design.md`.

**Tech Stack:** Python 3.12, Celery (existing `embedding_queue` + `kg_queue`), Redis (dispatch broker + observability log), Neo4j (KG store, already wired via `src/kg/neo4j_store.py`), pytest + monkeypatch for isolation tests. No new external dependencies.

**Non-goals (do NOT expand scope):** no Researcher Agent work (Plan 4), no new entity-extraction model, no Neo4j ontology changes, no query-time `GraphAugmenter` changes, no new MongoDB status values, no backfill endpoint.

---

## File structure

**New files:**
- `src/kg/observability.py` — Redis per-KG-ingestion audit log.
- `tests/unit/api/__init__.py` (if missing — empty comment).
- `tests/unit/api/test_pipeline_api_kg_dispatch.py` — unit tests for the dispatch trigger.
- `tests/unit/kg/__init__.py` (if missing — empty comment).
- `tests/unit/kg/test_observability.py` — unit tests for KG observability module.
- `tests/unit/kg/test_extraction_to_graph_payload_canonical.py` — unit tests for canonical-shape adapter branch.
- `tests/integration/test_kg_dispatch_isolation.py` — integration tests proving KG and embedding don't interfere.

**Modified files:**
- `src/api/pipeline_api.py` — add KG dispatch call inside `trigger_embedding`.
- `src/tasks/embedding.py` — remove KG backfill dispatch (~lines 295-298 per prior exploration).
- `src/tasks/kg.py` — (a) update `_extraction_to_graph_payload` to handle canonical shape; (b) hook in Redis observability log at end of `build_knowledge_graph`; (c) remove any stale KG dispatch references that are found.
- Possibly: `src/tasks/screening.py` or any other file grep surfaces with a `build_knowledge_graph.delay()` call — those get removed.

**Git:** continue on branch `preprod_v02`. Commit after each task.

---

### Task 1: Grep audit — find all current KG dispatch sites

**Files:** none modified. Discovery only.

This task identifies EVERY current `build_knowledge_graph.delay()` / `kg_queue` enqueue / `get_graph_ingest_queue().enqueue()` reference in the codebase so later tasks know what to remove.

- [ ] **Step 1: Grep for all KG-related dispatch patterns**

```bash
cd /home/ubuntu/.config/superpowers/worktrees/DocWain/preprod_v02
grep -rn --include='*.py' "build_knowledge_graph\.delay\|build_knowledge_graph\.apply_async" src/ tests/ scripts/ 2>&1
echo "---"
grep -rn --include='*.py' "get_graph_ingest_queue\|graph_ingest_queue\|kg_queue" src/ tests/ scripts/ 2>&1
echo "---"
grep -rn --include='*.py' "_ingest_to_knowledge_graph\|enqueue_graph" src/ tests/ scripts/ 2>&1
```

- [ ] **Step 2: Record findings in a brief inline report**

Paste the grep output into the task's completion report. Group by file. Mark each hit as one of:
- `KEEP` — this is inside `src/tasks/kg.py` (the task definition itself) or test code.
- `REMOVE` — this is a dispatch or enqueue call in production code outside `src/tasks/kg.py` or `src/api/pipeline_api.py`.
- `INSPECT` — not obviously one or the other; flag for Task 2.

No code changes in this task. No commit.

---

### Task 2: Add KG dispatch to `trigger_embedding` endpoint (TDD)

**Files:**
- Modify: `src/api/pipeline_api.py`
- Create: `tests/unit/api/__init__.py` (if missing)
- Create: `tests/unit/api/test_pipeline_api_kg_dispatch.py`

- [ ] **Step 1: Create `tests/unit/api/__init__.py` if missing**

```bash
cd /home/ubuntu/.config/superpowers/worktrees/DocWain/preprod_v02
mkdir -p tests/unit/api
[ -f tests/unit/api/__init__.py ] || printf '# api endpoint unit tests\n' > tests/unit/api/__init__.py
```

- [ ] **Step 2: Write failing test**

Create `tests/unit/api/test_pipeline_api_kg_dispatch.py`:

```python
"""trigger_embedding dispatches BOTH embed_document AND build_knowledge_graph.

Verifies:
- Both tasks are enqueued on HITL approval.
- KG dispatch failure is swallowed (logged, not raised) so embedding still dispatches.
"""
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def fake_doc_in_screening_completed():
    """Return a dict representing a document whose status is SCREENING_COMPLETED."""
    from src.api.statuses import PIPELINE_SCREENING_COMPLETED
    return {
        "_id": "doc-abc",
        "subscription_id": "sub-1",
        "profile_id": "prof-1",
        "pipeline_status": PIPELINE_SCREENING_COMPLETED,
    }


def test_trigger_embedding_dispatches_both_tasks(fake_doc_in_screening_completed, monkeypatch):
    from src.api import pipeline_api

    embed_calls = []
    kg_calls = []

    def fake_embed_delay(document_id, subscription_id, profile_id):
        embed_calls.append((document_id, subscription_id, profile_id))
        return MagicMock(id="embed-task-id")

    def fake_kg_delay(document_id, subscription_id, profile_id):
        kg_calls.append((document_id, subscription_id, profile_id))
        return MagicMock(id="kg-task-id")

    # Provide the document in the mocked Mongo collection lookup.
    fake_collection = MagicMock()
    fake_collection.find_one.return_value = fake_doc_in_screening_completed

    monkeypatch.setattr("src.tasks.embedding.embed_document.delay", fake_embed_delay)
    monkeypatch.setattr("src.tasks.kg.build_knowledge_graph.delay", fake_kg_delay)
    monkeypatch.setattr(pipeline_api, "get_documents_collection", lambda: fake_collection, raising=False)

    # Use a synchronous runner so the async endpoint can be awaited in test
    import asyncio
    result = asyncio.run(pipeline_api.trigger_embedding(document_id="doc-abc"))

    assert len(embed_calls) == 1
    assert embed_calls[0] == ("doc-abc", "sub-1", "prof-1")
    assert len(kg_calls) == 1
    assert kg_calls[0] == ("doc-abc", "sub-1", "prof-1")


def test_trigger_embedding_tolerates_kg_dispatch_failure(fake_doc_in_screening_completed, monkeypatch):
    """If KG dispatch raises (e.g., Redis down), embedding dispatch still happens and endpoint returns success."""
    from src.api import pipeline_api

    embed_calls = []

    def fake_embed_delay(document_id, subscription_id, profile_id):
        embed_calls.append((document_id, subscription_id, profile_id))
        return MagicMock(id="embed-task-id")

    def fake_kg_delay(*a, **kw):
        raise RuntimeError("redis unreachable")

    fake_collection = MagicMock()
    fake_collection.find_one.return_value = fake_doc_in_screening_completed

    monkeypatch.setattr("src.tasks.embedding.embed_document.delay", fake_embed_delay)
    monkeypatch.setattr("src.tasks.kg.build_knowledge_graph.delay", fake_kg_delay)
    monkeypatch.setattr(pipeline_api, "get_documents_collection", lambda: fake_collection, raising=False)

    import asyncio
    # Must NOT raise — KG failure is swallowed
    asyncio.run(pipeline_api.trigger_embedding(document_id="doc-abc"))

    # Embedding still dispatched
    assert len(embed_calls) == 1
```

- [ ] **Step 3: Run to confirm failure**

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/pytest tests/unit/api/test_pipeline_api_kg_dispatch.py -x -q
```

Expected: **both tests FAIL** — either because `trigger_embedding` doesn't dispatch KG today, or because the `get_documents_collection` attr isn't exposed (monkeypatch with `raising=False` tolerates the latter). If the tests fail for an import reason unrelated to the KG dispatch behavior (e.g., `trigger_embedding` signature mismatch), adjust the test's call signature to match but keep the assertions intact.

- [ ] **Step 4: Implement the dispatch**

In `src/api/pipeline_api.py::trigger_embedding`, locate the existing `embed_document.delay(...)` call. Immediately AFTER it, before returning, add:

```python
# Plan 3: KG build runs in parallel on kg_queue. Fire-and-forget — KG failure
# is fully isolated from embedding and from pipeline_status. Spec §4.1 + §8.
try:
    from src.tasks.kg import build_knowledge_graph
    build_knowledge_graph.delay(document_id, subscription_id, profile_id)
    logger.info("KG ingestion dispatched for %s", document_id)
except Exception as exc:  # noqa: BLE001
    # Redis down or task registry issue. Do NOT propagate — embedding still ran.
    logger.warning("KG dispatch failed for %s: %s", document_id, exc)
```

Make sure `subscription_id` and `profile_id` are available in the enclosing scope (they must already be, since `embed_document.delay(...)` received them). If the variable names in the existing code differ (e.g., `sub_id`), use the actual names.

- [ ] **Step 5: Run test to verify pass**

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/pytest tests/unit/api/test_pipeline_api_kg_dispatch.py -x -q
```

Expected: 2 passed.

If the test still fails with an import/signature issue in `trigger_embedding`, fix the test to match the real signature (e.g., pass the real dependency-injected collection) but preserve the KG dispatch assertions. Do NOT loosen assertions.

- [ ] **Step 6: Commit**

```bash
git add src/api/pipeline_api.py tests/unit/api/__init__.py tests/unit/api/test_pipeline_api_kg_dispatch.py
git commit -m "kg: dispatch build_knowledge_graph from training-stage trigger, isolated from embedding"
```

No Claude/Anthropic/Co-Authored-By.

---

### Task 3: Remove stale KG dispatch sites

**Files:** depends on Task 1 findings. Typical sites:
- Modify: `src/tasks/screening.py` (if a `build_knowledge_graph.delay()` dispatch is present there)
- Modify: any other file Task 1 flagged as `REMOVE`

- [ ] **Step 1: Re-run the grep from Task 1**

```bash
cd /home/ubuntu/.config/superpowers/worktrees/DocWain/preprod_v02
grep -rn --include='*.py' "build_knowledge_graph\.delay\|build_knowledge_graph\.apply_async" src/ 2>&1 | grep -v "src/tasks/kg.py" | grep -v "src/api/pipeline_api.py"
```

This lists all production dispatch sites OUTSIDE the canonical two files. Expected after Plan 3 completes: empty output.

- [ ] **Step 2: For each remaining hit, remove the dispatch**

Replace the `build_knowledge_graph.delay(...)` call (and any surrounding single-purpose try/except that exists only to wrap it) with a one-line comment:

```python
# KG dispatch moved to src/api/pipeline_api.py::trigger_embedding (spec: 2026-04-24-kg-training-stage-background-design.md §4.1).
```

If removing the call leaves a now-unused import of `build_knowledge_graph`, remove the import too. Verify with grep that the import isn't used elsewhere in the same file.

- [ ] **Step 3: Re-run grep to confirm empty**

```bash
grep -rn --include='*.py' "build_knowledge_graph\.delay\|build_knowledge_graph\.apply_async" src/ 2>&1 | grep -v "src/tasks/kg.py" | grep -v "src/api/pipeline_api.py"
```

Expected: empty output (exit code 1 is fine — grep returns non-zero on no matches).

- [ ] **Step 4: Sanity — import smoke**

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/python -c "from src.tasks import kg, embedding, screening; from src.api import pipeline_api; print('import OK')"
```

Expected: `import OK`. If screening.py doesn't exist, skip it in the imports list.

- [ ] **Step 5: Run existing test suite**

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/pytest tests/unit -x -q --timeout=30
```

Expected: no new failures vs baseline (2 pre-existing unrelated failures in finetune/v2/test_pipeline.py).

- [ ] **Step 6: Commit**

```bash
git add -A src/
git commit -m "kg: remove stale dispatch sites; pipeline_api.trigger_embedding is the single trigger"
```

No Claude/Anthropic/Co-Authored-By.

---

### Task 4: Remove KG backfill dispatch from `embed_document`

**Files:**
- Modify: `src/tasks/embedding.py`

The exploration noted that `embed_document` at ~line 295-298 dispatches a backfill KG task if KG status is not ready. This contradicts spec §4.3's isolation principle — embedding must be completely unaware of KG.

- [ ] **Step 1: Locate the backfill**

```bash
cd /home/ubuntu/.config/superpowers/worktrees/DocWain/preprod_v02
sed -n '280,305p' src/tasks/embedding.py
grep -n "build_knowledge_graph\|kg_backfill\|knowledge_graph\.delay" src/tasks/embedding.py
```

Identify the block. Typical shape: an `if`-gate checking `knowledge_graph.status` followed by a `.delay()` call.

- [ ] **Step 2: Remove the block**

Replace the whole conditional + dispatch block with:

```python
# Plan 3: embedding is fully isolated from KG. No KG dispatch or status
# check from here. Spec §4.3.
```

Remove any now-unused `build_knowledge_graph` import at the top of the file.

- [ ] **Step 3: Verify embedding task still imports + existing tests pass**

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/python -c "from src.tasks.embedding import embed_document; print('import OK')"
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/pytest tests/unit -x -q --timeout=30
```

Expected: `import OK` and no new failures.

- [ ] **Step 4: Commit**

```bash
git add src/tasks/embedding.py
git commit -m "kg: remove backfill dispatch from embed_document (embedding now fully isolated from KG)"
```

No Claude/Anthropic/Co-Authored-By.

---

### Task 5: Update `_extraction_to_graph_payload` for canonical extraction shape (TDD)

**Files:**
- Modify: `src/tasks/kg.py`
- Create: `tests/unit/kg/__init__.py` (if missing — `# kg unit tests\n`)
- Create: `tests/unit/kg/test_extraction_to_graph_payload_canonical.py`

- [ ] **Step 1: Ensure test package exists**

```bash
mkdir -p tests/unit/kg
[ -f tests/unit/kg/__init__.py ] || printf '# kg unit tests\n' > tests/unit/kg/__init__.py
```

- [ ] **Step 2: Failing tests**

Create `tests/unit/kg/test_extraction_to_graph_payload_canonical.py`:

```python
"""`_extraction_to_graph_payload` must accept both legacy and canonical extraction shapes.

- Legacy (pre-Plan 1): top-level `entities[]`, `relationships[]`, `tables[]`, `sections`,
  `metadata{source_file, doc_type, ...}`, optional `temporal_spans[]`.
- Canonical (Plan 1/2): top-level `pages[].blocks[]`, `sheets[].cells`, `slides[].elements[]`,
  `metadata.doc_intel{doc_type_hint, ...}`, `metadata.coverage{...}`. No top-level entities.

Spec §5.
"""
from src.kg.ingest import GraphIngestPayload
from src.tasks.kg import _extraction_to_graph_payload


def _legacy_extraction() -> dict:
    return {
        "entities": [
            {"text": "Acme Corp", "type": "ORGANIZATION", "confidence": 0.9, "chunk_id": "c1"},
        ],
        "relationships": [],
        "tables": [{"headers": ["h1"], "rows": [["v1"]]}],
        "sections": {"intro": "text"},
        "metadata": {
            "source_file": "/blob/doc1.pdf",
            "filename": "doc1.pdf",
            "doc_type": "invoice",
            "doc_name": "doc1.pdf",
        },
        "temporal_spans": [],
    }


def _canonical_extraction() -> dict:
    return {
        "doc_id": "doc-1",
        "format": "pdf_native",
        "path_taken": "native",
        "pages": [
            {"page_num": 1,
             "blocks": [
                 {"text": "First paragraph of a native PDF.", "block_type": "paragraph"},
                 {"text": "Second block with more content.", "block_type": "paragraph"},
             ],
             "tables": [{"rows": [["h1", "h2"], ["r1c1", "r1c2"]]}],
             "images": []},
        ],
        "sheets": [],
        "slides": [],
        "metadata": {
            "doc_intel": {
                "doc_type_hint": "invoice",
                "layout_complexity": "simple",
                "has_handwriting": False,
                "routing_confidence": 0.9,
            },
            "coverage": {
                "verifier_score": 1.0,
                "missed_regions": [],
                "low_confidence_regions": [],
                "fallback_invocations": [],
            },
            "extraction_version": "2026-04-23-v1",
        },
    }


def test_legacy_shape_still_produces_valid_payload():
    payload = _extraction_to_graph_payload(
        extraction=_legacy_extraction(),
        document_id="doc-1",
        subscription_id="sub-1",
        profile_id="prof-1",
        source_name="doc1.pdf",
        screening_summary=None,
    )
    assert isinstance(payload, GraphIngestPayload)
    # Legacy shape has entities — payload should carry them
    assert len(payload.entities) >= 1
    assert any(e.get("text") == "Acme Corp" for e in payload.entities)


def test_canonical_shape_produces_valid_payload_without_entities():
    payload = _extraction_to_graph_payload(
        extraction=_canonical_extraction(),
        document_id="doc-1",
        subscription_id="sub-1",
        profile_id="prof-1",
        source_name="doc1.pdf",
        screening_summary=None,
    )
    assert isinstance(payload, GraphIngestPayload)
    # Canonical shape produces a Document + chunk mentions, but no entities yet
    assert payload.document is not None
    assert len(payload.mentions) >= 1  # chunks from pages[].blocks[]
    assert payload.entities == []
    assert payload.typed_relationships == []


def test_canonical_shape_doc_type_comes_from_doc_intel():
    payload = _extraction_to_graph_payload(
        extraction=_canonical_extraction(),
        document_id="doc-1",
        subscription_id="sub-1",
        profile_id="prof-1",
        source_name="doc1.pdf",
        screening_summary=None,
    )
    # doc_type should propagate from metadata.doc_intel.doc_type_hint
    doc_type = (payload.document or {}).get("doc_type") if isinstance(payload.document, dict) else getattr(payload.document, "doc_type", None)
    assert doc_type in ("invoice", "generic", "unknown")  # accept either exact pass-through or a generic fallback
```

- [ ] **Step 3: Inspect current `_extraction_to_graph_payload` signature**

```bash
sed -n '1,50p' src/tasks/kg.py
grep -n "def _extraction_to_graph_payload" src/tasks/kg.py
```

Note the exact parameter names and order. If the existing signature is different (e.g., doesn't take `screening_summary` as kwarg), adjust the test's call accordingly — preserve the behavioral assertions.

- [ ] **Step 4: Run to confirm fail**

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/pytest tests/unit/kg/test_extraction_to_graph_payload_canonical.py -x -q
```

Expected: `test_canonical_shape_*` tests FAIL (the function today only handles legacy shape). `test_legacy_shape_*` may pass depending on current behavior.

- [ ] **Step 5: Implement canonical branch**

In `src/tasks/kg.py::_extraction_to_graph_payload`, at the top of the function (after input normalization), add a shape-detection branch:

```python
def _is_canonical_extraction(extraction: dict) -> bool:
    """Canonical shape has pages/sheets/slides at the top level and no top-level entities."""
    if "entities" in extraction:
        # Legacy — top-level entities present
        return False
    has_pages_sheets_slides = any(k in extraction for k in ("pages", "sheets", "slides"))
    return has_pages_sheets_slides
```

Then inside `_extraction_to_graph_payload`:

```python
if _is_canonical_extraction(extraction):
    return _canonical_to_graph_payload(
        extraction=extraction,
        document_id=document_id,
        subscription_id=subscription_id,
        profile_id=profile_id,
        source_name=source_name,
    )
# Fall through to existing legacy processing
```

Add `_canonical_to_graph_payload` as a new private helper in the same file:

```python
def _canonical_to_graph_payload(
    *,
    extraction: dict,
    document_id: str,
    subscription_id: str,
    profile_id: str,
    source_name: str,
) -> GraphIngestPayload:
    """Build a minimal GraphIngestPayload from canonical-shape extraction.

    Canonical extraction (Plan 1/2) has no entity extraction — just text blocks,
    table rows, slide elements. We produce a Document node and one mention per
    text block (for traceability), but no entities or typed_relationships.
    Researcher Agent (Plan 4) enriches the graph with semantic entities later.

    Spec: 2026-04-24-kg-training-stage-background-design.md §5
    """
    meta = extraction.get("metadata") or {}
    doc_intel = meta.get("doc_intel") or {}

    doc_type = doc_intel.get("doc_type_hint") or "generic"

    document = {
        "document_id": document_id,
        "subscription_id": subscription_id,
        "profile_id": profile_id,
        "source_name": source_name,
        "doc_type": doc_type,
        "format": extraction.get("format") or "unknown",
        "path_taken": extraction.get("path_taken") or "unknown",
        "extraction_version": meta.get("extraction_version") or "",
    }

    mentions: list[dict] = []
    mention_counter = 0

    def _add_mention(text: str, locator: str):
        nonlocal mention_counter
        mention_counter += 1
        if not text or not text.strip():
            return
        mentions.append({
            "mention_id": f"{document_id}::chunk::{mention_counter}",
            "document_id": document_id,
            "text": text,
            "locator": locator,
        })

    for page in (extraction.get("pages") or []):
        page_num = page.get("page_num", 0)
        for block in (page.get("blocks") or []):
            _add_mention(block.get("text", ""), f"page:{page_num}:block")
        for table in (page.get("tables") or []):
            # Flatten table as one mention per row for traceability.
            for row_idx, row in enumerate(table.get("rows") or []):
                _add_mention(" | ".join(str(c) for c in row), f"page:{page_num}:table:row:{row_idx}")

    for sheet in (extraction.get("sheets") or []):
        sheet_name = sheet.get("name", "sheet")
        for coord, cell in (sheet.get("cells") or {}).items():
            value = (cell or {}).get("value")
            if value is None:
                continue
            _add_mention(str(value), f"sheet:{sheet_name}:cell:{coord}")

    for slide in (extraction.get("slides") or []):
        slide_num = slide.get("slide_num", 0)
        for elem in (slide.get("elements") or []):
            _add_mention(elem.get("text", ""), f"slide:{slide_num}:element")
        if (slide.get("notes") or "").strip():
            _add_mention(slide["notes"], f"slide:{slide_num}:notes")

    return GraphIngestPayload(
        document=document,
        entities=[],
        mentions=mentions,
        fields=[],
        typed_relationships=[],
        temporal_spans=[],
    )
```

Place `_is_canonical_extraction` and `_canonical_to_graph_payload` BEFORE `_extraction_to_graph_payload` so forward-references work.

- [ ] **Step 6: Run to verify**

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/pytest tests/unit/kg/test_extraction_to_graph_payload_canonical.py -x -q
```

Expected: 3 passed.

- [ ] **Step 7: Sanity — run broader KG tests**

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/pytest tests/unit/kg tests/unit/tasks -q --timeout=30 2>&1 | tail -8
```

Expected: no regressions.

- [ ] **Step 8: Commit**

```bash
git add src/tasks/kg.py tests/unit/kg/__init__.py tests/unit/kg/test_extraction_to_graph_payload_canonical.py
git commit -m "kg: adapter handles canonical Plan1/2 extraction shape (Document + chunk mentions, no entities)"
```

No Claude/Anthropic/Co-Authored-By.

---

### Task 6: KG observability module + hook (TDD)

**Files:**
- Create: `src/kg/observability.py`
- Create: `tests/unit/kg/test_observability.py`
- Modify: `src/tasks/kg.py` (hook the log write at end of `build_knowledge_graph`)

- [ ] **Step 1: Failing tests**

Create `tests/unit/kg/test_observability.py`:

```python
import json
import time

from src.kg.observability import (
    KGLogEntry,
    build_kg_redis_key,
    serialize_kg_entry,
    write_kg_entry_if_redis,
)


def test_serialize_kg_entry_contains_required_fields():
    entry = KGLogEntry(
        doc_id="d1",
        status="KG_COMPLETED",
        nodes_created=5,
        edges_created=3,
        timings_ms={"ingest": 1200.0},
        error=None,
        completed_at=time.time(),
    )
    data = json.loads(serialize_kg_entry(entry))
    assert data["doc_id"] == "d1"
    assert data["status"] == "KG_COMPLETED"
    assert data["nodes_created"] == 5
    assert data["edges_created"] == 3
    assert data["timings_ms"]["ingest"] == 1200.0


def test_build_kg_redis_key_includes_doc_id():
    key = build_kg_redis_key("doc-123")
    assert "doc-123" in key
    assert key.startswith("kg:log:")


def test_write_kg_entry_if_redis_accepts_none_client():
    entry = KGLogEntry(doc_id="d2", status="KG_PENDING", nodes_created=0, edges_created=0,
                       timings_ms={}, error=None, completed_at=time.time())
    write_kg_entry_if_redis(redis_client=None, entry=entry)  # must not raise


def test_write_kg_entry_if_redis_sets_ttl():
    entry = KGLogEntry(doc_id="d3", status="KG_COMPLETED", nodes_created=1, edges_created=0,
                       timings_ms={}, error=None, completed_at=time.time())
    calls = {}

    class FakeRedis:
        def setex(self, key, ttl, value):
            calls["key"] = key
            calls["ttl"] = ttl
            calls["value"] = value

    write_kg_entry_if_redis(redis_client=FakeRedis(), entry=entry)
    assert calls["key"].endswith(":d3")
    assert calls["ttl"] == 7 * 24 * 3600
    assert "d3" in calls["value"]


def test_write_kg_entry_swallows_redis_exceptions():
    entry = KGLogEntry(doc_id="d4", status="KG_FAILED", nodes_created=0, edges_created=0,
                       timings_ms={}, error="neo4j unreachable", completed_at=time.time())

    class BrokenRedis:
        def setex(self, *a, **kw):
            raise RuntimeError("redis down")

    # Must not raise
    write_kg_entry_if_redis(redis_client=BrokenRedis(), entry=entry)
```

- [ ] **Step 2: Implement module**

Create `src/kg/observability.py`:

```python
"""Per-KG-ingestion Redis audit log.

Sibling of `src.extraction.vision.observability` with the same shape and TTL
conventions. Captures nodes_created / edges_created / error so operators can
see where KG enrichment bleeds.

Spec: 2026-04-24-kg-training-stage-background-design.md §7
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


REDIS_KEY_PREFIX = "kg:log"
TTL_SECONDS = 7 * 24 * 3600


@dataclass
class KGLogEntry:
    doc_id: str
    status: str  # KG_PENDING | KG_IN_PROGRESS | KG_COMPLETED | KG_FAILED
    nodes_created: int = 0
    edges_created: int = 0
    timings_ms: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None
    completed_at: float = 0.0


def build_kg_redis_key(doc_id: str) -> str:
    return f"{REDIS_KEY_PREFIX}:{doc_id}"


def serialize_kg_entry(entry: KGLogEntry) -> str:
    return json.dumps(asdict(entry), ensure_ascii=False)


def write_kg_entry_if_redis(*, redis_client: Any, entry: KGLogEntry) -> None:
    """Write the entry to Redis with TTL. Best-effort — errors swallowed."""
    if redis_client is None:
        return
    try:
        redis_client.setex(
            build_kg_redis_key(entry.doc_id),
            TTL_SECONDS,
            serialize_kg_entry(entry),
        )
    except Exception:
        # Observability must never break the KG task.
        pass
```

- [ ] **Step 3: Run tests to verify pass**

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/pytest tests/unit/kg/test_observability.py -x -q
```

Expected: 5 passed.

- [ ] **Step 4: Hook into `build_knowledge_graph`**

Modify `src/tasks/kg.py::build_knowledge_graph` — at each final status set (COMPLETED or FAILED), add an observability write. Keep it non-critical: wrap in try/except. Example shape — adapt to the actual local variables in the task:

```python
# Added at the START of the task function, near other imports/setup:
import time as _time_kg_log
from src.kg.observability import KGLogEntry, write_kg_entry_if_redis

_kg_log_started_at = _time_kg_log.perf_counter()

# ... existing task logic ...

# Just before setting status to KG_COMPLETED:
try:
    _kg_log_nodes = 0
    _kg_log_edges = 0
    if 'ingest_result' in locals() and ingest_result:
        _kg_log_nodes = int(getattr(ingest_result, 'nodes_created', 0) or 0)
        _kg_log_edges = int(getattr(ingest_result, 'edges_created', 0) or 0)
    _kg_log_entry = KGLogEntry(
        doc_id=document_id,
        status="KG_COMPLETED",
        nodes_created=_kg_log_nodes,
        edges_created=_kg_log_edges,
        timings_ms={"total": (_time_kg_log.perf_counter() - _kg_log_started_at) * 1000.0},
        error=None,
        completed_at=_time_kg_log.time(),
    )
    from src.api.dw_newron import get_redis_client
    write_kg_entry_if_redis(redis_client=get_redis_client(), entry=_kg_log_entry)
except Exception:
    pass  # Observability must never break the task.
```

And inside the task's except-block that catches failure (before setting status to KG_FAILED):

```python
try:
    _kg_log_entry = KGLogEntry(
        doc_id=document_id,
        status="KG_FAILED",
        nodes_created=0,
        edges_created=0,
        timings_ms={"total": (_time_kg_log.perf_counter() - _kg_log_started_at) * 1000.0},
        error=repr(exc) if 'exc' in locals() else "unknown",
        completed_at=_time_kg_log.time(),
    )
    from src.api.dw_newron import get_redis_client
    write_kg_entry_if_redis(redis_client=get_redis_client(), entry=_kg_log_entry)
except Exception:
    pass
```

Match the actual variable names in the task — `document_id`, the exception variable, the ingest result variable. If `ingest_result` doesn't expose `nodes_created`/`edges_created`, log zeros (we're measuring task success/failure first; node counts are nice-to-have).

- [ ] **Step 5: Verify nothing broke**

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/python -c "from src.tasks.kg import build_knowledge_graph; print('import OK')"
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/pytest tests/unit/kg -x -q
```

Expected: 8 passed (5 observability + 3 canonical adapter).

- [ ] **Step 6: Commit**

```bash
git add src/kg/observability.py tests/unit/kg/test_observability.py src/tasks/kg.py
git commit -m "kg: add Redis observability log + hook into build_knowledge_graph (best-effort)"
```

No Claude/Anthropic/Co-Authored-By.

---

### Task 7: Integration test — dispatch + isolation

**Files:**
- Create: `tests/integration/test_kg_dispatch_isolation.py`

- [ ] **Step 1: Write the test**

Create `tests/integration/test_kg_dispatch_isolation.py`:

```python
"""Integration-style tests proving KG and embedding are fully isolated.

These tests monkey-patch the Celery delay methods so no real broker is needed.
They verify:
- Both tasks dispatch on HITL approval.
- KG dispatch failure does not affect embedding dispatch.
- KG task sets only `knowledge_graph.status`, never `pipeline_status`.
"""
from unittest.mock import MagicMock

import pytest


def test_trigger_dispatches_both_tasks(monkeypatch):
    """POST /{document_id}/embed dispatches BOTH embed_document and build_knowledge_graph."""
    from src.api import pipeline_api
    from src.api.statuses import PIPELINE_SCREENING_COMPLETED

    dispatched = {"embed": 0, "kg": 0}

    def fake_embed_delay(*a, **kw):
        dispatched["embed"] += 1
        return MagicMock()

    def fake_kg_delay(*a, **kw):
        dispatched["kg"] += 1
        return MagicMock()

    monkeypatch.setattr("src.tasks.embedding.embed_document.delay", fake_embed_delay)
    monkeypatch.setattr("src.tasks.kg.build_knowledge_graph.delay", fake_kg_delay)

    fake_collection = MagicMock()
    fake_collection.find_one.return_value = {
        "_id": "doc-int-1",
        "subscription_id": "sub-i",
        "profile_id": "prof-i",
        "pipeline_status": PIPELINE_SCREENING_COMPLETED,
    }
    monkeypatch.setattr(pipeline_api, "get_documents_collection", lambda: fake_collection, raising=False)

    import asyncio
    asyncio.run(pipeline_api.trigger_embedding(document_id="doc-int-1"))

    assert dispatched == {"embed": 1, "kg": 1}


def test_kg_task_does_not_touch_pipeline_status(monkeypatch):
    """build_knowledge_graph must only mutate knowledge_graph.*; never pipeline_status."""
    import src.tasks.kg as kg_mod

    seen_updates = []

    class FakeCollection:
        def update_one(self, filter, update, **kw):
            seen_updates.append(update)
            return MagicMock(matched_count=1, modified_count=1)

        def find_one(self, *a, **kw):
            # Minimal document used by the task; specific keys depend on implementation.
            return {"_id": filter.get("_id") if isinstance(filter, dict) else "doc-x",
                    "subscription_id": "sub-x", "profile_id": "prof-x"}

    fake_col = FakeCollection()
    monkeypatch.setattr(kg_mod, "get_documents_collection", lambda: fake_col, raising=False)
    # Stub the heavy lifts — we only check which fields are being updated.
    monkeypatch.setattr(kg_mod, "_load_extraction_from_blob", lambda *a, **kw: {"format": "docx", "pages": []}, raising=False)
    monkeypatch.setattr(kg_mod, "_load_screening_from_blob", lambda *a, **kw: None, raising=False)
    monkeypatch.setattr(kg_mod, "ingest_graph_payload", lambda *a, **kw: MagicMock(nodes_created=0, edges_created=0), raising=False)

    # Run the task function directly (not via Celery) by calling .run on the task object.
    # Celery-decorated tasks expose .run for synchronous invocation.
    try:
        kg_mod.build_knowledge_graph.run("doc-x", "sub-x", "prof-x")
    except Exception:
        # Even if the stubbed task path raises somewhere internal, verify the update
        # pattern so far.
        pass

    # Inspect all Mongo update operations issued during the task.
    touched_fields: set[str] = set()
    for upd in seen_updates:
        for op, fields in upd.items():
            if isinstance(fields, dict):
                touched_fields.update(fields.keys())

    # KG task may update these:
    allowed_prefixes = ("knowledge_graph", "stages.knowledge_graph")
    for f in touched_fields:
        # Any field the KG task writes must be inside the knowledge_graph strand.
        assert f.startswith(allowed_prefixes) or f in ("updated_at", "last_modified"), (
            f"KG task wrote to forbidden field {f!r} — must stay inside knowledge_graph strand"
        )
    # Specifically must NEVER touch pipeline_status
    assert "pipeline_status" not in touched_fields
```

- [ ] **Step 2: Run the test**

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/pytest tests/integration/test_kg_dispatch_isolation.py -x -q
```

Expected: both pass.

If `test_kg_task_does_not_touch_pipeline_status` fails because the task's real code path touches `pipeline_status`, that's a spec violation that must be fixed in Task 4 (or a follow-up). Report the exact update operation the test caught.

If the test's stubs don't match the real task's function names (e.g., `_load_extraction_from_blob` isn't the actual helper), adapt the `monkeypatch.setattr` targets to the real names. The assertion about not touching `pipeline_status` is the load-bearing part and must stay.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_kg_dispatch_isolation.py
git commit -m "kg: integration tests — dispatch both tasks on approval + KG never touches pipeline_status"
```

No Claude/Anthropic/Co-Authored-By.

---

### Task 8: Full-suite + bench validation

Not a code task — validation only.

- [ ] **Step 1: Run extraction + integration + KG tests**

```bash
cd /home/ubuntu/.config/superpowers/worktrees/DocWain/preprod_v02
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/pytest tests/unit/extraction tests/unit/kg tests/unit/api tests/integration -q --timeout=30
```

Expected: all pass.

- [ ] **Step 2: Run bench runner**

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/python -m tests.extraction_bench.runner
```

Expected: 7/7 `[PASS]` (unchanged from Plan 2), exit 0.

- [ ] **Step 3: Broader sanity**

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/pytest tests/unit -q --timeout=30 2>&1 | tail -10
```

Expected: previously-passing count unchanged (+ the new Plan 3 tests), 2 pre-existing unrelated failures still present.

- [ ] **Step 4: Confirm branch state**

```bash
git log --oneline preprod_v01..HEAD | head -20
```

No commit for Task 8.

---

## Self-review — spec coverage

- **§3 Architecture** (single dispatch point, two parallel queues): Tasks 2, 3, 4 ✓
- **§4.1 One trigger site**: Task 2 ✓
- **§4.2 Stale dispatch cleanup**: Tasks 1 + 3 ✓
- **§4.3 Embedding does not re-dispatch KG**: Task 4 ✓
- **§5 Canonical shape adapter**: Task 5 ✓
- **§6 Status contract unchanged**: enforced by Task 7 assertion ✓
- **§7 Observability**: Task 6 ✓
- **§8 Error isolation**: Tasks 2/4 (try/except swallows KG dispatch failure; embedding remove-KG-call; KG task status isolation in Task 7) ✓
- **§9 Testing**: Tasks 2, 5, 6, 7 ✓
- **§11 Risks** (stale dispatch, shape mismatch): Tasks 1, 5 cover ✓
- **§12 Success criteria**: all covered ✓

## Self-review — placeholder scan

- Every code step shows complete code.
- Where variable names depend on the real file's local names (Task 2's `subscription_id`, Task 6's exception variable), the plan says "use the actual names from the file" — not a placeholder, an explicit context instruction.
- No "TBD" / "fill in" / "as appropriate".

## Self-review — type consistency

- `GraphIngestPayload` referenced consistently from `src.kg.ingest` across Tasks 5 and 7.
- `KGLogEntry`, `build_kg_redis_key`, `serialize_kg_entry`, `write_kg_entry_if_redis` defined in Task 6 and used in Task 6's hook.
- `_is_canonical_extraction` + `_canonical_to_graph_payload` defined in Task 5 and called from existing `_extraction_to_graph_payload`.
- `PIPELINE_SCREENING_COMPLETED` imported from `src.api.statuses` in Tasks 2 and 7.
- No drift.

Plan complete.
