"""Integration-style tests proving KG and embedding are fully isolated.

Verifies:
- POST /{document_id}/embed dispatches BOTH embed_document and build_knowledge_graph.
- The KG task NEVER writes to `pipeline_status` — stays within `knowledge_graph.*`.

These tests monkey-patch the Celery delay methods and Mongo helpers so no real
broker / DB is needed.
"""
import asyncio
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Test 1: trigger_embedding dispatches BOTH tasks
# ---------------------------------------------------------------------------

def test_trigger_dispatches_both_tasks(monkeypatch):
    """POST /{document_id}/embed dispatches BOTH embed_document and build_knowledge_graph."""
    from src.api import pipeline_api
    from src.api.statuses import PIPELINE_SCREENING_COMPLETED

    dispatched = {"embed": 0, "kg": 0}

    def fake_embed_delay(*a, **kw):
        dispatched["embed"] += 1
        return MagicMock(id="task-embed-1")

    def fake_kg_delay(*a, **kw):
        dispatched["kg"] += 1
        return MagicMock()

    monkeypatch.setattr("src.tasks.embedding.embed_document.delay", fake_embed_delay)
    monkeypatch.setattr("src.tasks.kg.build_knowledge_graph.delay", fake_kg_delay)

    fake_record = {
        "_id": "doc-int-1",
        "subscription_id": "sub-i",
        "profile_id": "prof-i",
        "pipeline_status": PIPELINE_SCREENING_COMPLETED,
    }

    # pipeline_api imports get_document_record and append_audit_log at module
    # level from src.api.document_status — patch them on the pipeline_api module.
    monkeypatch.setattr(pipeline_api, "get_document_record", lambda *a, **kw: fake_record)
    monkeypatch.setattr(pipeline_api, "append_audit_log", lambda *a, **kw: None)

    result = asyncio.run(pipeline_api.trigger_embedding(document_id="doc-int-1"))

    assert dispatched["embed"] == 1, f"embed dispatched {dispatched['embed']} times (expected 1)"
    assert dispatched["kg"] == 1, f"kg dispatched {dispatched['kg']} times (expected 1)"
    assert result["status"] == "EMBEDDING_IN_PROGRESS"


# ---------------------------------------------------------------------------
# Test 2: KG task never writes to pipeline_status
# ---------------------------------------------------------------------------

def test_kg_task_does_not_touch_pipeline_status(monkeypatch):
    """build_knowledge_graph must only mutate knowledge_graph.*; never pipeline_status.

    We patch update_stage (the sole Mongo-write surface used by the KG task) and
    capture every field name it tries to set.  Assertion: no write touches
    'pipeline_status'.
    """
    import src.tasks.kg as kg_mod

    # Track all (stage, field-prefix) pairs that update_stage is called with.
    update_stage_calls = []

    def recording_update_stage(document_id, stage, *args, **kwargs):
        update_stage_calls.append({"stage": stage, "args": args, "kwargs": kwargs})

    def recording_append_audit_log(document_id, event, *args, **kwargs):
        pass  # we only care about update_stage

    # Patch the imported names inside the kg module.
    monkeypatch.setattr(kg_mod, "update_stage", recording_update_stage)
    monkeypatch.setattr(kg_mod, "append_audit_log", recording_append_audit_log)
    monkeypatch.setattr(kg_mod, "get_document_record",
                        lambda *a, **kw: {"_id": "doc-x",
                                          "subscription_id": "sub-x",
                                          "profile_id": "prof-x"},
                        raising=False)

    # Patch blob downloads to avoid real Azure calls.
    fake_extraction = {
        "format": "docx",
        "pages": [],
        "common_data": {},
        "elements": [],
    }

    def fake_download_blob_json(blob_path):
        return fake_extraction

    def fake_try_download_blob_json(blob_path):
        return None  # no screening blob

    monkeypatch.setattr(kg_mod, "_download_blob_json", fake_download_blob_json)
    monkeypatch.setattr(kg_mod, "_try_download_blob_json", fake_try_download_blob_json)

    # Patch _extraction_to_graph_payload to return None so we hit the early-exit
    # path (no entities) — this exercises the update_stage(COMPLETED) call without
    # needing Neo4j.
    monkeypatch.setattr(kg_mod, "_extraction_to_graph_payload",
                        lambda **kw: None)

    # Patch Redis/observability (best-effort blocks inside the task).
    with patch("src.kg.observability.write_kg_entry_if_redis", return_value=None), \
         patch("src.api.dw_newron.get_redis_client", return_value=None):
        try:
            # Celery's .apply() executes the task synchronously without a broker.
            # For bind=True tasks it injects the request context automatically —
            # do NOT pass self manually.
            kg_mod.build_knowledge_graph.apply(
                args=("doc-x", "sub-x", "prof-x")
            )
        except Exception:
            # Even if an internal path raises (e.g. missing field), we still
            # inspect what update_stage was called with up to that point.
            pass

    # --- Assertions ---

    # 1. update_stage was called at least once (IN_PROGRESS + COMPLETED).
    assert update_stage_calls, "update_stage was never called — check task patching"

    # 2. Every update_stage call must target the 'knowledge_graph' stage, NOT
    #    'pipeline_status'.
    for call in update_stage_calls:
        stage = call["stage"]
        assert stage != "pipeline_status", (
            f"KG task called update_stage with stage='pipeline_status' — "
            f"this violates the isolation contract. Full call: {call}"
        )
        assert stage.startswith("knowledge_graph") or stage == "knowledge_graph", (
            f"KG task called update_stage with unexpected stage={stage!r}. "
            f"Must stay within knowledge_graph strand. Full call: {call}"
        )

    # 3. Confirm we did see the IN_PROGRESS call (task actually ran).
    stages_seen = [c["stage"] for c in update_stage_calls]
    assert "knowledge_graph" in stages_seen, (
        f"Expected update_stage('knowledge_graph', ...) call but saw: {stages_seen}"
    )
