"""trigger_embedding dispatches BOTH embed_document AND build_knowledge_graph.

Verifies:
- Both tasks are enqueued on HITL approval.
- KG dispatch failure is swallowed (logged, not raised) so embedding still dispatches.
"""
import asyncio
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def fake_doc_in_screening_completed():
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

    def fake_embed_delay(*a, **kw):
        embed_calls.append((a, kw))
        return MagicMock(id="embed-task-id")

    def fake_kg_delay(*a, **kw):
        kg_calls.append((a, kw))
        return MagicMock(id="kg-task-id")

    monkeypatch.setattr("src.tasks.embedding.embed_document.delay", fake_embed_delay)
    monkeypatch.setattr("src.tasks.kg.build_knowledge_graph.delay", fake_kg_delay)

    fake_collection = MagicMock()
    fake_collection.find_one.return_value = fake_doc_in_screening_completed
    # The pipeline_api module fetches the Mongo collection via some helper — try
    # multiple common names so this test survives minor refactors.
    for attr in ("get_documents_collection", "documents_collection", "get_mongo_collection"):
        monkeypatch.setattr(pipeline_api, attr, lambda: fake_collection, raising=False)

    # patch get_document_record directly since pipeline_api uses it
    monkeypatch.setattr(pipeline_api, "get_document_record", lambda doc_id: fake_doc_in_screening_completed)
    # patch append_audit_log to avoid real MongoDB calls
    monkeypatch.setattr(pipeline_api, "append_audit_log", lambda *a, **kw: None)

    try:
        asyncio.run(pipeline_api.trigger_embedding(document_id="doc-abc"))
    except Exception as exc:
        # If the endpoint has additional required params (e.g., background_tasks),
        # fail explicitly so the implementer sees the signature to match.
        pytest.fail(f"trigger_embedding raised: {exc!r} — adapt the test call to its actual signature")

    assert len(embed_calls) == 1, f"embed_document.delay not called exactly once: {embed_calls}"
    assert len(kg_calls) == 1, f"build_knowledge_graph.delay not called exactly once: {kg_calls}"


def test_trigger_embedding_tolerates_kg_dispatch_failure(fake_doc_in_screening_completed, monkeypatch):
    """If KG dispatch raises, embedding dispatch still happens and endpoint returns success."""
    from src.api import pipeline_api

    embed_calls = []

    def fake_embed_delay(*a, **kw):
        embed_calls.append((a, kw))
        return MagicMock(id="embed-task-id")

    def fake_kg_delay(*a, **kw):
        raise RuntimeError("redis unreachable")

    monkeypatch.setattr("src.tasks.embedding.embed_document.delay", fake_embed_delay)
    monkeypatch.setattr("src.tasks.kg.build_knowledge_graph.delay", fake_kg_delay)

    fake_collection = MagicMock()
    fake_collection.find_one.return_value = fake_doc_in_screening_completed
    for attr in ("get_documents_collection", "documents_collection", "get_mongo_collection"):
        monkeypatch.setattr(pipeline_api, attr, lambda: fake_collection, raising=False)

    # patch get_document_record directly since pipeline_api uses it
    monkeypatch.setattr(pipeline_api, "get_document_record", lambda doc_id: fake_doc_in_screening_completed)
    # patch append_audit_log to avoid real MongoDB calls
    monkeypatch.setattr(pipeline_api, "append_audit_log", lambda *a, **kw: None)

    # Must NOT raise — KG failure is swallowed
    try:
        asyncio.run(pipeline_api.trigger_embedding(document_id="doc-abc"))
    except Exception as exc:
        pytest.fail(f"trigger_embedding raised on KG dispatch failure: {exc!r}")

    # Embedding still dispatched
    assert len(embed_calls) == 1
