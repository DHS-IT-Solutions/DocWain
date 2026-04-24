"""trigger_embedding dispatches embed + KG + researcher when Researcher is enabled."""
import asyncio
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def fake_doc_in_screening_completed():
    from src.api.statuses import PIPELINE_SCREENING_COMPLETED
    return {"_id": "doc-r1", "subscription_id": "sub-r", "profile_id": "prof-r",
            "pipeline_status": PIPELINE_SCREENING_COMPLETED}


def test_dispatches_all_three_when_researcher_enabled(fake_doc_in_screening_completed, monkeypatch):
    from src.api import config as cfg_mod
    monkeypatch.setattr(cfg_mod.Config.Researcher, "ENABLED", True, raising=False)

    from src.api import pipeline_api

    counts = {"embed": 0, "kg": 0, "researcher": 0}
    monkeypatch.setattr("src.tasks.embedding.embed_document.delay",
                        lambda *a, **kw: (counts.__setitem__("embed", counts["embed"] + 1) or MagicMock()))
    monkeypatch.setattr("src.tasks.kg.build_knowledge_graph.delay",
                        lambda *a, **kw: (counts.__setitem__("kg", counts["kg"] + 1) or MagicMock()))
    monkeypatch.setattr("src.tasks.researcher.run_researcher_agent.delay",
                        lambda *a, **kw: (counts.__setitem__("researcher", counts["researcher"] + 1) or MagicMock()))

    for attr in ("get_document_record", "get_documents_collection"):
        monkeypatch.setattr(pipeline_api, attr, lambda *a, **kw: fake_doc_in_screening_completed, raising=False)
    monkeypatch.setattr(pipeline_api, "append_audit_log", lambda *a, **kw: None, raising=False)

    try:
        asyncio.run(pipeline_api.trigger_embedding(document_id="doc-r1"))
    except Exception as exc:
        pytest.fail(f"trigger_embedding raised: {exc!r}")

    assert counts == {"embed": 1, "kg": 1, "researcher": 1}


def test_does_not_dispatch_researcher_when_disabled(fake_doc_in_screening_completed, monkeypatch):
    from src.api import config as cfg_mod
    monkeypatch.setattr(cfg_mod.Config.Researcher, "ENABLED", False, raising=False)

    from src.api import pipeline_api

    counts = {"embed": 0, "kg": 0, "researcher": 0}
    monkeypatch.setattr("src.tasks.embedding.embed_document.delay",
                        lambda *a, **kw: (counts.__setitem__("embed", counts["embed"] + 1) or MagicMock()))
    monkeypatch.setattr("src.tasks.kg.build_knowledge_graph.delay",
                        lambda *a, **kw: (counts.__setitem__("kg", counts["kg"] + 1) or MagicMock()))
    monkeypatch.setattr("src.tasks.researcher.run_researcher_agent.delay",
                        lambda *a, **kw: (counts.__setitem__("researcher", counts["researcher"] + 1) or MagicMock()))

    for attr in ("get_document_record", "get_documents_collection"):
        monkeypatch.setattr(pipeline_api, attr, lambda *a, **kw: fake_doc_in_screening_completed, raising=False)
    monkeypatch.setattr(pipeline_api, "append_audit_log", lambda *a, **kw: None, raising=False)

    asyncio.run(pipeline_api.trigger_embedding(document_id="doc-r1"))

    # embed + kg still dispatch; researcher suppressed by flag
    assert counts["embed"] == 1
    assert counts["kg"] == 1
    assert counts["researcher"] == 0
