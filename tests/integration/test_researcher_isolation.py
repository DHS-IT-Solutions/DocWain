"""run_researcher_agent never touches pipeline_status; only researcher.* strand."""
from unittest.mock import MagicMock


def test_researcher_task_does_not_touch_pipeline_status(monkeypatch):
    import src.tasks.researcher as r_mod

    seen_updates = []

    class FakeCol:
        def update_one(self, filter, update, **kw):
            seen_updates.append(update)
            return MagicMock(matched_count=1, modified_count=1)

    def fake_get_mongo_collection(name):
        return FakeCol()

    # Patch the Mongo helper the researcher uses.
    try:
        monkeypatch.setattr("src.api.dw_newron.get_mongo_collection",
                             fake_get_mongo_collection, raising=False)
    except Exception:
        pass

    # Stub the heavy lifts so the task runs fully.
    monkeypatch.setattr(r_mod, "_load_extraction",
                         lambda *a, **kw: {"format": "docx", "pages": [{"page_num": 1, "blocks": [{"text": "hi"}]}]},
                         raising=False)
    monkeypatch.setattr(r_mod, "_call_docwain_for_insights",
                         lambda *a, **kw: __import__("src.docwain.prompts.researcher",
                                                     fromlist=["ResearcherInsights"]).ResearcherInsights(
                             summary="s", confidence=0.5
                         ),
                         raising=False)
    monkeypatch.setattr(r_mod, "_write_insights_to_qdrant", lambda *a, **kw: None, raising=False)
    monkeypatch.setattr(r_mod, "_write_insight_to_neo4j", lambda *a, **kw: None, raising=False)

    # Run synchronously via .apply (Celery bind=True pattern).
    try:
        r_mod.run_researcher_agent.apply(args=("doc-iso", "sub-iso", "prof-iso"))
    except Exception:
        pass

    # No captured update should set pipeline_status or stages.* — only researcher.*
    forbidden_prefixes = ("pipeline_status", "stages.", "knowledge_graph")
    for upd in seen_updates:
        if not isinstance(upd, dict):
            continue
        for op, body in upd.items():
            if isinstance(body, dict):
                for k in body.keys():
                    for forbidden in forbidden_prefixes:
                        assert not k.startswith(forbidden), f"researcher wrote forbidden field {k!r}"
