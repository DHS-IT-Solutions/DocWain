"""Weekend Researcher refresh — enumerates profiles + dispatches run_researcher_agent."""
from unittest.mock import MagicMock


def test_weekly_refresh_dispatches_researcher_per_embedded_doc(monkeypatch):
    """For each doc whose pipeline_status is TRAINING_COMPLETED, dispatch run_researcher_agent.delay(...)."""
    from src.tasks import researcher_refresh as rr

    docs = [
        {"document_id": "d1", "subscription_id": "sub-a", "profile_id": "prof-a",
         "pipeline_status": "TRAINING_COMPLETED"},
        {"document_id": "d2", "subscription_id": "sub-a", "profile_id": "prof-a",
         "pipeline_status": "TRAINING_COMPLETED"},
        {"document_id": "d3", "subscription_id": "sub-b", "profile_id": "prof-b",
         "pipeline_status": "TRAINING_COMPLETED"},
        {"document_id": "d4", "subscription_id": "sub-a", "profile_id": "prof-a",
         "pipeline_status": "EMBEDDING_IN_PROGRESS"},
    ]

    class FakeCol:
        def find(self, filter, projection=None):
            if filter and "pipeline_status" in filter:
                status = filter["pipeline_status"]
                if isinstance(status, dict) and "$in" in status:
                    return iter([d for d in docs if d.get("pipeline_status") in status["$in"]])
                return iter([d for d in docs if d.get("pipeline_status") == status])
            return iter(docs)

    monkeypatch.setattr(rr, "_get_documents_collection", lambda: FakeCol(), raising=False)

    dispatched = []
    def fake_delay(document_id, subscription_id, profile_id):
        dispatched.append((document_id, subscription_id, profile_id))
        return MagicMock()

    monkeypatch.setattr("src.tasks.researcher.run_researcher_agent.delay", fake_delay)

    result = rr.researcher_weekly_refresh.apply().get()

    assert result["dispatched_count"] == 3
    assert sorted(dispatched) == sorted([
        ("d1", "sub-a", "prof-a"),
        ("d2", "sub-a", "prof-a"),
        ("d3", "sub-b", "prof-b"),
    ])


def test_weekly_refresh_returns_zero_when_disabled(monkeypatch):
    from src.tasks import researcher_refresh as rr
    from src.api import config as cfg_mod
    monkeypatch.setattr(cfg_mod.Config.Researcher, "ENABLED", False, raising=False)

    dispatched = []
    monkeypatch.setattr("src.tasks.researcher.run_researcher_agent.delay",
                        lambda *a, **kw: dispatched.append(a) or MagicMock())

    result = rr.researcher_weekly_refresh.apply().get()
    assert result.get("skipped") is True
    assert result.get("dispatched_count", 0) == 0
    assert not dispatched


def test_weekly_refresh_swallows_individual_dispatch_failures(monkeypatch):
    from src.tasks import researcher_refresh as rr

    docs = [
        {"document_id": f"d{i}", "subscription_id": "sub", "profile_id": "prof",
         "pipeline_status": "TRAINING_COMPLETED"}
        for i in range(3)
    ]
    class FakeCol:
        def find(self, filter, projection=None):
            return iter(docs)
    monkeypatch.setattr(rr, "_get_documents_collection", lambda: FakeCol(), raising=False)

    dispatched = []
    def flaky_delay(document_id, subscription_id, profile_id):
        if document_id == "d1":
            raise RuntimeError("broker full")
        dispatched.append(document_id)
        return MagicMock()
    monkeypatch.setattr("src.tasks.researcher.run_researcher_agent.delay", flaky_delay)

    result = rr.researcher_weekly_refresh.apply().get()
    assert result["dispatched_count"] == 2
    assert result["failed_count"] == 1
    assert sorted(dispatched) == ["d0", "d2"]
