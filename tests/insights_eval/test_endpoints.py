from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def app_with_dashboard_flag(monkeypatch):
    monkeypatch.setenv("INSIGHTS_DASHBOARD_ENABLED", "true")
    from src.main import app
    return TestClient(app)


@pytest.fixture
def app_with_no_flags(monkeypatch):
    from src.api.feature_flags import FLAG_NAMES
    for name in FLAG_NAMES:
        monkeypatch.delenv(name, raising=False)
    from src.main import app
    return TestClient(app)


def test_insights_list_404_when_flag_off(app_with_no_flags):
    r = app_with_no_flags.get("/api/profiles/v2/P1/insights")
    assert r.status_code == 404


def test_insights_list_returns_when_flag_on(app_with_dashboard_flag):
    rows = [
        {"insight_id": "i1", "profile_id": "P1", "insight_type": "anomaly",
         "severity": "warn", "domain": "insurance", "refreshed_at": "2026-04-25",
         "stale": False, "tags": []},
    ]
    with patch("src.api.insights_api.list_insights_for_profile", return_value=rows):
        r = app_with_dashboard_flag.get("/api/profiles/v2/P1/insights")
    assert r.status_code == 200
    body = r.json()
    assert body["profile_id"] == "P1"
    assert body["total"] == 1
    assert body["insights"][0]["insight_id"] == "i1"


def test_insights_detail_returns_full_object(app_with_dashboard_flag):
    full = {
        "insight_id": "i1", "profile_id": "P1", "subscription_id": "S",
        "document_ids": ["D1"], "domain": "insurance", "insight_type": "gap",
        "headline": "h", "body": "b",
        "evidence_doc_spans": [{"document_id": "D1", "page": 1, "char_start": 0, "char_end": 1, "quote": "x"}],
        "external_kb_refs": [], "confidence": 0.9, "severity": "warn",
        "adapter_version": "insurance@1.0", "refreshed_at": "2026-04-25",
        "stale": False, "tags": [], "feature_flags": [], "suggested_actions": [],
        "created_at": "2026-04-25",
    }
    with patch("src.api.insights_api.get_insight_full", return_value=full):
        r = app_with_dashboard_flag.get("/api/profiles/v2/P1/insights/i1")
    assert r.status_code == 200
    body = r.json()
    assert body["headline"] == "h"
    assert body["evidence_doc_spans"][0]["document_id"] == "D1"


def test_actions_list_404_when_flag_off(app_with_no_flags):
    r = app_with_no_flags.get("/api/profiles/v2/P1/actions")
    assert r.status_code == 404


def test_actions_list_returns_when_flag_on(monkeypatch):
    monkeypatch.setenv("ACTIONS_ARTIFACT_ENABLED", "true")
    from src.main import app
    client = TestClient(app)
    actions = [{
        "action_id": "a1", "title": "Generate summary",
        "action_type": "artifact", "requires_confirmation": False,
        "preview": "Will produce a coverage summary PDF",
    }]
    with patch("src.api.actions_api.list_actions_for_profile", return_value=actions):
        r = client.get("/api/profiles/v2/P1/actions")
    assert r.status_code == 200
    body = r.json()
    assert body["actions"][0]["action_id"] == "a1"


def test_action_execute_returns_preview_when_unconfirmed(monkeypatch):
    monkeypatch.setenv("ACTIONS_ARTIFACT_ENABLED", "true")
    from src.main import app
    client = TestClient(app)

    def fake_execute(**kwargs):
        return {"status": "needs_confirmation", "preview": "preview text"}

    with patch("src.api.actions_api.execute_action", side_effect=fake_execute):
        r = client.post("/api/profiles/v2/P1/actions/a1/execute", json={"inputs": {}, "confirmed": False})
    assert r.status_code == 200
    assert r.json()["status"] == "needs_confirmation"


def test_viz_endpoint_gated_by_flag(app_with_no_flags):
    r = app_with_no_flags.get("/api/profiles/v2/P1/visualizations")
    assert r.status_code == 404


def test_viz_endpoint_returns_when_flag_on(monkeypatch):
    monkeypatch.setenv("VIZ_ENABLED", "true")
    from src.main import app
    client = TestClient(app)
    vizs = [{"viz_id": "timeline", "type": "timeline", "data": {"events": []}}]
    with patch("src.api.visualizations_api.list_visualizations_for_profile", return_value=vizs):
        r = client.get("/api/profiles/v2/P1/visualizations")
    assert r.status_code == 200
    assert r.json()["visualizations"][0]["viz_id"] == "timeline"


def test_artifacts_endpoint_gated(app_with_no_flags):
    r = app_with_no_flags.get("/api/profiles/v2/P1/artifacts")
    assert r.status_code == 404
