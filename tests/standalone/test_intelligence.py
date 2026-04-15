import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _make_app():
    from standalone.endpoints.intelligence import router
    from standalone.app import get_db, get_vllm_client

    app = FastAPI()
    app.include_router(router)

    mock_db = MagicMock()
    mock_keys = MagicMock()
    mock_keys.find_one = MagicMock(return_value={
        "key_hash": "abc", "active": True, "name": "test",
    })
    mock_db.__getitem__ = MagicMock(return_value=mock_keys)
    mock_logs = MagicMock()
    mock_logs.insert_one = MagicMock()
    mock_db.request_logs = mock_logs

    mock_vllm = MagicMock()
    mock_vllm.analyze = AsyncMock(return_value='{"summary": "This is an invoice.", "findings": ["Due date: April 30"], "evidence": ["Page 1: Payment terms"]}')

    app.dependency_overrides[get_db] = lambda: mock_db
    app.dependency_overrides[get_vllm_client] = lambda: mock_vllm

    return app, mock_vllm


def test_intelligence_success():
    app, _ = _make_app()
    client = TestClient(app)

    with patch("standalone.endpoints.intelligence.validate_api_key", new_callable=AsyncMock, return_value={"name": "test"}):
        response = client.post(
            "/api/v1/standalone/intelligence",
            headers={"X-Api-Key": "dw_sa_test"},
            files={"file": ("contract.txt", b"This contract is between A and B...", "text/plain")},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["analysis_type"] == "auto"
    assert "insights" in data
    assert "request_id" in data


def test_intelligence_with_analysis_type():
    app, _ = _make_app()
    client = TestClient(app)

    with patch("standalone.endpoints.intelligence.validate_api_key", new_callable=AsyncMock, return_value={"name": "test"}):
        response = client.post(
            "/api/v1/standalone/intelligence",
            headers={"X-Api-Key": "dw_sa_test"},
            data={"analysis_type": "risk_assessment"},
            files={"file": ("contract.txt", b"This contract includes a penalty clause...", "text/plain")},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["analysis_type"] == "risk_assessment"


def test_intelligence_with_prompt():
    app, mock_vllm = _make_app()
    client = TestClient(app)

    with patch("standalone.endpoints.intelligence.validate_api_key", new_callable=AsyncMock, return_value={"name": "test"}):
        response = client.post(
            "/api/v1/standalone/intelligence",
            headers={"X-Api-Key": "dw_sa_test"},
            data={"prompt": "focus on compliance risks"},
            files={"file": ("policy.txt", b"Company policy document...", "text/plain")},
        )

    assert response.status_code == 200
    call_args = mock_vllm.analyze.call_args
    assert "focus on compliance risks" in str(call_args)


def test_intelligence_no_api_key():
    app, _ = _make_app()
    client = TestClient(app)

    response = client.post(
        "/api/v1/standalone/intelligence",
        files={"file": ("test.txt", b"content", "text/plain")},
    )

    assert response.status_code == 401


def test_intelligence_invalid_analysis_type():
    app, _ = _make_app()
    client = TestClient(app)

    with patch("standalone.endpoints.intelligence.validate_api_key", new_callable=AsyncMock, return_value={"name": "test"}):
        response = client.post(
            "/api/v1/standalone/intelligence",
            headers={"X-Api-Key": "dw_sa_test"},
            data={"analysis_type": "magic"},
            files={"file": ("test.txt", b"content", "text/plain")},
        )

    assert response.status_code == 422
