"""
Smoke tests that verify the full standalone app works end-to-end
with mocked vLLM responses.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient


@pytest.fixture
def full_app():
    """Create the full standalone app with mocked dependencies."""
    # Mock MongoDB
    mock_db = MagicMock()
    mock_keys = MagicMock()
    mock_keys.find_one = MagicMock(return_value={
        "key_hash": "abc", "active": True, "name": "test",
    })
    mock_keys.insert_one = MagicMock()
    mock_keys.find = MagicMock(return_value=[])
    mock_keys.update_one = MagicMock(return_value=MagicMock(modified_count=1))
    mock_db.api_keys = mock_keys

    mock_logs = MagicMock()
    mock_logs.insert_one = MagicMock()
    mock_db.request_logs = mock_logs
    mock_db.__getitem__ = MagicMock(return_value=mock_keys)

    # Mock vLLM
    mock_vllm = MagicMock()
    mock_vllm.extract = AsyncMock(return_value='{"document_type": "report", "sections": [{"title": "Summary", "content": "Test content"}]}')
    mock_vllm.analyze = AsyncMock(return_value='{"summary": "Test report summary", "findings": ["Finding 1"], "evidence": ["Page 1"]}')
    mock_vllm.health_check = AsyncMock(return_value=True)

    from standalone.app import app
    from standalone.dependencies import get_db, get_vllm_client
    app.dependency_overrides[get_db] = lambda: mock_db
    app.dependency_overrides[get_vllm_client] = lambda: mock_vllm

    yield TestClient(app), mock_vllm, mock_db

    app.dependency_overrides.clear()


def test_health_check(full_app):
    client, _, _ = full_app
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_full_extract_flow(full_app):
    client, _, _ = full_app

    with patch("standalone.endpoints.extract.validate_api_key", new_callable=AsyncMock, return_value={"name": "test"}):
        response = client.post(
            "/api/v1/standalone/extract",
            headers={"X-Api-Key": "dw_sa_testkey123"},
            data={"output_format": "json"},
            files={"file": ("report.txt", b"Quarterly report with revenue data...", "text/plain")},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["output_format"] == "json"
    assert data["content"]["document_type"] == "report"


def test_full_intelligence_flow(full_app):
    client, _, _ = full_app

    with patch("standalone.endpoints.intelligence.validate_api_key", new_callable=AsyncMock, return_value={"name": "test"}):
        response = client.post(
            "/api/v1/standalone/intelligence",
            headers={"X-Api-Key": "dw_sa_testkey123"},
            data={"analysis_type": "summary"},
            files={"file": ("report.txt", b"Quarterly report content...", "text/plain")},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["analysis_type"] == "summary"
    assert "summary" in data["insights"]


def test_full_key_lifecycle(full_app):
    client, _, mock_db = full_app

    with patch("standalone.endpoints.keys.Config") as mock_config:
        mock_config.ADMIN_SECRET = "admin-secret"

        # Create key
        response = client.post(
            "/admin/keys",
            headers={"X-Admin-Secret": "admin-secret"},
            json={"name": "new-key"},
        )
        assert response.status_code == 201
        key_data = response.json()
        assert key_data["raw_key"].startswith("dw_sa_")

        # List keys
        response = client.get(
            "/admin/keys",
            headers={"X-Admin-Secret": "admin-secret"},
        )
        assert response.status_code == 200

        # Delete key
        response = client.delete(
            f"/admin/keys/{key_data['key_id']}",
            headers={"X-Admin-Secret": "admin-secret"},
        )
        assert response.status_code == 200


def test_extract_then_intelligence_same_file(full_app):
    """Verify both endpoints can process the same file independently."""
    client, _, _ = full_app
    file_content = b"Contract between Company A and Company B for $1M services."

    with patch("standalone.endpoints.extract.validate_api_key", new_callable=AsyncMock, return_value={"name": "test"}):
        extract_resp = client.post(
            "/api/v1/standalone/extract",
            headers={"X-Api-Key": "dw_sa_testkey123"},
            data={"output_format": "json"},
            files={"file": ("contract.txt", file_content, "text/plain")},
        )

    with patch("standalone.endpoints.intelligence.validate_api_key", new_callable=AsyncMock, return_value={"name": "test"}):
        intel_resp = client.post(
            "/api/v1/standalone/intelligence",
            headers={"X-Api-Key": "dw_sa_testkey123"},
            data={"analysis_type": "risk_assessment"},
            files={"file": ("contract.txt", file_content, "text/plain")},
        )

    assert extract_resp.status_code == 200
    assert intel_resp.status_code == 200
    # Both should have different request_ids
    assert extract_resp.json()["request_id"] != intel_resp.json()["request_id"]
