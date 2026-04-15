import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _make_app():
    from standalone.endpoints.extract import router
    from standalone.dependencies import get_db, get_vllm_client

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
    mock_vllm.extract = AsyncMock(return_value='{"document_type": "invoice", "total": 500}')

    app.dependency_overrides[get_db] = lambda: mock_db
    app.dependency_overrides[get_vllm_client] = lambda: mock_vllm

    return app, mock_vllm


def test_extract_success():
    app, mock_vllm = _make_app()
    client = TestClient(app)

    with patch("standalone.endpoints.extract.validate_api_key", new_callable=AsyncMock, return_value={"name": "test"}):
        response = client.post(
            "/api/v1/standalone/extract",
            headers={"X-Api-Key": "dw_sa_test"},
            data={"output_format": "json"},
            files={"file": ("test.txt", b"Invoice total: $500", "text/plain")},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["output_format"] == "json"
    assert "request_id" in data
    assert "content" in data
    assert "metadata" in data


def test_extract_missing_file():
    app, _ = _make_app()
    client = TestClient(app)

    with patch("standalone.endpoints.extract.validate_api_key", new_callable=AsyncMock, return_value={"name": "test"}):
        response = client.post(
            "/api/v1/standalone/extract",
            headers={"X-Api-Key": "dw_sa_test"},
            data={"output_format": "json"},
        )

    assert response.status_code == 422


def test_extract_invalid_format():
    app, _ = _make_app()
    client = TestClient(app)

    with patch("standalone.endpoints.extract.validate_api_key", new_callable=AsyncMock, return_value={"name": "test"}):
        response = client.post(
            "/api/v1/standalone/extract",
            headers={"X-Api-Key": "dw_sa_test"},
            data={"output_format": "xml"},
            files={"file": ("test.txt", b"content", "text/plain")},
        )

    assert response.status_code == 422


def test_extract_no_api_key():
    app, _ = _make_app()
    client = TestClient(app)

    response = client.post(
        "/api/v1/standalone/extract",
        data={"output_format": "json"},
        files={"file": ("test.txt", b"content", "text/plain")},
    )

    assert response.status_code == 401


def test_extract_csv_format():
    app, mock_vllm = _make_app()
    mock_vllm.extract = AsyncMock(
        return_value='{"tables": [{"headers": ["Item", "Price"], "rows": [["Widget", "10"]]}]}'
    )
    client = TestClient(app)

    with patch("standalone.endpoints.extract.validate_api_key", new_callable=AsyncMock, return_value={"name": "test"}):
        response = client.post(
            "/api/v1/standalone/extract",
            headers={"X-Api-Key": "dw_sa_test"},
            data={"output_format": "csv"},
            files={"file": ("data.txt", b"Item: Widget, Price: 10", "text/plain")},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["output_format"] == "csv"
