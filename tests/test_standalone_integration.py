"""Integration tests for the Standalone API router using FastAPI TestClient."""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


@pytest.fixture
def api_key_doc():
    return {
        "key_hash": "testhash",
        "name": "Test Key",
        "subscription_id": "test-sub",
        "active": True,
        "permissions": ["process", "extract", "batch", "query"],
        "usage": {"total_requests": 0, "last_used": None},
    }


@pytest.fixture
def client(api_key_doc):
    """Create a test client with mocked auth."""
    from src.api.standalone_api import standalone_router
    from src.api.standalone_auth import require_api_key
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(standalone_router)

    async def mock_auth():
        return {
            "key_hash": api_key_doc["key_hash"],
            "name": api_key_doc["name"],
            "subscription_id": api_key_doc["subscription_id"],
            "permissions": api_key_doc["permissions"],
            "keys_collection": MagicMock(),
        }

    app.dependency_overrides[require_api_key] = mock_auth
    return TestClient(app)


def test_templates_endpoint(client):
    """GET /v1/docwain/templates returns 200 with >= 6 templates."""
    response = client.get("/v1/docwain/templates")
    assert response.status_code == 200
    data = response.json()
    assert "templates" in data
    assert isinstance(data["templates"], list)
    assert len(data["templates"]) >= 6
    # Each entry should have name, description, modes
    for tmpl in data["templates"]:
        assert "name" in tmpl
        assert "description" in tmpl
        assert "modes" in tmpl


def test_process_endpoint(client):
    """POST /v1/docwain/process returns 200 with expected response shape."""
    mock_result = {
        "request_id": "test-req-001",
        "status": "completed",
        "answer": "This is a test answer.",
        "sources": [],
        "confidence": 0.9,
        "grounded": True,
        "low_confidence": False,
        "low_confidence_reasons": [],
        "structured_output": None,
        "document_id": None,
        "output_format": "json",
        "partial_answer": None,
        "usage": {
            "extraction_ms": 10,
            "intelligence_ms": 20,
            "retrieval_ms": 5,
            "generation_ms": 50,
            "total_ms": 85,
        },
    }

    with patch("src.api.standalone_processor.process_document", return_value=mock_result) as mock_proc, \
         patch("src.api.standalone_api._log_request") as mock_log:
        response = client.post(
            "/v1/docwain/process",
            data={"prompt": "What is this document about?"},
            files={"file": ("test.txt", b"Hello, this is a test document.", "text/plain")},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["request_id"] == "test-req-001"
    assert data["status"] == "completed"
    assert "answer" in data
    assert "confidence" in data
    mock_log.assert_called_once()


def test_process_requires_auth():
    """POST /v1/docwain/process without auth override returns 401."""
    from src.api.standalone_api import standalone_router
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(standalone_router)

    unauthenticated_client = TestClient(app, raise_server_exceptions=False)

    response = unauthenticated_client.post(
        "/v1/docwain/process",
        data={"prompt": "What is this document about?"},
        files={"file": ("test.txt", b"Hello.", "text/plain")},
    )
    assert response.status_code == 401


def test_extract_endpoint_requires_mode(client):
    """POST /v1/docwain/extract without mode field returns 422."""
    response = client.post(
        "/v1/docwain/extract",
        files={"file": ("test.txt", b"Hello, this is a test document.", "text/plain")},
    )
    assert response.status_code == 422
