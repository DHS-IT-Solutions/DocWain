import pytest
from unittest.mock import MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _make_app():
    from standalone.endpoints.keys import router
    from standalone.app import get_db

    app = FastAPI()
    app.include_router(router)

    mock_db = MagicMock()
    mock_keys = MagicMock()
    mock_keys.insert_one = MagicMock()
    mock_keys.find = MagicMock(return_value=[
        {
            "_id": "k-1",
            "key_prefix": "dw_sa_abc",
            "name": "production",
            "created_at": "2026-04-15T00:00:00Z",
            "total_requests": 100,
            "active": True,
        }
    ])
    mock_keys.update_one = MagicMock()
    mock_db.api_keys = mock_keys

    app.dependency_overrides[get_db] = lambda: mock_db

    return app, mock_keys


def test_create_key_success():
    app, mock_keys = _make_app()
    client = TestClient(app)

    with patch("standalone.endpoints.keys.Config") as mock_config:
        mock_config.ADMIN_SECRET = "test-secret"
        response = client.post(
            "/admin/keys",
            headers={"X-Admin-Secret": "test-secret"},
            json={"name": "my-new-key"},
        )

    assert response.status_code == 201
    data = response.json()
    assert data["raw_key"].startswith("dw_sa_")
    assert data["name"] == "my-new-key"
    assert "key_id" in data
    mock_keys.insert_one.assert_called_once()


def test_create_key_no_admin_secret():
    app, _ = _make_app()
    client = TestClient(app)

    response = client.post(
        "/admin/keys",
        json={"name": "my-key"},
    )

    assert response.status_code == 401


def test_create_key_wrong_admin_secret():
    app, _ = _make_app()
    client = TestClient(app)

    with patch("standalone.endpoints.keys.Config") as mock_config:
        mock_config.ADMIN_SECRET = "correct-secret"
        response = client.post(
            "/admin/keys",
            headers={"X-Admin-Secret": "wrong-secret"},
            json={"name": "my-key"},
        )

    assert response.status_code == 401


def test_list_keys():
    app, _ = _make_app()
    client = TestClient(app)

    with patch("standalone.endpoints.keys.Config") as mock_config:
        mock_config.ADMIN_SECRET = "test-secret"
        response = client.get(
            "/admin/keys",
            headers={"X-Admin-Secret": "test-secret"},
        )

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["name"] == "production"
    # Raw key should never appear in list
    assert "raw_key" not in data[0]
    assert "key_hash" not in data[0]


def test_delete_key():
    app, mock_keys = _make_app()
    mock_keys.update_one = MagicMock(return_value=MagicMock(modified_count=1))
    client = TestClient(app)

    with patch("standalone.endpoints.keys.Config") as mock_config:
        mock_config.ADMIN_SECRET = "test-secret"
        response = client.delete(
            "/admin/keys/k-123",
            headers={"X-Admin-Secret": "test-secret"},
        )

    assert response.status_code == 200
    mock_keys.update_one.assert_called_once()


def test_delete_key_not_found():
    app, mock_keys = _make_app()
    mock_keys.update_one = MagicMock(return_value=MagicMock(modified_count=0))
    client = TestClient(app)

    with patch("standalone.endpoints.keys.Config") as mock_config:
        mock_config.ADMIN_SECRET = "test-secret"
        response = client.delete(
            "/admin/keys/nonexistent",
            headers={"X-Admin-Secret": "test-secret"},
        )

    assert response.status_code == 404
