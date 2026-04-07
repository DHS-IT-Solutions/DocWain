import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from teams_app.storage.tenant import TenantManager


@pytest.fixture
def mock_mongo():
    db = MagicMock()
    db.teams_tenants = MagicMock()
    db.teams_users = MagicMock()
    return db

@pytest.fixture
def mock_qdrant():
    return MagicMock()

@pytest.fixture
def manager(mock_mongo, mock_qdrant):
    return TenantManager(db=mock_mongo, qdrant_client=mock_qdrant)

def test_extract_identity_from_activity():
    activity = {
        "channelData": {"tenant": {"id": "tenant_abc"}},
        "from": {"id": "user_123", "name": "Alice"},
    }
    tenant_id, user_id, display_name = TenantManager.extract_identity(activity)
    assert tenant_id == "tenant_abc"
    assert user_id == "user_123"
    assert display_name == "Alice"

def test_extract_identity_missing_tenant():
    activity = {"from": {"id": "user_123"}}
    tenant_id, user_id, _ = TenantManager.extract_identity(activity)
    assert tenant_id is None
    assert user_id == "user_123"

def test_ensure_tenant_existing(manager, mock_mongo):
    mock_mongo.teams_tenants.find_one.return_value = {"tenant_id": "t1", "qdrant_collection": "teams_t1"}
    result = manager.ensure_tenant("t1", "Contoso")
    assert result["tenant_id"] == "t1"
    mock_mongo.teams_tenants.insert_one.assert_not_called()

def test_ensure_tenant_new_creates_record(manager, mock_mongo, mock_qdrant):
    mock_mongo.teams_tenants.find_one.return_value = None
    result = manager.ensure_tenant("t_new", "NewCorp")
    assert result["tenant_id"] == "t_new"
    assert result["qdrant_collection"] == "teams_t_new"
    mock_mongo.teams_tenants.insert_one.assert_called_once()
    mock_qdrant.create_collection.assert_called_once()

def test_ensure_user_existing(manager, mock_mongo):
    mock_mongo.teams_users.find_one.return_value = {"user_id": "u1"}
    result = manager.ensure_user("u1", "t1", "Alice")
    assert result["user_id"] == "u1"
    mock_mongo.teams_users.insert_one.assert_not_called()

def test_ensure_user_new_creates_record(manager, mock_mongo):
    mock_mongo.teams_users.find_one.return_value = None
    result = manager.ensure_user("u_new", "t1", "Bob")
    assert result["user_id"] == "u_new"
    mock_mongo.teams_users.insert_one.assert_called_once()
