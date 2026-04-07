import pytest
from teams_app.storage.namespace import (
    qdrant_collection_name,
    redis_key,
    blob_prefix,
    mongo_collection,
)


def test_qdrant_collection_name():
    assert qdrant_collection_name("tenant_abc") == "teams_tenant_abc"

def test_qdrant_collection_sanitizes():
    assert qdrant_collection_name("tenant-abc!@#") == "teams_tenant_abc___"

def test_redis_key_session():
    assert redis_key("tenant_abc", "user_123", "session") == "teams:tenant_abc:user_123:session"

def test_redis_key_uploads():
    assert redis_key("tenant_abc", "user_123", "uploads") == "teams:tenant_abc:user_123:uploads"

def test_blob_prefix():
    assert blob_prefix("tenant_abc") == "teams/tenant_abc/"

def test_mongo_documents_collection():
    assert mongo_collection("documents") == "teams_documents"

def test_mongo_tenants_collection():
    assert mongo_collection("tenants") == "teams_tenants"

def test_mongo_users_collection():
    assert mongo_collection("users") == "teams_users"
