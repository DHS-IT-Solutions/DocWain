"""Tests for standalone API key authentication and usage tracking."""
import hashlib
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# hash_api_key
# ---------------------------------------------------------------------------

def test_hash_api_key_deterministic():
    from src.api.standalone_auth import hash_api_key

    key = "dw_abc123"
    result1 = hash_api_key(key)
    result2 = hash_api_key(key)

    assert result1 == result2, "hash_api_key must be deterministic"
    assert len(result1) == 64, "SHA-256 hex digest must be 64 characters"
    # Verify it's actually SHA-256
    expected = hashlib.sha256(key.encode()).hexdigest()
    assert result1 == expected


# ---------------------------------------------------------------------------
# generate_api_key
# ---------------------------------------------------------------------------

def test_generate_api_key_format():
    from src.api.standalone_auth import generate_api_key

    raw_key, key_hash = generate_api_key()

    assert raw_key.startswith("dw_"), "Key must start with 'dw_'"
    assert len(raw_key) == 51, f"Key length must be 51 (dw_ + 48 hex chars), got {len(raw_key)}"
    assert len(key_hash) == 64, f"Hash length must be 64, got {len(key_hash)}"

    # Hash must match
    from src.api.standalone_auth import hash_api_key
    assert key_hash == hash_api_key(raw_key)


# ---------------------------------------------------------------------------
# validate_api_key_sync
# ---------------------------------------------------------------------------

def test_validate_api_key_success():
    from src.api.standalone_auth import validate_api_key_sync, hash_api_key

    raw_key = "dw_" + "a" * 48
    key_hash = hash_api_key(raw_key)

    mock_doc = {"key_hash": key_hash, "active": True, "name": "Test Key"}
    mock_collection = MagicMock()
    mock_collection.find_one.return_value = mock_doc

    result = validate_api_key_sync(raw_key, mock_collection)

    assert result == mock_doc
    mock_collection.find_one.assert_called_once_with({"key_hash": key_hash})


def test_validate_api_key_inactive():
    from src.api.standalone_auth import validate_api_key_sync, hash_api_key

    raw_key = "dw_" + "b" * 48
    key_hash = hash_api_key(raw_key)

    mock_doc = {"key_hash": key_hash, "active": False, "name": "Inactive Key"}
    mock_collection = MagicMock()
    mock_collection.find_one.return_value = mock_doc

    result = validate_api_key_sync(raw_key, mock_collection)

    assert result is None


def test_validate_api_key_not_found():
    from src.api.standalone_auth import validate_api_key_sync

    raw_key = "dw_" + "c" * 48
    mock_collection = MagicMock()
    mock_collection.find_one.return_value = None

    result = validate_api_key_sync(raw_key, mock_collection)

    assert result is None


# ---------------------------------------------------------------------------
# track_usage
# ---------------------------------------------------------------------------

def test_track_usage_increments():
    from src.api.standalone_auth import track_usage

    mock_collection = MagicMock()
    key_hash = "abc123hash"
    endpoint = "/api/ask"
    mode = "rag"

    track_usage(mock_collection, key_hash, endpoint, mode)

    mock_collection.update_one.assert_called_once()
    call_args = mock_collection.update_one.call_args
    filter_doc, update_doc = call_args[0]

    assert filter_doc == {"key_hash": key_hash}
    assert "$inc" in update_doc
    assert "$set" in update_doc

    inc_fields = update_doc["$inc"]
    assert inc_fields.get("total_requests") == 1
    assert inc_fields.get("requests_today") == 1
    assert inc_fields.get(f"by_endpoint.{endpoint}") == 1
    assert inc_fields.get(f"by_mode.{mode}") == 1

    set_fields = update_doc["$set"]
    assert "last_used" in set_fields


def test_track_usage_swallows_exceptions():
    """track_usage must not propagate exceptions (fire-and-forget)."""
    from src.api.standalone_auth import track_usage

    mock_collection = MagicMock()
    mock_collection.update_one.side_effect = Exception("DB is down")

    # Should not raise
    track_usage(mock_collection, "hash", "/endpoint", "mode")


# ---------------------------------------------------------------------------
# track_document_processed
# ---------------------------------------------------------------------------

def test_track_document_processed():
    from src.api.standalone_auth import track_document_processed

    mock_collection = MagicMock()
    key_hash = "testhash"

    track_document_processed(mock_collection, key_hash)

    mock_collection.update_one.assert_called_once()
    call_args = mock_collection.update_one.call_args
    filter_doc, update_doc = call_args[0]

    assert filter_doc == {"key_hash": key_hash}
    assert update_doc["$inc"].get("documents_processed") == 1
