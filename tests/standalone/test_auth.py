import hashlib
import secrets
import pytest
from unittest.mock import MagicMock, AsyncMock
from fastapi import HTTPException


def test_hash_api_key_deterministic():
    from standalone.auth import hash_api_key

    key = "dw_sa_" + "a" * 48
    h1 = hash_api_key(key)
    h2 = hash_api_key(key)
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex digest


def test_hash_api_key_different_keys_differ():
    from standalone.auth import hash_api_key

    h1 = hash_api_key("dw_sa_" + "a" * 48)
    h2 = hash_api_key("dw_sa_" + "b" * 48)
    assert h1 != h2


def test_generate_api_key_format():
    from standalone.auth import generate_api_key

    key = generate_api_key()
    assert key.startswith("dw_sa_")
    assert len(key) == 6 + 48  # prefix + 48 hex chars


def test_validate_api_key_success():
    from standalone.auth import validate_api_key

    raw_key = "dw_sa_" + "a" * 48
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

    mock_collection = MagicMock()
    mock_collection.find_one = MagicMock(return_value={
        "key_hash": key_hash, "active": True, "name": "Test",
    })

    result = validate_api_key(raw_key, mock_collection)
    assert result["name"] == "Test"
    mock_collection.find_one.assert_called_once_with({"key_hash": key_hash, "active": True})


def test_validate_api_key_not_found():
    from standalone.auth import validate_api_key

    mock_collection = MagicMock()
    mock_collection.find_one = MagicMock(return_value=None)

    result = validate_api_key("dw_sa_bad", mock_collection)
    assert result is None


def test_verify_admin_secret_success():
    from standalone.auth import verify_admin_secret

    assert verify_admin_secret("correct-secret", "correct-secret") is True


def test_verify_admin_secret_failure():
    from standalone.auth import verify_admin_secret

    assert verify_admin_secret("wrong", "correct-secret") is False


def test_verify_admin_secret_empty():
    from standalone.auth import verify_admin_secret

    assert verify_admin_secret("anything", "") is False
