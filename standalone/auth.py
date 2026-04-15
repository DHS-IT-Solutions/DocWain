import hashlib
import secrets
from typing import Optional


def hash_api_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()


def generate_api_key() -> str:
    return "dw_sa_" + secrets.token_hex(24)


def validate_api_key(raw_key: str, keys_collection) -> Optional[dict]:
    key_hash = hash_api_key(raw_key)
    return keys_collection.find_one({"key_hash": key_hash, "active": True})


def verify_admin_secret(provided: str, expected: str) -> bool:
    if not expected:
        return False
    return secrets.compare_digest(provided, expected)
