"""API key authentication and usage tracking for the DocWain Standalone API."""
import hashlib
import logging
import secrets
from datetime import datetime, timezone
from typing import Dict, Optional

from fastapi import Header, HTTPException
from pymongo import MongoClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Key helpers
# ---------------------------------------------------------------------------

def hash_api_key(raw_key: str) -> str:
    """Return the SHA-256 hex digest of *raw_key*."""
    return hashlib.sha256(raw_key.encode()).hexdigest()


def generate_api_key() -> tuple[str, str]:
    """Generate a new API key.

    Returns:
        (raw_key, key_hash) where raw_key is ``"dw_"`` + 48 hex characters
        and key_hash is the SHA-256 hex digest of raw_key.
    """
    raw_key = "dw_" + secrets.token_hex(24)
    key_hash = hash_api_key(raw_key)
    return raw_key, key_hash


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_api_key_sync(
    raw_key: str,
    keys_collection,
) -> Optional[Dict]:
    """Validate *raw_key* against *keys_collection*.

    Hashes the key, looks it up in the collection, and returns the document
    only when ``active == True``.  Returns ``None`` otherwise.
    """
    key_hash = hash_api_key(raw_key)
    doc = keys_collection.find_one({"key_hash": key_hash})
    if doc is None:
        return None
    if not doc.get("active", False):
        return None
    return doc


# ---------------------------------------------------------------------------
# Usage tracking (fire-and-forget)
# ---------------------------------------------------------------------------

def track_usage(
    keys_collection,
    key_hash: str,
    endpoint: str,
    mode: str,
) -> None:
    """Increment usage counters for *key_hash*.

    Updates:
    - ``$inc``: total_requests, requests_today, by_endpoint.<endpoint>,
      by_mode.<mode>
    - ``$set``: last_used (UTC timestamp)

    Exceptions are caught and logged so callers are never interrupted.
    """
    try:
        keys_collection.update_one(
            {"key_hash": key_hash},
            {
                "$inc": {
                    "total_requests": 1,
                    "requests_today": 1,
                    f"by_endpoint.{endpoint}": 1,
                    f"by_mode.{mode}": 1,
                },
                "$set": {
                    "last_used": datetime.now(tz=timezone.utc),
                },
            },
        )
    except Exception:
        logger.exception("track_usage: failed to update key_hash=%s", key_hash)


def track_document_processed(keys_collection, key_hash: str) -> None:
    """Increment the ``documents_processed`` counter for *key_hash*.

    Exceptions are caught and logged so callers are never interrupted.
    """
    try:
        keys_collection.update_one(
            {"key_hash": key_hash},
            {"$inc": {"documents_processed": 1}},
        )
    except Exception:
        logger.exception(
            "track_document_processed: failed to update key_hash=%s", key_hash
        )


# ---------------------------------------------------------------------------
# MongoDB collection helper
# ---------------------------------------------------------------------------

def _get_keys_collection():
    """Return the MongoDB ``api_keys`` collection.

    Uses ``Config.MongoDB.URI``, ``Config.MongoDB.DB``, and
    ``Config.Standalone.API_KEYS_COLLECTION`` (falls back to ``"api_keys"``).
    """
    from src.api.config import Config

    uri = Config.MongoDB.URI
    db_name = Config.MongoDB.DB
    standalone = getattr(Config, "Standalone", None)
    collection_name = (
        getattr(standalone, "API_KEYS_COLLECTION", None) or "api_keys"
    )

    client = MongoClient(uri)
    db = client[db_name]
    return db[collection_name]


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------

async def require_api_key(
    request=None,
    x_api_key: Optional[str] = Header(None),
) -> Dict:
    """FastAPI dependency that validates the ``X-Api-Key`` header.

    Returns a dict with:
    - ``key_hash``
    - ``name``
    - ``subscription_id``
    - ``permissions``
    - ``keys_collection``

    Raises ``HTTPException(401)`` when the key is missing or invalid.
    """
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing X-Api-Key header")

    keys_collection = _get_keys_collection()
    doc = validate_api_key_sync(x_api_key, keys_collection)

    if doc is None:
        raise HTTPException(status_code=401, detail="Invalid or inactive API key")

    return {
        "key_hash": doc.get("key_hash"),
        "name": doc.get("name"),
        "subscription_id": doc.get("subscription_id"),
        "permissions": doc.get("permissions", []),
        "keys_collection": keys_collection,
    }
