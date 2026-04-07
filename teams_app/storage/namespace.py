"""Namespace helpers for isolating Teams data from the main app."""

from __future__ import annotations

import re

_QDRANT_SAFE = re.compile(r"[^a-zA-Z0-9_]")


def qdrant_collection_name(tenant_id: str) -> str:
    """Build a Teams-namespaced Qdrant collection name."""
    safe = _QDRANT_SAFE.sub("_", tenant_id)
    return f"teams_{safe}"


def redis_key(tenant_id: str, user_id: str, suffix: str) -> str:
    """Build a Teams-namespaced Redis key."""
    return f"teams:{tenant_id}:{user_id}:{suffix}"


def blob_prefix(tenant_id: str) -> str:
    """Build a Teams-namespaced Azure Blob path prefix."""
    return f"teams/{tenant_id}/"


def mongo_collection(name: str) -> str:
    """Build a Teams-namespaced MongoDB collection name."""
    return f"teams_{name}"
