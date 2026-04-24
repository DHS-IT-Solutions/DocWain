"""Content-hash Redis cache for expensive LLM sub-calls.

Identical document text produces identical identify/understand results, so
re-uploads and near-duplicate content can skip the ~30-60s LLM round trip
entirely. Key strategy:

    sha256(masked_text[:_HASH_SAMPLE_BYTES]) + ":" + op + ":" + prompt_version

Tuning:
- Truncation (32 KB) stabilises the hash on very large docs while still
  capturing enough content to make collisions negligible.
- TTL defaults to 7 days. Bump ``prompt_version`` in the calling site
  whenever the prompt text or output schema changes — that invalidates
  every stale entry without explicit flushing.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_HASH_SAMPLE_BYTES = 32 * 1024
_DEFAULT_TTL_SECONDS = 7 * 24 * 3600


def _redis_client():
    try:
        from src.api.dw_newron import get_redis_client
        return get_redis_client()
    except Exception:
        return None


def content_hash(text: str) -> str:
    """Stable sha256 over the first 32 KB of *text*."""
    if not text:
        return ""
    raw = text[:_HASH_SAMPLE_BYTES].encode("utf-8", errors="ignore")
    return hashlib.sha256(raw).hexdigest()


def cache_key(op: str, text_hash: str, prompt_version: str) -> str:
    return f"dw:llm:{op}:{prompt_version}:{text_hash}"


def get_cached(op: str, text: str, prompt_version: str) -> Optional[dict]:
    """Return cached payload or None. Best-effort, swallows Redis errors."""
    if not text:
        return None
    client = _redis_client()
    if not client:
        return None
    try:
        key = cache_key(op, content_hash(text), prompt_version)
        raw = client.get(key)
        if not raw:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="ignore")
        return json.loads(raw)
    except Exception:
        return None


def set_cached(op: str, text: str, prompt_version: str, payload: Any, ttl: int = _DEFAULT_TTL_SECONDS) -> None:
    if not text or payload is None:
        return
    client = _redis_client()
    if not client:
        return
    try:
        key = cache_key(op, content_hash(text), prompt_version)
        client.setex(key, ttl, json.dumps(payload, default=str))
    except Exception:
        pass


__all__ = ["get_cached", "set_cached", "content_hash", "cache_key"]
