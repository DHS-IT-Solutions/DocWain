"""Per-extraction Redis audit log.

Every extraction writes a structured log entry so operators can see where
accuracy bleeds and where DocWain training should focus next. Entry shape is
defined in spec §7. Writes are best-effort; a None or offline Redis client
does not break extraction.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


REDIS_KEY_PREFIX = "extraction:log"
TTL_SECONDS = 7 * 24 * 3600


@dataclass
class ExtractionLogEntry:
    doc_id: str
    format: str
    path_taken: str
    timings_ms: Dict[str, float] = field(default_factory=dict)
    routing_decision: Dict[str, Any] = field(default_factory=dict)
    coverage_score: float = 1.0
    fallback_invocations: List[Dict[str, Any]] = field(default_factory=list)
    human_review: bool = False
    completed_at: float = 0.0


def build_redis_key(doc_id: str) -> str:
    return f"{REDIS_KEY_PREFIX}:{doc_id}"


def serialize_entry(entry: ExtractionLogEntry) -> str:
    return json.dumps(asdict(entry), ensure_ascii=False)


def write_entry_if_redis(*, redis_client: Any, entry: ExtractionLogEntry) -> None:
    """Write the entry to Redis with TTL. Best-effort — errors swallowed."""
    if redis_client is None:
        return
    try:
        redis_client.setex(
            build_redis_key(entry.doc_id),
            TTL_SECONDS,
            serialize_entry(entry),
        )
    except Exception:
        pass
