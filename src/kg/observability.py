"""Per-KG-ingestion Redis audit log.

Sibling of `src.extraction.vision.observability` with the same shape and TTL
conventions. Captures nodes_created / edges_created / error so operators can
see where KG enrichment bleeds.

Spec: 2026-04-24-kg-training-stage-background-design.md §7
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


REDIS_KEY_PREFIX = "kg:log"
TTL_SECONDS = 7 * 24 * 3600


@dataclass
class KGLogEntry:
    doc_id: str
    status: str  # KG_PENDING | KG_IN_PROGRESS | KG_COMPLETED | KG_FAILED
    nodes_created: int = 0
    edges_created: int = 0
    timings_ms: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None
    completed_at: float = 0.0


def build_kg_redis_key(doc_id: str) -> str:
    return f"{REDIS_KEY_PREFIX}:{doc_id}"


def serialize_kg_entry(entry: KGLogEntry) -> str:
    return json.dumps(asdict(entry), ensure_ascii=False)


def write_kg_entry_if_redis(*, redis_client: Any, entry: KGLogEntry) -> None:
    """Write the entry to Redis with TTL. Best-effort — errors swallowed."""
    if redis_client is None:
        return
    try:
        redis_client.setex(
            build_kg_redis_key(entry.doc_id),
            TTL_SECONDS,
            serialize_kg_entry(entry),
        )
    except Exception:
        # Observability must never break the KG task.
        pass
