"""Build synthetic synthesis-trace JSONL for tests."""
from __future__ import annotations

import json
from datetime import datetime, timedelta


def make_synth_jsonl(
    *,
    synthesis_id: str = "syn_001",
    subscription_id: str = "sub_a",
    profile_id: str = "prof_a",
    profile_domain: str = "finance",
    adapter_version: str = "1.2.0",
    adapter_content_hash: str = "abc123",
    started_at: datetime | None = None,
    duration_s: float = 120.0,
    builders_ok: tuple[str, ...] = (
        "dossier",
        "insight_index",
        "comparative_register",
        "kg_materializer",
        "recommendation_bank",
    ),
    drop_count: int = 0,
) -> str:
    """Produce a multi-line JSONL string conforming to spec Section 11."""
    started_at = started_at or datetime(2026, 4, 1, 2, 0, 0)
    ended_at = started_at + timedelta(seconds=duration_s)
    lines: list[str] = []

    lines.append(
        json.dumps(
            {
                "event": "synthesis_started",
                "synthesis_id": synthesis_id,
                "subscription_id": subscription_id,
                "profile_id": profile_id,
                "profile_domain": profile_domain,
                "adapter_version": adapter_version,
                "adapter_content_hash": adapter_content_hash,
                "started_at": started_at.isoformat(),
            }
        )
    )

    for b in builders_ok:
        lines.append(
            json.dumps(
                {
                    "event": "builder_complete",
                    "builder": b,
                    "items_produced": 10,
                    "items_persisted": 10 - (drop_count if b == builders_ok[0] else 0),
                    "duration_ms": 1500.0,
                    "errors": [],
                }
            )
        )

    for i in range(drop_count):
        lines.append(
            json.dumps(
                {
                    "event": "verifier_drop",
                    "item_id": f"{builders_ok[0]}_{i}",
                    "builder": builders_ok[0],
                    "reason_code": "evidence_presence",
                    "detail": f"dropped item {i}",
                }
            )
        )

    lines.append(
        json.dumps(
            {
                "event": "synthesis_completed",
                "synthesis_id": synthesis_id,
                "completed_at": ended_at.isoformat(),
            }
        )
    )

    return "\n".join(lines) + "\n"
