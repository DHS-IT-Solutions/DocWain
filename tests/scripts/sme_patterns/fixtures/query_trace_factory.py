"""Build synthetic query-trace JSONL for tests."""
from __future__ import annotations

import json
from datetime import datetime


def make_query_jsonl(
    *,
    query_id: str = "q_001",
    subscription_id: str = "sub_a",
    profile_id: str = "prof_a",
    profile_domain: str = "finance",
    query_text: str = "analyze Q3 trend",
    query_fingerprint: str = "abc",
    intent: str = "analyze",
    format_hint: str | None = None,
    adapter_version: str = "1.2.0",
    adapter_persona_role: str = "senior financial analyst",
    sme_artifacts: int = 5,
    citation_verifier_drops: int = 0,
    honest_compact_fallback: bool = False,
    rating: int | None = 1,
    captured_at: datetime | None = None,
) -> str:
    captured_at = captured_at or datetime(2026, 4, 5, 10, 0, 0)
    payload = {
        "event": "query_complete",
        "query_id": query_id,
        "subscription_id": subscription_id,
        "profile_id": profile_id,
        "profile_domain": profile_domain,
        "query_text": query_text,
        "query_fingerprint": query_fingerprint,
        "intent": intent,
        "format_hint": format_hint,
        "adapter_version": adapter_version,
        "adapter_persona_role": adapter_persona_role,
        "retrieval_layers": {
            "chunks": 12,
            "kg": 5,
            "sme_artifacts": sme_artifacts,
            "url": 0,
        },
        "pack_tokens": 4200,
        "reasoner_prompt_hash": "hashy",
        "response_len_tokens": 780,
        "citation_verifier_drops": citation_verifier_drops,
        "honest_compact_fallback": honest_compact_fallback,
        "url_present": False,
        "url_fetch_ok": None,
        "timing_ms": {
            "understand": 40,
            "retrieval": 210,
            "reasoner": 8400,
            "compose": 60,
            "total": 8710,
        },
        "feedback": (
            {
                "rating": rating,
                "edited": False,
                "follow_up_count": 0,
                "source": "feedback_tracker",
            }
            if rating is not None
            else None
        ),
        "captured_at": captured_at.isoformat(),
    }
    return json.dumps(payload) + "\n"
