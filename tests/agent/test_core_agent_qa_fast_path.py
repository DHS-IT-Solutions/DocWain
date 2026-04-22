"""Phase 3 Task 7 — QA fast-path short-circuit tests on CoreAgent.

The QA fast path probes Redis for a pre-grounded Q&A pair at the query
fingerprint; a hit short-circuits the entire UNDERSTAND+RETRIEVE+REASON
pipeline and returns the cached answer. These tests use a fake Redis
client so the hot path is exercised end-to-end without touching the live
Redis instance.
"""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.agent.core_agent import CoreAgent
from src.intelligence.qa_generator import qa_index_fingerprint


class _FakeRedis:
    """Minimal Redis stand-in — only ``get`` is used by the fast path."""

    def __init__(self, mapping: dict[str, Any] | None = None) -> None:
        self._m: dict[str, Any] = dict(mapping or {})

    def set(self, key: str, value: Any) -> None:
        self._m[key] = value

    def get(self, key: str) -> Any:
        v = self._m.get(key)
        if isinstance(v, str):
            return v.encode("utf-8")
        return v


def _build_agent(redis_client: Any = None) -> CoreAgent:
    llm = MagicMock()
    llm.backend = "test"
    return CoreAgent(
        llm_gateway=llm,
        qdrant_client=MagicMock(),
        embedder=MagicMock(),
        mongodb=MagicMock(),
        redis_client=redis_client,
    )


# ---------------------------------------------------------------------------
# Hits
# ---------------------------------------------------------------------------


def test_hit_returns_cached_answer_without_reasoner() -> None:
    sub, prof = "sub_fin", "prof_fin"
    question = "What is our Q3 revenue?"
    fp = qa_index_fingerprint(question)
    payload = {
        "question": question,
        "answer": "Q3 revenue was $12.4M.",
        "metadata": {"confidence": 0.95, "task_type": "lookup", "sources": []},
        "qa_id": "qa_42",
    }
    fake = _FakeRedis({f"qa_idx:{sub}:{prof}:{fp}": json.dumps(payload)})
    agent = _build_agent(redis_client=fake)

    out = agent._qa_fast_path_lookup(
        query=question, subscription_id=sub, profile_id=prof
    )
    assert out is not None
    assert out["response"] == "Q3 revenue was $12.4M."
    assert out["grounded"] is True
    assert out["context_found"] is True
    assert out["metadata"]["qa_fast_path_hit"] is True
    assert out["metadata"]["qa_id"] == "qa_42"


def test_handle_short_circuits_on_hit() -> None:
    sub, prof = "sub_a", "prof_a"
    question = "What is our Q3 revenue?"
    fp = qa_index_fingerprint(question)
    payload = {
        "question": question,
        "answer": "cached answer",
        "metadata": {"confidence": 0.9, "task_type": "lookup"},
        "qa_id": "qa_1",
    }
    fake = _FakeRedis({f"qa_idx:{sub}:{prof}:{fp}": json.dumps(payload)})
    agent = _build_agent(redis_client=fake)

    # If the Reasoner is invoked, the MagicMock would be called. The fast
    # path must not reach it.
    result = agent.handle(
        query=question,
        subscription_id=sub,
        profile_id=prof,
        user_id="u1",
        session_id="s1",
        conversation_history=None,
    )
    assert result["response"] == "cached answer"
    assert result["metadata"]["qa_fast_path_hit"] is True
    # Timing block attached even on the fast path.
    assert "qa_fast_path_ms" in result["metadata"]["timing"]


def test_hit_respects_key_scope_per_profile() -> None:
    # Seed with sub=sub_a, prof=prof_a; a lookup against prof_b misses.
    question = "identical text"
    fp = qa_index_fingerprint(question)
    payload = {"answer": "x", "metadata": {"confidence": 0.9}, "qa_id": "q"}
    fake = _FakeRedis({f"qa_idx:sub_a:prof_a:{fp}": json.dumps(payload)})
    agent = _build_agent(redis_client=fake)
    hit_a = agent._qa_fast_path_lookup(
        query=question, subscription_id="sub_a", profile_id="prof_a"
    )
    hit_b = agent._qa_fast_path_lookup(
        query=question, subscription_id="sub_a", profile_id="prof_b"
    )
    hit_c = agent._qa_fast_path_lookup(
        query=question, subscription_id="sub_b", profile_id="prof_a"
    )
    assert hit_a is not None
    assert hit_b is None
    assert hit_c is None


def test_fingerprint_is_canonical_normalized() -> None:
    # Trailing whitespace + mixed case must not change the cache key.
    assert qa_index_fingerprint("  WHAT is the  Q3  revenue  ?") == qa_index_fingerprint(
        "what is the q3 revenue ?"
    )


# ---------------------------------------------------------------------------
# Misses
# ---------------------------------------------------------------------------


def test_miss_returns_none() -> None:
    fake = _FakeRedis({})
    agent = _build_agent(redis_client=fake)
    assert (
        agent._qa_fast_path_lookup(
            query="anything", subscription_id="s", profile_id="p"
        )
        is None
    )


def test_low_confidence_is_treated_as_miss() -> None:
    question = "test q"
    fp = qa_index_fingerprint(question)
    payload = {
        "answer": "weak cached",
        "metadata": {"confidence": 0.5},
        "qa_id": "qa",
    }
    fake = _FakeRedis({f"qa_idx:s:p:{fp}": json.dumps(payload)})
    agent = _build_agent(redis_client=fake)
    assert (
        agent._qa_fast_path_lookup(
            query=question,
            subscription_id="s",
            profile_id="p",
            min_confidence=0.85,
        )
        is None
    )


def test_missing_answer_field_treated_as_miss() -> None:
    question = "q"
    fp = qa_index_fingerprint(question)
    # No 'answer' key → miss.
    payload = {"metadata": {"confidence": 0.9}, "qa_id": "x"}
    fake = _FakeRedis({f"qa_idx:s:p:{fp}": json.dumps(payload)})
    agent = _build_agent(redis_client=fake)
    assert (
        agent._qa_fast_path_lookup(
            query=question, subscription_id="s", profile_id="p"
        )
        is None
    )


def test_no_redis_client_means_miss_no_exception() -> None:
    agent = _build_agent(redis_client=None)
    # _get_redis_client may also return None in unit tests; mock to force.
    agent._get_redis_client = lambda: None  # type: ignore[assignment]
    assert (
        agent._qa_fast_path_lookup(
            query="q", subscription_id="s", profile_id="p"
        )
        is None
    )


def test_redis_get_exception_falls_through() -> None:
    class _BadRedis:
        def get(self, key: str) -> Any:
            raise RuntimeError("redis down")

    agent = _build_agent(redis_client=_BadRedis())
    assert (
        agent._qa_fast_path_lookup(
            query="q", subscription_id="s", profile_id="p"
        )
        is None
    )


def test_non_json_payload_falls_through() -> None:
    # A malformed entry (not JSON) must degrade to a miss — Phase 2
    # emitters always write JSON, but defense in depth.
    question = "q"
    fp = qa_index_fingerprint(question)
    fake = _FakeRedis({f"qa_idx:s:p:{fp}": "this is not json"})
    agent = _build_agent(redis_client=fake)
    assert (
        agent._qa_fast_path_lookup(
            query=question, subscription_id="s", profile_id="p"
        )
        is None
    )


def test_handle_continues_when_fast_path_misses() -> None:
    """On miss, handle must not return the fast-path dict — it must drop
    through to the full pipeline. We assert by patching the downstream
    doc-intelligence load to raise a marker exception; if the fast path
    returned, we'd never see it."""
    agent = _build_agent(redis_client=_FakeRedis({}))

    marker = RuntimeError("reached full pipeline")

    def _boom(*a, **kw):
        raise marker

    agent._load_doc_intelligence = _boom  # type: ignore[assignment]
    with pytest.raises(RuntimeError, match="reached full pipeline"):
        agent.handle(
            query="q",
            subscription_id="s",
            profile_id="p",
            user_id="u",
            session_id="sess",
            conversation_history=None,
        )
