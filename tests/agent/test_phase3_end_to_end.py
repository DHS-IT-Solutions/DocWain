"""Phase 3 Task 11 — end-to-end integration tests for the CoreAgent.

Covers the five Phase 3 scenarios:

1. Flag OFF — Layer C never called; ``doc_context["sme_pack"]`` absent
   or has no SME items.
2. Flag ON — Layer C dispatches; pack contains SME-backed items.
3. Retrieval-cache hit — identical query returns cached bundle without
   calling any layer the second time.
4. Flag-set version bump invalidates the cache — next identical query
   re-dispatches.
5. ``PIPELINE_TRAINING_COMPLETED`` transition invalidates the cache.

The agent pipeline is a long flow — to exercise the Phase 3 splice points
without running the real LLM + Qdrant + Mongo stack we short-circuit the
heavy steps with lightweight stubs. The goal is to prove the Phase 3
surface behaves correctly — not to re-test UNDERSTAND or REASON.
"""
from __future__ import annotations

import json
import types
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.agent.core_agent import CoreAgent
from src.config import feature_flags as ff
from src.config.feature_flags import (
    ENABLE_SME_RETRIEVAL,
    SME_REDESIGN_ENABLED,
    bump_flag_set_version,
    init_flag_resolver,
    reset_flag_set_version,
)


class _FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    def get(self, key):
        v = self.store.get(str(key))
        if isinstance(v, str):
            return v.encode("utf-8")
        return v

    def setex(self, key, ttl, value):
        self.store[str(key)] = value

    def scan_iter(self, match=None):
        pattern = (match or "").rstrip("*")
        return [k for k in list(self.store.keys()) if k.startswith(pattern)]

    def delete(self, key):
        self.store.pop(str(key), None)


class _MutableStore:
    def __init__(self, overrides: dict[str, dict[str, bool]] | None = None) -> None:
        self._by: dict[str, dict[str, bool]] = overrides or {}

    def get_subscription_overrides(self, sub: str) -> dict[str, bool]:
        return dict(self._by.get(sub, {}))

    def set_subscription_override(self, sub: str, flag: str, value: bool) -> None:
        self._by.setdefault(sub, {})[flag] = bool(value)


@pytest.fixture(autouse=True)
def _clean_flags():
    reset_flag_set_version()
    yield
    reset_flag_set_version()


def _agent_with_stubs(
    *,
    sme_hits: list[dict] | None = None,
    kg_direct: list[dict] | None = None,
    redis_client: Any = None,
    flags_sub: str = "sub_fin",
    sme_on: bool = True,
) -> CoreAgent:
    """Construct a CoreAgent with the minimum viable stubs for the Phase
    3 hot path. The reasoner is mocked to return a deterministic string;
    the qdrant + embedder + mongo stack is MagicMock."""
    store = _MutableStore(
        overrides={
            flags_sub: {
                SME_REDESIGN_ENABLED: True,
                ENABLE_SME_RETRIEVAL: sme_on,
            }
        }
    )
    init_flag_resolver(store=store)

    llm = MagicMock()
    llm.backend = "test"
    llm.generate_text = MagicMock(return_value="stub text")

    sme = MagicMock()
    sme.retrieve.return_value = sme_hits or []
    kg = MagicMock()
    kg.one_hop.return_value = kg_direct or []
    kg.inferred_relations.return_value = []

    agent = CoreAgent(
        llm_gateway=llm,
        qdrant_client=MagicMock(),
        embedder=MagicMock(),
        mongodb=MagicMock(),
        sme_retriever=sme,
        sme_kg_client=kg,
        redis_client=redis_client,
    )
    return agent


# ---------------------------------------------------------------------------
# Scenario 1 — Flag OFF
# ---------------------------------------------------------------------------


def test_flag_off_pack_has_no_sme_items():
    """With SME retrieval OFF, _build_sme_pack is never reached from
    handle(), and direct pack assembly (if called) yields no SME items."""
    agent = _agent_with_stubs(
        sme_hits=[{"text": "should be skipped", "confidence": 0.9}],
        sme_on=False,
    )
    pack = agent._build_sme_pack(
        query="Analyze Q3 trend",
        subscription_id="sub_fin",
        profile_id="p",
        profile_domain="finance",
        intent="analyze",
        query_understanding={"intent": "analyze"},
    )
    # Layer C was gated off at the retrieval layer, so no SME items
    # reach the pack regardless of what sme.retrieve would have returned.
    assert not any(p.sme_backed for p in pack)
    agent._sme_retriever.retrieve.assert_not_called()


# ---------------------------------------------------------------------------
# Scenario 2 — Flag ON
# ---------------------------------------------------------------------------


def test_flag_on_pack_contains_sme_backed_items():
    agent = _agent_with_stubs(
        sme_hits=[
            {
                "kind": "sme_artifact",
                "artifact_type": "insight",
                "text": "Q3 revenue grew on recurring contracts",
                "confidence": 0.92,
                "evidence": ["d1#c1", "d2#c3"],
                "score": 0.88,
            }
        ],
        sme_on=True,
    )
    pack = agent._build_sme_pack(
        query="Analyze Q3 revenue",
        subscription_id="sub_fin",
        profile_id="prof_fin",
        profile_domain="finance",
        intent="analyze",
        query_understanding={"intent": "analyze"},
    )
    assert pack, "flag ON should produce at least one pack item"
    sme_items = [p for p in pack if p.sme_backed]
    assert sme_items, "expected sme_backed items with flag ON"
    # Compressed SME text renders "[SME/insight] ..."
    assert any("[SME/insight]" in p.text for p in sme_items)
    # Provenance carries Layer C evidence refs.
    assert any(len(p.provenance) >= 1 for p in sme_items)


# ---------------------------------------------------------------------------
# Scenario 3 — Retrieval cache hit
# ---------------------------------------------------------------------------


def test_cache_hit_skips_retrieval_layers_on_second_call():
    redis = _FakeRedis()
    agent = _agent_with_stubs(
        sme_hits=[
            {"kind": "sme_artifact", "artifact_type": "dossier",
             "text": "cached narrative", "confidence": 0.9,
             "evidence": ["d1#c1"], "score": 0.8}
        ],
        redis_client=redis,
    )
    # First call — cache miss. Retrieval layers fire.
    agent._build_sme_pack(
        query="Diagnose Q3 drop",
        subscription_id="sub_fin",
        profile_id="prof_fin",
        profile_domain="finance",
        intent="diagnose",
        query_understanding={"intent": "diagnose"},
    )
    assert agent._sme_retriever.retrieve.call_count == 1
    # Exactly one bundle key is now in Redis.
    matching = [k for k in redis.store.keys() if "sub_fin:prof_fin" in k]
    assert len(matching) == 1
    # Second call — cache hit. Layers must NOT fire again.
    agent._build_sme_pack(
        query="Diagnose Q3 drop",
        subscription_id="sub_fin",
        profile_id="prof_fin",
        profile_domain="finance",
        intent="diagnose",
        query_understanding={"intent": "diagnose"},
    )
    assert agent._sme_retriever.retrieve.call_count == 1
    assert agent._sme_kg_client.one_hop.call_count == 1


# ---------------------------------------------------------------------------
# Scenario 4 — Flag-set version bump invalidates via key change
# ---------------------------------------------------------------------------


def test_flag_set_version_bump_invalidates_via_key_change():
    redis = _FakeRedis()
    agent = _agent_with_stubs(
        sme_hits=[
            {"kind": "sme_artifact", "artifact_type": "insight",
             "text": "n1", "confidence": 0.9}
        ],
        redis_client=redis,
    )
    # First call — populate the cache under v0.
    agent._build_sme_pack(
        query="q",
        subscription_id="sub_fin",
        profile_id="p",
        profile_domain="finance",
        intent="analyze",
        query_understanding={"intent": "analyze"},
    )
    assert agent._sme_retriever.retrieve.call_count == 1
    # Simulate an admin flag flip — bump_flag_set_version is the public
    # primitive that the PATCH handler + set_subscription_override call.
    bump_flag_set_version()
    agent._build_sme_pack(
        query="q",
        subscription_id="sub_fin",
        profile_id="p",
        profile_domain="finance",
        intent="analyze",
        query_understanding={"intent": "analyze"},
    )
    # Second call must miss the cache → retrieval layers fire again.
    assert agent._sme_retriever.retrieve.call_count == 2


# ---------------------------------------------------------------------------
# Scenario 5 — PIPELINE_TRAINING_COMPLETED invalidates the cache
# ---------------------------------------------------------------------------


def test_training_complete_invalidates_retrieval_cache_for_profile():
    """The pipeline_api hook evicts every key matching
    ``dwx:retrieval:{sub}:{prof}:*`` — next _build_sme_pack call misses
    the cache even at the same flag-set version."""
    from src.api import pipeline_api
    from src.retrieval.retrieval_cache import RetrievalCache

    redis = _FakeRedis()
    # Install a cache backed by the same fake redis so both the agent and
    # the pipeline hook see the same store.
    pipeline_api.set_retrieval_cache_for_tests(
        RetrievalCache(redis_client=redis)
    )
    agent = _agent_with_stubs(
        sme_hits=[
            {"kind": "sme_artifact", "artifact_type": "dossier",
             "text": "x", "confidence": 0.9}
        ],
        redis_client=redis,
    )
    agent._build_sme_pack(
        query="q",
        subscription_id="sub_fin",
        profile_id="prof_fin",
        profile_domain="finance",
        intent="analyze",
        query_understanding={"intent": "analyze"},
    )
    assert agent._sme_retriever.retrieve.call_count == 1
    assert any("sub_fin:prof_fin" in k for k in redis.store.keys())

    # Pipeline transitions to PIPELINE_TRAINING_COMPLETED — hook fires.
    pipeline_api._on_pipeline_training_completed(
        subscription_id="sub_fin", profile_id="prof_fin"
    )
    # Cache entries for that (sub, prof) are gone.
    assert not any("sub_fin:prof_fin" in k for k in redis.store.keys())

    # Next build → cache miss → retrieval layers fire again.
    agent._build_sme_pack(
        query="q",
        subscription_id="sub_fin",
        profile_id="prof_fin",
        profile_domain="finance",
        intent="analyze",
        query_understanding={"intent": "analyze"},
    )
    assert agent._sme_retriever.retrieve.call_count == 2
    pipeline_api.reset_retrieval_cache_for_tests()


# ---------------------------------------------------------------------------
# Handle-level integration: ensure doc_context["sme_pack"] lands for the
# reasoner when the flag is ON.
# ---------------------------------------------------------------------------


def test_splice_point_writes_sme_pack_when_flag_on():
    """Directly exercise the Phase 3 splice-point helper that handle()
    uses to populate ``doc_context["sme_pack"]``.

    We verify that calling ``_build_sme_pack`` returns a list[PackedItem]
    with SME-backed entries when the flag is on — that's exactly what
    handle() writes into doc_context. Testing the splice point keeps the
    integration assertion tight without needing to stub the whole
    UNDERSTAND + RETRIEVE pipeline (which drags in real Redis/Mongo
    client construction in non-injected paths).
    """
    agent = _agent_with_stubs(
        sme_hits=[
            {"kind": "sme_artifact", "artifact_type": "dossier",
             "text": "Q3 up 12%", "confidence": 0.92,
             "evidence": ["d1#c1"], "score": 0.9}
        ],
        sme_on=True,
    )
    pack = agent._build_sme_pack(
        query="Analyze Q3 revenue",
        subscription_id="sub_fin",
        profile_id="prof_fin",
        profile_domain="finance",
        intent="analyze",
        query_understanding={"intent": "analyze"},
    )
    assert pack
    assert any(getattr(p, "sme_backed", False) for p in pack)


def test_splice_point_no_sme_pack_when_flag_off():
    agent = _agent_with_stubs(
        sme_hits=[{"text": "hidden"}],
        sme_on=False,
    )
    pack = agent._build_sme_pack(
        query="q",
        subscription_id="sub_fin",
        profile_id="prof_fin",
        profile_domain="finance",
        intent="analyze",
        query_understanding={"intent": "analyze"},
    )
    # Flag OFF → Layer C never calls the retriever, so no SME-backed
    # items reach the pack; the pack may still contain Layer A chunks
    # (none in this stub), but nothing SME-backed.
    assert not any(getattr(p, "sme_backed", False) for p in pack)


def test_handle_signature_accepts_same_call_shape_as_before():
    """The Phase 3 splice does not change ``handle``'s signature — Phase 2
    callers keep working. Smoke assertion: construct + inspect only,
    don't actually invoke ``handle`` (that would drag in real clients)."""
    import inspect
    agent = _agent_with_stubs(sme_on=True)
    sig = inspect.signature(agent.handle)
    params = list(sig.parameters.keys())
    # Required Phase 2 shape — positional args preserved.
    for required in (
        "query",
        "subscription_id",
        "profile_id",
        "user_id",
        "session_id",
        "conversation_history",
    ):
        assert required in params, f"handle missing Phase 2 parameter {required!r}"
