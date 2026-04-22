"""Phase 3 Task 10 — CoreAgent SME pack wiring.

Flag ON → four-layer retrieval → merge/rerank/MMR/PackAssembler pipeline
writes the assembled list[PackedItem] into ``doc_context["sme_pack"]``
for Phase 4's rich-mode consumer. Flag OFF → Layer C never runs and no
``sme_pack`` key is written (the reasoner's existing evidence list is
untouched).

These tests exercise the helper :meth:`CoreAgent._build_sme_pack` directly
so they don't need the full :meth:`handle` flow — the splice point in
``handle`` itself is covered by the end-to-end tests in
``test_phase3_end_to_end.py``.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from src.agent.core_agent import CoreAgent
from src.config import feature_flags as ff
from src.config.feature_flags import (
    ENABLE_CROSS_ENCODER_RERANK,
    ENABLE_SME_RETRIEVAL,
    SME_REDESIGN_ENABLED,
    init_flag_resolver,
    reset_flag_set_version,
)


class _MutableStore:
    def __init__(self, overrides: dict[str, dict[str, bool]] | None = None) -> None:
        self._by: dict[str, dict[str, bool]] = overrides or {}

    def get_subscription_overrides(self, sub: str) -> dict[str, bool]:
        return dict(self._by.get(sub, {}))

    def set_subscription_override(self, sub: str, flag: str, value: bool) -> None:
        self._by.setdefault(sub, {})[flag] = bool(value)


def _build_agent(*, sme=None, kg=None, qdrant=None, cross_encoder=None) -> CoreAgent:
    llm = MagicMock()
    llm.backend = "test"
    return CoreAgent(
        llm_gateway=llm,
        qdrant_client=qdrant or MagicMock(),
        embedder=MagicMock(),
        mongodb=MagicMock(),
        cross_encoder=cross_encoder,
        sme_retriever=sme,
        sme_kg_client=kg,
    )


@pytest.fixture(autouse=True)
def _clean_slate() -> None:
    reset_flag_set_version()
    yield
    reset_flag_set_version()


def test_flag_on_pack_contains_sme_items():
    store = _MutableStore(
        overrides={"sub_fin": {SME_REDESIGN_ENABLED: True, ENABLE_SME_RETRIEVAL: True}}
    )
    init_flag_resolver(store=store)
    sme = MagicMock()
    sme.retrieve.return_value = [
        {
            "kind": "sme_artifact",
            "artifact_type": "dossier",
            "text": "Q3 revenue analysis narrative",
            "confidence": 0.92,
            "evidence": ["d1#c1"],
            "score": 0.85,
        }
    ]
    kg = MagicMock()
    kg.one_hop.return_value = []
    kg.inferred_relations.return_value = []
    agent = _build_agent(sme=sme, kg=kg)
    pack = agent._build_sme_pack(
        query="Analyze Q3 revenue",
        subscription_id="sub_fin",
        profile_id="prof_fin",
        profile_domain="finance",
        intent="analyze",
        query_understanding={"intent": "analyze"},
    )
    # Pack is a list[PackedItem]
    assert pack
    # Layer C item must be in the pack with sme_backed=True.
    sme_items = [p for p in pack if p.sme_backed]
    assert sme_items, "expected at least one SME-backed pack item"
    # Compressed SME text shape — "[SME/dossier] ..."
    assert any("[SME/" in p.text for p in sme_items)


def test_flag_off_skips_layer_c_no_sme_items():
    store = _MutableStore(
        overrides={"sub_off": {SME_REDESIGN_ENABLED: True, ENABLE_SME_RETRIEVAL: False}}
    )
    init_flag_resolver(store=store)
    sme = MagicMock()
    sme.retrieve.return_value = [
        {"kind": "sme_artifact", "text": "should not appear", "confidence": 0.9}
    ]
    kg = MagicMock()
    kg.one_hop.return_value = []
    kg.inferred_relations.return_value = []
    agent = _build_agent(sme=sme, kg=kg)
    pack = agent._build_sme_pack(
        query="Analyze Q3",
        subscription_id="sub_off",
        profile_id="prof_off",
        profile_domain="finance",
        intent="analyze",
        query_understanding={"intent": "analyze"},
    )
    sme.retrieve.assert_not_called()
    assert not any(p.sme_backed for p in pack)


def test_retrieval_cache_hit_short_circuits_layer_calls():
    store = _MutableStore(
        overrides={"sub_c": {SME_REDESIGN_ENABLED: True, ENABLE_SME_RETRIEVAL: True}}
    )
    init_flag_resolver(store=store)
    sme = MagicMock()
    sme.retrieve.return_value = [
        {"kind": "sme_artifact", "text": "cached once", "confidence": 0.9,
         "artifact_type": "insight"}
    ]
    kg = MagicMock()
    kg.one_hop.return_value = []
    kg.inferred_relations.return_value = []

    # Fake Redis that records both get + setex calls so we can prove the
    # cache round-trips.
    class _FakeRedis:
        def __init__(self):
            self.store: dict[str, str] = {}
        def get(self, key):
            return self.store.get(str(key))
        def setex(self, key, ttl, value):
            self.store[str(key)] = value
        def scan_iter(self, match=None):
            return []
        def delete(self, key):
            self.store.pop(str(key), None)

    fake_redis = _FakeRedis()
    agent = _build_agent(sme=sme, kg=kg)
    agent._redis_client_injected = fake_redis  # type: ignore[assignment]

    pack1 = agent._build_sme_pack(
        query="Analyze Q3",
        subscription_id="sub_c",
        profile_id="prof_c",
        profile_domain="finance",
        intent="analyze",
        query_understanding={"intent": "analyze"},
    )
    # First call dispatched layers.
    assert sme.retrieve.call_count == 1
    assert pack1

    # Second call with identical params must HIT the cache.
    pack2 = agent._build_sme_pack(
        query="Analyze Q3",
        subscription_id="sub_c",
        profile_id="prof_c",
        profile_domain="finance",
        intent="analyze",
        query_understanding={"intent": "analyze"},
    )
    assert sme.retrieve.call_count == 1, "cache hit should skip layer dispatch"
    # Pack should be reconstructed from the cached bundle — same shape.
    assert [(p.layer, p.sme_backed) for p in pack2] == [
        (p.layer, p.sme_backed) for p in pack1
    ]


def test_flag_bump_invalidates_via_version_change():
    store = _MutableStore(
        overrides={"sub_d": {SME_REDESIGN_ENABLED: True, ENABLE_SME_RETRIEVAL: True}}
    )
    init_flag_resolver(store=store)
    sme = MagicMock()
    sme.retrieve.return_value = [
        {"kind": "sme_artifact", "text": "v1", "confidence": 0.9,
         "artifact_type": "insight"}
    ]
    kg = MagicMock()
    kg.one_hop.return_value = []
    kg.inferred_relations.return_value = []

    class _FakeRedis:
        def __init__(self):
            self.store: dict[str, str] = {}
        def get(self, k):
            return self.store.get(str(k))
        def setex(self, k, ttl, v):
            self.store[str(k)] = v
        def scan_iter(self, match=None):
            return []
        def delete(self, k):
            self.store.pop(str(k), None)

    fake_redis = _FakeRedis()
    agent = _build_agent(sme=sme, kg=kg)
    agent._redis_client_injected = fake_redis  # type: ignore[assignment]

    agent._build_sme_pack(
        query="q",
        subscription_id="sub_d",
        profile_id="p",
        profile_domain="finance",
        intent="analyze",
        query_understanding={"intent": "analyze"},
    )
    assert sme.retrieve.call_count == 1
    # Bump the version — retrieval cache key changes so next call MISSES.
    ff.bump_flag_set_version()
    agent._build_sme_pack(
        query="q",
        subscription_id="sub_d",
        profile_id="p",
        profile_domain="finance",
        intent="analyze",
        query_understanding={"intent": "analyze"},
    )
    assert sme.retrieve.call_count == 2


def test_cross_encoder_rerank_flag_off_skips_model_call():
    store = _MutableStore(
        overrides={"sub_e": {SME_REDESIGN_ENABLED: True, ENABLE_SME_RETRIEVAL: True,
                             ENABLE_CROSS_ENCODER_RERANK: False}}
    )
    init_flag_resolver(store=store)
    sme = MagicMock()
    sme.retrieve.return_value = []
    kg = MagicMock()
    kg.one_hop.return_value = []
    kg.inferred_relations.return_value = []
    ce = MagicMock()
    ce.predict.return_value = [0.5]
    agent = _build_agent(sme=sme, kg=kg, cross_encoder=ce)
    # Inject a single Layer A chunk via the qdrant mock so merge_layers
    # produces at least one PackedItem.
    agent._retrieve_four_layers = MagicMock(return_value=_bundle_with_chunk())  # type: ignore[assignment]

    agent._build_sme_pack(
        query="q",
        subscription_id="sub_e",
        profile_id="p",
        profile_domain="generic",
        intent="analyze",
        query_understanding={"intent": "analyze"},
    )
    ce.predict.assert_not_called()


def _bundle_with_chunk():
    from src.retrieval.types import RetrievalBundle

    return RetrievalBundle(
        layer_a_chunks=[
            {"doc_id": "d1", "chunk_id": "c1", "text": "chunk text",
             "score": 0.9, "confidence": 0.9}
        ],
        layer_b_kg=[],
        layer_c_sme=[],
        layer_d_url=[],
    )


def test_profile_domain_default_falls_back_to_generic():
    store = _MutableStore(
        overrides={"sub_g": {SME_REDESIGN_ENABLED: True, ENABLE_SME_RETRIEVAL: True}}
    )
    init_flag_resolver(store=store)
    sme = MagicMock()
    sme.retrieve.return_value = []
    kg = MagicMock()
    kg.one_hop.return_value = []
    kg.inferred_relations.return_value = []
    agent = _build_agent(sme=sme, kg=kg)
    # Even without a profile_domain, the stub adapter path is used and
    # the pack is assembled (or empty). Must not raise.
    pack = agent._build_sme_pack(
        query="q",
        subscription_id="sub_g",
        profile_id="p",
        profile_domain="",
        intent="analyze",
        query_understanding={"intent": "analyze"},
    )
    assert isinstance(pack, list)


def test_empty_bundle_returns_empty_pack():
    store = _MutableStore(
        overrides={"sub_h": {SME_REDESIGN_ENABLED: True, ENABLE_SME_RETRIEVAL: True}}
    )
    init_flag_resolver(store=store)
    sme = MagicMock()
    sme.retrieve.return_value = []
    kg = MagicMock()
    kg.one_hop.return_value = []
    kg.inferred_relations.return_value = []
    agent = _build_agent(sme=sme, kg=kg)
    pack = agent._build_sme_pack(
        query="q",
        subscription_id="sub_h",
        profile_id="p",
        profile_domain="generic",
        intent="analyze",
        query_understanding={"intent": "analyze"},
    )
    assert pack == []
