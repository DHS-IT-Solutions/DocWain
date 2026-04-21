"""Cross-module flag-gating tests (user Task 12).

Unit tests for Layer B (``UnifiedRetriever.retrieve_layer_b``) and Layer C
(``SMERetrieval.retrieve``) live in the module-specific test files; this
file covers the cross-cutting contract that both paths consume the
*same* ``SMEFeatureFlags`` resolver from ``src.config.feature_flags`` and
that the three canonical flags resolve correctly through the full
``FlagStore`` → ``SMEFeatureFlags`` → module-gate call path.

The tests use a tiny :class:`_DictFlagStore` so per-subscription overrides
are the same shape MongoDB will produce in production — any drift in the
override schema shows up here first.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.config.feature_flags import (
    ENABLE_KG_SYNTHESIZED_EDGES,
    ENABLE_SME_RETRIEVAL,
    SME_REDESIGN_ENABLED,
    FlagStore,
    SMEFeatureFlags,
    init_flag_resolver,
)
from src.retrieval.sme_retrieval import SMERetrieval
from src.retrieval.unified_retriever import UnifiedRetriever


class _DictFlagStore:
    """Minimal :class:`FlagStore` shim — one dict per subscription."""

    def __init__(self, overrides: dict[str, dict[str, bool]]) -> None:
        self._overrides = overrides

    def get_subscription_overrides(self, subscription_id: str) -> dict[str, bool]:
        return dict(self._overrides.get(subscription_id, {}))


def _prime_resolver(store: FlagStore) -> None:
    init_flag_resolver(store=store)


def _qdrant_row(payload: dict, score: float) -> SimpleNamespace:
    return SimpleNamespace(payload=payload, score=score, id=payload.get("snippet_id"))


# ---------------------------------------------------------------------------
# Layer C (SMERetrieval) gating via the real resolver
# ---------------------------------------------------------------------------
def test_layer_c_flag_off_globally_returns_empty() -> None:
    _prime_resolver(_DictFlagStore({}))
    qdrant = MagicMock()
    embedder = MagicMock()
    r = SMERetrieval(qdrant=qdrant, embedder=embedder)
    assert (
        r.retrieve(
            query="q",
            subscription_id="sub_x",
            profile_id="prof_y",
            artifact_types=None,
            top_k=5,
        )
        == []
    )
    qdrant.search.assert_not_called()


def test_layer_c_flag_on_via_per_sub_override_returns_results() -> None:
    _prime_resolver(
        _DictFlagStore(
            {
                "sub_x": {
                    SME_REDESIGN_ENABLED: True,
                    ENABLE_SME_RETRIEVAL: True,
                }
            }
        )
    )
    qdrant = MagicMock()
    embedder = MagicMock()
    embedder.embed.return_value = [0.0] * 8
    qdrant.search.return_value = [
        _qdrant_row(
            payload={
                "artifact_type": "insight_index",
                "snippet_id": "insight:1",
                "text": "Rev rose 4%",
                "confidence": 0.9,
                "evidence": [],
            },
            score=0.88,
        )
    ]
    r = SMERetrieval(qdrant=qdrant, embedder=embedder)
    hits = r.retrieve(
        query="q",
        subscription_id="sub_x",
        profile_id="prof_y",
        artifact_types=None,
        top_k=5,
    )
    assert len(hits) == 1
    assert hits[0]["artifact_type"] == "insight_index"


def test_layer_c_master_off_dependent_on_stays_off() -> None:
    # Per-sub turns on enable_sme_retrieval but leaves master off:
    # dependent flag must resolve False (master-gate precedence).
    _prime_resolver(
        _DictFlagStore(
            {"sub_x": {ENABLE_SME_RETRIEVAL: True}}
        )
    )
    qdrant = MagicMock()
    r = SMERetrieval(qdrant=qdrant, embedder=MagicMock())
    assert (
        r.retrieve(
            query="q",
            subscription_id="sub_x",
            profile_id="prof_y",
            artifact_types=None,
            top_k=5,
        )
        == []
    )


# ---------------------------------------------------------------------------
# Layer B (UnifiedRetriever.retrieve_layer_b) gating via the real resolver
# ---------------------------------------------------------------------------
def test_layer_b_inferred_off_when_master_off() -> None:
    _prime_resolver(_DictFlagStore({}))
    kg = MagicMock()
    kg.one_hop.return_value = [{"src": "a", "dst": "b", "type": "X", "evidence": []}]
    kg.inferred_relations.return_value = [
        {
            "src": "a",
            "dst": "c",
            "relation_type": "inf",
            "confidence": 0.9,
            "evidence": [],
            "inference_path": [],
        }
    ]
    r = UnifiedRetriever(kg_client=kg, qdrant=MagicMock(), sme=MagicMock())
    hits = r.retrieve_layer_b(
        query="q", subscription_id="sub_x", profile_id="prof_y", top_k=5
    )
    # Direct-only: the original KG view survives master rollback.
    assert [h["kind"] for h in hits] == ["kg_direct"]
    kg.inferred_relations.assert_not_called()


def test_layer_b_inferred_on_requires_both_flags() -> None:
    _prime_resolver(
        _DictFlagStore(
            {
                "sub_x": {
                    SME_REDESIGN_ENABLED: True,
                    ENABLE_SME_RETRIEVAL: True,
                    ENABLE_KG_SYNTHESIZED_EDGES: True,
                }
            }
        )
    )
    kg = MagicMock()
    kg.one_hop.return_value = [{"src": "a", "dst": "b", "type": "X", "evidence": []}]
    kg.inferred_relations.return_value = [
        {
            "src": "a",
            "dst": "c",
            "relation_type": "inf",
            "confidence": 0.9,
            "evidence": [],
            "inference_path": [],
        }
    ]
    r = UnifiedRetriever(kg_client=kg, qdrant=MagicMock(), sme=MagicMock())
    hits = r.retrieve_layer_b(
        query="q", subscription_id="sub_x", profile_id="prof_y", top_k=5
    )
    assert {h["kind"] for h in hits} == {"kg_direct", "kg_inferred"}


def test_layer_b_synth_edges_off_still_returns_direct_kg() -> None:
    _prime_resolver(
        _DictFlagStore(
            {
                "sub_x": {
                    SME_REDESIGN_ENABLED: True,
                    ENABLE_SME_RETRIEVAL: True,
                }
            }
        )
    )
    kg = MagicMock()
    kg.one_hop.return_value = [{"src": "a", "dst": "b", "type": "X", "evidence": []}]
    r = UnifiedRetriever(kg_client=kg, qdrant=MagicMock(), sme=MagicMock())
    hits = r.retrieve_layer_b(
        query="q", subscription_id="sub_x", profile_id="prof_y", top_k=5
    )
    assert [h["kind"] for h in hits] == ["kg_direct"]
    kg.inferred_relations.assert_not_called()
