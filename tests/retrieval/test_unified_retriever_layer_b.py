"""Layer B retrieval tests — KG 1-hop + ``INFERRED_RELATION`` synthesized edges.

These tests verify the canonical Phase 2 helper :meth:`UnifiedRetriever.
retrieve_layer_b` per ERRATA §7. They use a fake KG client (MagicMock) so
nothing talks to a real Neo4j, and a stub flag resolver so flag state is
explicit in every case.

Contract pinned here:

* Layer B always returns ``kg_direct`` rows.
* ``kg_inferred`` rows only appear when BOTH ``enable_sme_retrieval`` AND
  ``enable_kg_synthesized_edges`` are on.
* The confidence floor is threaded through to the KG client as
  ``min_confidence`` so the Cypher filter matches Phase 2 spec.
* Profile isolation is enforced at the helper boundary — missing
  ``subscription_id`` or ``profile_id`` raises ``ValueError``.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.config.feature_flags import (
    ENABLE_KG_SYNTHESIZED_EDGES,
    ENABLE_SME_RETRIEVAL,
)
from src.retrieval.unified_retriever import UnifiedRetriever


def _resolver_stub(enabled_set: set[str]) -> MagicMock:
    resolver = MagicMock()
    resolver.is_enabled.side_effect = lambda sub, flag: flag in enabled_set
    return resolver


def _kg_stub(
    direct_rows: list[dict] | None = None,
    inferred_rows: list[dict] | None = None,
) -> MagicMock:
    kg = MagicMock()
    kg.one_hop.return_value = direct_rows or []
    kg.inferred_relations.return_value = inferred_rows or []
    return kg


def test_layer_b_includes_inferred_with_profile_filter_and_conf_floor() -> None:
    direct = [
        {
            "src": "n1",
            "dst": "n2",
            "type": "MENTIONS",
            "evidence": ["d1#c1"],
        }
    ]
    inferred = [
        {
            "src": "n1",
            "dst": "n9",
            "relation_type": "indirectly_funds",
            "confidence": 0.78,
            "evidence": ["d1#c1", "d2#c3"],
            "inference_path": ["n1-[PAYS]->n4", "n4-[FUNDS]->n9"],
        }
    ]
    kg = _kg_stub(direct_rows=direct, inferred_rows=inferred)
    retriever = UnifiedRetriever(kg_client=kg, qdrant=MagicMock(), sme=MagicMock())
    with patch(
        "src.retrieval.unified_retriever.get_flag_resolver",
        return_value=_resolver_stub(
            {ENABLE_SME_RETRIEVAL, ENABLE_KG_SYNTHESIZED_EDGES}
        ),
    ):
        hits = retriever.retrieve_layer_b(
            query="funds?",
            subscription_id="s",
            profile_id="p",
            top_k=10,
            inferred_confidence_floor=0.75,
        )

    assert {h["kind"] for h in hits} == {"kg_direct", "kg_inferred"}
    kg.one_hop.assert_called_once_with(
        subscription_id="s", profile_id="p", entities=None, top_k=10
    )
    kg.inferred_relations.assert_called_once_with(
        subscription_id="s", profile_id="p", min_confidence=0.75, top_k=10
    )


def test_layer_b_excludes_inferred_when_synthesized_edges_flag_off() -> None:
    kg = _kg_stub(
        direct_rows=[
            {
                "src": "n1",
                "dst": "n2",
                "type": "MENTIONS",
                "evidence": [],
            }
        ]
    )
    retriever = UnifiedRetriever(kg_client=kg, qdrant=MagicMock(), sme=MagicMock())
    # sme_retrieval on, synthesized_edges off — inferred must NOT fire.
    with patch(
        "src.retrieval.unified_retriever.get_flag_resolver",
        return_value=_resolver_stub({ENABLE_SME_RETRIEVAL}),
    ):
        hits = retriever.retrieve_layer_b(
            query="q", subscription_id="s", profile_id="p", top_k=5
        )
    kg.inferred_relations.assert_not_called()
    assert all(h["kind"] != "kg_inferred" for h in hits)


def test_layer_b_excludes_inferred_when_sme_retrieval_flag_off() -> None:
    kg = _kg_stub(
        direct_rows=[{"src": "n1", "dst": "n2", "type": "CITES", "evidence": []}]
    )
    retriever = UnifiedRetriever(kg_client=kg, qdrant=MagicMock(), sme=MagicMock())
    with patch(
        "src.retrieval.unified_retriever.get_flag_resolver",
        return_value=_resolver_stub({ENABLE_KG_SYNTHESIZED_EDGES}),
    ):
        hits = retriever.retrieve_layer_b(
            query="q", subscription_id="s", profile_id="p", top_k=5
        )
    kg.inferred_relations.assert_not_called()
    assert all(h["kind"] != "kg_inferred" for h in hits)


def test_layer_b_default_confidence_floor_is_0p6() -> None:
    kg = _kg_stub(direct_rows=[])
    retriever = UnifiedRetriever(kg_client=kg, qdrant=MagicMock(), sme=MagicMock())
    with patch(
        "src.retrieval.unified_retriever.get_flag_resolver",
        return_value=_resolver_stub(
            {ENABLE_SME_RETRIEVAL, ENABLE_KG_SYNTHESIZED_EDGES}
        ),
    ):
        retriever.retrieve_layer_b(
            query="q", subscription_id="s", profile_id="p", top_k=5
        )
    call = kg.inferred_relations.call_args
    assert call.kwargs["min_confidence"] == 0.6


def test_layer_b_include_inferred_false_suppresses_inferred_even_when_flag_on() -> None:
    kg = _kg_stub(direct_rows=[])
    retriever = UnifiedRetriever(kg_client=kg, qdrant=MagicMock(), sme=MagicMock())
    with patch(
        "src.retrieval.unified_retriever.get_flag_resolver",
        return_value=_resolver_stub(
            {ENABLE_SME_RETRIEVAL, ENABLE_KG_SYNTHESIZED_EDGES}
        ),
    ):
        retriever.retrieve_layer_b(
            query="q",
            subscription_id="s",
            profile_id="p",
            top_k=5,
            include_inferred=False,
        )
    kg.inferred_relations.assert_not_called()


def test_layer_b_hard_profile_isolation_guards() -> None:
    kg = _kg_stub()
    retriever = UnifiedRetriever(kg_client=kg, qdrant=MagicMock(), sme=MagicMock())
    with pytest.raises(ValueError, match="subscription_id"):
        retriever.retrieve_layer_b(
            query="q", subscription_id="", profile_id="p", top_k=5
        )
    with pytest.raises(ValueError, match="profile_id"):
        retriever.retrieve_layer_b(
            query="q", subscription_id="s", profile_id="", top_k=5
        )


def test_layer_b_no_kg_client_returns_empty() -> None:
    retriever = UnifiedRetriever()
    assert (
        retriever.retrieve_layer_b(
            query="q", subscription_id="s", profile_id="p", top_k=5
        )
        == []
    )


def test_layer_b_flag_resolver_uninitialised_returns_direct_only() -> None:
    kg = _kg_stub(
        direct_rows=[{"src": "n1", "dst": "n2", "type": "X", "evidence": []}]
    )
    retriever = UnifiedRetriever(kg_client=kg, qdrant=MagicMock(), sme=MagicMock())
    # Simulate pre-Phase-2 deploy: flag resolver not wired yet.
    with patch(
        "src.retrieval.unified_retriever.get_flag_resolver",
        side_effect=RuntimeError("not initialized"),
    ):
        hits = retriever.retrieve_layer_b(
            query="q", subscription_id="s", profile_id="p", top_k=5
        )
    assert hits and all(h["kind"] == "kg_direct" for h in hits)
    kg.inferred_relations.assert_not_called()
