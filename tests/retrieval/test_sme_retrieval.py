"""Layer C retrieval tests — SME artifact snippets from Qdrant.

Uses a fake Qdrant client (MagicMock) so nothing talks to a real service
and a stub flag resolver so flag state is explicit in each case.

Contract pinned here:

* With ``enable_sme_retrieval`` OFF, the method returns ``[]`` without
  ever calling Qdrant or the embedder.
* With the flag ON, the method computes a dense vector via the embedder,
  searches the per-subscription collection ``sme_artifacts_{sub}``, and
  returns rows normalized to the ``{kind, artifact_type, ...}`` shape.
* ``subscription_id`` / ``profile_id`` are added to the Qdrant filter on
  every call (profile isolation is hard).
* ``artifact_types`` adds an extra ``artifact_type`` ``MatchAny`` filter.
* Empty ``subscription_id`` or ``profile_id`` raises ``ValueError``.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.config.feature_flags import ENABLE_SME_RETRIEVAL
from src.retrieval.sme_retrieval import SMERetrieval


def _resolver_stub(enabled: bool) -> MagicMock:
    r = MagicMock()
    r.is_enabled.return_value = enabled
    return r


def _qdrant_row(payload: dict, score: float) -> SimpleNamespace:
    return SimpleNamespace(payload=payload, score=score, id=payload.get("snippet_id"))


def test_layer_c_hard_filter_and_artifact_type_constraint() -> None:
    qdrant = MagicMock()
    embedder = MagicMock()
    embedder.embed.return_value = [0.1] * 768
    qdrant.search.return_value = [
        _qdrant_row(
            payload={
                "artifact_type": "insight_index",
                "snippet_id": "insight:trend:0",
                "text": "Rev QoQ rose.",
                "confidence": 0.8,
                "evidence": [{"doc_id": "d1", "chunk_id": "c1"}],
                "subscription_id": "sub_a",
                "profile_id": "prof_x",
            },
            score=0.92,
        )
    ]
    r = SMERetrieval(qdrant=qdrant, embedder=embedder)
    with patch(
        "src.retrieval.sme_retrieval.get_flag_resolver",
        return_value=_resolver_stub(True),
    ):
        hits = r.retrieve(
            query="rev",
            subscription_id="sub_a",
            profile_id="prof_x",
            artifact_types=["insight_index"],
            top_k=10,
        )
    assert len(hits) == 1
    assert hits[0]["kind"] == "sme_artifact"
    assert hits[0]["artifact_type"] == "insight_index"
    call_kwargs = qdrant.search.call_args.kwargs
    assert call_kwargs["collection_name"] == "sme_artifacts_sub_a"
    must_keys = {m.key for m in call_kwargs["query_filter"].must}
    assert {"subscription_id", "profile_id", "artifact_type"} <= must_keys


def test_layer_c_flag_off_returns_empty_without_calling_services() -> None:
    qdrant = MagicMock()
    embedder = MagicMock()
    r = SMERetrieval(qdrant=qdrant, embedder=embedder)
    with patch(
        "src.retrieval.sme_retrieval.get_flag_resolver",
        return_value=_resolver_stub(False),
    ):
        hits = r.retrieve(
            query="q",
            subscription_id="s",
            profile_id="p",
            artifact_types=None,
            top_k=10,
        )
    assert hits == []
    qdrant.search.assert_not_called()
    embedder.embed.assert_not_called()


def test_layer_c_cross_sub_guard_missing_subscription_raises() -> None:
    r = SMERetrieval(qdrant=MagicMock(), embedder=MagicMock())
    with pytest.raises(ValueError, match="subscription_id"):
        r.retrieve(
            query="q",
            subscription_id="",
            profile_id="p",
            artifact_types=None,
            top_k=5,
        )


def test_layer_c_cross_sub_guard_missing_profile_raises() -> None:
    r = SMERetrieval(qdrant=MagicMock(), embedder=MagicMock())
    with pytest.raises(ValueError, match="profile_id"):
        r.retrieve(
            query="q",
            subscription_id="s",
            profile_id="",
            artifact_types=None,
            top_k=5,
        )


def test_layer_c_no_artifact_type_filter_when_unspecified() -> None:
    qdrant = MagicMock()
    qdrant.search.return_value = []
    embedder = MagicMock()
    embedder.embed.return_value = [0.0] * 8
    r = SMERetrieval(qdrant=qdrant, embedder=embedder)
    with patch(
        "src.retrieval.sme_retrieval.get_flag_resolver",
        return_value=_resolver_stub(True),
    ):
        r.retrieve(
            query="q",
            subscription_id="s",
            profile_id="p",
            artifact_types=None,
            top_k=7,
        )
    must_keys = {m.key for m in qdrant.search.call_args.kwargs["query_filter"].must}
    assert must_keys == {"subscription_id", "profile_id"}


def test_layer_c_resolver_uninitialised_returns_empty() -> None:
    qdrant = MagicMock()
    embedder = MagicMock()
    r = SMERetrieval(qdrant=qdrant, embedder=embedder)
    with patch(
        "src.retrieval.sme_retrieval.get_flag_resolver",
        side_effect=RuntimeError("not initialised"),
    ):
        hits = r.retrieve(
            query="q",
            subscription_id="s",
            profile_id="p",
            artifact_types=None,
            top_k=5,
        )
    assert hits == []
    qdrant.search.assert_not_called()


def test_layer_c_normalises_row_shape() -> None:
    qdrant = MagicMock()
    embedder = MagicMock()
    embedder.embed.return_value = [0.0] * 8
    qdrant.search.return_value = [
        _qdrant_row(
            payload={
                "artifact_type": "dossier",
                "snippet_id": "dossier:overview:0",
                "text": "Overview narrative",
                "confidence": 0.7,
                "evidence": [],
            },
            score=0.41,
        )
    ]
    r = SMERetrieval(qdrant=qdrant, embedder=embedder)
    with patch(
        "src.retrieval.sme_retrieval.get_flag_resolver",
        return_value=_resolver_stub(True),
    ):
        hits = r.retrieve(
            query="q",
            subscription_id="s",
            profile_id="p",
            artifact_types=None,
            top_k=5,
        )
    assert hits[0]["snippet_id"] == "dossier:overview:0"
    assert hits[0]["text"] == "Overview narrative"
    assert hits[0]["confidence"] == 0.7
    assert hits[0]["score"] == 0.41
