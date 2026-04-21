"""Tests for :class:`HybridSearcher` — RRF fusion + filter forwarding."""
from __future__ import annotations

from unittest.mock import MagicMock

from src.retrieval.hybrid_search import HybridConfig, HybridSearcher


def _make(dense, sparse):
    q = MagicMock()
    q.search_dense.return_value = dense
    q.search_sparse.return_value = sparse
    return HybridSearcher(q, HybridConfig(rrf_k=60)), q


def test_rrf_fuses_dense_and_sparse() -> None:
    # "b" shows up in both lists near the top → wins. "c" and "d" each show
    # up in only one list, so rank by remaining RRF contribution.
    searcher, _ = _make(
        [{"id": "a"}, {"id": "b"}, {"id": "c"}],
        [{"id": "b"}, {"id": "d"}, {"id": "a"}],
    )
    out = searcher.search(query_text="x", collection="c", top_k=5)
    ids = [r.item_id for r in out]
    assert ids[0] == "b"
    assert set(ids) == {"a", "b", "c", "d"}


def test_falls_back_to_dense_when_sparse_unavailable() -> None:
    q = MagicMock()
    q.search_sparse.side_effect = NotImplementedError
    q.search_dense.return_value = [{"id": "a"}]
    searcher = HybridSearcher(q, HybridConfig())
    out = searcher.search(query_text="x", collection="c", top_k=5)
    assert [r.item_id for r in out] == ["a"]


def test_respects_top_k_cap() -> None:
    searcher, _ = _make(
        [{"id": f"d{i}"} for i in range(50)],
        [{"id": f"s{i}"} for i in range(50)],
    )
    out = searcher.search(query_text="x", collection="c", top_k=10)
    assert len(out) == 10


def test_filter_forwarded_to_both_backends() -> None:
    searcher, q = _make([], [])
    f = {
        "must": [
            {"key": "subscription_id", "value": "sub_a"},
            {"key": "profile_id", "value": "prof_x"},
        ]
    }
    searcher.search(query_text="x", collection="c", top_k=5, query_filter=f)
    assert q.search_dense.call_args.kwargs["query_filter"] == f
    assert q.search_sparse.call_args.kwargs["query_filter"] == f


def test_per_backend_topk_read_from_config() -> None:
    searcher, q = _make([], [])
    searcher._c = HybridConfig(k_dense=17, k_sparse=23)
    searcher.search(query_text="x", collection="c", top_k=5)
    assert q.search_dense.call_args.kwargs["top_k"] == 17
    assert q.search_sparse.call_args.kwargs["top_k"] == 23


def test_payload_preserved() -> None:
    searcher, _ = _make(
        [{"id": "a", "payload": {"meta": 1}}],
        [{"id": "a", "payload": {"other": 2}}],
    )
    out = searcher.search(query_text="x", collection="c", top_k=5)
    # Dense fills first, so its payload is preserved.
    assert out[0].payload == {"meta": 1}
    assert out[0].dense_rank == 0
    assert out[0].sparse_rank == 0


def test_no_filter_defaults_to_none() -> None:
    searcher, q = _make([], [])
    searcher.search(query_text="x", collection="c", top_k=5)
    assert q.search_dense.call_args.kwargs["query_filter"] is None
    assert q.search_sparse.call_args.kwargs["query_filter"] is None
