"""Tests for :class:`CrossEncoderReranker` — SME stage-2 wrapper."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.retrieval.reranker import (
    DEFAULT_CROSS_ENCODER_MODEL,
    CrossEncoderReranker,
    RerankCandidate,
    RerankScore,
)


def _cand(ident: str, text: str) -> RerankCandidate:
    return RerankCandidate(id=ident, text=text)


def test_default_model_name_matches_spec() -> None:
    r = CrossEncoderReranker()
    assert r.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert DEFAULT_CROSS_ENCODER_MODEL == r.model_name


def test_loads_model_lazily_on_first_rerank() -> None:
    with patch("src.retrieval.reranker._load_model") as loader:
        loader.return_value = MagicMock()
        r = CrossEncoderReranker(model_name="m")
        assert loader.call_count == 0
        r.rerank("q", [_cand("a", "x")])
        assert loader.call_count == 1
        # Second call reuses the cached model.
        r.rerank("q", [_cand("b", "y")])
        assert loader.call_count == 1


def test_preserves_top_k_ordering() -> None:
    m = MagicMock()
    m.predict.return_value = [0.1, 0.9, 0.5]
    with patch("src.retrieval.reranker._load_model", return_value=m):
        out = CrossEncoderReranker(model_name="m").rerank(
            "q",
            [_cand("a", "1"), _cand("b", "2"), _cand("c", "3")],
            top_k=2,
        )
    assert [c.id for c in out] == ["b", "c"]


def test_empty_candidates_short_circuits_without_loading() -> None:
    with patch("src.retrieval.reranker._load_model") as loader:
        out = CrossEncoderReranker(model_name="m").rerank("q", [])
    assert out == []
    loader.assert_not_called()


def test_caps_at_candidate_count_when_top_k_larger() -> None:
    m = MagicMock()
    m.predict.return_value = [0.5, 0.6]
    with patch("src.retrieval.reranker._load_model", return_value=m):
        out = CrossEncoderReranker(model_name="m").rerank(
            "q",
            [_cand("a", "1"), _cand("b", "2")],
            top_k=10,
        )
    assert len(out) == 2
    assert [c.id for c in out] == ["b", "a"]


def test_rerank_with_scores_returns_scores() -> None:
    m = MagicMock()
    m.predict.return_value = [0.4, 0.2]
    with patch("src.retrieval.reranker._load_model", return_value=m):
        out = CrossEncoderReranker(model_name="m").rerank_with_scores(
            "q",
            [_cand("a", "1"), _cand("b", "2")],
        )
    assert [rs.candidate.id for rs in out] == ["a", "b"]
    assert [rs.score for rs in out] == [0.4, 0.2]
    assert all(isinstance(rs, RerankScore) for rs in out)


def test_predict_receives_query_candidate_pairs() -> None:
    m = MagicMock()
    m.predict.return_value = [0.3, 0.7]
    with patch("src.retrieval.reranker._load_model", return_value=m):
        CrossEncoderReranker(model_name="m").rerank(
            "how does X work?",
            [_cand("a", "about x"), _cand("b", "about y")],
        )
    pairs = m.predict.call_args[0][0]
    assert pairs == [("how does X work?", "about x"), ("how does X work?", "about y")]


def test_rerank_with_scores_empty_candidates_short_circuits() -> None:
    with patch("src.retrieval.reranker._load_model") as loader:
        out = CrossEncoderReranker(model_name="m").rerank_with_scores("q", [])
    assert out == []
    loader.assert_not_called()
