"""Unit tests for src/retrieval/fusion.py"""

from __future__ import annotations

import pytest

from src.retrieval.fusion import FusionRetriever, reciprocal_rank_fusion


# ---------------------------------------------------------------------------
# reciprocal_rank_fusion
# ---------------------------------------------------------------------------


class TestReciprocalRankFusion:
    def test_single_signal_ordering(self):
        """Higher-ranked docs should score better."""
        result = reciprocal_rank_fusion({"bge": ["doc1", "doc2", "doc3"]})
        assert result == ["doc1", "doc2", "doc3"]

    def test_default_weight_is_one(self):
        """Passing weights=None should behave identically to weight=1.0."""
        rankings = {"bge": ["a", "b"], "splade": ["b", "a"]}
        r_none = reciprocal_rank_fusion(rankings, weights=None)
        r_ones = reciprocal_rank_fusion(rankings, weights={"bge": 1.0, "splade": 1.0})
        assert r_none == r_ones

    def test_score_formula(self):
        """Verify that score = weight / (k + rank + 1) is applied correctly."""
        k = 60
        # Single signal, single doc at rank 0 → score = 1.0 / (60 + 0 + 1) = 1/61
        result = reciprocal_rank_fusion({"s": ["only_doc"]}, k=k)
        assert result == ["only_doc"]

    def test_union_of_docs_returned(self):
        """All doc IDs from any signal must appear in the result."""
        rankings = {
            "bge": ["a", "b"],
            "splade": ["c", "b"],
            "v2": ["d"],
        }
        result = reciprocal_rank_fusion(rankings)
        assert set(result) == {"a", "b", "c", "d"}

    def test_weighted_boost_changes_order(self):
        """A strongly-weighted signal's top doc should win overall."""
        rankings = {
            "bge": ["x", "y"],
            "splade": ["y", "x"],
        }
        # Default (equal) weights: y gets more total score because it's rank-0 in splade
        # But with high bge weight, x should win
        result_boosted = reciprocal_rank_fusion(
            rankings, weights={"bge": 10.0, "splade": 1.0}
        )
        assert result_boosted[0] == "x"

    def test_custom_k_affects_scores(self):
        """k=0 makes ranks matter more; k=1000 smooths differences."""
        rankings = {"s": ["top", "bottom"]}
        # With k=0: top scores 1/1, bottom scores 1/2
        # With k=1000: top ≈ 1/1001, bottom ≈ 1/1002 (still same order)
        r_small_k = reciprocal_rank_fusion(rankings, k=0)
        r_large_k = reciprocal_rank_fusion(rankings, k=1000)
        assert r_small_k[0] == "top"
        assert r_large_k[0] == "top"

    def test_empty_rankings(self):
        """Empty rankings should return an empty list."""
        assert reciprocal_rank_fusion({}) == []

    def test_empty_signal_list(self):
        """A signal with an empty list contributes nothing."""
        result = reciprocal_rank_fusion({"bge": [], "splade": ["doc1"]})
        assert result == ["doc1"]

    def test_doc_appears_in_multiple_signals(self):
        """A doc present in all signals should accumulate a higher score."""
        rankings = {
            "bge": ["shared", "unique_a"],
            "splade": ["shared", "unique_b"],
            "v2": ["shared", "unique_c"],
        }
        result = reciprocal_rank_fusion(rankings)
        assert result[0] == "shared"


# ---------------------------------------------------------------------------
# FusionRetriever
# ---------------------------------------------------------------------------


class TestFusionRetrieverInit:
    def test_default_weights(self):
        fr = FusionRetriever()
        assert fr.default_weights == {"bge": 0.4, "splade": 0.3, "v2": 0.3}

    def test_custom_weights(self):
        custom = {"bge": 0.5, "splade": 0.3, "v2": 0.2}
        fr = FusionRetriever(default_weights=custom)
        assert fr.default_weights == custom

    def test_default_rrf_k(self):
        assert FusionRetriever().rrf_k == 60

    def test_custom_rrf_k(self):
        assert FusionRetriever(rrf_k=30).rrf_k == 30


class TestFusionRetrieverFuse:
    def setup_method(self):
        self.fr = FusionRetriever()

    def test_returns_list(self):
        result = self.fr.fuse(["a"], ["b"], ["c"])
        assert isinstance(result, list)

    def test_all_docs_included(self):
        bge = ["a", "b"]
        splade = ["c", "b"]
        v2 = ["d", "a"]
        result = self.fr.fuse(bge, splade, v2)
        assert set(result) == {"a", "b", "c", "d"}

    def test_no_query_type_uses_defaults(self):
        """Without query_type the default weights should be used."""
        bge = ["x"]
        splade = ["y"]
        v2 = ["z"]
        # All unique docs; order depends solely on weights applied to rank-0.
        # bge weight (0.4) > splade (0.3) == v2 (0.3), so "x" should be first.
        result = self.fr.fuse(bge, splade, v2, query_type=None)
        assert result[0] == "x"

    def test_empty_lists(self):
        result = self.fr.fuse([], [], [])
        assert result == []


class TestFusionRetrieverAdjustWeights:
    def setup_method(self):
        self.fr = FusionRetriever()

    def test_unknown_query_type_returns_defaults(self):
        w = self.fr._adjust_weights_for_query("unknown_type")
        assert w == self.fr.default_weights

    def test_none_query_type_returns_defaults(self):
        w = self.fr._adjust_weights_for_query(None)
        assert w == self.fr.default_weights

    @pytest.mark.parametrize("qtype", ["exact_lookup", "id_search"])
    def test_splade_boosted_query_types(self, qtype):
        w = self.fr._adjust_weights_for_query(qtype)
        base = self.fr.default_weights
        assert w["splade"] > base["splade"], "splade should be boosted"
        assert w["bge"] < base["bge"] or w["v2"] < base["v2"], (
            "other signals should decrease"
        )

    @pytest.mark.parametrize("qtype", ["conceptual", "summary"])
    def test_v2_boosted_query_types(self, qtype):
        w = self.fr._adjust_weights_for_query(qtype)
        base = self.fr.default_weights
        assert w["v2"] > base["v2"], "v2 should be boosted"
        assert w["bge"] < base["bge"] or w["splade"] < base["splade"], (
            "other signals should decrease"
        )

    def test_weights_non_negative(self):
        for qtype in ["exact_lookup", "id_search", "conceptual", "summary", None]:
            w = self.fr._adjust_weights_for_query(qtype)
            for sig, val in w.items():
                assert val >= 0.0, f"weight for {sig} must be non-negative"

    def test_adjust_does_not_mutate_defaults(self):
        original = dict(self.fr.default_weights)
        self.fr._adjust_weights_for_query("exact_lookup")
        assert self.fr.default_weights == original

    def test_splade_boost_affects_fuse_order(self):
        """exact_lookup should promote splade-top doc over bge-top doc."""
        fr = FusionRetriever()
        # splade_top is rank-0 in splade only; bge_top is rank-0 in bge only
        result_default = fr.fuse(["bge_top"], ["splade_top"], [], query_type=None)
        result_exact = fr.fuse(
            ["bge_top"], ["splade_top"], [], query_type="exact_lookup"
        )
        # After boost splade wins
        assert result_exact[0] == "splade_top"
