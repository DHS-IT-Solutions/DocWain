"""Unit tests for src/embedding/feedback.py"""

from __future__ import annotations

import pytest

from src.embedding.feedback import RetrievalFeedbackTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tracker() -> RetrievalFeedbackTracker:
    return RetrievalFeedbackTracker()


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------


class TestInitialState:
    def test_empty_metrics(self):
        t = _tracker()
        m = t.get_metrics()
        assert m["total_queries"] == 0
        assert m["precision_at_k"] == 0.0
        assert m["recall_at_k"] == 0.0
        assert m["mrr"] == 0.0

    def test_empty_hard_negatives(self):
        assert _tracker().get_hard_negatives() == []


# ---------------------------------------------------------------------------
# record_outcome — metric calculations
# ---------------------------------------------------------------------------


class TestRecordOutcomePrecision:
    def test_perfect_precision(self):
        t = _tracker()
        t.record_outcome("q", ["a", "b"], ["a", "b"])
        assert t.get_metrics()["precision_at_k"] == 1.0

    def test_zero_precision(self):
        t = _tracker()
        t.record_outcome("q", ["x", "y"], ["a", "b"])
        assert t.get_metrics()["precision_at_k"] == 0.0

    def test_partial_precision(self):
        t = _tracker()
        t.record_outcome("q", ["a", "x"], ["a", "b"])
        assert t.get_metrics()["precision_at_k"] == pytest.approx(0.5)

    def test_empty_retrieved(self):
        t = _tracker()
        t.record_outcome("q", [], ["a"])
        assert t.get_metrics()["precision_at_k"] == 0.0


class TestRecordOutcomeRecall:
    def test_perfect_recall(self):
        t = _tracker()
        t.record_outcome("q", ["a", "b", "c"], ["a", "b"])
        assert t.get_metrics()["recall_at_k"] == 1.0

    def test_zero_recall(self):
        t = _tracker()
        t.record_outcome("q", ["x"], ["a", "b"])
        assert t.get_metrics()["recall_at_k"] == 0.0

    def test_partial_recall(self):
        t = _tracker()
        t.record_outcome("q", ["a", "x"], ["a", "b"])
        assert t.get_metrics()["recall_at_k"] == pytest.approx(0.5)

    def test_empty_relevant(self):
        t = _tracker()
        t.record_outcome("q", ["a"], [])
        assert t.get_metrics()["recall_at_k"] == 0.0


class TestRecordOutcomeMRR:
    def test_first_doc_relevant(self):
        t = _tracker()
        t.record_outcome("q", ["a", "b", "c"], ["a"])
        assert t.get_metrics()["mrr"] == pytest.approx(1.0)

    def test_second_doc_relevant(self):
        t = _tracker()
        t.record_outcome("q", ["x", "a", "c"], ["a"])
        assert t.get_metrics()["mrr"] == pytest.approx(0.5)

    def test_third_doc_relevant(self):
        t = _tracker()
        t.record_outcome("q", ["x", "y", "a"], ["a"])
        assert t.get_metrics()["mrr"] == pytest.approx(1.0 / 3)

    def test_no_relevant_in_retrieved(self):
        t = _tracker()
        t.record_outcome("q", ["x", "y"], ["a"])
        assert t.get_metrics()["mrr"] == 0.0

    def test_empty_retrieved_mrr_zero(self):
        t = _tracker()
        t.record_outcome("q", [], ["a"])
        assert t.get_metrics()["mrr"] == 0.0

    def test_uses_first_relevant_rank(self):
        """When multiple relevant docs exist, MRR uses the first one found."""
        t = _tracker()
        # "b" is rank 2 (0-indexed 1), "c" is rank 3 (0-indexed 2)
        t.record_outcome("q", ["x", "b", "c"], {"b", "c"})
        assert t.get_metrics()["mrr"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Hard negatives
# ---------------------------------------------------------------------------


class TestHardNegatives:
    def test_irrelevant_docs_captured(self):
        t = _tracker()
        t.record_outcome("q", ["a", "b", "c"], ["a"])
        hn = t.get_hard_negatives()
        assert "b" in hn
        assert "c" in hn
        assert "a" not in hn

    def test_sorted_by_frequency_descending(self):
        t = _tracker()
        # "b" appears as hard negative twice; "c" once
        t.record_outcome("q1", ["b", "c"], ["x"])
        t.record_outcome("q2", ["b"], ["x"])
        hn = t.get_hard_negatives()
        assert hn[0] == "b"

    def test_no_hard_negatives_when_all_relevant(self):
        t = _tracker()
        t.record_outcome("q", ["a", "b"], ["a", "b"])
        assert t.get_hard_negatives() == []

    def test_deterministic_tie_breaking(self):
        """Ties broken alphabetically so output is deterministic."""
        t = _tracker()
        t.record_outcome("q", ["z", "a", "m"], [])
        hn = t.get_hard_negatives()
        assert hn == ["a", "m", "z"]


# ---------------------------------------------------------------------------
# Averaging across multiple queries
# ---------------------------------------------------------------------------


class TestAveraging:
    def test_total_queries_count(self):
        t = _tracker()
        t.record_outcome("q1", ["a"], ["a"])
        t.record_outcome("q2", ["b"], ["b"])
        assert t.get_metrics()["total_queries"] == 2

    def test_averaged_precision(self):
        t = _tracker()
        # query 1: precision = 1.0
        t.record_outcome("q1", ["a"], ["a"])
        # query 2: precision = 0.0
        t.record_outcome("q2", ["x"], ["b"])
        assert t.get_metrics()["precision_at_k"] == pytest.approx(0.5)

    def test_averaged_recall(self):
        t = _tracker()
        t.record_outcome("q1", ["a", "b"], ["a", "b"])  # recall = 1.0
        t.record_outcome("q2", ["x"], ["a", "b"])        # recall = 0.0
        assert t.get_metrics()["recall_at_k"] == pytest.approx(0.5)

    def test_averaged_mrr(self):
        t = _tracker()
        t.record_outcome("q1", ["a"], ["a"])   # mrr = 1.0
        t.record_outcome("q2", ["x", "a"], ["a"])  # mrr = 0.5
        assert t.get_metrics()["mrr"] == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# clear()
# ---------------------------------------------------------------------------


class TestClear:
    def test_clear_resets_metrics(self):
        t = _tracker()
        t.record_outcome("q", ["a", "b"], ["a"])
        t.clear()
        m = t.get_metrics()
        assert m["total_queries"] == 0
        assert m["precision_at_k"] == 0.0
        assert m["recall_at_k"] == 0.0
        assert m["mrr"] == 0.0

    def test_clear_resets_hard_negatives(self):
        t = _tracker()
        t.record_outcome("q", ["x", "y"], ["a"])
        t.clear()
        assert t.get_hard_negatives() == []

    def test_record_after_clear(self):
        t = _tracker()
        t.record_outcome("q1", ["a"], ["b"])
        t.clear()
        t.record_outcome("q2", ["a"], ["a"])
        m = t.get_metrics()
        assert m["total_queries"] == 1
        assert m["precision_at_k"] == 1.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_relevant_ids_as_set(self):
        """relevant_ids can be any iterable (set, list, generator)."""
        t = _tracker()
        t.record_outcome("q", ["a", "b"], {"a"})
        assert t.get_metrics()["precision_at_k"] == pytest.approx(0.5)

    def test_query_string_ignored_in_calculations(self):
        """Different query strings don't affect metric values."""
        t1, t2 = _tracker(), _tracker()
        t1.record_outcome("query A", ["a"], ["a"])
        t2.record_outcome("query B", ["a"], ["a"])
        assert t1.get_metrics() == t2.get_metrics()

    def test_duplicate_retrieved_ids(self):
        """Duplicate entries in retrieved_ids are deduplicated via set ops."""
        t = _tracker()
        # "a" appears twice but the set intersection still counts it once
        t.record_outcome("q", ["a", "a", "b"], ["a"])
        m = t.get_metrics()
        # retrieved_set = {"a", "b"}, relevant_set = {"a"}
        # precision = 1/2, recall = 1/1
        assert m["precision_at_k"] == pytest.approx(0.5)
        assert m["recall_at_k"] == pytest.approx(1.0)
