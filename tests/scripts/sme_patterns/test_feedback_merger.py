"""Lean tests for feedback merger."""
from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

from scripts.sme_patterns.feedback_merger import merge_feedback
from scripts.sme_patterns.schema import QueryFeedback, QueryRun


def _run(qid, *, feedback=None):
    return QueryRun(
        subscription_id="sub_a",
        profile_id="prof_a",
        profile_domain="finance",
        query_id=qid,
        query_text="analyze Q3",
        query_fingerprint="abc",
        intent="analyze",
        adapter_version="1.0.0",
        captured_at=datetime(2026, 4, 5, 10, 0, 0),
        feedback=feedback,
    )


def test_merge_preserves_explicit_and_fills_from_redis_aggregates():
    explicit_run = _run("q1", feedback=QueryFeedback(rating=1, source="feedback_tracker"))
    missing_runs = [_run("q2"), _run("q3")]
    tracker = MagicMock()
    tracker.get_profile_metrics.return_value = {
        "total_queries": 2,
        "low_confidence_count": 1,  # 50% low confidence -> implicit signal fires
    }
    out = merge_feedback([explicit_run, *missing_runs], tracker)
    assert out[0].feedback.rating == 1
    for q in out[1:]:
        assert q.feedback is not None
        assert q.feedback.source == "implicit"
        assert q.feedback.rating == -1


def test_merge_handles_low_confidence_below_threshold_and_tracker_errors():
    tracker_ok = MagicMock()
    tracker_ok.get_profile_metrics.return_value = {
        "total_queries": 10,
        "low_confidence_count": 0,
    }
    out = merge_feedback([_run("q1")], tracker_ok)
    assert out[0].feedback is None

    tracker_error = MagicMock()
    tracker_error.get_profile_metrics.side_effect = RuntimeError("redis down")
    out2 = merge_feedback([_run("q1")], tracker_error)
    assert out2[0].feedback is None
