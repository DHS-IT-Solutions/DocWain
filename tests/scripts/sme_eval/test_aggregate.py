from datetime import datetime

from scripts.sme_eval.aggregate import aggregate_latency_per_intent
from scripts.sme_eval.schema import EvalQuery, EvalResult, LatencyBreakdown


def _result(qid, intent, total_ms):
    return EvalResult(
        query=EvalQuery(
            query_id=qid,
            query_text="q",
            intent=intent,
            profile_domain="finance",
            subscription_id="s",
            profile_id="p",
        ),
        response_text="r",
        sources=[],
        metadata={},
        latency=LatencyBreakdown(total_ms=total_ms),
        run_id="run_test",
        captured_at=datetime(2026, 4, 20, 10, 0, 0),
        api_status=200,
    )


def test_aggregates_per_intent():
    results = [
        _result("l1", "lookup", 1000),
        _result("l2", "lookup", 2000),
        _result("l3", "lookup", 3000),
        _result("a1", "analyze", 5000),
        _result("a2", "analyze", 7000),
    ]
    agg = aggregate_latency_per_intent(results)
    assert "lookup" in agg
    assert "analyze" in agg
    assert agg["lookup"]["p50"] == 2000
    assert agg["analyze"]["p50"] == 6000


def test_empty_returns_empty_dict():
    assert aggregate_latency_per_intent([]) == {}


def test_skips_failed_calls():
    """API status != 200 excluded from latency stats."""
    r1 = _result("a", "lookup", 1000)
    r2 = _result("b", "lookup", 99999)
    r2 = r2.model_copy(update={"api_status": 500})
    agg = aggregate_latency_per_intent([r1, r2])
    assert agg["lookup"]["count"] == 1
    assert agg["lookup"]["p50"] == 1000
