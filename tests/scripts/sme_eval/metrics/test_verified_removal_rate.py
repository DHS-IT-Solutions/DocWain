from datetime import datetime

from scripts.sme_eval.metrics.verified_removal_rate import VerifiedRemovalRate
from scripts.sme_eval.schema import EvalQuery, EvalResult, LatencyBreakdown


def _result(qid, metadata):
    return EvalResult(
        query=EvalQuery(
            query_id=qid,
            query_text="q",
            intent="analyze",
            profile_domain="finance",
            subscription_id="s",
            profile_id="p",
        ),
        response_text="r",
        sources=[],
        metadata=metadata,
        latency=LatencyBreakdown(total_ms=100.0),
        run_id="run_test",
        captured_at=datetime(2026, 4, 20, 10, 0, 0),
        api_status=200,
    )


def test_no_drops_flagged_returns_one():
    metric = VerifiedRemovalRate()
    results = [_result("a", {}), _result("b", {"citation_verifier_dropped": 0})]
    batch = metric.compute(results)
    assert batch.value == 1.0


def test_some_drops_reduce_value():
    metric = VerifiedRemovalRate()
    results = [
        _result("a", {"citation_verifier_dropped": 0}),
        _result("b", {"citation_verifier_dropped": 2}),
    ]
    batch = metric.compute(results)
    assert batch.value == 0.5
    assert batch.details["num_with_drops"] == 1


def test_empty_batch():
    metric = VerifiedRemovalRate()
    batch = metric.compute([])
    assert batch.value == 1.0
    assert batch.details["num_results"] == 0
