from datetime import datetime

from scripts.sme_eval.metrics.cross_doc_integration_rate import CrossDocIntegrationRate
from scripts.sme_eval.schema import EvalQuery, EvalResult, LatencyBreakdown

_ANALYTICAL_INTENTS = ("analyze", "diagnose", "recommend", "investigate", "compare")


def _result(qid, intent, sources):
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
        sources=sources,
        metadata={},
        latency=LatencyBreakdown(total_ms=100.0),
        run_id="run_test",
        captured_at=datetime(2026, 4, 20, 10, 0, 0),
        api_status=200,
    )


def test_no_analytical_queries_returns_one():
    metric = CrossDocIntegrationRate()
    results = [_result("a", "lookup", [{"doc_id": "d1"}])]
    batch = metric.compute(results)
    assert batch.value == 1.0


def test_integrates_across_docs():
    metric = CrossDocIntegrationRate()
    results = [
        _result("a", "analyze", [{"doc_id": "d1"}, {"doc_id": "d2"}]),
    ]
    batch = metric.compute(results)
    assert batch.value == 1.0
    assert batch.details["num_integrated"] == 1


def test_single_doc_not_integrated():
    metric = CrossDocIntegrationRate()
    results = [_result("a", "analyze", [{"doc_id": "d1"}, {"doc_id": "d1"}])]
    batch = metric.compute(results)
    assert batch.value == 0.0
    assert batch.details["num_integrated"] == 0


def test_mixed_batch():
    metric = CrossDocIntegrationRate()
    results = [
        _result("a", "analyze", [{"doc_id": "d1"}, {"doc_id": "d2"}]),
        _result("b", "diagnose", [{"doc_id": "d3"}]),
        _result("c", "recommend", [{"doc_id": "d4"}, {"doc_id": "d5"}, {"doc_id": "d6"}]),
        _result("d", "lookup", [{"doc_id": "d7"}]),
    ]
    batch = metric.compute(results)
    assert batch.details["num_analytical"] == 3
    assert batch.details["num_integrated"] == 2
    assert abs(batch.value - 2 / 3) < 1e-9
