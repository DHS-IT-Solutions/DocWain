from datetime import datetime
from unittest.mock import MagicMock

from scripts.sme_eval.metrics.sme_persona_consistency import (
    SmePersonaConsistency,
    _REFERENCE_PERSONAS,
)
from scripts.sme_eval.schema import EvalQuery, EvalResult, LatencyBreakdown


def _result(qid, domain, response):
    return EvalResult(
        query=EvalQuery(
            query_id=qid,
            query_text="q",
            intent="analyze",
            profile_domain=domain,
            subscription_id="s",
            profile_id="p",
        ),
        response_text=response,
        sources=[],
        metadata={},
        latency=LatencyBreakdown(total_ms=100.0),
        run_id="run_test",
        captured_at=datetime(2026, 4, 20, 10, 0, 0),
        api_status=200,
    )


def test_reference_personas_cover_all_domains():
    for dom in ("finance", "legal", "hr", "medical", "it_support", "generic"):
        assert dom in _REFERENCE_PERSONAS


def test_metric_aggregates_judge_scores():
    judge = MagicMock(return_value=4.2)
    metric = SmePersonaConsistency(judge_fn=judge)
    results = [
        _result("a", "finance", "Q3 revenue rose 12%, driven by..."),
        _result("b", "legal", "The contract obliges..."),
    ]
    batch = metric.compute(results)
    assert batch.value == 4.2
    assert batch.details["num_judged"] == 2
    assert judge.call_count == 2


def test_metric_handles_judge_failure():
    calls = [5.0, Exception("gateway down")]

    def judge(*_args, **_kwargs):
        val = calls.pop(0)
        if isinstance(val, Exception):
            raise val
        return val

    metric = SmePersonaConsistency(judge_fn=judge)
    results = [
        _result("a", "finance", "Answer A"),
        _result("b", "finance", "Answer B"),
    ]
    batch = metric.compute(results)
    # One failed judgment — excluded from numerator and denominator
    assert batch.details["num_judged"] == 1
    assert batch.details["num_failed"] == 1
    assert batch.value == 5.0


def test_metric_empty_batch():
    judge = MagicMock()
    metric = SmePersonaConsistency(judge_fn=judge)
    batch = metric.compute([])
    assert batch.value == 0.0
    assert batch.details["num_judged"] == 0
    judge.assert_not_called()
