from datetime import datetime

from scripts.sme_eval.metrics.insight_novelty import InsightNovelty
from scripts.sme_eval.schema import EvalQuery, EvalResult, LatencyBreakdown


def _result(qid, intent, response, source_excerpts):
    return EvalResult(
        query=EvalQuery(
            query_id=qid,
            query_text="q",
            intent=intent,
            profile_domain="finance",
            subscription_id="s",
            profile_id="p",
        ),
        response_text=response,
        sources=[
            {"doc_id": f"d{i}", "chunk_id": f"c{i}", "excerpt": ex}
            for i, ex in enumerate(source_excerpts)
        ],
        metadata={},
        latency=LatencyBreakdown(total_ms=100.0),
        run_id="run_test",
        captured_at=datetime(2026, 4, 20, 10, 0, 0),
        api_status=200,
    )


def test_skips_non_analytical():
    metric = InsightNovelty()
    results = [_result("a", "lookup", "Answer is 42.", ["Source says 42."])]
    batch = metric.compute(results)
    # Non-analytical queries don't contribute; value is 0.0 (no novelty to measure)
    assert batch.details["num_analytical"] == 0


def test_response_fully_lifted_has_zero_novelty():
    metric = InsightNovelty()
    results = [
        _result(
            "a",
            "analyze",
            "Revenue rose 12 percent in Q3.",
            ["Revenue rose 12 percent in Q3."],
        )
    ]
    batch = metric.compute(results)
    assert batch.value < 0.1  # almost entirely lifted


def test_response_with_new_ngrams_has_some_novelty():
    metric = InsightNovelty()
    results = [
        _result(
            "a",
            "analyze",
            "Revenue rose 12 percent. This indicates accelerating growth from prior quarters.",
            ["Revenue rose 12 percent."],
        )
    ]
    batch = metric.compute(results)
    assert batch.value > 0.2


def test_multiple_results_aggregated():
    metric = InsightNovelty()
    results = [
        _result("a", "analyze", "Revenue rose 12%. Growth is accelerating.", ["Revenue rose 12%."]),
        _result("b", "diagnose", "The error is caused by auth token expiry.", ["Log shows token expired."]),
    ]
    batch = metric.compute(results)
    assert batch.details["num_analytical"] == 2
    assert 0.0 <= batch.value <= 1.0
