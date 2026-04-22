from datetime import datetime

from scripts.sme_eval.metrics.recommendation_groundedness import (
    RecommendationGroundedness,
    extract_recommendation_sentences,
)
from scripts.sme_eval.schema import EvalQuery, EvalResult, LatencyBreakdown


def _result(qid, intent, response, sources=None):
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
        sources=sources or [{"doc_id": "d1", "chunk_id": "c1"}],
        metadata={},
        latency=LatencyBreakdown(total_ms=100.0),
        run_id="run_test",
        captured_at=datetime(2026, 4, 20, 10, 0, 0),
        api_status=200,
    )


def test_extract_recommendation_sentences_picks_imperatives():
    text = (
        "Revenue is up 12%. Consider consolidating your SaaS vendors. "
        "We should reduce redundant expenses."
    )
    sents = extract_recommendation_sentences(text)
    assert any("consolidat" in s.lower() for s in sents)
    assert any("reduce" in s.lower() for s in sents)


def test_extract_recommendation_sentences_ignores_descriptive():
    text = "Revenue is up 12%. Expenses rose 5%. The trend is stable."
    sents = extract_recommendation_sentences(text)
    # No imperatives — should return empty
    assert sents == []


def test_metric_skips_non_recommend_intent():
    """Only recommend-intent results count toward this metric."""
    metric = RecommendationGroundedness()
    results = [_result("a", "lookup", "Some lookup answer.")]
    batch = metric.compute(results)
    # With no recommend-intent queries, value defaults to 1.0 (nothing to fail)
    assert batch.value == 1.0
    assert batch.details["num_recommend_queries"] == 0


def test_metric_grounded_when_recommendations_cite_sources():
    metric = RecommendationGroundedness()
    results = [
        _result(
            "rec1",
            "recommend",
            "Consolidate vendors (see invoice_2026_03) to reduce cost by 12%.",
            sources=[{"doc_id": "invoice_2026_03", "chunk_id": "c1"}],
        )
    ]
    batch = metric.compute(results)
    assert batch.value == 1.0
    assert batch.details["num_grounded"] == 1


def test_metric_ungrounded_when_recommendation_has_no_evidence():
    metric = RecommendationGroundedness()
    results = [
        _result(
            "rec1",
            "recommend",
            "You should consolidate your vendors.",  # no citation, no sources
            sources=[],
        )
    ]
    batch = metric.compute(results)
    assert batch.value == 0.0
    assert batch.details["num_ungrounded"] == 1


def test_metric_mixed_batch():
    metric = RecommendationGroundedness()
    results = [
        _result("rec_good", "recommend",
                "Reduce cloud spend by 10% (from aws_bill_q3).",
                sources=[{"doc_id": "aws_bill_q3", "chunk_id": "c1"}]),
        _result("rec_bad", "recommend",
                "You could try switching providers.",
                sources=[]),
        _result("ana", "analyze",
                "Revenue is stable."),
    ]
    batch = metric.compute(results)
    assert batch.details["num_recommend_queries"] == 2
    assert batch.value == 0.5
