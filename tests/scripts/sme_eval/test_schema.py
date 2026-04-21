"""Tests for eval query schema."""
from datetime import datetime

import pytest
from pydantic import ValidationError

from scripts.sme_eval.schema import (
    EvalQuery,
    EvalResult,
    LatencyBreakdown,
    MetricResult,
)


def _valid_query_dict():
    return {
        "query_id": "finance_001",
        "query_text": "Summarize our Q3 revenue trends.",
        "intent": "analyze",
        "profile_domain": "finance",
        "subscription_id": "test_sub_finance",
        "profile_id": "test_prof_finance_1",
        "expected_behavior": "Should identify QoQ trend and cite Q1/Q2/Q3 reports",
        "tags": ["trend", "revenue"],
    }


def test_eval_query_valid():
    q = EvalQuery(**_valid_query_dict())
    assert q.query_id == "finance_001"
    assert q.intent == "analyze"
    assert q.profile_domain == "finance"


def test_eval_query_rejects_missing_required_field():
    d = _valid_query_dict()
    del d["query_text"]
    with pytest.raises(ValidationError):
        EvalQuery(**d)


def test_eval_query_rejects_invalid_domain():
    d = _valid_query_dict()
    d["profile_domain"] = "rocket_science"
    with pytest.raises(ValidationError):
        EvalQuery(**d)


def test_eval_query_tags_default_empty():
    d = _valid_query_dict()
    del d["tags"]
    q = EvalQuery(**d)
    assert q.tags == []


def test_latency_breakdown_required_fields():
    lb = LatencyBreakdown(ttft_ms=820.5, total_ms=4200.0)
    assert lb.ttft_ms == 820.5
    assert lb.total_ms == 4200.0


def test_eval_result_serializes_to_dict():
    q = EvalQuery(**_valid_query_dict())
    r = EvalResult(
        query=q,
        response_text="Q3 revenue rose 12%...",
        sources=[{"doc_id": "d1", "chunk_id": "c1"}],
        metadata={"grounded": True},
        latency=LatencyBreakdown(ttft_ms=800.0, total_ms=4000.0),
        run_id="run_20260420",
        captured_at=datetime(2026, 4, 20, 10, 0, 0),
        api_status=200,
    )
    d = r.model_dump(mode="json")
    assert d["query"]["query_id"] == "finance_001"
    assert d["latency"]["ttft_ms"] == 800.0
    assert d["api_status"] == 200


def test_metric_result_has_value_and_details():
    m = MetricResult(
        metric_name="recommendation_groundedness",
        value=0.92,
        details={"passed": 46, "failed": 4},
    )
    assert m.metric_name == "recommendation_groundedness"
    assert m.value == 0.92
    assert m.details["passed"] == 46
