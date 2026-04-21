"""Tests for the RAGAS wrapper."""
from datetime import datetime
from unittest.mock import patch

import pytest

from scripts.sme_eval.metrics.ragas_wrapper import RagasMetrics
from scripts.sme_eval.schema import EvalQuery, EvalResult, LatencyBreakdown


def _result(qid, response, sources=None, grounded=True):
    return EvalResult(
        query=EvalQuery(
            query_id=qid,
            query_text="q",
            intent="lookup",
            profile_domain="finance",
            subscription_id="s",
            profile_id="p",
        ),
        response_text=response,
        sources=sources or [{"doc_id": "d1", "chunk_id": "c1"}],
        metadata={"grounded": grounded},
        latency=LatencyBreakdown(total_ms=100.0),
        run_id="run_test",
        captured_at=datetime(2026, 4, 20, 10, 0, 0),
        api_status=200,
    )


def test_ragas_wrapper_computes_four_metrics():
    metric = RagasMetrics()
    results = [
        _result("a", "The answer is 42."),
        _result("b", "The answer is 17."),
    ]
    batch = metric.compute(results)

    assert batch.metric_name == "ragas"
    assert "answer_faithfulness" in batch.details
    assert "hallucination_rate" in batch.details
    assert "context_recall" in batch.details
    assert "grounding_bypass_rate" in batch.details
    # Value is the faithfulness score (primary gate metric)
    assert 0.0 <= batch.value <= 1.0


def test_ragas_wrapper_flags_hallucination_markers():
    metric = RagasMetrics()
    hallucinating = _result("a", "As an AI language model, I cannot access...")
    clean = _result("b", "The answer is 42.")
    batch = metric.compute([hallucinating, clean])

    # At least one result flagged as hallucination
    assert batch.details["hallucination_rate"] > 0.0


def test_ragas_wrapper_empty_batch_returns_zero():
    metric = RagasMetrics()
    batch = metric.compute([])
    assert batch.value == 0.0
    assert batch.details["num_results"] == 0
