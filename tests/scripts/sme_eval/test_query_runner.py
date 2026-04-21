"""Tests for the query runner."""
from datetime import datetime
from unittest.mock import patch, MagicMock

import httpx
import pytest

from scripts.sme_eval.query_runner import QueryRunner, RunnerConfig
from scripts.sme_eval.schema import EvalQuery, LatencyBreakdown


@pytest.fixture
def sample_query():
    return EvalQuery(
        query_id="finance_001",
        query_text="Summarize Q3 revenue trends.",
        intent="analyze",
        profile_domain="finance",
        subscription_id="test_sub",
        profile_id="test_prof",
    )


@pytest.fixture
def config():
    return RunnerConfig(
        base_url="http://localhost:8000",
        path="/api/ask",
        user_id="test@docwain.internal",
        fetch_timeout_s=120.0,
    )


def _mock_response(status=200, payload=None):
    r = MagicMock(spec=httpx.Response)
    r.status_code = status
    r.json.return_value = payload or {
        "payload": {
            "response": "Q3 revenue rose 12%.",
            "sources": [{"doc_id": "d1", "chunk_id": "c1"}],
            "grounded": True,
            "context_found": True,
            "metadata": {},
        }
    }
    r.raise_for_status = MagicMock()
    return r


def test_runner_executes_one_query(config, sample_query):
    runner = QueryRunner(config)

    with patch.object(runner._client, "post", return_value=_mock_response()) as mock_post:
        result = runner.run_one(sample_query, run_id="run_test")

    assert result.query.query_id == "finance_001"
    assert result.response_text == "Q3 revenue rose 12%."
    assert result.api_status == 200
    assert result.latency.total_ms > 0
    assert result.run_id == "run_test"
    assert isinstance(result.captured_at, datetime)

    # Correct payload submitted
    call_args = mock_post.call_args
    assert call_args[1]["json"]["query"] == "Summarize Q3 revenue trends."
    assert call_args[1]["json"]["subscription_id"] == "test_sub"
    assert call_args[1]["json"]["profile_id"] == "test_prof"


def test_runner_captures_http_error(config, sample_query):
    runner = QueryRunner(config)
    err_response = _mock_response(status=500, payload={"detail": "server broke"})
    err_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "500", request=MagicMock(), response=err_response
    )

    with patch.object(runner._client, "post", return_value=err_response):
        result = runner.run_one(sample_query, run_id="run_test")

    assert result.api_status == 500
    assert result.api_error is not None
    assert result.response_text == ""


def test_runner_captures_network_timeout(config, sample_query):
    runner = QueryRunner(config)
    with patch.object(runner._client, "post", side_effect=httpx.TimeoutException("timed out")):
        result = runner.run_one(sample_query, run_id="run_test")

    assert result.api_status == 0
    assert "timed out" in result.api_error.lower()


def test_runner_batch_runs_all(config, sample_query):
    runner = QueryRunner(config)
    queries = [sample_query, sample_query.model_copy(update={"query_id": "finance_002"})]

    with patch.object(runner._client, "post", return_value=_mock_response()):
        results = list(runner.run_batch(queries, run_id="run_test"))

    assert len(results) == 2
    assert {r.query.query_id for r in results} == {"finance_001", "finance_002"}
