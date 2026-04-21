from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml

from scripts.sme_eval.run_baseline import (
    load_queries_from_yaml,
    compose_snapshot,
    DEFAULT_METRICS,
)
from scripts.sme_eval.schema import EvalQuery, EvalResult, LatencyBreakdown


def test_load_queries_parses_yaml(tmp_path: Path):
    y = tmp_path / "finance.yaml"
    y.write_text(
        yaml.safe_dump({
            "queries": [
                {
                    "query_id": "finance_001",
                    "query_text": "q1",
                    "intent": "analyze",
                    "profile_domain": "finance",
                    "subscription_id": "s",
                    "profile_id": "p",
                },
                {
                    "query_id": "finance_002",
                    "query_text": "q2",
                    "intent": "lookup",
                    "profile_domain": "finance",
                    "subscription_id": "s",
                    "profile_id": "p",
                },
            ]
        })
    )
    queries = load_queries_from_yaml(y)
    assert len(queries) == 2
    assert queries[0].query_id == "finance_001"


def test_load_queries_validates_each(tmp_path: Path):
    y = tmp_path / "bad.yaml"
    y.write_text(
        yaml.safe_dump({
            "queries": [
                {"query_id": "x", "query_text": "q", "intent": "analyze",
                 "profile_domain": "unknown_domain", "subscription_id": "s", "profile_id": "p"},
            ]
        })
    )
    try:
        load_queries_from_yaml(y)
        raised = False
    except Exception:
        raised = True
    assert raised


def test_compose_snapshot_rolls_up_metrics(tmp_path: Path):
    def _result(qid, intent):
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
            sources=[{"doc_id": "d1"}],
            metadata={"grounded": True},
            latency=LatencyBreakdown(total_ms=1000.0),
            run_id="run1",
            captured_at=datetime(2026, 4, 20, 10, 0, 0),
            api_status=200,
        )

    results = [_result("a", "analyze"), _result("b", "lookup")]
    judge_fn = MagicMock(return_value=4.0)
    snap = compose_snapshot(
        results,
        run_id="run1",
        git_sha="abcd123",
        api_base_url="http://localhost:8000",
        judge_fn=judge_fn,
    )
    assert snap.num_queries == 2
    assert snap.per_domain_counts["finance"] == 2
    assert "analyze" in snap.latency_p50_per_intent
    assert "sme_persona_consistency" in snap.sme_metrics
