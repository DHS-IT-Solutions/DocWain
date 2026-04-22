from datetime import datetime

from scripts.sme_eval.metrics.sme_artifact_hit_rate import SmeArtifactHitRate
from scripts.sme_eval.schema import EvalQuery, EvalResult, LatencyBreakdown


def _result(qid, intent, metadata=None):
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
        metadata=metadata or {},
        latency=LatencyBreakdown(total_ms=100.0),
        run_id="run_test",
        captured_at=datetime(2026, 4, 20, 10, 0, 0),
        api_status=200,
    )


def test_phase0_baseline_zero():
    """At Phase 0 no SME artifacts exist — value should be 0.0."""
    metric = SmeArtifactHitRate()
    results = [_result("a", "analyze"), _result("b", "diagnose")]
    batch = metric.compute(results)
    assert batch.value == 0.0


def test_with_sme_artifacts_counted():
    metric = SmeArtifactHitRate()
    results = [
        _result("a", "analyze", metadata={"retrieval_layers": {"sme_artifacts_count": 3}}),
        _result("b", "analyze", metadata={"retrieval_layers": {"sme_artifacts_count": 0}}),
    ]
    batch = metric.compute(results)
    assert batch.value == 0.5


def test_skips_non_analytical():
    metric = SmeArtifactHitRate()
    results = [_result("a", "lookup", metadata={"retrieval_layers": {"sme_artifacts_count": 3}})]
    batch = metric.compute(results)
    # No analytical queries — value is 0.0 with num_analytical=0 in details
    assert batch.details["num_analytical"] == 0


def test_empty_batch():
    metric = SmeArtifactHitRate()
    batch = metric.compute([])
    assert batch.value == 0.0
    assert batch.details["num_analytical"] == 0
