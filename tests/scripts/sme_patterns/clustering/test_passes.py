"""Lean tests for the four clustering passes + shared helpers."""
from __future__ import annotations

from datetime import datetime

from scripts.sme_patterns.clustering._shared import choose_k, cluster_texts
from scripts.sme_patterns.clustering.artifact_utility import (
    ArtifactUtilityConfig,
    analyze_artifact_utility,
)
from scripts.sme_patterns.clustering.failure_patterns import (
    FailurePatternsConfig,
    cluster_failure_patterns,
    is_failure_query,
)
from scripts.sme_patterns.clustering.persona_effect import (
    PersonaEffectConfig,
    analyze_persona_effect,
    sme_score_proxy,
)
from scripts.sme_patterns.clustering.success_patterns import (
    SuccessPatternsConfig,
    cluster_success_patterns,
    is_success_query,
)
from scripts.sme_patterns.schema import ClusterType, QueryFeedback, QueryRun


def _run(
    qid,
    *,
    domain="finance",
    intent="analyze",
    rating=1,
    sme=3,
    drops=0,
    honest_fallback=False,
    persona="senior financial analyst",
    text="analyze Q3 revenue trend",
    fingerprint=None,
):
    fb = QueryFeedback(rating=rating, source="feedback_tracker") if rating is not None else None
    return QueryRun(
        subscription_id="sub_a",
        profile_id="prof_a",
        profile_domain=domain,
        query_id=qid,
        query_text=text,
        query_fingerprint=fingerprint or qid,
        intent=intent,
        adapter_version="1.0.0",
        adapter_persona_role=persona,
        retrieval_layers={"chunks": 12, "kg": 5, "sme_artifacts": sme, "url": 0},
        pack_tokens=4200,
        citation_verifier_drops=drops,
        honest_compact_fallback=honest_fallback,
        captured_at=datetime(2026, 4, 5, 10, 0, 0),
        feedback=fb,
    )


def test_shared_cluster_groups_similar_and_bounds_k():
    assert choose_k(0) == 1
    assert choose_k(50) >= 3
    assert choose_k(500) <= 20

    texts = [
        "analyze Q3 revenue trend",
        "analyze Q3 revenue pattern",
        "diagnose login error symptom",
        "diagnose login error cause",
        "recommend cost reduction plan",
    ]
    clusters = cluster_texts(texts, k=3)
    assert len(clusters) == 3
    all_idxs = sorted(i for c in clusters for i in c.member_indexes)
    assert all_idxs == list(range(5))
    assert any({2, 3}.issubset(set(c.member_indexes)) for c in clusters)


def test_success_pass_filters_and_clusters_winning_queries():
    # 6 revenue-trend + 3 cost + 1 reco + 1 non-success
    runs = (
        [_run(f"q_rev_{i}", text="analyze Q3 revenue trend growth") for i in range(6)]
        + [_run(f"q_cost_{i}", text="analyze cost structure breakdown") for i in range(3)]
        + [_run("q_rec", intent="recommend", text="recommend SaaS consolidation plan")]
        + [_run("bad", intent="analyze", rating=-1, drops=3)]
    )
    clusters = cluster_success_patterns(runs, SuccessPatternsConfig(top_n=5))
    assert clusters, "expected at least one success cluster"
    assert all(c.cluster_type == ClusterType.SUCCESS for c in clusters)
    assert sum(c.size for c in clusters) == 10  # bad excluded
    # Eligibility predicate itself
    assert is_success_query(
        _run("ok", intent="analyze", rating=1, sme=3)
    ) is True
    assert is_success_query(
        _run("nope", intent="lookup", rating=1)
    ) is False
    assert is_success_query(
        _run("no_sme", intent="analyze", rating=1, sme=0)
    ) is False


def test_failure_pass_detects_drops_rating_fallback_and_recurring():
    # Failure by explicit rating + drops; plus recurring-bad fingerprint group.
    runs = (
        [
            _run(
                f"bad_rec_{i}",
                intent="recommend",
                rating=-1,
                text="recommend SaaS consolidation plan",
            )
            for i in range(5)
        ]
        + [
            _run(
                f"drop_{i}",
                intent="diagnose",
                rating=None,
                drops=3,
                text="diagnose login authentication failure",
            )
            for i in range(2)
        ]
        + [
            _run("r1", intent="diagnose", rating=-1, fingerprint="fp_r",
                 text="why is authentication broken"),
            _run("r2", intent="diagnose", rating=0, fingerprint="fp_r",
                 text="why is authentication broken"),
            _run("r3", intent="diagnose", rating=-1, fingerprint="fp_r",
                 text="why is authentication broken"),
        ]
        + [_run("ok", intent="analyze", rating=1)]
    )
    clusters = cluster_failure_patterns(runs, FailurePatternsConfig(top_n=10))
    assert clusters, "expected at least one failure cluster"
    assert all(c.cluster_type == ClusterType.FAILURE for c in clusters)
    assert sum(c.size for c in clusters) == 10
    sizes = [c.size for c in clusters]
    assert sizes == sorted(sizes, reverse=True)

    assert is_failure_query(_run("x", rating=-1)) is True
    assert is_failure_query(_run("x", rating=None, drops=2)) is True
    assert is_failure_query(_run("x", rating=None, honest_fallback=True)) is True
    assert is_failure_query(_run("x", rating=1, drops=0)) is False


def test_artifact_utility_computes_rates_and_flags_dead_weight():
    # Empty input emits one row per layer with zero rates
    empty_rows = analyze_artifact_utility([], ArtifactUtilityConfig())
    assert len(empty_rows) == 4
    assert all(c.size == 0 for c in empty_rows)

    runs = [
        _run("q1", drops=0, sme=3),
        _run("q2", drops=0, sme=3),
        _run("q3", drops=0, sme=0),
        _run("q4", drops=0, sme=0),
    ]
    rows = analyze_artifact_utility(runs, ArtifactUtilityConfig())
    by_id = {c.cluster_id: c for c in rows}
    assert {c.cluster_id for c in rows} == {
        "artifact_chunks",
        "artifact_kg",
        "artifact_sme_artifacts",
        "artifact_url",
    }
    assert by_id["artifact_sme_artifacts"].evidence["retrieval_rate"] == 0.5
    assert by_id["artifact_url"].evidence["retrieval_rate"] == 0.0

    bad_runs = [
        _run(f"b{i}", drops=2, rating=-1, sme=3)
        for i in range(4)
    ]
    dw_rows = analyze_artifact_utility(bad_runs, ArtifactUtilityConfig())
    dw_by_id = {c.cluster_id: c for c in dw_rows}
    assert dw_by_id["artifact_sme_artifacts"].evidence["dead_weight_flag"] is True


def test_persona_effect_flags_regression_and_sorts_worst_first():
    good = [
        _run(f"ok_{i}", persona="senior financial analyst", rating=1)
        for i in range(10)
    ]
    bad = [
        _run(
            f"rogue_{i}",
            persona="rogue persona",
            rating=-1,
            drops=2,
            honest_fallback=True,
        )
        for i in range(10)
    ]
    rows = analyze_persona_effect(good + bad, PersonaEffectConfig(regression_delta=0.1))
    assert rows, "expected at least one persona row"
    assert all(c.cluster_type == ClusterType.PERSONA_EFFECT for c in rows)
    # Worst-first
    assert rows[0].evidence["persona_role"] == "rogue persona"
    # Regression flag set
    assert next(
        r.evidence["regression_flag"]
        for r in rows
        if r.evidence["persona_role"] == "rogue persona"
    ) is True
    # Score proxy obeys penalty ordering
    assert sme_score_proxy(good) > sme_score_proxy(bad)
