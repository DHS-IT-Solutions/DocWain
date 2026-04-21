"""Lean tests for the training-trigger evaluator."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from scripts.sme_patterns.schema import Cluster, ClusterType, PatternReport
from scripts.sme_patterns.training_trigger import (
    TrainingTriggerConfig,
    evaluate_candidates,
    load_reports_from_dir,
    match_clusters_across_months,
    stabilization_score,
)


def _fail_cluster(cid, size, severity, domain="finance", intent="recommend", terms=None):
    return Cluster(
        cluster_id=cid,
        cluster_type=ClusterType.FAILURE,
        size=size,
        profile_domain=domain,
        primary_intent=intent,
        short_description="x",
        signal_score=severity,
        fingerprint_samples=["fp_x", "fp_y"],
        evidence={"top_terms": terms or ["recommend", "saas", "cost"]},
    )


def _report(month, clusters):
    return PatternReport(
        run_id=f"patterns_2026-{month:02d}",
        period_start=datetime(2026, month, 1),
        period_end=datetime(2026, month, 28),
        num_synth_runs=0,
        num_query_runs=0,
        successes=[],
        failures=clusters,
        artifact_utility=[],
        persona_effect=[],
        training_candidates=[],
        rollback_links=[],
    )


def test_matching_groups_clusters_by_domain_intent_and_top_terms():
    m1 = _fail_cluster("fail_finance_recommend_0", 20, 0.6)
    m2 = _fail_cluster("fail_finance_recommend_0", 30, 0.5)
    groups = match_clusters_across_months([_report(3, [m1]), _report(4, [m2])])
    assert len(groups) == 1
    assert groups[0].months_present == 2
    assert groups[0].total_volume == 50

    # Differing intent must NOT merge
    groups2 = match_clusters_across_months(
        [
            _report(
                3,
                [
                    _fail_cluster("fail_finance_recommend_0", 20, 0.6, intent="recommend"),
                    _fail_cluster(
                        "fail_finance_diagnose_0",
                        30,
                        0.5,
                        intent="diagnose",
                        terms=["login", "auth", "error"],
                    ),
                ],
            )
        ]
    )
    assert len(groups2) == 2


def test_thresholds_filter_single_month_and_low_volume():
    cfg = TrainingTriggerConfig(
        min_months=2, min_volume=20, stabilization_threshold=0.0, total_months_window=2
    )
    # Single-month fails min_months
    assert evaluate_candidates(
        [_report(4, [_fail_cluster("fail_x", 100, 1.0)])], cfg
    ) == []

    # Low total volume fails min_volume
    m1 = _fail_cluster("fail_x", 2, 0.9)
    m2 = _fail_cluster("fail_x", 3, 0.9)
    assert evaluate_candidates([_report(3, [m1]), _report(4, [m2])], cfg) == []


def test_happy_path_produces_candidate_and_scores_correctly(tmp_path: Path):
    m1 = _fail_cluster("fail_finance_recommend_0", 20, 0.6)
    m2 = _fail_cluster("fail_finance_recommend_0", 30, 0.55)
    cfg = TrainingTriggerConfig(
        min_months=2,
        min_volume=20,
        stabilization_threshold=0.5,
        total_months_window=2,
        volume_ref=50,
    )
    cands = evaluate_candidates([_report(3, [m1]), _report(4, [m2])], cfg)
    assert len(cands) == 1
    assert cands[0].months_present == 2
    assert cands[0].total_volume == 50
    # Manually verify score formula
    expected = stabilization_score(
        months_present=2,
        total_volume=50,
        severity_avg=(0.6 + 0.55) / 2,
        config=cfg,
    )
    assert abs(cands[0].stabilization_score - expected) < 1e-6
    assert cands[0].stabilization_score >= cfg.stabilization_threshold

    # load_reports_from_dir round-trip
    r3 = _report(3, [m1])
    r4 = _report(4, [m2])
    (tmp_path / "sme_patterns_2026-03.json").write_text(r3.model_dump_json())
    (tmp_path / "sme_patterns_2026-04.json").write_text(r4.model_dump_json())
    (tmp_path / "notes.txt").write_text("ignored")
    reps = load_reports_from_dir(tmp_path)
    assert [r.run_id for r in reps] == ["patterns_2026-03", "patterns_2026-04"]
