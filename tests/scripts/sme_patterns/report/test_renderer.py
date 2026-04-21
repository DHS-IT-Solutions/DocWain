"""Lean tests for the report composer + Jinja2 markdown renderer."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from scripts.sme_patterns.report.model import compose_pattern_report
from scripts.sme_patterns.report.renderer import render_pattern_report
from scripts.sme_patterns.schema import Cluster, ClusterType, TrainingCandidate


def _sample_report(rollback_links=None):
    suc = Cluster(
        cluster_id="succ_finance_analyze_0",
        cluster_type=ClusterType.SUCCESS,
        size=18,
        subscription_ids=["sub_a"],
        profile_domain="finance",
        primary_intent="analyze",
        short_description="Successful analyze queries on finance — top terms: revenue, q3, trend",
        signal_score=0.88,
        evidence={"top_terms": ["revenue", "q3", "trend"], "avg_sme_artifacts": 3.2},
    )
    fail = Cluster(
        cluster_id="fail_finance_recommend_0",
        cluster_type=ClusterType.FAILURE,
        size=12,
        subscription_ids=["sub_a", "sub_b"],
        profile_domain="finance",
        primary_intent="recommend",
        short_description="Failing recommend queries on finance — drops ≈ 2.1",
        signal_score=0.54,
        evidence={"avg_verifier_drops": 2.1, "thumbs_down_rate": 0.6},
    )
    art = Cluster(
        cluster_id="artifact_sme_artifacts",
        cluster_type=ClusterType.ARTIFACT_UTILITY,
        size=900,
        short_description="Layer 'sme_artifacts': used in 80% of queries",
        signal_score=0.72,
        evidence={
            "layer": "sme_artifacts",
            "retrieval_rate": 0.8,
            "citation_rate": 0.72,
            "dead_weight_flag": False,
        },
    )
    persona = Cluster(
        cluster_id="persona_finance_abcd",
        cluster_type=ClusterType.PERSONA_EFFECT,
        size=50,
        profile_domain="finance",
        short_description="Persona 'cfo advisor' on finance: proxy=0.88 (baseline 0.85)",
        signal_score=0.88,
        evidence={
            "persona_role": "cfo advisor",
            "regression_flag": False,
            "sme_score_proxy": 0.88,
            "domain_baseline": 0.85,
            "queries": 50,
        },
    )
    tc = TrainingCandidate(
        candidate_id="tc_001",
        cluster_ids=["fail_finance_recommend_0", "fail_finance_recommend_0_prev"],
        months_present=2,
        total_volume=48,
        stabilization_score=0.7,
        dominant_intent="recommend",
        dominant_domain="finance",
        short_description="recurring ungrounded recommendations on finance",
    )
    return compose_pattern_report(
        query_runs=[],
        synth_runs=[],
        successes=[suc],
        failures=[fail],
        artifact_utility=[art],
        persona_effect=[persona],
        training_candidates=[tc],
        period_start=datetime(2026, 4, 1),
        period_end=datetime(2026, 4, 30),
        rollback_links=rollback_links or [],
    )


def test_composer_and_renderer_emit_all_sections(tmp_path: Path):
    rep = _sample_report()
    assert rep.run_id == "patterns_2026-04"
    out = tmp_path / "sme_patterns_2026-04.md"
    path = render_pattern_report(rep, out)
    text = Path(path).read_text()
    for header in (
        "# DocWain SME Patterns — 2026-04",
        "## Executive summary",
        "## Success patterns",
        "## Failure patterns",
        "## Artifact utility",
        "## Persona performance",
        "## Training candidates",
    ):
        assert header in text, f"missing section: {header}"
    # Content propagates from evidence dicts
    assert "cfo advisor" in text
    assert "tc_001" in text
    assert "sub_a" in text


def test_renderer_toggles_rollback_section(tmp_path: Path):
    rep_with = _sample_report(rollback_links=["analytics/sme_rollback_2026-04-12.md"])
    rep_without = _sample_report(rollback_links=[])
    out_with = render_pattern_report(rep_with, tmp_path / "with.md")
    out_without = render_pattern_report(rep_without, tmp_path / "without.md")
    assert "## Rollback post-mortems" in Path(out_with).read_text()
    assert "## Rollback post-mortems" not in Path(out_without).read_text()
    assert "analytics/sme_rollback_2026-04-12.md" in Path(out_with).read_text()
