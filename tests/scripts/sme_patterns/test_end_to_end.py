"""End-to-end integration for the monthly pattern-mining pipeline.

Uses in-memory Azure Blob stubs — no network I/O, no Redis. Validates the
whole chain from blob bytes to rendered Markdown + JSON + training candidates.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

from scripts.sme_patterns.run import (
    RunConfig,
    default_window,
    run_monthly_mining,
    window_from_days,
)
from scripts.sme_patterns.schema import PatternReport
from tests.scripts.sme_patterns.fixtures.query_trace_factory import make_query_jsonl
from tests.scripts.sme_patterns.fixtures.synth_trace_factory import make_synth_jsonl


def _build_blobs() -> dict[str, str]:
    blobs: dict[str, str] = {}

    blobs["sme_traces/synthesis/sub_a/prof_a/syn_clean.jsonl"] = make_synth_jsonl(
        synthesis_id="syn_clean",
        started_at=datetime(2026, 4, 2, 2, 0, 0),
        drop_count=0,
    )
    blobs["sme_traces/synthesis/sub_a/prof_a/syn_drops.jsonl"] = make_synth_jsonl(
        synthesis_id="syn_drops",
        started_at=datetime(2026, 4, 9, 2, 0, 0),
        drop_count=4,
    )

    # 10 successful analyze queries
    for i in range(10):
        blobs[
            f"sme_traces/queries/sub_a/prof_a/2026-04-05/succ_{i}.jsonl"
        ] = make_query_jsonl(
            query_id=f"succ_{i}",
            intent="analyze",
            rating=1,
            sme_artifacts=3,
            citation_verifier_drops=0,
            adapter_persona_role="senior financial analyst",
            captured_at=datetime(2026, 4, 5, 10, 0, 0),
        )

    # 6 failing recommend queries (one cluster)
    for i in range(6):
        blobs[
            f"sme_traces/queries/sub_a/prof_a/2026-04-12/fail_rec_{i}.jsonl"
        ] = make_query_jsonl(
            query_id=f"fail_rec_{i}",
            intent="recommend",
            rating=-1,
            sme_artifacts=1,
            citation_verifier_drops=2,
            adapter_persona_role="experimental cfo persona",
            query_text="recommend cost reduction across SaaS stack",
            captured_at=datetime(2026, 4, 12, 10, 0, 0),
        )

    # Out-of-window query (must not be counted)
    blobs[
        "sme_traces/queries/sub_a/prof_a/2026-03-25/skip.jsonl"
    ] = make_query_jsonl(
        query_id="skip",
        captured_at=datetime(2026, 3, 25, 10, 0, 0),
    )

    return blobs


def test_default_window_and_window_from_days_are_sane():
    w = default_window(now=datetime(2026, 5, 3, 2, 0, 0))
    assert w.start == datetime(2026, 4, 1, 0, 0, 0)
    assert w.end.month == 4
    # Rolling window
    roll = window_from_days(days=30, now=datetime(2026, 5, 3, 0, 0, 0))
    assert (roll.end - roll.start).days == 30


def test_end_to_end_pipeline_writes_all_outputs(tmp_path: Path):
    blobs = _build_blobs()
    list_blobs = MagicMock(
        side_effect=lambda prefix: [k for k in blobs if k.startswith(prefix)]
    )
    read_blob = MagicMock(side_effect=lambda name: blobs[name])

    tracker = MagicMock()
    tracker.get_profile_metrics.return_value = {
        "total_queries": 16,
        "avg_confidence": 0.7,
        "grounded_ratio": 0.7,
        "low_confidence_count": 0,
    }

    cfg = RunConfig(
        window_start=datetime(2026, 4, 1),
        window_end=datetime(2026, 4, 30, 23, 59, 59),
        analytics_dir=tmp_path,
    )
    md_path = run_monthly_mining(
        cfg, list_blobs=list_blobs, read_blob=read_blob, feedback_tracker=tracker
    )

    md_text = Path(md_path).read_text()
    for header in (
        "# DocWain SME Patterns — 2026-04",
        "## Executive summary",
        "## Success patterns",
        "## Failure patterns",
        "## Artifact utility",
        "## Persona performance",
        "## Training candidates",
    ):
        assert header in md_text, f"missing section: {header}"

    # JSON snapshot matches
    json_path = Path(md_path).with_suffix(".json")
    rep = PatternReport.model_validate_json(json_path.read_text())
    assert rep.num_query_runs == 16  # 10 success + 6 failure
    assert rep.num_synth_runs == 2
    assert len(rep.failures) >= 1
    assert len(rep.successes) >= 1

    # Artifact utility rows cover all four layers
    artifact_layers = {c.evidence["layer"] for c in rep.artifact_utility}
    assert artifact_layers == {"chunks", "kg", "sme_artifacts", "url"}

    # Single month -> no training candidates
    assert rep.training_candidates == []

    # Separate training-candidates JSON is valid (and empty)
    tc_path = tmp_path / "training_candidates_2026-04.json"
    assert tc_path.exists()
    assert json.loads(tc_path.read_text()) == []


def test_end_to_end_surfaces_candidates_on_second_month(tmp_path: Path):
    """Simulate March + April runs; April's evaluation must produce a
    stabilized candidate from two consecutive months of the same failure
    cluster.
    """
    def _month_blobs(year: int, month: int, day_of_month: int) -> dict[str, str]:
        blobs: dict[str, str] = {}
        day_prefix = f"{year:04d}-{month:02d}-{day_of_month:02d}"
        for i in range(12):
            blobs[
                f"sme_traces/queries/sub_a/prof_a/{day_prefix}/fail_rec_{i}.jsonl"
            ] = make_query_jsonl(
                query_id=f"fail_rec_{month}_{i}",
                intent="recommend",
                rating=-1,
                sme_artifacts=1,
                citation_verifier_drops=2,
                adapter_persona_role="experimental cfo persona",
                query_text="recommend cost reduction across SaaS stack",
                captured_at=datetime(year, month, day_of_month, 10, 0, 0),
            )
        return blobs

    blobs_march = _month_blobs(2026, 3, 12)
    blobs_april = _month_blobs(2026, 4, 12)

    def _run(month: int, blobs: dict[str, str]) -> Path:
        list_blobs = MagicMock(
            side_effect=lambda prefix: [k for k in blobs if k.startswith(prefix)]
        )
        read_blob = MagicMock(side_effect=lambda name: blobs[name])
        tracker = MagicMock()
        tracker.get_profile_metrics.return_value = {
            "total_queries": 12,
            "low_confidence_count": 0,
        }
        cfg = RunConfig(
            window_start=datetime(2026, month, 1),
            window_end=datetime(2026, month, 28, 23, 59, 59),
            analytics_dir=tmp_path,
        )
        return Path(
            run_monthly_mining(
                cfg,
                list_blobs=list_blobs,
                read_blob=read_blob,
                feedback_tracker=tracker,
            )
        )

    _run(3, blobs_march)
    april_md = _run(4, blobs_april)

    april_json = april_md.with_suffix(".json")
    rep = PatternReport.model_validate_json(april_json.read_text())
    assert rep.training_candidates, "expected stabilized candidate after 2 months"
    tc = rep.training_candidates[0]
    assert tc.months_present == 2
    assert tc.dominant_intent == "recommend"
    assert tc.dominant_domain == "finance"
    assert tc.stabilization_score > 0.0

    tc_json = tmp_path / "training_candidates_2026-04.json"
    payload = json.loads(tc_json.read_text())
    assert payload and payload[0]["candidate_id"].startswith("tc_")
