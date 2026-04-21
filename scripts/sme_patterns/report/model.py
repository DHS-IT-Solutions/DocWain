"""Compose PatternReport from clusters + input counts.

Thin constructor — no I/O here. Rendering lives in renderer.py.
"""
from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime

from scripts.sme_patterns.schema import (
    Cluster,
    PatternReport,
    QueryRun,
    SynthesisRun,
    TrainingCandidate,
)


def compose_pattern_report(
    *,
    query_runs: Iterable[QueryRun],
    synth_runs: Iterable[SynthesisRun],
    successes: list[Cluster],
    failures: list[Cluster],
    artifact_utility: list[Cluster],
    persona_effect: list[Cluster],
    training_candidates: list[TrainingCandidate],
    period_start: datetime,
    period_end: datetime,
    rollback_links: list[str],
) -> PatternReport:
    query_runs = list(query_runs)
    synth_runs = list(synth_runs)
    return PatternReport(
        run_id=f"patterns_{period_start.strftime('%Y-%m')}",
        period_start=period_start,
        period_end=period_end,
        num_synth_runs=len(synth_runs),
        num_query_runs=len(query_runs),
        successes=successes,
        failures=failures,
        artifact_utility=artifact_utility,
        persona_effect=persona_effect,
        training_candidates=training_candidates,
        rollback_links=rollback_links,
    )
