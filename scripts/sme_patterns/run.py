"""Monthly pattern-mining orchestrator.

Wires loader + feedback merger + four clustering passes + renderer into a
single monthly run. No LLM calls, no destructive writes — output is an
idempotent Markdown + JSON pair under ``analytics/``.

Memory rules honored:
 - ``datetime.now(timezone.utc)`` everywhere; no ``datetime.utcnow``.
 - Traces pulled from Azure Blob only; no Mongo writes.
 - No retraining invocation; evaluator is evidence-only.
 - Redis path via ``src.utils.redis_cache`` (NOT ``src.utils.redis_client``,
   which does not exist).
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts.sme_patterns.clustering.artifact_utility import (
    ArtifactUtilityConfig,
    analyze_artifact_utility,
)
from scripts.sme_patterns.clustering.failure_patterns import (
    FailurePatternsConfig,
    cluster_failure_patterns,
)
from scripts.sme_patterns.clustering.persona_effect import (
    PersonaEffectConfig,
    analyze_persona_effect,
)
from scripts.sme_patterns.clustering.success_patterns import (
    SuccessPatternsConfig,
    cluster_success_patterns,
)
from scripts.sme_patterns.feedback_merger import merge_feedback
from scripts.sme_patterns.report.model import compose_pattern_report
from scripts.sme_patterns.report.renderer import render_pattern_report
from scripts.sme_patterns.schema import PatternReport
from scripts.sme_patterns.trace_loader import TraceLoader, TraceWindow
from scripts.sme_patterns.training_trigger import (
    TrainingTriggerConfig,
    evaluate_candidates,
    load_reports_from_dir,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunConfig:
    window_start: datetime
    window_end: datetime
    analytics_dir: Path
    rollback_glob: str = "sme_rollback_*.md"


def default_window(*, now: datetime | None = None) -> TraceWindow:
    """Return the prior calendar month as a TraceWindow."""
    now = now or datetime.now(timezone.utc).replace(tzinfo=None)
    first_of_this = datetime(now.year, now.month, 1, 0, 0, 0)
    if first_of_this.month == 1:
        prior_first = datetime(first_of_this.year - 1, 12, 1, 0, 0, 0)
    else:
        prior_first = datetime(
            first_of_this.year, first_of_this.month - 1, 1, 0, 0, 0
        )
    prior_end = first_of_this - timedelta(seconds=1)
    return TraceWindow(start=prior_first, end=prior_end)


def window_from_days(*, days: int, now: datetime | None = None) -> TraceWindow:
    """Return a rolling window ending now and starting ``days`` ago."""
    end = now or datetime.now(timezone.utc).replace(tzinfo=None)
    start = end - timedelta(days=days)
    return TraceWindow(start=start, end=end)


def _collect_rollback_links(
    analytics_dir: Path, glob: str, window: TraceWindow
) -> list[str]:
    links: list[str] = []
    for f in sorted(analytics_dir.glob(glob)):
        # Filename pattern: sme_rollback_YYYY-MM-DD.md
        name = f.stem  # 'sme_rollback_2026-04-12'
        parts = name.rsplit("_", 1)
        if len(parts) != 2:
            continue
        try:
            ts = datetime.strptime(parts[1], "%Y-%m-%d")
        except ValueError:
            continue
        if window.start.date() <= ts.date() <= window.end.date():
            links.append(f"analytics/{f.name}")
    return links


def run_monthly_mining(
    config: RunConfig,
    *,
    list_blobs: Callable[[str], Iterable[str]],
    read_blob: Callable[[str], str],
    feedback_tracker,
    success_cfg: SuccessPatternsConfig | None = None,
    failure_cfg: FailurePatternsConfig | None = None,
    artifact_cfg: ArtifactUtilityConfig | None = None,
    persona_cfg: PersonaEffectConfig | None = None,
    trigger_cfg: TrainingTriggerConfig | None = None,
) -> str:
    """Execute one month of mining. Returns the written Markdown path."""
    success_cfg = success_cfg or SuccessPatternsConfig()
    failure_cfg = failure_cfg or FailurePatternsConfig()
    artifact_cfg = artifact_cfg or ArtifactUtilityConfig()
    persona_cfg = persona_cfg or PersonaEffectConfig()
    trigger_cfg = trigger_cfg or TrainingTriggerConfig()

    window = TraceWindow(start=config.window_start, end=config.window_end)
    loader = TraceLoader(list_blobs=list_blobs, read_blob=read_blob)

    synth_runs = list(loader.iter_synthesis_runs(window))
    query_runs = list(loader.iter_query_runs(window))
    logger.info(
        "loaded %d synth runs and %d query runs for window %s -> %s",
        len(synth_runs),
        len(query_runs),
        window.start,
        window.end,
    )

    query_runs = merge_feedback(query_runs, feedback_tracker)

    successes = cluster_success_patterns(query_runs, success_cfg)
    failures = cluster_failure_patterns(query_runs, failure_cfg)
    artifacts = analyze_artifact_utility(query_runs, artifact_cfg)
    personas = analyze_persona_effect(query_runs, persona_cfg)

    rollback_links = _collect_rollback_links(
        config.analytics_dir, config.rollback_glob, window
    )

    # Render the monthly file first with an empty candidates list. Then write
    # the month's JSON snapshot so the training-trigger evaluator can pick it
    # up alongside prior months.
    month_slug = config.window_start.strftime("%Y-%m")
    config.analytics_dir.mkdir(parents=True, exist_ok=True)
    md_path = config.analytics_dir / f"sme_patterns_{month_slug}.md"
    json_path = config.analytics_dir / f"sme_patterns_{month_slug}.json"

    interim_report = compose_pattern_report(
        query_runs=query_runs,
        synth_runs=synth_runs,
        successes=successes,
        failures=failures,
        artifact_utility=artifacts,
        persona_effect=personas,
        training_candidates=[],
        period_start=config.window_start,
        period_end=config.window_end,
        rollback_links=rollback_links,
    )
    json_path.write_text(interim_report.model_dump_json(indent=2), encoding="utf-8")

    # Now evaluate training candidates across current + prior months.
    all_reports = load_reports_from_dir(config.analytics_dir)
    candidates = evaluate_candidates(all_reports, trigger_cfg)

    final_report: PatternReport = compose_pattern_report(
        query_runs=query_runs,
        synth_runs=synth_runs,
        successes=successes,
        failures=failures,
        artifact_utility=artifacts,
        persona_effect=personas,
        training_candidates=candidates,
        period_start=config.window_start,
        period_end=config.window_end,
        rollback_links=rollback_links,
    )

    render_pattern_report(final_report, md_path)
    json_path.write_text(final_report.model_dump_json(indent=2), encoding="utf-8")

    # Also drop the standalone candidates JSON alongside the monthly file so
    # the runbook's `training_candidates_YYYY-MM.json` contract is satisfied.
    tc_path = config.analytics_dir / f"training_candidates_{month_slug}.json"
    import json as _json

    tc_path.write_text(
        _json.dumps([c.model_dump() for c in candidates], indent=2, default=str),
        encoding="utf-8",
    )

    logger.info("wrote %s, %s, %s", md_path, json_path, tc_path)
    return str(md_path)


def _real_list_blobs(prefix: str) -> Iterable[str]:
    # Imported lazily so unit tests don't require Azure credentials at import.
    from src.storage.azure_blob_client import get_document_container_client

    container = get_document_container_client()
    for blob in container.list_blobs(name_starts_with=prefix):
        yield blob.name


def _real_read_blob(name: str) -> str:
    from src.storage.azure_blob_client import get_document_container_client

    container = get_document_container_client()
    client = container.get_blob_client(name)
    return client.download_blob().readall().decode("utf-8")


def _real_feedback_tracker():
    """Build the Redis-backed FeedbackTracker for production runs.

    Falls back to an offline stub if Redis is unreachable — pattern mining
    must tolerate Redis being down (it is evidence, not a critical path).

    Import guard: ``src.utils.redis_cache`` is the project's Redis helper
    (per ERRATA §18). The top-level Redis client factory lives at
    ``src.api.dw_newron.get_redis_client``; if its import fails we drop back
    to the offline stub so the batch job still completes.
    """
    logger_ = logging.getLogger(__name__)
    try:
        from src.utils import redis_cache  # noqa: F401
    except ImportError:
        logger_.warning("src.utils.redis_cache unavailable; using offline stub tracker")
        return _OfflineTrackerStub()

    try:
        from src.api.dw_newron import get_redis_client
        from src.intelligence.feedback_tracker import FeedbackTracker

        r = get_redis_client()
        return FeedbackTracker(r)
    except Exception:
        logger_.exception("failed to wire FeedbackTracker; using offline stub")
        return _OfflineTrackerStub()


class _OfflineTrackerStub:
    def get_profile_metrics(self, _profile_id: str) -> dict:
        return {
            "total_queries": 0,
            "avg_confidence": 0.0,
            "grounded_ratio": 0.0,
            "low_confidence_count": 0,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Monthly SME pattern-mining batch job"
    )
    parser.add_argument("--analytics-dir", type=Path, default=Path("analytics"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Alias for --analytics-dir",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=None,
        help="Rolling window in days (e.g. 30); default = prior calendar month",
    )
    parser.add_argument(
        "--window-start",
        type=str,
        default=None,
        help="ISO date; overrides --window-days",
    )
    parser.add_argument(
        "--window-end",
        type=str,
        default=None,
        help="ISO date; overrides --window-days",
    )
    args = parser.parse_args(argv)

    analytics_dir = args.out_dir or args.analytics_dir

    if args.window_start and args.window_end:
        start = datetime.fromisoformat(args.window_start)
        end = datetime.fromisoformat(args.window_end)
    elif args.window_days:
        w = window_from_days(days=args.window_days)
        start, end = w.start, w.end
    else:
        w = default_window()
        start, end = w.start, w.end

    config = RunConfig(
        window_start=start,
        window_end=end,
        analytics_dir=analytics_dir,
    )

    try:
        out_path = run_monthly_mining(
            config,
            list_blobs=_real_list_blobs,
            read_blob=_real_read_blob,
            feedback_tracker=_real_feedback_tracker(),
        )
    except Exception:
        logger.exception("pattern mining failed")
        return 2

    print(f"[mine_sme_patterns] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
