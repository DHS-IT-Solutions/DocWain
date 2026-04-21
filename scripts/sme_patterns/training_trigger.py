"""Training-trigger evaluator.

Reads the last N monthly pattern reports and outputs ``TrainingCandidate``
records for failure clusters that have stabilized. Sub-project F is a
separate human-gated project; this module writes evidence, nothing more.

Stabilization formula::

    stabilization = 0.5 * months_present_ratio
                  + 0.3 * volume_ratio
                  + 0.2 * severity_avg

    months_present_ratio = months_present_for_cluster / total_months_window
    volume_ratio         = min(1.0, total_volume / volume_ref)
    severity_avg         = mean of cluster.signal_score across months

Threshold gate: ``months_present >= min_months`` AND
``total_volume >= min_volume`` AND ``stabilization >= stabilization_threshold``.
All thresholds are CLI-configurable.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from scripts.sme_patterns.schema import Cluster, PatternReport, TrainingCandidate

logger = logging.getLogger(__name__)

_REPORT_GLOB = "sme_patterns_*.json"


@dataclass(frozen=True)
class TrainingTriggerConfig:
    min_months: int = 2
    min_volume: int = 20
    stabilization_threshold: float = 0.55
    total_months_window: int = 2
    volume_ref: float = 50.0


@dataclass
class _Group:
    key: tuple[str, str, tuple[str, ...]]  # (domain, intent, terms)
    clusters: list[Cluster] = field(default_factory=list)
    months_present: int = 0
    total_volume: int = 0


def _cluster_terms_key(c: Cluster) -> tuple[str, ...]:
    """Produce a stable signature for cross-month matching.

    Prefer ``evidence.top_terms`` (set by each clustering pass); fall back to
    the cluster_id shape prefix (which already bakes in domain+intent). This
    keeps matches interpretable even when top-terms drift slightly month to
    month.
    """
    terms = c.evidence.get("top_terms") or []
    if isinstance(terms, list) and terms:
        return tuple(sorted(str(t) for t in terms)[:3])
    parts = c.cluster_id.rsplit("_", 1)
    return (parts[0],)


def match_clusters_across_months(reports: list[PatternReport]) -> list[_Group]:
    """Group failure clusters across monthly reports by (domain, intent, top-terms)."""
    buckets: dict[tuple[str, str, tuple[str, ...]], _Group] = {}
    months_seen: dict[tuple[str, str, tuple[str, ...]], set[str]] = defaultdict(set)

    for rep in reports:
        month_key = rep.period_start.strftime("%Y-%m")
        for c in rep.failures:
            k = (
                c.profile_domain or "",
                c.primary_intent or "",
                _cluster_terms_key(c),
            )
            if k not in buckets:
                buckets[k] = _Group(key=k)
            buckets[k].clusters.append(c)
            months_seen[k].add(month_key)

    out = list(buckets.values())
    for g in out:
        g.months_present = len(months_seen[g.key])
        g.total_volume = sum(c.size for c in g.clusters)
    return out


def stabilization_score(
    *,
    months_present: int,
    total_volume: int,
    severity_avg: float,
    config: TrainingTriggerConfig,
) -> float:
    """Return the stabilization score in the range [0.0, ~1.0]."""
    months_ratio = months_present / max(1, config.total_months_window)
    volume_ratio = min(1.0, total_volume / max(1.0, config.volume_ref))
    return round(
        0.5 * months_ratio + 0.3 * volume_ratio + 0.2 * severity_avg,
        3,
    )


def evaluate_candidates(
    reports: list[PatternReport],
    config: TrainingTriggerConfig,
) -> list[TrainingCandidate]:
    """Return the list of TrainingCandidates for stabilized failure clusters."""
    groups = match_clusters_across_months(reports)
    candidates: list[TrainingCandidate] = []
    for g in groups:
        if g.months_present < config.min_months:
            continue
        if g.total_volume < config.min_volume:
            continue

        severity_avg = (
            sum(c.signal_score for c in g.clusters) / len(g.clusters)
            if g.clusters
            else 0.0
        )
        stab = stabilization_score(
            months_present=g.months_present,
            total_volume=g.total_volume,
            severity_avg=severity_avg,
            config=config,
        )
        if stab < config.stabilization_threshold:
            continue

        domain, intent, terms = g.key
        candidate_id = "tc_" + ("_".join(filter(None, [domain, intent, *terms])) or "root")
        candidate_id = candidate_id[:120]
        short = (
            f"recurring {intent or 'failure'} clusters on {domain or 'any domain'} "
            f"— {g.months_present} months present, volume {g.total_volume}, "
            f"severity≈{severity_avg:.2f}"
        )
        candidates.append(
            TrainingCandidate(
                candidate_id=candidate_id,
                cluster_ids=[c.cluster_id for c in g.clusters],
                months_present=g.months_present,
                total_volume=g.total_volume,
                stabilization_score=stab,
                dominant_intent=intent or "unknown",
                dominant_domain=domain or "unknown",
                short_description=short,
            )
        )

    candidates.sort(key=lambda c: c.stabilization_score, reverse=True)
    return candidates


def load_reports_from_dir(dirpath: Path | str) -> list[PatternReport]:
    """Load all ``sme_patterns_YYYY-MM.json`` reports under ``dirpath``."""
    p = Path(dirpath)
    reports: list[PatternReport] = []
    for file in sorted(p.glob(_REPORT_GLOB)):
        if file.suffix != ".json":
            continue
        try:
            reports.append(
                PatternReport.model_validate_json(file.read_text(encoding="utf-8"))
            )
        except Exception:
            logger.exception("failed to load report %s; skipping", file)
            continue
    return reports


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate training triggers from monthly reports"
    )
    parser.add_argument("--reports-dir", type=Path, default=Path("analytics"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-months", type=int, default=2)
    parser.add_argument("--min-volume", type=int, default=20)
    parser.add_argument("--stabilization-threshold", type=float, default=0.55)
    parser.add_argument("--total-months-window", type=int, default=2)
    args = parser.parse_args(argv)

    reports = load_reports_from_dir(args.reports_dir)
    if not reports:
        print(
            f"[evaluate_training_trigger] no reports under {args.reports_dir}",
            file=sys.stderr,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("[]", encoding="utf-8")
        return 0

    config = TrainingTriggerConfig(
        min_months=args.min_months,
        min_volume=args.min_volume,
        stabilization_threshold=args.stabilization_threshold,
        total_months_window=args.total_months_window,
    )
    candidates = evaluate_candidates(reports, config)
    payload = [c.model_dump() for c in candidates]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(
        f"[evaluate_training_trigger] wrote {len(candidates)} candidates to {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
