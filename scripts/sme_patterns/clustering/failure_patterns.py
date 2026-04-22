"""Pass 2 — Failure-pattern clustering.

A 'failure' query is any where the SME reasoning stack visibly stumbled:
explicit thumbs-down, citation-verifier drops, honest-compact fallback, or
a recurring fingerprint with net-negative aggregate rating.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from scripts.sme_patterns.clustering._shared import cluster_texts
from scripts.sme_patterns.schema import Cluster, ClusterType, QueryRun


@dataclass(frozen=True)
class FailurePatternsConfig:
    top_n: int = 10
    k_per_group: int | None = None
    recurring_min: int = 3
    recurring_net_neg_threshold: float = 0.0  # avg rating <= this is "net-negative"


def is_failure_query(q: QueryRun) -> bool:
    if q.feedback and q.feedback.rating == -1:
        return True
    if q.citation_verifier_drops > 0:
        return True
    if q.honest_compact_fallback:
        return True
    return False


def _recurring_bad_fingerprints(
    runs: list[QueryRun], cfg: FailurePatternsConfig
) -> set[str]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for r in runs:
        if r.feedback and r.feedback.rating is not None:
            buckets[r.query_fingerprint].append(float(r.feedback.rating))
    bad: set[str] = set()
    for fp, ratings in buckets.items():
        if len(ratings) < cfg.recurring_min:
            continue
        if (sum(ratings) / len(ratings)) <= cfg.recurring_net_neg_threshold:
            bad.add(fp)
    return bad


def cluster_failure_patterns(
    runs: Iterable[QueryRun],
    config: FailurePatternsConfig,
) -> list[Cluster]:
    runs = list(runs)
    if not runs:
        return []

    recurring = _recurring_bad_fingerprints(runs, config)
    eligible = [
        r for r in runs if is_failure_query(r) or r.query_fingerprint in recurring
    ]
    if not eligible:
        return []

    grouped: dict[tuple[str, str], list[QueryRun]] = defaultdict(list)
    for q in eligible:
        grouped[(q.profile_domain, q.intent)].append(q)

    produced: list[Cluster] = []
    for (domain, intent), group in grouped.items():
        texts = [q.query_text for q in group]
        text_clusters = cluster_texts(texts, k=config.k_per_group)
        for tc in text_clusters:
            member_runs = [group[i] for i in tc.member_indexes]
            subs = sorted({r.subscription_id for r in member_runs})
            fp_counter = Counter(r.query_fingerprint for r in member_runs)
            fps = [fp for fp, _ in fp_counter.most_common(5)]
            avg_drops = _avg(r.citation_verifier_drops for r in member_runs)
            fallback_rate = _avg(
                1.0 if r.honest_compact_fallback else 0.0 for r in member_runs
            )
            neg_rate = _avg(
                1.0 if (r.feedback and r.feedback.rating == -1) else 0.0
                for r in member_runs
            )
            severity = 0.4 * neg_rate + 0.3 * min(1.0, avg_drops / 3.0) + 0.3 * fallback_rate

            short = (
                f"Failing {intent} queries on {domain} — drops≈{avg_drops:.1f}, "
                f"neg-rate={neg_rate:.0%}; top terms: {', '.join(tc.terms[:3]) or 'n/a'}"
            )
            produced.append(
                Cluster(
                    cluster_id=f"fail_{domain}_{intent}_{tc.centroid_index}",
                    cluster_type=ClusterType.FAILURE,
                    size=len(member_runs),
                    subscription_ids=subs,
                    primary_intent=intent,
                    profile_domain=domain,
                    fingerprint_samples=fps,
                    short_description=short,
                    signal_score=round(min(1.0, severity), 3),
                    evidence={
                        "avg_verifier_drops": round(avg_drops, 2),
                        "honest_compact_fallback_rate": round(fallback_rate, 2),
                        "thumbs_down_rate": round(neg_rate, 2),
                        "recurring_fingerprints": sorted(recurring & set(fps)),
                        "top_terms": tc.terms,
                    },
                )
            )

    produced.sort(key=lambda c: (c.size, c.signal_score), reverse=True)
    return produced[: config.top_n]


def _avg(values: Iterable[float]) -> float:
    vs = [float(v) for v in values]
    return (sum(vs) / len(vs)) if vs else 0.0
