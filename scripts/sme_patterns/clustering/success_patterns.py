"""Pass 1 — Success-pattern clustering.

A 'success' query is a high-signal query where the SME reasoning layer
demonstrably helped: clean citations, SME artifacts contributed, either
explicit thumbs-up or no negative implicit signal. We cluster these to
answer "what kinds of queries are we winning on?".
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from scripts.sme_patterns.clustering._shared import cluster_texts
from scripts.sme_patterns.schema import Cluster, ClusterType, QueryRun

_ANALYTICAL_INTENTS: frozenset[str] = frozenset(
    {"analyze", "diagnose", "recommend", "investigate", "compare", "summarize"}
)


@dataclass(frozen=True)
class SuccessPatternsConfig:
    top_n: int = 10
    k_per_group: int | None = None  # None -> choose_k()


def is_success_query(q: QueryRun) -> bool:
    if q.intent not in _ANALYTICAL_INTENTS:
        return False
    if q.retrieval_layers.get("sme_artifacts", 0) < 1:
        return False
    if q.feedback and q.feedback.rating == 1:
        return True
    if q.feedback and q.feedback.rating == -1:
        return False
    if q.citation_verifier_drops > 0:
        return False
    if q.honest_compact_fallback:
        return False
    return True


def cluster_success_patterns(
    runs: Iterable[QueryRun],
    config: SuccessPatternsConfig,
) -> list[Cluster]:
    eligible = [q for q in runs if is_success_query(q)]
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
            fps = sorted({r.query_fingerprint for r in member_runs})[:5]
            avg_artifacts = _avg(
                r.retrieval_layers.get("sme_artifacts", 0) for r in member_runs
            )
            pos_rate = _avg(
                1.0 if (r.feedback and r.feedback.rating == 1) else 0.0
                for r in member_runs
            )
            short = (
                f"Successful {intent} queries on {domain} — top terms: "
                f"{', '.join(tc.terms[:3]) or 'n/a'}"
            )
            produced.append(
                Cluster(
                    cluster_id=f"succ_{domain}_{intent}_{tc.centroid_index}",
                    cluster_type=ClusterType.SUCCESS,
                    size=len(member_runs),
                    subscription_ids=subs,
                    primary_intent=intent,
                    profile_domain=domain,
                    fingerprint_samples=fps,
                    short_description=short,
                    signal_score=round(pos_rate, 3),
                    evidence={
                        "avg_sme_artifacts": round(avg_artifacts, 2),
                        "explicit_thumbs_up_rate": round(pos_rate, 2),
                        "top_terms": tc.terms,
                    },
                )
            )

    produced.sort(key=lambda c: c.size, reverse=True)
    return produced[: config.top_n]


def _avg(values: Iterable[float]) -> float:
    vs = list(values)
    return (sum(vs) / len(vs)) if vs else 0.0
