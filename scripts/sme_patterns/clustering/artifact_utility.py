"""Pass 3 — Artifact-utility ranking.

For each retrieval layer, compute retrieval_rate (how often it fires) and
citation_rate (proxy — share of positive-outcome queries among those that
pulled from the layer). Dead-weight layers (high retrieval + low citation)
are flagged for review.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from scripts.sme_patterns.schema import Cluster, ClusterType, QueryRun

_LAYERS: tuple[str, ...] = ("chunks", "kg", "sme_artifacts", "url")


@dataclass(frozen=True)
class ArtifactUtilityConfig:
    dead_weight_retrieval_min: float = 0.5
    dead_weight_citation_max: float = 0.25


def _positive_outcome(q: QueryRun) -> bool:
    if q.citation_verifier_drops > 0:
        return False
    if q.feedback and q.feedback.rating == -1:
        return False
    return True


def analyze_artifact_utility(
    runs: Iterable[QueryRun],
    config: ArtifactUtilityConfig,
) -> list[Cluster]:
    runs = list(runs)
    total = len(runs)

    out: list[Cluster] = []
    for layer in _LAYERS:
        used = [q for q in runs if q.retrieval_layers.get(layer, 0) >= 1]
        retrieval_rate = (len(used) / total) if total else 0.0
        if used:
            positive = sum(1 for q in used if _positive_outcome(q))
            citation_rate = positive / len(used)
        else:
            citation_rate = 0.0
        dead_weight = (
            retrieval_rate >= config.dead_weight_retrieval_min
            and citation_rate < config.dead_weight_citation_max
        )
        subs = sorted({q.subscription_id for q in used})
        short = (
            f"Layer '{layer}': used in {retrieval_rate:.0%} of queries, "
            f"positive-outcome rate {citation_rate:.0%}"
            + (" — DEAD WEIGHT" if dead_weight else "")
        )
        out.append(
            Cluster(
                cluster_id=f"artifact_{layer}",
                cluster_type=ClusterType.ARTIFACT_UTILITY,
                size=len(used),
                subscription_ids=subs,
                primary_intent=None,
                profile_domain=None,
                fingerprint_samples=[],
                short_description=short,
                signal_score=round(citation_rate, 3),
                evidence={
                    "layer": layer,
                    "retrieval_rate": round(retrieval_rate, 3),
                    "citation_rate": round(citation_rate, 3),
                    "total_queries": total,
                    "dead_weight_flag": dead_weight,
                },
            )
        )
    return out
