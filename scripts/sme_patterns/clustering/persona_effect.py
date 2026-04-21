"""Pass 4 — Persona effect.

Compute a per-persona SME-score proxy per domain; flag personas whose proxy
regresses under the domain baseline by more than a configurable delta.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from scripts.sme_patterns.schema import Cluster, ClusterType, QueryRun


@dataclass(frozen=True)
class PersonaEffectConfig:
    regression_delta: float = 0.15
    min_queries_per_persona: int = 5


def _positive_outcome(q: QueryRun) -> bool:
    if q.citation_verifier_drops > 0:
        return False
    if q.feedback and q.feedback.rating == -1:
        return False
    return True


def sme_score_proxy(runs: list[QueryRun]) -> float:
    if not runs:
        return 0.0
    n = len(runs)
    pos = sum(1 for q in runs if _positive_outcome(q)) / n
    drops = sum(q.citation_verifier_drops for q in runs) / n
    drops_norm = max(0.0, 1.0 - min(1.0, drops / 3.0))
    fallback_rate = sum(1 for q in runs if q.honest_compact_fallback) / n
    score = 0.5 * pos + 0.3 * drops_norm + 0.2 * (1.0 - fallback_rate)
    return round(max(0.0, min(1.0, score)), 3)


def analyze_persona_effect(
    runs: Iterable[QueryRun],
    config: PersonaEffectConfig,
) -> list[Cluster]:
    runs = list(runs)
    if not runs:
        return []

    by_domain: dict[str, list[QueryRun]] = defaultdict(list)
    by_pair: dict[tuple[str, str], list[QueryRun]] = defaultdict(list)
    for q in runs:
        by_domain[q.profile_domain].append(q)
        by_pair[(q.profile_domain, q.adapter_persona_role)].append(q)

    baseline_score = {dom: sme_score_proxy(rs) for dom, rs in by_domain.items()}

    produced: list[Cluster] = []
    for (domain, persona), group in by_pair.items():
        if len(group) < config.min_queries_per_persona:
            continue
        score = sme_score_proxy(group)
        base = baseline_score.get(domain, 0.0)
        regression_flag = (base - score) >= config.regression_delta
        subs = sorted({q.subscription_id for q in group})
        short = (
            f"Persona '{persona}' on {domain}: proxy={score:.2f} "
            f"(domain baseline {base:.2f})"
            + ("  REGRESSION" if regression_flag else "")
        )
        produced.append(
            Cluster(
                cluster_id=f"persona_{domain}_{abs(hash(persona)) & 0xFFFF:04x}",
                cluster_type=ClusterType.PERSONA_EFFECT,
                size=len(group),
                subscription_ids=subs,
                primary_intent=None,
                profile_domain=domain,
                fingerprint_samples=[],
                short_description=short,
                signal_score=score,
                evidence={
                    "persona_role": persona,
                    "sme_score_proxy": score,
                    "domain_baseline": base,
                    "regression_flag": regression_flag,
                    "queries": len(group),
                },
            )
        )

    produced.sort(key=lambda c: c.signal_score)  # ascending -> worst first
    return produced
