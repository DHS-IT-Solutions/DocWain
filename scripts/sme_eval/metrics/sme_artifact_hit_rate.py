"""sme_artifact_hit_rate metric.

Fraction of analytical-intent queries whose response included at least one
SME artifact in its evidence. Phase 0 baseline is 0.0 (no SME artifacts
exist yet); the metric is provisioned so Phase 3+ can measure uplift against
an already-calibrated before-picture.
"""
from __future__ import annotations

from collections.abc import Iterable

from scripts.sme_eval.metrics._base import Metric
from scripts.sme_eval.metrics.cross_doc_integration_rate import ANALYTICAL_INTENTS
from scripts.sme_eval.schema import EvalResult, MetricResult


def _has_sme_artifact(result: EvalResult) -> bool:
    layers = result.metadata.get("retrieval_layers", {}) or {}
    return int(layers.get("sme_artifacts_count", 0)) > 0


class SmeArtifactHitRate(Metric):
    name = "sme_artifact_hit_rate"

    def compute(self, results: Iterable[EvalResult]) -> MetricResult:
        analytical = [r for r in results if r.query.intent in ANALYTICAL_INTENTS]
        if not analytical:
            return MetricResult(
                metric_name=self.name,
                value=0.0,
                details={"num_analytical": 0, "num_hit": 0},
            )
        hits = sum(1 for r in analytical if _has_sme_artifact(r))
        total = len(analytical)
        return MetricResult(
            metric_name=self.name,
            value=hits / total,
            details={"num_analytical": total, "num_hit": hits},
        )
