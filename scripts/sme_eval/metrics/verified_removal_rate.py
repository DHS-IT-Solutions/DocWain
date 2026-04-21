"""verified_removal_rate metric.

Reads `metadata.citation_verifier_dropped` from each response's metadata.
Value is fraction of responses with zero drops. The gate threshold is ≥0.85 —
drops should happen sometimes (proving the verifier works) but not often.

At Phase 0, the production API may not emit this field. Treat missing-or-zero
as "no drops"; that's honest for the baseline.
"""
from __future__ import annotations

from collections.abc import Iterable

from scripts.sme_eval.metrics._base import Metric
from scripts.sme_eval.schema import EvalResult, MetricResult


def _had_drops(result: EvalResult) -> bool:
    return int(result.metadata.get("citation_verifier_dropped", 0)) > 0


class VerifiedRemovalRate(Metric):
    name = "verified_removal_rate"

    def compute(self, results: Iterable[EvalResult]) -> MetricResult:
        results = list(results)
        if not results:
            return MetricResult(
                metric_name=self.name,
                value=1.0,
                details={"num_results": 0, "num_with_drops": 0},
            )

        with_drops = sum(1 for r in results if _had_drops(r))
        total = len(results)
        return MetricResult(
            metric_name=self.name,
            value=(total - with_drops) / total,
            details={"num_results": total, "num_with_drops": with_drops},
        )
