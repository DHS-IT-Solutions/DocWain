"""cross_doc_integration_rate metric.

Measures: fraction of analytical-intent responses that cite evidence from
≥2 distinct documents in the profile. Analytical intents are compare,
analyze, diagnose, recommend, investigate.
"""
from __future__ import annotations

from collections.abc import Iterable

from scripts.sme_eval.metrics._base import Metric
from scripts.sme_eval.schema import EvalResult, MetricResult

ANALYTICAL_INTENTS: frozenset[str] = frozenset(
    {"analyze", "diagnose", "recommend", "investigate", "compare"}
)


def _distinct_docs(result: EvalResult) -> int:
    doc_ids = {s.get("doc_id") for s in result.sources if s.get("doc_id")}
    return len(doc_ids)


class CrossDocIntegrationRate(Metric):
    name = "cross_doc_integration_rate"

    def compute(self, results: Iterable[EvalResult]) -> MetricResult:
        analytical = [r for r in results if r.query.intent in ANALYTICAL_INTENTS]

        if not analytical:
            return MetricResult(
                metric_name=self.name,
                value=1.0,
                details={"num_analytical": 0, "num_integrated": 0},
            )

        integrated = sum(1 for r in analytical if _distinct_docs(r) >= 2)
        total = len(analytical)
        return MetricResult(
            metric_name=self.name,
            value=integrated / total,
            details={
                "num_analytical": total,
                "num_integrated": integrated,
                "per_query": [
                    {
                        "query_id": r.query.query_id,
                        "distinct_docs": _distinct_docs(r),
                    }
                    for r in analytical
                ],
            },
        )
