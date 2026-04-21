"""insight_novelty metric (Phase 0 lexical proxy).

Measures: fraction of analytical-intent response n-grams (trigrams) that do
not appear verbatim in any single source excerpt attached to the response.

Phase 0 implementation uses source `excerpt` fields when present; falls back
to empty excerpt (treating the response as maximally novel) when excerpts
aren't available. Phase 2 will re-target this against pre-computed per-doc
summaries for a more semantic measure.
"""
from __future__ import annotations

import re
from collections.abc import Iterable

from scripts.sme_eval.metrics._base import Metric
from scripts.sme_eval.metrics.cross_doc_integration_rate import ANALYTICAL_INTENTS
from scripts.sme_eval.schema import EvalResult, MetricResult

_WORD = re.compile(r"\w+")


def _trigrams(text: str) -> set[tuple[str, str, str]]:
    toks = [t.lower() for t in _WORD.findall(text)]
    if len(toks) < 3:
        return set()
    return {(toks[i], toks[i + 1], toks[i + 2]) for i in range(len(toks) - 2)}


def _novelty_ratio(response: str, source_text: str) -> float:
    resp = _trigrams(response)
    if not resp:
        return 0.0
    src = _trigrams(source_text)
    novel = resp - src
    return len(novel) / len(resp)


class InsightNovelty(Metric):
    name = "insight_novelty"

    def compute(self, results: Iterable[EvalResult]) -> MetricResult:
        analytical = [r for r in results if r.query.intent in ANALYTICAL_INTENTS]

        if not analytical:
            return MetricResult(
                metric_name=self.name,
                value=0.0,
                details={"num_analytical": 0},
            )

        per_query: list[dict] = []
        total_ratio = 0.0
        for r in analytical:
            src_text = " ".join(s.get("excerpt", "") for s in r.sources)
            ratio = _novelty_ratio(r.response_text, src_text)
            per_query.append(
                {"query_id": r.query.query_id, "novelty_ratio": round(ratio, 3)}
            )
            total_ratio += ratio

        return MetricResult(
            metric_name=self.name,
            value=total_ratio / len(analytical),
            details={"num_analytical": len(analytical), "per_query": per_query},
        )
