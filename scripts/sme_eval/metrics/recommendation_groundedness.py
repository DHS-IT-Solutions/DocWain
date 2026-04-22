"""recommendation_groundedness metric.

Measures: fraction of `recommend`-intent responses where every extracted
recommendation sentence is grounded in evidence (either cited inline,
supported by sources in the response payload, or exposed as explicit ad-hoc
reasoning).

At Phase 0 (no Recommendation Bank yet) this is a baseline measurement — it
will be re-run post-Phase 4 to measure the uplift.
"""
from __future__ import annotations

import re
from collections.abc import Iterable

from scripts.sme_eval.metrics._base import Metric
from scripts.sme_eval.schema import EvalResult, MetricResult

# Imperative verbs that commonly open a recommendation sentence
_RECOMMENDATION_VERBS: tuple[str, ...] = (
    "consolidate", "reduce", "increase", "eliminate", "switch",
    "adopt", "implement", "replace", "review", "renegotiate",
    "investigate", "consider", "explore", "audit", "prioritize",
    "streamline", "automate", "outsource", "hire", "defer",
)

_SHOULD_PATTERN = re.compile(
    r"\b(should|recommend(ed)?|suggest(ed)?|propose(d)?|advise[d]?|could|might)\b",
    re.IGNORECASE,
)

_INLINE_CITATION = re.compile(r"\[[^\]]+?\]|\(see [^)]+\)|\bfrom\s+[A-Za-z0-9_\-§.]+", re.IGNORECASE)

_ADHOC_REASONING_MARKERS: tuple[str, ...] = (
    "because", "since", "given that", "the data shows", "this implies",
)


def extract_recommendation_sentences(text: str) -> list[str]:
    """Split text into sentences and return only those that look like recommendations."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    found: list[str] = []
    for s in sentences:
        low = s.lower().strip()
        if not low:
            continue
        opens_imperative = any(low.startswith(v) for v in _RECOMMENDATION_VERBS)
        has_recommend_verb = (
            _SHOULD_PATTERN.search(low) is not None
            or any(v in low for v in _RECOMMENDATION_VERBS)
        )
        if opens_imperative or has_recommend_verb:
            found.append(s.strip())
    return found


def _is_grounded(sentence: str, result: EvalResult) -> bool:
    """A sentence counts as grounded if any of:
    - It contains an inline citation pattern, OR
    - The response has at least one source AND the sentence references a doc_id,
    - The sentence exposes ad-hoc reasoning (because/since/given that/...)
    """
    if _INLINE_CITATION.search(sentence):
        return True
    if result.sources:
        for src in result.sources:
            doc_id = src.get("doc_id", "")
            if doc_id and doc_id.lower() in sentence.lower():
                return True
    low = sentence.lower()
    if any(m in low for m in _ADHOC_REASONING_MARKERS):
        return True
    return False


class RecommendationGroundedness(Metric):
    name = "recommendation_groundedness"

    def compute(self, results: Iterable[EvalResult]) -> MetricResult:
        rec_results = [r for r in results if r.query.intent == "recommend"]

        if not rec_results:
            return MetricResult(
                metric_name=self.name,
                value=1.0,
                details={
                    "num_recommend_queries": 0,
                    "num_grounded": 0,
                    "num_ungrounded": 0,
                },
            )

        grounded = 0
        ungrounded = 0
        per_query: list[dict] = []
        for r in rec_results:
            recs = extract_recommendation_sentences(r.response_text)
            if not recs:
                # Response is a recommend-intent query with no recommendations.
                # Counts as ungrounded — the model produced no actionable output.
                ungrounded += 1
                per_query.append(
                    {"query_id": r.query.query_id, "recommendations_found": 0}
                )
                continue
            all_grounded = all(_is_grounded(s, r) for s in recs)
            if all_grounded:
                grounded += 1
            else:
                ungrounded += 1
            per_query.append(
                {
                    "query_id": r.query.query_id,
                    "recommendations_found": len(recs),
                    "all_grounded": all_grounded,
                }
            )

        total = len(rec_results)
        return MetricResult(
            metric_name=self.name,
            value=grounded / total,
            details={
                "num_recommend_queries": total,
                "num_grounded": grounded,
                "num_ungrounded": ungrounded,
                "per_query": per_query,
            },
        )
