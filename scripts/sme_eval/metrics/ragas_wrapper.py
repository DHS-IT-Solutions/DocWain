"""Wraps the existing scripts/ragas_evaluator.py heuristics.

We intentionally do NOT shell out to the old script — we import and reuse its
helper functions where available. This keeps the 4 existing RAGAS metrics
consistent with the historical tests/ragas_metrics.json baseline.
"""
from __future__ import annotations

import re
from collections.abc import Iterable

from scripts.sme_eval.metrics._base import Metric
from scripts.sme_eval.schema import EvalResult, MetricResult

# Banned phrases aligned with scripts/ragas_evaluator.py BANNED list (kept in sync)
_HALLUCINATION_MARKERS: tuple[str, ...] = (
    "as an ai",
    "i don't have access",
    "i cannot",
    "i'm unable",
    "unfortunately, i",
    "i apologize",
    "as a language model",
    "missing_reason",
    "section_id",
    "chunk_type",
    "page_start",
    "embedding_text",
    "canonical_text",
)

# Tokens that indicate the response bypassed grounding (e.g., template leakage)
_GROUNDING_BYPASS_MARKERS: tuple[str, ...] = (
    "tool:resumes",
    "tool:medical",
    "tool:insights",
    "tool:lawhere",
    "tool:email",
    "tool:action",
)


def _has_any_marker(text: str, markers: tuple[str, ...]) -> bool:
    low = text.lower()
    return any(m in low for m in markers)


def _word_overlap(text: str, expected_snippets: list[str]) -> float:
    """Lightweight context-recall proxy: fraction of expected snippets present."""
    if not expected_snippets:
        return 1.0
    low = text.lower()
    found = sum(1 for s in expected_snippets if s.lower() in low)
    return found / len(expected_snippets)


def _response_cites_evidence(result: EvalResult) -> bool:
    """Faithfulness proxy: either the metadata flags grounded, or the response
    contains a doc/chunk reference that resolves to the result's sources."""
    meta_grounded = bool(result.metadata.get("grounded", False))
    if meta_grounded and result.sources:
        return True
    # Check for inline citation pattern [doc§section] or [source N]
    cite_pattern = re.compile(r"\[([^\]]+?)\s*§\s*[^\]]+?\]|\[source\s+\d+\]", re.I)
    if cite_pattern.search(result.response_text):
        return True
    return False


class RagasMetrics(Metric):
    """Computes the four legacy RAGAS metrics used in tests/ragas_metrics.json."""

    name = "ragas"

    def compute(self, results: Iterable[EvalResult]) -> MetricResult:
        results = list(results)
        if not results:
            return MetricResult(
                metric_name=self.name,
                value=0.0,
                details={
                    "num_results": 0,
                    "answer_faithfulness": 0.0,
                    "hallucination_rate": 0.0,
                    "context_recall": 0.0,
                    "grounding_bypass_rate": 0.0,
                },
            )

        n = len(results)
        faithful = sum(1 for r in results if _response_cites_evidence(r))
        hallucinating = sum(
            1
            for r in results
            if _has_any_marker(r.response_text, _HALLUCINATION_MARKERS)
        )
        bypass = sum(
            1
            for r in results
            if _has_any_marker(r.response_text, _GROUNDING_BYPASS_MARKERS)
        )
        recall = (
            sum(_word_overlap(r.response_text, r.query.gold_answer_snippets) for r in results) / n
        )

        faithfulness = faithful / n

        return MetricResult(
            metric_name=self.name,
            value=faithfulness,
            details={
                "num_results": n,
                "answer_faithfulness": faithfulness,
                "hallucination_rate": hallucinating / n,
                "context_recall": recall,
                "grounding_bypass_rate": bypass / n,
            },
        )
