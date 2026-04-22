"""Base class for SME eval metrics.

Each metric implements compute(results) -> MetricResult. Metrics operate
over a full batch of EvalResult records and return a single aggregated
score with per-query details.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from scripts.sme_eval.schema import EvalResult, MetricResult


class Metric(ABC):
    """A metric computed over a batch of eval results."""

    name: str  # must be set on subclass

    @abstractmethod
    def compute(self, results: Iterable[EvalResult]) -> MetricResult:
        """Compute the metric value over a batch.

        Returns a MetricResult whose `value` is the aggregated score in
        [0.0, 1.0] (or natural unit if non-fractional) and `details`
        carries per-query breakdowns needed for debugging.
        """
