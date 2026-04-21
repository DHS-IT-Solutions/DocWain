"""Latency aggregation per intent.

Computes p50/p95/p99 of total_ms per intent, excluding failed calls.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import numpy as np

from scripts.sme_eval.schema import EvalResult


def aggregate_latency_per_intent(results: Iterable[EvalResult]) -> dict[str, dict[str, float]]:
    """Return { intent: {count, p50, p95, p99, mean, max} }."""
    grouped: dict[str, list[float]] = defaultdict(list)
    for r in results:
        if r.api_status != 200:
            continue
        grouped[r.query.intent].append(r.latency.total_ms)

    out: dict[str, dict[str, float]] = {}
    for intent, samples in grouped.items():
        arr = np.asarray(samples, dtype=float)
        out[intent] = {
            "count": int(arr.size),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "mean": float(arr.mean()),
            "max": float(arr.max()),
        }
    return out
