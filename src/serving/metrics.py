"""In-memory latency and request metrics for the vLLM serving layer.

Tracks request count, latency histogram percentiles (p50/p95/p99), and error
rate.  No external dependencies — pure Python stdlib only.
"""

from __future__ import annotations

import bisect
import threading
from typing import Optional

# ---------------------------------------------------------------------------
# Internal state — module-level, protected by a lock
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_request_count: int = 0
_error_count: int = 0
# Sorted list of latency samples (ms).  Kept sorted via bisect for O(log n)
# insert so percentile queries are O(1) after sort.
_latencies: list[float] = []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def record_request(latency_ms: float, status_code: int) -> None:
    """Record a single completed request.

    Args:
        latency_ms:  End-to-end request latency in milliseconds.
        status_code: HTTP status code returned to the caller.
    """
    global _request_count, _error_count

    with _lock:
        _request_count += 1
        if status_code >= 400:
            _error_count += 1
        # bisect.insort keeps the list sorted in O(log n) + O(n) shift —
        # acceptable for an in-process metrics store.
        bisect.insort(_latencies, float(latency_ms))


def get_metrics() -> dict:
    """Return a snapshot of current metrics.

    Returns:
        dict with keys:
            request_count   — total requests recorded
            error_count     — requests with status_code >= 400
            error_rate      — error_count / request_count (0.0 when no data)
            p50_ms          — median latency, or None when no data
            p95_ms          — 95th-percentile latency, or None when no data
            p99_ms          — 99th-percentile latency, or None when no data
    """
    with _lock:
        count = _request_count
        errors = _error_count
        samples = list(_latencies)  # snapshot while holding the lock

    error_rate = errors / count if count else 0.0
    p50 = _percentile(samples, 50)
    p95 = _percentile(samples, 95)
    p99 = _percentile(samples, 99)

    return {
        "request_count": count,
        "error_count": errors,
        "error_rate": round(error_rate, 6),
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
    }


def reset_metrics() -> None:
    """Reset all counters and latency samples (useful in tests)."""
    global _request_count, _error_count, _latencies

    with _lock:
        _request_count = 0
        _error_count = 0
        _latencies = []


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _percentile(sorted_samples: list[float], pct: int) -> Optional[float]:
    """Nearest-rank percentile from a pre-sorted sample list."""
    n = len(sorted_samples)
    if n == 0:
        return None
    # Nearest-rank formula: ceil(pct/100 * n) — 1-indexed then convert
    rank = max(1, int((pct / 100.0) * n + 0.5))
    rank = min(rank, n)
    return round(sorted_samples[rank - 1], 3)
