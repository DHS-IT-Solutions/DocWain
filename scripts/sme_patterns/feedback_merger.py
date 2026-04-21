"""Merge feedback signals into QueryRuns.

Preference:
  1. Explicit feedback already on the QueryRun (set by the Phase 1 query-trace
     writer at generation time).
  2. Redis ``FeedbackTracker`` aggregates — used only when (a) the QueryRun has
     no explicit feedback and (b) the profile's current low-confidence ratio
     indicates the reasoner struggled. Everything merged this way is tagged
     ``source="implicit"`` so downstream clustering can weight it accordingly.

Redis client path: we do NOT import ``src.utils.redis_client`` (that module
does not exist — per ERRATA §18). The production wiring uses
``src.utils.redis_cache`` through the tracker constructor in the orchestrator.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Iterable

from scripts.sme_patterns.schema import QueryFeedback, QueryRun

logger = logging.getLogger(__name__)


def merge_feedback(
    runs: Iterable[QueryRun],
    feedback_tracker,
    *,
    implicit_rating_when_low: int = -1,
    low_confidence_threshold: float = 0.3,
) -> list[QueryRun]:
    """Return a new list of QueryRuns with feedback filled where missing."""
    runs = list(runs)

    missing_by_profile: dict[str, list[int]] = defaultdict(list)
    for idx, r in enumerate(runs):
        if r.feedback is None:
            missing_by_profile[r.profile_id].append(idx)

    if not missing_by_profile:
        return runs

    out = list(runs)
    for profile_id, idxs in missing_by_profile.items():
        try:
            metrics = feedback_tracker.get_profile_metrics(profile_id) or {}
        except Exception:
            # Narrow catch is not possible — tracker impl may raise RedisError,
            # network errors, etc. We treat "no signal" as "leave as None".
            logger.exception(
                "feedback_tracker raised for profile %s; leaving implicit None",
                profile_id,
            )
            continue
        total = int(metrics.get("total_queries", 0) or 0)
        low_count = int(metrics.get("low_confidence_count", 0) or 0)
        ratio = (low_count / total) if total > 0 else 0.0
        if ratio <= low_confidence_threshold:
            continue
        implicit = QueryFeedback(
            rating=implicit_rating_when_low,
            edited=False,
            follow_up_count=0,
            source="implicit",
        )
        for i in idxs:
            out[i] = out[i].model_copy(update={"feedback": implicit})
    return out
