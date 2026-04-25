"""Insight lookup latency — p95 must be ≤ 50ms per spec Section 13.2.

This test exercises the InsightStore.list_for_profile path with a
realistic-size in-memory index (1000 insights across 50 profiles) and
asserts p95 over 1000 lookups stays under budget.
"""
from __future__ import annotations

import statistics
import time

import pytest

from src.intelligence.insights.schema import Insight, EvidenceSpan
from src.intelligence.insights.store import InsightStore, MongoIndexBackend


class _Coll:
    def __init__(self):
        self.docs = []

    def update_one(self, filter, update, upsert=False):
        match = next(
            (d for d in self.docs if all(d.get(k) == v for k, v in filter.items())),
            None,
        )
        if match is None:
            d = {**filter, **update.get("$set", {})}
            self.docs.append(d)
        else:
            match.update(update.get("$set", {}))
        return type("R", (), {"matched_count": int(match is not None)})()

    def find(self, query):
        def matches(d):
            for k, v in query.items():
                if isinstance(v, dict) and "$in" in v:
                    field_value = d.get(k)
                    if isinstance(field_value, list):
                        if not any(item in v["$in"] for item in field_value):
                            return False
                    else:
                        if field_value not in v["$in"]:
                            return False
                elif d.get(k) != v:
                    return False
            return True
        return [d for d in self.docs if matches(d)]


def _seed(store, *, n_profiles: int, per_profile: int) -> None:
    for p in range(n_profiles):
        for i in range(per_profile):
            insight = Insight(
                insight_id=f"p{p}-i{i}",
                profile_id=f"profile-{p}",
                subscription_id="s",
                document_ids=[f"DOC-{i}"],
                domain="generic",
                insight_type="anomaly",
                headline=f"H{i}",
                body=f"H{i} body content",
                evidence_doc_spans=[EvidenceSpan(
                    document_id=f"DOC-{i}", page=1, char_start=0, char_end=2,
                    quote=f"H{i}",
                )],
                confidence=0.5,
                severity="notice",
                adapter_version="generic@1.0",
            )
            store.write(insight)


@pytest.mark.perf
def test_p95_under_50ms():
    coll = _Coll()
    store = InsightStore(mongo_index=MongoIndexBackend(collection=coll), qdrant=None, neo4j=None)
    _seed(store, n_profiles=50, per_profile=20)

    timings = []
    for _ in range(1000):
        start = time.perf_counter_ns()
        rows = store.list_for_profile(profile_id="profile-25")
        end = time.perf_counter_ns()
        assert len(rows) == 20
        timings.append((end - start) / 1_000_000.0)
    p95 = statistics.quantiles(timings, n=100)[94]
    assert p95 <= 50.0, f"p95 {p95:.2f}ms exceeds 50ms budget"
