import json
import time

import pytest

from src.extraction.vision.observability import (
    ExtractionLogEntry,
    build_redis_key,
    serialize_entry,
    write_entry_if_redis,
)


def test_serialize_entry_contains_required_fields():
    entry = ExtractionLogEntry(
        doc_id="d1",
        format="pdf_scanned",
        path_taken="vision",
        timings_ms={"file_adapter": 10.0, "docintel_route": 300.0, "vision_pass": 1200.0,
                    "coverage_verify": 400.0, "fallback": 0.0},
        routing_decision={"format": "pdf_scanned", "suggested_path": "vision", "confidence": 0.8},
        coverage_score=0.98,
        fallback_invocations=[],
        human_review=False,
        completed_at=time.time(),
    )
    out = serialize_entry(entry)
    data = json.loads(out)
    assert data["doc_id"] == "d1"
    assert data["path_taken"] == "vision"
    assert data["coverage_score"] == 0.98
    assert data["timings_ms"]["vision_pass"] == 1200.0


def test_build_redis_key_includes_doc_id():
    key = build_redis_key("doc-123")
    assert "doc-123" in key


def test_write_entry_if_redis_accepts_none_client():
    entry = ExtractionLogEntry(doc_id="d2", format="docx", path_taken="native",
                               timings_ms={}, routing_decision={}, coverage_score=1.0,
                               fallback_invocations=[], human_review=False, completed_at=time.time())
    write_entry_if_redis(redis_client=None, entry=entry)


def test_write_entry_if_redis_sets_ttl(monkeypatch):
    entry = ExtractionLogEntry(doc_id="d3", format="docx", path_taken="native",
                               timings_ms={}, routing_decision={}, coverage_score=1.0,
                               fallback_invocations=[], human_review=False, completed_at=time.time())

    calls = {}

    class FakeRedis:
        def setex(self, key, ttl, value):
            calls["key"] = key
            calls["ttl"] = ttl
            calls["value"] = value

    write_entry_if_redis(redis_client=FakeRedis(), entry=entry)
    assert "d3" in calls["key"]
    assert calls["ttl"] == 7 * 24 * 3600
    assert "d3" in calls["value"]
