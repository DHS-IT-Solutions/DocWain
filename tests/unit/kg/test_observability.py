import json
import time

from src.kg.observability import (
    KGLogEntry,
    build_kg_redis_key,
    serialize_kg_entry,
    write_kg_entry_if_redis,
)


def test_serialize_kg_entry_contains_required_fields():
    entry = KGLogEntry(
        doc_id="d1",
        status="KG_COMPLETED",
        nodes_created=5,
        edges_created=3,
        timings_ms={"ingest": 1200.0},
        error=None,
        completed_at=time.time(),
    )
    data = json.loads(serialize_kg_entry(entry))
    assert data["doc_id"] == "d1"
    assert data["status"] == "KG_COMPLETED"
    assert data["nodes_created"] == 5
    assert data["edges_created"] == 3
    assert data["timings_ms"]["ingest"] == 1200.0


def test_build_kg_redis_key_includes_doc_id():
    key = build_kg_redis_key("doc-123")
    assert "doc-123" in key
    assert key.startswith("kg:log:")


def test_write_kg_entry_if_redis_accepts_none_client():
    entry = KGLogEntry(doc_id="d2", status="KG_PENDING", nodes_created=0, edges_created=0,
                       timings_ms={}, error=None, completed_at=time.time())
    write_kg_entry_if_redis(redis_client=None, entry=entry)  # must not raise


def test_write_kg_entry_if_redis_sets_ttl():
    entry = KGLogEntry(doc_id="d3", status="KG_COMPLETED", nodes_created=1, edges_created=0,
                       timings_ms={}, error=None, completed_at=time.time())
    calls = {}

    class FakeRedis:
        def setex(self, key, ttl, value):
            calls["key"] = key
            calls["ttl"] = ttl
            calls["value"] = value

    write_kg_entry_if_redis(redis_client=FakeRedis(), entry=entry)
    assert calls["key"].endswith(":d3")
    assert calls["ttl"] == 7 * 24 * 3600
    assert "d3" in calls["value"]


def test_write_kg_entry_swallows_redis_exceptions():
    entry = KGLogEntry(doc_id="d4", status="KG_FAILED", nodes_created=0, edges_created=0,
                       timings_ms={}, error="neo4j unreachable", completed_at=time.time())

    class BrokenRedis:
        def setex(self, *a, **kw):
            raise RuntimeError("redis down")

    write_kg_entry_if_redis(redis_client=BrokenRedis(), entry=entry)  # must not raise
