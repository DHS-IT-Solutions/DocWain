"""Unit tests for src/embedding/kg_enrichment.py"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.embedding.kg_enrichment import (
    _CACHE_KEY_TPL,
    _CACHE_TTL,
    enrich_chunk_text,
    fetch_kg_context_for_chunk,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


SAMPLE_KG_CONTEXT = {
    "document": "Invoice #4521",
    "entities": {
        "Acme Corp": "ORGANIZATION",
        "Project Alpha": "PROJECT",
    },
}


# ---------------------------------------------------------------------------
# enrich_chunk_text
# ---------------------------------------------------------------------------


class TestEnrichChunkText:
    def test_no_context_returns_original(self):
        text = "Some chunk text."
        assert enrich_chunk_text(text, None) == text

    def test_empty_context_returns_original(self):
        text = "Some chunk text."
        assert enrich_chunk_text(text, {}) == text

    def test_prefix_format(self):
        text = "Original body."
        result = enrich_chunk_text(text, SAMPLE_KG_CONTEXT)
        assert result.startswith("[Doc: Invoice #4521]")
        assert "[Acme Corp]" in result
        assert result.endswith(text)

    def test_entities_prepended(self):
        text = "Body text."
        result = enrich_chunk_text(text, SAMPLE_KG_CONTEXT)
        assert "[Acme Corp]" in result
        assert "[Project Alpha]" in result

    def test_prefix_does_not_exceed_max(self):
        many_entities = {f"Entity{i}": "TYPE" for i in range(50)}
        context = {"document": "Doc", "entities": many_entities}
        text = "Body."
        result = enrich_chunk_text(text, context, max_prefix_chars=50)
        prefix = result[: result.index("Body.")]
        assert len(prefix) <= 60  # small tolerance for separator

    def test_no_entities_only_doc_name(self):
        context = {"document": "My Document", "entities": {}}
        text = "Some text."
        result = enrich_chunk_text(text, context)
        assert "[Doc: My Document]" in result
        assert result.endswith(text)

    def test_no_doc_name_only_entities(self):
        context = {"document": "", "entities": {"Alpha": "ORG"}}
        text = "Body text."
        result = enrich_chunk_text(text, context)
        assert "[Alpha]" in result
        assert "[Doc:" not in result

    def test_max_prefix_chars_zero_returns_original(self):
        text = "Body text."
        result = enrich_chunk_text(text, SAMPLE_KG_CONTEXT, max_prefix_chars=0)
        # With zero budget nothing can be prepended
        assert result == text

    def test_entity_order_preserved(self):
        context = {
            "document": "",
            "entities": {"First": "A", "Second": "B", "Third": "C"},
        }
        text = "body"
        result = enrich_chunk_text(text, context, max_prefix_chars=200)
        first_pos = result.index("[First]")
        second_pos = result.index("[Second]")
        third_pos = result.index("[Third]")
        assert first_pos < second_pos < third_pos

    def test_large_doc_name_truncates_entities(self):
        long_name = "A" * 190
        context = {"document": long_name, "entities": {"Entity1": "TYPE"}}
        text = "body"
        result = enrich_chunk_text(text, context, max_prefix_chars=200)
        # Entity1 should not appear because prefix is already full from doc name
        assert "[Entity1]" not in result
        assert f"[Doc: {long_name}]" in result


# ---------------------------------------------------------------------------
# fetch_kg_context_for_chunk — Redis cache hit
# ---------------------------------------------------------------------------


class TestFetchKgContextRedisHit:
    def _make_redis(self, cached_value):
        client = MagicMock()
        if cached_value is None:
            client.get.return_value = None
        else:
            client.get.return_value = json.dumps(cached_value).encode()
        return client

    def test_returns_cached_value(self):
        redis = self._make_redis(SAMPLE_KG_CONTEXT)
        result = fetch_kg_context_for_chunk("doc1", "chunk1", redis_client=redis)
        assert result == SAMPLE_KG_CONTEXT

    def test_uses_correct_cache_key(self):
        redis = self._make_redis(SAMPLE_KG_CONTEXT)
        fetch_kg_context_for_chunk("doc1", "chunk1", redis_client=redis)
        expected_key = _CACHE_KEY_TPL.format(doc_id="doc1", chunk_id="chunk1")
        redis.get.assert_called_once_with(expected_key)

    def test_does_not_query_neo4j_on_cache_hit(self):
        redis = self._make_redis(SAMPLE_KG_CONTEXT)
        neo4j = MagicMock()
        fetch_kg_context_for_chunk("doc1", "chunk1", neo4j_store=neo4j, redis_client=redis)
        neo4j.query_entities.assert_not_called()


# ---------------------------------------------------------------------------
# fetch_kg_context_for_chunk — Redis miss, Neo4j query
# ---------------------------------------------------------------------------


class TestFetchKgContextNeo4jQuery:
    def _make_redis_miss(self):
        client = MagicMock()
        client.get.return_value = None
        return client

    def _make_neo4j(self, rows):
        store = MagicMock()
        store.query_entities.return_value = rows
        return store

    def test_queries_neo4j_on_cache_miss(self):
        redis = self._make_redis_miss()
        neo4j = self._make_neo4j([
            {"name": "Acme Corp", "type": "ORGANIZATION", "document": "Invoice #4521"},
        ])
        result = fetch_kg_context_for_chunk("doc1", "chunk1", neo4j_store=neo4j, redis_client=redis)
        neo4j.query_entities.assert_called_once_with("doc1")
        assert result is not None
        assert result["entities"]["Acme Corp"] == "ORGANIZATION"
        assert result["document"] == "Invoice #4521"

    def test_caches_result_after_neo4j_query(self):
        redis = self._make_redis_miss()
        neo4j = self._make_neo4j([
            {"name": "Acme Corp", "type": "ORGANIZATION", "document": "Invoice #4521"},
        ])
        fetch_kg_context_for_chunk("doc1", "chunk1", neo4j_store=neo4j, redis_client=redis)
        # Redis.set should have been called with correct TTL
        redis.set.assert_called_once()
        call_args = redis.set.call_args
        assert call_args.kwargs.get("ex") == _CACHE_TTL or (
            len(call_args.args) >= 3 and call_args.args[2] == _CACHE_TTL
        )

    def test_cache_key_used_for_set(self):
        redis = self._make_redis_miss()
        neo4j = self._make_neo4j([{"name": "E", "type": "T", "document": "D"}])
        fetch_kg_context_for_chunk("doc99", "chunk7", neo4j_store=neo4j, redis_client=redis)
        expected_key = _CACHE_KEY_TPL.format(doc_id="doc99", chunk_id="chunk7")
        set_key = redis.set.call_args.args[0]
        assert set_key == expected_key

    def test_no_neo4j_returns_none(self):
        redis = self._make_redis_miss()
        result = fetch_kg_context_for_chunk("doc1", "chunk1", redis_client=redis)
        assert result is None

    def test_no_redis_no_neo4j_returns_none(self):
        result = fetch_kg_context_for_chunk("doc1", "chunk1")
        assert result is None

    def test_neo4j_exception_returns_none(self):
        redis = self._make_redis_miss()
        neo4j = MagicMock()
        neo4j.query_entities.side_effect = RuntimeError("connection failed")
        result = fetch_kg_context_for_chunk("doc1", "chunk1", neo4j_store=neo4j, redis_client=redis)
        assert result is None

    def test_empty_neo4j_rows_returns_empty_entities(self):
        redis = self._make_redis_miss()
        neo4j = self._make_neo4j([])
        result = fetch_kg_context_for_chunk("doc1", "chunk1", neo4j_store=neo4j, redis_client=redis)
        assert result is not None
        assert result["entities"] == {}

    def test_multiple_entities_aggregated(self):
        redis = self._make_redis_miss()
        rows = [
            {"name": "Acme Corp", "type": "ORGANIZATION", "document": "Invoice #4521"},
            {"name": "Project Alpha", "type": "PROJECT", "document": "Invoice #4521"},
        ]
        neo4j = self._make_neo4j(rows)
        result = fetch_kg_context_for_chunk("doc1", "chunk1", neo4j_store=neo4j, redis_client=redis)
        assert "Acme Corp" in result["entities"]
        assert "Project Alpha" in result["entities"]


# ---------------------------------------------------------------------------
# fetch_kg_context_for_chunk — no Redis, only Neo4j
# ---------------------------------------------------------------------------


class TestFetchKgContextNoRedis:
    def test_queries_neo4j_without_redis(self):
        rows = [{"name": "Entity1", "type": "TYPE", "document": "Doc"}]
        neo4j = MagicMock()
        neo4j.query_entities.return_value = rows
        result = fetch_kg_context_for_chunk("doc1", "chunk1", neo4j_store=neo4j)
        assert result is not None
        assert result["entities"]["Entity1"] == "TYPE"


# ---------------------------------------------------------------------------
# fetch_kg_context_for_chunk — Redis error tolerance
# ---------------------------------------------------------------------------


class TestFetchKgContextRedisErrors:
    def test_redis_get_error_falls_through_to_neo4j(self):
        redis = MagicMock()
        redis.get.side_effect = ConnectionError("redis down")
        rows = [{"name": "E", "type": "T", "document": "D"}]
        neo4j = MagicMock()
        neo4j.query_entities.return_value = rows
        result = fetch_kg_context_for_chunk("doc1", "chunk1", neo4j_store=neo4j, redis_client=redis)
        assert result is not None

    def test_redis_set_error_still_returns_result(self):
        redis = MagicMock()
        redis.get.return_value = None
        redis.set.side_effect = ConnectionError("redis down")
        rows = [{"name": "E", "type": "T", "document": "D"}]
        neo4j = MagicMock()
        neo4j.query_entities.return_value = rows
        result = fetch_kg_context_for_chunk("doc1", "chunk1", neo4j_store=neo4j, redis_client=redis)
        assert result is not None
