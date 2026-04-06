from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from src.cache.query_cache import QueryCache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cache(get_return=None, side_effect=None):
    mock_redis = MagicMock()
    if side_effect:
        mock_redis.get.side_effect = side_effect
    else:
        mock_redis.get.return_value = get_return
    return QueryCache(redis_client=mock_redis), mock_redis


# ---------------------------------------------------------------------------
# Tier 1 – Embedding cache
# ---------------------------------------------------------------------------

def test_embedding_cache_roundtrip():
    cache, mock_redis = _make_cache(get_return=None)

    # Miss
    assert cache.get_embedding("test query") is None

    # Set
    cache.set_embedding("test query", [0.1, 0.2, 0.3])
    mock_redis.setex.assert_called_once()
    args = mock_redis.setex.call_args
    assert args[0][1] == QueryCache.EMBEDDING_TTL
    assert json.loads(args[0][2]) == [0.1, 0.2, 0.3]

    # Hit
    mock_redis.get.return_value = json.dumps([0.1, 0.2, 0.3])
    result = cache.get_embedding("test query")
    assert result == [0.1, 0.2, 0.3]


# ---------------------------------------------------------------------------
# Tier 2 – Search results cache
# ---------------------------------------------------------------------------

def test_search_results_cache_roundtrip():
    results = [{"id": "doc1", "score": 0.95}, {"id": "doc2", "score": 0.8}]
    cache, mock_redis = _make_cache(get_return=None)

    # Miss
    assert cache.get_search_results("find revenue", "tenant_abc") is None

    # Set
    cache.set_search_results("find revenue", "tenant_abc", results)
    assert mock_redis.setex.called
    set_args = mock_redis.setex.call_args
    assert json.loads(set_args[0][2]) == results

    # Hit
    mock_redis.get.return_value = json.dumps(results)
    got = cache.get_search_results("find revenue", "tenant_abc")
    assert got == results


# ---------------------------------------------------------------------------
# Tier 3 – Response cache
# ---------------------------------------------------------------------------

def test_response_cache_roundtrip():
    response = {"answer": "Revenue is $1M", "sources": ["doc1"]}
    cache, mock_redis = _make_cache(get_return=None)

    # Miss
    assert cache.get_response("what is revenue", "profile_1") is None

    # Set
    cache.set_response("what is revenue", "profile_1", response)
    assert mock_redis.setex.called
    set_args = mock_redis.setex.call_args
    assert json.loads(set_args[0][2]) == response

    # Hit
    mock_redis.get.return_value = json.dumps(response)
    got = cache.get_response("what is revenue", "profile_1")
    assert got == response


# ---------------------------------------------------------------------------
# Resilience
# ---------------------------------------------------------------------------

def test_cache_survives_redis_failure():
    cache, _ = _make_cache(side_effect=Exception("Redis down"))

    # None of these should raise
    assert cache.get_embedding("test") is None
    assert cache.get_search_results("test", "col") is None
    assert cache.get_response("test", "p") is None

    # set/invalidate should also silently swallow
    cache.set_embedding("test", [1.0])
    cache.set_search_results("test", "col", [])
    cache.set_response("test", "p", {})
    cache.invalidate_collection("col")


def test_cache_returns_none_when_no_redis():
    cache = QueryCache(redis_client=None)
    # Patch lazy resolution to also return None
    cache._get_redis = lambda: None

    assert cache.get_embedding("x") is None
    assert cache.get_search_results("x", "c") is None
    assert cache.get_response("x", "p") is None


# ---------------------------------------------------------------------------
# Invalidation
# ---------------------------------------------------------------------------

def test_invalidate_collection():
    cache, mock_redis = _make_cache(get_return=None)

    cache.invalidate_collection("tenant_abc")
    mock_redis.incr.assert_called_once_with("dw:qcache:version:tenant_abc")


def test_invalidation_changes_search_key():
    """After invalidation the version counter changes, so the search key changes."""
    mock_redis = MagicMock()
    cache = QueryCache(redis_client=mock_redis)

    # Before invalidation, version = 0
    mock_redis.get.return_value = None
    cache.set_search_results("q", "col", [{"id": "1"}])
    key_before = mock_redis.setex.call_args[0][0]

    # Simulate version bump
    mock_redis.get.return_value = b"1"
    mock_redis.setex.reset_mock()
    cache.set_search_results("q", "col", [{"id": "1"}])
    key_after = mock_redis.setex.call_args[0][0]

    assert key_before != key_after
    assert ":v0:" in key_before
    assert ":v1:" in key_after


# ---------------------------------------------------------------------------
# Key uniqueness
# ---------------------------------------------------------------------------

def test_cache_keys_are_different_for_different_queries():
    key1 = QueryCache._hash_key("query one")
    key2 = QueryCache._hash_key("query two")
    assert key1 != key2


def test_hash_key_deterministic():
    assert QueryCache._hash_key("hello") == QueryCache._hash_key("hello")
