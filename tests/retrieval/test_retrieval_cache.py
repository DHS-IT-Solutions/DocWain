"""Phase 3 Task 8 — Redis retrieval cache tests."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.retrieval.retrieval_cache import (
    RetrievalCache,
    _bundle_from_jsonable,
    _bundle_to_jsonable,
    _query_fingerprint,
)
from src.retrieval.types import RetrievalBundle


def _fake_redis():
    r = MagicMock()
    store: dict[str, str] = {}

    def _setex(k, ttl, v):
        store[str(k)] = v

    def _get(k):
        return store.get(str(k))

    def _scan_iter(match=None):
        # naive prefix match
        pat = (match or "").rstrip("*")
        return [k for k in list(store.keys()) if k.startswith(pat)]

    def _delete(k):
        store.pop(str(k), None)

    r.setex.side_effect = _setex
    r.get.side_effect = _get
    r.scan_iter.side_effect = _scan_iter
    r.delete.side_effect = _delete
    r._store = store
    return r


def _bundle(layer_a_chunks=None, layer_c_sme=None, degraded=None) -> RetrievalBundle:
    return RetrievalBundle(
        layer_a_chunks=layer_a_chunks or [{"doc_id": "d1", "chunk_id": "c1", "text": "hello", "score": 0.9}],
        layer_b_kg=[],
        layer_c_sme=layer_c_sme or [],
        layer_d_url=[],
        degraded_layers=degraded or [],
        per_layer_ms={"layer_a": 12.3},
    )


def test_fingerprint_is_stable_and_case_insensitive():
    assert _query_fingerprint("What is Q3?") == _query_fingerprint(" what is q3? ")
    assert _query_fingerprint("a") != _query_fingerprint("b")
    # 20 characters, hex
    fp = _query_fingerprint("hello world")
    assert len(fp) == 20
    assert all(c in "0123456789abcdef" for c in fp)


def test_set_and_get_roundtrip_preserves_bundle_shape():
    r = _fake_redis()
    cache = RetrievalCache(redis_client=r, ttl_seconds=300)
    bundle = _bundle(layer_c_sme=[{"kind": "sme_artifact", "narrative": "Q3 up", "score": 0.8}])
    fp = _query_fingerprint("Q3 revenue trend")
    cache.set(
        subscription_id="s1",
        profile_id="p1",
        query_fingerprint=fp,
        flag_set_version="v1",
        bundle=bundle,
    )
    got = cache.get(
        subscription_id="s1",
        profile_id="p1",
        query_fingerprint=fp,
        flag_set_version="v1",
    )
    assert isinstance(got, RetrievalBundle)
    assert got.layer_a_chunks == bundle.layer_a_chunks
    assert got.layer_c_sme == bundle.layer_c_sme
    assert got.per_layer_ms == bundle.per_layer_ms


def test_get_miss_on_different_flag_set_version():
    r = _fake_redis()
    cache = RetrievalCache(redis_client=r)
    fp = _query_fingerprint("q")
    cache.set(
        subscription_id="s",
        profile_id="p",
        query_fingerprint=fp,
        flag_set_version="v1",
        bundle=_bundle(),
    )
    assert (
        cache.get(
            subscription_id="s",
            profile_id="p",
            query_fingerprint=fp,
            flag_set_version="v2",
        )
        is None
    )


def test_get_miss_on_different_profile():
    r = _fake_redis()
    cache = RetrievalCache(redis_client=r)
    fp = _query_fingerprint("q")
    cache.set(
        subscription_id="s",
        profile_id="pA",
        query_fingerprint=fp,
        flag_set_version="v1",
        bundle=_bundle(),
    )
    assert (
        cache.get(
            subscription_id="s",
            profile_id="pB",
            query_fingerprint=fp,
            flag_set_version="v1",
        )
        is None
    )


def test_get_miss_on_different_subscription():
    r = _fake_redis()
    cache = RetrievalCache(redis_client=r)
    fp = _query_fingerprint("q")
    cache.set(
        subscription_id="sA",
        profile_id="p",
        query_fingerprint=fp,
        flag_set_version="v1",
        bundle=_bundle(),
    )
    assert (
        cache.get(
            subscription_id="sB",
            profile_id="p",
            query_fingerprint=fp,
            flag_set_version="v1",
        )
        is None
    )


def test_cache_disabled_when_redis_none():
    cache = RetrievalCache(redis_client=None)
    cache.set(
        subscription_id="s",
        profile_id="p",
        query_fingerprint="abc",
        flag_set_version="v",
        bundle=_bundle(),
    )
    assert (
        cache.get(
            subscription_id="s",
            profile_id="p",
            query_fingerprint="abc",
            flag_set_version="v",
        )
        is None
    )
    assert cache.invalidate_profile(subscription_id="s", profile_id="p") == 0


def test_redis_get_error_returns_none():
    r = MagicMock()
    r.get.side_effect = RuntimeError("down")
    cache = RetrievalCache(redis_client=r)
    assert (
        cache.get(
            subscription_id="s",
            profile_id="p",
            query_fingerprint="f",
            flag_set_version="v",
        )
        is None
    )


def test_redis_set_serialise_error_does_not_raise():
    r = _fake_redis()
    cache = RetrievalCache(redis_client=r)

    class Weird:
        pass

    # Inject an un-serialisable attribute by coercing to dataclass-like
    bundle = _bundle()
    bundle.per_layer_ms = {"a": Weird()}  # type: ignore[assignment]
    cache.set(
        subscription_id="s",
        profile_id="p",
        query_fingerprint="f",
        flag_set_version="v",
        bundle=bundle,
    )
    # Nothing stored is OK; the key must not have been indirectly set
    # with a half-serialised value either.
    assert list(r._store.keys()) == [] or all(isinstance(v, str) for v in r._store.values())


def test_invalidate_profile_removes_matching_keys_only():
    r = _fake_redis()
    cache = RetrievalCache(redis_client=r)
    cache.set(
        subscription_id="s",
        profile_id="pA",
        query_fingerprint="abc",
        flag_set_version="v1",
        bundle=_bundle(),
    )
    cache.set(
        subscription_id="s",
        profile_id="pA",
        query_fingerprint="def",
        flag_set_version="v1",
        bundle=_bundle(),
    )
    cache.set(
        subscription_id="s",
        profile_id="pB",
        query_fingerprint="abc",
        flag_set_version="v1",
        bundle=_bundle(),
    )
    deleted = cache.invalidate_profile(subscription_id="s", profile_id="pA")
    assert deleted == 2
    # pB key must still be readable
    assert (
        cache.get(
            subscription_id="s",
            profile_id="pB",
            query_fingerprint="abc",
            flag_set_version="v1",
        )
        is not None
    )
    # pA keys are gone
    assert (
        cache.get(
            subscription_id="s",
            profile_id="pA",
            query_fingerprint="abc",
            flag_set_version="v1",
        )
        is None
    )


def test_invalidate_profile_scan_error_returns_zero_no_raise():
    r = MagicMock()
    r.scan_iter.side_effect = RuntimeError("boom")
    cache = RetrievalCache(redis_client=r)
    assert cache.invalidate_profile(subscription_id="s", profile_id="p") == 0


def test_bump_flag_set_version_invalidates_naturally():
    """Cache keyed by old flag-set version is unreachable after a bump —
    the cache surface doesn't need to know; callers just re-key."""
    from src.config import feature_flags as ff

    ff.reset_flag_set_version()
    r = _fake_redis()
    cache = RetrievalCache(redis_client=r)
    fp = _query_fingerprint("q")
    v1 = ff.get_flag_set_version()
    cache.set(
        subscription_id="s",
        profile_id="p",
        query_fingerprint=fp,
        flag_set_version=f"v{v1}",
        bundle=_bundle(),
    )
    ff.bump_flag_set_version()
    v2 = ff.get_flag_set_version()
    assert v1 != v2
    # Caller keys off v2 → no hit
    assert (
        cache.get(
            subscription_id="s",
            profile_id="p",
            query_fingerprint=fp,
            flag_set_version=f"v{v2}",
        )
        is None
    )
    ff.reset_flag_set_version()


def test_key_layout_matches_spec():
    r = _fake_redis()
    cache = RetrievalCache(redis_client=r)
    cache.set(
        subscription_id="sub_A",
        profile_id="prof_X",
        query_fingerprint="abc123",
        flag_set_version="v7",
        bundle=_bundle(),
    )
    assert "dwx:retrieval:sub_A:prof_X:abc123:v7" in r._store
