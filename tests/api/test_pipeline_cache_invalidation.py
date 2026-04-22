"""Phase 3 Task 8.5 — ``PIPELINE_TRAINING_COMPLETED`` must evict the
retrieval cache for the (sub, prof) pair.

These tests drive the wire-up point (:func:`_on_pipeline_training_completed`)
directly and also verify the ``_safe_invalidate_qa_index`` hook — which
fires on every training-complete branch in :mod:`src.api.pipeline_api`
— cascades into the retrieval cache invalidation.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def _fake_redis_with_keys(keys: list[str]):
    r = MagicMock()
    r.scan_iter.return_value = list(keys)
    return r


def test_training_complete_invalidates_retrieval_cache(monkeypatch):
    from src.api import pipeline_api
    from src.retrieval.retrieval_cache import RetrievalCache

    fake_redis = _fake_redis_with_keys(
        [
            "dwx:retrieval:sub_A:prof_X:abc123:v1",
            "dwx:retrieval:sub_A:prof_X:def456:v1",
        ]
    )
    cache = RetrievalCache(redis_client=fake_redis)
    pipeline_api.set_retrieval_cache_for_tests(cache)

    pipeline_api._on_pipeline_training_completed(
        subscription_id="sub_A", profile_id="prof_X"
    )
    assert fake_redis.delete.called
    deleted_keys = [c.args[0] for c in fake_redis.delete.call_args_list]
    assert all("sub_A:prof_X" in str(k) for k in deleted_keys)
    assert len(deleted_keys) == 2
    pipeline_api.reset_retrieval_cache_for_tests()


def test_training_complete_ignores_cache_when_redis_unavailable(monkeypatch):
    from src.api import pipeline_api
    from src.retrieval.retrieval_cache import RetrievalCache

    pipeline_api.set_retrieval_cache_for_tests(
        RetrievalCache(redis_client=None)
    )
    # Must not raise.
    pipeline_api._on_pipeline_training_completed(
        subscription_id="sub_A", profile_id="prof_X"
    )
    pipeline_api.reset_retrieval_cache_for_tests()


def test_safe_invalidate_cascades_to_retrieval_cache(monkeypatch):
    """The existing ``_safe_invalidate_qa_index`` hook ALSO bumps the
    retrieval cache — that's the only call site that needs to know about
    Task 8.5 since every ``PIPELINE_TRAINING_COMPLETED`` write goes
    through it per Phase 3 Task 7 wiring."""
    from src.api import pipeline_api

    called = {}

    def _fake_invalidate(*, subscription_id, profile_id):
        called["sub"] = subscription_id
        called["prof"] = profile_id

    monkeypatch.setattr(
        pipeline_api, "_on_pipeline_training_completed", _fake_invalidate
    )
    # qa_index invalidation uses its own Redis client; stub both to
    # avoid a live call.
    monkeypatch.setattr(
        pipeline_api, "invalidate_qa_index", lambda **_: None
    )

    pipeline_api._safe_invalidate_qa_index(
        subscription_id="sub_B", profile_id="prof_Y"
    )
    assert called["sub"] == "sub_B"
    assert called["prof"] == "prof_Y"


def test_retrieval_cache_invalidation_survives_cache_exception(monkeypatch):
    """A cache-level exception must never escape the pipeline hook."""
    from src.api import pipeline_api

    class Exploding:
        def invalidate_profile(self, **_):
            raise RuntimeError("redis is angry")

    pipeline_api.set_retrieval_cache_for_tests(Exploding())
    # Must not raise.
    pipeline_api._on_pipeline_training_completed(
        subscription_id="s", profile_id="p"
    )
    pipeline_api.reset_retrieval_cache_for_tests()


def test_get_retrieval_cache_lazy_instantiates(monkeypatch):
    """First call constructs the singleton with the process-wide redis."""
    from src.api import pipeline_api
    from src.retrieval.retrieval_cache import RetrievalCache

    pipeline_api.reset_retrieval_cache_for_tests()

    fake = MagicMock()
    monkeypatch.setattr(
        "src.api.dw_newron.get_redis_client", lambda: fake, raising=False
    )
    c = pipeline_api.get_retrieval_cache()
    assert isinstance(c, RetrievalCache)
    # Second call returns the same instance.
    assert pipeline_api.get_retrieval_cache() is c
    pipeline_api.reset_retrieval_cache_for_tests()
