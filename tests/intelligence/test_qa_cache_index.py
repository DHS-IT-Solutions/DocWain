"""Q&A cache-index emission + invalidation (user Task 14, ERRATA §13).

Pins the contract Phase 3's ``QAFastPath.lookup`` depends on:

* Keys are shaped ``qa_idx:{sub}:{prof}:{fingerprint}``.
* The fingerprint is SHA-256 over the whitespace-normalized lowercased
  question. Any drift here is a silent cache miss.
* Emission writes one key per Q&A pair with a finite TTL (default 24h).
* Invalidation scans + deletes every ``qa_idx:{sub}:{prof}:*`` key on
  ``PIPELINE_TRAINING_COMPLETED`` transition for that profile.
* Both operations are best-effort: a Redis outage logs + continues —
  the pipeline status flip is NEVER blocked on cache housekeeping.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.api import pipeline_api
from src.intelligence import qa_generator as qg


def test_fingerprint_is_normalized_sha256() -> None:
    import hashlib

    q = "  What is Revenue growth? "
    expected = hashlib.sha256(
        "what is revenue growth?".encode("utf-8")
    ).hexdigest()
    assert qg.qa_index_fingerprint(q) == expected


def test_fingerprint_collapses_internal_whitespace() -> None:
    assert qg.qa_index_fingerprint("a   b\tc") == qg.qa_index_fingerprint("a b c")


def test_emit_qa_index_writes_one_key_per_pair() -> None:
    redis = MagicMock()
    written = qg.emit_qa_index(
        subscription_id="s",
        profile_id="p",
        pairs=[
            {"question": "Q1?", "answer": "A1"},
            {"question": "Q2?", "answer": "A2"},
        ],
        redis_client=redis,
    )
    assert written == 2
    assert redis.set.call_count == 2
    keys = [c.args[0] for c in redis.set.call_args_list]
    assert all(k.startswith("qa_idx:s:p:") for k in keys)


def test_emit_qa_index_passes_ttl_via_ex_kwarg() -> None:
    redis = MagicMock()
    qg.emit_qa_index(
        subscription_id="s",
        profile_id="p",
        pairs=[{"question": "Q?", "answer": "A"}],
        redis_client=redis,
        ttl_s=1234,
    )
    assert redis.set.call_args.kwargs["ex"] == 1234


def test_emit_qa_index_skips_blank_questions() -> None:
    redis = MagicMock()
    written = qg.emit_qa_index(
        subscription_id="s",
        profile_id="p",
        pairs=[
            {"question": "", "answer": "A1"},
            {"question": None, "answer": "A2"},
            {"question": "Real?", "answer": "A3"},
        ],
        redis_client=redis,
    )
    assert written == 1
    assert redis.set.call_count == 1


def test_emit_qa_index_no_client_returns_zero() -> None:
    written = qg.emit_qa_index(
        subscription_id="s",
        profile_id="p",
        pairs=[{"question": "Q?", "answer": "A"}],
        redis_client=None,
    )
    # No real redis in this test environment; the fallback path returns 0.
    with patch.object(qg, "_resolve_redis_client", return_value=None):
        written = qg.emit_qa_index(
            subscription_id="s",
            profile_id="p",
            pairs=[{"question": "Q?", "answer": "A"}],
        )
    assert written == 0


def test_emit_qa_index_empty_sub_or_prof_raises() -> None:
    with pytest.raises(ValueError, match="subscription_id"):
        qg.emit_qa_index(
            subscription_id="",
            profile_id="p",
            pairs=[{"question": "Q?", "answer": "A"}],
        )
    with pytest.raises(ValueError, match="profile_id"):
        qg.emit_qa_index(
            subscription_id="s",
            profile_id="",
            pairs=[{"question": "Q?", "answer": "A"}],
        )


def test_emit_qa_index_redis_write_failure_is_nonfatal() -> None:
    redis = MagicMock()
    redis.set.side_effect = RuntimeError("redis overloaded")
    written = qg.emit_qa_index(
        subscription_id="s",
        profile_id="p",
        pairs=[{"question": "Q?", "answer": "A"}],
        redis_client=redis,
    )
    assert written == 0


# ---------------------------------------------------------------------------
# Invalidation lives in pipeline_api. Test via the module directly.
# ---------------------------------------------------------------------------
def test_invalidate_qa_index_deletes_every_matching_key() -> None:
    redis = MagicMock()
    redis.scan_iter.return_value = iter(
        ["qa_idx:s:p:f1", "qa_idx:s:p:f2", "qa_idx:s:p:f3"]
    )
    with patch(
        "src.api.dw_newron.get_redis_client",
        return_value=redis,
    ):
        pipeline_api.invalidate_qa_index(
            subscription_id="s", profile_id="p"
        )
    assert redis.delete.call_count == 3
    deleted_keys = {c.args[0] for c in redis.delete.call_args_list}
    assert deleted_keys == {"qa_idx:s:p:f1", "qa_idx:s:p:f2", "qa_idx:s:p:f3"}


def test_invalidate_qa_index_scans_with_profile_scoped_pattern() -> None:
    redis = MagicMock()
    redis.scan_iter.return_value = iter([])
    with patch(
        "src.api.dw_newron.get_redis_client",
        return_value=redis,
    ):
        pipeline_api.invalidate_qa_index(
            subscription_id="sub_a", profile_id="prof_x"
        )
    redis.scan_iter.assert_called_once_with("qa_idx:sub_a:prof_x:*")


def test_invalidate_qa_index_redis_unavailable_no_error() -> None:
    with patch(
        "src.api.dw_newron.get_redis_client",
        return_value=None,
    ):
        pipeline_api.invalidate_qa_index(
            subscription_id="s", profile_id="p"
        )
    # No raise => success.


def test_invalidate_qa_index_scan_failure_is_nonfatal() -> None:
    redis = MagicMock()
    redis.scan_iter.side_effect = RuntimeError("mitm")
    with patch(
        "src.api.dw_newron.get_redis_client",
        return_value=redis,
    ):
        # Must not raise — pipeline advance is already committed.
        pipeline_api.invalidate_qa_index(
            subscription_id="s", profile_id="p"
        )
