"""Tests for QueryTraceWriter (Task 8).

Path: ``sme_traces/queries/{sub}/{prof}/{YYYY-MM-DD}/{query_id}.jsonl``.
Same ``_Base`` as :class:`SynthesisTraceWriter` — the date partition is
captured at ``open(...)`` time using a UTC-aware ``now`` callable (ERRATA §5,
no ``datetime.utcnow``).
"""
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from src.intelligence.sme.trace import QueryTraceWriter, TraceBlobAppender


@pytest.fixture
def appender():
    return MagicMock(spec=TraceBlobAppender)


def test_query_path_uses_date(appender):
    fixed = datetime(2026, 4, 20, 14, 30, tzinfo=timezone.utc)
    w = QueryTraceWriter(appender, now=lambda: fixed)
    w.open(subscription_id="s", profile_id="p", query_id="q42")
    w.append({"stage": "retrieval"})
    assert (
        appender.append.call_args_list[0][0][0]
        == "sme_traces/queries/s/p/2026-04-20/q42.jsonl"
    )


def test_query_path_crossing_midnight(appender):
    """Date partition is captured at ``open(...)`` time, not per ``append``.
    Long-running queries that span midnight still write to the opening day's
    directory — important so Phase 3 retrieval-trace tooling can locate every
    event for a given query by scanning a single directory."""
    t1 = datetime(2026, 4, 20, 23, 59, 59, tzinfo=timezone.utc)
    now_ref = {"now": t1}
    w = QueryTraceWriter(appender, now=lambda: now_ref["now"])
    w.open(subscription_id="s", profile_id="p", query_id="q99")
    # Advance clock past midnight before the next append.
    now_ref["now"] = datetime(2026, 4, 21, 0, 0, 5, tzinfo=timezone.utc)
    w.append({"stage": "rerank"})
    assert (
        appender.append.call_args_list[0][0][0]
        == "sme_traces/queries/s/p/2026-04-20/q99.jsonl"
    )


def test_default_now_returns_utc_aware(appender):
    """Default ``now`` callable must return a timezone-aware UTC datetime so
    the date partition never drifts with local TZ (constraint: no
    ``datetime.utcnow``)."""
    w = QueryTraceWriter(appender)
    value = w._now()  # noqa: SLF001 — verifying contract
    assert value.tzinfo is not None
    assert value.tzinfo.utcoffset(value) == timezone.utc.utcoffset(value)


def test_refuses_append_before_open(appender):
    with pytest.raises(RuntimeError, match="open"):
        QueryTraceWriter(appender).append({"x": 1})


def test_jsonl_payload_shape(appender):
    fixed = datetime(2026, 4, 20, tzinfo=timezone.utc)
    w = QueryTraceWriter(appender, now=lambda: fixed)
    w.open(subscription_id="s", profile_id="p", query_id="q1")
    w.append({"stage": "retrieval", "layers": ["a", "b"]})
    line = appender.append.call_args[0][1]
    assert line.endswith("\n")
    assert '"layers"' in line
    assert '"a"' in line


def test_close_resets_path(appender):
    fixed = datetime(2026, 4, 20, tzinfo=timezone.utc)
    w = QueryTraceWriter(appender, now=lambda: fixed)
    w.open(subscription_id="s", profile_id="p", query_id="q1")
    w.append({"x": 1})
    w.close()
    with pytest.raises(RuntimeError, match="open"):
        w.append({"x": 2})
