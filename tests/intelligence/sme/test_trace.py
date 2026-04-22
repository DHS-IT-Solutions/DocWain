"""Tests for SynthesisTraceWriter (Task 7).

Canonical surface per ERRATA §5: ``.append(event)``; ``.record`` is a kept
back-compat alias. QueryTraceWriter is added in Task 8 with its own test file.
"""
from unittest.mock import MagicMock

import pytest

from src.intelligence.sme.trace import SynthesisTraceWriter, TraceBlobAppender


@pytest.fixture
def appender():
    return MagicMock(spec=TraceBlobAppender)


def test_synthesis_path(appender):
    w = SynthesisTraceWriter(appender)
    w.open(subscription_id="s", profile_id="p", synthesis_id="syn1")
    w.append({"stage": "start"})
    w.close()
    assert (
        appender.append.call_args_list[0][0][0]
        == "sme_traces/synthesis/s/p/syn1.jsonl"
    )


def test_appends_jsonl(appender):
    w = SynthesisTraceWriter(appender)
    w.open(subscription_id="s", profile_id="p", synthesis_id="syn1")
    w.append({"stage": "builder_start", "builder": "dossier"})
    lines = [c[0][1] for c in appender.append.call_args_list]
    assert lines[0].endswith("\n")
    assert '"builder": "dossier"' in lines[0]


def test_refuses_append_before_open(appender):
    with pytest.raises(RuntimeError, match="open"):
        SynthesisTraceWriter(appender).append({"x": 1})


def test_record_alias_matches_append(appender):
    # ERRATA §5 keeps ``record`` as a deprecated alias; compare underlying
    # function since bound-methods are fresh descriptors each access.
    w = SynthesisTraceWriter(appender)
    assert w.record.__func__ is w.append.__func__


def test_multiple_appends_accumulate(appender):
    w = SynthesisTraceWriter(appender)
    w.open(subscription_id="s", profile_id="p", synthesis_id="syn1")
    w.append({"stage": "start"})
    w.append({"stage": "builder_start", "builder": "dossier"})
    w.append({"stage": "builder_end", "builder": "dossier", "ok": True})
    assert appender.append.call_count == 3
    # Every call targets the same path.
    paths = {c[0][0] for c in appender.append.call_args_list}
    assert paths == {"sme_traces/synthesis/s/p/syn1.jsonl"}


def test_close_blocks_further_append(appender):
    w = SynthesisTraceWriter(appender)
    w.open(subscription_id="s", profile_id="p", synthesis_id="syn1")
    w.append({"stage": "start"})
    w.close()
    with pytest.raises(RuntimeError, match="open"):
        w.append({"stage": "after_close"})


def test_non_ascii_payload_preserved(appender):
    w = SynthesisTraceWriter(appender)
    w.open(subscription_id="s", profile_id="p", synthesis_id="syn1")
    w.append({"note": "résumé日本"})
    line = appender.append.call_args[0][1]
    # ensure_ascii=False keeps characters intact (spec §11).
    assert "résumé日本" in line
