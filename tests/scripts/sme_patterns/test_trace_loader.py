"""Lean tests for the trace loader (schema + Blob iteration)."""
from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

from scripts.sme_patterns.schema import QueryRun, SynthesisRun
from scripts.sme_patterns.trace_loader import (
    TraceLoader,
    TraceWindow,
    parse_query_jsonl,
    parse_synth_jsonl,
)
from tests.scripts.sme_patterns.fixtures.query_trace_factory import make_query_jsonl
from tests.scripts.sme_patterns.fixtures.synth_trace_factory import make_synth_jsonl


def test_parse_synth_jsonl_happy_path_and_drops():
    text = make_synth_jsonl(synthesis_id="syn_42", drop_count=2)
    run = parse_synth_jsonl(text)
    assert isinstance(run, SynthesisRun)
    assert run.synthesis_id == "syn_42"
    assert len(run.verifier_drops) == 2
    assert run.per_builder["dossier"].items_persisted == 8
    assert run.completed_at is not None


def test_parse_query_jsonl_tolerates_malformed_and_filters_events():
    good = make_query_jsonl(query_id="q_1", rating=-1, citation_verifier_drops=3)
    bad_event = '{"event": "not_query_complete"}\n'
    malformed = "this is not json\n"
    runs = list(parse_query_jsonl(good + bad_event + malformed))
    assert len(runs) == 1
    assert isinstance(runs[0], QueryRun)
    assert runs[0].feedback.rating == -1
    assert runs[0].citation_verifier_drops == 3


def test_loader_filters_by_window_for_both_prefixes():
    blobs = {
        "sme_traces/synthesis/sub_a/prof_a/syn_in.jsonl": make_synth_jsonl(
            synthesis_id="syn_in", started_at=datetime(2026, 4, 10, 0, 0, 0),
        ),
        "sme_traces/synthesis/sub_a/prof_a/syn_out.jsonl": make_synth_jsonl(
            synthesis_id="syn_out", started_at=datetime(2026, 3, 1, 0, 0, 0),
        ),
        "sme_traces/queries/sub_a/prof_a/2026-04-05/q1.jsonl": make_query_jsonl(
            query_id="q1", captured_at=datetime(2026, 4, 5, 10, 0, 0),
        ),
        "sme_traces/queries/sub_a/prof_a/2026-03-05/q_old.jsonl": make_query_jsonl(
            query_id="q_old", captured_at=datetime(2026, 3, 5, 10, 0, 0),
        ),
    }
    list_blobs = MagicMock(side_effect=lambda prefix: [k for k in blobs if k.startswith(prefix)])
    read_blob = MagicMock(side_effect=lambda name: blobs[name])

    loader = TraceLoader(list_blobs=list_blobs, read_blob=read_blob)
    window = TraceWindow(
        start=datetime(2026, 4, 1),
        end=datetime(2026, 4, 30, 23, 59, 59),
    )
    synth_ids = [r.synthesis_id for r in loader.iter_synthesis_runs(window)]
    query_ids = [q.query_id for q in loader.iter_query_runs(window)]
    assert synth_ids == ["syn_in"]
    assert query_ids == ["q1"]


def test_loader_never_raises_on_bad_blob():
    blobs = {
        "sme_traces/queries/sub_a/prof_a/2026-04-05/ok.jsonl": make_query_jsonl(query_id="ok"),
        "sme_traces/queries/sub_a/prof_a/2026-04-05/bad.jsonl": "not jsonl",
    }
    list_blobs = MagicMock(side_effect=lambda prefix: [k for k in blobs if k.startswith(prefix)])
    read_blob = MagicMock(side_effect=lambda name: blobs[name])

    loader = TraceLoader(list_blobs=list_blobs, read_blob=read_blob)
    window = TraceWindow(
        start=datetime(2026, 4, 1),
        end=datetime(2026, 4, 30, 23, 59, 59),
    )
    qs = list(loader.iter_query_runs(window))
    assert [q.query_id for q in qs] == ["ok"]
