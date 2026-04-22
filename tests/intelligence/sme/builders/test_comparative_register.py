"""Phase 2 tests for :class:`ComparativeRegisterBuilder`.

Mirrors the dossier and insight_index tests: structural flow + error paths.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from src.intelligence.sme.artifact_models import ArtifactItem
from src.intelligence.sme.builders.comparative_register import (
    ComparativeRegisterBuilder,
)


def _ctx():
    ctx = MagicMock()
    ctx.iter_profile_chunks.return_value = [
        {"doc_id": "q3", "chunk_id": "a", "text": "Revenue $6M Q3."},
        {"doc_id": "q2", "chunk_id": "b", "text": "Revenue $5.4M Q2."},
    ]
    return ctx


def _axis(name, dimension, unit=None):
    a = MagicMock()
    a.name = name
    a.dimension = dimension
    a.unit = unit
    return a


def _adapter(axes=None):
    a = MagicMock()
    a.version = "1.0.0"
    persona = MagicMock()
    persona.role = "SME"
    persona.voice = "direct"
    persona.grounding_rules = []
    a.persona = persona
    if axes is None:
        axes = [_axis("revenue_qoq", "monetary", "USD")]
    a.comparison_axes = axes
    return a


def _items_payload(items):
    return json.dumps({"items": items})


def test_returns_comparative_items():
    llm = MagicMock()
    llm.complete.side_effect = [
        _items_payload(
            [
                {
                    "type": "delta",
                    "axis": "revenue_qoq",
                    "compared_items": ["q2", "q3"],
                    "analysis": "Q3 up $600K",
                    "resolution": None,
                    "evidence": [
                        {"doc_id": "q3", "chunk_id": "a"},
                        {"doc_id": "q2", "chunk_id": "b"},
                    ],
                    "confidence": 0.8,
                }
            ]
        ),
    ]
    trace = MagicMock()
    builder = ComparativeRegisterBuilder(ctx=_ctx(), llm=llm, trace=trace)

    items = builder.build(
        subscription_id="s", profile_id="p", adapter=_adapter(), version=1
    )
    assert len(items) == 1
    item = items[0]
    assert isinstance(item, ArtifactItem)
    assert item.artifact_type == "comparison"
    assert item.metadata["comparison_type"] == "delta"
    assert item.metadata["axis"] == "revenue_qoq"
    assert item.metadata["compared_items"] == ["q2", "q3"]
    assert item.text == "Q3 up $600K"


def test_pools_multiple_axes():
    llm = MagicMock()
    llm.complete.side_effect = [
        _items_payload(
            [
                {
                    "type": "delta",
                    "axis": "revenue",
                    "analysis": "Up",
                    "compared_items": ["q2", "q3"],
                    "evidence": [
                        {"doc_id": "q3", "chunk_id": "a"},
                        {"doc_id": "q2", "chunk_id": "b"},
                    ],
                    "confidence": 0.8,
                }
            ]
        ),
        _items_payload(
            [
                {
                    "type": "timeline",
                    "axis": "period",
                    "analysis": "Q2\u2192Q3",
                    "compared_items": ["q2", "q3"],
                    "evidence": [
                        {"doc_id": "q2", "chunk_id": "b"},
                        {"doc_id": "q3", "chunk_id": "a"},
                    ],
                    "confidence": 0.7,
                }
            ]
        ),
    ]
    trace = MagicMock()
    adapter = _adapter(
        axes=[
            _axis("revenue", "monetary", "USD"),
            _axis("period", "temporal"),
        ]
    )
    builder = ComparativeRegisterBuilder(ctx=_ctx(), llm=llm, trace=trace)
    items = builder.build(
        subscription_id="s", profile_id="p", adapter=adapter, version=1
    )
    assert len(items) == 2
    assert {i.metadata["axis"] for i in items} == {"revenue", "period"}


def test_drops_unknown_type_and_records_trace():
    llm = MagicMock()
    llm.complete.side_effect = [
        _items_payload(
            [
                {
                    "type": "vibes",
                    "axis": "revenue",
                    "analysis": "gibberish",
                    "evidence": [{"doc_id": "q3", "chunk_id": "a"}],
                    "confidence": 0.5,
                }
            ]
        ),
    ]
    trace = MagicMock()
    builder = ComparativeRegisterBuilder(ctx=_ctx(), llm=llm, trace=trace)
    items = builder.build(
        subscription_id="s", profile_id="p", adapter=_adapter(), version=1
    )
    assert items == []
    events = [c.args[0] for c in trace.append.call_args_list]
    invalid = [e for e in events if e.get("stage") == "builder_invalid_type"]
    assert len(invalid) == 1
    assert invalid[0]["got"] == "vibes"


def test_parse_failure_logged_and_continues():
    llm = MagicMock()
    llm.complete.side_effect = ["not-json"]
    trace = MagicMock()
    builder = ComparativeRegisterBuilder(ctx=_ctx(), llm=llm, trace=trace)
    items = builder.build(
        subscription_id="s", profile_id="p", adapter=_adapter(), version=1
    )
    assert items == []
    events = [c.args[0] for c in trace.append.call_args_list]
    parse = [e for e in events if e.get("stage") == "builder_parse_failure"]
    assert len(parse) == 1


def test_llm_error_logs_and_continues():
    llm = MagicMock()
    llm.complete.side_effect = [
        RuntimeError("gateway flaky"),
        _items_payload(
            [
                {
                    "type": "conflict",
                    "axis": "period",
                    "analysis": "Dates conflict.",
                    "evidence": [{"doc_id": "q3", "chunk_id": "a"}],
                    "confidence": 0.6,
                }
            ]
        ),
    ]
    trace = MagicMock()
    adapter = _adapter(
        axes=[
            _axis("revenue", "monetary"),
            _axis("period", "temporal"),
        ]
    )
    builder = ComparativeRegisterBuilder(ctx=_ctx(), llm=llm, trace=trace)
    items = builder.build(
        subscription_id="s", profile_id="p", adapter=adapter, version=1
    )
    assert len(items) == 1
    assert items[0].metadata["axis"] == "period"
    llm_errors = [
        c.args[0]
        for c in trace.append.call_args_list
        if c.args[0].get("stage") == "builder_llm_error"
    ]
    assert len(llm_errors) == 1


def test_falls_back_to_evidence_doc_ids_when_compared_items_missing():
    llm = MagicMock()
    llm.complete.side_effect = [
        _items_payload(
            [
                {
                    "type": "delta",
                    "axis": "revenue",
                    "analysis": "Up",
                    "evidence": [
                        {"doc_id": "q3", "chunk_id": "a"},
                        {"doc_id": "q2", "chunk_id": "b"},
                    ],
                    "confidence": 0.8,
                }
            ]
        ),
    ]
    trace = MagicMock()
    builder = ComparativeRegisterBuilder(
        ctx=_ctx(),
        llm=llm,
        trace=trace,
    )
    items = builder.build(
        subscription_id="s", profile_id="p", adapter=_adapter(), version=1
    )
    assert items[0].metadata["compared_items"] == sorted({"q3", "q2"})


def test_uses_analysis_as_text_per_errata_section_3():
    analysis = "Q3 revenue beats Q2 by 13%."
    llm = MagicMock()
    llm.complete.side_effect = [
        _items_payload(
            [
                {
                    "type": "delta",
                    "axis": "revenue",
                    "analysis": analysis,
                    "evidence": [
                        {"doc_id": "q3", "chunk_id": "a"},
                        {"doc_id": "q2", "chunk_id": "b"},
                    ],
                    "confidence": 0.8,
                }
            ]
        ),
    ]
    trace = MagicMock()
    builder = ComparativeRegisterBuilder(ctx=_ctx(), llm=llm, trace=trace)
    items = builder.build(
        subscription_id="s", profile_id="p", adapter=_adapter(), version=1
    )
    assert items[0].text == analysis
