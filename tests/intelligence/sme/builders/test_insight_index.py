"""Phase 2 tests for :class:`InsightIndexBuilder`.

Replaces the Phase 1 skeleton test. Covers the full per-detector LLM flow:
multiple detectors produce pooled items, invalid types drop, missing evidence
drops, and LLM failures are logged without aborting the builder.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from src.intelligence.sme.artifact_models import ArtifactItem
from src.intelligence.sme.builders.insight_index import InsightIndexBuilder


def _ctx():
    ctx = MagicMock()
    ctx.iter_profile_chunks.return_value = [
        {"doc_id": "d1", "chunk_id": "c1", "text": "Revenue rose QoQ."},
        {"doc_id": "d2", "chunk_id": "c2", "text": "Customer churn up 3%."},
    ]
    return ctx


def _adapter(detectors=None):
    a = MagicMock()
    a.version = "1.2.0"
    persona = MagicMock()
    persona.role = "finance SME"
    persona.voice = "quant"
    persona.grounding_rules = []
    a.persona = persona
    if detectors is None:
        detectors = [
            _detector("trend", "temporal_sweep", {"scope": "quarterly"}),
            _detector("anomaly", "entity_cluster", {"entity": "Customer"}),
        ]
    a.insight_detectors = detectors
    return a


def _detector(type_, rule, params):
    d = MagicMock()
    d.type = type_
    d.rule = rule
    d.params = params
    return d


def _items_payload(items):
    return json.dumps({"items": items})


def test_pools_items_across_detectors():
    llm = MagicMock()
    llm.complete.side_effect = [
        _items_payload(
            [
                {
                    "type": "trend",
                    "narrative": "QoQ revenue +8%",
                    "evidence": [{"doc_id": "d1", "chunk_id": "c1"}],
                    "confidence": 0.75,
                    "domain_tags": ["finance", "revenue"],
                }
            ]
        ),
        _items_payload(
            [
                {
                    "type": "anomaly",
                    "narrative": "Churn up 3%",
                    "evidence": [{"doc_id": "d2", "chunk_id": "c2"}],
                    "confidence": 0.7,
                    "domain_tags": ["finance", "churn"],
                }
            ]
        ),
    ]
    trace = MagicMock()
    builder = InsightIndexBuilder(ctx=_ctx(), llm=llm, trace=trace)

    items = builder.build(
        subscription_id="sub", profile_id="prof", adapter=_adapter(), version=1
    )

    assert len(items) == 2
    assert all(isinstance(i, ArtifactItem) for i in items)
    assert {i.metadata["detector"] for i in items} == {"trend", "anomaly"}
    for item in items:
        assert item.artifact_type == "insight"
        # detector type must be in domain_tags for retrieval filtering
        assert item.metadata["detector"] in item.domain_tags


def test_drops_items_missing_evidence():
    llm = MagicMock()
    llm.complete.side_effect = [
        _items_payload(
            [
                {
                    "type": "trend",
                    "narrative": "no evidence claim",
                    "evidence": [],
                    "confidence": 0.6,
                }
            ]
        ),
        _items_payload(
            [
                {
                    "type": "anomaly",
                    "narrative": "good one",
                    "evidence": [{"doc_id": "d1", "chunk_id": "c1"}],
                    "confidence": 0.6,
                }
            ]
        ),
    ]
    trace = MagicMock()
    builder = InsightIndexBuilder(ctx=_ctx(), llm=llm, trace=trace)

    items = builder.build(
        subscription_id="s", profile_id="p", adapter=_adapter(), version=1
    )
    assert len(items) == 1
    assert items[0].metadata["detector"] == "anomaly"


def test_parse_failure_logs_and_continues():
    llm = MagicMock()
    llm.complete.side_effect = [
        "not-json",
        _items_payload(
            [
                {
                    "type": "anomaly",
                    "narrative": "Churn up 3%",
                    "evidence": [{"doc_id": "d2", "chunk_id": "c2"}],
                    "confidence": 0.7,
                }
            ]
        ),
    ]
    trace = MagicMock()
    builder = InsightIndexBuilder(ctx=_ctx(), llm=llm, trace=trace)

    items = builder.build(
        subscription_id="s", profile_id="p", adapter=_adapter(), version=1
    )
    assert len(items) == 1
    events = [c.args[0] for c in trace.append.call_args_list]
    parse = [e for e in events if e.get("stage") == "builder_parse_failure"]
    assert len(parse) == 1
    assert parse[0]["detector"] == "trend"


def test_llm_error_logged_and_next_detector_still_runs():
    llm = MagicMock()
    llm.complete.side_effect = [
        RuntimeError("gateway flaky"),
        _items_payload(
            [
                {
                    "type": "anomaly",
                    "narrative": "Churn up 3%",
                    "evidence": [{"doc_id": "d2", "chunk_id": "c2"}],
                    "confidence": 0.7,
                }
            ]
        ),
    ]
    trace = MagicMock()
    builder = InsightIndexBuilder(ctx=_ctx(), llm=llm, trace=trace)

    items = builder.build(
        subscription_id="s", profile_id="p", adapter=_adapter(), version=1
    )
    assert len(items) == 1
    assert items[0].metadata["detector"] == "anomaly"
    llm_errors = [
        c.args[0]
        for c in trace.append.call_args_list
        if c.args[0].get("stage") == "builder_llm_error"
    ]
    assert len(llm_errors) == 1
    assert llm_errors[0]["detector"] == "trend"


def test_adapter_type_wins_over_llm_type_when_adapter_valid():
    llm = MagicMock()
    llm.complete.side_effect = [
        _items_payload(
            [
                {
                    "type": "not_a_real_type",
                    "narrative": "x",
                    "evidence": [{"doc_id": "d1", "chunk_id": "c1"}],
                    "confidence": 0.7,
                }
            ]
        ),
    ]
    trace = MagicMock()
    adapter = _adapter(
        detectors=[_detector("trend", "temporal_sweep", {})]
    )
    builder = InsightIndexBuilder(ctx=_ctx(), llm=llm, trace=trace)

    items = builder.build(
        subscription_id="s", profile_id="p", adapter=adapter, version=1
    )
    assert len(items) == 1
    assert items[0].metadata["detector"] == "trend"
    assert "trend" in items[0].domain_tags


def test_confidence_clipped_into_range():
    llm = MagicMock()
    llm.complete.side_effect = [
        _items_payload(
            [
                {
                    "type": "trend",
                    "narrative": "x",
                    "evidence": [{"doc_id": "d1", "chunk_id": "c1"}],
                    "confidence": 1.5,
                },
                {
                    "type": "trend",
                    "narrative": "y",
                    "evidence": [{"doc_id": "d1", "chunk_id": "c1"}],
                    "confidence": -0.3,
                },
            ]
        ),
    ]
    trace = MagicMock()
    adapter = _adapter(
        detectors=[_detector("trend", "temporal_sweep", {})]
    )
    builder = InsightIndexBuilder(ctx=_ctx(), llm=llm, trace=trace)

    items = builder.build(
        subscription_id="s", profile_id="p", adapter=adapter, version=1
    )
    assert items[0].confidence == 1.0
    assert items[1].confidence == 0.0


def test_uses_narrative_as_text_per_errata_section_3():
    llm = MagicMock()
    narrative = "Customer churn increased 3% in Q3."
    llm.complete.side_effect = [
        _items_payload(
            [
                {
                    "type": "anomaly",
                    "narrative": narrative,
                    "evidence": [{"doc_id": "d2", "chunk_id": "c2"}],
                    "confidence": 0.7,
                }
            ]
        ),
    ]
    trace = MagicMock()
    adapter = _adapter(
        detectors=[_detector("anomaly", "entity_cluster", {"entity": "Customer"})]
    )
    builder = InsightIndexBuilder(ctx=_ctx(), llm=llm, trace=trace)
    items = builder.build(
        subscription_id="s", profile_id="p", adapter=adapter, version=1
    )
    assert items[0].text == narrative


def test_temporal_scope_preserved_in_metadata():
    llm = MagicMock()
    llm.complete.side_effect = [
        _items_payload(
            [
                {
                    "type": "trend",
                    "narrative": "x",
                    "evidence": [{"doc_id": "d1", "chunk_id": "c1"}],
                    "confidence": 0.7,
                    "temporal_scope": "Q1-Q3 2026",
                    "entity_refs": ["Acme Corp"],
                }
            ]
        ),
    ]
    trace = MagicMock()
    adapter = _adapter(
        detectors=[_detector("trend", "temporal_sweep", {})]
    )
    builder = InsightIndexBuilder(ctx=_ctx(), llm=llm, trace=trace)
    items = builder.build(
        subscription_id="s", profile_id="p", adapter=adapter, version=1
    )
    assert items[0].metadata["temporal_scope"] == "Q1-Q3 2026"
    assert items[0].metadata["entity_refs"] == ["Acme Corp"]
