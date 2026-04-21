"""Phase 2 tests for :class:`RecommendationBankBuilder`.

Replaces the Phase 1 skeleton — exercises frame filtering by insight type,
the ≥1 linked_insights + ≥1 evidence invariants, and the trace drop path.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from src.intelligence.sme.artifact_models import ArtifactItem, EvidenceRef
from src.intelligence.sme.builders.recommendation_bank import (
    RecommendationBankBuilder,
)


def _ctx():
    ctx = MagicMock()
    ctx.iter_profile_chunks.return_value = [
        {"doc_id": "d1", "chunk_id": "c1", "text": "Q3 DSO widened to 60 days."},
    ]
    return ctx


def _frame(frame_name, template, required_types=None):
    f = MagicMock()
    f.frame = frame_name
    f.template = template
    f.requires = {"insight_types": required_types or []}
    return f


def _adapter(frames=None):
    a = MagicMock()
    a.version = "1.0.0"
    persona = MagicMock()
    persona.role = "SME"
    persona.voice = "direct"
    persona.grounding_rules = []
    a.persona = persona
    if frames is None:
        frames = [
            _frame(
                "trend_based",
                "Given the {trend}, consider {action}.",
                required_types=["trend"],
            )
        ]
    a.recommendation_frames = frames
    return a


def _insight(item_id, detector="trend", narrative="Rev QoQ rose"):
    return ArtifactItem(
        item_id=item_id,
        artifact_type="insight",
        subscription_id="s",
        profile_id="p",
        text=narrative,
        evidence=[EvidenceRef(doc_id="d1", chunk_id="c1")],
        confidence=0.7,
        domain_tags=[detector],
        metadata={"detector": detector},
    )


def _items_payload(items):
    return json.dumps({"items": items})


def test_returns_recommendation_linked_to_insight():
    ins = _insight("insight:1")
    llm = MagicMock()
    llm.complete.side_effect = [
        _items_payload(
            [
                {
                    "recommendation": "Tighten AR",
                    "rationale": "Q3 DSO widened",
                    "linked_insights": ["insight:1"],
                    "estimated_impact": {"qualitative": "moderate"},
                    "assumptions": ["pricing stable"],
                    "caveats": ["seasonal"],
                    "evidence": [{"doc_id": "d1", "chunk_id": "c1"}],
                    "confidence": 0.75,
                    "domain_tags": ["finance"],
                }
            ]
        ),
    ]
    trace = MagicMock()
    builder = RecommendationBankBuilder(ctx=_ctx(), llm=llm, trace=trace)

    items = builder.build(
        subscription_id="s",
        profile_id="p",
        adapter=_adapter(),
        version=1,
        insight_items=[ins],
    )

    assert len(items) == 1
    item = items[0]
    assert isinstance(item, ArtifactItem)
    assert item.artifact_type == "recommendation"
    assert item.text == "Tighten AR"
    assert item.metadata["frame"] == "trend_based"
    assert item.metadata["linked_insights"] == ["insight:1"]
    assert item.evidence[0].doc_id == "d1"


def test_frame_skipped_when_no_matching_insights():
    llm = MagicMock()
    trace = MagicMock()
    builder = RecommendationBankBuilder(ctx=_ctx(), llm=llm, trace=trace)

    items = builder.build(
        subscription_id="s",
        profile_id="p",
        adapter=_adapter(),  # frame requires ["trend"]
        version=1,
        insight_items=[_insight("insight:2", detector="gap")],
    )
    assert items == []
    llm.complete.assert_not_called()
    events = [c.args[0] for c in trace.append.call_args_list]
    skipped = [
        e for e in events if e.get("stage") == "builder_no_matching_insights"
    ]
    assert len(skipped) == 1


def test_drops_recommendation_without_linked_insights():
    ins = _insight("insight:1")
    llm = MagicMock()
    llm.complete.side_effect = [
        _items_payload(
            [
                {
                    "recommendation": "dangling rec",
                    "rationale": "",
                    "linked_insights": [],
                    "evidence": [{"doc_id": "d1", "chunk_id": "c1"}],
                    "confidence": 0.6,
                }
            ]
        ),
    ]
    trace = MagicMock()
    builder = RecommendationBankBuilder(ctx=_ctx(), llm=llm, trace=trace)

    items = builder.build(
        subscription_id="s",
        profile_id="p",
        adapter=_adapter(),
        version=1,
        insight_items=[ins],
    )
    assert items == []
    events = [c.args[0] for c in trace.append.call_args_list]
    drops = [
        e for e in events if e.get("stage") == "builder_no_linked_insights"
    ]
    assert len(drops) == 1


def test_drops_recommendation_with_out_of_scope_linked_insights():
    ins = _insight("insight:1")
    llm = MagicMock()
    llm.complete.side_effect = [
        _items_payload(
            [
                {
                    "recommendation": "Tighten AR",
                    "rationale": "",
                    "linked_insights": ["insight:hallucinated"],
                    "evidence": [{"doc_id": "d1", "chunk_id": "c1"}],
                    "confidence": 0.7,
                }
            ]
        ),
    ]
    trace = MagicMock()
    builder = RecommendationBankBuilder(ctx=_ctx(), llm=llm, trace=trace)

    items = builder.build(
        subscription_id="s",
        profile_id="p",
        adapter=_adapter(),
        version=1,
        insight_items=[ins],
    )
    assert items == []
    events = [c.args[0] for c in trace.append.call_args_list]
    oos = [
        e
        for e in events
        if e.get("stage") == "builder_linked_insights_out_of_scope"
    ]
    assert len(oos) == 1


def test_drops_recommendation_without_evidence():
    ins = _insight("insight:1")
    llm = MagicMock()
    llm.complete.side_effect = [
        _items_payload(
            [
                {
                    "recommendation": "Tighten AR",
                    "rationale": "",
                    "linked_insights": ["insight:1"],
                    "evidence": [],
                    "confidence": 0.7,
                }
            ]
        ),
    ]
    trace = MagicMock()
    builder = RecommendationBankBuilder(ctx=_ctx(), llm=llm, trace=trace)

    items = builder.build(
        subscription_id="s",
        profile_id="p",
        adapter=_adapter(),
        version=1,
        insight_items=[ins],
    )
    assert items == []


def test_parse_failure_logged_and_continues():
    ins = _insight("insight:1")
    llm = MagicMock()
    llm.complete.side_effect = ["not-json"]
    trace = MagicMock()
    builder = RecommendationBankBuilder(ctx=_ctx(), llm=llm, trace=trace)
    items = builder.build(
        subscription_id="s",
        profile_id="p",
        adapter=_adapter(),
        version=1,
        insight_items=[ins],
    )
    assert items == []
    events = [c.args[0] for c in trace.append.call_args_list]
    parse = [e for e in events if e.get("stage") == "builder_parse_failure"]
    assert len(parse) == 1


def test_uses_recommendation_as_text_per_errata_section_3():
    ins = _insight("insight:1")
    llm = MagicMock()
    llm.complete.side_effect = [
        _items_payload(
            [
                {
                    "recommendation": "Shorten invoicing cycle to 30 days.",
                    "rationale": "Q3 DSO widened",
                    "linked_insights": ["insight:1"],
                    "evidence": [{"doc_id": "d1", "chunk_id": "c1"}],
                    "confidence": 0.75,
                }
            ]
        ),
    ]
    trace = MagicMock()
    builder = RecommendationBankBuilder(ctx=_ctx(), llm=llm, trace=trace)
    items = builder.build(
        subscription_id="s",
        profile_id="p",
        adapter=_adapter(),
        version=1,
        insight_items=[ins],
    )
    assert items[0].text == "Shorten invoicing cycle to 30 days."


def test_frame_without_required_types_accepts_all_insights():
    ins1 = _insight("insight:1", detector="trend")
    ins2 = _insight("insight:2", detector="anomaly")
    llm = MagicMock()
    llm.complete.side_effect = [
        _items_payload(
            [
                {
                    "recommendation": "Diversify customers",
                    "rationale": "reduce risk",
                    "linked_insights": ["insight:2"],
                    "evidence": [{"doc_id": "d1", "chunk_id": "c1"}],
                    "confidence": 0.7,
                }
            ]
        ),
    ]
    trace = MagicMock()
    adapter = _adapter(
        frames=[_frame("generic", "{action}", required_types=[])]
    )
    builder = RecommendationBankBuilder(ctx=_ctx(), llm=llm, trace=trace)
    items = builder.build(
        subscription_id="s",
        profile_id="p",
        adapter=adapter,
        version=1,
        insight_items=[ins1, ins2],
    )
    assert len(items) == 1
    assert items[0].metadata["linked_insights"] == ["insight:2"]


def test_public_build_without_insight_items_skips_typed_frames():
    """Calling the base ``build`` without insight_items disables any frame
    that has a non-empty ``requires.insight_types`` list."""
    llm = MagicMock()
    trace = MagicMock()
    builder = RecommendationBankBuilder(ctx=_ctx(), llm=llm, trace=trace)
    # Direct base-class-compatible call (no insight_items kwarg).
    items = builder.build(
        subscription_id="s",
        profile_id="p",
        adapter=_adapter(),
        version=1,
    )
    assert items == []
    # LLM was never invoked — the frame requires trends but none were
    # threaded, so the frame skipped the LLM call entirely.
    llm.complete.assert_not_called()
