"""Phase 2 tests for :class:`SMEDossierBuilder`.

Phase 1's skeleton test asserted the builder returned ``[]``; Phase 2 replaces
that with full LLM-driven per-section synthesis tests. The tests inject a
MagicMock LLM and MagicMock trace sink so the builder runs in isolation — the
verifier and storage wire-up live in the orchestrator (Task 7).
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from src.intelligence.sme.artifact_models import ArtifactItem
from src.intelligence.sme.builders.dossier import SMEDossierBuilder


def _adapter(section_weights=None):
    a = MagicMock()
    a.version = "1.2.0"
    persona = MagicMock()
    persona.role = "senior financial analyst"
    persona.voice = "direct, quantitative"
    persona.grounding_rules = ["cite each claim"]
    a.persona = persona
    dossier_cfg = MagicMock()
    dossier_cfg.section_weights = section_weights or {
        "overview": 0.4,
        "trends": 0.4,
        "risks": 0.2,
    }
    dossier_cfg.prompt_template = "prompts/finance_dossier.md"
    a.dossier = dossier_cfg
    return a


def _ctx():
    ctx = MagicMock()
    ctx.iter_profile_chunks.return_value = [
        {"doc_id": "d1", "chunk_id": "c1", "text": "Q1 revenue $5M."},
        {"doc_id": "d1", "chunk_id": "c2", "text": "Q2 revenue $5.4M."},
        {"doc_id": "d2", "chunk_id": "c3", "text": "Q3 revenue $6.1M."},
    ]
    return ctx


def _llm_body(section, narrative, citations, confidence=0.85):
    return json.dumps(
        {
            "section": section,
            "narrative": narrative,
            "evidence": [
                {"doc_id": d, "chunk_id": c} for d, c in citations
            ],
            "confidence": confidence,
            "entity_refs": ["Acme Corp"],
        }
    )


def _three_responses():
    return [
        _llm_body("overview", "Revenue grew QoQ.", [("d1", "c1"), ("d2", "c3")]),
        _llm_body("trends", "QoQ growth 8% then 13%.", [("d1", "c2"), ("d2", "c3")]),
        _llm_body("risks", "Concentration risk.", [("d1", "c1")]),
    ]


def test_build_produces_one_item_per_section_with_evidence():
    llm = MagicMock()
    llm.complete.side_effect = _three_responses()
    trace = MagicMock()
    builder = SMEDossierBuilder(ctx=_ctx(), llm=llm, trace=trace)

    items = builder.build(
        subscription_id="sub_test",
        profile_id="prof_fin",
        adapter=_adapter(),
        version=3,
    )

    assert len(items) == 3
    assert all(isinstance(i, ArtifactItem) for i in items)
    assert {i.metadata["section"] for i in items} == {"overview", "trends", "risks"}
    for item in items:
        assert item.artifact_type == "dossier"
        assert item.subscription_id == "sub_test"
        assert item.profile_id == "prof_fin"
        assert item.text  # narrative surfaces via .text per ERRATA §3
        assert item.evidence, "every section carries ≥1 evidence ref"
        assert 0.0 <= item.confidence <= 1.0
    assert llm.complete.call_count == 3


def test_build_propagates_adapter_version_to_trace_tag():
    llm = MagicMock()
    llm.complete.side_effect = _three_responses()
    trace = MagicMock()
    builder = SMEDossierBuilder(ctx=_ctx(), llm=llm, trace=trace)

    builder.build(
        subscription_id="s",
        profile_id="p",
        adapter=_adapter(),
        version=1,
    )

    for call in llm.complete.call_args_list:
        kw = call.kwargs
        assert kw["adapter_version"] == "1.2.0"
        assert kw["trace_tag"].startswith("dossier:s:p:")


def test_build_skips_sections_with_invalid_json():
    llm = MagicMock()
    llm.complete.side_effect = [
        "not-json",
        _llm_body("trends", "QoQ growth.", [("d1", "c2")]),
        _llm_body("risks", "Concentration.", [("d1", "c1")]),
    ]
    trace = MagicMock()
    builder = SMEDossierBuilder(ctx=_ctx(), llm=llm, trace=trace)

    items = builder.build(
        subscription_id="s", profile_id="p", adapter=_adapter(), version=1
    )
    # Parse failure → section dropped, not persisted
    assert {i.metadata["section"] for i in items} == {"trends", "risks"}
    # Trace records the parse failure event
    events = [c.args[0] for c in trace.append.call_args_list]
    parse_failures = [e for e in events if e.get("stage") == "builder_parse_failure"]
    assert len(parse_failures) == 1
    assert parse_failures[0]["section"] == "overview"


def test_build_skips_sections_with_no_evidence():
    llm = MagicMock()
    llm.complete.side_effect = [
        json.dumps(
            {
                "section": "overview",
                "narrative": "text",
                "evidence": [],
                "confidence": 0.7,
            }
        ),
        _llm_body("trends", "QoQ growth.", [("d1", "c2")]),
        _llm_body("risks", "Concentration.", [("d1", "c1")]),
    ]
    trace = MagicMock()
    builder = SMEDossierBuilder(ctx=_ctx(), llm=llm, trace=trace)

    items = builder.build(
        subscription_id="s", profile_id="p", adapter=_adapter(), version=1
    )
    assert {i.metadata["section"] for i in items} == {"trends", "risks"}
    no_evidence_events = [
        c.args[0]
        for c in trace.append.call_args_list
        if c.args[0].get("stage") == "builder_no_evidence"
    ]
    assert len(no_evidence_events) == 1


def test_build_logs_llm_error_and_continues():
    llm = MagicMock()
    llm.complete.side_effect = [
        RuntimeError("gateway unreachable"),
        _llm_body("trends", "QoQ growth.", [("d1", "c2")]),
        _llm_body("risks", "Concentration.", [("d1", "c1")]),
    ]
    trace = MagicMock()
    builder = SMEDossierBuilder(ctx=_ctx(), llm=llm, trace=trace)

    items = builder.build(
        subscription_id="s", profile_id="p", adapter=_adapter(), version=1
    )
    assert {i.metadata["section"] for i in items} == {"trends", "risks"}
    llm_errors = [
        c.args[0]
        for c in trace.append.call_args_list
        if c.args[0].get("stage") == "builder_llm_error"
    ]
    assert len(llm_errors) == 1
    assert llm_errors[0]["section"] == "overview"
    assert "unreachable" in llm_errors[0]["error"]


def test_build_uses_narrative_as_text_per_errata_section_3():
    """ERRATA §3 mandates every artifact type expose .text carrying the
    substantively-checkable string. For dossier sections that string is the
    narrative, so SMEVerifier's evidence-validity + contradiction checks run
    over the right content."""
    llm = MagicMock()
    narrative = "Revenue grew QoQ from $5M to $6.1M."
    llm.complete.side_effect = [
        _llm_body("overview", narrative, [("d1", "c1"), ("d2", "c3")]),
        _llm_body("trends", "QoQ growth 8% then 13%.", [("d1", "c2")]),
        _llm_body("risks", "Concentration risk.", [("d1", "c1")]),
    ]
    trace = MagicMock()
    builder = SMEDossierBuilder(ctx=_ctx(), llm=llm, trace=trace)

    items = builder.build(
        subscription_id="s", profile_id="p", adapter=_adapter(), version=1
    )
    overview = next(i for i in items if i.metadata["section"] == "overview")
    assert overview.text == narrative


def test_build_passes_grounding_rules_into_system_prompt():
    llm = MagicMock()
    llm.complete.side_effect = _three_responses()
    trace = MagicMock()
    adapter = _adapter()
    builder = SMEDossierBuilder(ctx=_ctx(), llm=llm, trace=trace)

    builder.build(
        subscription_id="s", profile_id="p", adapter=adapter, version=1
    )

    first_call = llm.complete.call_args_list[0].kwargs
    sys_prompt = first_call["system_prompt"]
    assert "senior financial analyst" in sys_prompt
    # Persona voice threaded
    assert "direct, quantitative" in sys_prompt
