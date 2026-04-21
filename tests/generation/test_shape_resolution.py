"""Lean sanity tests for resolve_response_shape + rich prompt builders."""
from types import SimpleNamespace

import pytest

from src.generation.pack_summary import PackSummary
from src.generation.prompts import (
    AnalyzePromptInputs,
    DiagnosePromptInputs,
    PersonaBundle,
    RecommendPromptInputs,
    ResponseShape,
    TASK_FORMATS,
    build_analyze_rich_prompt,
    build_diagnose_rich_prompt,
    build_honest_compact_prompt,
    build_recommend_rich_prompt,
    build_system_prompt,
    persona_bundle_from_adapter,
    resolve_response_shape,
)
from src.serving.model_router import FormatHint


def _pack(
    has_sme_artifacts: bool = True,
    total_chunks: int = 12,
    distinct_docs: int = 3,
) -> PackSummary:
    return PackSummary(
        total_chunks=total_chunks,
        distinct_docs=distinct_docs,
        has_sme_artifacts=has_sme_artifacts,
    )


# ---------------------------------------------------------------------------
# Shape resolution invariants
# ---------------------------------------------------------------------------


def test_phase4_task_formats_added():
    for intent in ("analyze", "diagnose", "recommend"):
        assert intent in TASK_FORMATS
        assert "## Executive summary" in TASK_FORMATS[intent] or "## Symptom" in TASK_FORMATS[intent] or "## Observations" in TASK_FORMATS[intent] or "## Recommendations" in TASK_FORMATS[intent]


def test_compact_override_always_wins():
    assert (
        resolve_response_shape(
            intent="analyze",
            format_hint=FormatHint.COMPACT,
            pack=_pack(),
            enable_rich_mode=True,
        )
        is ResponseShape.COMPACT
    )


def test_rich_mode_off_forces_compact():
    assert (
        resolve_response_shape(
            intent="analyze",
            format_hint=FormatHint.AUTO,
            pack=_pack(),
            enable_rich_mode=False,
        )
        is ResponseShape.COMPACT
    )


def test_trivial_intent_gets_compact_with_rich_on():
    assert (
        resolve_response_shape(
            intent="lookup",
            format_hint=FormatHint.AUTO,
            pack=_pack(),
            enable_rich_mode=True,
        )
        is ResponseShape.COMPACT
    )


def test_analytical_rich_when_artifacts_present():
    assert (
        resolve_response_shape(
            intent="analyze",
            format_hint=FormatHint.AUTO,
            pack=_pack(has_sme_artifacts=True),
            enable_rich_mode=True,
        )
        is ResponseShape.RICH
    )


def test_analytical_honest_compact_when_pack_thin():
    assert (
        resolve_response_shape(
            intent="analyze",
            format_hint=FormatHint.AUTO,
            pack=_pack(has_sme_artifacts=False, total_chunks=2, distinct_docs=1),
            enable_rich_mode=True,
        )
        is ResponseShape.HONEST_COMPACT
    )


def test_borderline_rich_when_sme_backed():
    assert (
        resolve_response_shape(
            intent="compare",
            format_hint=FormatHint.AUTO,
            pack=_pack(has_sme_artifacts=True),
            enable_rich_mode=True,
        )
        is ResponseShape.RICH
    )


def test_borderline_honest_compact_without_sme():
    assert (
        resolve_response_shape(
            intent="compare",
            format_hint=FormatHint.AUTO,
            pack=_pack(has_sme_artifacts=False),
            enable_rich_mode=True,
        )
        is ResponseShape.HONEST_COMPACT
    )


# ---------------------------------------------------------------------------
# Rich prompt builders
# ---------------------------------------------------------------------------


def test_analyze_rich_prompt_has_all_sections_and_persona():
    p = build_analyze_rich_prompt(
        AnalyzePromptInputs(
            query_text="Analyze Q3 revenue trends.",
            persona_role="senior financial analyst",
            persona_voice="direct, quantitative",
            grounding_rules=("Cite every claim.",),
            pack_tokens=3000,
            output_cap_tokens=1200,
            evidence_items=[
                {"doc_id": "q3", "chunk_id": "c1", "text": "Q3 revenue was $5.3M."}
            ],
            insight_refs=[{"type": "trend", "narrative": "QoQ growth 14%."}],
            domain="finance",
        )
    )
    assert "senior financial analyst" in p
    assert "## Executive summary" in p
    assert "## Analysis" in p
    assert "## Patterns" in p
    assert "## Assumptions & caveats" in p
    assert "## Evidence" in p
    assert "q3:c1" in p
    assert "QoQ growth 14%" in p
    assert "1200" in p


def test_diagnose_rich_prompt_has_symptom_and_causes():
    p = build_diagnose_rich_prompt(
        DiagnosePromptInputs(
            query_text="Why is backup failing?",
            persona_role="support engineer",
            persona_voice="methodical",
            grounding_rules=("Rank causes by evidence.",),
            pack_tokens=2000,
            output_cap_tokens=1500,
            evidence_items=[
                {"doc_id": "incident_1", "chunk_id": "c3", "text": "disk full"}
            ],
            diagnostic_hits=[
                {
                    "symptom": "backup job exits 1",
                    "doc_id": "incident_1",
                    "chunk_id": "c3",
                    "rank": 1,
                }
            ],
            domain="it_support",
        )
    )
    assert "support engineer" in p
    assert "## Symptom" in p
    assert "## Causes (ranked)" in p
    assert "## Executive summary" in p


def test_recommend_rich_prompt_injects_bank_entries_and_forbids_speculation():
    p = build_recommend_rich_prompt(
        RecommendPromptInputs(
            query_text="How to improve margins?",
            persona_role="CFO-advisor",
            persona_voice="decisive",
            grounding_rules=("Cite a bank entry.",),
            pack_tokens=3000,
            output_cap_tokens=1000,
            evidence_items=[
                {"doc_id": "q3_pl", "chunk_id": "c2", "text": "COGS 62%"}
            ],
            bank_entries=[
                {
                    "recommendation": "Renegotiate top-3 vendor contracts",
                    "rationale": "COGS concentration",
                    "evidence": ["q3_pl:c2"],
                    "estimated_impact": "margin +1.2–1.8pp",
                    "assumptions": ["current volume holds"],
                    "confidence": 0.74,
                }
            ],
            domain="finance",
        )
    )
    assert "Renegotiate top-3 vendor contracts" in p
    assert "margin +1.2" in p
    assert "## Recommendations" in p
    assert "Recommendation Bank" in p


def test_honest_compact_includes_caveat():
    p = build_honest_compact_prompt(
        query_text="Analyze trends",
        pack_summary=_pack(has_sme_artifacts=False, total_chunks=1, distinct_docs=1),
    )
    assert "necessarily compact" in p
    assert "Analyze trends" in p


# ---------------------------------------------------------------------------
# Persona bundle
# ---------------------------------------------------------------------------


def _adapter(role="senior financial analyst", body="ANALYZE BODY"):
    return SimpleNamespace(
        persona=SimpleNamespace(
            role=role, voice="direct", grounding_rules=["cite every claim"]
        ),
        response_persona_prompts=SimpleNamespace(
            analyze=body, diagnose="", recommend=""
        ),
        version="1.2.0",
        content_hash="deadbeef",
    )


def test_persona_bundle_carries_adapter_fields():
    b = persona_bundle_from_adapter(_adapter(), intent="analyze")
    assert isinstance(b, PersonaBundle)
    assert b.role == "senior financial analyst"
    assert b.intent_template_body == "ANALYZE BODY"
    assert b.adapter_version == "1.2.0"
    assert b.adapter_content_hash == "deadbeef"


def test_persona_bundle_rejects_unknown_intent():
    with pytest.raises(ValueError):
        persona_bundle_from_adapter(_adapter(), intent="telepathy")


# ---------------------------------------------------------------------------
# build_system_prompt rich path
# ---------------------------------------------------------------------------


def test_build_system_prompt_compact_is_legacy():
    s = build_system_prompt(profile_domain="finance")
    assert "DocWain" in s
    assert "You are acting as" not in s


def test_build_system_prompt_rich_injects_persona():
    s = build_system_prompt(
        profile_domain="finance",
        shape="rich",
        persona={
            "role": "CFO-advisor",
            "voice": "decisive",
            "grounding_rules": ["cite every claim"],
        },
    )
    assert "You are acting as a CFO-advisor" in s
    assert "Voice: decisive" in s
    # DocWain identity still present
    assert "DocWain" in s
