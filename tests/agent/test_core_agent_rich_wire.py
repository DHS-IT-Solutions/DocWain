"""Lean tests for CoreAgent rich-mode wiring seam."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from src.agent.core_agent import CoreAgent
from src.generation.pack_summary import PackSummary
from src.retrieval.types import PackedItem
from src.serving.model_router import ClassifiedQuery, FormatHint


def _stub_adapter() -> SimpleNamespace:
    return SimpleNamespace(
        persona=SimpleNamespace(
            role="senior financial analyst",
            voice="direct, quantitative",
            grounding_rules=["cite every claim"],
        ),
        response_persona_prompts=SimpleNamespace(
            analyze="ANALYZE BODY", diagnose="", recommend=""
        ),
        output_caps=SimpleNamespace(
            analyze=1200, diagnose=1500, recommend=1000, investigate=2000
        ),
        retrieval_caps=SimpleNamespace(
            max_pack_tokens={"analyze": 6000, "diagnose": 5000, "recommend": 4500}
        ),
        version="1.2.0",
        content_hash="deadbeef",
    )


def _bare_agent() -> CoreAgent:
    """Construct a CoreAgent with None dependencies — rich-mode wiring
    exercises the formatting path only, not retrieval / reasoning."""
    return CoreAgent(
        llm_gateway=None,
        qdrant_client=None,
        embedder=None,
        mongodb=None,
    )


def _rich_pack(has_sme_artifacts: bool = True) -> PackSummary:
    if not has_sme_artifacts:
        return PackSummary(
            total_chunks=1, distinct_docs=1, has_sme_artifacts=False
        )
    evidence = (
        PackedItem(
            text="Q3 revenue $5.3M",
            provenance=(("q3_report", "c1"),),
            layer="a",
            confidence=0.9,
            rerank_score=0.8,
            sme_backed=True,
            metadata={"artifact_type": "dossier"},
        ),
    )
    return PackSummary(
        total_chunks=10,
        distinct_docs=3,
        has_sme_artifacts=True,
        evidence_items=evidence,
    )


@pytest.mark.asyncio
async def test_rich_mode_off_uses_compact_path():
    agent = _bare_agent()
    classified = ClassifiedQuery(
        query_text="Analyze Q3 trends.",
        intent="analyze",
        format_hint=FormatHint.AUTO,
        entities=[],
        urls=[],
    )
    with patch(
        "src.agent.core_agent._load_adapter",
        new=AsyncMock(return_value=_stub_adapter()),
    ), patch(
        "src.agent.core_agent._is_rich_mode_enabled",
        new=AsyncMock(return_value=False),
    ), patch.object(
        CoreAgent, "_resolve_profile_domain",
        new=AsyncMock(return_value="finance"),
    ):
        prompt = await agent._build_prompt_for_test(
            classified=classified,
            pack_summary=_rich_pack(),
            subscription_id="s",
            profile_id="p",
        )
    # Compact stub returns "COMPACT:<query_text>"
    assert prompt.startswith("COMPACT:")


@pytest.mark.asyncio
async def test_rich_mode_on_builds_analyze_rich_prompt():
    agent = _bare_agent()
    classified = ClassifiedQuery(
        query_text="Analyze Q3 revenue trends across quarters.",
        intent="analyze",
        format_hint=FormatHint.AUTO,
        entities=[],
        urls=[],
    )
    with patch(
        "src.agent.core_agent._load_adapter",
        new=AsyncMock(return_value=_stub_adapter()),
    ), patch(
        "src.agent.core_agent._is_rich_mode_enabled",
        new=AsyncMock(return_value=True),
    ), patch.object(
        CoreAgent, "_resolve_profile_domain",
        new=AsyncMock(return_value="finance"),
    ):
        prompt = await agent._build_prompt_for_test(
            classified=classified,
            pack_summary=_rich_pack(),
            subscription_id="s",
            profile_id="p",
        )
    assert "## Executive summary" in prompt
    assert "## Analysis" in prompt
    assert "senior financial analyst" in prompt
    assert "Analyze Q3 revenue trends across quarters." in prompt


@pytest.mark.asyncio
async def test_compact_override_bypasses_rich_even_with_flag_on():
    agent = _bare_agent()
    classified = ClassifiedQuery(
        query_text="Keep it short please.",
        intent="analyze",
        format_hint=FormatHint.COMPACT,
        entities=[],
        urls=[],
    )
    with patch(
        "src.agent.core_agent._load_adapter",
        new=AsyncMock(return_value=_stub_adapter()),
    ), patch(
        "src.agent.core_agent._is_rich_mode_enabled",
        new=AsyncMock(return_value=True),
    ):
        prompt = await agent._build_prompt_for_test(
            classified=classified,
            pack_summary=_rich_pack(),
            subscription_id="s",
            profile_id="p",
        )
    assert prompt.startswith("COMPACT:")


@pytest.mark.asyncio
async def test_honest_compact_taken_for_thin_pack():
    agent = _bare_agent()
    classified = ClassifiedQuery(
        query_text="Analyze with thin evidence.",
        intent="analyze",
        format_hint=FormatHint.AUTO,
        entities=[],
        urls=[],
    )
    with patch(
        "src.agent.core_agent._load_adapter",
        new=AsyncMock(return_value=_stub_adapter()),
    ), patch(
        "src.agent.core_agent._is_rich_mode_enabled",
        new=AsyncMock(return_value=True),
    ), patch.object(
        CoreAgent, "_resolve_profile_domain",
        new=AsyncMock(return_value="finance"),
    ):
        prompt = await agent._build_prompt_for_test(
            classified=classified,
            pack_summary=_rich_pack(has_sme_artifacts=False),
            subscription_id="s",
            profile_id="p",
        )
    assert "necessarily compact" in prompt


@pytest.mark.asyncio
async def test_investigate_routes_through_analyze_template():
    agent = _bare_agent()
    classified = ClassifiedQuery(
        query_text="Investigate patterns across Q3.",
        intent="investigate",
        format_hint=FormatHint.AUTO,
        entities=[],
        urls=[],
    )
    with patch(
        "src.agent.core_agent._load_adapter",
        new=AsyncMock(return_value=_stub_adapter()),
    ), patch(
        "src.agent.core_agent._is_rich_mode_enabled",
        new=AsyncMock(return_value=True),
    ), patch.object(
        CoreAgent, "_resolve_profile_domain",
        new=AsyncMock(return_value="finance"),
    ):
        prompt = await agent._build_prompt_for_test(
            classified=classified,
            pack_summary=_rich_pack(),
            subscription_id="s",
            profile_id="p",
        )
    # Analyze template sections present, confirming investigate is mapped.
    assert "## Analysis" in prompt
    assert "## Patterns" in prompt


@pytest.mark.asyncio
async def test_recommend_grounding_post_pass_drops_unverified_claims():
    agent = _bare_agent()
    classified = ClassifiedQuery(
        query_text="What should we do?",
        intent="recommend",
        format_hint=FormatHint.AUTO,
        entities=[],
        urls=[],
    )
    pack_summary = PackSummary(
        total_chunks=5,
        distinct_docs=2,
        has_sme_artifacts=True,
        bank_entries=(
            {
                "recommendation": "Renegotiate top-3 vendor contracts",
                "evidence": ["q3_pl:c2"],
            },
        ),
    )
    from src.generation.prompts import ResponseShape

    bad_response = (
        "## Executive summary\nS.\n\n"
        "## Recommendations\n"
        "1. Renegotiate top-3 vendor contracts [q3_pl:c2].\n"
        "2. Launch a moonbase colony.\n\n"
        "## Evidence\n- q3_pl:c2\n"
    )
    result = await agent._apply_recommend_grounding(
        response_text=bad_response,
        classified=classified,
        shape=ResponseShape.RICH,
        pack_summary=pack_summary,
    )
    assert "Launch a moonbase" not in result
    assert "Renegotiate" in result
    assert "could not be verified" in result


@pytest.mark.asyncio
async def test_recommend_grounding_noop_for_non_recommend_intent():
    agent = _bare_agent()
    classified = ClassifiedQuery(
        query_text="Analyze trends.",
        intent="analyze",
        format_hint=FormatHint.AUTO,
        entities=[],
        urls=[],
    )
    from src.generation.prompts import ResponseShape

    original = "## Executive summary\nS\n\n## Analysis\nfoo\n"
    result = await agent._apply_recommend_grounding(
        response_text=original,
        classified=classified,
        shape=ResponseShape.RICH,
        pack_summary=_rich_pack(),
    )
    assert result == original
