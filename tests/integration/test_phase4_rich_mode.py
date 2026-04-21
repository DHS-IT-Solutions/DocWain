"""Phase 4 integration smoke — classifier → shape → adapter → prompt.

One end-to-end test per major path. Covers the interaction between the
extended classifier, the resolver, and the rich-mode prompt builders via
the CoreAgent test seam.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from src.agent.core_agent import CoreAgent
from src.generation.pack_summary import PackSummary
from src.retrieval.types import PackedItem
from src.serving.model_router import ClassifiedQuery, FormatHint


def _fake_adapter() -> SimpleNamespace:
    return SimpleNamespace(
        persona=SimpleNamespace(
            role="senior SME",
            voice="clear, hedged",
            grounding_rules=["cite every claim"],
        ),
        response_persona_prompts=SimpleNamespace(
            analyze="ANALYZE",
            diagnose="DIAGNOSE",
            recommend="RECOMMEND",
        ),
        output_caps=SimpleNamespace(
            analyze=1200, diagnose=1500, recommend=1000, investigate=2000
        ),
        retrieval_caps=SimpleNamespace(
            max_pack_tokens={
                "analyze": 6000,
                "diagnose": 5000,
                "recommend": 4500,
            }
        ),
        version="1.0.0",
        content_hash="abc",
    )


def _fake_pack(
    has_sme_artifacts: bool = True,
    total_chunks: int = 10,
    distinct_docs: int = 3,
) -> PackSummary:
    if not has_sme_artifacts:
        return PackSummary(
            total_chunks=total_chunks,
            distinct_docs=distinct_docs,
            has_sme_artifacts=False,
        )
    evidence = (
        PackedItem(
            text="Sample evidence chunk.",
            provenance=(("d1", "c1"),),
            layer="a",
            confidence=0.9,
            rerank_score=0.8,
            sme_backed=True,
            metadata={"artifact_type": "dossier"},
        ),
    )
    bank_entries = (
        {
            "recommendation": "Sample recommendation",
            "evidence": ["d1:c1"],
        },
    )
    return PackSummary(
        total_chunks=total_chunks,
        distinct_docs=distinct_docs,
        has_sme_artifacts=True,
        evidence_items=evidence,
        bank_entries=bank_entries,
    )


def _bare_agent() -> CoreAgent:
    return CoreAgent(
        llm_gateway=None,
        qdrant_client=None,
        embedder=None,
        mongodb=None,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("intent", ["analyze", "diagnose", "recommend"])
async def test_rich_mode_on_produces_rich_skeleton_for_all_new_intents(intent):
    agent = _bare_agent()
    classified = ClassifiedQuery(
        query_text=f"Sample {intent} query.",
        intent=intent,
        format_hint=FormatHint.AUTO,
        entities=[],
        urls=[],
    )
    with patch(
        "src.agent.core_agent._is_rich_mode_enabled",
        new=AsyncMock(return_value=True),
    ), patch(
        "src.agent.core_agent._load_adapter",
        new=AsyncMock(return_value=_fake_adapter()),
    ), patch.object(
        CoreAgent,
        "_resolve_profile_domain",
        new=AsyncMock(return_value="generic"),
    ):
        prompt = await agent._build_prompt_for_test(
            classified=classified,
            pack_summary=_fake_pack(),
            subscription_id="s",
            profile_id="p",
        )
    assert "## Executive summary" in prompt
    assert "## Assumptions & caveats" in prompt
    assert "senior SME" in prompt


@pytest.mark.asyncio
async def test_compact_override_bypasses_rich_for_every_new_intent():
    agent = _bare_agent()
    for intent in ("analyze", "diagnose", "recommend"):
        classified = ClassifiedQuery(
            query_text=f"Compact {intent} please.",
            intent=intent,
            format_hint=FormatHint.COMPACT,
            entities=[],
            urls=[],
        )
        with patch(
            "src.agent.core_agent._is_rich_mode_enabled",
            new=AsyncMock(return_value=True),
        ), patch(
            "src.agent.core_agent._load_adapter",
            new=AsyncMock(return_value=_fake_adapter()),
        ):
            prompt = await agent._build_prompt_for_test(
                classified=classified,
                pack_summary=_fake_pack(),
                subscription_id="s",
                profile_id="p",
            )
        assert "## Executive summary" not in prompt


@pytest.mark.asyncio
async def test_honest_compact_used_for_thin_pack():
    agent = _bare_agent()
    classified = ClassifiedQuery(
        query_text="Analyze trends with scant evidence.",
        intent="analyze",
        format_hint=FormatHint.AUTO,
        entities=[],
        urls=[],
    )
    with patch(
        "src.agent.core_agent._is_rich_mode_enabled",
        new=AsyncMock(return_value=True),
    ), patch(
        "src.agent.core_agent._load_adapter",
        new=AsyncMock(return_value=_fake_adapter()),
    ), patch.object(
        CoreAgent,
        "_resolve_profile_domain",
        new=AsyncMock(return_value="generic"),
    ):
        prompt = await agent._build_prompt_for_test(
            classified=classified,
            pack_summary=_fake_pack(
                has_sme_artifacts=False, total_chunks=1, distinct_docs=1
            ),
            subscription_id="s",
            profile_id="p",
        )
    assert "necessarily compact" in prompt
