"""Phase 4 — extended intent classifier.

Lean sanity coverage: one test per invariant that the Phase 4 plan cares
about (new labels, compact override, URL detection, safe fallback).
"""
from unittest.mock import AsyncMock, patch

import pytest

from src.serving.model_router import (
    ClassifiedQuery,
    FormatHint,
    VALID_INTENTS,
    classify_query,
)


def test_classified_query_is_frozen_dataclass():
    cq = ClassifiedQuery(
        query_text="hello",
        intent="greeting",
        format_hint=FormatHint.AUTO,
        entities=[],
        urls=[],
    )
    with pytest.raises(Exception):
        cq.intent = "lookup"  # type: ignore[misc]


def test_valid_intents_include_phase4_additions():
    for label in ("analyze", "diagnose", "recommend"):
        assert label in VALID_INTENTS


@pytest.mark.asyncio
async def test_classifier_recognises_recommend_intent():
    with patch(
        "src.serving.model_router._call_classifier_llm",
        new=AsyncMock(
            return_value='{"intent": "recommend", "format_hint": "auto", '
            '"entities": [], "urls": []}'
        ),
    ):
        result = await classify_query("What should we do to improve margins?")
    assert result.intent == "recommend"
    assert result.format_hint is FormatHint.AUTO
    assert result.query_text == "What should we do to improve margins?"


@pytest.mark.asyncio
async def test_compact_override_from_text_beats_auto_hint():
    with patch(
        "src.serving.model_router._call_classifier_llm",
        new=AsyncMock(
            return_value='{"intent": "analyze", "format_hint": "auto", '
            '"entities": [], "urls": []}'
        ),
    ):
        result = await classify_query(
            "Analyze Q3 trends. tl;dr please, one paragraph."
        )
    assert result.format_hint is FormatHint.COMPACT


@pytest.mark.asyncio
async def test_urls_detected_deterministically():
    with patch(
        "src.serving.model_router._call_classifier_llm",
        new=AsyncMock(
            return_value='{"intent": "analyze", "format_hint": "auto", '
            '"entities": [], "urls": []}'
        ),
    ):
        result = await classify_query(
            "Analyze this report: https://example.com/q3-report.pdf"
        )
    assert result.urls == ["https://example.com/q3-report.pdf"]


@pytest.mark.asyncio
async def test_unparseable_llm_output_falls_back_to_overview():
    with patch(
        "src.serving.model_router._call_classifier_llm",
        new=AsyncMock(return_value="not-json ~~~"),
    ):
        result = await classify_query("something opaque")
    assert result.intent == "overview"
    assert result.format_hint is FormatHint.AUTO


@pytest.mark.asyncio
async def test_unknown_label_falls_back_to_overview():
    with patch(
        "src.serving.model_router._call_classifier_llm",
        new=AsyncMock(
            return_value='{"intent": "telepathy", "format_hint": "auto", '
            '"entities": [], "urls": []}'
        ),
    ):
        result = await classify_query("anything")
    assert result.intent == "overview"
