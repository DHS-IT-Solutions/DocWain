import json

from src.docwain.prompts.chart_generation import (
    CHART_GENERATION_SYSTEM_PROMPT,
    extract_viz_block,
    should_emit_chart,
)


def test_prompt_is_non_empty():
    assert len(CHART_GENERATION_SYSTEM_PROMPT) > 100
    assert "DOCWAIN_VIZ" in CHART_GENERATION_SYSTEM_PROMPT


def test_should_emit_chart_for_comparison_query():
    assert should_emit_chart("Compare revenue between Q1 and Q2") is True
    assert should_emit_chart("show me a chart of monthly expenses") is True
    assert should_emit_chart("plot the trend over time") is True


def test_should_not_emit_chart_for_factual_query():
    assert should_emit_chart("What is the candidate's name?") is False
    assert should_emit_chart("List the vendors") is False


def test_extract_viz_block_parses_html_comment():
    response = """
Some text before.
<!--DOCWAIN_VIZ
{"chart_type": "bar", "title": "Expenses", "labels": ["Jan", "Feb"], "values": [100, 200], "unit": "USD"}
-->
Some text after.
"""
    viz = extract_viz_block(response)
    assert viz is not None
    assert viz["chart_type"] == "bar"
    assert viz["labels"] == ["Jan", "Feb"]
    assert viz["values"] == [100, 200]


def test_extract_viz_block_returns_none_when_absent():
    assert extract_viz_block("no viz here") is None


def test_extract_viz_block_tolerates_malformed_json():
    response = "<!--DOCWAIN_VIZ\n{malformed json}\n-->"
    assert extract_viz_block(response) is None
