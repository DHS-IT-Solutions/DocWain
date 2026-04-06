"""Tests for the Analytical Reasoner module."""

import json
import pytest

from src.generation.analytical_reasoner import (
    AnalyticalResponse,
    DOMAIN_PROMPTS,
    build_analytical_prompt,
    parse_analytical_response,
    reason_analytical,
    reason_analytical_stream,
)


# ---------------------------------------------------------------------------
# Prompt construction tests
# ---------------------------------------------------------------------------


def test_build_analytical_prompt_includes_query():
    prompt = build_analytical_prompt(
        "What areas need improvement?",
        [{"text": "Revenue declined 10%"}],
    )
    assert "What areas need improvement?" in prompt
    assert "Revenue declined 10%" in prompt


def test_build_prompt_includes_enrichment():
    enrichment = {
        "temporal": {"date_range": {"earliest": "2024-01-01"}},
        "numerical": {"trends": [{"metric": "revenue", "direction": "decreasing"}]},
    }
    prompt = build_analytical_prompt("Analyze", [{"text": "data"}], enrichment=enrichment)
    assert "2024-01-01" in prompt
    assert "decreasing" in prompt


def test_build_prompt_uses_domain():
    prompt = build_analytical_prompt("Analyze", [{"text": "data"}], domain="financial")
    assert "financial analyst" in prompt.lower()


def test_build_prompt_uses_healthcare_domain():
    prompt = build_analytical_prompt("Analyze", [{"text": "data"}], domain="healthcare")
    assert "clinical analyst" in prompt.lower()


def test_build_prompt_defaults_to_general_domain():
    prompt = build_analytical_prompt("Analyze", [{"text": "data"}])
    assert "document analyst" in prompt.lower()


def test_enrichment_labeled_separately():
    prompt = build_analytical_prompt(
        "Q",
        [{"text": "data"}],
        enrichment={"temporal": {"gaps": ["Q3 missing"]}},
    )
    assert "COMPUTED CONTEXT" in prompt
    assert "do NOT cite directly" in prompt


def test_build_prompt_no_enrichment():
    prompt = build_analytical_prompt("Q", [{"text": "data"}], enrichment=None)
    assert "COMPUTED CONTEXT" in prompt
    assert "None available" in prompt


def test_build_prompt_formats_multiple_sources():
    chunks = [
        {"text": "First chunk", "source_name": "Report A", "page": 1},
        {"text": "Second chunk", "source_name": "Report B", "page": 5},
    ]
    prompt = build_analytical_prompt("Q", chunks)
    assert "Report A" in prompt
    assert "Report B" in prompt
    assert "First chunk" in prompt
    assert "Second chunk" in prompt


# ---------------------------------------------------------------------------
# Parsing tests
# ---------------------------------------------------------------------------


def test_parse_well_formed_response():
    raw = """## Key Data Points
- Revenue was $1.5M (Source: Q2 Report page 2)
- Costs rose 15% (Source: Q2 Report page 4)

## Analysis
Revenue shows steady growth of 20% QoQ while costs are rising faster.

## Risks
- Margin compression from rising COGS — Severity: HIGH
- Customer concentration risk — Severity: MEDIUM

## Recommendations
- Renegotiate supplier contracts — Confidence: MEDIUM
- Diversify customer base — Confidence: HIGH"""

    result = parse_analytical_response(raw)
    assert len(result.key_data_points) == 2
    assert result.key_data_points[0]["fact"] == "Revenue was $1.5M"
    assert "Q2 Report" in result.key_data_points[0]["source"]
    assert "steady growth" in result.analysis
    assert len(result.risks) == 2
    assert result.risks[0]["severity"] == "HIGH"
    assert len(result.recommendations) == 2
    assert result.recommendations[1]["confidence"] == "HIGH"
    assert result.raw_answer == raw


def test_parse_unparseable_response():
    raw = "Just a plain text response without sections."
    result = parse_analytical_response(raw)
    assert result.raw_answer == raw
    assert result.key_data_points == []
    assert result.analysis == ""
    assert result.risks == []
    assert result.recommendations == []
    assert result.confidence_level == "LOW"
    assert result.data_sufficiency == "insufficient"


# ---------------------------------------------------------------------------
# reason_analytical tests
# ---------------------------------------------------------------------------


def test_reason_analytical_returns_structured():
    def mock_llm(prompt):
        return """## Key Data Points
- Revenue was $1.5M (Source: Q2 Report page 2)

## Analysis
Revenue shows steady growth of 20% QoQ.

## Risks
- Margin compression from rising COGS — Severity: HIGH

## Recommendations
- Renegotiate supplier contracts — Confidence: MEDIUM"""

    result = reason_analytical(
        "What needs improvement?",
        [{"text": "Revenue $1.5M"}],
        llm_fn=mock_llm,
    )
    assert isinstance(result, AnalyticalResponse)
    assert len(result.key_data_points) >= 1
    assert len(result.risks) >= 1
    assert len(result.recommendations) >= 1
    assert result.raw_answer != ""


def test_reason_analytical_handles_unparseable():
    def mock_llm(prompt):
        return "Just a plain text response without sections."

    result = reason_analytical("Analyze", [{"text": "data"}], llm_fn=mock_llm)
    assert result.raw_answer == "Just a plain text response without sections."
    assert result.key_data_points == []


def test_reason_analytical_passes_domain_to_prompt():
    captured = {}

    def mock_llm(prompt):
        captured["prompt"] = prompt
        return "## Key Data Points\n- None\n\n## Analysis\nN/A\n\n## Risks\n- None — Severity: LOW\n\n## Recommendations\n- None — Confidence: LOW"

    reason_analytical("Q", [{"text": "data"}], domain="legal", llm_fn=mock_llm)
    assert "legal analyst" in captured["prompt"].lower()


def test_reason_analytical_passes_enrichment():
    captured = {}

    def mock_llm(prompt):
        captured["prompt"] = prompt
        return "## Key Data Points\n- X (Source: Y)\n\n## Analysis\nOK\n\n## Risks\n- R — Severity: LOW\n\n## Recommendations\n- A — Confidence: LOW"

    enrichment = {"temporal": {"period": "2024-Q1"}}
    reason_analytical("Q", [{"text": "data"}], enrichment=enrichment, llm_fn=mock_llm)
    assert "2024-Q1" in captured["prompt"]


# ---------------------------------------------------------------------------
# Streaming tests
# ---------------------------------------------------------------------------


def test_reason_analytical_stream_yields_chunks():
    def mock_llm(prompt):
        return ["chunk1", "chunk2", "chunk3"]

    chunks = list(reason_analytical_stream("Q", [{"text": "data"}], llm_fn=mock_llm))
    # Should have the 3 content chunks plus the metadata chunk
    assert len(chunks) == 4
    assert chunks[0] == "chunk1"
    assert "---ANALYTICAL_META---" in chunks[-1]

    # Metadata should be valid JSON
    meta_json = chunks[-1].split("---ANALYTICAL_META---\n")[1]
    meta = json.loads(meta_json)
    assert "raw_answer" in meta


def test_reason_analytical_stream_string_fallback():
    """When llm_fn returns a plain string, streaming still works."""
    def mock_llm(prompt):
        return "single response"

    chunks = list(reason_analytical_stream("Q", [{"text": "data"}], llm_fn=mock_llm))
    assert chunks[0] == "single response"
    assert "---ANALYTICAL_META---" in chunks[-1]


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------


def test_analytical_response_to_dict():
    resp = AnalyticalResponse(
        key_data_points=[{"fact": "x", "source": "y"}],
        analysis="test",
        risks=[],
        recommendations=[],
        confidence_level="HIGH",
        data_sufficiency="sufficient",
        raw_answer="raw",
    )
    d = resp.to_dict()
    assert isinstance(d, dict)
    assert d["confidence_level"] == "HIGH"
    assert d["key_data_points"] == [{"fact": "x", "source": "y"}]


# ---------------------------------------------------------------------------
# Confidence / sufficiency logic
# ---------------------------------------------------------------------------


def test_confidence_scales_with_data_points():
    # 5+ data points => HIGH
    raw = """## Key Data Points
- A (Source: S1)
- B (Source: S2)
- C (Source: S3)
- D (Source: S4)
- E (Source: S5)

## Analysis
Good.

## Risks
- R — Severity: LOW

## Recommendations
- X — Confidence: LOW"""

    result = parse_analytical_response(raw)
    assert result.confidence_level == "HIGH"
    assert result.data_sufficiency == "sufficient"


def test_low_confidence_with_few_data_points():
    raw = """## Key Data Points
- A (Source: S1)

## Analysis
Limited.

## Risks
- R — Severity: HIGH

## Recommendations
- X — Confidence: LOW"""

    result = parse_analytical_response(raw)
    assert result.confidence_level == "LOW"
    assert result.data_sufficiency == "insufficient"
