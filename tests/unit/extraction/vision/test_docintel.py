import json

from src.extraction.vision.docintel import (
    CLASSIFIER_SYSTEM_PROMPT,
    COVERAGE_SYSTEM_PROMPT,
    RoutingDecision,
    parse_coverage_response,
    parse_routing_response,
)


def test_parse_routing_response_accepts_well_formed_json():
    text = json.dumps({
        "format": "pdf_scanned",
        "doc_type_hint": "invoice",
        "layout_complexity": "moderate",
        "has_handwriting": False,
        "suggested_path": "vision",
        "confidence": 0.8,
    })
    out = parse_routing_response(text)
    assert isinstance(out, RoutingDecision)
    assert out.suggested_path == "vision"
    assert out.layout_complexity == "moderate"
    assert 0.0 <= out.confidence <= 1.0


def test_parse_routing_response_tolerates_code_fences():
    text = "```json\n" + json.dumps({
        "format": "image",
        "doc_type_hint": "receipt",
        "layout_complexity": "simple",
        "has_handwriting": True,
        "suggested_path": "vision",
        "confidence": 0.7,
    }) + "\n```"
    out = parse_routing_response(text)
    assert out.has_handwriting is True


def test_parse_routing_response_returns_safe_default_on_garbage():
    out = parse_routing_response("this is not json")
    assert out.suggested_path == "vision"
    assert out.confidence < 0.2


def test_parse_coverage_response_accepts_complete_true():
    text = json.dumps({"complete": True, "missed_regions": [], "low_confidence_regions": []})
    out = parse_coverage_response(text)
    assert out["complete"] is True
    assert out["missed_regions"] == []


def test_parse_coverage_response_defaults_to_incomplete_on_garbage():
    out = parse_coverage_response("??? not json ???")
    assert out["complete"] is False
    assert out["missed_regions"] == []


def test_prompts_are_non_empty_strings():
    assert isinstance(CLASSIFIER_SYSTEM_PROMPT, str) and len(CLASSIFIER_SYSTEM_PROMPT) > 50
    assert isinstance(COVERAGE_SYSTEM_PROMPT, str) and len(COVERAGE_SYSTEM_PROMPT) > 50
