import pytest
from pydantic import ValidationError


def test_extract_request_valid_modes():
    from standalone.schemas import ExtractRequest

    for fmt in ("json", "csv", "sections", "flatfile", "tables"):
        req = ExtractRequest(output_format=fmt)
        assert req.output_format == fmt


def test_extract_request_invalid_mode():
    from standalone.schemas import ExtractRequest

    with pytest.raises(ValidationError):
        ExtractRequest(output_format="xml")


def test_extract_request_optional_prompt():
    from standalone.schemas import ExtractRequest

    req = ExtractRequest(output_format="json")
    assert req.prompt is None

    req2 = ExtractRequest(output_format="json", prompt="focus on tables")
    assert req2.prompt == "focus on tables"


def test_intelligence_request_defaults():
    from standalone.schemas import IntelligenceRequest

    req = IntelligenceRequest()
    assert req.analysis_type == "auto"
    assert req.prompt is None


def test_intelligence_request_valid_types():
    from standalone.schemas import IntelligenceRequest

    for t in ("summary", "key_facts", "risk_assessment", "recommendations", "auto"):
        req = IntelligenceRequest(analysis_type=t)
        assert req.analysis_type == t


def test_intelligence_request_invalid_type():
    from standalone.schemas import IntelligenceRequest

    with pytest.raises(ValidationError):
        IntelligenceRequest(analysis_type="magic")


def test_extract_response_structure():
    from standalone.schemas import ExtractResponse, ResponseMetadata

    resp = ExtractResponse(
        request_id="abc-123",
        document_type="invoice",
        output_format="json",
        content={"items": [{"name": "Widget", "price": 10}]},
        metadata=ResponseMetadata(pages=3, processing_time_ms=1200),
    )
    assert resp.request_id == "abc-123"
    assert resp.metadata.pages == 3


def test_intelligence_response_structure():
    from standalone.schemas import IntelligenceResponse, ResponseMetadata

    resp = IntelligenceResponse(
        request_id="def-456",
        document_type="contract",
        analysis_type="risk_assessment",
        insights={"summary": "High risk", "findings": [], "evidence": []},
        metadata=ResponseMetadata(pages=12, processing_time_ms=3400),
    )
    assert resp.analysis_type == "risk_assessment"
    assert resp.insights["summary"] == "High risk"


def test_key_create_request():
    from standalone.schemas import KeyCreateRequest

    req = KeyCreateRequest(name="production-key")
    assert req.name == "production-key"


def test_key_create_request_empty_name_rejected():
    from standalone.schemas import KeyCreateRequest

    with pytest.raises(ValidationError):
        KeyCreateRequest(name="")


def test_key_response_structure():
    from standalone.schemas import KeyCreateResponse

    resp = KeyCreateResponse(
        key_id="k-123",
        raw_key="dw_sa_abc123",
        key_prefix="dw_sa_abc",
        name="test",
        created_at="2026-04-15T00:00:00Z",
    )
    assert resp.raw_key == "dw_sa_abc123"


def test_key_list_item():
    from standalone.schemas import KeyListItem

    item = KeyListItem(
        key_id="k-123",
        key_prefix="dw_sa_abc",
        name="test",
        created_at="2026-04-15T00:00:00Z",
        total_requests=42,
    )
    assert item.total_requests == 42
