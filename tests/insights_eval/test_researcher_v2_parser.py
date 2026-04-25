import pytest

from src.intelligence.researcher_v2.parser import (
    parse_typed_insight_response,
    ParsedInsight,
    ParseError,
)


SAMPLE_GOOD = """
{
  "insights": [
    {
      "headline": "No flood coverage",
      "body": "The policy excludes flood damage. Flood damage exclusion is listed under exclusions.",
      "evidence_doc_spans": [
        {"document_id": "DOC-1", "page": 1, "char_start": 100, "char_end": 130, "quote": "Excludes: flood damage"}
      ],
      "external_kb_refs": [
        {"kb_id": "insurance_taxonomy_v1", "ref": "exclusions/flood", "label": "Flood exclusion"}
      ],
      "confidence": 0.95,
      "severity": "warn"
    }
  ]
}
"""


def test_parses_good_response():
    items = parse_typed_insight_response(SAMPLE_GOOD)
    assert len(items) == 1
    p = items[0]
    assert isinstance(p, ParsedInsight)
    assert p.headline == "No flood coverage"
    assert p.evidence_doc_spans[0]["document_id"] == "DOC-1"
    assert p.external_kb_refs[0]["kb_id"] == "insurance_taxonomy_v1"
    assert p.confidence == 0.95
    assert p.severity == "warn"


def test_strips_markdown_code_fence():
    fenced = "```json\n" + SAMPLE_GOOD + "\n```"
    items = parse_typed_insight_response(fenced)
    assert len(items) == 1


def test_empty_insights_list_returns_empty():
    items = parse_typed_insight_response('{"insights": []}')
    assert items == []


def test_malformed_json_raises():
    with pytest.raises(ParseError):
        parse_typed_insight_response("not json")


def test_missing_required_field_skipped():
    txt = '{"insights": [{"headline": "x"}]}'
    items = parse_typed_insight_response(txt)
    assert items == []
