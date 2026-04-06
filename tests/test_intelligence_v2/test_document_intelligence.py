"""Tests for pre-computed document intelligence module."""

from unittest.mock import MagicMock, patch

import pytest


def test_compute_returns_document_intelligence():
    from src.intelligence_v2.document_intelligence import compute_document_intelligence

    def mock_llm(prompt):
        if "top 10 most important facts" in prompt:
            return '[{"fact": "Revenue was $1.5M", "source": "page 1", "confidence": 0.95}]'
        if "relationships" in prompt.lower():
            return '[{"subject": "Acme Corp", "predicate": "employs", "object": "John Smith", "evidence": "CEO"}]'
        if "temporal gaps" in prompt.lower():
            return '{"gaps": [], "events_timeline": [{"date": "2024-06-15", "description": "Q2 report"}]}'
        if "trends and anomalies" in prompt.lower():
            return '{"trends": [{"description": "Revenue growth", "direction": "up", "confidence": 0.8}], "anomalies": []}'
        if "summarise" in prompt.lower() or "summary" in prompt.lower():
            return "Acme Corp reported $1.5M revenue in Q2 2024. John Smith is CEO."
        return "{}"

    result = compute_document_intelligence(
        document_id="doc1",
        profile_id="prof1",
        subscription_id="sub1",
        extracted_text="Acme Corp reported revenue of $1,500,000 in Q2 2024. John Smith is the CEO.",
        llm_fn=mock_llm,
    )
    assert result.document_id == "doc1"
    assert result.profile_id == "prof1"
    assert len(result.entities) > 0  # heuristic should find Acme Corp, John Smith
    assert result.summary != ""
    assert result.processing_duration_seconds >= 0
    assert result.processed_at != ""


def test_compute_merges_existing_entities():
    from src.intelligence_v2.document_intelligence import compute_document_intelligence

    existing = [{"name": "Acme Corp", "type": "ORG", "mentions": 5, "confidence": 0.99}]
    result = compute_document_intelligence(
        document_id="doc2",
        profile_id="prof1",
        subscription_id="sub1",
        extracted_text="Acme Corp is based in London. John Smith leads Acme Corp.",
        existing_entities=existing,
    )
    # Should have Acme Corp from existing + John Smith from heuristic
    names = [e["name"].lower() for e in result.entities]
    assert "acme corp" in names
    assert "john smith" in names
    # Should not duplicate Acme Corp
    acme_count = sum(1 for n in names if n == "acme corp")
    assert acme_count == 1


def test_temporal_extraction_finds_dates():
    from src.intelligence_v2.document_intelligence import _extract_temporal_info

    result = _extract_temporal_info(
        "The contract was signed on 2024-03-15 and expires 2025-12-31."
    )
    assert result["date_range"]["earliest"] == "2024-03-15"
    assert result["date_range"]["latest"] == "2025-12-31"
    assert len(result["events_timeline"]) == 2


def test_temporal_extraction_handles_quarter_dates():
    from src.intelligence_v2.document_intelligence import _extract_temporal_info

    result = _extract_temporal_info("Revenue was reported in Q1 2024 and Q3 2024.")
    assert result["date_range"]["earliest"] == "2024-01-01"
    assert result["date_range"]["latest"] == "2024-07-01"


def test_temporal_extraction_handles_month_year():
    from src.intelligence_v2.document_intelligence import _extract_temporal_info

    result = _extract_temporal_info("The report covers January 2023 to December 2024.")
    assert result["date_range"]["earliest"] == "2023-01-01"
    assert result["date_range"]["latest"] == "2024-12-01"


def test_temporal_extraction_empty_text():
    from src.intelligence_v2.document_intelligence import _extract_temporal_info

    result = _extract_temporal_info("No dates here at all.")
    assert result["date_range"] == {}
    assert result["events_timeline"] == []


def test_numerical_extraction_finds_currency():
    from src.intelligence_v2.document_intelligence import _extract_numerical_info

    result = _extract_numerical_info(
        "Revenue was $1,500,000 and expenses were $1,200,000."
    )
    assert len(result["key_figures"]) >= 2
    types = [f["type"] for f in result["key_figures"]]
    assert "currency" in types or "number" in types


def test_numerical_extraction_finds_percentages():
    from src.intelligence_v2.document_intelligence import _extract_numerical_info

    result = _extract_numerical_info("Growth was 25.5% year-over-year, margin at 12%.")
    pct_figures = [f for f in result["key_figures"] if f["type"] == "percentage"]
    assert len(pct_figures) >= 2


def test_numerical_extraction_empty_text():
    from src.intelligence_v2.document_intelligence import _extract_numerical_info

    result = _extract_numerical_info("No numbers in this text whatsoever.")
    assert result["key_figures"] == []
    assert result["trends"] == []


def test_to_dict_serializable():
    from src.intelligence_v2.document_intelligence import DocumentIntelligence

    di = DocumentIntelligence(
        document_id="doc1", profile_id="p1", subscription_id="s1"
    )
    d = di.to_dict()
    assert d["document_id"] == "doc1"
    assert isinstance(d, dict)
    assert isinstance(d["entities"], list)
    assert isinstance(d["temporal"], dict)


def test_to_dict_roundtrip():
    from src.intelligence_v2.document_intelligence import DocumentIntelligence

    di = DocumentIntelligence(
        document_id="doc1",
        profile_id="p1",
        subscription_id="s1",
        entities=[{"name": "Test", "type": "ORG"}],
        summary="A test summary.",
        key_facts=[{"fact": "test", "source": "p1", "confidence": 0.9}],
    )
    d = di.to_dict()
    restored = DocumentIntelligence(**d)
    assert restored.document_id == di.document_id
    assert restored.entities == di.entities
    assert restored.summary == di.summary


def test_load_returns_none_when_not_found():
    from src.intelligence_v2.document_intelligence import load_document_intelligence

    with patch(
        "src.intelligence_v2.document_intelligence._get_intel_collection"
    ) as mock_coll:
        mock_coll.return_value = MagicMock()
        mock_coll.return_value.find_one.return_value = None
        result = load_document_intelligence("nonexistent")
        assert result is None


def test_load_returns_intelligence_when_found():
    from src.intelligence_v2.document_intelligence import load_document_intelligence

    stored_doc = {
        "document_id": "doc1",
        "profile_id": "p1",
        "subscription_id": "s1",
        "entities": [{"name": "Test"}],
        "relationships": [],
        "temporal": {},
        "numerical": {},
        "cross_doc": {},
        "summary": "Test summary",
        "key_facts": [],
        "processed_at": "2026-04-06T00:00:00",
        "processing_duration_seconds": 1.5,
        "model_version": "v2",
    }

    with patch(
        "src.intelligence_v2.document_intelligence._get_intel_collection"
    ) as mock_coll:
        mock_coll.return_value = MagicMock()
        mock_coll.return_value.find_one.return_value = {**stored_doc, "_id": "mongo_id"}
        result = load_document_intelligence("doc1")
        assert result is not None
        assert result.document_id == "doc1"
        assert result.summary == "Test summary"
        assert result.entities == [{"name": "Test"}]


def test_store_document_intelligence():
    from src.intelligence_v2.document_intelligence import (
        DocumentIntelligence,
        store_document_intelligence,
    )

    di = DocumentIntelligence(
        document_id="doc1", profile_id="p1", subscription_id="s1"
    )

    with patch(
        "src.intelligence_v2.document_intelligence._get_intel_collection"
    ) as mock_coll:
        mock_collection = MagicMock()
        mock_coll.return_value = mock_collection

        result = store_document_intelligence(di)
        assert result is True
        mock_collection.update_one.assert_called_once()
        call_args = mock_collection.update_one.call_args
        assert call_args[0][0] == {"document_id": "doc1"}
        assert call_args[1]["upsert"] is True


def test_store_returns_false_on_error():
    from src.intelligence_v2.document_intelligence import (
        DocumentIntelligence,
        store_document_intelligence,
    )

    di = DocumentIntelligence(
        document_id="doc1", profile_id="p1", subscription_id="s1"
    )

    with patch(
        "src.intelligence_v2.document_intelligence._get_intel_collection"
    ) as mock_coll:
        mock_coll.side_effect = Exception("DB down")
        result = store_document_intelligence(di)
        assert result is False


def test_key_facts_extraction():
    from src.intelligence_v2.document_intelligence import _extract_key_facts

    def mock_llm(prompt):
        return '[{"fact": "Revenue grew 20%", "source": "paragraph 3", "confidence": 0.9}]'

    facts = _extract_key_facts(
        "Revenue grew 20% year over year.", llm_fn=mock_llm
    )
    assert len(facts) >= 1
    assert facts[0]["fact"] == "Revenue grew 20%"


def test_key_facts_no_llm_returns_empty():
    from src.intelligence_v2.document_intelligence import _extract_key_facts

    facts = _extract_key_facts("Some document text without LLM.")
    assert facts == []


def test_generate_summary_with_llm():
    from src.intelligence_v2.document_intelligence import _generate_summary

    def mock_llm(prompt):
        return "This is a concise summary."

    result = _generate_summary("A long document text...", llm_fn=mock_llm)
    assert result == "This is a concise summary."


def test_generate_summary_without_llm():
    from src.intelligence_v2.document_intelligence import _generate_summary

    text = " ".join(["word"] * 300)
    result = _generate_summary(text)
    assert result.endswith("...")
    assert len(result.split()) <= 210  # 200 words + "..."


def test_safe_parse_json_with_fences():
    from src.intelligence_v2.document_intelligence import _safe_parse_json

    raw = '```json\n{"key": "value"}\n```'
    result = _safe_parse_json(raw)
    assert result == {"key": "value"}


def test_safe_parse_json_plain():
    from src.intelligence_v2.document_intelligence import _safe_parse_json

    raw = '{"key": "value"}'
    result = _safe_parse_json(raw)
    assert result == {"key": "value"}


def test_entities_heuristic():
    from src.intelligence_v2.document_intelligence import _extract_entities_heuristic

    text = "John Smith works at Acme Corp. John Smith is the CEO. Jane Doe is CFO."
    entities = _extract_entities_heuristic(text)
    names = [e["name"] for e in entities]
    assert "John Smith" in names
    assert "Acme Corp" in names
    assert "Jane Doe" in names
    # John Smith appears twice, so should be first (sorted by mentions)
    john = next(e for e in entities if e["name"] == "John Smith")
    assert john["mentions"] >= 2
