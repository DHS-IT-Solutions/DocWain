"""Unit tests for src/kg/llm_entity_extractor.py"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.kg.llm_entity_extractor import (
    LLMEntityExtractor,
    _build_extraction_prompt,
    _call_llm,
)


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

SAMPLE_LEGAL_TEXT = (
    "This Agreement is entered into by Acme Corp and John Smith on 2024-01-15. "
    "The contract is governed by the laws of California."
)

SAMPLE_HR_TEXT = (
    "Jane Doe is employed by TechCo Inc. She reports to Bob Manager and holds "
    "a Python certification. Her email is jane@techco.com."
)

_FAKE_LLM_RESPONSE = {
    "entities": [
        {"name": "Acme Corp", "type": "ORGANIZATION", "aliases": [], "confidence": 0.95},
        {"name": "John Smith", "type": "PERSON", "aliases": ["J. Smith"], "confidence": 0.9},
        {"name": "2024-01-15", "type": "DATE", "aliases": [], "confidence": 0.85},
        {"name": "California", "type": "LOCATION", "aliases": [], "confidence": 0.6},
    ],
    "relationships": [
        {
            "source": "John Smith",
            "target": "Agreement",
            "type": "party_to",
            "evidence": "entered into by Acme Corp and John Smith",
            "confidence": 0.88,
            "temporal_bounds": None,
        }
    ],
}

_FAKE_LLM_RESPONSE_HR = {
    "entities": [
        {"name": "Jane Doe", "type": "PERSON", "aliases": [], "confidence": 0.92},
        {"name": "TechCo Inc", "type": "ORGANIZATION", "aliases": [], "confidence": 0.9},
        {"name": "Bob Manager", "type": "PERSON", "aliases": [], "confidence": 0.85},
        {"name": "Python", "type": "SKILL", "aliases": [], "confidence": 0.95},
        {"name": "jane@techco.com", "type": "EMAIL", "aliases": [], "confidence": 0.99},
        {"name": "Low Conf Widget", "type": "PRODUCT", "aliases": [], "confidence": 0.3},
    ],
    "relationships": [
        {
            "source": "Jane Doe",
            "target": "TechCo Inc",
            "type": "employed_by",
            "evidence": "is employed by TechCo Inc",
            "confidence": 0.92,
            "temporal_bounds": None,
        }
    ],
}


# ---------------------------------------------------------------------------
# _build_extraction_prompt
# ---------------------------------------------------------------------------


class TestBuildExtractionPrompt:
    def test_contains_domain_name(self):
        prompt = _build_extraction_prompt("some text", domain="legal")
        assert "legal" in prompt

    def test_contains_domain_relationship_types(self):
        prompt = _build_extraction_prompt("some text", domain="legal")
        # party_to is a legal relationship type from ontology
        assert "party_to" in prompt

    def test_contains_text(self):
        prompt = _build_extraction_prompt("hello world", domain="generic")
        assert "hello world" in prompt

    def test_generic_domain_includes_related_to(self):
        prompt = _build_extraction_prompt("some text", domain="generic")
        assert "related_to" in prompt

    def test_schema_hints_present(self):
        prompt = _build_extraction_prompt("some text", domain="financial")
        assert "entities" in prompt
        assert "relationships" in prompt
        assert "confidence" in prompt

    def test_unknown_domain_falls_back_to_generic(self):
        # get_domain_relationships falls back to generic for unknown domains
        prompt = _build_extraction_prompt("some text", domain="unknown_domain")
        assert "related_to" in prompt

    def test_hr_relationships_present(self):
        prompt = _build_extraction_prompt("some text", domain="hr")
        assert "employed_by" in prompt

    def test_medical_relationships_present(self):
        prompt = _build_extraction_prompt("some text", domain="medical")
        assert "diagnosed_with" in prompt


# ---------------------------------------------------------------------------
# _call_llm (mocked at requests level)
# ---------------------------------------------------------------------------


class TestCallLlm:
    def _make_response(self, payload: dict) -> MagicMock:
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"response": json.dumps(payload)}
        return mock_resp

    def test_returns_entities_and_relationships(self):
        with patch("requests.post") as mock_post:
            mock_post.return_value = self._make_response(_FAKE_LLM_RESPONSE)
            result = _call_llm(
                "dummy prompt",
                ollama_host="http://localhost:11434",
                model="docwain:v2",
            )
        assert "entities" in result
        assert "relationships" in result
        assert len(result["entities"]) == 4
        assert len(result["relationships"]) == 1

    def test_posts_to_generate_endpoint(self):
        with patch("requests.post") as mock_post:
            mock_post.return_value = self._make_response(_FAKE_LLM_RESPONSE)
            _call_llm("prompt", ollama_host="http://myhost:11434", model="test-model")
        args, kwargs = mock_post.call_args
        assert "http://myhost:11434/api/generate" in args[0]

    def test_uses_specified_model(self):
        with patch("requests.post") as mock_post:
            mock_post.return_value = self._make_response(_FAKE_LLM_RESPONSE)
            _call_llm("prompt", ollama_host="http://localhost:11434", model="my-custom-model")
        _, kwargs = mock_post.call_args
        assert kwargs["json"]["model"] == "my-custom-model"

    def test_missing_entities_key_defaults_to_empty_list(self):
        incomplete = {"relationships": []}
        with patch("requests.post") as mock_post:
            mock_post.return_value = self._make_response(incomplete)
            result = _call_llm("prompt", ollama_host="http://localhost:11434", model="m")
        assert result["entities"] == []

    def test_missing_relationships_key_defaults_to_empty_list(self):
        incomplete = {"entities": [{"name": "X", "type": "Y", "aliases": [], "confidence": 0.5}]}
        with patch("requests.post") as mock_post:
            mock_post.return_value = self._make_response(incomplete)
            result = _call_llm("prompt", ollama_host="http://localhost:11434", model="m")
        assert result["relationships"] == []

    def test_raises_value_error_on_non_json_response(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"response": "Sorry, I cannot help with that."}
        with patch("requests.post", return_value=mock_resp):
            with pytest.raises(ValueError, match="No JSON object found"):
                _call_llm("prompt", ollama_host="http://localhost:11434", model="m")

    def test_json_embedded_in_prose_is_extracted(self):
        payload = {"entities": [], "relationships": []}
        prose_with_json = f"Here is the result:\n{json.dumps(payload)}\nEnd."
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"response": prose_with_json}
        with patch("requests.post", return_value=mock_resp):
            result = _call_llm("prompt", ollama_host="http://localhost:11434", model="m")
        assert result["entities"] == []


# ---------------------------------------------------------------------------
# LLMEntityExtractor.extract — mocking _call_llm
# ---------------------------------------------------------------------------


class TestLLMEntityExtractorExtract:
    def setup_method(self):
        self.extractor = LLMEntityExtractor(
            ollama_host="http://localhost:11434", model="docwain:v2"
        )

    def test_extract_returns_entities_and_relationships_keys(self):
        with patch("src.kg.llm_entity_extractor._call_llm", return_value=_FAKE_LLM_RESPONSE):
            result = self.extractor.extract(SAMPLE_LEGAL_TEXT, domain="legal")
        assert "entities" in result
        assert "relationships" in result

    def test_extract_returns_correct_entity_count(self):
        with patch("src.kg.llm_entity_extractor._call_llm", return_value=_FAKE_LLM_RESPONSE):
            result = self.extractor.extract(SAMPLE_LEGAL_TEXT, domain="legal")
        assert len(result["entities"]) == 4

    def test_extract_returns_correct_relationship_count(self):
        with patch("src.kg.llm_entity_extractor._call_llm", return_value=_FAKE_LLM_RESPONSE):
            result = self.extractor.extract(SAMPLE_LEGAL_TEXT, domain="legal")
        assert len(result["relationships"]) == 1

    def test_extract_passes_domain_hint_to_prompt(self):
        """Verify the prompt built for extract() contains domain-specific rel types."""
        captured_prompts = []

        def fake_call_llm(prompt, *, ollama_host, model):
            captured_prompts.append(prompt)
            return _FAKE_LLM_RESPONSE

        with patch("src.kg.llm_entity_extractor._call_llm", side_effect=fake_call_llm):
            self.extractor.extract(SAMPLE_LEGAL_TEXT, domain="legal")

        assert len(captured_prompts) == 1
        assert "party_to" in captured_prompts[0]
        assert "legal" in captured_prompts[0]

    def test_extract_with_financial_domain(self):
        financial_response = {
            "entities": [
                {"name": "INV-001", "type": "INVOICE", "aliases": [], "confidence": 0.9}
            ],
            "relationships": [
                {
                    "source": "INV-001",
                    "target": "2024-03-01",
                    "type": "billed_on",
                    "evidence": "billed on 2024-03-01",
                    "confidence": 0.85,
                    "temporal_bounds": None,
                }
            ],
        }
        captured = []

        def fake_call_llm(prompt, *, ollama_host, model):
            captured.append(prompt)
            return financial_response

        with patch("src.kg.llm_entity_extractor._call_llm", side_effect=fake_call_llm):
            result = self.extractor.extract("Invoice INV-001 billed on 2024-03-01", domain="financial")

        assert result["entities"][0]["name"] == "INV-001"
        assert "invoiced_by" in captured[0]  # financial domain rel type

    def test_extract_with_generic_domain_default(self):
        with patch("src.kg.llm_entity_extractor._call_llm", return_value=_FAKE_LLM_RESPONSE) as mock_fn:
            result = self.extractor.extract(SAMPLE_LEGAL_TEXT)
        # Should not raise; generic domain used by default
        assert "entities" in result

    def test_extract_entity_schema_shape(self):
        with patch("src.kg.llm_entity_extractor._call_llm", return_value=_FAKE_LLM_RESPONSE):
            result = self.extractor.extract(SAMPLE_LEGAL_TEXT, domain="legal")
        for entity in result["entities"]:
            assert "name" in entity
            assert "type" in entity
            assert "aliases" in entity
            assert "confidence" in entity

    def test_extract_relationship_schema_shape(self):
        with patch("src.kg.llm_entity_extractor._call_llm", return_value=_FAKE_LLM_RESPONSE):
            result = self.extractor.extract(SAMPLE_LEGAL_TEXT, domain="legal")
        for rel in result["relationships"]:
            assert "source" in rel
            assert "target" in rel
            assert "type" in rel
            assert "evidence" in rel
            assert "confidence" in rel
            assert "temporal_bounds" in rel


# ---------------------------------------------------------------------------
# LLMEntityExtractor.validate_entities
# ---------------------------------------------------------------------------


class TestValidateEntities:
    """Tests for the cross-validation logic in validate_entities."""

    def setup_method(self):
        self.extractor = LLMEntityExtractor()

    # ------------------------------------------------------------------
    # Helper: create a minimal LLM entity dict
    # ------------------------------------------------------------------

    @staticmethod
    def _entity(name: str, entity_type: str, confidence: float, aliases: list | None = None) -> dict:
        return {
            "name": name,
            "type": entity_type,
            "aliases": aliases or [],
            "confidence": confidence,
        }

    # ------------------------------------------------------------------
    # cross_validated=True when entity is also found by baseline
    # ------------------------------------------------------------------

    def test_entity_confirmed_by_baseline_is_cross_validated(self):
        """An entity found by both LLM and regex/spaCy gets cross_validated=True."""
        # jane@techco.com will be found by EMAIL_RE in EntityExtractor
        entities = [self._entity("jane@techco.com", "EMAIL", 0.6)]
        result = self.extractor.validate_entities(entities, SAMPLE_HR_TEXT)
        assert len(result) == 1
        assert result[0]["cross_validated"] is True

    # ------------------------------------------------------------------
    # cross_validated=False for high-confidence entities not in baseline
    # ------------------------------------------------------------------

    def test_high_confidence_entity_passes_without_cross_validation(self):
        """High-confidence (>=0.8) LLM entities pass with cross_validated=False."""
        # "TechCo Inc" may or may not be found by regex; use confidence=0.99 to
        # guarantee the high-confidence path.
        entities = [self._entity("Totally Unknown Thing XYZ", "ORGANIZATION", 0.99)]
        result = self.extractor.validate_entities(entities, "Some unrelated sentence.")
        assert len(result) == 1
        assert result[0]["cross_validated"] is False

    # ------------------------------------------------------------------
    # Low-confidence entities without baseline confirmation are rejected
    # ------------------------------------------------------------------

    def test_low_confidence_entity_without_baseline_is_rejected(self):
        """Low-confidence entities not found by baseline extractor are dropped."""
        entities = [self._entity("Nonexistent Banana Corp", "ORGANIZATION", 0.3)]
        result = self.extractor.validate_entities(entities, "Some text with no relevant content.")
        assert len(result) == 0

    # ------------------------------------------------------------------
    # Mixed batch
    # ------------------------------------------------------------------

    def test_mixed_batch_validates_correctly(self):
        """
        Batch with:
          - email (low conf, confirmed by baseline) -> cross_validated=True
          - high-confidence unknown entity -> cross_validated=False
          - low-confidence unknown entity -> rejected
        """
        entities = [
            self._entity("jane@techco.com", "EMAIL", 0.5),         # confirmed by regex
            self._entity("Mystery Corp XYZ", "ORGANIZATION", 0.95), # high-conf, unknown
            self._entity("Ghost Entity", "PERSON", 0.2),            # rejected
        ]
        result = self.extractor.validate_entities(entities, SAMPLE_HR_TEXT)

        names = [e["name"] for e in result]
        assert "jane@techco.com" in names
        assert "Mystery Corp XYZ" in names
        assert "Ghost Entity" not in names

        confirmed = {e["name"]: e["cross_validated"] for e in result}
        assert confirmed["jane@techco.com"] is True
        assert confirmed["Mystery Corp XYZ"] is False

    # ------------------------------------------------------------------
    # Empty inputs
    # ------------------------------------------------------------------

    def test_empty_entity_list_returns_empty(self):
        result = self.extractor.validate_entities([], SAMPLE_HR_TEXT)
        assert result == []

    def test_empty_text_with_high_conf_entities_still_passes(self):
        """High-confidence entities still pass even when baseline finds nothing."""
        entities = [self._entity("Quantum Nexus Corp", "ORGANIZATION", 0.9)]
        result = self.extractor.validate_entities(entities, "")
        assert len(result) == 1
        assert result[0]["cross_validated"] is False

    # ------------------------------------------------------------------
    # Boundary: confidence exactly at threshold
    # ------------------------------------------------------------------

    def test_entity_at_exactly_threshold_passes(self):
        """Confidence exactly 0.8 meets the high-confidence threshold."""
        entities = [self._entity("Unknown Galaxy Z", "ORGANIZATION", 0.8)]
        result = self.extractor.validate_entities(entities, "unrelated text blah")
        assert len(result) == 1
        assert result[0]["cross_validated"] is False

    def test_entity_just_below_threshold_is_rejected_when_not_in_baseline(self):
        """Confidence 0.79 does not meet threshold; rejected if not in baseline."""
        entities = [self._entity("Invisible Widget Corp", "ORGANIZATION", 0.79)]
        result = self.extractor.validate_entities(entities, "completely unrelated text")
        assert len(result) == 0

    # ------------------------------------------------------------------
    # Original entity fields preserved
    # ------------------------------------------------------------------

    def test_original_fields_preserved_after_validation(self):
        """Validated entities retain all original fields plus cross_validated."""
        entities = [
            self._entity("jane@techco.com", "EMAIL", 0.7, aliases=["jane"]),
        ]
        result = self.extractor.validate_entities(entities, SAMPLE_HR_TEXT)
        assert len(result) == 1
        assert result[0]["aliases"] == ["jane"]
        assert result[0]["confidence"] == 0.7
        assert result[0]["type"] == "EMAIL"
        assert "cross_validated" in result[0]
