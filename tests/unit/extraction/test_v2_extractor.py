"""Unit tests for V2Extractor (src/extraction/v2_extractor.py)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.extraction.v2_extractor import (
    V2Extractor,
    _build_extraction_prompt,
    _call_v2_model,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FULL_RESULT = {
    "think": "The document appears to be a standard invoice with clear fields.",
    "entities": [
        {"name": "Acme Corp", "type": "organization", "confidence": 0.95},
        {"name": "John Doe", "type": "person", "confidence": 0.88},
    ],
    "tables": [
        {
            "headers": ["Item", "Qty", "Price"],
            "rows": [["Widget A", "10", "$5.00"], ["Widget B", "3", "$12.00"]],
        }
    ],
    "fields": {"invoice_number": "INV-001", "date": "2026-04-01", "total": "$62.00"},
    "confidence": 0.91,
}


def _make_extractor(**kwargs) -> V2Extractor:
    return V2Extractor(
        ollama_host=kwargs.get("ollama_host", "http://localhost:11434"),
        model=kwargs.get("model", "docwain:v2"),
    )


# ---------------------------------------------------------------------------
# _build_extraction_prompt
# ---------------------------------------------------------------------------


class TestBuildExtractionPrompt:

    def test_includes_doc_type(self):
        prompt = _build_extraction_prompt(doc_type="invoice", page_type="body")
        assert "invoice" in prompt

    def test_includes_page_type(self):
        prompt = _build_extraction_prompt(doc_type="contract", page_type="cover")
        assert "cover" in prompt

    def test_instructs_think_tags(self):
        prompt = _build_extraction_prompt()
        assert "<think>" in prompt

    def test_requests_entities_key(self):
        prompt = _build_extraction_prompt()
        assert "entities" in prompt

    def test_requests_tables_key(self):
        prompt = _build_extraction_prompt()
        assert "tables" in prompt

    def test_requests_fields_key(self):
        prompt = _build_extraction_prompt()
        assert "fields" in prompt

    def test_requests_confidence_key(self):
        prompt = _build_extraction_prompt()
        assert "confidence" in prompt

    def test_requests_json_response(self):
        prompt = _build_extraction_prompt()
        assert "JSON" in prompt or "json" in prompt.lower()

    def test_defaults_are_unknown_and_body(self):
        prompt = _build_extraction_prompt()
        assert "unknown" in prompt
        assert "body" in prompt

    def test_returns_string(self):
        assert isinstance(_build_extraction_prompt(), str)


# ---------------------------------------------------------------------------
# _call_v2_model — direct unit tests
# ---------------------------------------------------------------------------


class TestCallV2Model:

    def _mock_response(self, content: str):
        """Build a mock requests.Response-like object."""
        import json as _json

        class MockResp:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self_inner):
                return {"message": {"content": content}}

        return MockResp()

    def test_returns_structured_dict(self):
        raw = '{"think":"ok","entities":[],"tables":[],"fields":{},"confidence":0.9}'
        with patch("requests.post", return_value=self._mock_response(raw)):
            result = _call_v2_model(
                "prompt",
                ollama_host="http://localhost:11434",
                model="docwain:v2",
            )
        assert set(result.keys()) == {"think", "entities", "tables", "fields", "confidence"}

    def test_parses_embedded_json(self):
        """JSON that is surrounded by prose should still be parsed."""
        raw = 'Here is the result:\n{"think":"reasoning","entities":[],"tables":[],"fields":{},"confidence":0.75}\nDone.'
        with patch("requests.post", return_value=self._mock_response(raw)):
            result = _call_v2_model(
                "prompt",
                ollama_host="http://localhost:11434",
                model="docwain:v2",
            )
        assert result["confidence"] == pytest.approx(0.75)

    def test_returns_default_on_json_parse_failure(self):
        with patch("requests.post", return_value=self._mock_response("not json at all")):
            result = _call_v2_model(
                "prompt",
                ollama_host="http://localhost:11434",
                model="docwain:v2",
            )
        assert result["confidence"] == pytest.approx(0.3)
        assert result["entities"] == []
        assert result["tables"] == []

    def test_returns_confidence_zero_on_exception(self):
        with patch("requests.post", side_effect=ConnectionError("offline")):
            result = _call_v2_model(
                "prompt",
                ollama_host="http://localhost:11434",
                model="docwain:v2",
            )
        assert result["confidence"] == pytest.approx(0.0)

    def test_passes_images_in_payload(self):
        raw = '{"think":"","entities":[],"tables":[],"fields":{},"confidence":0.5}'
        captured = {}

        def fake_post(url, json=None, timeout=None):
            captured["payload"] = json
            return self._mock_response(raw)

        with patch("requests.post", side_effect=fake_post):
            _call_v2_model(
                "prompt",
                images=["base64imgdata"],
                ollama_host="http://localhost:11434",
                model="docwain:v2",
            )

        messages = captured["payload"]["messages"]
        assert messages[0].get("images") == ["base64imgdata"]

    def test_confidence_is_float(self):
        raw = '{"think":"","entities":[],"tables":[],"fields":{},"confidence":0.88}'
        with patch("requests.post", return_value=self._mock_response(raw)):
            result = _call_v2_model(
                "prompt",
                ollama_host="http://localhost:11434",
                model="docwain:v2",
            )
        assert isinstance(result["confidence"], float)


# ---------------------------------------------------------------------------
# V2Extractor.extract — integration with mocked _call_v2_model
# ---------------------------------------------------------------------------


class TestV2ExtractorExtract:

    @pytest.fixture
    def extractor(self):
        return _make_extractor()

    def _patch(self, return_value: dict):
        return patch(
            "src.extraction.v2_extractor._call_v2_model",
            return_value=return_value,
        )

    def test_returns_structured_output(self, extractor):
        with self._patch(_FULL_RESULT):
            result = extractor.extract(b"bytes", "pdf")
        assert set(result.keys()) == {"think", "entities", "tables", "fields", "confidence"}

    def test_includes_think_reasoning(self, extractor):
        with self._patch(_FULL_RESULT):
            result = extractor.extract(b"bytes", "pdf")
        assert result["think"] == _FULL_RESULT["think"]
        assert len(result["think"]) > 0

    def test_entities_propagated(self, extractor):
        with self._patch(_FULL_RESULT):
            result = extractor.extract(b"bytes", "pdf")
        assert result["entities"] == _FULL_RESULT["entities"]

    def test_tables_propagated(self, extractor):
        with self._patch(_FULL_RESULT):
            result = extractor.extract(b"bytes", "pdf")
        assert result["tables"] == _FULL_RESULT["tables"]

    def test_fields_propagated(self, extractor):
        with self._patch(_FULL_RESULT):
            result = extractor.extract(b"bytes", "pdf")
        assert result["fields"] == _FULL_RESULT["fields"]

    def test_confidence_propagated(self, extractor):
        with self._patch(_FULL_RESULT):
            result = extractor.extract(b"bytes", "pdf")
        assert result["confidence"] == pytest.approx(0.91)

    def test_prompt_includes_doc_type(self, extractor):
        """Verify the prompt passed to _call_v2_model contains the doc_type."""
        captured_prompt: list[str] = []

        def fake_call(prompt, images=None, *, ollama_host, model, timeout=120):
            captured_prompt.append(prompt)
            return _FULL_RESULT

        with patch("src.extraction.v2_extractor._call_v2_model", side_effect=fake_call):
            extractor.extract(b"bytes", "pdf", doc_type="invoice")

        assert len(captured_prompt) == 1
        assert "invoice" in captured_prompt[0]

    def test_prompt_includes_page_type(self, extractor):
        captured_prompt: list[str] = []

        def fake_call(prompt, images=None, *, ollama_host, model, timeout=120):
            captured_prompt.append(prompt)
            return _FULL_RESULT

        with patch("src.extraction.v2_extractor._call_v2_model", side_effect=fake_call):
            extractor.extract(b"bytes", "pdf", page_type="cover")

        assert "cover" in captured_prompt[0]

    def test_page_images_forwarded(self, extractor):
        captured_images: list = []

        def fake_call(prompt, images=None, *, ollama_host, model, timeout=120):
            captured_images.append(images)
            return _FULL_RESULT

        with patch("src.extraction.v2_extractor._call_v2_model", side_effect=fake_call):
            extractor.extract(b"bytes", "pdf", page_images=["img1", "img2"])

        assert captured_images[0] == ["img1", "img2"]

    def test_text_content_appended_to_prompt(self, extractor):
        captured_prompt: list[str] = []

        def fake_call(prompt, images=None, *, ollama_host, model, timeout=120):
            captured_prompt.append(prompt)
            return _FULL_RESULT

        with patch("src.extraction.v2_extractor._call_v2_model", side_effect=fake_call):
            extractor.extract(b"bytes", "pdf", text_content="Invoice total $100")

        assert "Invoice total $100" in captured_prompt[0]

    def test_default_doc_type_is_unknown(self, extractor):
        captured_prompt: list[str] = []

        def fake_call(prompt, images=None, *, ollama_host, model, timeout=120):
            captured_prompt.append(prompt)
            return _FULL_RESULT

        with patch("src.extraction.v2_extractor._call_v2_model", side_effect=fake_call):
            extractor.extract(b"bytes", "pdf")

        assert "unknown" in captured_prompt[0]

    def test_default_page_type_is_body(self, extractor):
        captured_prompt: list[str] = []

        def fake_call(prompt, images=None, *, ollama_host, model, timeout=120):
            captured_prompt.append(prompt)
            return _FULL_RESULT

        with patch("src.extraction.v2_extractor._call_v2_model", side_effect=fake_call):
            extractor.extract(b"bytes", "pdf")

        assert "body" in captured_prompt[0]

    def test_constructor_defaults(self):
        extractor = V2Extractor()
        assert extractor.ollama_host == "http://localhost:11434"
        assert extractor.model == "docwain:v2"

    def test_constructor_custom_values(self):
        extractor = V2Extractor(
            ollama_host="http://gpu-host:11434",
            model="docwain:v2-finetuned",
        )
        assert extractor.ollama_host == "http://gpu-host:11434"
        assert extractor.model == "docwain:v2-finetuned"

    def test_model_forwarded_to_call(self, extractor):
        captured: dict = {}

        def fake_call(prompt, images=None, *, ollama_host, model, timeout=120):
            captured["model"] = model
            return _FULL_RESULT

        with patch("src.extraction.v2_extractor._call_v2_model", side_effect=fake_call):
            extractor.extract(b"bytes", "pdf")

        assert captured["model"] == "docwain:v2"
