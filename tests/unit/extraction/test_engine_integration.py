"""Unit tests for ExtractionEngine integration (v2_extractor + triage)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.extraction.engine import ExtractionEngine
from src.extraction.models import ExtractionResult, TriageResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_triage_result() -> TriageResult:
    return TriageResult(
        document_type="clean_digital",
        engine_weights={"structural": 0.9, "semantic": 0.8, "vision": 0.3, "v2": 0.7},
        preprocessing_directives=[],
        page_types=["digital"],
        confidence=0.9,
    )


def _make_extraction_result() -> ExtractionResult:
    return ExtractionResult(
        document_id="doc-test",
        subscription_id="sub-1",
        profile_id="prof-1",
        clean_text="",
        structure={},
        entities=[],
        relationships=[],
        tables=[],
        metadata={"extraction_confidence": 0.85},
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine_with_mocks():
    """Return an ExtractionEngine with all sub-components replaced by mocks."""
    with (
        patch("src.extraction.engine.StructuralExtractor") as MockStructural,
        patch("src.extraction.engine.SemanticExtractor") as MockSemantic,
        patch("src.extraction.engine.VisionExtractor") as MockVision,
        patch("src.extraction.engine.ExtractionMerger") as MockMerger,
        patch("src.extraction.engine.V2Extractor") as MockV2,
        patch("src.extraction.engine.DocumentTriager") as MockTriager,
    ):
        triage_result = _make_triage_result()
        extraction_result = _make_extraction_result()

        MockStructural.return_value.extract.return_value = {}
        MockSemantic.return_value.extract.return_value = {}
        MockVision.return_value.extract.return_value = {}
        MockV2.return_value.extract.return_value = {}
        MockTriager.return_value.triage.return_value = triage_result
        MockMerger.return_value.merge.return_value = extraction_result

        engine = ExtractionEngine(ollama_host="http://localhost:11434")
        yield engine, MockTriager, MockMerger, triage_result, extraction_result


# ---------------------------------------------------------------------------
# Attribute presence tests
# ---------------------------------------------------------------------------

class TestEngineAttributes:
    def test_has_v2_extractor(self, engine_with_mocks):
        engine, *_ = engine_with_mocks
        assert hasattr(engine, "v2_extractor")

    def test_has_triager(self, engine_with_mocks):
        engine, *_ = engine_with_mocks
        assert hasattr(engine, "triager")


# ---------------------------------------------------------------------------
# Triage integration
# ---------------------------------------------------------------------------

class TestEngineTriage:
    def test_triage_is_called_during_extract(self, engine_with_mocks):
        engine, MockTriager, MockMerger, triage_result, _ = engine_with_mocks

        engine.extract(
            document_id="doc-1",
            subscription_id="sub-1",
            profile_id="prof-1",
            document_bytes=b"bytes",
            file_type="pdf",
            text_content="some text",
        )

        engine.triager.triage.assert_called_once()

    def test_triage_receives_file_type(self, engine_with_mocks):
        engine, *_ = engine_with_mocks

        engine.extract(
            document_id="doc-1",
            subscription_id="sub-1",
            profile_id="prof-1",
            document_bytes=b"bytes",
            file_type="pdf",
            text_content="text",
        )

        call_kwargs = engine.triager.triage.call_args
        assert call_kwargs.kwargs.get("file_type") == "pdf" or call_kwargs.args[0] == "pdf"


# ---------------------------------------------------------------------------
# Merger receives v2 and triage
# ---------------------------------------------------------------------------

class TestMergerReceivesV2AndTriage:
    def test_merger_receives_v2_result(self, engine_with_mocks):
        engine, MockTriager, MockMerger, triage_result, _ = engine_with_mocks

        engine.extract(
            document_id="doc-1",
            subscription_id="sub-1",
            profile_id="prof-1",
            document_bytes=b"bytes",
            file_type="pdf",
        )

        merge_kwargs = engine.merger.merge.call_args.kwargs
        assert "v2" in merge_kwargs

    def test_merger_receives_triage_result(self, engine_with_mocks):
        engine, MockTriager, MockMerger, triage_result, _ = engine_with_mocks

        engine.extract(
            document_id="doc-1",
            subscription_id="sub-1",
            profile_id="prof-1",
            document_bytes=b"bytes",
            file_type="pdf",
        )

        merge_kwargs = engine.merger.merge.call_args.kwargs
        assert "triage" in merge_kwargs
        assert merge_kwargs["triage"] is triage_result

    def test_extract_returns_extraction_result(self, engine_with_mocks):
        engine, _, _, _, expected_result = engine_with_mocks

        result = engine.extract(
            document_id="doc-1",
            subscription_id="sub-1",
            profile_id="prof-1",
            document_bytes=b"bytes",
            file_type="pdf",
        )

        assert result is expected_result
