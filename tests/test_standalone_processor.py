"""Unit tests for src.api.standalone_processor.

Heavy external dependencies (Qdrant, embedder, LLM) are mocked.
"""
from __future__ import annotations

import io
import json
import struct
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# detect_file_type
# ---------------------------------------------------------------------------


def test_detect_file_type_pdf_by_extension():
    from src.api.standalone_processor import detect_file_type

    result = detect_file_type("report.pdf", b"anything")
    assert result == "pdf"


def test_detect_file_type_pdf_by_magic_bytes():
    from src.api.standalone_processor import detect_file_type

    result = detect_file_type("unknown.bin", b"%PDF-1.4 binary garbage")
    assert result == "pdf"


def test_detect_file_type_docx_by_extension():
    from src.api.standalone_processor import detect_file_type

    result = detect_file_type("contract.docx", b"")
    assert result == "docx"


def test_detect_file_type_doc_by_extension():
    from src.api.standalone_processor import detect_file_type

    result = detect_file_type("old.doc", b"")
    assert result == "docx"


def test_detect_file_type_pptx_by_extension():
    from src.api.standalone_processor import detect_file_type

    result = detect_file_type("slides.pptx", b"")
    assert result == "pptx"


def test_detect_file_type_ppt_by_extension():
    from src.api.standalone_processor import detect_file_type

    result = detect_file_type("old_slides.ppt", b"")
    assert result == "pptx"


def test_detect_file_type_png_by_extension():
    from src.api.standalone_processor import detect_file_type

    result = detect_file_type("photo.png", b"")
    assert result == "image"


def test_detect_file_type_jpg_by_extension():
    from src.api.standalone_processor import detect_file_type

    result = detect_file_type("photo.jpg", b"")
    assert result == "image"


def test_detect_file_type_jpeg_by_extension():
    from src.api.standalone_processor import detect_file_type

    result = detect_file_type("photo.jpeg", b"")
    assert result == "image"


def test_detect_file_type_image_by_png_magic():
    from src.api.standalone_processor import detect_file_type

    png_magic = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
    result = detect_file_type("unknown", png_magic)
    assert result == "image"


def test_detect_file_type_image_by_jpeg_magic():
    from src.api.standalone_processor import detect_file_type

    jpeg_magic = b"\xff\xd8\xff" + b"\x00" * 100
    result = detect_file_type("unknown", jpeg_magic)
    assert result == "image"


def test_detect_file_type_csv_by_extension():
    from src.api.standalone_processor import detect_file_type

    result = detect_file_type("data.csv", b"col1,col2\n1,2\n")
    assert result == "csv"


def test_detect_file_type_xlsx_by_extension():
    from src.api.standalone_processor import detect_file_type

    result = detect_file_type("spreadsheet.xlsx", b"")
    assert result == "excel"


def test_detect_file_type_xls_by_extension():
    from src.api.standalone_processor import detect_file_type

    result = detect_file_type("old.xls", b"")
    assert result == "excel"


def test_detect_file_type_txt_by_extension():
    from src.api.standalone_processor import detect_file_type

    result = detect_file_type("notes.txt", b"hello world")
    assert result == "txt"


def test_detect_file_type_md_by_extension():
    from src.api.standalone_processor import detect_file_type

    result = detect_file_type("README.md", b"# Title")
    assert result == "txt"


def test_detect_file_type_pk_magic_bytes():
    """PK magic bytes (zip-like) should produce docx when extension is also docx."""
    from src.api.standalone_processor import detect_file_type

    pk_bytes = b"PK\x03\x04" + b"\x00" * 100
    # If extension is .docx, extension wins
    result = detect_file_type("file.docx", pk_bytes)
    assert result == "docx"


def test_detect_file_type_unknown_fallback():
    from src.api.standalone_processor import detect_file_type

    result = detect_file_type("mystery.xyz", b"\x00\x01\x02\x03")
    assert result == "txt"  # default fallback


# ---------------------------------------------------------------------------
# extract_from_bytes
# ---------------------------------------------------------------------------


def _make_extracted():
    from src.api.pipeline_models import ExtractedDocument

    return ExtractedDocument(
        full_text="sample text",
        sections=[],
        tables=[],
        figures=[],
        chunk_candidates=[],
    )


def test_extract_from_bytes_pdf():
    """PDF bytes should route to extract_text_from_pdf."""
    from src.api.standalone_processor import extract_from_bytes

    mock_extractor = MagicMock()
    mock_extractor.extract_text_from_pdf.return_value = _make_extracted()

    with patch("src.api.standalone_processor._get_document_extractor", return_value=mock_extractor):
        result = extract_from_bytes(b"%PDF-1.4 data", "test.pdf")

    mock_extractor.extract_text_from_pdf.assert_called_once()
    assert result.full_text == "sample text"


def test_extract_from_bytes_docx():
    """DOCX bytes should route to extract_text_from_docx."""
    from src.api.standalone_processor import extract_from_bytes

    mock_extractor = MagicMock()
    mock_extractor.extract_text_from_docx.return_value = _make_extracted()

    with patch("src.api.standalone_processor._get_document_extractor", return_value=mock_extractor):
        result = extract_from_bytes(b"PK fake docx", "contract.docx")

    mock_extractor.extract_text_from_docx.assert_called_once()
    assert result.full_text == "sample text"


def test_extract_from_bytes_pptx():
    """PPTX bytes should route to extract_text_from_pptx."""
    from src.api.standalone_processor import extract_from_bytes

    mock_extractor = MagicMock()
    mock_extractor.extract_text_from_pptx.return_value = _make_extracted()

    with patch("src.api.standalone_processor._get_document_extractor", return_value=mock_extractor):
        result = extract_from_bytes(b"PK fake pptx", "slides.pptx")

    mock_extractor.extract_text_from_pptx.assert_called_once()
    assert result.full_text == "sample text"


def test_extract_from_bytes_txt():
    """TXT bytes should route to extract_text_from_txt."""
    from src.api.standalone_processor import extract_from_bytes

    mock_extractor = MagicMock()
    mock_extractor.extract_text_from_txt.return_value = _make_extracted()

    with patch("src.api.standalone_processor._get_document_extractor", return_value=mock_extractor):
        result = extract_from_bytes(b"plain text content", "notes.txt")

    mock_extractor.extract_text_from_txt.assert_called_once()
    assert result.full_text == "sample text"


def test_extract_from_bytes_csv():
    """CSV bytes should call pd.read_csv then extract_dataframe."""
    import pandas as pd

    from src.api.standalone_processor import extract_from_bytes

    mock_extractor = MagicMock()
    mock_extractor.extract_dataframe.return_value = _make_extracted()

    csv_bytes = b"a,b,c\n1,2,3\n4,5,6\n"

    with patch("src.api.standalone_processor._get_document_extractor", return_value=mock_extractor):
        result = extract_from_bytes(csv_bytes, "data.csv")

    mock_extractor.extract_dataframe.assert_called_once()
    called_df = mock_extractor.extract_dataframe.call_args[0][0]
    assert isinstance(called_df, pd.DataFrame)


def test_extract_from_bytes_image():
    """Image bytes should call extract_text_from_txt on OCR output."""
    from src.api.standalone_processor import extract_from_bytes

    mock_extractor = MagicMock()
    mock_extractor.extract_text_from_txt.return_value = _make_extracted()

    # Minimal valid PNG header — we just need detect_file_type to return "image"
    png_header = b"\x89PNG\r\n\x1a\n" + b"\x00" * 200

    with (
        patch("src.api.standalone_processor._get_document_extractor", return_value=mock_extractor),
        patch("src.api.standalone_processor._ocr_image_bytes", return_value="ocr text"),
    ):
        result = extract_from_bytes(png_header, "scan.png")

    mock_extractor.extract_text_from_txt.assert_called_once()


# ---------------------------------------------------------------------------
# build_structured_prompt
# ---------------------------------------------------------------------------


def test_build_structured_prompt_for_qa_mode():
    from src.api.standalone_processor import build_structured_prompt

    result = build_structured_prompt("qa", "What is the total?", "doc text here")
    assert "system_prompt" in result
    assert "user_prompt" in result
    assert "What is the total?" in result["user_prompt"]


def test_build_structured_prompt_for_table_mode():
    from src.api.standalone_processor import build_structured_prompt

    result = build_structured_prompt("table", "Extract tables", "doc text here")
    sp = result["system_prompt"].lower()
    # The table mode system prompt should mention table or tabular extraction
    assert "table" in sp or "tabular" in sp
    assert result["user_prompt"] == "Extract tables"


def test_build_structured_prompt_for_entities_mode():
    from src.api.standalone_processor import build_structured_prompt

    result = build_structured_prompt("entities", "Find all people", "some text")
    sp = result["system_prompt"].lower()
    assert "entit" in sp or "extract" in sp
    assert result["user_prompt"] == "Find all people"


def test_build_structured_prompt_for_summary_mode():
    from src.api.standalone_processor import build_structured_prompt

    result = build_structured_prompt("summary", "Summarise this", "long text")
    sp = result["system_prompt"].lower()
    assert "summar" in sp
    assert result["user_prompt"] == "Summarise this"


def test_build_structured_prompt_with_template():
    from src.api.standalone_processor import build_structured_prompt

    mock_template = MagicMock()
    with patch(
        "src.api.standalone_processor.apply_template",
        return_value={"system_prompt": "template sys", "user_prompt": "template user"},
    ) as mock_apply:
        result = build_structured_prompt("table", "Extract invoice", "text", template=mock_template)

    mock_apply.assert_called_once_with(mock_template, "Extract invoice")
    assert result["system_prompt"] == "template sys"


# ---------------------------------------------------------------------------
# _parse_structured_response
# ---------------------------------------------------------------------------


def test_parse_structured_response_json_block():
    from src.api.standalone_processor import _parse_structured_response

    text = 'Some intro\n```json\n{"key": "value", "amount": 42}\n```\nTrailing text'
    result = _parse_structured_response(text, "table")
    assert result == {"key": "value", "amount": 42}


def test_parse_structured_response_direct_json():
    from src.api.standalone_processor import _parse_structured_response

    text = '{"entities": ["Alice", "Bob"]}'
    result = _parse_structured_response(text, "entities")
    assert result == {"entities": ["Alice", "Bob"]}


def test_parse_structured_response_braces_extraction():
    from src.api.standalone_processor import _parse_structured_response

    text = 'Here is your answer: {"summary": "short text"} end'
    result = _parse_structured_response(text, "summary")
    assert result is not None
    assert result.get("summary") == "short text"


def test_parse_structured_response_invalid_returns_none():
    from src.api.standalone_processor import _parse_structured_response

    result = _parse_structured_response("This is just plain text with no JSON.", "qa")
    assert result is None


# ---------------------------------------------------------------------------
# _build_low_confidence_reasons
# ---------------------------------------------------------------------------


def test_build_low_confidence_reasons_very_low():
    from src.api.standalone_processor import _build_low_confidence_reasons

    reasons = _build_low_confidence_reasons("some answer", 0.1)
    assert len(reasons) >= 1
    # Should mention low confidence
    combined = " ".join(reasons).lower()
    assert "confidence" in combined or "uncertain" in combined or "low" in combined


def test_build_low_confidence_reasons_medium():
    from src.api.standalone_processor import _build_low_confidence_reasons

    reasons = _build_low_confidence_reasons("an answer", 0.5)
    assert isinstance(reasons, list)


def test_build_low_confidence_reasons_high_returns_empty_or_minimal():
    from src.api.standalone_processor import _build_low_confidence_reasons

    reasons = _build_low_confidence_reasons("a good answer", 0.95)
    assert isinstance(reasons, list)


# ---------------------------------------------------------------------------
# run_intelligence
# ---------------------------------------------------------------------------


def test_run_intelligence_import_error_handled():
    """If the intelligence module is missing, should return empty dict gracefully."""
    from src.api.standalone_processor import run_intelligence

    extracted = _make_extracted()
    with patch.dict("sys.modules", {"src.intelligence.integration": None}):
        result = run_intelligence(extracted, "doc-123")
    assert isinstance(result, dict)


def test_run_intelligence_exception_handled():
    """Any exception during intelligence should be caught, returning partial/empty dict."""
    from src.api.standalone_processor import run_intelligence

    extracted = _make_extracted()
    mock_module = MagicMock()
    mock_module.process_document_intelligence.side_effect = RuntimeError("LLM unavailable")

    with patch.dict(
        "sys.modules",
        {"src.intelligence.integration": mock_module},
    ):
        result = run_intelligence(extracted, "doc-xyz")
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# chunk_and_embed
# ---------------------------------------------------------------------------


def test_chunk_and_embed_returns_chunk_count():
    from src.api.standalone_processor import chunk_and_embed

    extracted = _make_extracted()
    extracted.full_text = "This is some document content that will be chunked."

    mock_chunker = MagicMock()
    mock_chunker.chunk.return_value = [
        {"text": "chunk one", "metadata": {"page": 1}},
        {"text": "chunk two", "metadata": {"page": 2}},
    ]

    mock_model = MagicMock()
    mock_model.encode.return_value = [[0.1] * 128, [0.2] * 128]

    mock_qdrant = MagicMock()

    with (
        patch("src.api.standalone_processor.SectionChunker", return_value=mock_chunker),
        patch("src.api.standalone_processor.get_embedding_model", return_value=(mock_model, 128)),
        patch("src.api.standalone_processor.QdrantClient", return_value=mock_qdrant),
    ):
        count = chunk_and_embed(extracted, "doc-001", "col-001")

    assert count == 2
    mock_qdrant.upsert.assert_called_once()


def test_chunk_and_embed_empty_document():
    from src.api.standalone_processor import chunk_and_embed

    extracted = _make_extracted()
    extracted.full_text = ""

    mock_chunker = MagicMock()
    mock_chunker.chunk.return_value = []

    mock_model = MagicMock()
    mock_model.encode.return_value = []

    mock_qdrant = MagicMock()

    with (
        patch("src.api.standalone_processor.SectionChunker", return_value=mock_chunker),
        patch("src.api.standalone_processor.get_embedding_model", return_value=(mock_model, 128)),
        patch("src.api.standalone_processor.QdrantClient", return_value=mock_qdrant),
    ):
        count = chunk_and_embed(extracted, "doc-empty", "col-empty")

    assert count == 0


# ---------------------------------------------------------------------------
# cleanup_collection
# ---------------------------------------------------------------------------


def test_cleanup_collection_deletes():
    from src.api.standalone_processor import cleanup_collection

    mock_qdrant = MagicMock()
    with patch("src.api.standalone_processor.QdrantClient", return_value=mock_qdrant):
        cleanup_collection("tmp-col-xyz")

    mock_qdrant.delete_collection.assert_called_once_with("tmp-col-xyz")


def test_cleanup_collection_tolerates_errors():
    from src.api.standalone_processor import cleanup_collection

    mock_qdrant = MagicMock()
    mock_qdrant.delete_collection.side_effect = Exception("collection not found")
    with patch("src.api.standalone_processor.QdrantClient", return_value=mock_qdrant):
        # Should not raise
        cleanup_collection("nonexistent-col")


# ---------------------------------------------------------------------------
# retrieve_and_generate
# ---------------------------------------------------------------------------


def test_retrieve_and_generate_calls_execute_request():
    from src.api.standalone_processor import retrieve_and_generate

    mock_result = MagicMock()
    mock_result.answer = {
        "answer": "The total is $100.",
        "sources": [],
        "confidence": 0.85,
        "grounded": True,
        "context_found": True,
    }

    with patch("src.api.standalone_processor.execute_request", return_value=mock_result) as mock_exec:
        result = retrieve_and_generate(
            query="What is the total?",
            collection_name="col-001",
            subscription_id="sub-001",
        )

    mock_exec.assert_called_once()
    assert "answer" in result


# ---------------------------------------------------------------------------
# _capture_learning_signal
# ---------------------------------------------------------------------------


def test_capture_learning_signal_high_quality():
    from src.api.standalone_processor import _capture_learning_signal

    mock_store = MagicMock()
    with patch("src.api.standalone_processor.LearningSignalStore", return_value=mock_store):
        _capture_learning_signal(
            query="q",
            context="ctx",
            answer_text="ans",
            sources=[],
            confidence=0.8,
            mode="qa",
        )

    mock_store.record_high_quality.assert_called_once()
    mock_store.record_low_confidence.assert_not_called()


def test_capture_learning_signal_low_confidence():
    from src.api.standalone_processor import _capture_learning_signal

    mock_store = MagicMock()
    with patch("src.api.standalone_processor.LearningSignalStore", return_value=mock_store):
        _capture_learning_signal(
            query="q",
            context="ctx",
            answer_text="ans",
            sources=[],
            confidence=0.4,
            mode="qa",
        )

    mock_store.record_low_confidence.assert_called_once()
    mock_store.record_high_quality.assert_not_called()


# ---------------------------------------------------------------------------
# process_document (integration-level with heavy mocking)
# ---------------------------------------------------------------------------


def test_process_document_full_pipeline():
    """Smoke test: full pipeline with all external deps mocked."""
    from src.api.standalone_processor import process_document

    fake_extracted = _make_extracted()
    fake_extracted.full_text = "Invoice total: $500."

    mock_store = MagicMock()

    with (
        patch("src.api.standalone_processor.extract_from_bytes", return_value=fake_extracted),
        patch("src.api.standalone_processor.run_intelligence", return_value={}),
        patch("src.api.standalone_processor.chunk_and_embed", return_value=3),
        patch(
            "src.api.standalone_processor.retrieve_and_generate",
            return_value={
                "answer": "The total is $500.",
                "sources": [{"page": 1}],
                "confidence": 0.9,
                "grounded": True,
                "context_found": True,
            },
        ),
        patch("src.api.standalone_processor.cleanup_collection"),
        patch("src.api.standalone_processor.LearningSignalStore", return_value=mock_store),
    ):
        result = process_document(
            content=b"%PDF-1.4 fake",
            filename="invoice.pdf",
            prompt="What is the total?",
        )

    assert result["status"] == "completed"
    assert result["answer"] == "The total is $500."
    assert result["confidence"] == 0.9
    assert result["grounded"] is True


def test_process_document_persist_skips_cleanup():
    """When persist=True, cleanup_collection should not be called."""
    from src.api.standalone_processor import process_document

    fake_extracted = _make_extracted()
    mock_cleanup = MagicMock()
    mock_store = MagicMock()

    with (
        patch("src.api.standalone_processor.extract_from_bytes", return_value=fake_extracted),
        patch("src.api.standalone_processor.run_intelligence", return_value={}),
        patch("src.api.standalone_processor.chunk_and_embed", return_value=1),
        patch(
            "src.api.standalone_processor.retrieve_and_generate",
            return_value={
                "answer": "ok",
                "sources": [],
                "confidence": 0.7,
                "grounded": False,
                "context_found": True,
            },
        ),
        patch("src.api.standalone_processor.cleanup_collection", mock_cleanup),
        patch("src.api.standalone_processor.LearningSignalStore", return_value=mock_store),
    ):
        result = process_document(
            content=b"text content",
            filename="doc.txt",
            prompt="Summarise",
            persist=True,
        )

    mock_cleanup.assert_not_called()
    assert "document_id" in result
    assert result["document_id"] is not None


def test_process_document_confidence_gate():
    """When response confidence < threshold, low_confidence flag should be True."""
    from src.api.standalone_processor import process_document

    fake_extracted = _make_extracted()
    mock_store = MagicMock()

    with (
        patch("src.api.standalone_processor.extract_from_bytes", return_value=fake_extracted),
        patch("src.api.standalone_processor.run_intelligence", return_value={}),
        patch("src.api.standalone_processor.chunk_and_embed", return_value=1),
        patch(
            "src.api.standalone_processor.retrieve_and_generate",
            return_value={
                "answer": "maybe",
                "sources": [],
                "confidence": 0.3,
                "grounded": False,
                "context_found": False,
            },
        ),
        patch("src.api.standalone_processor.cleanup_collection"),
        patch("src.api.standalone_processor.LearningSignalStore", return_value=mock_store),
    ):
        result = process_document(
            content=b"text",
            filename="doc.txt",
            prompt="What is X?",
            confidence_threshold=0.5,
        )

    assert result["low_confidence"] is True
    assert len(result["low_confidence_reasons"]) >= 1


# ---------------------------------------------------------------------------
# query_persisted_document
# ---------------------------------------------------------------------------


def test_query_persisted_document_skips_extraction():
    from src.api.standalone_processor import query_persisted_document

    mock_store = MagicMock()

    with (
        patch("src.api.standalone_processor.extract_from_bytes") as mock_extract,
        patch(
            "src.api.standalone_processor.retrieve_and_generate",
            return_value={
                "answer": "persisted answer",
                "sources": [],
                "confidence": 0.8,
                "grounded": True,
                "context_found": True,
            },
        ),
        patch("src.api.standalone_processor.LearningSignalStore", return_value=mock_store),
    ):
        result = query_persisted_document(
            document_id="doc-persisted-001",
            prompt="What does it say?",
            subscription_id="sub-001",
        )

    mock_extract.assert_not_called()
    assert result["answer"] == "persisted answer"
    assert result["status"] == "completed"
