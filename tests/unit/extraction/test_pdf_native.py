import io

import fitz  # PyMuPDF
import pytest

from src.extraction.adapters.errors import NotNativePathError
from src.extraction.adapters.pdf_native import extract_pdf_native
from src.extraction.canonical_schema import ExtractionResult


def _make_pdf_with_text(pages_text: list[str]) -> bytes:
    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


def _make_scanned_pdf_without_text_layer(num_pages: int = 1) -> bytes:
    """Create a PDF with only an image, no text layer (simulates scan)."""
    doc = fitz.open()
    for _ in range(num_pages):
        page = doc.new_page()
        # Insert a blank rectangle so the page isn't completely empty but has no text.
        page.draw_rect(fitz.Rect(72, 72, 200, 200), color=(0.5, 0.5, 0.5))
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


def test_pdf_native_extracts_single_page_text():
    pdf = _make_pdf_with_text(["Hello world"])
    result = extract_pdf_native(pdf, doc_id="d1", filename="t.pdf")
    assert isinstance(result, ExtractionResult)
    assert result.format == "pdf_native"
    assert result.path_taken == "native"
    assert len(result.pages) == 1
    assert result.pages[0].page_num == 1
    assert any("Hello world" in b.text for b in result.pages[0].blocks)


def test_pdf_native_extracts_multi_page():
    pdf = _make_pdf_with_text(["Page one text", "Page two text", "Page three text"])
    result = extract_pdf_native(pdf, doc_id="d2", filename="multi.pdf")
    assert len(result.pages) == 3
    assert any("Page one" in b.text for b in result.pages[0].blocks)
    assert any("Page two" in b.text for b in result.pages[1].blocks)
    assert any("Page three" in b.text for b in result.pages[2].blocks)


def test_pdf_native_raises_on_scanned_pdf():
    pdf = _make_scanned_pdf_without_text_layer()
    with pytest.raises(NotNativePathError):
        extract_pdf_native(pdf, doc_id="d3", filename="scan.pdf")
