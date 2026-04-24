import io

import fitz
from docx import Document

from src.extraction.adapters.dispatcher import dispatch_native
from src.extraction.adapters.errors import NotNativePathError


def _tiny_pdf() -> bytes:
    d = fitz.open()
    p = d.new_page()
    p.insert_text((72, 72), "Hello world from a native PDF document under test.")
    buf = io.BytesIO()
    d.save(buf)
    d.close()
    return buf.getvalue()


def _tiny_docx() -> bytes:
    d = Document()
    d.add_paragraph("hello")
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


def test_dispatch_native_routes_pdf_by_extension():
    result = dispatch_native(_tiny_pdf(), filename="doc.pdf", doc_id="d1")
    assert result.format == "pdf_native"


def test_dispatch_native_routes_docx_by_extension():
    result = dispatch_native(_tiny_docx(), filename="doc.docx", doc_id="d2")
    assert result.format == "docx"


def test_dispatch_native_raises_for_unknown_extension():
    import pytest
    with pytest.raises(NotNativePathError):
        dispatch_native(b"\x89PNG\r\n\x1a\n", filename="a.png", doc_id="d3")


def test_dispatch_native_raises_for_scanned_pdf():
    import pytest
    d = fitz.open()
    d.new_page()  # blank page, no text
    buf = io.BytesIO()
    d.save(buf)
    d.close()
    with pytest.raises(NotNativePathError):
        dispatch_native(buf.getvalue(), filename="scan.pdf", doc_id="d4")


def test_dispatch_native_routes_png_to_image_adapter_which_raises():
    import pytest
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20
    with pytest.raises(NotNativePathError):
        dispatch_native(png, filename="pic.png", doc_id="di1")
