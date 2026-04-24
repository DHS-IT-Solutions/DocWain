import io

import fitz

from src.extraction.vision.images import (
    b64_to_bytes,
    bytes_to_b64,
    render_pdf_page_to_png,
)


def _make_pdf(num_pages: int = 2) -> bytes:
    d = fitz.open()
    for _ in range(num_pages):
        p = d.new_page()
        p.insert_text((72, 72), "page content")
    buf = io.BytesIO()
    d.save(buf)
    d.close()
    return buf.getvalue()


def test_render_pdf_page_returns_png_bytes():
    pdf = _make_pdf()
    png = render_pdf_page_to_png(pdf, page_index=0, dpi=72)
    assert png.startswith(b"\x89PNG\r\n\x1a\n")


def test_render_pdf_page_index_in_range():
    pdf = _make_pdf(num_pages=3)
    png2 = render_pdf_page_to_png(pdf, page_index=2, dpi=72)
    assert png2.startswith(b"\x89PNG\r\n\x1a\n")


def test_bytes_to_b64_roundtrips():
    original = b"\x00\x01\x02\x03" * 4
    b64 = bytes_to_b64(original)
    restored = b64_to_bytes(b64)
    assert restored == original
