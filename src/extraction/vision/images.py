"""Image rendering + base64 helpers for the vision path."""
from __future__ import annotations

import base64
import io

import fitz


def render_pdf_page_to_png(pdf_bytes: bytes, *, page_index: int, dpi: int = 144) -> bytes:
    """Render the given page of a PDF to PNG bytes at the given DPI."""
    doc = fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
    try:
        if page_index < 0 or page_index >= len(doc):
            raise IndexError(f"page_index {page_index} out of range for PDF with {len(doc)} pages")
        page = doc[page_index]
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        return pix.tobytes("png")
    finally:
        doc.close()


def bytes_to_b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def b64_to_bytes(data: str) -> bytes:
    return base64.b64decode(data.encode("ascii"))
