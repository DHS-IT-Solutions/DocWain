"""Native PDF adapter using PyMuPDF (fitz).

Reads the PDF's text layer directly. If the document has no meaningful text layer
(scan / image-only), raises NotNativePathError so the caller can delegate to the
vision path (Plan 2) or the existing fallback.

Spec: docs/superpowers/specs/2026-04-23-extraction-accuracy-design.md §4.1
"""
from __future__ import annotations

import io

import fitz

from src.extraction.adapters.errors import NativeAdapterError, NotNativePathError
from src.extraction.canonical_schema import (
    Block,
    ExtractionResult,
    Image,
    Page,
    Table,
)

# If fewer than this many characters are extractable across the whole document,
# treat as non-native (scanned).
MIN_TEXT_CHARS_FOR_NATIVE = 10


def extract_pdf_native(file_bytes: bytes, *, doc_id: str, filename: str) -> ExtractionResult:
    try:
        doc = fitz.open(stream=io.BytesIO(file_bytes), filetype="pdf")
    except Exception as exc:
        raise NativeAdapterError(f"failed to open PDF {filename!r}: {exc}") from exc

    try:
        total_chars = sum(len(page.get_text("text")) for page in doc)
        if total_chars < MIN_TEXT_CHARS_FOR_NATIVE:
            raise NotNativePathError(
                f"PDF {filename!r} has only {total_chars} characters in text layer — "
                "likely scanned; route to vision path."
            )

        pages: list[Page] = []
        for page_index, page in enumerate(doc):
            page_num = page_index + 1

            # Text blocks via PyMuPDF's built-in block segmentation (reading order).
            blocks: list[Block] = []
            for b in page.get_text("blocks"):
                # b: (x0, y0, x1, y1, text, block_no, block_type)
                x0, y0, x1, y1, text = b[0], b[1], b[2], b[3], b[4]
                if not text.strip():
                    continue
                bbox = (float(x0), float(y0), float(x1 - x0), float(y1 - y0))
                blocks.append(Block(text=text.strip(), bbox=bbox, block_type="paragraph"))

            # Tables via PyMuPDF find_tables.
            tables: list[Table] = []
            try:
                found = page.find_tables()
                for t in found.tables:
                    rows = t.extract()
                    cleaned_rows = [[(c or "").strip() for c in row] for row in rows]
                    bx0, by0, bx1, by1 = t.bbox
                    tables.append(
                        Table(
                            rows=cleaned_rows,
                            bbox=(float(bx0), float(by0), float(bx1 - bx0), float(by1 - by0)),
                            header_row_index=0 if cleaned_rows else None,
                        )
                    )
            except Exception:
                # find_tables() can raise on exotic PDFs; absence of detected tables
                # is not an adapter failure, just a signal there were none.
                pass

            # Images (metadata only in Plan 1; vision sub-pass fills ocr_text in Plan 2).
            images: list[Image] = []
            for img_info in page.get_images(full=True):
                # img_info: (xref, smask, width, height, bpc, colorspace, alt, name, filter)
                images.append(Image(bbox=None, ocr_text="", caption=""))

            pages.append(Page(page_num=page_num, blocks=blocks, tables=tables, images=images))

        result = ExtractionResult(
            doc_id=doc_id,
            format="pdf_native",
            path_taken="native",
            pages=pages,
        )
        return result
    finally:
        doc.close()
