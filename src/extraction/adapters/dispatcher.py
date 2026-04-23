"""Native adapter dispatcher.

Routes a document to the right native-format adapter by filename extension.
Unsupported formats raise NotNativePathError; callers should delegate to the
vision path (Plan 2) or the existing fallback for those.

Spec: docs/superpowers/specs/2026-04-23-extraction-accuracy-design.md §4.1
"""
from __future__ import annotations

import os

from src.extraction.adapters.csv_native import extract_csv_native
from src.extraction.adapters.docx_native import extract_docx_native
from src.extraction.adapters.errors import NotNativePathError
from src.extraction.adapters.pdf_native import extract_pdf_native
from src.extraction.adapters.pptx_native import extract_pptx_native
from src.extraction.adapters.xlsx_native import extract_xlsx_native
from src.extraction.canonical_schema import ExtractionResult

_ADAPTERS = {
    ".pdf": extract_pdf_native,
    ".docx": extract_docx_native,
    ".xlsx": extract_xlsx_native,
    ".xls": extract_xlsx_native,
    ".pptx": extract_pptx_native,
    ".csv": extract_csv_native,
}


def dispatch_native(file_bytes: bytes, *, filename: str, doc_id: str) -> ExtractionResult:
    """Pick the native adapter for this file; raise NotNativePathError if unsupported."""
    _, ext = os.path.splitext(filename.lower())
    adapter = _ADAPTERS.get(ext)
    if adapter is None:
        raise NotNativePathError(
            f"no native adapter for extension {ext!r} (filename={filename!r})"
        )
    return adapter(file_bytes, doc_id=doc_id, filename=filename)
