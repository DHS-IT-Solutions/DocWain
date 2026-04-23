"""Native CSV adapter using stdlib csv with dialect sniffing.

Spec: docs/superpowers/specs/2026-04-23-extraction-accuracy-design.md §4.1
"""
from __future__ import annotations

import csv
import io

from src.extraction.adapters.errors import NativeAdapterError
from src.extraction.canonical_schema import (
    ExtractionResult,
    Page,
    Table,
)


def extract_csv_native(file_bytes: bytes, *, doc_id: str, filename: str) -> ExtractionResult:
    try:
        text = file_bytes.decode("utf-8", errors="replace")
    except Exception as exc:
        raise NativeAdapterError(f"failed to decode CSV {filename!r}: {exc}") from exc

    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel

    reader = csv.reader(io.StringIO(text), dialect)
    rows = [list(r) for r in reader]

    page = Page(
        page_num=1,
        blocks=[],
        tables=[Table(rows=rows, bbox=None, header_row_index=0 if rows else None)],
        images=[],
    )

    return ExtractionResult(
        doc_id=doc_id,
        format="csv",
        path_taken="native",
        pages=[page],
    )
