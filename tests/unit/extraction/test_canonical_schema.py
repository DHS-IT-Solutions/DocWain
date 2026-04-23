import json
from dataclasses import asdict

import pytest

from src.extraction.canonical_schema import (
    Block,
    CoverageMetadata,
    DocIntelMetadata,
    ExtractionMetadata,
    ExtractionResult,
    Image,
    Page,
    Sheet,
    Slide,
    Table,
)


def test_extraction_result_roundtrips_through_json():
    result = ExtractionResult(
        doc_id="doc-1",
        format="pdf_native",
        path_taken="native",
        pages=[
            Page(
                page_num=1,
                blocks=[Block(text="hello", bbox=None, block_type="paragraph")],
                tables=[Table(rows=[["a", "b"], ["1", "2"]], bbox=None, header_row_index=0)],
                images=[Image(bbox=None, ocr_text="", caption="")],
            )
        ],
        sheets=[],
        slides=[],
        metadata=ExtractionMetadata(
            doc_intel=DocIntelMetadata(
                doc_type_hint="invoice",
                layout_complexity="simple",
                has_handwriting=False,
                routing_confidence=0.9,
            ),
            coverage=CoverageMetadata(
                verifier_score=1.0,
                missed_regions=[],
                low_confidence_regions=[],
                fallback_invocations=[],
            ),
            extraction_version="2026-04-23-v1",
        ),
    )

    payload = json.dumps(asdict(result))
    data = json.loads(payload)

    assert data["doc_id"] == "doc-1"
    assert data["pages"][0]["blocks"][0]["text"] == "hello"
    assert data["pages"][0]["tables"][0]["rows"] == [["a", "b"], ["1", "2"]]
    assert data["metadata"]["doc_intel"]["doc_type_hint"] == "invoice"


def test_sheet_preserves_hidden_flag():
    sheet = Sheet(name="Hidden", cells={}, hidden=True, merged_cells=[], named_ranges=[])
    assert sheet.hidden is True


def test_slide_preserves_hidden_flag():
    slide = Slide(slide_num=1, elements=[], notes="", hidden=True)
    assert slide.hidden is True
