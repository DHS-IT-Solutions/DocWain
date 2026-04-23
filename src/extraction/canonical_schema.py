"""Canonical extraction output schema.

All extraction adapters (native + vision) produce `ExtractionResult` instances.
Downstream consumers (embedding, researcher, KG) read this shape only.

Spec: docs/superpowers/specs/2026-04-23-extraction-accuracy-design.md §4.3
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

BBox = Optional[Tuple[float, float, float, float]]  # x, y, w, h


@dataclass
class Block:
    """A text block within a page."""
    text: str
    bbox: BBox = None
    block_type: str = "paragraph"  # paragraph | heading | list_item | footnote | header | footer


@dataclass
class Table:
    """A table within a page, sheet, or slide."""
    rows: List[List[str]]
    bbox: BBox = None
    header_row_index: Optional[int] = None  # 0-based index of header row; None if no header


@dataclass
class Image:
    """An embedded image that may contain text (queued for vision sub-pass)."""
    bbox: BBox = None
    ocr_text: str = ""  # filled by later vision sub-pass; empty in Plan 1
    caption: str = ""


@dataclass
class Page:
    """A page within a PDF or image-based document."""
    page_num: int
    blocks: List[Block] = field(default_factory=list)
    tables: List[Table] = field(default_factory=list)
    images: List[Image] = field(default_factory=list)


@dataclass
class Sheet:
    """A worksheet within an XLSX/XLS document."""
    name: str
    cells: dict  # {(row, col): {"value": Any, "formula": Optional[str], "type": str}}
    hidden: bool = False
    merged_cells: List[str] = field(default_factory=list)  # e.g. ["A1:B2"]
    named_ranges: List[str] = field(default_factory=list)


@dataclass
class Slide:
    """A slide within a PPTX document."""
    slide_num: int
    elements: List[Block] = field(default_factory=list)  # text frames, shape text
    tables: List[Table] = field(default_factory=list)
    images: List[Image] = field(default_factory=list)
    notes: str = ""
    hidden: bool = False


@dataclass
class DocIntelMetadata:
    doc_type_hint: str = ""
    layout_complexity: str = "simple"  # simple | moderate | complex
    has_handwriting: bool = False
    routing_confidence: float = 1.0


@dataclass
class CoverageMetadata:
    verifier_score: float = 1.0  # native path always 1.0; vision path variable
    missed_regions: List[Any] = field(default_factory=list)
    low_confidence_regions: List[Any] = field(default_factory=list)
    fallback_invocations: List[Any] = field(default_factory=list)


@dataclass
class ExtractionMetadata:
    doc_intel: DocIntelMetadata = field(default_factory=DocIntelMetadata)
    coverage: CoverageMetadata = field(default_factory=CoverageMetadata)
    extraction_version: str = "2026-04-23-v1"


@dataclass
class ExtractionResult:
    """The canonical extraction output shape.

    One of `pages`, `sheets`, `slides` is populated depending on the source format.
    CSV uses `pages` with a single Page containing one Table.
    """
    doc_id: str
    format: str  # pdf_native | pdf_scanned | docx | xlsx | pptx | csv | image | handwritten
    path_taken: str  # native | vision | mixed
    pages: List[Page] = field(default_factory=list)
    sheets: List[Sheet] = field(default_factory=list)
    slides: List[Slide] = field(default_factory=list)
    metadata: ExtractionMetadata = field(default_factory=ExtractionMetadata)
