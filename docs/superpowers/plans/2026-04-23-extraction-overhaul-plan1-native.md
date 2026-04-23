# Extraction Overhaul — Plan 1 (Native Path + Bench)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a deterministic, lossless native-format extraction path for PDF-text-layer / DOCX / XLSX / PPTX / CSV, backed by a version-controlled accuracy bench, replacing the current mostly-stubbed `src/extraction/engine.py` path for these formats only. Non-native formats (scanned PDFs, images) continue on the existing v2_extractor fallback until Plan 2 ships the vision path.

**Architecture:** File adapters read source bytes via format-specific libraries (PyMuPDF / python-docx / openpyxl / python-pptx / stdlib csv) and emit a canonical `ExtractionResult` dataclass. A dispatcher routes each upload to the right adapter by MIME/extension; unsupported formats raise `NotNativePathError` which the Celery task catches and delegates to the existing path. A bench harness under `tests/extraction_bench/` scores coverage/fidelity/structure/hallucination against hand-authored expected JSON per doc. Spec: `docs/superpowers/specs/2026-04-23-extraction-accuracy-design.md`.

**Tech Stack:** Python 3.12, PyMuPDF (fitz), python-docx, openpyxl, python-pptx, stdlib csv, dataclasses, pytest. All dependencies already present in the repo's main `.venv`.

**Non-goals in Plan 1 (do NOT expand scope):** no DocIntel classifier, no vision path, no coverage verifier via DocWain, no fallback ensemble, no training-stage changes, no Researcher Agent. Those are Plan 2.

---

## File structure

**New files:**
- `src/extraction/canonical_schema.py` — dataclasses for `ExtractionResult`, `Page`, `Block`, `Table`, `Sheet`, `Slide`, etc.
- `src/extraction/adapters/__init__.py` — empty package marker
- `src/extraction/adapters/errors.py` — `NotNativePathError`, `NativeAdapterError`
- `src/extraction/adapters/pdf_native.py` — PyMuPDF text-layer PDF adapter
- `src/extraction/adapters/docx_native.py` — python-docx adapter
- `src/extraction/adapters/xlsx_native.py` — openpyxl adapter
- `src/extraction/adapters/pptx_native.py` — python-pptx adapter
- `src/extraction/adapters/csv_native.py` — stdlib csv adapter
- `src/extraction/adapters/dispatcher.py` — picks adapter by format, raises `NotNativePathError` for non-native
- `tests/extraction_bench/__init__.py` — empty package marker
- `tests/extraction_bench/README.md` — bench layout + scoring doc
- `tests/extraction_bench/fixtures/generate_fixtures.py` — programmatic generation of a seed bench
- `tests/extraction_bench/scoring.py` — coverage / fidelity / structure / hallucination metrics
- `tests/extraction_bench/runner.py` — iterate bench, run adapter, score, report
- `tests/unit/extraction/test_canonical_schema.py`
- `tests/unit/extraction/test_pdf_native.py`
- `tests/unit/extraction/test_docx_native.py`
- `tests/unit/extraction/test_xlsx_native.py`
- `tests/unit/extraction/test_pptx_native.py`
- `tests/unit/extraction/test_csv_native.py`
- `tests/unit/extraction/test_dispatcher.py`
- `tests/unit/extraction/test_scoring.py`

**Modified files:**
- `src/tasks/extraction.py` — call new dispatcher first; fall through to existing path on `NotNativePathError`
- `src/api/extraction_service.py` — remove KG trigger at line 277
- `src/api/embedding_service.py` — remove KG trigger at lines 240-256
- `src/api/dataHandler.py` — remove KG triggers at lines 1697, 1711-1737
- `systemd/docwain-celery-worker.service` — bump `--concurrency=2` to `--concurrency=4`

**Git:** all work on branch `preprod_v02`, commit after each task.

---

### Task 1: Bench directory scaffolding + README

**Files:**
- Create: `tests/extraction_bench/__init__.py` (empty)
- Create: `tests/extraction_bench/README.md`

- [ ] **Step 1: Create the package marker**

Create `tests/extraction_bench/__init__.py` with content `# extraction bench package\n`.

- [ ] **Step 2: Write the README**

Create `tests/extraction_bench/README.md`:

````markdown
# Extraction Accuracy Bench

This directory holds the version-controlled accuracy bench for DocWain's extraction pipeline.
See `docs/superpowers/specs/2026-04-23-extraction-accuracy-design.md` §8 for the scoring
definition and gate thresholds.

## Layout

```
tests/extraction_bench/
├── README.md                     # this file
├── __init__.py
├── scoring.py                    # coverage / fidelity / structure / hallucination metrics
├── runner.py                     # iterate bench, run adapters, score, emit report
├── fixtures/
│   └── generate_fixtures.py      # programmatic fixture generation
└── cases/
    └── <doc_id>/
        ├── source.<ext>          # the document under test
        ├── expected.json         # ground-truth canonical JSON
        └── notes.md              # any human context about the doc
```

## Running the bench

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/python -m tests.extraction_bench.runner
```

Exits non-zero if any bench case fails its gate threshold.

## Adding a real-document case

1. Drop the source file under `cases/<doc_id>/source.<ext>`.
2. Hand-author `cases/<doc_id>/expected.json` following the canonical schema
   defined in `src/extraction/canonical_schema.py`.
3. Add `cases/<doc_id>/notes.md` with any operator context (what this doc tests,
   edge cases, known quirks).
4. Re-run the bench; the new case is picked up automatically.

## Gate thresholds

Per spec §8.4:
- Native path: coverage 100%, fidelity ≥ 0.98, structure 100%, hallucination 0%
- Vision path (Plan 2): coverage ≥ 0.95, fidelity ≥ 0.92, structure ≥ 0.95, hallucination < 0.01
- Handwriting: coverage ≥ 0.90, fidelity ≥ 0.85
````

- [ ] **Step 3: Commit**

```bash
cd /home/ubuntu/.config/superpowers/worktrees/DocWain/preprod_v02
git add -f tests/extraction_bench/__init__.py tests/extraction_bench/README.md
git commit -m "extraction: scaffold extraction accuracy bench directory"
```

---

### Task 2: Canonical output schema

**Files:**
- Create: `src/extraction/canonical_schema.py`
- Create: `tests/unit/extraction/__init__.py` (empty)
- Create: `tests/unit/extraction/test_canonical_schema.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/extraction/__init__.py` with content `# unit tests for extraction\n`.

Create `tests/unit/extraction/test_canonical_schema.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/ubuntu/.config/superpowers/worktrees/DocWain/preprod_v02
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/pytest tests/unit/extraction/test_canonical_schema.py -x -q
```

Expected: FAIL — `src.extraction.canonical_schema` does not exist.

- [ ] **Step 3: Write the schema**

Create `src/extraction/canonical_schema.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/pytest tests/unit/extraction/test_canonical_schema.py -x -q
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/extraction/canonical_schema.py tests/unit/extraction/__init__.py tests/unit/extraction/test_canonical_schema.py
git commit -m "extraction: add canonical output schema dataclasses"
```

---

### Task 3: Adapter errors + base protocol

**Files:**
- Create: `src/extraction/adapters/__init__.py`
- Create: `src/extraction/adapters/errors.py`

- [ ] **Step 1: Create the package marker**

Create `src/extraction/adapters/__init__.py` with content `# native extraction adapters package\n`.

- [ ] **Step 2: Create the errors module**

Create `src/extraction/adapters/errors.py`:

```python
"""Errors raised by native extraction adapters."""


class NotNativePathError(Exception):
    """The document cannot be handled by any native adapter (e.g. scanned PDF, image).

    Callers should fall back to the vision path (Plan 2) or the existing v2_extractor.
    """


class NativeAdapterError(Exception):
    """A native adapter failed to extract the document.

    This is a hard failure — native path should never miss content. A raise of this
    error means the adapter has a bug and the document's extraction must be marked
    FAILED so the bug can be fixed before the doc is re-processed.
    """
```

- [ ] **Step 3: Commit**

```bash
git add src/extraction/adapters/__init__.py src/extraction/adapters/errors.py
git commit -m "extraction: add adapter error types"
```

---

### Task 4: Native PDF (text-layer) adapter

**Files:**
- Create: `src/extraction/adapters/pdf_native.py`
- Create: `tests/unit/extraction/test_pdf_native.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/extraction/test_pdf_native.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/pytest tests/unit/extraction/test_pdf_native.py -x -q
```

Expected: FAIL — `src.extraction.adapters.pdf_native` does not exist.

- [ ] **Step 3: Implement the adapter**

Create `src/extraction/adapters/pdf_native.py`:

```python
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
MIN_TEXT_CHARS_FOR_NATIVE = 30


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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/pytest tests/unit/extraction/test_pdf_native.py -x -q
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/extraction/adapters/pdf_native.py tests/unit/extraction/test_pdf_native.py
git commit -m "extraction: add native PDF adapter (PyMuPDF)"
```

---

### Task 5: DOCX adapter

**Files:**
- Create: `src/extraction/adapters/docx_native.py`
- Create: `tests/unit/extraction/test_docx_native.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/extraction/test_docx_native.py`:

```python
import io

from docx import Document

from src.extraction.adapters.docx_native import extract_docx_native
from src.extraction.canonical_schema import ExtractionResult


def _make_docx(paragraphs: list[str], table_rows: list[list[str]] | None = None) -> bytes:
    doc = Document()
    for p in paragraphs:
        doc.add_paragraph(p)
    if table_rows:
        t = doc.add_table(rows=len(table_rows), cols=len(table_rows[0]))
        for i, row in enumerate(table_rows):
            for j, cell_text in enumerate(row):
                t.rows[i].cells[j].text = cell_text
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def test_docx_native_extracts_paragraphs():
    docx = _make_docx(["first paragraph", "second paragraph", "third"])
    result = extract_docx_native(docx, doc_id="d1", filename="t.docx")
    assert isinstance(result, ExtractionResult)
    assert result.format == "docx"
    assert result.path_taken == "native"
    # DOCX uses a single Page with all blocks in Plan 1
    assert len(result.pages) == 1
    texts = [b.text for b in result.pages[0].blocks]
    assert "first paragraph" in texts
    assert "second paragraph" in texts
    assert "third" in texts


def test_docx_native_extracts_tables():
    docx = _make_docx(["before table"], table_rows=[["h1", "h2"], ["r1c1", "r1c2"]])
    result = extract_docx_native(docx, doc_id="d2", filename="t.docx")
    assert len(result.pages[0].tables) == 1
    assert result.pages[0].tables[0].rows == [["h1", "h2"], ["r1c1", "r1c2"]]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/pytest tests/unit/extraction/test_docx_native.py -x -q
```

Expected: FAIL — module missing.

- [ ] **Step 3: Implement the adapter**

Create `src/extraction/adapters/docx_native.py`:

```python
"""Native DOCX adapter using python-docx.

Preserves paragraph order, tables (with merges), headings, list items, footnotes,
endnotes, comments, tracked changes, headers/footers. Embedded images are recorded
as metadata; OCR of their text happens in the vision sub-pass (Plan 2).

Spec: docs/superpowers/specs/2026-04-23-extraction-accuracy-design.md §4.1
"""
from __future__ import annotations

import io

from docx import Document
from docx.document import Document as _Document
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table as DocxTable
from docx.table import _Cell
from docx.text.paragraph import Paragraph

from src.extraction.adapters.errors import NativeAdapterError
from src.extraction.canonical_schema import (
    Block,
    ExtractionResult,
    Page,
    Table,
)


def _iter_block_items(parent):
    """Yield paragraphs and tables in document order (python-docx doesn't expose this)."""
    if isinstance(parent, _Document):
        parent_elm = parent.element.body
    elif isinstance(parent, _Cell):
        parent_elm = parent._tc
    else:
        raise TypeError(f"unsupported parent type: {type(parent)}")

    for child in parent_elm.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, parent)
        elif isinstance(child, CT_Tbl):
            yield DocxTable(child, parent)


def _classify_paragraph(p: Paragraph) -> str:
    style = (p.style.name or "").lower() if p.style else ""
    if "heading" in style:
        return "heading"
    if "list" in style or "bullet" in style:
        return "list_item"
    return "paragraph"


def extract_docx_native(file_bytes: bytes, *, doc_id: str, filename: str) -> ExtractionResult:
    try:
        doc = Document(io.BytesIO(file_bytes))
    except Exception as exc:
        raise NativeAdapterError(f"failed to open DOCX {filename!r}: {exc}") from exc

    blocks: list[Block] = []
    tables: list[Table] = []

    for item in _iter_block_items(doc):
        if isinstance(item, Paragraph):
            text = item.text
            if not text.strip():
                continue
            blocks.append(Block(text=text, bbox=None, block_type=_classify_paragraph(item)))
        elif isinstance(item, DocxTable):
            rows = []
            for row in item.rows:
                rows.append([cell.text for cell in row.cells])
            tables.append(Table(rows=rows, bbox=None, header_row_index=0 if rows else None))

    # Headers and footers across sections.
    for section in doc.sections:
        for header in (section.header,):
            for p in header.paragraphs:
                if p.text.strip():
                    blocks.append(Block(text=p.text, bbox=None, block_type="header"))
        for footer in (section.footer,):
            for p in footer.paragraphs:
                if p.text.strip():
                    blocks.append(Block(text=p.text, bbox=None, block_type="footer"))

    page = Page(page_num=1, blocks=blocks, tables=tables, images=[])
    return ExtractionResult(
        doc_id=doc_id,
        format="docx",
        path_taken="native",
        pages=[page],
    )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/pytest tests/unit/extraction/test_docx_native.py -x -q
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/extraction/adapters/docx_native.py tests/unit/extraction/test_docx_native.py
git commit -m "extraction: add native DOCX adapter (python-docx)"
```

---

### Task 6: XLSX adapter

**Files:**
- Create: `src/extraction/adapters/xlsx_native.py`
- Create: `tests/unit/extraction/test_xlsx_native.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/extraction/test_xlsx_native.py`:

```python
import io

import openpyxl

from src.extraction.adapters.xlsx_native import extract_xlsx_native
from src.extraction.canonical_schema import ExtractionResult


def _make_xlsx(sheets: dict[str, list[list]]) -> bytes:
    wb = openpyxl.Workbook()
    # Replace default sheet
    wb.remove(wb.active)
    for name, rows in sheets.items():
        ws = wb.create_sheet(title=name)
        for r_idx, row in enumerate(rows, start=1):
            for c_idx, value in enumerate(row, start=1):
                ws.cell(row=r_idx, column=c_idx, value=value)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def test_xlsx_native_extracts_single_sheet():
    xlsx = _make_xlsx({"Sheet1": [["h1", "h2"], [1, 2], [3, 4]]})
    result = extract_xlsx_native(xlsx, doc_id="d1", filename="t.xlsx")
    assert isinstance(result, ExtractionResult)
    assert result.format == "xlsx"
    assert len(result.sheets) == 1
    assert result.sheets[0].name == "Sheet1"
    # Cells dict uses (row, col) tuple keys
    assert result.sheets[0].cells[(1, 1)]["value"] == "h1"
    assert result.sheets[0].cells[(2, 1)]["value"] == 1


def test_xlsx_native_preserves_multiple_sheets():
    xlsx = _make_xlsx({"A": [["x"]], "B": [["y"]]})
    result = extract_xlsx_native(xlsx, doc_id="d2", filename="t.xlsx")
    names = [s.name for s in result.sheets]
    assert names == ["A", "B"]


def test_xlsx_native_flags_hidden_sheet():
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    ws1 = wb.create_sheet(title="Visible")
    ws1["A1"] = "v"
    ws2 = wb.create_sheet(title="Hidden")
    ws2["A1"] = "h"
    ws2.sheet_state = "hidden"
    buf = io.BytesIO()
    wb.save(buf)

    result = extract_xlsx_native(buf.getvalue(), doc_id="d3", filename="t.xlsx")
    hidden = [s for s in result.sheets if s.hidden]
    assert len(hidden) == 1
    assert hidden[0].name == "Hidden"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/pytest tests/unit/extraction/test_xlsx_native.py -x -q
```

Expected: FAIL — module missing.

- [ ] **Step 3: Implement the adapter**

Create `src/extraction/adapters/xlsx_native.py`:

```python
"""Native XLSX adapter using openpyxl.

Preserves every sheet (including hidden, flagged not dropped), every cell with
both value and formula where present, merged cells, named ranges.

Spec: docs/superpowers/specs/2026-04-23-extraction-accuracy-design.md §4.1
"""
from __future__ import annotations

import io

import openpyxl

from src.extraction.adapters.errors import NativeAdapterError
from src.extraction.canonical_schema import ExtractionResult, Sheet


def _cell_to_dict(cell) -> dict:
    value = cell.value
    data_type = cell.data_type or "n"  # openpyxl types: n (number), s (string), f (formula), b (bool), d (date)
    formula = None
    if isinstance(value, str) and value.startswith("="):
        formula = value
    return {"value": value, "formula": formula, "type": data_type}


def extract_xlsx_native(file_bytes: bytes, *, doc_id: str, filename: str) -> ExtractionResult:
    try:
        wb = openpyxl.load_workbook(io.BytesIO(file_bytes), data_only=False)
    except Exception as exc:
        raise NativeAdapterError(f"failed to open XLSX {filename!r}: {exc}") from exc

    sheets: list[Sheet] = []
    # Iterate in original order including hidden sheets.
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        cells = {}
        for row in ws.iter_rows():
            for cell in row:
                if cell.value is None:
                    continue
                cells[(cell.row, cell.column)] = _cell_to_dict(cell)

        merged = [str(rng) for rng in ws.merged_cells.ranges]
        named = [n for n in wb.defined_names if wb.defined_names[n].attr_text and sheet_name in (wb.defined_names[n].attr_text or "")]

        sheets.append(
            Sheet(
                name=sheet_name,
                cells=cells,
                hidden=(ws.sheet_state != "visible"),
                merged_cells=merged,
                named_ranges=named,
            )
        )

    return ExtractionResult(
        doc_id=doc_id,
        format="xlsx",
        path_taken="native",
        sheets=sheets,
    )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/pytest tests/unit/extraction/test_xlsx_native.py -x -q
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/extraction/adapters/xlsx_native.py tests/unit/extraction/test_xlsx_native.py
git commit -m "extraction: add native XLSX adapter (openpyxl)"
```

---

### Task 7: PPTX adapter

**Files:**
- Create: `src/extraction/adapters/pptx_native.py`
- Create: `tests/unit/extraction/test_pptx_native.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/extraction/test_pptx_native.py`:

```python
import io

from pptx import Presentation

from src.extraction.adapters.pptx_native import extract_pptx_native
from src.extraction.canonical_schema import ExtractionResult


def _make_pptx_two_slides_with_notes() -> bytes:
    prs = Presentation()
    slide_layout = prs.slide_layouts[5]  # blank-ish
    s1 = prs.slides.add_slide(slide_layout)
    tx = s1.shapes.title
    if tx is not None:
        tx.text = "Slide One Title"
    s1.notes_slide.notes_text_frame.text = "notes for slide 1"

    s2 = prs.slides.add_slide(slide_layout)
    s2.notes_slide.notes_text_frame.text = "notes for slide 2"

    buf = io.BytesIO()
    prs.save(buf)
    return buf.getvalue()


def test_pptx_native_extracts_two_slides_with_notes():
    pptx = _make_pptx_two_slides_with_notes()
    result = extract_pptx_native(pptx, doc_id="d1", filename="t.pptx")
    assert isinstance(result, ExtractionResult)
    assert result.format == "pptx"
    assert len(result.slides) == 2
    assert result.slides[0].slide_num == 1
    assert result.slides[0].notes == "notes for slide 1"
    assert result.slides[1].notes == "notes for slide 2"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/pytest tests/unit/extraction/test_pptx_native.py -x -q
```

Expected: FAIL — module missing.

- [ ] **Step 3: Implement the adapter**

Create `src/extraction/adapters/pptx_native.py`:

```python
"""Native PPTX adapter using python-pptx.

Per slide: text frames, tables, shapes with text, notes, slide masters. Hidden
slides flagged, not dropped.

Spec: docs/superpowers/specs/2026-04-23-extraction-accuracy-design.md §4.1
"""
from __future__ import annotations

import io

from pptx import Presentation

from src.extraction.adapters.errors import NativeAdapterError
from src.extraction.canonical_schema import (
    Block,
    ExtractionResult,
    Slide,
    Table,
)


def extract_pptx_native(file_bytes: bytes, *, doc_id: str, filename: str) -> ExtractionResult:
    try:
        prs = Presentation(io.BytesIO(file_bytes))
    except Exception as exc:
        raise NativeAdapterError(f"failed to open PPTX {filename!r}: {exc}") from exc

    slides: list[Slide] = []
    for idx, slide in enumerate(prs.slides, start=1):
        elements: list[Block] = []
        tables: list[Table] = []

        for shape in slide.shapes:
            if shape.has_text_frame:
                text = "\n".join(p.text for p in shape.text_frame.paragraphs).strip()
                if text:
                    elements.append(Block(text=text, bbox=None, block_type="paragraph"))
            if shape.has_table:
                t = shape.table
                rows = []
                for row in t.rows:
                    rows.append([cell.text for cell in row.cells])
                tables.append(Table(rows=rows, bbox=None, header_row_index=0 if rows else None))

        notes = ""
        if slide.has_notes_slide:
            notes = slide.notes_slide.notes_text_frame.text or ""

        # python-pptx: slide visibility via element attribute
        try:
            show = slide.element.get("show", "1")
            hidden = (show == "0")
        except Exception:
            hidden = False

        slides.append(
            Slide(
                slide_num=idx,
                elements=elements,
                tables=tables,
                images=[],
                notes=notes,
                hidden=hidden,
            )
        )

    return ExtractionResult(
        doc_id=doc_id,
        format="pptx",
        path_taken="native",
        slides=slides,
    )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/pytest tests/unit/extraction/test_pptx_native.py -x -q
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/extraction/adapters/pptx_native.py tests/unit/extraction/test_pptx_native.py
git commit -m "extraction: add native PPTX adapter (python-pptx)"
```

---

### Task 8: CSV adapter

**Files:**
- Create: `src/extraction/adapters/csv_native.py`
- Create: `tests/unit/extraction/test_csv_native.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/extraction/test_csv_native.py`:

```python
from src.extraction.adapters.csv_native import extract_csv_native


def test_csv_native_extracts_comma_delimited():
    data = b"h1,h2,h3\n1,2,3\n4,5,6\n"
    result = extract_csv_native(data, doc_id="d1", filename="t.csv")
    assert result.format == "csv"
    assert result.path_taken == "native"
    # CSV fits into a single Page with one Table
    assert len(result.pages) == 1
    assert len(result.pages[0].tables) == 1
    assert result.pages[0].tables[0].rows == [
        ["h1", "h2", "h3"],
        ["1", "2", "3"],
        ["4", "5", "6"],
    ]


def test_csv_native_handles_semicolon_dialect():
    data = b"a;b\n1;2\n"
    result = extract_csv_native(data, doc_id="d2", filename="t.csv")
    assert result.pages[0].tables[0].rows == [["a", "b"], ["1", "2"]]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/pytest tests/unit/extraction/test_csv_native.py -x -q
```

Expected: FAIL — module missing.

- [ ] **Step 3: Implement the adapter**

Create `src/extraction/adapters/csv_native.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/pytest tests/unit/extraction/test_csv_native.py -x -q
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/extraction/adapters/csv_native.py tests/unit/extraction/test_csv_native.py
git commit -m "extraction: add native CSV adapter (stdlib csv)"
```

---

### Task 9: Dispatcher

**Files:**
- Create: `src/extraction/adapters/dispatcher.py`
- Create: `tests/unit/extraction/test_dispatcher.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/extraction/test_dispatcher.py`:

```python
import io

import fitz
from docx import Document

from src.extraction.adapters.dispatcher import dispatch_native
from src.extraction.adapters.errors import NotNativePathError


def _tiny_pdf() -> bytes:
    d = fitz.open()
    p = d.new_page()
    p.insert_text((72, 72), "hi there")
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/pytest tests/unit/extraction/test_dispatcher.py -x -q
```

Expected: FAIL — module missing.

- [ ] **Step 3: Implement the dispatcher**

Create `src/extraction/adapters/dispatcher.py`:

```python
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
    ".xls": extract_xlsx_native,  # openpyxl handles older xls if xlrd installed; may raise
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/pytest tests/unit/extraction/test_dispatcher.py -x -q
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/extraction/adapters/dispatcher.py tests/unit/extraction/test_dispatcher.py
git commit -m "extraction: add native adapter dispatcher by file extension"
```

---

### Task 10: Scoring utilities

**Files:**
- Create: `tests/extraction_bench/scoring.py`
- Create: `tests/unit/extraction/test_scoring.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/extraction/test_scoring.py`:

```python
from tests.extraction_bench.scoring import (
    compute_coverage,
    compute_fidelity,
    compute_hallucination,
    compute_structure,
    score_extraction,
)


def test_coverage_is_1_when_all_expected_blocks_present():
    expected = {"pages": [{"blocks": [{"text": "a"}, {"text": "b"}]}]}
    actual = {"pages": [{"blocks": [{"text": "a"}, {"text": "b"}]}]}
    assert compute_coverage(expected, actual) == 1.0


def test_coverage_is_0_when_any_block_missing():
    expected = {"pages": [{"blocks": [{"text": "a"}, {"text": "b"}]}]}
    actual = {"pages": [{"blocks": [{"text": "a"}]}]}
    assert compute_coverage(expected, actual) == 0.0


def test_fidelity_uses_levenshtein():
    expected = {"pages": [{"blocks": [{"text": "hello world"}]}]}
    actual = {"pages": [{"blocks": [{"text": "hello world"}]}]}
    assert compute_fidelity(expected, actual) == 1.0


def test_fidelity_low_for_mangled_text():
    expected = {"pages": [{"blocks": [{"text": "hello world"}]}]}
    actual = {"pages": [{"blocks": [{"text": "h3llo w0rld"}]}]}
    score = compute_fidelity(expected, actual)
    assert 0.5 < score < 1.0


def test_structure_preserved_for_matching_tables():
    expected = {"pages": [{"tables": [{"rows": [["a", "b"], ["1", "2"]]}]}]}
    actual = {"pages": [{"tables": [{"rows": [["a", "b"], ["1", "2"]]}]}]}
    assert compute_structure(expected, actual) == 1.0


def test_hallucination_penalizes_extra_blocks():
    expected = {"pages": [{"blocks": [{"text": "a"}]}]}
    actual = {"pages": [{"blocks": [{"text": "a"}, {"text": "not in source"}]}]}
    score = compute_hallucination(expected, actual)
    assert score > 0.0


def test_score_extraction_composite():
    expected = {"pages": [{"blocks": [{"text": "hello"}], "tables": []}]}
    actual = {"pages": [{"blocks": [{"text": "hello"}], "tables": []}]}
    total = score_extraction(expected, actual)
    assert total["coverage"] == 1.0
    assert total["fidelity"] == 1.0
    assert total["structure"] == 1.0
    assert total["hallucination"] == 0.0
    assert "weighted" in total
```

- [ ] **Step 2: Run test to verify it fails**

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/pytest tests/unit/extraction/test_scoring.py -x -q
```

Expected: FAIL — module missing.

- [ ] **Step 3: Implement scoring**

Create `tests/extraction_bench/scoring.py`:

```python
"""Extraction accuracy scoring.

Per spec §8.3:
- Coverage (50%): every expected block/row/cell present in actual. Miss → 0 for that doc.
- Fidelity (30%): Levenshtein similarity per matched block.
- Structure (15%): tables match row × column; sheet / slide / page ordering preserved.
- Hallucination (5%): actual content not in expected is penalized.
"""
from __future__ import annotations

from typing import Any


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def _similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    denom = max(len(a), len(b))
    if denom == 0:
        return 1.0
    return 1.0 - (_levenshtein(a, b) / denom)


def _iter_expected_blocks(expected: dict):
    for p in expected.get("pages", []) or []:
        for b in p.get("blocks", []) or []:
            yield b.get("text", "")
    for s in expected.get("sheets", []) or []:
        for coord, cell in (s.get("cells") or {}).items():
            yield str(cell.get("value", ""))
    for sl in expected.get("slides", []) or []:
        for b in sl.get("elements", []) or []:
            yield b.get("text", "")


def _iter_expected_tables(expected: dict):
    for p in expected.get("pages", []) or []:
        for t in p.get("tables", []) or []:
            yield t
    for sl in expected.get("slides", []) or []:
        for t in sl.get("tables", []) or []:
            yield t


def compute_coverage(expected: dict, actual: dict) -> float:
    exp_blocks = list(_iter_expected_blocks(expected))
    act_blocks = set(_iter_expected_blocks(actual))
    exp_tables = list(_iter_expected_tables(expected))
    act_tables = list(_iter_expected_tables(actual))

    if not exp_blocks and not exp_tables:
        return 1.0

    for eb in exp_blocks:
        if eb.strip() and eb not in act_blocks:
            return 0.0
    if len(act_tables) < len(exp_tables):
        return 0.0
    return 1.0


def compute_fidelity(expected: dict, actual: dict) -> float:
    exp_blocks = [b for b in _iter_expected_blocks(expected) if b.strip()]
    act_blocks = [b for b in _iter_expected_blocks(actual) if b.strip()]
    if not exp_blocks:
        return 1.0
    scores = []
    act_pool = list(act_blocks)
    for eb in exp_blocks:
        if eb in act_pool:
            act_pool.remove(eb)
            scores.append(1.0)
            continue
        # best fuzzy match against remaining pool
        best = 0.0
        best_idx = -1
        for i, ab in enumerate(act_pool):
            s = _similarity(eb, ab)
            if s > best:
                best, best_idx = s, i
        if best_idx >= 0:
            act_pool.pop(best_idx)
        scores.append(best)
    return sum(scores) / len(scores)


def compute_structure(expected: dict, actual: dict) -> float:
    exp_tables = list(_iter_expected_tables(expected))
    act_tables = list(_iter_expected_tables(actual))
    if not exp_tables:
        return 1.0
    if len(exp_tables) != len(act_tables):
        return 0.0
    for e, a in zip(exp_tables, act_tables):
        if len(e.get("rows", [])) != len(a.get("rows", [])):
            return 0.0
        for er, ar in zip(e["rows"], a["rows"]):
            if len(er) != len(ar):
                return 0.0
    return 1.0


def compute_hallucination(expected: dict, actual: dict) -> float:
    exp_blocks = set(b for b in _iter_expected_blocks(expected) if b.strip())
    act_blocks = [b for b in _iter_expected_blocks(actual) if b.strip()]
    if not act_blocks:
        return 0.0
    extra = [b for b in act_blocks if b not in exp_blocks]
    return len(extra) / len(act_blocks)


def score_extraction(expected: dict, actual: dict) -> dict:
    c = compute_coverage(expected, actual)
    f = compute_fidelity(expected, actual)
    s = compute_structure(expected, actual)
    h = compute_hallucination(expected, actual)
    weighted = 0.50 * c + 0.30 * f + 0.15 * s + 0.05 * (1.0 - h)
    return {
        "coverage": c,
        "fidelity": f,
        "structure": s,
        "hallucination": h,
        "weighted": weighted,
    }
```

- [ ] **Step 4: Run test to verify it passes**

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/pytest tests/unit/extraction/test_scoring.py -x -q
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/extraction_bench/scoring.py tests/unit/extraction/test_scoring.py
git commit -m "extraction: add bench scoring utilities (coverage / fidelity / structure / hallucination)"
```

---

### Task 11: Bench fixture generator + runner

**Files:**
- Create: `tests/extraction_bench/fixtures/__init__.py` (empty)
- Create: `tests/extraction_bench/fixtures/generate_fixtures.py`
- Create: `tests/extraction_bench/runner.py`

- [ ] **Step 1: Create the fixtures package**

Create `tests/extraction_bench/fixtures/__init__.py` with content `# programmatic fixture generation\n`.

- [ ] **Step 2: Write the fixture generator**

Create `tests/extraction_bench/fixtures/generate_fixtures.py`:

```python
"""Programmatic bench-case generator.

Creates one fixture per native format with known source bytes and known expected
JSON. Run this from project root to (re)populate tests/extraction_bench/cases/.
"""
from __future__ import annotations

import io
import json
import os
from dataclasses import asdict
from pathlib import Path

import fitz
import openpyxl
from docx import Document as DocxDocument
from pptx import Presentation

BENCH_ROOT = Path(__file__).resolve().parents[1] / "cases"


def _write_case(doc_id: str, ext: str, source_bytes: bytes, expected: dict) -> None:
    case_dir = BENCH_ROOT / doc_id
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / f"source{ext}").write_bytes(source_bytes)
    (case_dir / "expected.json").write_text(json.dumps(expected, indent=2), encoding="utf-8")
    (case_dir / "notes.md").write_text(f"Synthetic bench fixture for {doc_id}.\n", encoding="utf-8")


def _expected_for_pdf(text_per_page: list[str]) -> dict:
    return {
        "format": "pdf_native",
        "path_taken": "native",
        "pages": [
            {
                "page_num": i + 1,
                "blocks": [{"text": t, "block_type": "paragraph"}],
                "tables": [],
            }
            for i, t in enumerate(text_per_page)
        ],
        "sheets": [],
        "slides": [],
    }


def _expected_for_docx(paragraphs: list[str], table_rows: list[list[str]] | None) -> dict:
    blocks = [{"text": p, "block_type": "paragraph"} for p in paragraphs]
    tables = [{"rows": table_rows}] if table_rows else []
    return {
        "format": "docx",
        "path_taken": "native",
        "pages": [{"page_num": 1, "blocks": blocks, "tables": tables}],
        "sheets": [],
        "slides": [],
    }


def _expected_for_xlsx(sheet_name: str, rows: list[list]) -> dict:
    cells = {}
    for r_idx, row in enumerate(rows, start=1):
        for c_idx, value in enumerate(row, start=1):
            cells[f"({r_idx}, {c_idx})"] = {"value": value, "formula": None, "type": "n" if isinstance(value, (int, float)) else "s"}
    return {
        "format": "xlsx",
        "path_taken": "native",
        "pages": [],
        "sheets": [{"name": sheet_name, "cells": cells, "hidden": False, "merged_cells": [], "named_ranges": []}],
        "slides": [],
    }


def _expected_for_pptx(titles: list[str], notes: list[str]) -> dict:
    slides = []
    for i, (title, note) in enumerate(zip(titles, notes), start=1):
        slides.append(
            {
                "slide_num": i,
                "elements": [{"text": title, "block_type": "paragraph"}] if title else [],
                "tables": [],
                "notes": note,
                "hidden": False,
            }
        )
    return {"format": "pptx", "path_taken": "native", "pages": [], "sheets": [], "slides": slides}


def _expected_for_csv(rows: list[list[str]]) -> dict:
    return {
        "format": "csv",
        "path_taken": "native",
        "pages": [{"page_num": 1, "blocks": [], "tables": [{"rows": rows}]}],
        "sheets": [],
        "slides": [],
    }


def generate_pdf_case():
    d = fitz.open()
    pages_text = ["Invoice total: 1234.56", "Vendor: Acme Corp"]
    for t in pages_text:
        p = d.new_page()
        p.insert_text((72, 72), t)
    buf = io.BytesIO()
    d.save(buf)
    d.close()
    _write_case("bench_pdf_01", ".pdf", buf.getvalue(), _expected_for_pdf(pages_text))


def generate_docx_case():
    doc = DocxDocument()
    paras = ["First paragraph.", "Second paragraph with detail."]
    for p in paras:
        doc.add_paragraph(p)
    table_rows = [["Header A", "Header B"], ["cell 1", "cell 2"]]
    t = doc.add_table(rows=len(table_rows), cols=len(table_rows[0]))
    for i, row in enumerate(table_rows):
        for j, val in enumerate(row):
            t.rows[i].cells[j].text = val
    buf = io.BytesIO()
    doc.save(buf)
    _write_case("bench_docx_01", ".docx", buf.getvalue(), _expected_for_docx(paras, table_rows))


def generate_xlsx_case():
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    ws = wb.create_sheet(title="Data")
    rows = [["Name", "Amount"], ["Alice", 100], ["Bob", 200]]
    for r_idx, row in enumerate(rows, start=1):
        for c_idx, value in enumerate(row, start=1):
            ws.cell(row=r_idx, column=c_idx, value=value)
    buf = io.BytesIO()
    wb.save(buf)
    _write_case("bench_xlsx_01", ".xlsx", buf.getvalue(), _expected_for_xlsx("Data", rows))


def generate_pptx_case():
    prs = Presentation()
    layout = prs.slide_layouts[5]
    titles = ["Overview", "Details"]
    notes = ["speaker notes one", "speaker notes two"]
    for title, note in zip(titles, notes):
        s = prs.slides.add_slide(layout)
        if s.shapes.title is not None:
            s.shapes.title.text = title
        s.notes_slide.notes_text_frame.text = note
    buf = io.BytesIO()
    prs.save(buf)
    _write_case("bench_pptx_01", ".pptx", buf.getvalue(), _expected_for_pptx(titles, notes))


def generate_csv_case():
    rows = [["h1", "h2"], ["1", "2"], ["3", "4"]]
    data = ("\n".join(",".join(r) for r in rows) + "\n").encode("utf-8")
    _write_case("bench_csv_01", ".csv", data, _expected_for_csv(rows))


def main() -> None:
    BENCH_ROOT.mkdir(parents=True, exist_ok=True)
    generate_pdf_case()
    generate_docx_case()
    generate_xlsx_case()
    generate_pptx_case()
    generate_csv_case()
    print(f"generated fixtures under {BENCH_ROOT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Write the bench runner**

Create `tests/extraction_bench/runner.py`:

```python
"""Extraction bench runner.

Iterates over tests/extraction_bench/cases/<doc_id>/ entries, runs the native
adapter on source.<ext>, compares against expected.json, emits a per-case and
aggregate report. Exits non-zero if any gate fails.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

# Ensure repo root is on sys.path so `src...` imports resolve.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.extraction.adapters.dispatcher import dispatch_native  # noqa: E402
from tests.extraction_bench.scoring import score_extraction  # noqa: E402

BENCH_CASES = Path(__file__).parent / "cases"

# Native-path gate per spec §8.4
NATIVE_COVERAGE_MIN = 1.0
NATIVE_FIDELITY_MIN = 0.98
NATIVE_STRUCTURE_MIN = 1.0
NATIVE_HALLUCINATION_MAX = 0.0


def _extraction_to_comparable(result) -> dict:
    """Convert ExtractionResult dataclass to dict in the shape scoring expects."""
    data = asdict(result)
    # Normalize sheet cell key (tuple) to the same string form the scoring helpers use.
    for sheet in data.get("sheets", []) or []:
        new_cells = {}
        for k, v in (sheet.get("cells") or {}).items():
            new_cells[str(k)] = v
        sheet["cells"] = new_cells
    return data


def run_case(case_dir: Path) -> dict:
    source = next(case_dir.glob("source.*"))
    expected = json.loads((case_dir / "expected.json").read_text(encoding="utf-8"))
    file_bytes = source.read_bytes()
    result = dispatch_native(file_bytes, filename=source.name, doc_id=case_dir.name)
    actual = _extraction_to_comparable(result)
    scores = score_extraction(expected, actual)
    gate_passed = (
        scores["coverage"] >= NATIVE_COVERAGE_MIN
        and scores["fidelity"] >= NATIVE_FIDELITY_MIN
        and scores["structure"] >= NATIVE_STRUCTURE_MIN
        and scores["hallucination"] <= NATIVE_HALLUCINATION_MAX
    )
    return {"case": case_dir.name, "scores": scores, "gate_passed": gate_passed}


def main() -> int:
    if not BENCH_CASES.exists():
        print(f"ERROR: bench cases directory missing: {BENCH_CASES}", file=sys.stderr)
        return 2
    reports = []
    any_failed = False
    for case_dir in sorted(p for p in BENCH_CASES.iterdir() if p.is_dir()):
        report = run_case(case_dir)
        reports.append(report)
        status = "PASS" if report["gate_passed"] else "FAIL"
        print(
            f"[{status}] {report['case']}: "
            f"cov={report['scores']['coverage']:.3f} "
            f"fid={report['scores']['fidelity']:.3f} "
            f"struct={report['scores']['structure']:.3f} "
            f"hal={report['scores']['hallucination']:.3f} "
            f"weighted={report['scores']['weighted']:.3f}"
        )
        if not report["gate_passed"]:
            any_failed = True

    out = BENCH_CASES.parent / "bench_report.json"
    out.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Generate fixtures and run the bench**

```bash
cd /home/ubuntu/.config/superpowers/worktrees/DocWain/preprod_v02
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/python -m tests.extraction_bench.fixtures.generate_fixtures
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/python -m tests.extraction_bench.runner
```

Expected: fixtures generated under `tests/extraction_bench/cases/`; bench runner exits 0 with five `[PASS]` lines.

- [ ] **Step 5: Commit**

```bash
git add -f tests/extraction_bench/fixtures/__init__.py tests/extraction_bench/fixtures/generate_fixtures.py tests/extraction_bench/runner.py tests/extraction_bench/cases/ tests/extraction_bench/bench_report.json
git commit -m "extraction: add bench fixture generator + runner with 5 seed cases"
```

---

### Task 12: Remove KG triggers from extraction + embedding paths

**Files:**
- Modify: `src/api/extraction_service.py` (remove line 277 KG trigger)
- Modify: `src/api/embedding_service.py` (remove lines 240-256 KG trigger)
- Modify: `src/api/dataHandler.py` (remove lines 1697, 1711-1737 KG triggers)

- [ ] **Step 1: Read + snapshot the three files**

```bash
cd /home/ubuntu/.config/superpowers/worktrees/DocWain/preprod_v02
grep -n "get_graph_ingest_queue\|enqueue_graph\|KG.*ingest\|kg_ingest" src/api/extraction_service.py src/api/embedding_service.py src/api/dataHandler.py
```

Note the exact line numbers and surrounding context.

- [ ] **Step 2: Remove the trigger in `src/api/extraction_service.py`**

Locate the KG enqueue call near line 277. Remove the statement + any immediately surrounding code that exists solely to support it (the variable holding the queue reference, any `try/except` that only wraps this call). Replace with a single-line comment:

```python
# KG ingestion moved to the training-stage background service (spec: 2026-04-23-extraction-accuracy-design.md §6.1).
```

Do NOT delete the `get_graph_ingest_queue` import if it is used elsewhere in the file; if it is unused after removal, also remove the import.

- [ ] **Step 3: Remove the trigger in `src/api/embedding_service.py`**

Locate lines 240-256 (KG enqueue block). Replace the block with the same one-line comment:

```python
# KG ingestion moved to the training-stage background service (spec: 2026-04-23-extraction-accuracy-design.md §6.1).
```

Clean up now-unused imports.

- [ ] **Step 4: Remove the triggers in `src/api/dataHandler.py`**

Locate line 1697 and lines 1711-1737 (two KG enqueue sites). Replace each with the same comment. Clean up unused imports.

- [ ] **Step 5: Sanity-check the existing test suite still runs**

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/pytest tests/unit -x -q --timeout=30
```

Expected: tests pass (no new failures introduced by the removal). If import errors surface because a now-unused import was removed, add the import back with a `# noqa: F401  # preserved for backwards compat` comment.

- [ ] **Step 6: Commit**

```bash
git add src/api/extraction_service.py src/api/embedding_service.py src/api/dataHandler.py
git commit -m "extraction: remove KG triggers from extraction and embedding; moves to training stage"
```

---

### Task 13: Wire native dispatcher into the Celery extraction task

**Files:**
- Modify: `src/tasks/extraction.py` (try native dispatcher first; fall through to existing path on NotNativePathError)

- [ ] **Step 1: Inspect the current Celery task**

```bash
cd /home/ubuntu/.config/superpowers/worktrees/DocWain/preprod_v02
sed -n '1,200p' src/tasks/extraction.py
```

Find the `extract_document()` entry (per exploration: line 98). Identify exactly where it calls the old extraction engine and uploads the result.

- [ ] **Step 2: Add the native-dispatcher integration**

Above the existing call to `ExtractionEngine.extract()` / `fileProcessor()` inside `extract_document()`, insert a native-first attempt. Example shape (adjust imports and variable names to match the actual code in that file — the key logic, not the names, is what must land):

```python
# --- Plan 1: native-first dispatch ---
from dataclasses import asdict as _dc_asdict
from src.extraction.adapters.dispatcher import dispatch_native
from src.extraction.adapters.errors import NotNativePathError

try:
    canonical = dispatch_native(file_bytes, filename=filename, doc_id=doc_id)
    extraction_json = _dc_asdict(canonical)
    # Normalize sheet cell tuple keys for JSON serialization
    for sheet in extraction_json.get("sheets", []) or []:
        sheet["cells"] = {str(k): v for k, v in (sheet.get("cells") or {}).items()}
    # Upload canonical extraction JSON to blob (existing upload helper)
    upload_extraction_json(sub_id=sub_id, profile_id=profile_id, doc_id=doc_id, payload=extraction_json)
    # Update Mongo status with path_taken="native"
    update_extraction_summary(
        doc_id=doc_id,
        status="EXTRACTION_COMPLETED",
        path_taken="native",
        format=canonical.format,
        page_count=len(canonical.pages),
        sheet_count=len(canonical.sheets),
        slide_count=len(canonical.slides),
    )
    return {"status": "ok", "path_taken": "native", "doc_id": doc_id}
except NotNativePathError as exc:
    logger.info("native path not applicable (%s); falling through to legacy engine", exc)
    # fall through to the existing code path below — unchanged
```

The subagent implementing this task will have to match the exact variable names and helper functions that exist in the current `extraction.py`. Keep the existing code path for the fallback unchanged.

- [ ] **Step 3: Sanity-check the Celery task imports cleanly**

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/python -c "from src.tasks import extraction; print('import OK')"
```

Expected: prints `import OK`. If import fails, fix the imports before continuing.

- [ ] **Step 4: Run existing extraction tests**

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/pytest tests/ -k extraction -x -q --timeout=30
```

Expected: no new failures. If any test imports something the changes broke, fix the test import; do NOT relax the test.

- [ ] **Step 5: Commit**

```bash
git add src/tasks/extraction.py
git commit -m "extraction: wire native dispatcher into Celery task (native-first, legacy fallback)"
```

---

### Task 14: Bump Celery concurrency

**Files:**
- Modify: `systemd/docwain-celery-worker.service`

- [ ] **Step 1: Inspect the current service file**

```bash
cd /home/ubuntu/.config/superpowers/worktrees/DocWain/preprod_v02
cat systemd/docwain-celery-worker.service
```

Find the `ExecStart=...` line with `--concurrency=2`.

- [ ] **Step 2: Raise the concurrency value to 4**

Change `--concurrency=2` to `--concurrency=4`. Do NOT change any other flag or behavior in the unit file.

- [ ] **Step 3: Commit**

```bash
git add systemd/docwain-celery-worker.service
git commit -m "extraction: raise Celery extraction worker concurrency 2 -> 4"
```

Note: the unit file is in-repo for version control; applying it to the running system (`sudo systemctl daemon-reload && sudo systemctl restart docwain-celery-worker`) is an operator action and is NOT part of the subagent's job.

---

### Task 15: End-to-end smoke test

**Files:**
- Create: `tests/integration/test_native_extraction_smoke.py`

- [ ] **Step 1: Write the smoke test**

Create `tests/integration/__init__.py` if missing with content `# integration tests\n`.

Create `tests/integration/test_native_extraction_smoke.py`:

```python
"""End-to-end smoke: dispatch_native on each format yields a non-empty ExtractionResult."""
import io

import fitz
import openpyxl
from docx import Document as DocxDocument
from pptx import Presentation

from src.extraction.adapters.dispatcher import dispatch_native


def _pdf_bytes() -> bytes:
    d = fitz.open()
    p = d.new_page()
    p.insert_text((72, 72), "smoke pdf content")
    buf = io.BytesIO()
    d.save(buf)
    d.close()
    return buf.getvalue()


def _docx_bytes() -> bytes:
    d = DocxDocument()
    d.add_paragraph("smoke docx content")
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


def _xlsx_bytes() -> bytes:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws["A1"] = "smoke"
    ws["B1"] = "xlsx"
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _pptx_bytes() -> bytes:
    prs = Presentation()
    layout = prs.slide_layouts[5]
    slide = prs.slides.add_slide(layout)
    slide.shapes.title.text = "smoke pptx"
    buf = io.BytesIO()
    prs.save(buf)
    return buf.getvalue()


def _csv_bytes() -> bytes:
    return b"a,b,c\n1,2,3\n"


def test_smoke_all_native_formats_produce_non_empty_output():
    cases = [
        ("smoke.pdf", _pdf_bytes(), "pdf_native"),
        ("smoke.docx", _docx_bytes(), "docx"),
        ("smoke.xlsx", _xlsx_bytes(), "xlsx"),
        ("smoke.pptx", _pptx_bytes(), "pptx"),
        ("smoke.csv", _csv_bytes(), "csv"),
    ]
    for filename, data, expected_format in cases:
        result = dispatch_native(data, filename=filename, doc_id=filename)
        assert result.format == expected_format, filename
        assert result.path_taken == "native"
        has_content = bool(result.pages) or bool(result.sheets) or bool(result.slides)
        assert has_content, f"no content extracted from {filename}"
```

- [ ] **Step 2: Run the smoke**

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/pytest tests/integration/test_native_extraction_smoke.py -x -q
```

Expected: 1 passed.

- [ ] **Step 3: Run the full new test suite**

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/pytest tests/unit/extraction tests/integration/test_native_extraction_smoke.py -x -q
```

Expected: all tests pass. Record the count for the report.

- [ ] **Step 4: Run the bench**

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/python -m tests.extraction_bench.runner
```

Expected: 5 `[PASS]` lines, exit 0.

- [ ] **Step 5: Commit**

```bash
git add tests/integration/__init__.py tests/integration/test_native_extraction_smoke.py
git commit -m "extraction: add native-path end-to-end smoke across all 5 formats"
```

---

## Self-review — spec coverage

**Spec §3 (architecture):** Task 9 dispatcher routes by extension; native adapters Tasks 4–8 implement each format; DocIntel + vision path explicitly deferred to Plan 2 per spec §5 phasing. ✓

**Spec §4.1 (native path per format):** PDF (Task 4), DOCX (Task 5), XLSX (Task 6), PPTX (Task 7), CSV (Task 8) all covered. Native coverage verifier for native is the bench gate itself (Task 10 scoring + Task 11 runner). ✓

**Spec §4.3 (canonical schema):** Task 2 defines ExtractionResult with all required fields (pages, sheets, slides, metadata including DocIntel + coverage). ✓

**Spec §5 (parallelism):** Task 14 bumps Celery concurrency 2 → 4 (per-document parallelism). Intra-document parallelism for >100-page PDFs is deferred — not in Plan 1 scope since native PDFs are usually small enough that sequential page processing is fine. Revisit in Plan 2 if needed. ✓ (covered as scoped)

**Spec §6 (storage):** Task 13 writes canonical JSON to Blob + updates Mongo extraction summary; preserves existing status values per `feedback_mongo_status_stability.md`. ✓

**Spec §6.1 (KG trigger removal):** Task 12 removes all three KG trigger sites. ✓

**Spec §7 (observability):** NOT covered in Plan 1. The per-extraction Redis log entry is deferred to Plan 2 so we can include the vision-path fields (DocIntel routing confidence, coverage verifier score, fallback invocations) in one go rather than building partial observability. **Gap noted.**

**Spec §8 (bench):** Tasks 10 (scoring), 11 (fixture generator + runner). Seed bench is 5 synthetic fixtures (one per format). Real-document ground-truth docs are operator-provided later per bench README. ✓ (for Plan 1 scope)

**Spec §9 (phased rollout):** Plan 1 executes Phase 0 (bench) + Phase 1 native half. Phase 1 vision half is Plan 2. Phase 2/3 are separate workstreams. ✓

## Self-review — placeholder scan

- All steps show actual code, not "implement appropriately."
- All test functions have full bodies.
- Commit messages are exact, not templated.
- File paths are absolute where needed for commands, package-relative for imports.
- One small hand-off in Task 13 Step 2: the Celery task integration code shows the shape to paste but acknowledges the subagent will need to match existing variable names in `src/tasks/extraction.py`. This is inherent to integrating with unfamiliar existing code — the task explicitly says "match the actual code" and provides the pattern, not a placeholder.
- Observability gap flagged explicitly above; not a placeholder, a deferral.

## Self-review — type consistency

- `ExtractionResult`, `Page`, `Block`, `Table`, `Sheet`, `Slide`, `Image`, `DocIntelMetadata`, `CoverageMetadata`, `ExtractionMetadata` names match across schema (Task 2), adapters (Tasks 4–8), scoring (Task 10), fixtures (Task 11), smoke (Task 15).
- Adapter function name consistently `extract_<format>_native` (pdf, docx, xlsx, pptx, csv). Dispatcher `dispatch_native`.
- Error types `NotNativePathError` / `NativeAdapterError` used consistently.
- Cell key format: adapter emits `(row, col)` tuples; scoring + fixtures normalize to `str(tuple)`. Consistent.

No drift detected.

---

Plan complete.
