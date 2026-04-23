# Extraction Accuracy Overhaul — Design

**Date:** 2026-04-23
**Branch:** `preprod_v02` (implementation line, off `preprod_v01`)
**Status:** Approved for implementation
**Owner:** Muthu (brainstormed collaboratively; pair-working session)

## 1. Purpose

DocWain's extraction stage is the load-bearing layer for every downstream capability — embeddings, Researcher Agent, KG, RAG, response generation. Any content miss, alteration, or structural corruption at extraction propagates to every downstream consumer. This spec defines the architecture that replaces the current mostly-stubbed extraction engine with one that delivers verbatim, structure-preserving extraction across seven formats.

**Definition of accurate extraction (owner's words, 2026-04-23):** "whatever content provided in PDF, PPT, Word, Excel, CSV, image, scanned PDFs should accurately be preserved and extracted as it is. There should be no miss in extraction of the content accurately as it is the true source."

Operationally:
- **No miss** — every section, field, table cell, slide, sheet, and page present in the source must appear in the output.
- **Verbatim fidelity** — extracted content matches the source character-by-character, within OCR's physical ceiling for scanned content.
- **Structure preservation** — tables keep row/column structure; sheets keep cell coordinates; slides keep order; pages keep reading order.
- **No hallucination** — nothing added that isn't in the source.

## 2. Scope

### In-scope formats
PDF (native text-layer), PDF (scanned), PPT / PPTX, Word / DOCX, Excel / XLSX, CSV, image files (JPG / PNG / TIFF), and handwritten content within any of the above.

### Non-goals (explicit)
- Researcher Agent (separate spec — training stage, reads extraction output)
- KG ingestion design (separate spec — training-stage background service; this spec *removes* the broken KG triggers from extraction but does not design their replacement)
- DocWain V2 vision-upgrade training plan (separate spec — Phase 2 training workstream)
- Query-time generation / RAG (separate; uses extraction output unchanged)
- HITL screening UI (untouched)
- New MongoDB `pipeline_status` values (forbidden per existing contract)

## 3. Architecture

Two-path routing, one model (DocWain), one verifier.

```
Upload
   ↓
File adapter (no models) — hash, MIME, page/sheet/slide counts
   ↓
DocIntel Classifier + Router (DocWain call) — decides path
   ↓
   ├─ NATIVE PATH (deterministic, lossless)          ← text-layer PDFs,
   │                                                   DOCX, XLSX, PPTX, CSV
   │    file-format libraries read source bytes
   │
   └─ VISION PATH (DocWain-primary, OCR fallback)    ← scanned PDFs,
                                                       images, handwriting,
                                                       embedded images within
                                                       native-path docs
        1. DocWain vision pass (page image → structured regions)
        2. DocIntel coverage verifier (missed regions? low confidence?)
        3. Region-scoped fallback ensemble only if needed
           (Tesseract, EasyOCR run on specific bboxes — not whole page)
        4. Verifier re-checks fallback output
   ↓
Merge into canonical extraction JSON
   ↓
Store to Azure Blob (content) + MongoDB (control-plane metadata)
```

### 3.1 DocIntel role

DocIntel is NOT a separate model. It is a **set of capabilities inside the unified DocWain model**, invoked at three points per document via specific prompting:

1. **Classify + route** — inputs: filename, first-page image, first 2 KB of text layer if any. Output: routing decision JSON.
2. **Extract with hints** — for vision path: DocWain receives the page image plus routing hints and emits structured regions.
3. **Coverage verify** — DocWain receives the source image and the extracted JSON; answers whether every visible region is represented.

DocIntel improves over time via a training workstream (out of scope here). It does not persist a separate artifact post-extraction; its effect is embedded in the accuracy of the extracted output.

### 3.2 No secondary ML/DL models

DocWain is the ONLY model in the extraction stage for OCR, layout detection, handwriting recognition, form parsing, and table recovery. Native file-format libraries (python-docx, openpyxl, PyMuPDF, python-pptx, stdlib csv) are NOT models — they read file format bytes deterministically and stay.

**One exception, gated as fallback only:** Tesseract + EasyOCR, invoked region-by-region when DocWain fails or when the coverage verifier flags a miss. Fallback output is re-verified by DocWain before acceptance. Fallback invocation is measured per-extraction and is expected to trend toward zero as DocWain training matures.

## 4. Per-path pipelines

### 4.1 Native path (deterministic, lossless)

Triggered when DocIntel route = `native` AND format is Word / Excel / CSV / PPT / text-layer PDF.

**PDF (text-layer):**
- PyMuPDF: text blocks per page in reading order.
- PyMuPDF `find_tables()`: table structure (rows, cells, headers).
- Embedded images that contain text → queued for vision sub-pass.
- Preserved: page numbers, reading order, hyperlinks, bookmarks, footnotes, annotations (comments, highlights).

**DOCX:**
- python-docx: paragraphs in order with semantic run-level formatting (headings, bullets, numbering).
- Tables: `table.rows` → `table.cells` (preserves merges).
- Preserved: footnotes, endnotes, comments, tracked changes, embedded images (vision sub-pass), headers/footers, section breaks.

**XLSX / XLS:**
- openpyxl (XLSX) / xlrd (XLS) per sheet.
- Cells: values AND formulas (both) with types.
- Preserved: hidden sheets (flag, not drop), merged cells, named ranges, defined tables, pivot sources.

**PPTX:**
- python-pptx per slide.
- Text frames, tables, shapes with text, slide notes, slide masters.
- Preserved: hidden slides (flag), embedded images (vision sub-pass).

**CSV:**
- stdlib csv with auto-dialect detection → rows.

**Coverage verifier for native:** deterministic cross-check. For a DOCX, every `<w:p>` and `<w:tbl>` element in `document.xml` must have an entry in the output. Any miss → extraction fails and we fix the adapter. Native is never "best effort"; a miss on native means a bug.

### 4.2 Vision path (DocWain-primary, OCR fallback)

Triggered when DocIntel route = `vision`, OR native path encounters an embedded image with text, OR handwriting is present.

1. **DocWain vision pass** — inputs: page image + DocIntel hints. Output:
    ```json
    {
      "regions": [
        {"type": "text_block|table|form_field|figure|handwriting",
         "bbox": [x,y,w,h],
         "content": "...",
         "confidence": 0.0..1.0}
      ],
      "reading_order": [region_id...],
      "page_confidence": 0.0..1.0
    }
    ```

2. **DocIntel coverage verifier** — inputs: source image + DocWain output. Answers: "is every visible region represented? Any missed or low-confidence regions?"

3. **Decision:**
    - Complete + no low-confidence → accept.
    - Complete + low-confidence regions → retry those regions with DocWain at higher decoding temperature / extended reasoning.
    - Not complete OR retry still low-confidence → **region-scoped fallback ensemble:**
       - Tesseract on the missed / low-confidence bboxes only.
       - EasyOCR on same bboxes.
       - Coverage verifier re-checks fallback output.
       - Highest-agreement answer wins; otherwise flag for human review.

4. **Merge** into the canonical JSON shape (identical to native path), so downstream consumers never branch on path.

**Region-scoped fallback is critical.** If DocWain handles a page's header and first paragraph correctly but misses a handwritten signature block, Tesseract runs on just the signature bbox — not the whole page. This confines fallback scope and keeps DocWain as the primary extractor.

### 4.3 Common output schema

```json
{
  "doc_id": "...",
  "format": "pdf_native|pdf_scanned|docx|xlsx|pptx|csv|image|handwritten",
  "path_taken": "native|vision|mixed",
  "pages": [
    {"page_num": 1, "blocks": [...], "tables": [...], "images": [...]}
  ],
  "sheets": [],   // xlsx only
  "slides": [],   // pptx only
  "metadata": {
    "doc_intel": {"doc_type_hint": "...", "layout_complexity": "...",
                  "has_handwriting": bool, "routing_confidence": 0.0..1.0},
    "coverage": {"verifier_score": 0.0..1.0, "missed_regions": [],
                 "low_confidence_regions": [], "fallback_invocations": []},
    "extraction_version": "2026-04-23-v1"
  }
}
```

## 5. Parallelism

- **Per-document parallelism in Celery** — raise `extraction_queue` concurrency from 2 to 4–8 workers; one DocWain vLLM call in flight per worker. Each document takes its own path independently.
- **Within a single document, pages are processed sequentially** — DocIntel builds a per-document context map (doc type, layout template, entities seen so far) that later pages benefit from.
- **Exception for very large PDFs (>100 pages):** pages batched in groups of 20; each batch a separate DocWain call; DocIntel carries context between batches. Configurable cap.

## 6. Storage (respects existing rules)

- **Azure Blob:** raw file at `raw/{doc_id}/{filename}`. Extraction JSON at `{sub_id}/{profile_id}/{doc_id}/extraction.json`. Content lives here.
- **MongoDB control plane:** `documents.{doc_id}.extraction` gets summary metadata (status, page/sheet/slide counts, coverage score, path_taken, blob path). Never document content. Status values preserved per `feedback_mongo_status_stability.md`: `UPLOADED` → `EXTRACTION_IN_PROGRESS` → `EXTRACTION_COMPLETED | EXTRACTION_FAILED`.
- **Qdrant, Neo4j, Researcher outputs:** NOT written by extraction. Those belong to later stages.

### 6.1 KG trigger removal

Current code builds KG in three locations (per canonical pipeline gap):
- `src/api/extraction_service.py:277`
- `src/api/embedding_service.py:240-256`
- `src/api/dataHandler.py:1697,1711-1737`

This spec **removes** all three KG triggers from the extraction stage. KG ingestion moves to a single post-screening async task during the training stage (designed in a separate spec, part of roadmap item 4). Extraction does not talk to Neo4j.

## 7. Observability

Every extraction writes a structured log entry to Redis (`extraction:log:{doc_id}`, 7-day TTL) with:
- Timings per stage (file adapter / DocIntel route / path execution / coverage verify / fallback if any)
- DocIntel routing decision + confidence
- Coverage verifier score + missed regions
- Fallback invocation count (per region, which engine, final confidence)
- Any human-review flag

This feeds the training workstream's pattern capture.

No wall-clock timeout on extraction per `feedback_no_timeouts.md`. Per-operation network timeouts on embedded URL fetches are fine.

## 8. Extraction accuracy bench

This is the gate. Per `feedback_measure_before_change.md`, no code change in extraction until the bench exists. It is the definition of "accurate."

### 8.1 Corpus

30–50 documents covering every format and edge case:
- Native PDFs (simple, multi-column, with tables, with embedded images, with footnotes/annotations)
- Scanned PDFs (clean, skewed, low-quality, multi-page)
- Images (JPG/PNG with text, photographs of documents, receipts)
- DOCX (tracked changes, comments, footnotes, headers/footers, complex tables)
- XLSX (multi-sheet, hidden sheets, merged cells, formulas, pivot-source data)
- PPTX (slide notes, hidden slides, embedded images with text)
- CSV (varied dialects)
- Handwritten content (forms, notes, mixed hand+print)

### 8.2 Ground truth

Per document: hand-authored canonical JSON matching the output schema from §4.3. Every block, table cell, sheet cell, slide element, and page present. Stored at `tests/extraction_bench/<doc_id>/{source.ext, expected.json, notes.md}`, version-controlled.

### 8.3 Scoring dimensions

1. **Coverage (50%)** — every block/row/cell/slide/page in expected.json must appear in extracted.json. Miss = 0 for that doc. Absolute rule.
2. **Verbatim fidelity (30%)** — Levenshtein similarity per matched block. Threshold ≥ 0.98 native, ≥ 0.92 vision, ≥ 0.85 handwriting.
3. **Structure preservation (15%)** — tables keep row × column match; slide / sheet / page ordering preserved.
4. **No hallucination (5%)** — content in extraction not in expected.json penalizes.

### 8.4 Gate thresholds

- **Native path:** coverage 100% (absolute), fidelity ≥ 0.98, structure 100%, hallucination 0%. No promotion below this.
- **Vision path:** coverage ≥ 0.95, fidelity ≥ 0.92, structure ≥ 0.95, hallucination < 0.01. Promotes on meeting bar; fallback rate tracked.
- **Handwriting:** coverage ≥ 0.90, fidelity ≥ 0.85. Promotes if beats previous checkpoint.

### 8.5 Feedback loop

Production extraction failures (human-review flags, coverage verifier misses) feed back as new bench entries monthly. The bench grows; the gate stays absolute on native, catches up on vision via training.

## 9. Phased rollout

**Phase 0 — Bench first (no code changes).**
- Build 30–50-doc bench with ground-truth JSONs.
- Run current extraction (mostly-stubbed v2 path) against the bench → baseline report per format.
- Estimated 2–3 days, mostly labeling.

**Phase 1 — Engineering on `preprod_v02`.**
- Implement file-format adapters + canonical JSON schema.
- DocIntel routing + coverage verifier via DocWain prompting.
- Vision path uses existing DocWain V2 capability + Tesseract/EasyOCR region fallback.
- KG trigger removal from three locations.
- Celery concurrency tuning (`extraction_queue` 2 → 4–8).
- Gate: native path must hit bench thresholds (100% coverage). Vision path promotes whatever it achieves today — that becomes Phase 2's starting line.
- Estimated 2–3 weeks.

**Phase 2 — Training workstream (parallel, separate spec).**
- DocWain V2 vision upgrade: training data from DocLayNet, DocVQA, FUNSD, CORD, IAM.
- Monthly SFT + targeted DPO on failure-case pairs mined from Phase 1 production logs.
- Gate: each promoted checkpoint must beat previous bench score on vision path.
- Cadence: quarterly sprints until Azure DocAI parity.

**Phase 3 — Fallback retirement.**
- As DocWain vision-path bench score exceeds fallback ensemble on every category, fallback invocation trends to zero in production.
- Tesseract / EasyOCR code retained but unreachable; eventually deleted.

## 10. Risks

1. **DocWain V2 on scanned/handwritten today is an unknown floor.** Phase 0 bench run gives the real number. Could be uncomfortably low. Mitigation: region-scoped fallback catches coverage gaps so Phase 1 still ships usable; Phase 2 training closes the gap.
2. **Bench labeling cost.** Hand-authoring 30–50 ground-truth JSONs is 2–3 days. Mitigation: start with 15 docs covering the most critical formats (native PDF, Excel, scanned PDF, DOCX), grow from there.
3. **Training workstream velocity.** If training sprints can't move the bench within a quarter, fallback stays load-bearing longer than planned. Mitigation: fallback is a feature during transition, not a shame.
4. **Extraction latency after the overhaul.** Three DocWain calls per document (route, extract-vision, coverage-verify) plus possible fallback regions = more model calls than today. Mitigation: native path replaces model calls with deterministic reads (faster); vision path is comparable to today per page.

## 11. Dependencies

- **DocWain vLLM local serving** must be alive and wired through `src/llm/gateway.py` (per roadmap item 5). If the serving swap isn't done yet, extraction will call Ollama Cloud, defeating the "no cloud" constraint for the vision path. This is a prerequisite.
- **Adapter YAML framework for domain-specific extraction hints** (per `feedback_adapter_yaml_blob.md`) — optional Phase 1 extension; not blocking.
- **Celery, Redis, Azure Blob, MongoDB** — all currently functional on preprod_v01.

## 12. Success criteria

The spec is delivered when:
- Phase 0 bench exists at `tests/extraction_bench/`, version-controlled, with ≥ 15 ground-truth documents covering all 7 formats.
- Phase 1 engineering on `preprod_v02` passes the native-path gate absolutely (100% coverage on every native-path bench doc) and produces a measured vision-path baseline.
- Fallback ensemble is region-scoped and gated on DocWain vision + coverage-verifier decisions (not a default path).
- KG trigger removal verified in the three named locations.
- Canonical JSON output shape from §4.3 is the ONLY downstream-facing schema.
- Per-extraction observability log exists in Redis with all fields from §7.
- All existing feedback rules honored: HITL gates, MongoDB status immutability, storage separation, no-timeouts, engineering-first-training-last, no customer data in training, no Claude attribution in commits.
