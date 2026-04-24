"""Vision-path orchestrator.

Entry point: extract_via_vision(file_bytes, doc_id, filename, format_hint) →
ExtractionResult. Internally:

1. For PDF: render each page to PNG. For images: use the image bytes directly.
2. Per page:
   a. DocIntel classifier call → routing decision (format, layout, handwriting).
      (Called on page 0 only; later pages carry the page-0 decision.)
   b. Vision extractor call → structured regions JSON.
   c. Coverage verifier call → complete?, missed_regions, low_confidence_regions.
   d. If incomplete: run fallback ensemble on each missed region's bbox.
3. Collect per-page Block/Table/Image into canonical Page.
4. Assemble ExtractionResult with metadata (routing, coverage, fallback).
"""
from __future__ import annotations

import io
import time
from typing import Any, Dict, List, Optional

import fitz
from PIL import Image

from src.extraction.canonical_schema import (
    Block,
    CoverageMetadata,
    DocIntelMetadata,
    ExtractionMetadata,
    ExtractionResult,
    Image as CanonicalImage,
    Page,
    Table,
)
from src.extraction.vision.client import VisionClient, VisionClientError, VisionResponse
from src.extraction.vision.docintel import (
    CLASSIFIER_SYSTEM_PROMPT,
    COVERAGE_SYSTEM_PROMPT,
    RoutingDecision,
    parse_coverage_response,
    parse_routing_response,
)
from src.extraction.vision.extractor import (
    EXTRACTOR_SYSTEM_PROMPT,
    VisionExtraction,
    parse_extractor_response,
)
from src.extraction.vision.fallback import run_fallback_ensemble
from src.extraction.vision.images import render_pdf_page_to_png


VLLM_BASE_URL = "http://localhost:8100/v1"
VLLM_MODEL = "docwain-fast"


def _build_client() -> VisionClient:
    return VisionClient(base_url=VLLM_BASE_URL, model=VLLM_MODEL)


def _route(client: VisionClient, *, image_bytes: bytes, filename: str) -> RoutingDecision:
    try:
        resp = client.call(
            system=CLASSIFIER_SYSTEM_PROMPT,
            user_text=f"Filename: {filename}. Return the routing decision JSON for this page.",
            image_bytes=image_bytes,
            max_tokens=256,
            temperature=0.0,
        )
        return parse_routing_response(resp.text)
    except VisionClientError:
        return RoutingDecision(
            format="image", doc_type_hint="unknown", layout_complexity="simple",
            has_handwriting=False, suggested_path="vision", confidence=0.1,
        )


def _extract_page(client: VisionClient, *, image_bytes: bytes, hints: RoutingDecision) -> VisionExtraction:
    user_text = (
        f"Hints: doc_type={hints.doc_type_hint}, layout={hints.layout_complexity}, "
        f"handwriting={hints.has_handwriting}. Emit the regions JSON for this page."
    )
    try:
        resp = client.call(
            system=EXTRACTOR_SYSTEM_PROMPT,
            user_text=user_text,
            image_bytes=image_bytes,
            max_tokens=4096,
            temperature=0.0,
        )
        return parse_extractor_response(resp.text)
    except VisionClientError:
        return VisionExtraction()


def _verify(client: VisionClient, *, image_bytes: bytes, extraction: VisionExtraction) -> Dict[str, Any]:
    import json as _json
    payload_preview = _json.dumps({
        "regions": extraction.regions[:30],
        "reading_order": extraction.reading_order,
        "page_confidence": extraction.page_confidence,
    })
    try:
        resp = client.call(
            system=COVERAGE_SYSTEM_PROMPT,
            user_text=(
                "Here is the extraction JSON for the page:\n"
                f"{payload_preview}\n"
                "Return the coverage verdict JSON."
            ),
            image_bytes=image_bytes,
            max_tokens=1024,
            temperature=0.0,
        )
        return parse_coverage_response(resp.text)
    except VisionClientError:
        return {"complete": False, "missed_regions": [], "low_confidence_regions": []}


def _page_image_bytes_to_pil(image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def _regions_to_canonical_page(regions: List[Dict[str, Any]], *, page_num: int) -> Page:
    blocks: List[Block] = []
    tables: List[Table] = []
    images: List[CanonicalImage] = []
    for r in regions:
        rtype = r.get("type", "text_block")
        content = r.get("content", "")
        if rtype == "table" and isinstance(content, dict) and isinstance(content.get("rows"), list):
            tables.append(Table(rows=content["rows"], bbox=None, header_row_index=0 if content["rows"] else None))
            continue
        if rtype in ("text_block", "handwriting", "form_field"):
            text = content if isinstance(content, str) else str(content)
            block_type = "paragraph" if rtype == "text_block" else rtype
            blocks.append(Block(text=text, bbox=None, block_type=block_type))
            continue
        if rtype == "figure":
            caption = content if isinstance(content, str) else ""
            images.append(CanonicalImage(bbox=None, ocr_text="", caption=caption))
            continue
    return Page(page_num=page_num, blocks=blocks, tables=tables, images=images)


def _apply_fallback(
    *,
    page_image_bytes: bytes,
    coverage: Dict[str, Any],
    extraction: VisionExtraction,
) -> List[Dict[str, Any]]:
    """Run fallback on missed regions; return augmented regions list with trailing __invocations__ record."""
    regions = list(extraction.regions)
    invocations: List[Dict[str, Any]] = []
    if coverage.get("complete"):
        return regions + [{"__invocations__": invocations}]
    pil_img = _page_image_bytes_to_pil(page_image_bytes)
    for missed in coverage.get("missed_regions", []):
        bbox = missed.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            result = run_fallback_ensemble(pil_img, bbox=bbox)
        except Exception as exc:
            invocations.append({"bbox": bbox, "status": "error", "error": repr(exc)})
            continue
        if result.text.strip():
            regions.append({
                "type": "text_block",
                "bbox": bbox,
                "content": result.text,
                "confidence": max(0.3, result.agreement),
                "source": f"fallback:{result.engine_winner}",
            })
        invocations.append({
            "bbox": bbox,
            "engine_winner": result.engine_winner,
            "agreement": result.agreement,
            "chars": len(result.text),
        })
    return regions + [{"__invocations__": invocations}]


def extract_via_vision(
    file_bytes: bytes,
    *,
    doc_id: str,
    filename: str,
    format_hint: str,
) -> ExtractionResult:
    """Main entry. Returns a canonical ExtractionResult built from vision + fallback."""
    client = _build_client()

    # Obtain page images. For PDFs, render each page. For image files, use bytes as page 0.
    page_images: List[bytes] = []
    if format_hint in ("pdf_scanned", "pdf_mixed", "pdf_native"):
        doc = fitz.open(stream=io.BytesIO(file_bytes), filetype="pdf")
        try:
            for i in range(len(doc)):
                page_images.append(render_pdf_page_to_png(file_bytes, page_index=i, dpi=144))
        finally:
            doc.close()
    else:
        page_images.append(file_bytes)

    # Classifier called on page 0 only.
    routing = _route(client, image_bytes=page_images[0], filename=filename)

    pages: List[Page] = []
    all_fallback_invocations: List[Dict[str, Any]] = []
    verifier_scores: List[float] = []

    for idx, img_bytes in enumerate(page_images):
        extraction = _extract_page(client, image_bytes=img_bytes, hints=routing)
        coverage = _verify(client, image_bytes=img_bytes, extraction=extraction)
        regions_with_invocations = _apply_fallback(
            page_image_bytes=img_bytes, coverage=coverage, extraction=extraction
        )
        invocations: List[Dict[str, Any]] = []
        cleaned_regions: List[Dict[str, Any]] = []
        for r in regions_with_invocations:
            if isinstance(r, dict) and "__invocations__" in r:
                invocations.extend(r["__invocations__"])
            else:
                cleaned_regions.append(r)
        page = _regions_to_canonical_page(cleaned_regions, page_num=idx + 1)
        pages.append(page)
        all_fallback_invocations.extend(invocations)
        if coverage.get("complete"):
            verifier_scores.append(1.0)
        else:
            missed_n = len(coverage.get("missed_regions", []))
            total_n = max(1, len(cleaned_regions))
            verifier_scores.append(max(0.0, 1.0 - missed_n / (missed_n + total_n)))

    avg_coverage = sum(verifier_scores) / len(verifier_scores) if verifier_scores else 0.0

    return ExtractionResult(
        doc_id=doc_id,
        format=routing.format if routing.format != "native" else format_hint,
        path_taken="vision",
        pages=pages,
        sheets=[],
        slides=[],
        metadata=ExtractionMetadata(
            doc_intel=DocIntelMetadata(
                doc_type_hint=routing.doc_type_hint,
                layout_complexity=routing.layout_complexity,
                has_handwriting=routing.has_handwriting,
                routing_confidence=routing.confidence,
            ),
            coverage=CoverageMetadata(
                verifier_score=avg_coverage,
                missed_regions=[],
                low_confidence_regions=[],
                fallback_invocations=all_fallback_invocations,
            ),
            extraction_version="2026-04-24-v2",
        ),
    )
