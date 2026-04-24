"""DocIntel capabilities inside the unified DocWain model.

DocIntel is NOT a separate ML/DL model — it is a set of capabilities invoked
by specific prompting of DocWain. This module holds the prompts and the
response parsers. Actual model calls happen via VisionClient in
`src.extraction.vision.client`.

Three capabilities:
1. Classifier + router — decides native vs vision, doc type hint, handwriting
   presence.
2. Coverage verifier — given the source image and the extracted JSON, answers
   whether every visible region is represented.
3. Extractor prompts live in `src.extraction.vision.extractor` (kept separate
   so this module stays pure parsing + prompt strings).

Spec: docs/superpowers/specs/2026-04-23-extraction-accuracy-design.md §3.1
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List


CLASSIFIER_SYSTEM_PROMPT = (
    "You are DocIntel, DocWain's document understanding capability. Given a "
    "document's filename, the first rendered page image, and any text already "
    "extractable from the file's text layer, you output a JSON routing decision.\n\n"
    "Output ONLY valid JSON with this shape (no prose, no markdown fences):\n"
    "{\n"
    '  "format": "pdf_native" | "pdf_scanned" | "pdf_mixed" | "pptx" | "docx" | '
    '"xlsx" | "csv" | "image" | "handwritten",\n'
    '  "doc_type_hint": string (e.g. "invoice", "resume", "contract", "receipt", '
    '"report", or "unknown"),\n'
    '  "layout_complexity": "simple" | "moderate" | "complex",\n'
    '  "has_handwriting": true | false,\n'
    '  "suggested_path": "native" | "vision" | "mixed",\n'
    '  "confidence": number between 0.0 and 1.0\n'
    "}\n\n"
    "Rules:\n"
    "- If the text layer is substantial and machine-readable, prefer 'native'.\n"
    "- If the page is visibly a scan or image of a document, prefer 'vision'.\n"
    "- If any visible writing looks handwritten, set has_handwriting=true.\n"
    "- Be deterministic and conservative on confidence — do not claim >0.9 unless "
    "the evidence is unambiguous."
)

COVERAGE_SYSTEM_PROMPT = (
    "You are DocIntel's coverage verifier for DocWain's vision extraction path. "
    "You receive the original page image and a JSON extraction output. Your job: "
    "decide whether every visible region of the image is represented in the "
    "extraction.\n\n"
    "Output ONLY valid JSON with this shape (no prose, no markdown fences):\n"
    "{\n"
    '  "complete": true | false,\n'
    '  "missed_regions": [ { "bbox": [x, y, w, h], "description": string } ],\n'
    '  "low_confidence_regions": [ { "region_id": string, "reason": string } ]\n'
    "}\n\n"
    "Rules:\n"
    "- If any visible text, table, form field, or handwritten patch lacks a "
    "corresponding entry in the extraction, add it to missed_regions.\n"
    "- Use normalized 0..1 coordinates for bbox (fractions of page width/height).\n"
    "- Set complete=true only when every visible content region is represented.\n"
    "- Empty regions (pure whitespace, decorative lines) do not count as missed."
)


@dataclass
class RoutingDecision:
    format: str
    doc_type_hint: str
    layout_complexity: str
    has_handwriting: bool
    suggested_path: str
    confidence: float


def _strip_code_fence(text: str) -> str:
    t = text.strip()
    m = re.match(r"```(?:json)?\s*(.*?)```", t, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    return t


def _first_json_object(text: str) -> str:
    t = _strip_code_fence(text)
    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return t
    return t[start : end + 1]


def parse_routing_response(text: str) -> RoutingDecision:
    """Best-effort parse of classifier output into a RoutingDecision."""
    try:
        data = json.loads(_first_json_object(text))
    except Exception:
        return RoutingDecision(
            format="image",
            doc_type_hint="unknown",
            layout_complexity="simple",
            has_handwriting=False,
            suggested_path="vision",
            confidence=0.1,
        )
    try:
        return RoutingDecision(
            format=str(data.get("format", "image")),
            doc_type_hint=str(data.get("doc_type_hint", "unknown")),
            layout_complexity=str(data.get("layout_complexity", "simple")),
            has_handwriting=bool(data.get("has_handwriting", False)),
            suggested_path=str(data.get("suggested_path", "vision")),
            confidence=float(data.get("confidence", 0.1)),
        )
    except Exception:
        return RoutingDecision(
            format="image",
            doc_type_hint="unknown",
            layout_complexity="simple",
            has_handwriting=False,
            suggested_path="vision",
            confidence=0.1,
        )


def parse_coverage_response(text: str) -> Dict[str, Any]:
    """Best-effort parse of coverage-verifier output."""
    try:
        data = json.loads(_first_json_object(text))
    except Exception:
        return {"complete": False, "missed_regions": [], "low_confidence_regions": []}
    missed = data.get("missed_regions") or []
    if not isinstance(missed, list):
        missed = []
    low_conf = data.get("low_confidence_regions") or []
    if not isinstance(low_conf, list):
        low_conf = []
    return {
        "complete": bool(data.get("complete", False)),
        "missed_regions": missed,
        "low_confidence_regions": low_conf,
    }
