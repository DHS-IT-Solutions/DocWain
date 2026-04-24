"""DocWain vision-extraction prompt + response parser.

Prompts DocWain to return a structured regions JSON for a given page image.
Safe parsing: malformed output returns an empty VisionExtraction (no regions,
page_confidence=0) so the coverage verifier correctly flags everything as
missed and the fallback catches the full page.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List


EXTRACTOR_SYSTEM_PROMPT = (
    "You are DocWain's vision extractor. You receive a single document page as "
    "an image plus routing hints (document type, layout complexity, whether "
    "handwriting is present).\n\n"
    "Your task: emit a JSON object describing every visible content region on "
    "the page, preserving reading order, bboxes (normalized 0..1 coordinates), "
    "and the extracted content verbatim. You MUST NOT paraphrase, summarize, "
    "or add content not present on the page.\n\n"
    "Output ONLY valid JSON (no prose, no markdown fences):\n"
    "{\n"
    '  "regions": [\n'
    "    {\n"
    '      "type": "text_block" | "table" | "form_field" | "figure" | "handwriting",\n'
    '      "bbox": [x, y, w, h]  (normalized 0..1),\n'
    '      "content": (string for text; {"rows": [...]} for table; {"label": s, '
    '"value": v} for form_field; string caption for figure; string for handwriting),\n'
    '      "confidence": number between 0.0 and 1.0\n'
    "    }\n"
    "  ],\n"
    '  "reading_order": [region_index, ...],\n'
    '  "page_confidence": number between 0.0 and 1.0\n'
    "}\n\n"
    "Rules:\n"
    "- Cover every visible text or content region; missing regions are failures.\n"
    "- Do not hallucinate — if a region is unreadable, emit it with "
    'confidence < 0.5 so the coverage verifier can flag it.\n'
    "- Preserve exact text characters; do not correct spelling, expand "
    "abbreviations, or normalize whitespace beyond collapsing internal line "
    "breaks within a single text block."
)


@dataclass
class VisionExtraction:
    regions: List[Dict[str, Any]] = field(default_factory=list)
    reading_order: List[int] = field(default_factory=list)
    page_confidence: float = 0.0


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


def parse_extractor_response(text: str) -> VisionExtraction:
    try:
        data = json.loads(_first_json_object(text))
    except Exception:
        return VisionExtraction()
    regions = data.get("regions") or []
    if not isinstance(regions, list):
        regions = []
    reading_order = data.get("reading_order") or []
    if not isinstance(reading_order, list):
        reading_order = []
    try:
        page_conf = float(data.get("page_confidence", 0.0))
    except Exception:
        page_conf = 0.0
    return VisionExtraction(
        regions=regions,
        reading_order=reading_order,
        page_confidence=page_conf,
    )
