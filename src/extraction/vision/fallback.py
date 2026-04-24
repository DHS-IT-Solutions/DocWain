"""Region-scoped OCR fallback ensemble: Tesseract + EasyOCR.

Invoked by the vision path when the coverage verifier flags missed or low-
confidence regions. Runs on cropped page-image regions (not full pages) to
keep fallback scope narrow. DocWain-primary is preserved; fallback is
catch-all for what DocWain misses.

As DocWain training improves (Phase 2 workstream), fallback invocation rate
drops and eventually this code becomes unreachable. Retained for safety.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from PIL import Image


@dataclass
class FallbackRegionResult:
    text: str
    agreement: float
    engine_winner: str


def crop_bbox_normalized(img: Image.Image, *, bbox: List[float]) -> Image.Image:
    """Crop a PIL image by a normalized (0..1) bbox [x, y, w, h]."""
    if len(bbox) != 4:
        raise ValueError(f"bbox must have 4 elements, got {bbox}")
    x, y, w, h = bbox
    W, H = img.size
    left = max(0, int(x * W))
    top = max(0, int(y * H))
    right = min(W, int((x + w) * W))
    bottom = min(H, int((y + h) * H))
    return img.crop((left, top, right, bottom))


def _run_tesseract(img: Image.Image) -> str:
    try:
        import pytesseract
        return pytesseract.image_to_string(img) or ""
    except Exception:
        return ""


def _run_easyocr(img: Image.Image) -> str:
    try:
        import easyocr
        import numpy as np
        reader = _get_easyocr_reader()
        results = reader.readtext(np.array(img))
        lines = [r[1] for r in results]
        return "\n".join(lines)
    except Exception:
        return ""


_EASYOCR_READER = None


def _get_easyocr_reader():
    global _EASYOCR_READER
    if _EASYOCR_READER is None:
        import easyocr
        _EASYOCR_READER = easyocr.Reader(["en"], gpu=False)
    return _EASYOCR_READER


def _levenshtein_ratio(a: str, b: str) -> float:
    if a == b:
        return 1.0 if a else 1.0
    if not a or not b:
        return 0.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = cur
    denom = max(len(a), len(b))
    return 1.0 - (prev[-1] / denom) if denom else 1.0


def run_fallback_ensemble(img: Image.Image, *, bbox: List[float]) -> FallbackRegionResult:
    """Run both OCR engines on the cropped region; return merged result."""
    crop = crop_bbox_normalized(img, bbox=bbox)
    tess_text = _run_tesseract(crop).strip()
    easy_text = _run_easyocr(crop).strip()

    agreement = _levenshtein_ratio(tess_text, easy_text)
    if tess_text == easy_text and tess_text:
        return FallbackRegionResult(text=tess_text, agreement=agreement, engine_winner="both")
    if agreement > 0.9:
        return FallbackRegionResult(text=tess_text or easy_text, agreement=agreement, engine_winner="tesseract")
    if len(easy_text) > len(tess_text):
        return FallbackRegionResult(text=easy_text, agreement=agreement, engine_winner="easyocr")
    return FallbackRegionResult(text=tess_text, agreement=agreement, engine_winner="tesseract")
