"""Vision-path smoke with a fully stubbed DocWain client — offline test of the
entire orchestrator pipeline (DocIntel → extractor → verifier → fallback)
without hitting a live vLLM endpoint.
"""
import io

import fitz
from PIL import Image

from src.extraction.canonical_schema import ExtractionResult
from src.extraction.vision.client import VisionResponse
from src.extraction.vision.fallback import FallbackRegionResult
from src.extraction.vision.orchestrator import extract_via_vision


def _scanned_pdf(num_pages: int = 1) -> bytes:
    d = fitz.open()
    for _ in range(num_pages):
        p = d.new_page()
        p.draw_rect(fitz.Rect(72, 72, 400, 400), color=(0.1, 0.1, 0.1))
    buf = io.BytesIO()
    d.save(buf)
    d.close()
    return buf.getvalue()


def _png_image() -> bytes:
    img = Image.new("RGB", (200, 100), (255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _stub_all(monkeypatch):
    def fake_call(self, *, system, user_text, image_bytes, image_mime="image/png",
                  max_tokens=1024, temperature=0.0):
        s = system.lower()
        if "classifier" in s or "routing decision" in s:
            return VisionResponse(
                text='{"format":"pdf_scanned","doc_type_hint":"unknown","layout_complexity":"simple",'
                     '"has_handwriting":false,"suggested_path":"vision","confidence":0.8}',
                prompt_tokens=5, completion_tokens=10, wall_ms=1.0, model="stub",
            )
        if "coverage verifier" in s or "verifier stage" in s:
            return VisionResponse(
                text='{"complete":false,"missed_regions":[{"bbox":[0.0,0.0,1.0,1.0],"description":"full"}],'
                     '"low_confidence_regions":[]}',
                prompt_tokens=5, completion_tokens=10, wall_ms=1.0, model="stub",
            )
        return VisionResponse(
            text='{"regions":[],"reading_order":[],"page_confidence":0.1}',
            prompt_tokens=5, completion_tokens=10, wall_ms=1.0, model="stub",
        )
    monkeypatch.setattr("src.extraction.vision.client.VisionClient.call", fake_call)

    def fake_fallback(img, *, bbox):
        return FallbackRegionResult(text="recovered content", agreement=1.0, engine_winner="both")
    monkeypatch.setattr("src.extraction.vision.orchestrator.run_fallback_ensemble", fake_fallback)


def test_vision_smoke_on_scanned_pdf(monkeypatch):
    _stub_all(monkeypatch)
    result = extract_via_vision(_scanned_pdf(), doc_id="s1", filename="scan.pdf", format_hint="pdf_scanned")
    assert isinstance(result, ExtractionResult)
    assert result.path_taken == "vision"
    assert result.pages
    assert any("recovered content" in b.text for b in result.pages[0].blocks)
    assert len(result.metadata.coverage.fallback_invocations) >= 1


def test_vision_smoke_on_image(monkeypatch):
    _stub_all(monkeypatch)
    result = extract_via_vision(_png_image(), doc_id="i1", filename="pic.png", format_hint="image")
    assert result.path_taken == "vision"
    assert result.pages
    assert any("recovered content" in b.text for b in result.pages[0].blocks)
