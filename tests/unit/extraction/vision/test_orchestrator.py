import io

import fitz

from src.extraction.canonical_schema import ExtractionResult
from src.extraction.vision.client import VisionResponse
from src.extraction.vision.extractor import VisionExtraction
from src.extraction.vision.orchestrator import extract_via_vision


def _make_pdf() -> bytes:
    d = fitz.open()
    p = d.new_page()
    p.insert_text((72, 72), "some text")
    buf = io.BytesIO()
    d.save(buf)
    d.close()
    return buf.getvalue()


def test_extract_via_vision_returns_canonical_result(monkeypatch):
    def fake_call(self, *, system, user_text, image_bytes, image_mime="image/png",
                  max_tokens=1024, temperature=0.0):
        s = system.lower()
        if "routing decision" in s or "classifier" in s or "routing" in s:
            return VisionResponse(
                text='{"format":"pdf_scanned","doc_type_hint":"unknown","layout_complexity":"simple",'
                     '"has_handwriting":false,"suggested_path":"vision","confidence":0.8}',
                prompt_tokens=10, completion_tokens=20, wall_ms=100.0, model="docwain-fast",
            )
        if "coverage verifier" in s:
            return VisionResponse(
                text='{"complete":true,"missed_regions":[],"low_confidence_regions":[]}',
                prompt_tokens=10, completion_tokens=20, wall_ms=100.0, model="docwain-fast",
            )
        return VisionResponse(
            text='{"regions":[{"type":"text_block","bbox":[0.1,0.1,0.3,0.1],'
                 '"content":"some text","confidence":0.95}],"reading_order":[0],"page_confidence":0.9}',
            prompt_tokens=10, completion_tokens=20, wall_ms=100.0, model="docwain-fast",
        )

    monkeypatch.setattr("src.extraction.vision.client.VisionClient.call", fake_call)

    pdf = _make_pdf()
    result = extract_via_vision(pdf, doc_id="dv1", filename="scan.pdf", format_hint="pdf_scanned")
    assert isinstance(result, ExtractionResult)
    assert result.format == "pdf_scanned"
    assert result.path_taken == "vision"
    assert len(result.pages) == 1
    assert any("some text" in b.text for b in result.pages[0].blocks)
    assert result.metadata.coverage.verifier_score >= 0.9


def test_extract_via_vision_invokes_fallback_on_missed_region(monkeypatch):
    call_counts = {"fallback": 0}

    def fake_call(self, *, system, user_text, image_bytes, image_mime="image/png",
                  max_tokens=1024, temperature=0.0):
        s = system.lower()
        if "coverage verifier" in s:
            return VisionResponse(
                text='{"complete":false,"missed_regions":[{"bbox":[0.0,0.0,1.0,0.5],'
                     '"description":"header"}],"low_confidence_regions":[]}',
                prompt_tokens=10, completion_tokens=20, wall_ms=100.0, model="docwain-fast",
            )
        if "classifier" in s or "routing" in s:
            return VisionResponse(
                text='{"format":"pdf_scanned","doc_type_hint":"unknown","layout_complexity":"simple",'
                     '"has_handwriting":false,"suggested_path":"vision","confidence":0.8}',
                prompt_tokens=10, completion_tokens=20, wall_ms=100.0, model="docwain-fast",
            )
        return VisionResponse(
            text='{"regions":[],"reading_order":[],"page_confidence":0.1}',
            prompt_tokens=10, completion_tokens=20, wall_ms=100.0, model="docwain-fast",
        )

    def fake_fallback(img, *, bbox):
        call_counts["fallback"] += 1
        from src.extraction.vision.fallback import FallbackRegionResult
        return FallbackRegionResult(text="recovered via fallback", agreement=1.0, engine_winner="both")

    monkeypatch.setattr("src.extraction.vision.client.VisionClient.call", fake_call)
    monkeypatch.setattr("src.extraction.vision.orchestrator.run_fallback_ensemble", fake_fallback)

    pdf = _make_pdf()
    result = extract_via_vision(pdf, doc_id="dv2", filename="scan.pdf", format_hint="pdf_scanned")
    assert call_counts["fallback"] >= 1
    assert any("recovered via fallback" in b.text for b in result.pages[0].blocks)
    assert len(result.metadata.coverage.fallback_invocations) >= 1
