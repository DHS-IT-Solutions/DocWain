from src.extraction.vision.fallback import (
    FallbackRegionResult,
    crop_bbox_normalized,
    run_fallback_ensemble,
)


def test_crop_bbox_normalized_returns_pil_crop():
    from PIL import Image
    img = Image.new("RGB", (100, 100), (255, 255, 255))
    crop = crop_bbox_normalized(img, bbox=[0.1, 0.2, 0.3, 0.4])
    assert crop.size == (30, 40)


def test_run_fallback_ensemble_stubs(monkeypatch):
    def fake_tess(img):
        return "hello from tesseract"

    def fake_easy(img):
        return "hello from tesseract"

    monkeypatch.setattr("src.extraction.vision.fallback._run_tesseract", fake_tess)
    monkeypatch.setattr("src.extraction.vision.fallback._run_easyocr", fake_easy)

    from PIL import Image
    img = Image.new("RGB", (100, 100), (255, 255, 255))
    result = run_fallback_ensemble(img, bbox=[0.0, 0.0, 1.0, 1.0])
    assert isinstance(result, FallbackRegionResult)
    assert result.text == "hello from tesseract"
    assert result.agreement > 0.9
    assert result.engine_winner in ("tesseract", "easyocr", "both")


def test_run_fallback_ensemble_picks_higher_confidence_when_disagreement(monkeypatch):
    monkeypatch.setattr("src.extraction.vision.fallback._run_tesseract", lambda img: "tess output")
    monkeypatch.setattr("src.extraction.vision.fallback._run_easyocr", lambda img: "easyocr output is longer here")

    from PIL import Image
    img = Image.new("RGB", (100, 100), (255, 255, 255))
    result = run_fallback_ensemble(img, bbox=[0.0, 0.0, 1.0, 1.0])
    assert "longer" in result.text
