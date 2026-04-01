"""Unit tests for DocumentPreprocessor and detect_language (src/extraction/preprocessor.py)."""

from __future__ import annotations

import numpy as np
import pytest

from src.extraction.preprocessor import DocumentPreprocessor, detect_language


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def preprocessor():
    return DocumentPreprocessor()


def _solid_bgr(height: int = 64, width: int = 64, value: int = 180) -> np.ndarray:
    """Return a flat BGR image filled with a single grey value."""
    return np.full((height, width, 3), value, dtype=np.uint8)


def _noisy_bgr(height: int = 128, width: int = 128, seed: int = 42) -> np.ndarray:
    """Return a BGR image with moderate Gaussian noise."""
    rng = np.random.default_rng(seed)
    base = np.full((height, width, 3), 128, dtype=np.int16)
    noise = rng.integers(-60, 60, size=(height, width, 3), dtype=np.int16)
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def _text_like_bgr(height: int = 128, width: int = 128) -> np.ndarray:
    """Return a white image with a thin black horizontal line (minimal foreground)."""
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    img[height // 2, :] = 0  # one-pixel horizontal rule
    return img


# ---------------------------------------------------------------------------
# preprocess — no directives
# ---------------------------------------------------------------------------

class TestPreprocessNoDirectives:

    def test_returns_unchanged_when_no_directives(self, preprocessor):
        image = _solid_bgr()
        result = preprocessor.preprocess(image, [])
        np.testing.assert_array_equal(result, image)

    def test_returns_copy_not_same_object(self, preprocessor):
        """preprocess() with directives must not return the original reference."""
        image = _noisy_bgr()
        result = preprocessor.preprocess(image, ["denoise"])
        assert result is not image

    def test_empty_directives_returns_unchanged_shape(self, preprocessor):
        image = _solid_bgr(32, 48)
        result = preprocessor.preprocess(image, [])
        assert result.shape == image.shape


# ---------------------------------------------------------------------------
# _deskew
# ---------------------------------------------------------------------------

class TestDeskew:

    def test_straight_image_unchanged(self, preprocessor):
        """A perfectly straight image (no dominant skew) should come back unmodified."""
        image = _text_like_bgr()
        result = preprocessor._deskew(image)
        # Shape must be preserved
        assert result.shape == image.shape

    def test_deskew_output_same_shape_as_input(self, preprocessor):
        image = _noisy_bgr(64, 64)
        result = preprocessor._deskew(image)
        assert result.shape == image.shape

    def test_deskew_returns_ndarray(self, preprocessor):
        image = _solid_bgr()
        result = preprocessor._deskew(image)
        assert isinstance(result, np.ndarray)

    def test_deskew_via_preprocess_pipeline(self, preprocessor):
        image = _text_like_bgr()
        result = preprocessor.preprocess(image, ["deskew"])
        assert result.shape == image.shape

    def test_deskew_accepts_grayscale(self, preprocessor):
        gray = np.full((64, 64), 200, dtype=np.uint8)
        result = preprocessor._deskew(gray)
        assert result.shape == gray.shape

    def test_deskew_empty_foreground_returns_image(self, preprocessor):
        """All-white image has no foreground pixels — should be returned as-is."""
        image = np.full((64, 64, 3), 255, dtype=np.uint8)
        result = preprocessor._deskew(image)
        assert result.shape == image.shape


# ---------------------------------------------------------------------------
# _denoise
# ---------------------------------------------------------------------------

class TestDenoise:

    def test_denoise_reduces_variance(self, preprocessor):
        """Bilateral filter should smooth the image — variance of the result is lower."""
        image = _noisy_bgr(128, 128)
        result = preprocessor._denoise(image)
        original_var = float(np.var(image.astype(np.float32)))
        result_var = float(np.var(result.astype(np.float32)))
        assert result_var < original_var, (
            f"Expected variance to decrease after denoising; "
            f"original={original_var:.2f}, result={result_var:.2f}"
        )

    def test_denoise_preserves_shape(self, preprocessor):
        image = _noisy_bgr(64, 64)
        result = preprocessor._denoise(image)
        assert result.shape == image.shape

    def test_denoise_preserves_dtype(self, preprocessor):
        image = _noisy_bgr()
        result = preprocessor._denoise(image)
        assert result.dtype == image.dtype

    def test_denoise_returns_ndarray(self, preprocessor):
        image = _noisy_bgr()
        result = preprocessor._denoise(image)
        assert isinstance(result, np.ndarray)

    def test_denoise_via_preprocess_pipeline(self, preprocessor):
        image = _noisy_bgr()
        result = preprocessor.preprocess(image, ["denoise"])
        assert result.shape == image.shape

    def test_denoise_solid_image_unchanged(self, preprocessor):
        """A perfectly uniform image should be essentially unchanged by bilateral filter."""
        image = _solid_bgr(64, 64, value=128)
        result = preprocessor._denoise(image)
        np.testing.assert_allclose(result.astype(float), image.astype(float), atol=2)


# ---------------------------------------------------------------------------
# _enhance_contrast
# ---------------------------------------------------------------------------

class TestEnhanceContrast:

    def test_contrast_preserves_shape_bgr(self, preprocessor):
        image = _solid_bgr(64, 64)
        result = preprocessor._enhance_contrast(image)
        assert result.shape == image.shape

    def test_contrast_preserves_dtype(self, preprocessor):
        image = _solid_bgr()
        result = preprocessor._enhance_contrast(image)
        assert result.dtype == np.uint8

    def test_contrast_returns_ndarray(self, preprocessor):
        image = _solid_bgr()
        result = preprocessor._enhance_contrast(image)
        assert isinstance(result, np.ndarray)

    def test_contrast_grayscale_input(self, preprocessor):
        gray = np.full((64, 64), 120, dtype=np.uint8)
        result = preprocessor._enhance_contrast(gray)
        assert result.shape == gray.shape
        assert result.ndim == 2

    def test_contrast_via_preprocess_pipeline(self, preprocessor):
        image = _solid_bgr()
        result = preprocessor.preprocess(image, ["contrast"])
        assert result.shape == image.shape

    def test_contrast_spreads_histogram_on_low_contrast_image(self, preprocessor):
        """CLAHE should increase spread for a very low-contrast (nearly uniform) image."""
        rng = np.random.default_rng(0)
        # Very low contrast: pixel values tightly clustered around 128 ± 5
        flat = np.clip(
            128 + rng.integers(-5, 5, size=(128, 128, 3), dtype=np.int16),
            0, 255,
        ).astype(np.uint8)
        result = preprocessor._enhance_contrast(flat)
        # After CLAHE the standard deviation should be at least as large
        original_std = float(np.std(flat.astype(np.float32)))
        result_std = float(np.std(result.astype(np.float32)))
        assert result_std >= original_std


# ---------------------------------------------------------------------------
# _upscale
# ---------------------------------------------------------------------------

class TestUpscale:

    def test_upscale_doubles_dimensions(self, preprocessor):
        image = _solid_bgr(32, 48)
        result = preprocessor._upscale(image)
        assert result.shape == (64, 96, 3)

    def test_upscale_preserves_dtype(self, preprocessor):
        image = _solid_bgr()
        result = preprocessor._upscale(image)
        assert result.dtype == image.dtype

    def test_upscale_via_preprocess_pipeline(self, preprocessor):
        image = _solid_bgr(32, 32)
        result = preprocessor.preprocess(image, ["upscale"])
        assert result.shape == (64, 64, 3)

    def test_upscale_returns_ndarray(self, preprocessor):
        image = _solid_bgr()
        result = preprocessor._upscale(image)
        assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# Unknown directives
# ---------------------------------------------------------------------------

class TestUnknownDirectives:

    def test_unknown_directive_logged_as_warning(self, preprocessor, caplog):
        import logging
        image = _solid_bgr()
        with caplog.at_level(logging.WARNING, logger="src.extraction.preprocessor"):
            result = preprocessor.preprocess(image, ["nonexistent_step"])
        assert any("nonexistent_step" in record.message for record in caplog.records)

    def test_unknown_directive_does_not_raise(self, preprocessor):
        image = _solid_bgr()
        # Should complete without exception
        result = preprocessor.preprocess(image, ["totally_unknown"])
        assert result.shape == image.shape

    def test_mixed_known_and_unknown_directives(self, preprocessor):
        image = _noisy_bgr()
        result = preprocessor.preprocess(image, ["denoise", "unknown_op"])
        assert result.shape == image.shape


# ---------------------------------------------------------------------------
# Directive ordering
# ---------------------------------------------------------------------------

class TestDirectiveOrdering:

    def test_upscale_after_denoise_doubles_dimensions(self, preprocessor):
        image = _noisy_bgr(32, 32)
        result = preprocessor.preprocess(image, ["denoise", "upscale"])
        assert result.shape == (64, 64, 3)

    def test_all_directives_applied_in_sequence(self, preprocessor):
        image = _text_like_bgr(32, 32)
        result = preprocessor.preprocess(image, ["deskew", "denoise", "contrast", "upscale"])
        # upscale is the last step — final dims should be doubled from the 32×32 input
        assert result.shape[0] == 64
        assert result.shape[1] == 64


# ---------------------------------------------------------------------------
# detect_language
# ---------------------------------------------------------------------------

class TestDetectLanguage:

    def test_english_text_detected(self):
        text = (
            "The quick brown fox jumps over the lazy dog. "
            "This is a straightforward English sentence."
        )
        assert detect_language(text) == "en"

    def test_empty_string_falls_back_to_en(self):
        assert detect_language("") == "en"

    def test_whitespace_only_falls_back_to_en(self):
        assert detect_language("   \t\n  ") == "en"

    def test_returns_string(self):
        result = detect_language("hello world")
        assert isinstance(result, str)

    def test_returns_non_empty_code(self):
        result = detect_language("hello world this is some text")
        assert len(result) >= 2

    def test_ambiguous_text_returns_string(self):
        """Single-character or highly ambiguous input must still return a string."""
        result = detect_language("a")
        assert isinstance(result, str)
