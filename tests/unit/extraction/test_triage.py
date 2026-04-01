"""Unit tests for DocumentTriager (src/extraction/triage.py)."""

import pytest

from src.extraction.triage import DocumentTriager
from src.extraction.models import TriageResult


@pytest.fixture
def triager():
    return DocumentTriager()


# ---------------------------------------------------------------------------
# Classification tests
# ---------------------------------------------------------------------------

class TestClassification:

    def test_clean_digital_pdf(self, triager):
        result = triager.triage("pdf", has_text_layer=True, noise_score=0.05)
        assert isinstance(result, TriageResult)
        assert result.document_type == "clean_digital"

    def test_clean_digital_noise_boundary_below(self, triager):
        # noise < 0.15 with text layer -> clean_digital
        result = triager.triage("pdf", has_text_layer=True, noise_score=0.14)
        assert result.document_type == "clean_digital"

    def test_clean_digital_noise_boundary_at_threshold(self, triager):
        # noise == 0.15 -> mixed (not < 0.15)
        result = triager.triage("pdf", has_text_layer=True, noise_score=0.15)
        assert result.document_type == "mixed"

    def test_mixed_document(self, triager):
        result = triager.triage("pdf", has_text_layer=True, noise_score=0.4)
        assert result.document_type == "mixed"

    def test_scanned_no_text_layer(self, triager):
        result = triager.triage("pdf", has_text_layer=False, noise_score=0.1)
        assert result.document_type == "scanned"

    def test_handwritten_high_noise(self, triager):
        result = triager.triage("pdf", has_text_layer=False, noise_score=0.7)
        assert result.document_type == "handwritten"

    def test_handwritten_noise_boundary_above(self, triager):
        # noise > 0.5 and no text layer -> handwritten
        result = triager.triage("pdf", has_text_layer=False, noise_score=0.51)
        assert result.document_type == "handwritten"

    def test_handwritten_noise_at_boundary(self, triager):
        # noise == 0.5 is NOT > 0.5 -> scanned
        result = triager.triage("pdf", has_text_layer=False, noise_score=0.5)
        assert result.document_type == "scanned"

    @pytest.mark.parametrize("ext", ["xlsx", "xls", "csv", "tsv"])
    def test_table_heavy_extensions(self, triager, ext):
        result = triager.triage(ext)
        assert result.document_type == "table_heavy"

    @pytest.mark.parametrize("ext", ["png", "jpg", "jpeg", "tiff", "bmp"])
    def test_scanned_image_extensions(self, triager, ext):
        # Image files classified as scanned even when has_text_layer=True
        result = triager.triage(ext, has_text_layer=True, noise_score=0.0)
        assert result.document_type == "scanned"

    def test_file_type_with_leading_dot(self, triager):
        # Extension normalisation
        result = triager.triage(".pdf", has_text_layer=True, noise_score=0.05)
        assert result.document_type == "clean_digital"

    def test_file_type_uppercase(self, triager):
        result = triager.triage("PDF", has_text_layer=True, noise_score=0.05)
        assert result.document_type == "clean_digital"


# ---------------------------------------------------------------------------
# Engine weights tests
# ---------------------------------------------------------------------------

class TestEngineWeights:

    def test_clean_digital_weights(self, triager):
        result = triager.triage("pdf", has_text_layer=True, noise_score=0.05)
        assert result.engine_weights == {
            "structural": 0.9, "semantic": 0.8, "vision": 0.3, "v2": 0.7
        }

    def test_scanned_weights(self, triager):
        result = triager.triage("pdf", has_text_layer=False, noise_score=0.1)
        assert result.engine_weights == {
            "structural": 0.3, "semantic": 0.5, "vision": 0.9, "v2": 0.8
        }

    def test_handwritten_weights(self, triager):
        result = triager.triage("pdf", has_text_layer=False, noise_score=0.8)
        assert result.engine_weights == {
            "structural": 0.2, "semantic": 0.3, "vision": 0.7, "v2": 0.9
        }

    def test_mixed_weights(self, triager):
        result = triager.triage("pdf", has_text_layer=True, noise_score=0.5)
        assert result.engine_weights == {
            "structural": 0.6, "semantic": 0.7, "vision": 0.7, "v2": 0.8
        }

    def test_table_heavy_weights(self, triager):
        result = triager.triage("xlsx")
        assert result.engine_weights == {
            "structural": 0.8, "semantic": 0.6, "vision": 0.5, "v2": 0.9
        }

    def test_engine_weights_are_independent_copies(self, triager):
        """Mutating one result's weights must not affect subsequent calls."""
        r1 = triager.triage("pdf", has_text_layer=True, noise_score=0.05)
        r1.engine_weights["structural"] = 0.0
        r2 = triager.triage("pdf", has_text_layer=True, noise_score=0.05)
        assert r2.engine_weights["structural"] == 0.9


# ---------------------------------------------------------------------------
# Preprocessing directives tests
# ---------------------------------------------------------------------------

class TestPreprocessingDirectives:

    def test_low_dpi_triggers_upscale(self, triager):
        result = triager.triage("pdf", has_text_layer=True, dpi=100)
        assert "upscale" in result.preprocessing_directives

    def test_dpi_at_threshold_no_upscale(self, triager):
        # dpi == 150 is NOT < 150 -> no upscale
        result = triager.triage("pdf", has_text_layer=True, dpi=150, noise_score=0.05)
        assert "upscale" not in result.preprocessing_directives

    def test_dpi_above_threshold_no_upscale(self, triager):
        result = triager.triage("pdf", has_text_layer=True, dpi=300, noise_score=0.05)
        assert "upscale" not in result.preprocessing_directives

    def test_high_noise_triggers_denoise(self, triager):
        result = triager.triage("pdf", has_text_layer=True, noise_score=0.3)
        assert "denoise" in result.preprocessing_directives

    def test_noise_at_denoise_boundary(self, triager):
        # noise == 0.2 is NOT > 0.2 -> no denoise
        result = triager.triage("pdf", has_text_layer=True, noise_score=0.2, dpi=300)
        assert "denoise" not in result.preprocessing_directives

    def test_scanned_triggers_deskew(self, triager):
        result = triager.triage("pdf", has_text_layer=False, noise_score=0.1)
        assert "deskew" in result.preprocessing_directives

    def test_handwritten_triggers_deskew(self, triager):
        result = triager.triage("pdf", has_text_layer=False, noise_score=0.8)
        assert "deskew" in result.preprocessing_directives

    def test_clean_digital_no_deskew(self, triager):
        result = triager.triage("pdf", has_text_layer=True, noise_score=0.05)
        assert "deskew" not in result.preprocessing_directives

    def test_contrast_triggered_high_noise_no_text_layer(self, triager):
        result = triager.triage("pdf", has_text_layer=False, noise_score=0.6)
        assert "contrast" in result.preprocessing_directives

    def test_contrast_not_triggered_with_text_layer(self, triager):
        # noise > 0.3 but has text layer -> no contrast
        result = triager.triage("pdf", has_text_layer=True, noise_score=0.6)
        assert "contrast" not in result.preprocessing_directives

    def test_contrast_not_triggered_low_noise(self, triager):
        # no text layer but noise <= 0.3 -> no contrast
        result = triager.triage("pdf", has_text_layer=False, noise_score=0.3)
        assert "contrast" not in result.preprocessing_directives

    def test_multiple_directives_combined(self, triager):
        # low dpi + high noise + no text layer -> upscale + denoise + deskew + contrast
        result = triager.triage("pdf", has_text_layer=False, dpi=100, noise_score=0.8)
        assert set(result.preprocessing_directives) >= {"upscale", "denoise", "deskew", "contrast"}

    def test_clean_document_no_directives(self, triager):
        result = triager.triage("pdf", has_text_layer=True, dpi=300, noise_score=0.05)
        assert result.preprocessing_directives == []


# ---------------------------------------------------------------------------
# Page types & other fields
# ---------------------------------------------------------------------------

class TestPageTypesAndMeta:

    def test_page_types_length_matches_page_count(self, triager):
        result = triager.triage("pdf", has_text_layer=True, noise_score=0.05, page_count=7)
        assert len(result.page_types) == 7

    def test_page_types_all_same_document_type(self, triager):
        result = triager.triage("pdf", has_text_layer=True, noise_score=0.05, page_count=3)
        assert all(pt == result.document_type for pt in result.page_types)

    def test_single_page_default(self, triager):
        result = triager.triage("pdf", has_text_layer=True, noise_score=0.05)
        assert len(result.page_types) == 1

    def test_confidence_is_float_in_range(self, triager):
        for ext, kw in [
            ("pdf", {"has_text_layer": True, "noise_score": 0.05}),
            ("pdf", {"has_text_layer": False, "noise_score": 0.1}),
            ("pdf", {"has_text_layer": False, "noise_score": 0.8}),
            ("pdf", {"has_text_layer": True, "noise_score": 0.5}),
            ("xlsx", {}),
        ]:
            result = triager.triage(ext, **kw)
            assert 0.0 <= result.confidence <= 1.0, (
                f"confidence {result.confidence} out of range for {ext} {kw}"
            )

    def test_page_images_accepted_without_error(self, triager):
        """page_images kwarg must be accepted even if not used."""
        result = triager.triage(
            "pdf", has_text_layer=True, noise_score=0.05,
            page_count=2, page_images=["img1", "img2"]
        )
        assert result.document_type == "clean_digital"
