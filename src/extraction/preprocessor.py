"""Pre-processing intelligence — applies image corrections before OCR/extraction."""

from __future__ import annotations

import logging
from typing import List

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Directive registry
# ---------------------------------------------------------------------------

_KNOWN_DIRECTIVES = {"deskew", "denoise", "contrast", "upscale"}


# ---------------------------------------------------------------------------
# DocumentPreprocessor
# ---------------------------------------------------------------------------

class DocumentPreprocessor:
    """Applies a sequence of image-correction steps driven by triage directives."""

    def preprocess(self, image: np.ndarray, directives: List[str]) -> np.ndarray:
        """Apply each directive in order and return the processed image.

        Parameters
        ----------
        image:
            Input image as a NumPy array (BGR or grayscale).
        directives:
            Ordered list of processing steps to apply.  Recognised values are
            ``"deskew"``, ``"denoise"``, ``"contrast"``, and ``"upscale"``.
            Unknown values are logged as warnings and skipped.

        Returns
        -------
        np.ndarray
            Processed image (same dtype as input unless a step converts it).
        """
        if not directives:
            return image

        result = image.copy()
        for directive in directives:
            if directive == "deskew":
                result = self._deskew(result)
            elif directive == "denoise":
                result = self._denoise(result)
            elif directive == "contrast":
                result = self._enhance_contrast(result)
            elif directive == "upscale":
                result = self._upscale(result)
            else:
                logger.warning("Unknown preprocessing directive: %r — skipping.", directive)

        return result

    # ------------------------------------------------------------------
    # Private processing steps
    # ------------------------------------------------------------------

    @staticmethod
    def _deskew(image: np.ndarray) -> np.ndarray:
        """Correct skew by detecting the dominant angle via minAreaRect.

        The rotation is only applied when the absolute angle is at least
        0.5 degrees, to avoid unnecessary resampling on straight pages.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()

        # Threshold to isolate foreground pixels
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        coords = np.column_stack(np.where(thresh > 0))
        if coords.size == 0:
            return image

        rect = cv2.minAreaRect(coords)
        angle = rect[-1]

        # minAreaRect returns angles in [-90, 0); normalise to (-45, 45]
        if angle < -45.0:
            angle += 90.0

        if abs(angle) < 0.5:
            return image

        h, w = image.shape[:2]
        centre = (w / 2.0, h / 2.0)
        rotation_matrix = cv2.getRotationMatrix2D(centre, angle, 1.0)
        rotated = cv2.warpAffine(
            image,
            rotation_matrix,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return rotated

    @staticmethod
    def _denoise(image: np.ndarray) -> np.ndarray:
        """Reduce noise with a bilateral filter that preserves edges."""
        return cv2.bilateralFilter(image, 9, 75, 75)

    @staticmethod
    def _enhance_contrast(image: np.ndarray) -> np.ndarray:
        """Apply CLAHE on the L channel of the LAB colour space.

        Works on both colour (BGR) and grayscale inputs.  Grayscale images
        are temporarily promoted to three-channel BGR for the LAB conversion.
        """
        was_gray = image.ndim == 2
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if was_gray else image.copy()

        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_channel)

        lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
        enhanced_bgr = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        if was_gray:
            return cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)
        return enhanced_bgr

    @staticmethod
    def _upscale(image: np.ndarray, factor: int = 2) -> np.ndarray:
        """Double the image dimensions using bicubic interpolation."""
        h, w = image.shape[:2]
        return cv2.resize(image, (w * factor, h * factor), interpolation=cv2.INTER_CUBIC)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def detect_language(text: str) -> str:
    """Detect the dominant language of *text*.

    Returns the ISO 639-1 language code (e.g. ``"en"``, ``"fr"``).
    Falls back to ``"en"`` on any detection failure (empty text, ambiguous
    input, or langdetect library errors).
    """
    try:
        from langdetect import detect  # type: ignore[import]
        return detect(text)
    except Exception:  # noqa: BLE001
        logger.warning("Language detection failed — defaulting to 'en'.")
        return "en"
