"""Adaptive document triage — classifies documents and routes them to extraction engines."""

from __future__ import annotations

import logging
from typing import List, Optional

from src.extraction.models import TriageResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TABLE_EXTENSIONS = {"xlsx", "xls", "csv", "tsv"}
_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "tiff", "bmp"}

# Engine weight profiles keyed by document_type.
# Keys inside each profile: structural, semantic, vision, v2
_ENGINE_WEIGHTS: dict[str, dict[str, float]] = {
    "clean_digital": {"structural": 0.9, "semantic": 0.8, "vision": 0.3, "v2": 0.7},
    "scanned":       {"structural": 0.3, "semantic": 0.5, "vision": 0.9, "v2": 0.8},
    "handwritten":   {"structural": 0.2, "semantic": 0.3, "vision": 0.7, "v2": 0.9},
    "mixed":         {"structural": 0.6, "semantic": 0.7, "vision": 0.7, "v2": 0.8},
    "table_heavy":   {"structural": 0.8, "semantic": 0.6, "vision": 0.5, "v2": 0.9},
}

# Confidence scores we assign per classification path (heuristic)
_CONFIDENCE: dict[str, float] = {
    "table_heavy":   0.95,
    "handwritten":   0.85,
    "scanned":       0.80,
    "clean_digital": 0.90,
    "mixed":         0.70,
}


# ---------------------------------------------------------------------------
# DocumentTriager
# ---------------------------------------------------------------------------

class DocumentTriager:
    """Classifies an incoming document and produces a TriageResult routing decision."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def triage(
        self,
        file_type: str,
        has_text_layer: bool = True,
        dpi: int = 300,
        noise_score: float = 0.0,
        page_count: int = 1,
        *,
        page_images: Optional[List] = None,
    ) -> TriageResult:
        """Classify the document and return a TriageResult.

        Parameters
        ----------
        file_type:
            File extension without leading dot, e.g. ``"pdf"``, ``"xlsx"``.
        has_text_layer:
            Whether the document has a machine-readable text layer.
        dpi:
            Scan or render resolution (dots per inch).
        noise_score:
            Estimated noise level in [0, 1].  Higher means noisier.
        page_count:
            Number of pages in the document.
        page_images:
            Optional list of per-page image arrays (unused internally;
            reserved for future per-page analysis).
        """
        ext = file_type.lower().lstrip(".")

        document_type = self._classify(ext, has_text_layer, noise_score)
        engine_weights = dict(_ENGINE_WEIGHTS[document_type])
        preprocessing = self._preprocessing_directives(
            dpi, noise_score, has_text_layer, document_type
        )
        page_types = [document_type] * page_count
        confidence = _CONFIDENCE[document_type]

        logger.debug(
            "Triage: ext=%s type=%s weights=%s directives=%s confidence=%.2f",
            ext, document_type, engine_weights, preprocessing, confidence,
        )

        return TriageResult(
            document_type=document_type,
            engine_weights=engine_weights,
            preprocessing_directives=preprocessing,
            page_types=page_types,
            confidence=confidence,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _classify(ext: str, has_text_layer: bool, noise_score: float) -> str:
        """Apply classification rules in priority order and return a document type."""

        # 1. Spreadsheet / tabular formats
        if ext in _TABLE_EXTENSIONS:
            return "table_heavy"

        # 2. No text layer — distinguish handwritten vs plain scanned
        if not has_text_layer:
            if noise_score > 0.5:
                return "handwritten"
            return "scanned"

        # 3. Image-only formats (always lack a real text layer, but has_text_layer
        #    defaults True, so catch them by extension)
        if ext in _IMAGE_EXTENSIONS:
            return "scanned"

        # 4. Has text layer — quality-based split
        if noise_score < 0.15:
            return "clean_digital"

        return "mixed"

    @staticmethod
    def _preprocessing_directives(
        dpi: int,
        noise_score: float,
        has_text_layer: bool,
        document_type: str,
    ) -> List[str]:
        """Build the list of preprocessing steps required for this document."""
        directives: List[str] = []

        if dpi < 150:
            directives.append("upscale")

        if noise_score > 0.2:
            directives.append("denoise")

        if document_type in {"scanned", "handwritten"}:
            directives.append("deskew")

        if noise_score > 0.3 and not has_text_layer:
            directives.append("contrast")

        return directives
