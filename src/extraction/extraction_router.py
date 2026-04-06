"""Smart extraction routing by file type and size.

Routes documents to the most efficient extraction pipeline based on
filename extension and content size.
"""

import os
from dataclasses import dataclass
from typing import Literal

ExtractionMethod = Literal[
    "native_structured", "standard", "parallel_pages", "background_bulk", "vision"
]

# Average bytes per page for PDF estimation
_PDF_BYTES_PER_PAGE = 50_000

# PDF page-count thresholds
_PDF_PARALLEL_THRESHOLD = 5   # pages
_PDF_BULK_THRESHOLD = 50      # pages

_NATIVE_STRUCTURED_EXTENSIONS = {".csv", ".tsv", ".xlsx", ".xls"}
_DIRECT_TEXT_EXTENSIONS = {".txt", ".md"}
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff"}


@dataclass
class ExtractionRoute:
    method: ExtractionMethod
    reason: str
    estimated_seconds: float  # rough estimate for progress reporting


def route_extraction(filename: str, content_size_bytes: int) -> ExtractionRoute:
    """Determine the best extraction method for a document.

    Args:
        filename: Original filename including extension.
        content_size_bytes: Size of the file content in bytes.

    Returns:
        ExtractionRoute with chosen method, reason, and time estimate.
    """
    ext = os.path.splitext(filename)[1].lower()

    if ext in _NATIVE_STRUCTURED_EXTENSIONS:
        return ExtractionRoute(
            method="native_structured",
            reason=f"Structured file ({ext}) — using native parsers",
            estimated_seconds=2.0,
        )

    if ext in _DIRECT_TEXT_EXTENSIONS:
        return ExtractionRoute(
            method="standard",
            reason=f"Plain text file ({ext}) — direct chunking",
            estimated_seconds=1.0,
        )

    if ext == ".docx":
        return ExtractionRoute(
            method="standard",
            reason="Word document — python-docx extraction",
            estimated_seconds=4.0,
        )

    if ext in _IMAGE_EXTENSIONS:
        return ExtractionRoute(
            method="vision",
            reason=f"Image file ({ext}) — vision model extraction",
            estimated_seconds=5.0,
        )

    if ext == ".pdf":
        estimated_pages = max(1, content_size_bytes / _PDF_BYTES_PER_PAGE)
        if estimated_pages > _PDF_BULK_THRESHOLD:
            return ExtractionRoute(
                method="background_bulk",
                reason=f"Large PDF (~{int(estimated_pages)} pages) — background bulk processing",
                estimated_seconds=estimated_pages * 1.5,
            )
        if estimated_pages >= _PDF_PARALLEL_THRESHOLD:
            return ExtractionRoute(
                method="parallel_pages",
                reason=f"Medium PDF (~{int(estimated_pages)} pages) — parallel page extraction",
                estimated_seconds=max(30.0, estimated_pages * 1.2),
            )
        return ExtractionRoute(
            method="standard",
            reason=f"Small PDF (~{int(estimated_pages)} pages) — standard extraction",
            estimated_seconds=max(10.0, estimated_pages * 5.0),
        )

    # Unknown extension — safe default
    return ExtractionRoute(
        method="standard",
        reason=f"Unknown file type ({ext or 'no extension'}) — defaulting to standard extraction",
        estimated_seconds=10.0,
    )
