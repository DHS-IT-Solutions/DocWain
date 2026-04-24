"""Image-format "adapter" — a pass-through that always raises NotNativePathError.

Image files (JPG, PNG, TIFF) have no native text layer. They must go through
the vision path. This adapter exists so the dispatcher's routing table can map
image extensions explicitly (distinguishing them from genuinely-unknown
extensions) while still triggering the vision-path fallthrough.
"""
from __future__ import annotations

from src.extraction.adapters.errors import NotNativePathError


def extract_image_native(file_bytes: bytes, *, doc_id: str, filename: str):
    raise NotNativePathError(
        f"image file {filename!r} routes to vision path (no native extraction applicable)"
    )
