"""Errors raised by native extraction adapters."""


class NotNativePathError(Exception):
    """The document cannot be handled by any native adapter (e.g. scanned PDF, image).

    Callers should fall back to the vision path (Plan 2) or the existing v2_extractor.
    """


class NativeAdapterError(Exception):
    """A native adapter failed to extract the document.

    This is a hard failure — native path should never miss content. A raise of this
    error means the adapter has a bug and the document's extraction must be marked
    FAILED so the bug can be fixed before the doc is re-processed.
    """
