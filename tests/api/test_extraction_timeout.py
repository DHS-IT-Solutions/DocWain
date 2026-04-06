import threading
import time
import pytest
from unittest.mock import patch, MagicMock


def _slow_extract(doc_id, doc_data, conn_data):
    """Simulate a stuck extraction that takes forever."""
    time.sleep(600)
    return {"document_id": doc_id, "status": "EXTRACTION_COMPLETED"}


def test_per_document_timeout_fires():
    """A document that exceeds DOC_EXTRACTION_TIMEOUT_SECONDS should be marked FAILED."""
    from src.api.extraction_service import _extract_single_with_timeout

    result = _extract_single_with_timeout(
        doc_id="test_doc_123",
        doc_data={"name": "slow.pdf"},
        conn_data={},
        timeout_seconds=2,
        extract_fn=_slow_extract,
    )
    assert result["status"] == "EXTRACTION_FAILED"
    assert "timed out" in result.get("error", "").lower()


def test_normal_extraction_completes_within_timeout():
    """A fast extraction should return normally."""
    def _fast_extract(doc_id, doc_data, conn_data):
        return {"document_id": doc_id, "status": "EXTRACTION_COMPLETED"}

    from src.api.extraction_service import _extract_single_with_timeout

    result = _extract_single_with_timeout(
        doc_id="test_doc_456",
        doc_data={"name": "fast.pdf"},
        conn_data={},
        timeout_seconds=30,
        extract_fn=_fast_extract,
    )
    assert result["status"] == "EXTRACTION_COMPLETED"
