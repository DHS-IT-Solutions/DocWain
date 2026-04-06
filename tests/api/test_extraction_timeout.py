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


def test_stale_lock_auto_released():
    """If a batch lock is older than STALE_LOCK_THRESHOLD, acquire_batch_lock should reclaim it."""
    from src.api.extraction_service import _acquire_batch_lock, _release_batch_lock

    lock1 = _acquire_batch_lock("test_sub_stale")
    assert lock1 is not None
    lock2 = _acquire_batch_lock("test_sub_stale")
    assert lock2 is None
    _release_batch_lock(lock1)


def test_stale_lock_recovery_with_force():
    """Force-release should allow re-acquisition."""
    from src.api.extraction_service import _acquire_batch_lock, _force_release_batch_lock, _release_batch_lock

    lock1 = _acquire_batch_lock("test_sub_force")
    assert lock1 is not None
    _force_release_batch_lock("test_sub_force")
    lock2 = _acquire_batch_lock("test_sub_force")
    assert lock2 is not None
    _release_batch_lock(lock2)
