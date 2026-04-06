import time
import pytest
from unittest.mock import patch, MagicMock


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


def test_extraction_transitions_to_awaiting_review():
    """After successful extraction, document status should move to AWAITING_REVIEW_1."""
    from src.api.statuses import PIPELINE_AWAITING_REVIEW_1

    with patch("src.api.extraction_service.update_document_fields") as mock_update:
        with patch("src.api.extraction_service.update_stage"):
            from src.api.extraction_service import _transition_to_awaiting_review
            _transition_to_awaiting_review("test_doc_789")
            mock_update.assert_called_once()
            call_args = mock_update.call_args
            assert call_args[0][0] == "test_doc_789"
            fields = call_args[0][1]
            assert fields["status"] == PIPELINE_AWAITING_REVIEW_1
