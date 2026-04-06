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


