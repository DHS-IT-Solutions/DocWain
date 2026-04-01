"""Simple in-memory TTL cache for secret values."""

from __future__ import annotations

import threading
import time
from typing import Optional


class TTLCache:
    """Thread-safe key-value cache with per-entry time-based expiry.

    Parameters
    ----------
    ttl:
        Default time-to-live in seconds for each entry (default 300 = 5 min).
    """

    def __init__(self, ttl: int = 300) -> None:
        self._ttl = ttl
        self._store: dict[str, tuple[str, float]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            value, expires_at = entry
            if time.monotonic() > expires_at:
                del self._store[key]
                return None
            return value

    def set(self, key: str, value: str) -> None:
        with self._lock:
            self._store[key] = (value, time.monotonic() + self._ttl)

    def invalidate(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
