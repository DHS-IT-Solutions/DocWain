"""Abstract base class for secret backends."""

from __future__ import annotations

import abc
from typing import Optional


class SecretBackend(abc.ABC):
    """Interface every secret-storage backend must implement."""

    @abc.abstractmethod
    def get(self, key: str) -> Optional[str]:
        """Return the secret value for *key*, or ``None`` if absent."""

    @abc.abstractmethod
    def set(self, key: str, value: str, metadata: dict | None = None) -> None:
        """Persist *value* under *key* with optional metadata."""

    @abc.abstractmethod
    def delete(self, key: str) -> None:
        """Remove *key* from the backend."""

    @abc.abstractmethod
    def list_keys(self) -> list[str]:
        """Return all known key names."""

    @abc.abstractmethod
    def rotate(self, key: str, new_value: str) -> None:
        """Replace the current value of *key* with *new_value*."""

    @abc.abstractmethod
    def health_check(self) -> bool:
        """Return ``True`` when the backend is operational."""
