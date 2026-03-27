"""Environment-variable fallback backend (always available)."""

from __future__ import annotations

import logging
import os
from typing import Optional

from src.vault._base import SecretBackend

log = logging.getLogger(__name__)


class EnvFallbackBackend(SecretBackend):
    """Read-only backend that delegates to ``os.getenv``.

    This is the migration-compatibility layer: it requires zero setup and
    always works so that the vault chain never returns empty-handed when a
    variable exists in the process environment.
    """

    def get(self, key: str) -> Optional[str]:
        return os.getenv(key)

    def set(self, key: str, value: str, metadata: dict | None = None) -> None:
        log.warning("EnvFallbackBackend.set() is a no-op; secrets cannot be written to the environment safely.")

    def delete(self, key: str) -> None:
        log.warning("EnvFallbackBackend.delete() is a no-op.")

    def list_keys(self) -> list[str]:
        return list(os.environ.keys())

    def rotate(self, key: str, new_value: str) -> None:
        log.warning("EnvFallbackBackend.rotate() is a no-op.")

    def health_check(self) -> bool:
        return True
