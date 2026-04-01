"""SecretVault -- chain-of-responsibility orchestrator for secret retrieval."""

from __future__ import annotations

import logging
import os
from typing import Optional

from src.vault._base import SecretBackend
from src.vault._cache import TTLCache

log = logging.getLogger(__name__)


class VaultKeyError(KeyError):
    """Raised by ``SecretVault.require()`` when a mandatory key is missing."""


class SecretVault:
    """Try each backend in order until a value is found, caching the result.

    Parameters
    ----------
    backends:
        Ordered list of ``SecretBackend`` instances.  The first backend to
        return a non-``None`` value wins.
    cache_ttl:
        Time-to-live (seconds) for the in-memory cache.  Set to ``0`` to
        disable caching.
    """

    def __init__(self, backends: list[SecretBackend], cache_ttl: int = 300) -> None:
        self._backends = list(backends)
        self._cache = TTLCache(ttl=cache_ttl) if cache_ttl > 0 else None

    # -- public API ------------------------------------------------------

    def get(self, key: str, default: str = "") -> str:
        """Return the secret for *key* or *default* if not found anywhere."""
        # 1. cache hit?
        if self._cache is not None:
            cached = self._cache.get(key)
            if cached is not None:
                return cached

        # 2. walk the backend chain
        for backend in self._backends:
            try:
                value = backend.get(key)
            except Exception:
                log.debug("Backend %s raised on get(%s); skipping.", type(backend).__name__, key)
                continue
            if value is not None:
                if self._cache is not None:
                    self._cache.set(key, value)
                return value

        return default

    def require(self, key: str) -> str:
        """Like ``get()`` but raises ``VaultKeyError`` when the key is absent."""
        value = self.get(key, default="")
        if not value:
            raise VaultKeyError(f"Required secret '{key}' not found in any vault backend.")
        return value

    def set(self, key: str, value: str, backend_index: int = 0) -> None:
        """Write *value* into the backend at *backend_index*."""
        self._backends[backend_index].set(key, value)
        if self._cache is not None:
            self._cache.set(key, value)

    def rotate(self, key: str, new_value: str, backend_index: int = 0) -> None:
        """Rotate *key* to *new_value* and invalidate the cache entry."""
        self._backends[backend_index].rotate(key, new_value)
        if self._cache is not None:
            self._cache.invalidate(key)

    def health(self) -> dict[str, bool]:
        """Return per-backend health status."""
        return {type(b).__name__: b.health_check() for b in self._backends}

    # -- factory ---------------------------------------------------------

    @classmethod
    def from_env(cls) -> "SecretVault":
        """Auto-detect available backends from the environment."""
        backends: list[SecretBackend] = []

        # Local encrypted vault
        if os.getenv("VAULT_MASTER_KEY"):
            try:
                from src.vault._local import LocalEncryptedBackend

                backends.append(LocalEncryptedBackend())
                log.info("Vault: LocalEncryptedBackend enabled.")
            except Exception:
                log.warning("Vault: VAULT_MASTER_KEY set but LocalEncryptedBackend failed to init.", exc_info=True)

        # Azure Key Vault
        if os.getenv("AZURE_VAULT_URL"):
            try:
                from src.vault._azure_keyvault import AzureKeyVaultBackend

                backends.append(AzureKeyVaultBackend())
                log.info("Vault: AzureKeyVaultBackend enabled.")
            except Exception:
                log.warning("Vault: AZURE_VAULT_URL set but AzureKeyVaultBackend failed to init.", exc_info=True)

        # Env fallback is always last
        from src.vault._env_fallback import EnvFallbackBackend

        backends.append(EnvFallbackBackend())

        return cls(backends=backends)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_vault: Optional[SecretVault] = None


def get_vault() -> SecretVault:
    """Return (or lazily create) the global ``SecretVault`` singleton."""
    global _vault
    if _vault is None:
        _vault = SecretVault.from_env()
    return _vault


def get_secret(key: str, default: str = "") -> str:
    """Convenience wrapper: ``get_vault().get(key, default)``."""
    return get_vault().get(key, default)
