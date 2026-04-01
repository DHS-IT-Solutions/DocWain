"""Azure Key Vault backend."""

from __future__ import annotations

import logging
import os
from typing import Optional

from src.vault._base import SecretBackend

log = logging.getLogger(__name__)


def _to_azure_name(key: str) -> str:
    """Azure Key Vault secret names only allow alphanumerics and hyphens."""
    return key.replace("_", "-")


def _from_azure_name(name: str) -> str:
    """Reverse the Azure name mapping back to underscore-delimited env-style keys."""
    return name.replace("-", "_")


class AzureKeyVaultBackend(SecretBackend):
    """Backend that reads/writes secrets in Azure Key Vault.

    Activated when ``AZURE_VAULT_URL`` is set.  Requires the
    ``azure-keyvault-secrets`` and ``azure-identity`` packages.
    """

    def __init__(self, vault_url: str | None = None) -> None:
        url = vault_url or os.getenv("AZURE_VAULT_URL", "")
        if not url:
            raise ValueError("AZURE_VAULT_URL is required for AzureKeyVaultBackend")

        try:
            from azure.identity import DefaultAzureCredential
            from azure.keyvault.secrets import SecretClient
        except ImportError as exc:
            raise ImportError(
                "azure-keyvault-secrets and azure-identity are required. "
                "Install them with: pip install azure-keyvault-secrets azure-identity"
            ) from exc

        credential = DefaultAzureCredential()
        self._client = SecretClient(vault_url=url, credential=credential)
        self._vault_url = url

    def get(self, key: str) -> Optional[str]:
        try:
            secret = self._client.get_secret(_to_azure_name(key))
            return secret.value
        except Exception:
            log.debug("Azure KV: key '%s' not found or inaccessible.", key)
            return None

    def set(self, key: str, value: str, metadata: dict | None = None) -> None:
        try:
            tags = {str(k): str(v) for k, v in (metadata or {}).items()}
            self._client.set_secret(_to_azure_name(key), value, tags=tags or None)
        except Exception:
            log.exception("Azure KV: failed to set key '%s'.", key)
            raise

    def delete(self, key: str) -> None:
        try:
            self._client.begin_delete_secret(_to_azure_name(key))
        except Exception:
            log.exception("Azure KV: failed to delete key '%s'.", key)
            raise

    def list_keys(self) -> list[str]:
        try:
            props = self._client.list_properties_of_secrets()
            return [_from_azure_name(p.name) for p in props]
        except Exception:
            log.exception("Azure KV: failed to list keys.")
            return []

    def rotate(self, key: str, new_value: str) -> None:
        self.set(key, new_value)
        log.info("Rotated secret '%s' in Azure Key Vault.", key)

    def health_check(self) -> bool:
        try:
            # A lightweight call to verify connectivity.
            list(self._client.list_properties_of_secrets(max_page_size=1))
            return True
        except Exception:
            log.debug("Azure Key Vault health check failed.")
            return False
