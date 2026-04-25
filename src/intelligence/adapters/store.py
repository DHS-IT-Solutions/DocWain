"""AdapterStore — resolves and caches adapter YAMLs.

Resolution order:
  1. sme_adapters/subscription/{subscription_id}/{domain}.yaml
  2. sme_adapters/global/{domain}.yaml
  3. sme_adapters/global/generic.yaml  (always succeeds)

Cache: in-memory TTL (default 5 min). Failure mode: serve last cached
version; if no cache, fall back to generic.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional, Protocol

from src.intelligence.adapters.schema import Adapter, parse_adapter_yaml

logger = logging.getLogger(__name__)


class AdapterNotFound(Exception):
    pass


class AdapterBackend(Protocol):
    def get_text(self, key: str) -> str: ...


@dataclass
class _CacheEntry:
    adapter: Adapter
    loaded_at: float


class AdapterStore:
    def __init__(self, *, backend: AdapterBackend, cache_ttl_seconds: int = 300):
        self._backend = backend
        self._ttl = cache_ttl_seconds
        self._cache: Dict[str, _CacheEntry] = {}

    def get(self, *, domain: str, subscription_id: str) -> Adapter:
        # Try subscription override first, then global, then generic
        for key in self._candidate_keys(domain=domain, subscription_id=subscription_id):
            adapter = self._load_or_cached(key)
            if adapter is not None:
                return adapter
        adapter = self._load_or_cached("sme_adapters/global/generic.yaml")
        if adapter is None:
            raise AdapterNotFound("generic adapter is missing — install required")
        return adapter

    def invalidate(self, *, domain: Optional[str] = None) -> None:
        if domain is None:
            self._cache.clear()
            return
        for key in list(self._cache.keys()):
            if f"/{domain}.yaml" in key:
                del self._cache[key]

    def _candidate_keys(self, *, domain: str, subscription_id: str):
        if subscription_id:
            yield f"sme_adapters/subscription/{subscription_id}/{domain}.yaml"
        yield f"sme_adapters/global/{domain}.yaml"

    def _load_or_cached(self, key: str) -> Optional[Adapter]:
        entry = self._cache.get(key)
        now = time.monotonic()
        if entry is not None and (now - entry.loaded_at) < self._ttl:
            return entry.adapter
        try:
            text = self._backend.get_text(key)
        except AdapterNotFound:
            return None
        except Exception as exc:
            logger.warning("adapter blob fetch failed for %s: %s", key, exc)
            return entry.adapter if entry is not None else None
        try:
            adapter = parse_adapter_yaml(text)
        except Exception as exc:
            logger.error("adapter YAML parse failed for %s: %s", key, exc)
            return entry.adapter if entry is not None else None
        self._cache[key] = _CacheEntry(adapter=adapter, loaded_at=now)
        return adapter


class FilesystemAdapterBackend:
    """Local-filesystem backend for tests + early dev. Production uses Blob."""

    def __init__(self, *, root: str):
        self._root = root

    def get_text(self, key: str) -> str:
        path = os.path.join(self._root, key)
        if not os.path.exists(path):
            raise AdapterNotFound(key)
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()


class AzureBlobAdapterBackend:
    """Azure Blob backend. Lazy-imports Azure SDK so tests don't need it."""

    def __init__(self, *, container: str, connection_string: str):
        self._container = container
        self._connection_string = connection_string
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from azure.storage.blob import BlobServiceClient
            self._client = BlobServiceClient.from_connection_string(
                self._connection_string
            )
        return self._client

    def get_text(self, key: str) -> str:
        client = self._ensure_client()
        blob = client.get_blob_client(container=self._container, blob=key)
        try:
            data = blob.download_blob().readall()
        except Exception as exc:
            raise AdapterNotFound(key) from exc
        return data.decode("utf-8")


def resolve_default_backend(*, blob_root: str) -> AdapterBackend:
    """Return the right backend for current environment.

    With ADAPTER_BLOB_LOADING_ENABLED=true and Azure config present, use
    AzureBlobAdapterBackend. Otherwise fall back to filesystem (the
    always-safe path matching ADAPTER_GENERIC_FALLBACK_ENABLED behavior).
    """
    from src.api.config import insight_flag_enabled

    if insight_flag_enabled("ADAPTER_BLOB_LOADING_ENABLED"):
        container = os.environ.get("ADAPTER_BLOB_CONTAINER")
        conn = os.environ.get("ADAPTER_BLOB_CONNECTION")
        if container and conn:
            return AzureBlobAdapterBackend(
                container=container, connection_string=conn
            )
    return FilesystemAdapterBackend(root=blob_root)
