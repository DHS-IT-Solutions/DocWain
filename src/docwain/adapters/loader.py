"""Domain adapter loader.

Fetches per-domain YAML from Azure Blob at:
    {BLOB_PREFIX}/{subscription_id}/{domain}.yaml  (per-subscription override)
    {BLOB_PREFIX}/global/{domain}.yaml             (global default)

Falls back to a baked-in generic seed YAML on any Blob failure. Parsed
adapters are cached in-process with a TTL.

Spec: feedback_adapter_yaml_blob.md + unified-docwain spec §5.5
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import re

import yaml

from src.docwain.adapters.schema import DomainAdapter


logger = logging.getLogger(__name__)


_SEED_DIR = Path(__file__).parent / "seed"


def _load_seed_yaml(domain: str) -> Optional[str]:
    path = _SEED_DIR / f"{domain}.yaml"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None


class AdapterLoader:
    def __init__(
        self,
        *,
        subscription_id: str = "",
        cache_ttl_seconds: Optional[int] = None,
        blob_prefix: Optional[str] = None,
    ):
        from src.api.config import Config
        da_cfg = getattr(Config, "DomainAdapters", None)
        self.subscription_id = subscription_id
        self.cache_ttl_seconds = (
            cache_ttl_seconds
            if cache_ttl_seconds is not None
            else (getattr(da_cfg, "CACHE_TTL_SECONDS", 300) if da_cfg else 300)
        )
        self.blob_prefix = (
            blob_prefix
            or (getattr(da_cfg, "BLOB_PREFIX", "sme_adapters") if da_cfg else "sme_adapters")
        )
        self._cache: Dict[str, tuple[float, DomainAdapter]] = {}

    def _fetch_from_blob(self, path: str) -> str:
        """Fetch a YAML text from Azure Blob. Raises FileNotFoundError if not found."""
        try:
            from azure.storage.blob import BlobServiceClient
            from src.api.config import Config
            import os
            conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
            if not conn_str:
                raise RuntimeError("AZURE_STORAGE_CONNECTION_STRING not set")
            svc = BlobServiceClient.from_connection_string(conn_str)
            da_cfg = getattr(Config, "DomainAdapters", None)
            container = getattr(da_cfg, "BLOB_CONTAINER", "docwain-configs") if da_cfg else "docwain-configs"
            blob = svc.get_blob_client(container=container, blob=path)
            return blob.download_blob().readall().decode("utf-8")
        except Exception as exc:
            # Re-raise as FileNotFoundError so callers can treat "not found" and
            # "blob down" uniformly.
            raise FileNotFoundError(f"blob {path}: {exc}") from exc

    @staticmethod
    def _sanitize_yaml(text: str) -> str:
        """Convert inline YAML flow sequences to block sequences so that strings
        containing YAML-special characters (e.g. '?' in questions) parse cleanly."""
        def _flow_seq_to_block(m: re.Match) -> str:
            key = m.group(1)
            # Split on commas that are not inside quotes, strip whitespace.
            raw_items = [item.strip() for item in m.group(2).split(",")]
            block_lines = "\n".join(f"  - {item}" for item in raw_items if item)
            return f"{key}:\n{block_lines}"

        return re.sub(
            r"^(\s*\w[\w_]*):\s*\[([^\]]+)\]\s*$",
            _flow_seq_to_block,
            text,
            flags=re.MULTILINE,
        )

    def _parse_yaml(self, yaml_text: str) -> DomainAdapter:
        try:
            data = yaml.safe_load(yaml_text)
        except yaml.YAMLError:
            # Retry after converting flow sequences to block sequences.
            data = yaml.safe_load(self._sanitize_yaml(yaml_text))
        data = data or {}
        if not isinstance(data, dict):
            raise ValueError(f"adapter YAML must be a mapping, got {type(data)}")
        return DomainAdapter(
            domain=str(data.get("domain", "generic")),
            version=str(data.get("version", "v1")),
            prompt_fragment=str(data.get("prompt_fragment", "")),
            key_entities=list(data.get("key_entities") or []),
            analysis_hints=dict(data.get("analysis_hints") or {}),
            questions_to_ask=list(data.get("questions_to_ask") or []),
        )

    def _cache_key(self, domain: str) -> str:
        return f"{self.subscription_id}:{domain}"

    def _get_cached(self, domain: str) -> Optional[DomainAdapter]:
        key = self._cache_key(domain)
        entry = self._cache.get(key)
        if not entry:
            return None
        expires_at, adapter = entry
        if time.time() > expires_at:
            self._cache.pop(key, None)
            return None
        return adapter

    def _put_cached(self, domain: str, adapter: DomainAdapter) -> None:
        self._cache[self._cache_key(domain)] = (time.time() + self.cache_ttl_seconds, adapter)

    def load(self, domain: str) -> DomainAdapter:
        """Return the best-available DomainAdapter for the given domain."""
        # Cache
        cached = self._get_cached(domain)
        if cached is not None:
            return cached

        # Subscription override
        if self.subscription_id:
            sub_path = f"{self.blob_prefix}/{self.subscription_id}/{domain}.yaml"
            try:
                text = self._fetch_from_blob(sub_path)
            except Exception:
                text = None
            if text is not None:
                try:
                    adapter = self._parse_yaml(text)
                    self._put_cached(domain, adapter)
                    return adapter
                except Exception as exc:
                    logger.warning("Failed to parse sub adapter YAML for %r: %s", domain, exc)

        # Global
        global_path = f"{self.blob_prefix}/global/{domain}.yaml"
        try:
            text = self._fetch_from_blob(global_path)
        except Exception:
            text = None
        if text is not None:
            try:
                adapter = self._parse_yaml(text)
                self._put_cached(domain, adapter)
                return adapter
            except Exception as exc:
                logger.warning("Failed to parse global adapter YAML for %r: %s", domain, exc)

        # Seed fallback — only `generic` is guaranteed to exist; any other
        # domain falls back to generic.
        seed_text = _load_seed_yaml(domain) or _load_seed_yaml("generic")
        if seed_text is None:
            logger.error("No seed YAML found for domain %r and no generic seed", domain)
            adapter = DomainAdapter()  # defaults
        else:
            try:
                adapter = self._parse_yaml(seed_text)
            except Exception as exc:
                logger.error("Failed to parse seed YAML for %r: %s", domain, exc)
                adapter = DomainAdapter()
        self._put_cached(domain, adapter)
        return adapter
