"""AdapterLoader: Blob fetch with TTL cache + layered fallback (spec §5).

Resolution order: subscription override → global domain → global generic.
Blob unreachable → last cached entry (health marked degraded); no cache → the
embedded last-resort YAML shipped at ``deploy/sme_adapters/last_resort/generic.yaml``.

Per ERRATA §1 the canonical method is :meth:`AdapterLoader.load` (``.get`` is a
kept alias); every loaded :class:`Adapter` carries ``content_hash`` and
``source_path`` populated on the model instance itself. A process-wide factory
pair :func:`get_adapter_loader` / :func:`init_adapter_loader` gives non-FastAPI
callers access to the same singleton that FastAPI lifespan wires into
``app.state``.
"""
from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import yaml

from src.intelligence.sme.adapter_schema import Adapter

_GLOBAL = "sme_adapters/global"
_SUB = "sme_adapters/subscription"


class AdapterLoadError(RuntimeError):
    """Raised when no adapter can be resolved (even through fallback)."""


class BlobReader(Protocol):
    """Minimal Blob surface used by the loader. Implementations raise
    ``FileNotFoundError`` on missing path, ``ConnectionError`` / ``OSError`` on
    transport failure. No ``timeout`` argument: per spec §3 invariant 8, no
    DocWain-layer wall-clock cutoffs on internal ops; the underlying Azure SDK
    enforces its own per-operation safety timeout."""

    def read_text(self, path: str) -> str: ...


@dataclass
class _Cached:
    """Internal cache entry: parsed adapter + bookkeeping for TTL/health."""

    adapter: Adapter
    loaded_at: float
    version: str
    content_hash: str
    source_path: str


def _hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


class AdapterLoader:
    """Per-process adapter resolver with TTL cache + layered fallback.

    ``ttl_seconds`` is *cache freshness*, NOT a fetch timeout (spec §3 inv. 8).
    """

    def __init__(
        self,
        *,
        blob: BlobReader,
        last_resort_path: Path,
        ttl_seconds: float = 300.0,
    ) -> None:
        if not last_resort_path.exists():
            raise AdapterLoadError(
                f"Embedded last-resort missing: {last_resort_path}"
            )
        self._blob = blob
        self._lr = last_resort_path
        self._ttl = ttl_seconds
        self._cache: dict[tuple[str, str], _Cached] = {}
        self._lock = threading.Lock()
        self._status = "healthy"

    def load(self, sub_id: str, domain: str) -> Adapter:
        """Resolve the adapter for ``(sub_id, domain)``.

        Returns an :class:`Adapter` with ``content_hash`` + ``source_path``
        populated on the instance itself (ERRATA §1). On Blob failure returns
        the last cached entry if available, else the embedded last-resort
        adapter, and marks ``health_status() == "degraded"``.
        """
        key = (sub_id, domain)
        now = time.monotonic()
        with self._lock:
            entry = self._cache.get(key)
            if entry is not None and (now - entry.loaded_at) < self._ttl:
                return entry.adapter

        try:
            fresh = self._fetch(sub_id, domain)
            with self._lock:
                self._cache[key] = fresh
                self._status = "healthy"
            return fresh.adapter
        except (ConnectionError, OSError):
            with self._lock:
                stale = self._cache.get(key)
                if stale is not None:
                    self._status = "degraded"
                    return stale.adapter
            last_resort = self._load_last_resort()
            with self._lock:
                self._cache[key] = last_resort
                self._status = "degraded"
            return last_resort.adapter

    # Back-compat alias — ERRATA §1 keeps ``.get`` as a thin alias.
    get = load

    def invalidate(self, sub_id: str, domain: str) -> None:
        """Drop a single cache entry so the next ``load`` refetches."""
        with self._lock:
            self._cache.pop((sub_id, domain), None)

    def invalidate_all(self) -> None:
        """Drop every cached entry (admin endpoint uses this on bulk updates)."""
        with self._lock:
            self._cache.clear()

    def last_load_metadata(self, sub_id: str, domain: str) -> dict[str, str]:
        """Return ``{version, content_hash, source_path}`` or ``{}`` if uncached."""
        with self._lock:
            entry = self._cache.get((sub_id, domain))
            if entry is None:
                return {}
            return {
                "version": entry.version,
                "content_hash": entry.content_hash,
                "source_path": entry.source_path,
            }

    def health_status(self) -> str:
        """``"healthy"`` when the last Blob fetch succeeded; ``"degraded"``
        otherwise (after a successful refetch returns to ``"healthy"``)."""
        with self._lock:
            return self._status

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _fetch(self, sub_id: str, domain: str) -> _Cached:
        paths = [
            f"{_SUB}/{sub_id}/{domain}.yaml",
            f"{_GLOBAL}/{domain}.yaml",
            f"{_GLOBAL}/generic.yaml",
        ]
        last: Exception | None = None
        for path in paths:
            try:
                raw = self._blob.read_text(path)
            except FileNotFoundError as exc:
                last = exc
                continue
            adapter = self._parse(raw, source_path=path)
            return _Cached(
                adapter=adapter,
                loaded_at=time.monotonic(),
                version=adapter.version,
                content_hash=_hash(raw),
                source_path=path,
            )
        raise AdapterLoadError(f"No adapter resolvable (last: {last})")

    def _load_last_resort(self) -> _Cached:
        raw = self._lr.read_text()
        source = f"embedded:{self._lr.name}"
        adapter = self._parse(raw, source_path=source)
        return _Cached(
            adapter=adapter,
            loaded_at=time.monotonic(),
            version=adapter.version,
            content_hash=_hash(raw),
            source_path=source,
        )

    @staticmethod
    def _parse(raw: str, *, source_path: str) -> Adapter:
        data = yaml.safe_load(raw)
        if not isinstance(data, dict):
            raise AdapterLoadError(
                f"Adapter at {source_path} is not a YAML mapping"
            )
        adapter = Adapter(**data)
        # Populate runtime-injected fields per ERRATA §1.
        return adapter.model_copy(
            update={
                "content_hash": _hash(raw),
                "source_path": source_path,
            }
        )


# ---------------------------------------------------------------------------
# Module-level singleton factory (ERRATA §1)
# ---------------------------------------------------------------------------
_adapter_loader_singleton: AdapterLoader | None = None


def get_adapter_loader() -> AdapterLoader:
    """Return the process-wide :class:`AdapterLoader`.

    Non-FastAPI callers (scripts, background workers, synthesis pipeline) use
    this. FastAPI lifespan wires the SAME instance into ``app.state`` so that
    admin endpoints, retrieval paths, and synthesis see identical caches.
    """
    global _adapter_loader_singleton
    if _adapter_loader_singleton is None:
        raise RuntimeError(
            "AdapterLoader not initialized — call init_adapter_loader() at startup"
        )
    return _adapter_loader_singleton


def init_adapter_loader(
    *,
    blob: BlobReader,
    last_resort_path: Path,
    ttl_seconds: float = 300.0,
) -> AdapterLoader:
    """Initialize the module-level singleton. Called once from app startup /
    CLI entrypoint. Returns the constructed loader so callers can also hold a
    direct reference for dependency injection into FastAPI ``app.state``."""
    global _adapter_loader_singleton
    _adapter_loader_singleton = AdapterLoader(
        blob=blob,
        last_resort_path=last_resort_path,
        ttl_seconds=ttl_seconds,
    )
    return _adapter_loader_singleton
