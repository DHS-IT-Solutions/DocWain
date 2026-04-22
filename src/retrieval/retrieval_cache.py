"""Phase 3 Task 8 — Redis-backed retrieval-pack cache.

Memoizes the merged retrieval :class:`~src.retrieval.types.RetrievalBundle`
for 5 minutes so near-duplicate queries don't re-hit Qdrant + Neo4j + SME
for the same ``(sub, prof, query_fingerprint, flag_set_version)`` tuple.
The flag-set version — incremented by
:func:`src.config.feature_flags.set_subscription_override` and the admin
``PATCH /admin/sme-flags/{subscription_id}`` endpoint — naturally
invalidates the cache: a flag flip bumps the version, old keys are
unreachable, and they age out on TTL.

Key layout: ``dwx:retrieval:{sub}:{prof}:{query_fingerprint}:{flag_set_version}``
where ``query_fingerprint`` is a SHA-256-derived 20-character prefix over
the lower-cased, stripped query text. Profile isolation is enforced at
the key level — there is no way to read a bundle keyed for ``(sub_A,
prof_X)`` from ``(sub_A, prof_Y)``.

The cache is best-effort: every Redis call is wrapped so a partial outage
degrades into "cache miss" and the retrieval pipeline keeps working. This
mirrors the Phase 3 rule: no internal wall-clock timeouts, no cached-path
exception that can leak into the hot path.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, is_dataclass
from typing import Any, Optional

from src.retrieval.types import PackedItem, RetrievalBundle

logger = logging.getLogger(__name__)

_PREFIX = "dwx:retrieval"
_DEFAULT_TTL_SECONDS = 300


def _query_fingerprint(query: str) -> str:
    """Compute a 20-character lowercase-stripped SHA-256 prefix."""
    h = hashlib.sha256((query or "").strip().lower().encode("utf-8"))
    return h.hexdigest()[:20]


def _bundle_to_jsonable(bundle: RetrievalBundle) -> dict:
    """Render a :class:`RetrievalBundle` into a JSON-safe dict.

    ``PackedItem`` tuples become lists of ``[doc_id, chunk_id]``, dict
    layers round-trip via their dict form. The inverse (:func:`_bundle_from_jsonable`)
    is careful to restore the frozen dataclass where applicable.
    """
    return {
        "layer_a_chunks": list(bundle.layer_a_chunks or []),
        "layer_b_kg": list(bundle.layer_b_kg or []),
        "layer_c_sme": list(bundle.layer_c_sme or []),
        "layer_d_url": list(bundle.layer_d_url or []),
        "degraded_layers": list(bundle.degraded_layers or []),
        "per_layer_ms": dict(bundle.per_layer_ms or {}),
    }


def _bundle_from_jsonable(data: dict) -> RetrievalBundle:
    return RetrievalBundle(
        layer_a_chunks=list(data.get("layer_a_chunks") or []),
        layer_b_kg=list(data.get("layer_b_kg") or []),
        layer_c_sme=list(data.get("layer_c_sme") or []),
        layer_d_url=list(data.get("layer_d_url") or []),
        degraded_layers=list(data.get("degraded_layers") or []),
        per_layer_ms=dict(data.get("per_layer_ms") or {}),
    )


def _packed_item_to_jsonable(p: PackedItem) -> dict:
    return {
        "text": p.text,
        "provenance": [list(t) for t in p.provenance],
        "layer": p.layer,
        "confidence": float(p.confidence),
        "rerank_score": float(p.rerank_score),
        "sme_backed": bool(p.sme_backed),
        "metadata": dict(p.metadata or {}),
    }


def _packed_item_from_jsonable(d: dict) -> PackedItem:
    prov_raw = d.get("provenance") or []
    prov = tuple(
        (str(p[0]), str(p[1])) for p in prov_raw if isinstance(p, (list, tuple)) and len(p) == 2
    )
    layer = d.get("layer") or "a"
    if layer not in ("a", "b", "c", "d"):
        layer = "a"
    return PackedItem(
        text=str(d.get("text") or ""),
        provenance=prov,
        layer=layer,  # type: ignore[arg-type]
        confidence=float(d.get("confidence") or 0.0),
        rerank_score=float(d.get("rerank_score") or 0.0),
        sme_backed=bool(d.get("sme_backed") or False),
        metadata=dict(d.get("metadata") or {}),
    )


class RetrievalCache:
    """Redis-backed retrieval-pack cache.

    Instantiate with a Redis client (``redis.Redis`` or any object exposing
    ``get``/``setex``/``scan_iter``/``delete``). Pass ``None`` to disable
    the cache — all methods become no-ops that return ``None`` on
    :meth:`get` so callers can always call the cache without branching.
    """

    PREFIX = _PREFIX

    def __init__(
        self,
        redis_client: Any = None,
        *,
        ttl_seconds: int = _DEFAULT_TTL_SECONDS,
    ) -> None:
        self._redis = redis_client
        self._ttl = int(ttl_seconds)

    # ------------------------------------------------------------------
    # Key assembly
    # ------------------------------------------------------------------

    def _key(
        self,
        *,
        subscription_id: str,
        profile_id: str,
        query_fingerprint: str,
        flag_set_version: Any,
    ) -> str:
        return (
            f"{self.PREFIX}:{subscription_id}:{profile_id}:"
            f"{query_fingerprint}:{flag_set_version}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(
        self,
        *,
        subscription_id: str,
        profile_id: str,
        query_fingerprint: str,
        flag_set_version: Any,
    ) -> Optional[RetrievalBundle]:
        """Return the cached :class:`RetrievalBundle` or ``None`` on miss.

        Any Redis-side error degrades to a miss — the retrieval hot path
        must never raise because the cache is unavailable.
        """
        if self._redis is None:
            return None
        key = self._key(
            subscription_id=subscription_id,
            profile_id=profile_id,
            query_fingerprint=query_fingerprint,
            flag_set_version=flag_set_version,
        )
        try:
            raw = self._redis.get(key)
        except Exception:  # noqa: BLE001
            logger.debug("retrieval_cache.get redis error", exc_info=True)
            return None
        if raw is None:
            return None
        if isinstance(raw, (bytes, bytearray)):
            try:
                raw = raw.decode("utf-8")
            except Exception:  # noqa: BLE001
                return None
        try:
            data = json.loads(raw)
        except Exception:  # noqa: BLE001
            logger.debug("retrieval_cache.get json decode failed", exc_info=True)
            return None
        if not isinstance(data, dict):
            return None
        try:
            return _bundle_from_jsonable(data)
        except Exception:  # noqa: BLE001
            logger.debug("retrieval_cache.get bundle hydrate failed", exc_info=True)
            return None

    def set(
        self,
        *,
        subscription_id: str,
        profile_id: str,
        query_fingerprint: str,
        flag_set_version: Any,
        bundle: RetrievalBundle,
    ) -> None:
        """Persist ``bundle`` under the composite key with the TTL.

        Silently no-ops when ``redis_client`` is ``None`` or ``bundle`` is
        falsy. Dataclass serialisation is handled by
        :func:`_bundle_to_jsonable`; errors on serialisation are logged
        and swallowed.
        """
        if self._redis is None or bundle is None:
            return
        key = self._key(
            subscription_id=subscription_id,
            profile_id=profile_id,
            query_fingerprint=query_fingerprint,
            flag_set_version=flag_set_version,
        )
        try:
            payload = json.dumps(_bundle_to_jsonable(bundle), default=str)
        except Exception:  # noqa: BLE001
            logger.debug("retrieval_cache.set serialise failed", exc_info=True)
            return
        try:
            self._redis.setex(key, self._ttl, payload)
        except Exception:  # noqa: BLE001
            logger.debug("retrieval_cache.set redis error", exc_info=True)

    def invalidate_profile(
        self, *, subscription_id: str, profile_id: str
    ) -> int:
        """Evict every cached pack for the ``(sub, prof)`` pair.

        Called from :mod:`src.api.pipeline_api` when a profile's pipeline
        transitions to ``PIPELINE_TRAINING_COMPLETED`` — new SME artifacts
        have been materialized and any cached pack would be stale.

        Returns the number of keys deleted (best-effort — partial scan
        failures are logged but never raised into the caller).
        """
        if self._redis is None:
            return 0
        pattern = f"{self.PREFIX}:{subscription_id}:{profile_id}:*"
        deleted = 0
        try:
            for key in self._redis.scan_iter(match=pattern):
                try:
                    self._redis.delete(key)
                    deleted += 1
                except Exception:  # noqa: BLE001
                    logger.debug(
                        "retrieval_cache.invalidate_profile delete failed",
                        exc_info=True,
                    )
        except Exception:  # noqa: BLE001
            logger.debug(
                "retrieval_cache.invalidate_profile scan_iter failed",
                exc_info=True,
            )
        return deleted


__all__ = [
    "RetrievalCache",
    "_query_fingerprint",
    "_bundle_to_jsonable",
    "_bundle_from_jsonable",
    "_packed_item_to_jsonable",
    "_packed_item_from_jsonable",
]
