"""Layer C retrieval — SME artifact snippets from per-subscription Qdrant.

Phase 2 writes retrievable snippets to ``sme_artifacts_{sub}`` via
:class:`src.intelligence.sme.storage.SMEArtifactStorage`. This module is the
read path (ERRATA §8): :class:`SMERetrieval.retrieve` performs a
dense-vector search over the subscription's collection, hard-filtered on
``subscription_id`` + ``profile_id`` (+ optional ``artifact_type`` /
``domain_tags``), and returns a list of dict rows that the Phase 3
retrieval fusion layer will merge into Layer C of the unified pack.

The retrieve call is flag-gated on ``enable_sme_retrieval``: with the flag
off we return ``[]`` WITHOUT ever hitting Qdrant, preserving the "flag-off
= pre-Phase-2 behaviour" invariant.

Cross-subscription read attempts are rejected at the method boundary:
missing ``subscription_id`` or ``profile_id`` raises ``ValueError`` so
that a bug in the caller can never degrade into a cross-profile read.
"""
from __future__ import annotations

from typing import Any

from src.config.feature_flags import ENABLE_SME_RETRIEVAL, get_flag_resolver


class SMERetrieval:
    """Hybrid-ready (dense-now, sparse Phase 3) SME artifact retriever.

    ``qdrant`` must expose the standard qdrant_client surface: a ``search``
    method taking ``collection_name``, ``query_vector``, ``query_filter``
    and ``limit``. ``embedder`` exposes ``embed(text) -> list[float]``; in
    production this is the same embedder used by the synthesis write path
    so vectors live in the same metric space.
    """

    def __init__(self, qdrant: Any, embedder: Any) -> None:
        self._qdrant = qdrant
        self._embedder = embedder

    def retrieve(
        self,
        *,
        query: str,
        subscription_id: str,
        profile_id: str,
        artifact_types: list[str] | None = None,
        top_k: int = 10,
    ) -> list[dict]:
        """Return up to ``top_k`` SME artifact rows matching ``query``.

        Profile isolation is hard: empty ``subscription_id`` or
        ``profile_id`` raises ``ValueError`` so the typical
        ``subscription_id=some_default_sub`` cross-profile-read bug cannot
        compile silently. The ``enable_sme_retrieval`` flag gate fires
        BEFORE any embed or search call so the flag-off path is free of
        side-effects.
        """
        if not subscription_id:
            raise ValueError("subscription_id required for SME retrieval")
        if not profile_id:
            raise ValueError("profile_id required for SME retrieval")
        resolver = _safe_flag_resolver()
        if resolver is None:
            return []
        try:
            if not resolver.is_enabled(subscription_id, ENABLE_SME_RETRIEVAL):
                return []
        except KeyError:
            return []

        from qdrant_client.http.models import (
            FieldCondition,
            Filter,
            MatchAny,
            MatchValue,
        )

        must = [
            FieldCondition(
                key="subscription_id",
                match=MatchValue(value=subscription_id),
            ),
            FieldCondition(
                key="profile_id",
                match=MatchValue(value=profile_id),
            ),
        ]
        if artifact_types:
            must.append(
                FieldCondition(
                    key="artifact_type",
                    match=MatchAny(any=list(artifact_types)),
                )
            )
        vector = self._embedder.embed(query)
        rows = self._qdrant.search(
            collection_name=f"sme_artifacts_{subscription_id}",
            query_vector=vector,
            query_filter=Filter(must=must),
            limit=int(top_k),
        )
        return [_row_to_hit(r) for r in rows]


def _row_to_hit(row: Any) -> dict:
    payload = getattr(row, "payload", {}) or {}
    # Qdrant's Python client returns either a ScoredPoint or a plain dict
    # depending on version; handle both while keeping retrieval-layer code
    # free of isinstance-on-external-type hacks.
    score = getattr(row, "score", None)
    if score is None and isinstance(row, dict):
        payload = row.get("payload", {}) or {}
        score = row.get("score")
    return {
        "kind": "sme_artifact",
        "artifact_type": payload.get("artifact_type"),
        "snippet_id": payload.get("snippet_id")
        or payload.get("id")
        or getattr(row, "id", None),
        "text": payload.get("text", ""),
        "confidence": payload.get("confidence"),
        "evidence": list(payload.get("evidence", []) or []),
        "score": score,
        "payload": payload,
    }


def _safe_flag_resolver():
    """Return the process-wide flag resolver or ``None`` if uninitialised.

    Mirrors the ``UnifiedRetriever`` helper: retrieval never blows up when
    the flag resolver is not wired; flag-off semantics apply instead.
    """
    try:
        return get_flag_resolver()
    except RuntimeError:
        return None
