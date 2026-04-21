"""Hybrid dense+sparse retrieval with Reciprocal Rank Fusion
(spec §7 stage 1, ERRATA §7 + §8).

The helper is deliberately flag-agnostic: a caller decides whether hybrid
retrieval is active (via :class:`src.config.feature_flags.SMEFeatureFlags`)
and either invokes :meth:`HybridSearcher.search` or falls back to dense-only
through its existing path. The fuser defaults to the canonical RRF formula
``score(d) = sum_i(1 / (k + rank_i(d)))`` with ``k = 60``.

Sparse availability is best-effort: if the Qdrant bridge raises
``NotImplementedError`` (Phase 1 deployments not yet re-indexed), the helper
returns the dense-only ranking unchanged. Phase 2 migrates collections via
``scripts/reindex_qdrant_sparse.py`` and the fallback stops firing.

Profile-isolation filters are forwarded verbatim to both backends — the helper
never mutates ``query_filter`` — so the ``subscription_id`` / ``profile_id``
guarantees hold uniformly across dense and sparse paths (spec §3 invariant 4).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


class QdrantBridge(Protocol):
    """Dense + sparse search surface exposed by the Qdrant integration layer.

    Implementations may take either an embedding vector or encode the query
    text internally; the helper is agnostic. Sparse backends that are not yet
    provisioned should raise ``NotImplementedError`` so the helper can
    gracefully degrade to dense-only during the rolling re-index window.
    """

    def search_dense(
        self,
        *,
        collection: str,
        query_text: str,
        top_k: int,
        query_filter: dict[str, Any] | None,
    ) -> list[dict[str, Any]]: ...

    def search_sparse(
        self,
        *,
        collection: str,
        query_text: str,
        top_k: int,
        query_filter: dict[str, Any] | None,
    ) -> list[dict[str, Any]]: ...


@dataclass
class HybridConfig:
    """Tunables for the fusion pass.

    ``k_dense`` / ``k_sparse`` cap how many candidates each backend returns
    before fusion. ``rrf_k`` is the RRF denominator floor; lower values skew
    toward top-ranked results, higher values smooth the combined ranking.
    Defaults match the user-specified canonical numbers (40 / 40 / 60).
    """

    k_dense: int = 40
    k_sparse: int = 40
    rrf_k: int = 60


@dataclass
class HybridResult:
    """Fused result carrying both per-backend ranks for diagnostics."""

    item_id: str
    rrf_score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None
    payload: dict[str, Any] = field(default_factory=dict)


class HybridSearcher:
    """RRF-fused dense + sparse searcher."""

    def __init__(self, qdrant: QdrantBridge, config: HybridConfig) -> None:
        self._q = qdrant
        self._c = config

    def search(
        self,
        query_text: str,
        collection: str,
        top_k: int,
        query_filter: dict[str, Any] | None = None,
    ) -> list[HybridResult]:
        """Run dense + sparse and return the top-``top_k`` fused results.

        The per-backend ``top_k`` is taken from the config (``k_dense`` /
        ``k_sparse``); the final cut to ``top_k`` happens after RRF so the
        two feeds can promote different candidates into the final list.
        """
        dense = self._q.search_dense(
            collection=collection,
            query_text=query_text,
            top_k=self._c.k_dense,
            query_filter=query_filter,
        )
        try:
            sparse = self._q.search_sparse(
                collection=collection,
                query_text=query_text,
                top_k=self._c.k_sparse,
                query_filter=query_filter,
            )
        except NotImplementedError:
            sparse = []

        k = self._c.rrf_k
        by_id: dict[str, HybridResult] = {}
        for rank, point in enumerate(dense):
            result = by_id.setdefault(
                point["id"],
                HybridResult(
                    item_id=point["id"],
                    rrf_score=0.0,
                    payload=point.get("payload", {}),
                ),
            )
            result.dense_rank = rank
            result.rrf_score += 1.0 / (k + rank + 1)
        for rank, point in enumerate(sparse):
            result = by_id.setdefault(
                point["id"],
                HybridResult(
                    item_id=point["id"],
                    rrf_score=0.0,
                    payload=point.get("payload", {}),
                ),
            )
            result.sparse_rank = rank
            result.rrf_score += 1.0 / (k + rank + 1)
        return sorted(by_id.values(), key=lambda r: -r.rrf_score)[:top_k]
