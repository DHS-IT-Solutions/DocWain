"""Three-signal retrieval fusion using Reciprocal Rank Fusion (RRF).

Combines ranked lists from BGE dense, SPLADE sparse, and DocWain V2
embedding signals into a single fused ranking.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Query types that favour a particular signal
_SPLADE_BOOSTED = {"exact_lookup", "id_search"}
_V2_BOOSTED = {"conceptual", "summary"}

_DEFAULT_WEIGHTS: dict[str, float] = {"bge": 0.4, "splade": 0.3, "v2": 0.3}


def reciprocal_rank_fusion(
    rankings: dict[str, list[str]],
    k: int = 60,
    weights: Optional[dict[str, float]] = None,
) -> list[str]:
    """Fuse multiple ranked lists via Reciprocal Rank Fusion.

    Args:
        rankings: Maps signal name to a ranked list of document IDs
                  (index 0 = highest rank).
        k:        RRF constant (default 60).
        weights:  Per-signal multipliers.  Defaults to 1.0 for every signal
                  when *None*.

    Returns:
        Document IDs sorted by descending fused score.
    """
    scores: dict[str, float] = {}

    for signal, ranked_ids in rankings.items():
        w = 1.0 if weights is None else weights.get(signal, 1.0)
        for rank, doc_id in enumerate(ranked_ids):
            scores[doc_id] = scores.get(doc_id, 0.0) + w / (k + rank + 1)

    return sorted(scores, key=lambda d: scores[d], reverse=True)


class FusionRetriever:
    """Fuses BGE, SPLADE, and DocWain-V2 result lists into one ranking.

    Args:
        default_weights: Per-signal weight mapping.  Defaults to
                         ``{"bge": 0.4, "splade": 0.3, "v2": 0.3}``.
        rrf_k:           RRF constant forwarded to
                         :func:`reciprocal_rank_fusion`.
    """

    def __init__(
        self,
        default_weights: Optional[dict[str, float]] = None,
        rrf_k: int = 60,
    ) -> None:
        self.default_weights: dict[str, float] = (
            default_weights if default_weights is not None else dict(_DEFAULT_WEIGHTS)
        )
        self.rrf_k = rrf_k

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fuse(
        self,
        bge_results: list[str],
        splade_results: list[str],
        v2_results: list[str],
        *,
        query_type: Optional[str] = None,
    ) -> list[str]:
        """Return a fused ranking from the three signal lists.

        Args:
            bge_results:    Ranked doc IDs from BGE dense retrieval.
            splade_results: Ranked doc IDs from SPLADE sparse retrieval.
            v2_results:     Ranked doc IDs from DocWain V2 embeddings.
            query_type:     Optional hint used to adjust signal weights.

        Returns:
            Fused and ranked list of document IDs.
        """
        weights = self._adjust_weights_for_query(query_type)
        rankings = {
            "bge": bge_results,
            "splade": splade_results,
            "v2": v2_results,
        }
        return reciprocal_rank_fusion(rankings, k=self.rrf_k, weights=weights)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _adjust_weights_for_query(
        self, query_type: Optional[str]
    ) -> dict[str, float]:
        """Return a weight dict adjusted for the given query type.

        - ``exact_lookup`` / ``id_search`` → boost *splade* by 0.2,
          reduce *bge* and *v2* proportionally.
        - ``conceptual`` / ``summary``     → boost *v2* by 0.2,
          reduce *bge* and *splade* proportionally.
        - Any other value                  → return default weights unchanged.
        """
        weights = dict(self.default_weights)

        if query_type in _SPLADE_BOOSTED:
            boost = 0.2
            weights["splade"] = weights.get("splade", 0.3) + boost
            # Distribute the boost reduction across the other two signals
            weights["bge"] = max(0.0, weights.get("bge", 0.4) - boost / 2)
            weights["v2"] = max(0.0, weights.get("v2", 0.3) - boost / 2)

        elif query_type in _V2_BOOSTED:
            boost = 0.2
            weights["v2"] = weights.get("v2", 0.3) + boost
            weights["bge"] = max(0.0, weights.get("bge", 0.4) - boost / 2)
            weights["splade"] = max(0.0, weights.get("splade", 0.3) - boost / 2)

        return weights
