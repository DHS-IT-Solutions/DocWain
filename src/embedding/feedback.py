"""Embedding feedback loop: tracks retrieval quality metrics and hard negatives.

Accumulated statistics drive the iterative fine-tuning pipeline by exposing
which retrieved documents were irrelevant (hard negatives) and how well the
embedding model performs on average (precision, recall, MRR).
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Iterable

logger = logging.getLogger(__name__)


class RetrievalFeedbackTracker:
    """Collects per-query retrieval outcomes and aggregates quality metrics.

    Metrics computed per call to :meth:`record_outcome`:

    - **Precision@k** — fraction of retrieved docs that are relevant.
    - **Recall@k**    — fraction of relevant docs that were retrieved.
    - **MRR**         — reciprocal rank of the first relevant doc
                        (0 when no relevant doc appears in the retrieved list).

    Hard negatives are documents that were retrieved but turned out to be
    irrelevant; they are accumulated across all recorded outcomes.
    """

    def __init__(self) -> None:
        self._total_queries: int = 0
        self._sum_precision: float = 0.0
        self._sum_recall: float = 0.0
        self._sum_mrr: float = 0.0
        # Counts how many times each doc appears as a hard negative
        self._hard_negative_counts: Counter[str] = Counter()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_outcome(
        self,
        query: str,  # kept for future per-query logging / indexing
        retrieved_ids: list[str],
        relevant_ids: Iterable[str],
    ) -> None:
        """Record one query's retrieval outcome and update running totals.

        Args:
            query:         The original query string (unused in calculations,
                           retained for signature completeness / future use).
            retrieved_ids: Ordered list of retrieved document IDs (rank 0 first).
            relevant_ids:  Collection of ground-truth relevant document IDs.
        """
        relevant_set = set(relevant_ids)
        retrieved_set = set(retrieved_ids)

        # --- Precision ---
        if retrieved_set:
            precision = len(relevant_set & retrieved_set) / len(retrieved_set)
        else:
            precision = 0.0

        # --- Recall ---
        if relevant_set:
            recall = len(relevant_set & retrieved_set) / len(relevant_set)
        else:
            recall = 0.0

        # --- MRR ---
        mrr = 0.0
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant_set:
                mrr = 1.0 / rank
                break

        # --- Hard negatives (retrieved but not relevant) ---
        for doc_id in retrieved_ids:
            if doc_id not in relevant_set:
                self._hard_negative_counts[doc_id] += 1

        self._total_queries += 1
        self._sum_precision += precision
        self._sum_recall += recall
        self._sum_mrr += mrr

    def get_metrics(self) -> dict:
        """Return averaged quality metrics across all recorded queries.

        Returns:
            Dictionary with keys:
            ``total_queries``, ``precision_at_k``, ``recall_at_k``, ``mrr``.
        """
        n = self._total_queries
        if n == 0:
            return {
                "total_queries": 0,
                "precision_at_k": 0.0,
                "recall_at_k": 0.0,
                "mrr": 0.0,
            }
        return {
            "total_queries": n,
            "precision_at_k": self._sum_precision / n,
            "recall_at_k": self._sum_recall / n,
            "mrr": self._sum_mrr / n,
        }

    def get_hard_negatives(self) -> list[str]:
        """Return hard-negative doc IDs sorted by descending occurrence count.

        Ties are broken alphabetically so the output is deterministic.
        """
        return sorted(
            self._hard_negative_counts,
            key=lambda d: (-self._hard_negative_counts[d], d),
        )

    def clear(self) -> None:
        """Reset all accumulated state."""
        self._total_queries = 0
        self._sum_precision = 0.0
        self._sum_recall = 0.0
        self._sum_mrr = 0.0
        self._hard_negative_counts.clear()
