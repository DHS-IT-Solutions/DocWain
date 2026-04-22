"""One-time rolling migration: add sparse vectors alongside existing dense
vectors on each chunk point (spec §12 Phase 1, ERRATA §21 bypasses master
rollback).

Idempotent — skips points whose payload already carries ``has_sparse=True``,
so partial runs resume cleanly after interruption. Dense-only queries
continue to work while the migration is in progress (the sparse branch of
:class:`src.retrieval.hybrid_search.HybridSearcher` falls back when the
sparse backend raises ``NotImplementedError``).

CLI invocation::

    python -m scripts.reindex_qdrant_sparse --subscription sub_abc [--dry-run]
    python -m scripts.reindex_qdrant_sparse --all [--dry-run]

The CLI glue (``main``) is intentionally thin in Phase 1: real deployment
goes through the app's lifespan in Phase 2 with proper credentials wiring.
The idempotent :class:`SparseReindexer` core is fully exercised by tests.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from typing import Any, Iterable, Protocol


LOGGER = logging.getLogger("scripts.reindex_qdrant_sparse")


class QdrantBridge(Protocol):
    """Minimal Qdrant write surface the reindexer needs.

    ``scroll_points`` yields batches (list[dict]) of the subscription's chunk
    collection. Each point carries ``id``, ``payload`` (with the source
    ``text``), and optionally ``vector`` (dense, unchanged).
    ``update_points_sparse`` applies the sparse vectors in bulk.
    """

    def scroll_points(
        self, collection: str, batch_size: int = 256
    ) -> Iterable[list[dict[str, Any]]]: ...

    def update_points_sparse(
        self, *, collection: str, point_updates: list[dict[str, Any]]
    ) -> None: ...


class SparseEncoder(Protocol):
    """Maps chunk text → sparse representation (e.g., SPLADE / BM25 signals).

    ``encode_batch`` returns one dict per input text with ``indices`` and
    ``values`` keys matching Qdrant's sparse-vector shape.
    """

    def encode_batch(self, texts: list[str]) -> list[dict[str, Any]]: ...


@dataclass
class ReindexDeps:
    """Injected dependencies for :class:`SparseReindexer`."""

    qdrant: QdrantBridge
    sparse_encoder: SparseEncoder


class SparseReindexer:
    """Idempotent sparse-vector reindexer for an existing chunk collection.

    Driver: :meth:`reindex_subscription` scrolls the subscription's
    collection, encodes the text of points that don't yet carry
    ``has_sparse=True``, and writes the sparse vectors + payload patch in one
    call per batch. Returns the count of points updated so the operator can
    reconcile against the corpus size.

    When ``dry_run=True``, the method calculates the needy count without
    touching Qdrant — useful for planning a migration window without risk.
    """

    def __init__(self, deps: ReindexDeps) -> None:
        self._d = deps

    def reindex_subscription(
        self,
        subscription_id: str,
        *,
        dry_run: bool = False,
    ) -> int:
        """Scroll + reindex. Returns the number of points updated."""
        total = 0
        for batch in self._d.qdrant.scroll_points(subscription_id):
            needy = [
                point
                for point in batch
                if not point.get("payload", {}).get("has_sparse")
            ]
            if not needy:
                continue
            if dry_run:
                total += len(needy)
                continue
            texts = [point.get("payload", {}).get("text", "") for point in needy]
            sparse_vectors = self._d.sparse_encoder.encode_batch(texts)
            updates = [
                {
                    "id": point["id"],
                    "sparse_vector": sparse,
                    "payload_patch": {"has_sparse": True},
                }
                for point, sparse in zip(needy, sparse_vectors)
            ]
            self._d.qdrant.update_points_sparse(
                collection=subscription_id, point_updates=updates
            )
            total += len(updates)
        return total


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="reindex_qdrant_sparse",
        description=(
            "One-time rolling migration to add sparse vectors on existing "
            "Qdrant chunk points. Idempotent: skips payload.has_sparse=True."
        ),
    )
    parser.add_argument(
        "--subscription",
        action="append",
        default=[],
        help="Subscription id to reindex (may be passed multiple times).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Reindex every configured subscription.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count needy points without mutating Qdrant.",
    )
    return parser.parse_args(argv)


def main() -> None:  # pragma: no cover — runtime path wired in Phase 2
    args = _parse_args()
    logging.basicConfig(level=logging.INFO)
    if not args.subscription and not args.all:
        raise SystemExit(
            "Specify --subscription <id> (repeatable) or --all. Runtime "
            "backing wires land in Phase 2; see scripts/__init__.py."
        )
    raise SystemExit(
        "Runtime wiring deferred to Phase 2 lifespan integration."
    )


if __name__ == "__main__":  # pragma: no cover
    main()
