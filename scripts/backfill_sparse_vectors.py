#!/usr/bin/env python3
"""One-shot backfill of SPLADE sparse vectors into existing Qdrant chunks.

Qdrant collections already have a sparse slot named `keywords_vector`
provisioned (see src/api/vector_store.py). Existing chunks were indexed
with sparse_vector=None. This script iterates every chunk, encodes its
canonical_text with src.embedding.sparse.SparseEncoder, and upserts the
sparse vector while leaving the dense vector and payload untouched.

Task 5 scope: skeleton + argparse + read-only inventory mode.
Task 6 will add dry-run and full processing.
Task 7 will operate the full run.

Usage (Task 5):
    # Size check, no writes
    python scripts/backfill_sparse_vectors.py --inventory-only
    python scripts/backfill_sparse_vectors.py --inventory-only --subscription-id X
"""
from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Optional

from qdrant_client import QdrantClient

from src.api.config import Config

logger = logging.getLogger("backfill_sparse")


def make_client() -> QdrantClient:
    return QdrantClient(url=Config.Qdrant.URL, api_key=Config.Qdrant.API)


def inventory_collections(
    client: QdrantClient, subscription_id: Optional[str] = None
) -> list[dict]:
    """Return [{collection, point_count}] for every (or filtered) collection.

    On Qdrant failure for a specific collection, includes the entry with
    point_count=-1 and an error message so inventory partial-failures are
    visible without aborting the whole pass.
    """
    out = []
    for col in client.get_collections().collections or []:
        name = getattr(col, "name", str(col))
        if subscription_id and subscription_id not in name:
            continue
        try:
            count_resp = client.count(collection_name=name, exact=True)
            out.append({"collection": name, "point_count": count_resp.count})
        except Exception as exc:
            logger.warning("Count failed for %s: %s", name, exc)
            out.append({"collection": name, "point_count": -1, "error": str(exc)})
    return out


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="One-shot SPLADE sparse backfill for Qdrant chunks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--subscription-id", help="Single subscription substring (matches collection name)")
    ap.add_argument("--all", action="store_true", help="Process every collection")
    ap.add_argument("--batch-size", type=int, default=64, help="SPLADE batch size")
    ap.add_argument("--inventory-only", action="store_true", help="List collections + counts, then exit (read-only)")
    ap.add_argument("--dry-run", action="store_true", help="Encode but do not upsert (added in Task 6)")
    ap.add_argument("--concurrency", type=int, default=1, help="Currently fixed to 1 while vLLM is resident")
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    args = parse_args(argv)

    if not (args.subscription_id or args.all or args.inventory_only):
        logger.error("Must specify --subscription-id, --all, or --inventory-only")
        return 2

    client = make_client()

    if args.inventory_only:
        inventory = inventory_collections(client, subscription_id=args.subscription_id)
        print(f"{'COLLECTION':<40} {'POINTS':>12}")
        print("-" * 54)
        valid_count = 0
        total = 0
        for row in inventory:
            print(f"{row['collection']:<40} {row['point_count']:>12}")
            if row["point_count"] > 0:
                total += row["point_count"]
                valid_count += 1
        print("-" * 54)
        print(f"{'TOTAL':<40} {total:>12}  ({valid_count} collections)")
        return 0

    # Mutating paths (dry-run / full) are implemented in Task 6.
    logger.error("Full processing not yet implemented — rerun with --inventory-only for now.")
    return 3


if __name__ == "__main__":
    sys.exit(main())
