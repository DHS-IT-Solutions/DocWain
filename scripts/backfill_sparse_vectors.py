#!/usr/bin/env python3
"""One-shot backfill of SPLADE sparse vectors into existing Qdrant chunks.

Qdrant collections already have a sparse slot named `keywords_vector`
provisioned (see src/api/vector_store.py). Existing chunks were indexed
with sparse_vector=None. This script iterates every chunk, encodes its
canonical_text with src.embedding.sparse.SparseEncoder, and upserts the
sparse vector while leaving the dense vector and payload untouched.

Modes:
    --inventory-only : list collections + counts, no Qdrant writes.
    --dry-run        : full scroll + encode path, skip upsert.
    (default)        : encode + upsert sparse vectors; mark each processed
                       point with payload `sparse_backfilled_at` so reruns
                       resume from where they left off.

Usage:
    # Size check, no writes
    python scripts/backfill_sparse_vectors.py --inventory-only
    python scripts/backfill_sparse_vectors.py --inventory-only --subscription-id X

    # Verify encode path end-to-end without mutating
    python scripts/backfill_sparse_vectors.py --dry-run --subscription-id X

    # Full run, single collection
    python scripts/backfill_sparse_vectors.py --subscription-id X --batch-size 64

    # Full run, all collections
    python scripts/backfill_sparse_vectors.py --all --batch-size 64
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from typing import Iterator, List, Optional

from qdrant_client import QdrantClient

from src.api.config import Config
from src.embedding.sparse import SparseEncoder, sparse_to_qdrant

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


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iter_batches(
    client: QdrantClient,
    collection_name: str,
    batch_size: int,
) -> Iterator[list]:
    """Yield successive batches of Qdrant points, skipping already-backfilled ones.

    Resumability marker: payload field `sparse_backfilled_at` is set on each
    processed point. Scroll filter excludes points that carry that key so a
    rerun picks up only untouched work.

    Uses `IsEmptyCondition` (not `IsNullCondition`) because existing chunks
    do not carry the marker key at all — `IsNullCondition` matches only
    explicit `null` values, whereas `IsEmptyCondition` matches missing keys.
    """
    from qdrant_client.models import Filter, IsEmptyCondition, PayloadField

    # Qdrant requires a payload index before IsEmptyCondition can filter on the
    # field. Create the index idempotently (no-op if it already exists).
    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name="sparse_backfilled_at",
            field_schema="datetime",
        )
    except Exception as exc:
        logger.debug(
            "create_payload_index(sparse_backfilled_at) on %s: %s (likely already exists)",
            collection_name, exc,
        )

    scroll_filter = Filter(
        must=[IsEmptyCondition(is_empty=PayloadField(key="sparse_backfilled_at"))]
    )

    next_offset = None
    while True:
        try:
            points, next_offset = client.scroll(
                collection_name=collection_name,
                scroll_filter=scroll_filter,
                limit=batch_size,
                with_payload=True,
                with_vectors=False,
                offset=next_offset,
            )
        except Exception as exc:
            logger.error("Scroll failed for %s: %s", collection_name, exc)
            return

        if not points:
            return

        yield points

        if next_offset is None:
            return


def process_collection(
    client: QdrantClient,
    collection_name: str,
    encoder: SparseEncoder,
    batch_size: int,
    *,
    dry_run: bool = False,
) -> dict:
    """Encode and (optionally) upsert sparse vectors for all un-backfilled chunks.

    Returns per-collection stats: encoded / upserted / skipped / errors.
    """
    from qdrant_client.models import PointVectors

    stats = {"encoded": 0, "upserted": 0, "skipped": 0, "errors": 0}

    for batch in _iter_batches(client, collection_name, batch_size):
        texts: list[str] = []
        point_ids: list = []

        for pt in batch:
            payload = pt.payload or {}
            text = (
                payload.get("canonical_text")
                or payload.get("embedding_text")
                or payload.get("text")
                or ""
            )
            if not text:
                stats["skipped"] += 1
                continue
            texts.append(text)
            point_ids.append(pt.id)

        if not texts:
            continue

        try:
            sparse_dicts = encoder.encode_batch(texts)
            stats["encoded"] += len(sparse_dicts)
        except Exception as exc:
            logger.error("Encode batch failed for %s: %s", collection_name, exc)
            stats["errors"] += len(texts)
            continue

        if dry_run:
            continue

        upsert_points = [
            PointVectors(id=pid, vector={"keywords_vector": sparse_to_qdrant(sd)})
            for pid, sd in zip(point_ids, sparse_dicts)
        ]
        try:
            client.update_vectors(collection_name=collection_name, points=upsert_points)
            client.set_payload(
                collection_name=collection_name,
                payload={"sparse_backfilled_at": _iso_now()},
                points=point_ids,
            )
            stats["upserted"] += len(upsert_points)
        except Exception as exc:
            logger.error("Upsert failed for %s: %s", collection_name, exc)
            stats["errors"] += len(upsert_points)

        logger.info(
            "collection=%s processed=%d upserted=%d errors=%d",
            collection_name, stats["encoded"], stats["upserted"], stats["errors"],
        )

    return stats


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="One-shot SPLADE sparse backfill for Qdrant chunks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--subscription-id", help="Single subscription substring (matches collection name)")
    ap.add_argument("--all", action="store_true", help="Process every collection")
    ap.add_argument("--batch-size", type=int, default=64, help="SPLADE batch size")
    ap.add_argument("--inventory-only", action="store_true", help="List collections + counts, then exit (read-only)")
    ap.add_argument("--dry-run", action="store_true", help="Encode but do not upsert")
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

    # Resolve collections to process
    inventory = inventory_collections(client, subscription_id=args.subscription_id)
    if not inventory:
        logger.error("No collections matched.")
        return 1

    encoder = SparseEncoder()
    grand_total = {"encoded": 0, "upserted": 0, "skipped": 0, "errors": 0}

    for row in inventory:
        name = row["collection"]
        if row.get("point_count", 0) < 0:
            logger.warning("Skipping %s (inventory reported error: %s)", name, row.get("error"))
            continue
        logger.info("=== Processing %s (%d points) ===", name, row["point_count"])
        t0 = time.time()
        stats = process_collection(
            client, name, encoder, batch_size=args.batch_size, dry_run=args.dry_run,
        )
        elapsed = time.time() - t0
        logger.info("%s done in %.1fs: %s", name, elapsed, stats)
        for k in grand_total:
            grand_total[k] += stats.get(k, 0)

    logger.info("=== GRAND TOTAL === %s", grand_total)
    return 0 if grand_total["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
