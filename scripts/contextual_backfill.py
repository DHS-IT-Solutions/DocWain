#!/usr/bin/env python3
"""Contextual Retrieval backfill — upgrade existing chunks in place.

For each chunk in the target profile, generate a 1-2 sentence situating
context via the live vLLM DocWain instance, then re-embed
``<context>\n\n<chunk_text>`` and upsert the Qdrant point with the same
``point_id``. ``canonical_text`` / ``content`` / ``chunk_id`` are left
untouched so answer synthesis and graph lookups are unaffected — only the
vector and the new ``chunk_context`` + updated ``embedding_text`` fields
change.

Why this shape:
  * No reingest of sections / entities / KG — context is the only new
    artifact, so it belongs on the Qdrant payload, not in a separate store.
  * Same ``point_id`` means no orphaned vectors and idempotent repeats.
  * Uses the same BGE-large embedder the live retriever calls at query
    time, so the added context actually lands in the same vector space.

Usage:
    python scripts/contextual_backfill.py <subscription_id> <profile_id> \\
        [--limit N] [--dry-run] [--doc <document_id>]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Prompt + generation logic live in the shared module so ingest and
# backfill use one definition.
from src.embedding.pipeline.contextual_embedding import (  # noqa: E402
    generate_context_for_chunk,
)


def _load_profile_chunks(qdrant, collection_name: str, subscription_id: str,
                         profile_id: str, doc_id_filter: Optional[str] = None) -> List[Any]:
    """Scroll all ``resolution=chunk`` points for the target profile."""
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    must = [
        FieldCondition(key="subscription_id", match=MatchValue(value=str(subscription_id))),
        FieldCondition(key="profile_id", match=MatchValue(value=str(profile_id))),
        FieldCondition(key="resolution", match=MatchValue(value="chunk")),
    ]
    if doc_id_filter:
        must.append(FieldCondition(key="document_id", match=MatchValue(value=str(doc_id_filter))))
    flt = Filter(must=must)

    records: List[Any] = []
    nxt = None
    while True:
        batch, nxt = qdrant.scroll(
            collection_name=collection_name,
            scroll_filter=flt,
            limit=256,
            with_payload=True,
            with_vectors=False,
            offset=nxt,
        )
        if not batch:
            break
        records.extend(batch)
        if nxt is None:
            break
    return records


def _group_by_doc(records: List[Any]) -> Dict[str, List[Any]]:
    """Group chunk points by document_id and sort by chunk_index when present."""
    by_doc: Dict[str, List[Any]] = {}
    for r in records:
        p = r.payload or {}
        doc_id = str(p.get("document_id") or "")
        if not doc_id:
            continue
        by_doc.setdefault(doc_id, []).append(r)
    for doc_id, chunks in by_doc.items():
        chunks.sort(key=lambda rec: int((rec.payload or {}).get("chunk_index") or 0))
    return by_doc


def _reconstruct_doc_text(chunks: List[Any]) -> str:
    """Concatenate canonical_text from chunks in order."""
    parts: List[str] = []
    for r in chunks:
        p = r.payload or {}
        t = p.get("canonical_text") or p.get("content") or p.get("embedding_text") or ""
        if t:
            parts.append(str(t))
    return "\n\n".join(parts)


def _doc_name_and_type(chunks: List[Any]) -> Tuple[str, str]:
    for r in chunks:
        p = r.payload or {}
        name = p.get("source_name") or ""
        dtype = p.get("doc_domain") or p.get("doc_type") or "document"
        if name:
            return str(name), str(dtype)
    return "document", "document"


def _embed(embedder, texts: List[str]) -> List[List[float]]:
    vectors = embedder.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return [v.tolist() for v in vectors]


def run(subscription_id: str, profile_id: str, *,
        limit: Optional[int] = None, dry_run: bool = False,
        single_doc: Optional[str] = None) -> int:
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct
    from src.api.config import Config
    from src.api.vector_store import build_collection_name
    from src.api.dw_newron import get_model_by_name
    from src.serving.vllm_manager import VLLMManager

    collection_name = build_collection_name(subscription_id)
    print(f"collection:      {collection_name}")
    print(f"profile:         {profile_id}")
    print(f"doc filter:      {single_doc or '(all)'}")
    print(f"dry-run:         {dry_run}")
    print()

    qdrant = QdrantClient(url=Config.Qdrant.URL, api_key=Config.Qdrant.API)
    vllm = VLLMManager()
    if not vllm.health_check():
        print("[fatal] vLLM unreachable at :8100 — aborting")
        return 2

    print("[1/4] embedder load (BGE-large)...")
    embedder = get_model_by_name("BAAI/bge-large-en-v1.5")

    print("[2/4] scrolling chunks...")
    records = _load_profile_chunks(qdrant, collection_name, subscription_id, profile_id,
                                   doc_id_filter=single_doc)
    by_doc = _group_by_doc(records)
    total_chunks = sum(len(v) for v in by_doc.values())
    print(f"       {total_chunks} chunk(s) across {len(by_doc)} doc(s)")
    if not total_chunks:
        return 0
    if limit:
        capped = {}
        count = 0
        for doc_id, chunks in by_doc.items():
            if count >= limit:
                break
            take = chunks[: max(1, limit - count)]
            capped[doc_id] = take
            count += len(take)
        by_doc = capped
        total_chunks = sum(len(v) for v in by_doc.values())
        print(f"       limited to {total_chunks} chunk(s)")

    print("[3/4] generating contexts + re-embedding...")
    points_to_upsert: List[Any] = []
    t0 = time.monotonic()
    gen_total = 0.0
    emb_total = 0.0
    for doc_id, chunks in by_doc.items():
        doc_name, doc_type = _doc_name_and_type(chunks)
        doc_text = _reconstruct_doc_text(chunks)

        ctx_list: List[str] = []
        chunk_texts: List[str] = []
        for rec in chunks:
            p = rec.payload or {}
            chunk_text = p.get("canonical_text") or p.get("content") or ""
            chunk_idx = int(p.get("chunk_index") or 0)
            tg = time.monotonic()
            ctx = generate_context_for_chunk(
                vllm,
                doc_name=doc_name,
                doc_type=doc_type,
                doc_text=doc_text,
                chunk_text=chunk_text,
                chunk_index=chunk_idx,
            )
            gen_total += time.monotonic() - tg
            ctx_list.append(ctx)
            chunk_texts.append(chunk_text)

        contextualized = [
            (f"{c}\n\n{t}" if c else t) for c, t in zip(ctx_list, chunk_texts)
        ]
        te = time.monotonic()
        vectors = _embed(embedder, contextualized)
        emb_total += time.monotonic() - te

        for rec, ctx, ctext, vec in zip(chunks, ctx_list, contextualized, vectors):
            p = dict(rec.payload or {})
            p["chunk_context"] = ctx
            p["embedding_text"] = ctext
            p["contextualized"] = True
            p["contextualized_at"] = int(time.time())
            points_to_upsert.append(PointStruct(
                id=rec.id,
                vector={"content_vector": vec},
                payload=p,
            ))

        print(f"   doc={doc_id} {doc_name!r} chunks={len(chunks)}")

    elapsed = time.monotonic() - t0
    print(f"       generated {total_chunks} contexts in {gen_total:.1f}s, "
          f"embedded in {emb_total:.1f}s, total {elapsed:.1f}s "
          f"({gen_total/max(total_chunks,1):.1f}s/chunk on LLM)")

    if dry_run:
        # Show a sample
        print("\n[dry-run] sample context (first chunk):")
        sample = points_to_upsert[0]
        print("  doc:", sample.payload.get("document_id"))
        print("  ctx:", sample.payload.get("chunk_context"))
        return 0

    print("[4/4] upserting points...")
    for i in range(0, len(points_to_upsert), 64):
        batch = points_to_upsert[i : i + 64]
        qdrant.upsert(collection_name=collection_name, points=batch)
        print(f"       upserted {min(i+64, len(points_to_upsert))}/{len(points_to_upsert)}")
    print("done.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("subscription_id")
    ap.add_argument("profile_id")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--doc", default=None, help="Restrict to a single document_id")
    args = ap.parse_args()
    sys.exit(run(args.subscription_id, args.profile_id,
                 limit=args.limit, dry_run=args.dry_run, single_doc=args.doc))


if __name__ == "__main__":
    main()
