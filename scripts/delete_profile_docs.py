#!/usr/bin/env python3
"""Delete all documents for a given (subscription, profile) across every store.

Clean-slate operation — intended for re-upload after a major pipeline
refactor. Removes, per document:
  * MongoDB ``documents`` record (and any audit_log it carries)
  * Qdrant points for that document across all resolutions
  * Neo4j Document / Section / Chunk / Entity / edges scoped to that
    (subscription, profile, document_id)
  * Azure Blob artifacts under ``{subscription}/{profile}/{document}/``
    (extraction.json, screening.json, etc.)
  * The raw upload blob too if its ``location`` is recorded on the doc

The script refuses to run without ``--confirm``. Dry-run shows exactly
which docs would be deleted, across which stores.

Usage:
    python scripts/delete_profile_docs.py <subscription_id> <profile_id>
    python scripts/delete_profile_docs.py <subscription_id> <profile_id> --dry-run
    python scripts/delete_profile_docs.py <subscription_id> <profile_id> --confirm
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _list_profile_docs(col, subscription_id: str, profile_id: str) -> List[Dict[str, Any]]:
    return list(col.find(
        {"subscription_id": subscription_id, "profile_id": profile_id},
        {"_id": 1, "name": 1, "location": 1, "extraction": 1, "screening": 1, "embedding": 1},
    ))


def _delete_qdrant_points(qdrant, collection_name: str, subscription_id: str, profile_id: str, doc_id: str) -> int:
    from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue
    flt = Filter(must=[
        FieldCondition(key="subscription_id", match=MatchValue(value=str(subscription_id))),
        FieldCondition(key="profile_id", match=MatchValue(value=str(profile_id))),
        FieldCondition(key="document_id", match=MatchValue(value=str(doc_id))),
    ])
    # Count before delete
    try:
        cnt = qdrant.count(collection_name=collection_name, count_filter=flt, exact=True).count
    except Exception:
        cnt = 0
    if cnt:
        try:
            qdrant.delete(collection_name=collection_name, points_selector=FilterSelector(filter=flt))
        except Exception as exc:  # noqa: BLE001
            print(f"    [warn] Qdrant delete failed doc={doc_id}: {exc}")
            return 0
    return cnt


def _delete_neo4j(driver, subscription_id: str, profile_id: str, doc_id: str) -> Dict[str, int]:
    """Detach-delete all nodes scoped to this (sub, prof, doc)."""
    counts = {"Document": 0, "Section": 0, "Chunk": 0, "Entity": 0}
    try:
        with driver.session() as session:
            # Count before
            for label in list(counts.keys()):
                if label == "Entity":
                    # Entity nodes are shared across docs; only delete ones
                    # whose only link is via this document's chunks/sections
                    continue
                r = session.run(
                    f"MATCH (n:{label}) "
                    "WHERE n.subscription_id = $sub AND n.profile_id = $prof "
                    f"{'AND n.document_id = $doc ' if label != 'Section' else ''}"
                    "RETURN count(n) AS c",
                    sub=str(subscription_id), prof=str(profile_id), doc=str(doc_id),
                ).single()
                counts[label] = r["c"] if r else 0

            # Section nodes: scope via HAS_SECTION edge from the Document
            r = session.run(
                "MATCH (:Document {document_id: $doc, subscription_id: $sub, profile_id: $prof})"
                "-[:HAS_SECTION]->(s:Section) RETURN count(s) AS c",
                sub=str(subscription_id), prof=str(profile_id), doc=str(doc_id),
            ).single()
            counts["Section"] = r["c"] if r else 0

            # Delete Document + linked Sections + linked Chunks
            session.run(
                "MATCH (d:Document {document_id: $doc, subscription_id: $sub, profile_id: $prof}) "
                "OPTIONAL MATCH (d)-[:HAS_SECTION]->(s:Section) "
                "OPTIONAL MATCH (s)-[:HAS_CHUNK]->(c:Chunk) "
                "DETACH DELETE d, s, c",
                sub=str(subscription_id), prof=str(profile_id), doc=str(doc_id),
            )
            # Delete Chunks pointing at this doc that weren't caught via Section
            session.run(
                "MATCH (c:Chunk {document_id: $doc, subscription_id: $sub, profile_id: $prof}) "
                "DETACH DELETE c",
                sub=str(subscription_id), prof=str(profile_id), doc=str(doc_id),
            )
            # Orphan Entity cleanup: entities with no remaining Chunk/Section/Document link
            r = session.run(
                "MATCH (e:Entity {subscription_id: $sub, profile_id: $prof}) "
                "WHERE NOT (e)<-[:ABOUT]-(:Section) "
                "  AND NOT (e)<-[:MENTIONS]-(:Chunk) "
                "  AND NOT (e)<-[:MENTIONS]-(:Document) "
                "WITH e, count(*) AS total "
                "DETACH DELETE e "
                "RETURN total",
                sub=str(subscription_id), prof=str(profile_id),
            ).single()
            counts["Entity"] = r["total"] if r else 0
    except Exception as exc:  # noqa: BLE001
        print(f"    [warn] Neo4j delete failed doc={doc_id}: {exc}")
    return counts


def _delete_blob_artifacts(subscription_id: str, profile_id: str, doc_id: str, location: str = "") -> int:
    """Delete blob artifacts and the raw upload blob."""
    from src.api.blob_content_store import get_blob_client
    from src.storage.azure_blob_client import get_container_client

    removed = 0
    # Artifact blobs under {sub}/{prof}/{doc}/
    try:
        container = get_blob_client()
        prefix = f"{subscription_id}/{profile_id}/{doc_id}/"
        for blob in container.list_blobs(name_starts_with=prefix):
            try:
                container.get_blob_client(blob.name).delete_blob()
                removed += 1
            except Exception as exc:  # noqa: BLE001
                print(f"    [warn] blob delete failed {blob.name}: {exc}")
    except Exception as exc:  # noqa: BLE001
        print(f"    [warn] artifact listing failed doc={doc_id}: {exc}")

    # Raw upload blob via location=az://container/blob_path
    if location and str(location).startswith("az://"):
        try:
            without_scheme = str(location)[len("az://"):]
            parts = without_scheme.split("/", 1)
            if len(parts) == 2:
                container_name, blob_path = parts
                try:
                    cc = get_container_client(container_name)
                    cc.get_blob_client(blob_path).delete_blob()
                    removed += 1
                except Exception as exc:  # noqa: BLE001
                    print(f"    [warn] raw blob delete failed {location}: {exc}")
        except Exception as exc:  # noqa: BLE001
            print(f"    [warn] location parse failed {location}: {exc}")
    return removed


def run(subscription_id: str, profile_id: str, *, confirm: bool = False, dry_run: bool = False) -> int:
    from qdrant_client import QdrantClient
    from neo4j import GraphDatabase
    from bson import ObjectId
    from src.api.config import Config
    from src.api.vector_store import build_collection_name
    from src.api.document_status import get_documents_collection

    col = get_documents_collection()
    docs = _list_profile_docs(col, subscription_id, profile_id)
    print(f"subscription:   {subscription_id}")
    print(f"profile:        {profile_id}")
    print(f"documents:      {len(docs)}")

    if not docs:
        return 0

    for d in docs:
        print(f"  {str(d['_id'])[-8:]}  {d.get('name','?')[:60]}")

    if not confirm:
        print()
        print(f"DRY-RUN — no changes. Pass --confirm to actually delete.")
        return 0

    qdrant = QdrantClient(url=Config.Qdrant.URL, api_key=Config.Qdrant.API)
    collection_name = build_collection_name(subscription_id)
    neo4j_driver = GraphDatabase.driver(
        Config.Neo4j.URI, auth=(Config.Neo4j.USER, Config.Neo4j.PASSWORD),
    )

    summary = {"qdrant_points": 0, "neo4j_docs": 0, "neo4j_sections": 0,
               "neo4j_chunks": 0, "neo4j_entities": 0, "blobs": 0, "mongo": 0}
    for d in docs:
        doc_id = str(d["_id"])
        name = d.get("name", "?")
        location = d.get("location", "")
        print(f"\ndeleting doc={doc_id[-8:]} {name[:50]}")

        # Qdrant
        qpts = _delete_qdrant_points(qdrant, collection_name, subscription_id, profile_id, doc_id)
        summary["qdrant_points"] += qpts
        print(f"  qdrant: {qpts} points")

        # Neo4j
        ncounts = _delete_neo4j(neo4j_driver, subscription_id, profile_id, doc_id)
        for label, count in ncounts.items():
            summary[f"neo4j_{label.lower()}s" if not label.endswith("s") else f"neo4j_{label.lower()}"] = \
                summary.get(f"neo4j_{label.lower()}s", 0) + count
        print(f"  neo4j: {ncounts}")

        # Blob artifacts
        blob_count = _delete_blob_artifacts(subscription_id, profile_id, doc_id, location)
        summary["blobs"] += blob_count
        print(f"  blob: {blob_count} artifacts")

        # Mongo doc
        try:
            res = col.delete_one({"_id": d["_id"]})
            summary["mongo"] += res.deleted_count
            print(f"  mongo: deleted={res.deleted_count}")
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] mongo delete failed: {exc}")

    print()
    print("=" * 60)
    print("DELETE SUMMARY")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("subscription_id")
    ap.add_argument("profile_id")
    ap.add_argument("--confirm", action="store_true",
                    help="Actually delete (otherwise dry-run)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would be deleted and exit (default)")
    args = ap.parse_args()
    sys.exit(run(args.subscription_id, args.profile_id,
                 confirm=args.confirm, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
