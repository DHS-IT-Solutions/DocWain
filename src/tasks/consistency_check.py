"""Post-ingest consistency check — chunk_id parity across stores.

Every document that reaches ``EMBEDDING_COMPLETED`` must have its
chunks visible in three stores:
    * Mongo: stage.embedding.summary.chunk_count
    * Qdrant: one point per chunk with ``resolution=chunk`` and
      ``chunk_id`` on the payload
    * Neo4j: one ``Chunk {chunk_id}`` node attached to a Section
      attached to the Document

Any mismatch means one of the write paths silently dropped chunks,
which in practice silently degrades retrieval ("why does the agent
not see page 7?"). This check runs at the tail of the Celery embed
task, compares the three sets of chunk_ids, and records the diff on
the Mongo document so it's visible to the UI and the RAG regression
harness.

It does not repair — repair is a different kind of operation that
needs operator awareness. The goal here is to *surface* drift early
so we never operate on a broken document without knowing.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from src.api.document_status import update_document_fields

logger = logging.getLogger(__name__)

_SEVERITY_OK = "ok"
_SEVERITY_WARNING = "warning"
_SEVERITY_ERROR = "error"


def _fetch_qdrant_chunk_ids(
    *, subscription_id: str, profile_id: str, document_id: str,
) -> Set[str]:
    """Return the set of chunk_ids present in Qdrant for a document."""
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import FieldCondition, Filter, MatchValue
        from src.api.config import Config
        from src.api.vector_store import build_collection_name
    except Exception as exc:  # noqa: BLE001
        logger.debug("consistency-check: Qdrant deps unavailable: %s", exc)
        return set()

    try:
        client = QdrantClient(url=Config.Qdrant.URL, api_key=Config.Qdrant.API)
        collection = build_collection_name(subscription_id)
        flt = Filter(must=[
            FieldCondition(key="subscription_id", match=MatchValue(value=str(subscription_id))),
            FieldCondition(key="profile_id", match=MatchValue(value=str(profile_id))),
            FieldCondition(key="document_id", match=MatchValue(value=str(document_id))),
            FieldCondition(key="resolution", match=MatchValue(value="chunk")),
        ])
        ids: Set[str] = set()
        next_offset = None
        while True:
            batch, next_offset = client.scroll(
                collection_name=collection,
                scroll_filter=flt,
                limit=512,
                with_payload=True,
                with_vectors=False,
                offset=next_offset,
            )
            for rec in batch or []:
                p = rec.payload or {}
                cid = (
                    p.get("chunk_id")
                    or (p.get("chunk") or {}).get("id")
                )
                if cid:
                    ids.add(str(cid))
            if next_offset is None:
                break
        return ids
    except Exception as exc:  # noqa: BLE001
        logger.warning("consistency-check: Qdrant scroll failed: %s", exc)
        return set()


def _fetch_neo4j_chunk_ids(
    *, subscription_id: str, profile_id: str, document_id: str,
) -> Set[str]:
    """Return the set of chunk_ids present in Neo4j for a document."""
    try:
        from neo4j import GraphDatabase
        from src.api.config import Config
    except Exception as exc:  # noqa: BLE001
        logger.debug("consistency-check: Neo4j deps unavailable: %s", exc)
        return set()

    try:
        driver = GraphDatabase.driver(
            Config.Neo4j.URI, auth=(Config.Neo4j.USER, Config.Neo4j.PASSWORD),
        )
        with driver.session() as session:
            cypher = (
                "MATCH (c:Chunk) "
                "WHERE c.subscription_id = $sub "
                "  AND c.profile_id = $prof "
                "  AND c.document_id = $doc "
                "RETURN c.chunk_id AS chunk_id"
            )
            records = session.run(
                cypher, sub=str(subscription_id), prof=str(profile_id), doc=str(document_id),
            )
            return {str(r["chunk_id"]) for r in records if r["chunk_id"]}
    except Exception as exc:  # noqa: BLE001
        logger.warning("consistency-check: Neo4j query failed: %s", exc)
        return set()


def verify(
    *,
    document_id: str,
    subscription_id: str,
    profile_id: str,
    expected_chunk_ids: List[str],
) -> Dict[str, Any]:
    """Compare the three stores' chunk_id sets and write a report to Mongo.

    ``expected_chunk_ids`` is the source of truth — usually the chunk_ids
    the embed task just upserted into Qdrant. The function returns the
    computed report dict even if writing it to Mongo fails, so callers
    can log or raise on severity themselves.
    """
    expected_set = set(str(c) for c in expected_chunk_ids if c)
    qdrant_set = _fetch_qdrant_chunk_ids(
        subscription_id=subscription_id, profile_id=profile_id, document_id=document_id,
    )
    neo4j_set = _fetch_neo4j_chunk_ids(
        subscription_id=subscription_id, profile_id=profile_id, document_id=document_id,
    )

    qdrant_missing = sorted(expected_set - qdrant_set)
    qdrant_extra = sorted(qdrant_set - expected_set)
    neo4j_missing = sorted(expected_set - neo4j_set)
    neo4j_extra = sorted(neo4j_set - expected_set)

    severity = _SEVERITY_OK
    # Qdrant drift is an error — retrieval will silently miss chunks.
    if qdrant_missing or qdrant_extra:
        severity = _SEVERITY_ERROR
    # Neo4j drift is a warning — KG/intelligence layers degrade but
    # dense retrieval still works.
    elif neo4j_missing or neo4j_extra:
        severity = _SEVERITY_WARNING

    report: Dict[str, Any] = {
        "severity": severity,
        "expected_count": len(expected_set),
        "qdrant_count": len(qdrant_set),
        "neo4j_count": len(neo4j_set),
        "qdrant_missing": qdrant_missing[:20],
        "qdrant_extra": qdrant_extra[:20],
        "neo4j_missing": neo4j_missing[:20],
        "neo4j_extra": neo4j_extra[:20],
    }

    logger.info(
        "consistency-check doc=%s severity=%s expected=%d qdrant=%d neo4j=%d "
        "qdrant_missing=%d neo4j_missing=%d",
        document_id, severity, len(expected_set), len(qdrant_set), len(neo4j_set),
        len(qdrant_missing), len(neo4j_missing),
    )

    try:
        update_document_fields(document_id, {"consistency_check": report})
    except Exception as exc:  # noqa: BLE001
        logger.warning("consistency-check: Mongo write failed for %s: %s", document_id, exc)

    return report


__all__ = ["verify"]
