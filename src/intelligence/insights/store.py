"""InsightStore — single writer for Mongo + Qdrant + Neo4j.

Mongo (control plane) holds index records only. Body, quotes, and KB
refs go to Qdrant payload + Neo4j. Per feedback_storage_separation.md:
Mongo is control plane only; document content goes to Blob/Qdrant.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

from src.intelligence.insights.schema import Insight
from src.intelligence.insights.validators import (
    require_doc_evidence,
    require_body_grounded,
    compute_dedup_key,
)

logger = logging.getLogger(__name__)


class MongoCollection(Protocol):
    def update_one(self, filter, update, upsert=False) -> Any: ...
    def find(self, query) -> Any: ...


@dataclass
class MongoIndexBackend:
    collection: MongoCollection

    def upsert(self, dedup_key: str, doc: Dict[str, Any]) -> None:
        self.collection.update_one(
            {"dedup_key": dedup_key},
            {"$set": doc},
            upsert=True,
        )

    def list(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Exclude Mongo internals so payload is JSON-serializable
        try:
            cursor = self.collection.find(query, {"_id": 0})
        except TypeError:
            # Fakes used in tests don't accept projection
            cursor = self.collection.find(query)
        return list(cursor)


class QdrantBackend(Protocol):
    def upsert_insight(self, *, insight: Insight) -> None: ...


class Neo4jBackend(Protocol):
    def upsert_insight(self, *, insight: Insight) -> None: ...


class InsightStore:
    def __init__(
        self,
        *,
        mongo_index: MongoIndexBackend,
        qdrant: Optional[QdrantBackend],
        neo4j: Optional[Neo4jBackend],
    ):
        self._mongo = mongo_index
        self._qdrant = qdrant
        self._neo4j = neo4j

    def write(self, insight: Insight) -> None:
        require_doc_evidence(insight)
        require_body_grounded(insight)
        dedup_key = compute_dedup_key(insight)

        index_doc = {
            "insight_id": insight.insight_id,
            "dedup_key": dedup_key,
            "profile_id": insight.profile_id,
            "subscription_id": insight.subscription_id,
            "document_ids": list(insight.document_ids),
            "domain": insight.domain,
            "insight_type": insight.insight_type,
            "severity": insight.severity,
            "tags": list(insight.tags),
            "refreshed_at": insight.refreshed_at,
            "stale": insight.stale,
            "adapter_version": insight.adapter_version,
        }
        self._mongo.upsert(dedup_key=dedup_key, doc=index_doc)

        if self._qdrant is not None:
            self._qdrant.upsert_insight(insight=insight)
        if self._neo4j is not None:
            self._neo4j.upsert_insight(insight=insight)

    def list_for_profile(
        self,
        *,
        profile_id: str,
        insight_types=None,
        severities=None,
        domain=None,
        since=None,
        limit: int = 50,
        offset: int = 0,
    ):
        query: Dict[str, Any] = {"profile_id": profile_id}
        if insight_types:
            query["insight_type"] = {"$in": list(insight_types)}
        if severities:
            query["severity"] = {"$in": list(severities)}
        if domain:
            query["domain"] = domain
        rows = self._mongo.list(query)
        if since:
            rows = [r for r in rows if r.get("refreshed_at", "") >= since]
        return rows[offset : offset + limit]

    def get_by_id(self, *, insight_id: str):
        rows = self._mongo.list({"insight_id": insight_id})
        return rows[0] if rows else None


class QdrantInsightBackend:
    """Real Qdrant backend for the `insights` collection.

    Uses an existing Qdrant client provided at wiring time. Embedding of
    headline+body is done by an injected embedder.
    """

    def __init__(self, *, client, collection_name: str = "insights", embedder=None):
        self._client = client
        self._collection = collection_name
        self._embedder = embedder

    def upsert_insight(self, *, insight: Insight) -> None:
        from qdrant_client.http.models import PointStruct
        text = f"{insight.headline}\n\n{insight.body}"
        vector = self._embedder.embed(text) if self._embedder else [0.0] * 384
        payload = {
            "insight_id": insight.insight_id,
            "profile_id": insight.profile_id,
            "subscription_id": insight.subscription_id,
            "document_ids": list(insight.document_ids),
            "insight_type": insight.insight_type,
            "severity": insight.severity,
            "tags": list(insight.tags),
            "domain": insight.domain,
            "headline": insight.headline,
            "body": insight.body,
            "evidence_doc_spans": [s.__dict__ for s in insight.evidence_doc_spans],
            "external_kb_refs": [r.__dict__ for r in insight.external_kb_refs],
            "adapter_version": insight.adapter_version,
            "refreshed_at": insight.refreshed_at,
        }
        self._client.upsert(
            collection_name=self._collection,
            points=[PointStruct(id=insight.insight_id, vector=vector, payload=payload)],
        )


class Neo4jInsightBackend:
    """Real Neo4j backend.

    Schema: (:Insight {insight_id, headline, severity, insight_type})
            -[:GROUNDED_IN]-> (:Document {document_id})
            (:Insight) -[:OF_PROFILE]-> (:Profile {profile_id})
    """

    def __init__(self, *, driver):
        self._driver = driver

    def upsert_insight(self, *, insight: Insight) -> None:
        cypher = (
            "MERGE (i:Insight {insight_id: $insight_id}) "
            "SET i.headline = $headline, "
            "    i.severity = $severity, "
            "    i.insight_type = $insight_type, "
            "    i.refreshed_at = $refreshed_at, "
            "    i.adapter_version = $adapter_version "
            "MERGE (p:Profile {profile_id: $profile_id}) "
            "MERGE (i)-[:OF_PROFILE]->(p) "
            "WITH i "
            "UNWIND $document_ids AS doc_id "
            "  MERGE (d:Document {document_id: doc_id}) "
            "  MERGE (i)-[:GROUNDED_IN]->(d)"
        )
        with self._driver.session() as sess:
            sess.run(
                cypher,
                insight_id=insight.insight_id,
                headline=insight.headline,
                severity=insight.severity,
                insight_type=insight.insight_type,
                refreshed_at=insight.refreshed_at,
                adapter_version=insight.adapter_version,
                profile_id=insight.profile_id,
                document_ids=list(insight.document_ids),
            )
