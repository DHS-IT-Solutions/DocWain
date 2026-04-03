"""Profile Intelligence Builder — pre-computes structured knowledge per profile.

Runs after document embedding to build domain-aware profile intelligence:
entity summaries, computed profiles, collection insights, and domain metadata.
Results are cached in MongoDB for injection into LLM context.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Domain detection mapping
# ---------------------------------------------------------------------------

_DOC_TYPE_TO_DOMAIN: Dict[str, str] = {
    "resume": "hr_recruitment",
    "cv": "hr_recruitment",
    "job_description": "hr_recruitment",
    "invoice": "finance",
    "purchase_order": "finance",
    "receipt": "finance",
    "financial_statement": "finance",
    "contract": "legal",
    "agreement": "legal",
    "nda": "legal",
    "lease": "legal",
    "lease_agreement": "legal",
    "inventory": "logistics",
    "shipping": "logistics",
    "manifest": "logistics",
    "warehouse": "logistics",
    "medical_record": "medical",
    "prescription": "medical",
    "lab_report": "medical",
}

VALID_PROFILE_TYPES = frozenset(
    {"hr_recruitment", "finance", "legal", "logistics", "medical", "generic"}
)


# ---------------------------------------------------------------------------
# ProfileIntelligence dataclass
# ---------------------------------------------------------------------------


@dataclass
class ProfileIntelligence:
    """Pre-computed intelligence for a document profile."""

    profile_id: str
    profile_type: str  # hr_recruitment | finance | legal | logistics | medical | generic
    document_count: int = 0
    last_updated: str = ""
    entities_summary: Dict[str, Any] = field(default_factory=dict)
    computed_profiles: List[Dict[str, Any]] = field(default_factory=list)
    collection_insights: Dict[str, Any] = field(default_factory=dict)
    domain_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.profile_type not in VALID_PROFILE_TYPES:
            self.profile_type = "generic"
        if not self.last_updated:
            self.last_updated = dt.datetime.utcnow().isoformat()


# ---------------------------------------------------------------------------
# ProfileBuilder
# ---------------------------------------------------------------------------


class ProfileBuilder:
    """Builds pre-computed profile intelligence from MongoDB + KG data."""

    # -- public API ----------------------------------------------------------

    @staticmethod
    def build(
        profile_id: str,
        subscription_id: str,
        mongo_client: Any,
        kg_client: Any = None,
    ) -> ProfileIntelligence:
        """Build or refresh profile intelligence for the given profile.

        Parameters
        ----------
        profile_id:
            The document-profile identifier.
        subscription_id:
            Tenant/subscription scope.
        mongo_client:
            A MongoDB client (or database handle) with access to ``documents``
            and ``computed_profiles`` collections.
        kg_client:
            Optional Neo4j store (``Neo4jStore`` instance). When provided,
            entities and relationships are queried for richer profiles.

        Returns
        -------
        ProfileIntelligence
            Fully populated intelligence object.
        """
        logger.info(
            "Building profile intelligence",
            extra={"profile_id": profile_id, "subscription_id": subscription_id},
        )

        # 1. Query document metadata from MongoDB
        doc_metadata = ProfileBuilder._fetch_doc_metadata(
            profile_id, subscription_id, mongo_client
        )

        # 2. Detect domain
        profile_type = ProfileBuilder._detect_domain(doc_metadata)

        # 3. Query KG for entities and relationships
        entities: List[Dict[str, Any]] = []
        relationships: List[Dict[str, Any]] = []
        if kg_client is not None:
            entities, relationships = ProfileBuilder._fetch_kg_data(
                profile_id, subscription_id, kg_client
            )

        # 4. Build entities summary
        entities_summary = ProfileBuilder._summarise_entities(entities)

        # 5. Compute domain-specific profiles
        from src.intelligence_v2.computed_profiles import compute_profiles

        computed = compute_profiles(profile_type, entities, relationships)

        # 6. Generate collection insights
        from src.intelligence_v2.collection_insights import generate_insights

        insights = generate_insights(profile_type, entities, relationships, doc_metadata)

        # 7. Assemble intelligence
        intelligence = ProfileIntelligence(
            profile_id=profile_id,
            profile_type=profile_type,
            document_count=len(doc_metadata),
            last_updated=dt.datetime.utcnow().isoformat(),
            entities_summary=entities_summary,
            computed_profiles=computed,
            collection_insights=insights,
            domain_metadata={
                "subscription_id": subscription_id,
                "detected_doc_types": list(
                    {d.get("doc_type", "unknown") for d in doc_metadata}
                ),
                "entity_count": len(entities),
                "relationship_count": len(relationships),
            },
        )

        # 8. Cache in MongoDB
        ProfileBuilder._store_cached(intelligence, mongo_client)

        logger.info(
            "Profile intelligence built",
            extra={
                "profile_id": profile_id,
                "profile_type": profile_type,
                "doc_count": len(doc_metadata),
                "entity_count": len(entities),
            },
        )
        return intelligence

    @staticmethod
    def get_cached(
        profile_id: str,
        mongo_client: Any,
    ) -> Optional[ProfileIntelligence]:
        """Retrieve cached profile intelligence from MongoDB.

        Returns ``None`` if no cached intelligence exists.
        """
        try:
            db = _get_db(mongo_client)
            doc = db["computed_profiles"].find_one({"profile_id": profile_id})
            if doc is None:
                return None
            doc.pop("_id", None)
            return ProfileIntelligence(**doc)
        except Exception:
            logger.warning(
                "Failed to retrieve cached intelligence",
                extra={"profile_id": profile_id},
                exc_info=True,
            )
            return None

    @staticmethod
    def to_context_block(intelligence: ProfileIntelligence) -> str:
        """Format intelligence as an XML context block for LLM injection.

        Example output::

            <profile_context profile_id="..." type="hr_recruitment" docs="5">
              ...structured summary...
            </profile_context>
        """
        lines: List[str] = [
            f'<profile_context profile_id="{intelligence.profile_id}" '
            f'type="{intelligence.profile_type}" '
            f'docs="{intelligence.document_count}">',
        ]

        # Entities summary
        if intelligence.entities_summary:
            lines.append("  <entities_summary>")
            for etype, names in intelligence.entities_summary.items():
                if isinstance(names, list):
                    lines.append(f"    {etype}: {', '.join(str(n) for n in names[:20])}")
                else:
                    lines.append(f"    {etype}: {names}")
            lines.append("  </entities_summary>")

        # Computed profiles (compact)
        if intelligence.computed_profiles:
            lines.append(f"  <computed_profiles count=\"{len(intelligence.computed_profiles)}\">")
            for cp in intelligence.computed_profiles[:10]:
                label = cp.get("label") or cp.get("name") or cp.get("id", "?")
                cp_type = cp.get("type", "item")
                lines.append(f"    <profile type=\"{cp_type}\" label=\"{label}\">")
                for k, v in cp.items():
                    if k not in ("label", "name", "id", "type"):
                        lines.append(f"      {k}: {v}")
                lines.append("    </profile>")
            lines.append("  </computed_profiles>")

        # Insights
        if intelligence.collection_insights:
            lines.append("  <insights>")
            for section, content in intelligence.collection_insights.items():
                lines.append(f"    <{section}>")
                if isinstance(content, list):
                    for item in content[:10]:
                        lines.append(f"      - {item}")
                elif isinstance(content, dict):
                    for k, v in content.items():
                        lines.append(f"      {k}: {v}")
                else:
                    lines.append(f"      {content}")
                lines.append(f"    </{section}>")
            lines.append("  </insights>")

        lines.append("</profile_context>")
        return "\n".join(lines)

    # -- internal helpers ----------------------------------------------------

    @staticmethod
    def _fetch_doc_metadata(
        profile_id: str,
        subscription_id: str,
        mongo_client: Any,
    ) -> List[Dict[str, Any]]:
        """Query MongoDB for all document metadata in this profile."""
        try:
            db = _get_db(mongo_client)
            cursor = db["documents"].find(
                {"profile_id": profile_id, "subscription_id": subscription_id},
                {
                    "_id": 0,
                    "document_id": 1,
                    "filename": 1,
                    "doc_type": 1,
                    "document_type": 1,
                    "created_at": 1,
                    "metadata": 1,
                },
            )
            return list(cursor)
        except Exception:
            logger.warning(
                "Failed to fetch document metadata from MongoDB",
                extra={"profile_id": profile_id},
                exc_info=True,
            )
            return []

    @staticmethod
    def _detect_domain(doc_metadata: List[Dict[str, Any]]) -> str:
        """Detect profile domain from document types and patterns."""
        if not doc_metadata:
            return "generic"

        domain_votes: Dict[str, int] = {}
        for doc in doc_metadata:
            raw_type = (doc.get("doc_type") or doc.get("document_type") or "").lower().strip()
            # Check direct mapping
            domain = _DOC_TYPE_TO_DOMAIN.get(raw_type)
            if domain is None:
                # Try partial match
                for keyword, d in _DOC_TYPE_TO_DOMAIN.items():
                    if keyword in raw_type:
                        domain = d
                        break
            if domain:
                domain_votes[domain] = domain_votes.get(domain, 0) + 1

        if not domain_votes:
            return "generic"

        # Return the domain with the most votes
        return max(domain_votes, key=domain_votes.get)  # type: ignore[arg-type]

    @staticmethod
    def _fetch_kg_data(
        profile_id: str,
        subscription_id: str,
        kg_client: Any,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Query Neo4j KG for entities and relationships in this profile."""
        entities: List[Dict[str, Any]] = []
        relationships: List[Dict[str, Any]] = []
        try:
            # Fetch entities
            entity_query = (
                "MATCH (e:Entity {subscription_id: $sid, profile_id: $pid}) "
                "RETURN e.entity_id AS id, e.name AS name, e.type AS type "
                "LIMIT 500"
            )
            entities = kg_client.run_query(
                entity_query, {"sid": subscription_id, "pid": profile_id}
            )

            # Fetch relationships
            rel_query = (
                "MATCH (e1:Entity {subscription_id: $sid, profile_id: $pid})"
                "-[r:RELATED_TO]->"
                "(e2:Entity {subscription_id: $sid, profile_id: $pid}) "
                "RETURN e1.name AS source, type(r) AS relation, "
                "r.relation_type AS relation_type, e2.name AS target, "
                "r.frequency AS frequency "
                "LIMIT 500"
            )
            relationships = kg_client.run_query(
                rel_query, {"sid": subscription_id, "pid": profile_id}
            )
        except Exception:
            logger.warning(
                "Failed to fetch KG data",
                extra={"profile_id": profile_id},
                exc_info=True,
            )

        return entities, relationships

    @staticmethod
    def _summarise_entities(entities: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Group entities by type and return top names per type."""
        by_type: Dict[str, List[str]] = {}
        for e in entities:
            etype = e.get("type", "OTHER")
            name = e.get("name", "")
            if name:
                by_type.setdefault(etype, []).append(name)

        # Deduplicate and limit
        return {
            etype: sorted(set(names))[:30]
            for etype, names in by_type.items()
        }

    @staticmethod
    def _store_cached(
        intelligence: ProfileIntelligence,
        mongo_client: Any,
    ) -> None:
        """Upsert intelligence into MongoDB ``computed_profiles`` collection."""
        try:
            db = _get_db(mongo_client)
            data = asdict(intelligence)
            db["computed_profiles"].replace_one(
                {"profile_id": intelligence.profile_id},
                data,
                upsert=True,
            )
        except Exception:
            logger.warning(
                "Failed to cache profile intelligence",
                extra={"profile_id": intelligence.profile_id},
                exc_info=True,
            )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _get_db(mongo_client: Any) -> Any:
    """Return the database handle from a client or pass through if already a db."""
    # Support both pymongo.MongoClient and a direct database handle
    if hasattr(mongo_client, "get_database"):
        return mongo_client.get_database()
    if hasattr(mongo_client, "documents"):
        # Already a database handle
        return mongo_client
    # Assume it's a client with a default db
    return mongo_client
