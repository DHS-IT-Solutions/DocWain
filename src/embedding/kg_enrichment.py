"""KG-enriched embedding text utilities.

Prepends knowledge-graph context (document name + entity tags) to chunk text
so that embedding models capture named entity signal.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Redis cache key template
_CACHE_KEY_TPL = "kg_ctx:{doc_id}:{chunk_id}"
_CACHE_TTL = 3600  # seconds


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def enrich_chunk_text(
    chunk_text: str,
    kg_context: Optional[Dict[str, Any]],
    *,
    max_prefix_chars: int = 200,
) -> str:
    """Prepend KG context tags to *chunk_text*.

    Format: ``[Doc: {name}] [{entity1}] [{entity2}] {original_text}``

    Entity tags are added in insertion order until adding the next tag would
    push the prefix beyond *max_prefix_chars*.  If *kg_context* is ``None``
    or empty the original text is returned unchanged.

    Parameters
    ----------
    chunk_text:
        The raw chunk text to enrich.
    kg_context:
        Dict with keys ``"document"`` (str) and ``"entities"``
        (dict mapping entity name → entity type).
    max_prefix_chars:
        Hard ceiling on the length of the prepended prefix (not including the
        original text).
    """
    if not kg_context:
        return chunk_text

    doc_name: str = kg_context.get("document", "")
    entities: Dict[str, str] = kg_context.get("entities", {}) or {}

    prefix_parts: list[str] = []
    prefix_len = 0

    if doc_name:
        tag = f"[Doc: {doc_name}]"
        if prefix_len + len(tag) + 1 <= max_prefix_chars:
            prefix_parts.append(tag)
            prefix_len += len(tag) + 1  # +1 for the space separator

    for entity_name in entities:
        tag = f"[{entity_name}]"
        needed = prefix_len + len(tag) + 1
        if needed > max_prefix_chars:
            break
        prefix_parts.append(tag)
        prefix_len += len(tag) + 1

    if not prefix_parts:
        return chunk_text

    prefix = " ".join(prefix_parts)
    return f"{prefix} {chunk_text}"


def fetch_kg_context_for_chunk(
    document_id: str,
    chunk_id: str,
    *,
    neo4j_store=None,
    redis_client=None,
) -> Optional[Dict[str, Any]]:
    """Retrieve KG context for a chunk, using Redis as a cache layer.

    Lookup order:
    1. Redis cache (key ``kg_ctx:{doc_id}:{chunk_id}``, TTL 3600 s).
    2. Neo4j query for document entities.

    Returns a dict ``{"document": str, "entities": {name: type}}`` or
    ``None`` when no context can be found.

    Parameters
    ----------
    document_id:
        The document identifier used to scope the Neo4j query.
    chunk_id:
        The chunk identifier; combined with *document_id* for the cache key.
    neo4j_store:
        An optional Neo4j store/session object.  Must support a
        ``query_entities(document_id)`` method returning
        ``list[dict]`` with ``name`` and ``type`` keys, OR a raw
        ``driver`` attribute for direct Cypher execution.
    redis_client:
        An optional Redis client.  Must support ``get(key)`` and
        ``set(key, value, ex=ttl)``.
    """
    cache_key = _CACHE_KEY_TPL.format(doc_id=document_id, chunk_id=chunk_id)

    # 1. Try Redis cache first
    if redis_client is not None:
        try:
            cached = redis_client.get(cache_key)
            if cached is not None:
                raw = cached if isinstance(cached, str) else cached.decode()
                return json.loads(raw)
        except Exception:  # noqa: BLE001
            logger.debug("Redis cache miss or error for key %s", cache_key)

    # 2. Query Neo4j
    context: Optional[Dict[str, Any]] = None

    if neo4j_store is not None:
        try:
            context = _query_neo4j(neo4j_store, document_id)
        except Exception:  # noqa: BLE001
            logger.warning(
                "Neo4j query failed for document_id=%s", document_id, exc_info=True
            )

    if context is None:
        return None

    # 3. Cache result in Redis
    if redis_client is not None:
        try:
            redis_client.set(cache_key, json.dumps(context), ex=_CACHE_TTL)
        except Exception:  # noqa: BLE001
            logger.debug("Failed to cache KG context for key %s", cache_key)

    return context


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _query_neo4j(neo4j_store: Any, document_id: str) -> Optional[Dict[str, Any]]:
    """Extract document name and entities from Neo4j.

    Supports stores that expose either:
    - ``query_entities(document_id)`` → list of {name, type, ...} dicts, or
    - ``driver`` attribute for raw Cypher execution.
    """
    # Prefer the high-level helper if available
    if hasattr(neo4j_store, "query_entities"):
        rows = neo4j_store.query_entities(document_id)
        if rows is None:
            return None
        entities: Dict[str, str] = {}
        doc_name = ""
        for row in rows:
            if isinstance(row, dict):
                name = row.get("name", "")
                etype = row.get("type", row.get("entity_type", ""))
                doc_label = row.get("document", "")
                if name:
                    entities[name] = etype
                if doc_label and not doc_name:
                    doc_name = doc_label
        return {"document": doc_name, "entities": entities}

    # Fallback: raw driver with Cypher
    if hasattr(neo4j_store, "driver"):
        cypher = (
            "MATCH (d:Document {document_id: $doc_id})-[:HAS_ENTITY]->(e:Entity) "
            "RETURN d.name AS document, e.name AS name, e.type AS type"
        )
        with neo4j_store.driver.session() as session:
            result = session.run(cypher, doc_id=document_id)
            entities = {}
            doc_name = ""
            for record in result:
                name = record.get("name", "")
                etype = record.get("type", "")
                doc_label = record.get("document", "")
                if name:
                    entities[name] = etype
                if doc_label and not doc_name:
                    doc_name = doc_label
        return {"document": doc_name, "entities": entities}

    return None
