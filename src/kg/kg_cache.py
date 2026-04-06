"""In-memory KG entity cache for fast query-time lookups."""

import threading
import time
from typing import Dict, List, Optional

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_ENTITY_CATALOG_CYPHER = (
    "MATCH (e:Entity) "
    "WITH e.name AS name, labels(e) AS types, e.doc_count AS doc_count "
    "RETURN name, head([t IN types WHERE t <> 'Entity']) AS type, "
    "coalesce(doc_count, 1) AS doc_count "
    "ORDER BY doc_count DESC LIMIT 10000"
)

_RELATIONSHIP_SCHEMA_CYPHER = (
    "MATCH ()-[r]->() "
    "WITH type(r) AS rel_type, count(*) AS freq "
    "RETURN rel_type, freq ORDER BY freq DESC LIMIT 100"
)

_REFRESH_INTERVAL = 15 * 60  # 15 minutes


class KGCache:
    """Pre-loaded KG entity catalog for O(1) entity lookups."""

    def __init__(self) -> None:
        self._entities: List[Dict] = []
        self._entity_index: Dict[str, Dict] = {}
        self._relationship_schema: List[Dict] = []
        self._is_warmed: bool = False
        self._last_refresh: float = 0.0
        self._lock = threading.Lock()

    @property
    def entity_count(self) -> int:
        return len(self._entities)

    @property
    def is_warmed(self) -> bool:
        return self._is_warmed

    def warm(self, neo4j_store) -> None:
        """Load KG index from Neo4j. Survives Neo4j failure gracefully."""
        try:
            entities = neo4j_store.run_query(_ENTITY_CATALOG_CYPHER)
            relationships = neo4j_store.run_query(_RELATIONSHIP_SCHEMA_CYPHER)

            with self._lock:
                self._entities = entities
                self._entity_index = {e["name"]: e for e in entities}
                self._relationship_schema = relationships
                self._is_warmed = True
                self._last_refresh = time.time()

            logger.info(
                "KG cache warmed: %d entities, %d relationship types",
                len(entities),
                len(relationships),
            )
        except Exception as exc:
            logger.warning("KG cache warming failed (will retry later): %s", exc)

    def get_entity_catalog(self) -> List[Dict]:
        return list(self._entities)

    def get_relationship_schema(self) -> List[Dict]:
        return list(self._relationship_schema)

    def lookup_entity(self, name: str) -> Optional[Dict]:
        return self._entity_index.get(name)

    def needs_refresh(self) -> bool:
        """True if > 15 min since last refresh."""
        return (time.time() - self._last_refresh) > _REFRESH_INTERVAL


# Module-level singleton
kg_cache = KGCache()
