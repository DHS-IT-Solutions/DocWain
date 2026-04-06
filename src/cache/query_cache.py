from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class QueryCache:
    """Three-tier query cache in Redis.

    Tier 1: Query embedding cache -- avoids re-encoding identical queries
    Tier 2: Search results cache -- avoids Qdrant round-trip
    Tier 3: Full response cache -- instant response for repeated questions (opt-in)
    """

    EMBEDDING_TTL = 3600      # 1 hour
    SEARCH_TTL = 300          # 5 minutes
    RESPONSE_TTL = 300        # 5 minutes

    PREFIX = "dw:qcache"

    def __init__(self, redis_client=None):
        self._redis = redis_client

    def _get_redis(self):
        if self._redis is not None:
            return self._redis
        try:
            from src.api.dw_newron import get_redis_client
            self._redis = get_redis_client()
        except Exception:
            logger.debug("QueryCache: Redis client unavailable")
        return self._redis

    @staticmethod
    def _hash_key(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    # -- Tier 1: Embedding cache ------------------------------------------------

    def get_embedding(self, query_text: str) -> Optional[List[float]]:
        try:
            r = self._get_redis()
            if r is None:
                return None
            key = f"{self.PREFIX}:emb:{self._hash_key(query_text)}"
            raw = r.get(key)
            if raw is None:
                return None
            if isinstance(raw, (bytes, bytearray)):
                raw = raw.decode("utf-8")
            return json.loads(raw)
        except Exception:
            logger.debug("QueryCache: get_embedding failed", exc_info=True)
            return None

    def set_embedding(self, query_text: str, embedding: List[float]) -> None:
        try:
            r = self._get_redis()
            if r is None:
                return
            key = f"{self.PREFIX}:emb:{self._hash_key(query_text)}"
            r.setex(key, self.EMBEDDING_TTL, json.dumps(embedding))
        except Exception:
            logger.debug("QueryCache: set_embedding failed", exc_info=True)

    # -- Tier 2: Search results cache -------------------------------------------

    def _collection_version(self, collection: str) -> int:
        """Return the current version counter for a collection."""
        try:
            r = self._get_redis()
            if r is None:
                return 0
            raw = r.get(f"{self.PREFIX}:version:{collection}")
            if raw is None:
                return 0
            if isinstance(raw, (bytes, bytearray)):
                raw = raw.decode("utf-8")
            return int(raw)
        except Exception:
            return 0

    def get_search_results(self, query_text: str, collection: str) -> Optional[List[Dict]]:
        try:
            r = self._get_redis()
            if r is None:
                return None
            ver = self._collection_version(collection)
            key = f"{self.PREFIX}:search:{collection}:v{ver}:{self._hash_key(query_text)}"
            raw = r.get(key)
            if raw is None:
                return None
            if isinstance(raw, (bytes, bytearray)):
                raw = raw.decode("utf-8")
            return json.loads(raw)
        except Exception:
            logger.debug("QueryCache: get_search_results failed", exc_info=True)
            return None

    def set_search_results(self, query_text: str, collection: str, results: List[Dict]) -> None:
        try:
            r = self._get_redis()
            if r is None:
                return
            ver = self._collection_version(collection)
            key = f"{self.PREFIX}:search:{collection}:v{ver}:{self._hash_key(query_text)}"
            r.setex(key, self.SEARCH_TTL, json.dumps(results))
        except Exception:
            logger.debug("QueryCache: set_search_results failed", exc_info=True)

    # -- Tier 3: Full response cache (opt-in per tenant) ------------------------

    def get_response(self, query_text: str, profile_id: str) -> Optional[Dict]:
        try:
            r = self._get_redis()
            if r is None:
                return None
            key = f"{self.PREFIX}:resp:{profile_id}:{self._hash_key(query_text)}"
            raw = r.get(key)
            if raw is None:
                return None
            if isinstance(raw, (bytes, bytearray)):
                raw = raw.decode("utf-8")
            return json.loads(raw)
        except Exception:
            logger.debug("QueryCache: get_response failed", exc_info=True)
            return None

    def set_response(self, query_text: str, profile_id: str, response: Dict) -> None:
        try:
            r = self._get_redis()
            if r is None:
                return
            key = f"{self.PREFIX}:resp:{profile_id}:{self._hash_key(query_text)}"
            r.setex(key, self.RESPONSE_TTL, json.dumps(response))
        except Exception:
            logger.debug("QueryCache: set_response failed", exc_info=True)

    # -- Invalidation -----------------------------------------------------------

    def invalidate_collection(self, collection: str) -> None:
        """Invalidate all search caches for a collection (called when new docs are embedded).

        Uses a version counter pattern: bumping the version makes all prior
        search-cache keys stale (they will simply never be read again and expire
        naturally via TTL).
        """
        try:
            r = self._get_redis()
            if r is None:
                return
            r.incr(f"{self.PREFIX}:version:{collection}")
        except Exception:
            logger.debug("QueryCache: invalidate_collection failed", exc_info=True)


# Singleton -- lazily resolves Redis on first use
query_cache = QueryCache()
