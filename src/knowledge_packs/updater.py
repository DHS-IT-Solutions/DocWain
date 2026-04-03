"""Monthly update job for knowledge packs.

Handles full initial loads and incremental updates by scraping, parsing,
and upserting documents into Qdrant pack-specific collections.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.knowledge_packs.base import KnowledgePackConfig, ParsedDocument
from src.knowledge_packs.registry import PackRegistry
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class PackUpdater:
    """Scrapes, parses, and indexes knowledge pack content into Qdrant."""

    def __init__(
        self,
        qdrant_client: Any,
        embedding_fn: Any = None,
        vector_size: int = 1024,
    ) -> None:
        """
        Parameters
        ----------
        qdrant_client:
            Qdrant client instance (``qdrant_client.QdrantClient``).
        embedding_fn:
            Callable that takes a list of strings and returns a list of
            vectors (list of floats). If ``None``, a placeholder zero-vector
            is used (useful for testing collection structure).
        vector_size:
            Dimension of embedding vectors.
        """
        self._qdrant = qdrant_client
        self._embed = embedding_fn
        self._vector_size = vector_size
        self._registry = PackRegistry()

    # -- Public API ----------------------------------------------------------

    def update_pack(self, domain: str, region: str) -> Dict[str, Any]:
        """Incremental update: scrape recent changes, parse, and upsert.

        Returns stats dict with counts of added, updated, and errors.
        """
        pack = self._registry.get_pack(domain, region)
        if pack is None:
            return {"error": f"No pack registered for {domain}/{region}"}

        config, scraper_cls, parser_cls = pack
        scraper = scraper_cls()
        parser = parser_cls()

        logger.info("Starting incremental update", extra={"pack": config.name})

        # Check for updates since 30 days ago (overlap for safety)
        since = datetime(
            datetime.utcnow().year,
            datetime.utcnow().month,
            1,
        )
        try:
            raw_items = scraper.check_updates(since)
        except Exception:
            logger.error("Scraper check_updates failed", extra={"pack": config.name}, exc_info=True)
            return {"error": "Scraper failed", "pack": config.name}

        return self._process_and_index(config, parser, raw_items)

    def update_all(self) -> Dict[str, Any]:
        """Update all registered packs. Returns per-pack stats."""
        results: Dict[str, Any] = {}
        for pack_info in self._registry.list_packs():
            domain = pack_info["domain"]
            region = pack_info["region"]
            key = f"{domain}/{region}"
            results[key] = self.update_pack(domain, region)
        return results

    def initial_load(self, domain: str, region: str) -> Dict[str, Any]:
        """Full scrape + index for first install of a pack.

        Creates the Qdrant collection if it does not exist, then performs
        a complete scrape and index.
        """
        pack = self._registry.get_pack(domain, region)
        if pack is None:
            return {"error": f"No pack registered for {domain}/{region}"}

        config, scraper_cls, parser_cls = pack
        scraper = scraper_cls()
        parser = parser_cls()

        logger.info("Starting initial load", extra={"pack": config.name})

        # Ensure collection exists
        self._ensure_collection(config.qdrant_collection)

        # Full scrape
        try:
            raw_items = scraper.scrape()
        except Exception:
            logger.error("Scraper full scrape failed", extra={"pack": config.name}, exc_info=True)
            return {"error": "Scraper failed", "pack": config.name}

        return self._process_and_index(config, parser, raw_items)

    # -- Internals -----------------------------------------------------------

    def _process_and_index(
        self,
        config: KnowledgePackConfig,
        parser: Any,
        raw_items: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Parse raw items and upsert into Qdrant."""
        stats = {"added": 0, "updated": 0, "errors": 0, "total_raw": len(raw_items)}

        self._ensure_collection(config.qdrant_collection)

        for raw in raw_items:
            try:
                documents = parser.parse(raw)
            except Exception:
                logger.warning("Parser failed for item", exc_info=True)
                stats["errors"] += 1
                continue

            for doc in documents:
                try:
                    self._upsert_document(config, doc)
                    stats["added"] += 1
                except Exception:
                    logger.warning("Upsert failed for document", extra={"title": doc.title}, exc_info=True)
                    stats["errors"] += 1

        logger.info(
            "Pack update complete",
            extra={"pack": config.name, **stats},
        )
        return stats

    def _upsert_document(
        self,
        config: KnowledgePackConfig,
        doc: ParsedDocument,
    ) -> None:
        """Upsert a single parsed document into Qdrant."""
        full_text = doc.full_text()
        if not full_text.strip():
            return

        # Generate a deterministic ID from source URL + title
        content_hash = hashlib.sha256(
            f"{doc.source_url}:{doc.title}".encode()
        ).hexdigest()[:16]
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, content_hash))

        # Embed
        vector = self._get_vector(full_text)

        # Build payload
        payload = {
            "title": doc.title,
            "source_url": doc.source_url,
            "pack_name": config.name,
            "domain": config.domain,
            "region": config.region,
            "citation_format": config.citation_format,
            "last_updated": doc.last_updated,
            "content": full_text[:10000],  # Cap stored content
            "sections": doc.sections[:50],
            "metadata": doc.metadata,
        }

        # Upsert to Qdrant
        try:
            from qdrant_client.models import PointStruct

            self._qdrant.upsert(
                collection_name=config.qdrant_collection,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload,
                    )
                ],
            )
        except ImportError:
            # Fallback for environments without qdrant_client models
            self._qdrant.upsert(
                collection_name=config.qdrant_collection,
                points=[{
                    "id": point_id,
                    "vector": vector,
                    "payload": payload,
                }],
            )

    def _get_vector(self, text: str) -> List[float]:
        """Embed text or return a zero vector if no embedding function is set."""
        if self._embed is not None:
            try:
                vectors = self._embed([text])
                if vectors and len(vectors) > 0:
                    return vectors[0]
            except Exception:
                logger.warning("Embedding failed, using zero vector", exc_info=True)
        return [0.0] * self._vector_size

    def _ensure_collection(self, collection_name: str) -> None:
        """Create Qdrant collection if it does not exist."""
        try:
            collections = self._qdrant.get_collections()
            existing = {c.name for c in collections.collections}
            if collection_name in existing:
                return
        except Exception:
            pass

        try:
            from qdrant_client.models import Distance, VectorParams

            self._qdrant.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self._vector_size,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("Created Qdrant collection", extra={"collection": collection_name})
        except ImportError:
            # Minimal fallback
            self._qdrant.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "size": self._vector_size,
                    "distance": "Cosine",
                },
            )
        except Exception:
            logger.warning(
                "Failed to create collection (may already exist)",
                extra={"collection": collection_name},
                exc_info=True,
            )
