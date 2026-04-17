"""Personalized-PageRank retriever over the Neo4j entity graph.

Inspired by HippoRAG — given a query, extract its entities, seed them on
the corpus's entity graph, run Personalized PageRank to surface
semantically-adjacent entities (including ones that DIDN'T lexically
match the query), then pull the chunks those entities live in.

This is the multi-hop complement to dense vector retrieval: dense catches
chunks whose text embeds near the query; PPR catches chunks that are
topically connected through the entity graph, even when the chunk's
own text doesn't lexically resemble the query.

Graph model used:
    * Nodes: ``Entity`` in Neo4j, keyed by ``entity_id``
    * Edges: ``CO_OCCURS_WITH`` between entities (undirected, weight 1.0)
    * Chunks are reached after PPR via ``Chunk-[MENTIONS]->Entity``

Graph is loaded lazily and cached per ``(subscription_id, profile_id)``.
Cache entries expire after ``_CACHE_TTL_S`` so newly-ingested docs become
visible without an API restart. Size of cached graph per profile is
bounded by the number of entities in that profile (typically a few K to
low tens of K — easy fit in memory).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_CACHE_TTL_S = 600.0  # 10 minutes
_MAX_SEED_ENTITIES = 8
_MAX_TOP_ENTITIES = 100
_PPR_ALPHA = 0.85
_PPR_MAX_ITER = 30
_PPR_TOL = 1.0e-4


class PPRRetriever:
    """Personalized-PageRank retrieval over the entity graph.

    Parameters
    ----------
    neo4j_driver
        An open ``neo4j.GraphDatabase.driver``. If ``None``, one is built
        on first use from ``Config.Neo4j``.
    entity_extractor
        Optional ``src.kg.entity_extractor.EntityExtractor`` — re-used from
        ``GraphAugmenter`` to keep query-entity extraction consistent.
    """

    def __init__(
        self,
        *,
        neo4j_driver: Any = None,
        entity_extractor: Any = None,
    ) -> None:
        self._driver = neo4j_driver
        self._extractor = entity_extractor
        # Cache: (subscription_id, profile_id) -> (networkx.Graph, loaded_at)
        self._graph_cache: Dict[Tuple[str, str], Tuple[Any, float]] = {}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        subscription_id: str,
        profile_id: str,
        *,
        qdrant_client: Any = None,
        collection_name: Optional[str] = None,
        top_k: int = 30,
    ) -> List[Dict[str, Any]]:
        """Return up to ``top_k`` chunks surfaced by PPR.

        Each result is ``{"chunk_id": str, "document_id": str, "score": float}``.
        Chunks have PPR-relevance scores; the caller is responsible for
        hydrating text (via Qdrant lookup) and merging with other signals.
        """
        if not query or not str(query).strip():
            return []

        seeds = self._extract_seed_entity_ids(query, subscription_id, profile_id)
        if not seeds:
            logger.debug("[PPR] no seed entities matched in KG for query=%r", query[:80])
            return []

        graph = self._load_graph(subscription_id, profile_id)
        if graph is None or graph.number_of_nodes() == 0:
            logger.debug("[PPR] empty entity graph for sub=%s prof=%s", subscription_id, profile_id)
            return []

        # Keep only seeds that actually exist in the graph
        valid_seeds = [s for s in seeds if s in graph]
        if not valid_seeds:
            logger.debug("[PPR] seeds present in KG but not in co-occurrence graph")
            return []

        try:
            import networkx as nx  # local import — heavy dep
        except Exception as exc:  # noqa: BLE001
            logger.warning("[PPR] networkx unavailable: %s", exc)
            return []

        personalization = {n: 0.0 for n in graph.nodes}
        w = 1.0 / len(valid_seeds)
        for s in valid_seeds:
            personalization[s] = w

        t0 = time.monotonic()
        try:
            scores = nx.pagerank(
                graph,
                alpha=_PPR_ALPHA,
                personalization=personalization,
                max_iter=_PPR_MAX_ITER,
                tol=_PPR_TOL,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[PPR] PageRank failed: %s", exc)
            return []
        ppr_elapsed = time.monotonic() - t0

        # Drop seeds from results (they match the query verbatim — dense
        # retrieval already finds their chunks). We want the NEW entities
        # PPR brings in through multi-hop propagation.
        seed_set = set(valid_seeds)
        top_entities = [
            (eid, s) for eid, s in
            sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            if eid not in seed_set and s > 0
        ][:_MAX_TOP_ENTITIES]

        if not top_entities:
            return []

        entity_to_score = dict(top_entities)
        chunks = self._fetch_chunks_by_entities(
            entity_ids=list(entity_to_score.keys()),
            subscription_id=subscription_id,
            profile_id=profile_id,
        )

        # Aggregate chunk scores: a chunk's PPR relevance is the sum of the
        # PPR scores of the entities it mentions. Capped at the max entity
        # score so a single chunk with many mentions doesn't dominate.
        chunk_scores: Dict[str, float] = {}
        chunk_docs: Dict[str, str] = {}
        for ch in chunks:
            cid = ch.get("chunk_id")
            if not cid:
                continue
            eid = ch.get("entity_id")
            if not eid:
                continue
            chunk_scores[cid] = chunk_scores.get(cid, 0.0) + entity_to_score.get(eid, 0.0)
            chunk_docs[cid] = ch.get("document_id") or chunk_docs.get(cid, "")

        if not chunk_scores:
            return []

        # Normalise scores to [0, 1] so they merge sanely with other signals
        max_s = max(chunk_scores.values()) or 1.0
        results = [
            {"chunk_id": cid, "document_id": chunk_docs.get(cid, ""),
             "score": s / max_s}
            for cid, s in sorted(chunk_scores.items(), key=lambda kv: kv[1], reverse=True)
        ][:top_k]

        logger.info(
            "[PPR] seeds=%d nodes=%d top_entities=%d chunks=%d ppr_elapsed=%.3fs",
            len(valid_seeds), graph.number_of_nodes(), len(entity_to_score), len(results),
            ppr_elapsed,
        )
        return results

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_driver(self):
        if self._driver is not None:
            return self._driver
        from neo4j import GraphDatabase
        from src.api.config import Config
        self._driver = GraphDatabase.driver(
            Config.Neo4j.URI, auth=(Config.Neo4j.USER, Config.Neo4j.PASSWORD)
        )
        return self._driver

    def _get_extractor(self):
        if self._extractor is not None:
            return self._extractor
        from src.kg.entity_extractor import EntityExtractor
        self._extractor = EntityExtractor()
        return self._extractor

    def _extract_seed_entity_ids(
        self,
        query: str,
        subscription_id: str,
        profile_id: str,
    ) -> List[str]:
        """Extract entities from the query and look up their entity_ids in Neo4j."""
        extractor = self._get_extractor()
        try:
            extracted = extractor.extract_with_metadata(query)
        except Exception as exc:  # noqa: BLE001
            logger.debug("[PPR] entity extraction failed: %s", exc)
            return []
        if not extracted:
            return []

        names = [e.normalized_name for e in extracted][:_MAX_SEED_ENTITIES]
        if not names:
            return []

        driver = self._get_driver()
        cypher = (
            "UNWIND $names AS name "
            "MATCH (e:Entity) "
            "WHERE e.subscription_id = $sub "
            "  AND e.profile_id = $prof "
            "  AND (e.normalized_name = name "
            "       OR toLower(e.name) = toLower(name) "
            "       OR toLower(e.value) = toLower(name)) "
            "RETURN DISTINCT e.entity_id AS entity_id"
        )
        try:
            with driver.session() as session:
                records = session.run(cypher, names=names, sub=str(subscription_id), prof=str(profile_id))
                return [r["entity_id"] for r in records if r["entity_id"]]
        except Exception as exc:  # noqa: BLE001
            logger.warning("[PPR] seed lookup failed: %s", exc)
            return []

    def _load_graph(self, subscription_id: str, profile_id: str):
        """Load (or fetch from cache) the entity co-occurrence graph for a profile.

        Edges are derived from the intelligence KG's ``Section-[:ABOUT]->Entity``
        layer: two entities share an edge if they're both ABOUT the same
        section, with weight equal to the count of shared sections. The legacy
        ``CO_OCCURS_WITH`` relationship (written by the separate ``kg_store``
        path) keys entities by ``entity_key`` and is disjoint from the
        ``entity_id``-keyed entity set that PPR seeds and traverses chunks
        through, so it can't be used directly.
        """
        key = (str(subscription_id), str(profile_id))
        now = time.monotonic()
        cached = self._graph_cache.get(key)
        if cached is not None:
            graph, loaded_at = cached
            if now - loaded_at < _CACHE_TTL_S:
                return graph

        try:
            import networkx as nx
        except Exception as exc:  # noqa: BLE001
            logger.warning("[PPR] networkx unavailable: %s", exc)
            return None

        driver = self._get_driver()
        cypher = (
            "MATCH (e1:Entity)<-[:ABOUT]-(sec:Section)-[:ABOUT]->(e2:Entity) "
            "WHERE e1.subscription_id = $sub AND e1.profile_id = $prof "
            "  AND e2.subscription_id = $sub AND e2.profile_id = $prof "
            "  AND e1.entity_id IS NOT NULL AND e2.entity_id IS NOT NULL "
            "  AND e1.entity_id < e2.entity_id "
            "RETURN e1.entity_id AS a, e2.entity_id AS b, "
            "       count(DISTINCT sec) AS w"
        )
        g = nx.Graph()
        t0 = time.monotonic()
        try:
            with driver.session() as session:
                records = session.run(cypher, sub=str(subscription_id), prof=str(profile_id))
                for r in records:
                    a, b, w = r["a"], r["b"], r["w"]
                    if a and b and a != b:
                        g.add_edge(a, b, weight=float(w or 1.0))
        except Exception as exc:  # noqa: BLE001
            logger.warning("[PPR] graph load failed: %s", exc)
            return None
        load_elapsed = time.monotonic() - t0

        # Also add isolated (unreferenced) entities so seeds that have no
        # co-occurrence still get matched — they just contribute their own
        # PPR mass with no propagation.
        entity_cypher = (
            "MATCH (e:Entity) "
            "WHERE e.subscription_id = $sub AND e.profile_id = $prof "
            "RETURN e.entity_id AS entity_id"
        )
        try:
            with driver.session() as session:
                records = session.run(entity_cypher, sub=str(subscription_id), prof=str(profile_id))
                for r in records:
                    eid = r["entity_id"]
                    if eid and eid not in g:
                        g.add_node(eid)
        except Exception as exc:  # noqa: BLE001
            logger.debug("[PPR] entity enumeration skipped: %s", exc)

        self._graph_cache[key] = (g, now)
        logger.info(
            "[PPR] graph loaded sub=%s prof=%s nodes=%d edges=%d in %.3fs",
            subscription_id, profile_id, g.number_of_nodes(), g.number_of_edges(),
            load_elapsed,
        )
        return g

    def _fetch_chunks_by_entities(
        self,
        *,
        entity_ids: List[str],
        subscription_id: str,
        profile_id: str,
    ) -> List[Dict[str, Any]]:
        """Fetch (chunk_id, document_id, entity_id) triples for the given entities.

        The live KG has no direct ``Chunk-[:MENTIONS]->Entity`` edge — chunks
        connect to entities only through the section layer
        (``Section-[:ABOUT]->Entity`` + ``Section-[:HAS_CHUNK]->Chunk``),
        written by the section-intelligence builder.
        """
        if not entity_ids:
            return []
        driver = self._get_driver()
        cypher = (
            "UNWIND $ids AS eid "
            "MATCH (e:Entity {entity_id: eid}) "
            "WHERE e.subscription_id = $sub AND e.profile_id = $prof "
            "MATCH (e)<-[:ABOUT]-(sec:Section)-[:HAS_CHUNK]->(c:Chunk) "
            "WHERE c.subscription_id = $sub AND c.profile_id = $prof "
            "RETURN DISTINCT c.chunk_id AS chunk_id, c.document_id AS document_id, "
            "       eid AS entity_id"
        )
        try:
            with driver.session() as session:
                records = session.run(
                    cypher, ids=entity_ids, sub=str(subscription_id), prof=str(profile_id),
                )
                return [dict(r) for r in records]
        except Exception as exc:  # noqa: BLE001
            logger.warning("[PPR] chunk fetch failed: %s", exc)
            return []


__all__ = ["PPRRetriever"]
