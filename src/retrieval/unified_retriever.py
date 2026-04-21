"""Three-layer unified retriever — Qdrant + Neo4j + MongoDB.

Phase 2 (SME) adds :meth:`UnifiedRetriever.retrieve_layer_b` — the canonical
helper (ERRATA §7) that returns 1-hop KG direct edges plus the Phase 2
``INFERRED_RELATION`` edges synthesized by
:class:`src.intelligence.sme.builders.kg_materializer.KGMultiHopMaterializer`.
The inferred subset is gated behind ``enable_sme_retrieval`` *and*
``enable_kg_synthesized_edges``; with either flag off the direct 1-hop view
is returned unchanged so Phase 3 routing stays stable.

Phase 3 adds :meth:`UnifiedRetriever.retrieve_four_layers` — the orchestrator
that dispatches Layer A (chunks), Layer B (KG), Layer C (SME), Layer D
(URL placeholder) in parallel via a single :class:`ThreadPoolExecutor`
with ``max_workers=4``. No wall-clock timeout — per spec §3 invariant 8,
DocWain layers never impose internal cutoffs. Layer failures degrade
gracefully into ``RetrievalBundle.degraded_layers``.

Profile isolation is hard — every call includes both ``subscription_id`` and
``profile_id`` and passes them through to the KG driver, which must scope its
MATCH on both properties (see :class:`UnifiedRetrieverKGClient`).
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from src.config.feature_flags import (
    ENABLE_KG_SYNTHESIZED_EDGES,
    ENABLE_SME_RETRIEVAL,
    get_flag_resolver,
)
from src.retrieval.types import RetrievalBundle

logger = logging.getLogger(__name__)


def inferred_edge_confidence_floor() -> float:
    """Return the default confidence floor for ``INFERRED_RELATION`` edges.

    Phase 2 fixes this at 0.6 per plan Task 12; Phase 3 measures whether to
    raise (precision) or lower (recall). Callers of
    :meth:`UnifiedRetriever.retrieve_layer_b` may override via the
    ``inferred_confidence_floor`` kwarg; this helper only supplies the
    default when the caller omits it.
    """
    return 0.6


class UnifiedRetriever:
    """Orchestrates retrieval across three data layers in parallel.

    Layer 1: Qdrant — dense + sparse hybrid search with metadata pre-filtering
    Layer 2: Neo4j — KG entity matching and relationship traversal (1-2 hops)
    Layer 3: MongoDB — document-level metadata and screening summaries
    """

    def __init__(
        self,
        qdrant_client=None,
        neo4j_driver=None,
        mongo_db=None,
        *,
        kg_client: Any = None,
        qdrant: Any = None,
        sme: Any = None,
    ):
        self._qdrant = qdrant_client if qdrant_client is not None else qdrant
        self._neo4j = neo4j_driver
        self._mongo_db = mongo_db
        # Phase 2 Layer B adds a pluggable KG client with ``one_hop`` +
        # ``inferred_relations`` methods. It is kept separate from the legacy
        # ``neo4j_driver`` so the existing three-layer ``retrieve()`` call
        # graph remains untouched. Real-world wiring is expected to pass a
        # :class:`UnifiedRetrieverKGClient` instance; tests pass a MagicMock.
        self._kg_client = kg_client
        self._sme = sme

    # ------------------------------------------------------------------
    # Phase 2 Layer B (ERRATA §7) — KG 1-hop + INFERRED_RELATION edges.
    # ------------------------------------------------------------------
    def retrieve_layer_b(
        self,
        *,
        query: str,
        subscription_id: str,
        profile_id: str,
        top_k: int,
        include_inferred: bool = True,
        inferred_confidence_floor: float | None = None,
        entities: list[str] | None = None,
    ) -> list[dict]:
        """Retrieve Layer B — 1-hop KG direct edges plus synthesized edges.

        ``include_inferred`` + both flags (``enable_sme_retrieval``,
        ``enable_kg_synthesized_edges``) must all be true for the Phase 2
        ``INFERRED_RELATION`` rows to be included. Original KG 1-hop rows are
        always returned (no regression for flag-off callers).

        Profile isolation is enforced by the underlying KG client — callers
        MUST pass a non-empty ``subscription_id`` and ``profile_id``; this
        method validates both and raises ``ValueError`` otherwise so that a
        bug in the caller cannot degrade into a cross-profile read.
        """
        if not subscription_id:
            raise ValueError("subscription_id required for retrieve_layer_b")
        if not profile_id:
            raise ValueError("profile_id required for retrieve_layer_b")
        if self._kg_client is None:
            return []
        hits: list[dict] = []
        for row in self._kg_client.one_hop(
            subscription_id=subscription_id,
            profile_id=profile_id,
            entities=entities,
            top_k=top_k,
        ):
            hits.append({"kind": "kg_direct", **row})
        if not include_inferred:
            return hits
        flags = _safe_flag_resolver()
        if flags is None:
            return hits
        try:
            retrieval_on = flags.is_enabled(
                subscription_id, ENABLE_SME_RETRIEVAL
            )
            edges_on = flags.is_enabled(
                subscription_id, ENABLE_KG_SYNTHESIZED_EDGES
            )
        except KeyError:
            return hits
        if not (retrieval_on and edges_on):
            return hits
        floor = (
            inferred_confidence_floor
            if inferred_confidence_floor is not None
            else inferred_edge_confidence_floor()
        )
        for row in self._kg_client.inferred_relations(
            subscription_id=subscription_id,
            profile_id=profile_id,
            min_confidence=floor,
            top_k=top_k,
        ):
            hits.append({"kind": "kg_inferred", **row})
        return hits

    # ------------------------------------------------------------------
    # Phase 3 — four-layer parallel orchestration (ERRATA §11).
    # ------------------------------------------------------------------
    def retrieve_four_layers(
        self,
        *,
        query: str,
        subscription_id: str,
        profile_id: str,
        query_understanding: dict | None = None,
        top_k_a: int = 10,
        top_k_b: int = 5,
        top_k_c: int = 5,
        layer_a_fn=None,
        layer_b_fn=None,
        layer_c_fn=None,
        layer_d_fn=None,
        gate: Any = None,
        user_requested_compact: bool = False,
    ) -> RetrievalBundle:
        """Parallel-dispatch the four Phase 3 retrieval layers.

        ``layer_a_fn``, ``layer_b_fn``, ``layer_c_fn``, ``layer_d_fn`` are
        optional override callables (zero-arg closures returning
        ``list[dict]``). If omitted, sensible defaults are wired:

        * Layer A: :meth:`retrieve_layer_a` (hybrid when
          :data:`src.config.feature_flags.ENABLE_HYBRID_RETRIEVAL` is on,
          else dense-only).
        * Layer B: :meth:`retrieve_layer_b` (always runs — no master-flag
          gating because direct KG edges are pre-SME infrastructure).
        * Layer C: the injected SME retriever's ``retrieve`` (gated on
          :data:`ENABLE_SME_RETRIEVAL`).
        * Layer D: returns ``[]`` — Phase 5 wires URL fetch.

        Layer C only dispatches when ``enable_sme_retrieval`` is ON for
        the subscription, per the Phase 3 rollout contract. Per ERRATA §11
        any layer that raises is appended once to
        :attr:`RetrievalBundle.degraded_layers` with its full name
        (``"layer_a"``, ``"layer_b"``, ``"layer_c"``, ``"layer_d"``) — no
        short-name append.

        Phase 3 Task 9: intent-aware gating. ``gate`` is an optional
        :class:`src.retrieval.intent_gating.IntentGate` — when provided,
        the orchestrator calls ``gate.decide(intent, user_requested_compact)``
        and skips the submit for any layer whose decision is ``False``.
        Gated layers return empty lists without ever hitting the backing
        store (that's the whole latency win). Callers that don't want
        gating simply omit ``gate`` and every wired layer dispatches.
        """
        if not subscription_id:
            raise ValueError("subscription_id required for retrieve_four_layers")
        if not profile_id:
            raise ValueError("profile_id required for retrieve_four_layers")

        resolver = _safe_flag_resolver()
        layer_c_on = False
        if resolver is not None:
            try:
                layer_c_on = resolver.is_enabled(
                    subscription_id, ENABLE_SME_RETRIEVAL
                )
            except KeyError:
                layer_c_on = False

        qu = query_understanding or {}
        intent = (qu.get("intent") or qu.get("task_type") or "lookup")

        # Intent gate — skip layers the gate decides are not needed.
        if gate is None:
            run_a = True
            run_b = True
            run_c = True
        else:
            decision = gate.decide(
                intent, user_requested_compact=user_requested_compact
            )
            run_a = bool(decision.run_a)
            run_b = bool(decision.run_b)
            run_c = bool(decision.run_c)

        if layer_a_fn is None:
            def layer_a_fn() -> list[dict]:
                return self.retrieve_layer_a(
                    query=query,
                    subscription_id=subscription_id,
                    profile_id=profile_id,
                    query_understanding=qu,
                    top_k=top_k_a,
                )

        if layer_b_fn is None:
            def layer_b_fn() -> list[dict]:
                return self.retrieve_layer_b(
                    query=query,
                    subscription_id=subscription_id,
                    profile_id=profile_id,
                    top_k=top_k_b,
                    entities=(qu.get("entities") or None),
                )

        if layer_c_fn is None:
            def layer_c_fn() -> list[dict]:
                if self._sme is None:
                    return []
                hits = self._sme.retrieve(
                    query=query,
                    subscription_id=subscription_id,
                    profile_id=profile_id,
                    top_k=top_k_c,
                )
                return [_sme_hit_to_dict(h) for h in hits]

        if layer_d_fn is None:
            def layer_d_fn() -> list[dict]:
                # Phase 5 wires URL fetch; Phase 3 placeholder.
                return []

        bundle = RetrievalBundle()

        jobs: Dict[Any, tuple[str, float]] = {}
        # Single ThreadPoolExecutor, max_workers=4, parallel dispatch, no timeout
        # on as_completed per memory rule (no internal wall-clock aborts).
        with ThreadPoolExecutor(max_workers=4, thread_name_prefix="dw-retrieve") as ex:
            if run_a:
                jobs[ex.submit(layer_a_fn)] = ("layer_a", time.perf_counter())
            if run_b:
                jobs[ex.submit(layer_b_fn)] = ("layer_b", time.perf_counter())
            if run_c and layer_c_on:
                jobs[ex.submit(layer_c_fn)] = ("layer_c", time.perf_counter())
            jobs[ex.submit(layer_d_fn)] = ("layer_d", time.perf_counter())

            for fut in as_completed(jobs):
                name, t0 = jobs[fut]
                try:
                    result = fut.result()  # NO timeout — memory rule.
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "%s degraded (%s): %s", name, type(exc).__name__, exc
                    )
                    bundle.degraded_layers.append(name)
                    bundle.per_layer_ms[name] = (
                        time.perf_counter() - t0
                    ) * 1000.0
                    continue
                setattr(
                    bundle,
                    {
                        "layer_a": "layer_a_chunks",
                        "layer_b": "layer_b_kg",
                        "layer_c": "layer_c_sme",
                        "layer_d": "layer_d_url",
                    }[name],
                    list(result or []),
                )
                bundle.per_layer_ms[name] = (time.perf_counter() - t0) * 1000.0

        return bundle

    def retrieve_layer_a(
        self,
        *,
        query: str,
        subscription_id: str,
        profile_id: str,
        query_understanding: dict | None = None,
        top_k: int = 10,
    ) -> list[dict]:
        """Layer A — dense-or-hybrid chunk retrieval from Qdrant.

        Returns a list of ``{"doc_id", "chunk_id", "text", "score", ...}``
        dicts so the merge step (Task 4) sees a uniform shape regardless
        of whether the underlying backend was the legacy dense-only
        Qdrant client or the Phase 1 :class:`HybridSearcher`. The
        hybrid path is flag-gated on :data:`ENABLE_HYBRID_RETRIEVAL`.
        """
        # Reuse the legacy _qdrant_search output and flatten to the Layer A
        # list-of-dicts contract the merge step expects. The flag routing
        # for the hybrid path is driven by the caller — this method
        # accepts either shape via ``self._qdrant``.
        out = self._qdrant_search(
            query,
            profile_id,
            subscription_id,
            query_understanding or {},
            top_k,
        )
        chunks = out.get("chunks", []) if isinstance(out, dict) else []
        normalized: list[dict] = []
        for c in chunks:
            payload = c.get("payload", {}) or {}
            normalized.append(
                {
                    "kind": "chunk",
                    "doc_id": payload.get("document_id") or payload.get("doc_id"),
                    "chunk_id": c.get("id") or payload.get("chunk_id"),
                    "text": c.get("text", "") or payload.get("text", ""),
                    "score": c.get("score"),
                    "raw_score": c.get("score"),
                    "confidence": c.get("score") or 0.5,
                    "payload": payload,
                }
            )
        return normalized

    def retrieve(self, query: str, profile_id: str, subscription_id: str,
                 query_understanding: dict = None, top_k: int = 20) -> dict:
        """Execute three-layer retrieval in parallel.

        Args:
            query: User's search query
            profile_id: Profile scope
            subscription_id: Subscription scope (collection name)
            query_understanding: Dict with intent, entities, suggested_format, sub_queries
            top_k: Number of results to return

        Returns:
            dict with: chunks, kg_context, doc_metadata, merged_context
        """
        query_understanding = query_understanding or {}
        results = {}

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(
                    self._qdrant_search, query, profile_id, subscription_id,
                    query_understanding, top_k
                ): "qdrant",
                executor.submit(
                    self._kg_search, query, profile_id, subscription_id,
                    query_understanding
                ): "neo4j",
                executor.submit(
                    self._metadata_search, profile_id, subscription_id,
                    query_understanding
                ): "mongodb",
            }

            for future in as_completed(futures):
                layer = futures[future]
                try:
                    results[layer] = future.result(timeout=30)
                except Exception as e:
                    logger.error(f"{layer} retrieval failed: {e}")
                    results[layer] = {}

        # Merge and assemble context
        merged = self._assemble_context(results, query, top_k)
        return merged

    def _qdrant_search(self, query: str, profile_id: str,
                       subscription_id: str, query_understanding: dict,
                       top_k: int) -> dict:
        """Layer 1: Qdrant dense + sparse hybrid search with metadata filtering."""
        try:
            qdrant_client = self._get_qdrant_client()
            collection_name = subscription_id

            # Build payload filter
            must_filters = [
                {"key": "profile_id", "match": {"value": profile_id}}
            ]

            # Add metadata pre-filters from query understanding
            if query_understanding.get("domain_tags"):
                must_filters.append({
                    "key": "domain_tags",
                    "match": {"any": query_understanding["domain_tags"]}
                })
            if query_understanding.get("doc_category"):
                must_filters.append({
                    "key": "doc_category",
                    "match": {"value": query_understanding["doc_category"]}
                })

            # Generate query embedding
            query_vector = self._embed_query(query)

            # Dense search
            from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

            filter_conditions = []
            filter_conditions.append(
                FieldCondition(key="profile_id", match=MatchValue(value=profile_id))
            )

            search_filter = Filter(must=filter_conditions)

            dense_results = qdrant_client.search(
                collection_name=collection_name,
                query_vector=("dense", query_vector),
                query_filter=search_filter,
                limit=top_k * 2,
                with_payload=True
            )

            # TODO: Add sparse search and RRF merge
            # For now, return dense results only

            chunks = []
            for hit in dense_results:
                chunks.append({
                    "id": hit.id,
                    "score": hit.score,
                    "text": hit.payload.get("text", ""),
                    "payload": hit.payload
                })

            return {"chunks": chunks, "total_hits": len(chunks)}

        except Exception as e:
            logger.error(f"Qdrant search failed: {e}")
            return {"chunks": [], "total_hits": 0}

    def _kg_search(self, query: str, profile_id: str,
                   subscription_id: str, query_understanding: dict) -> dict:
        """Layer 2: Neo4j KG entity matching and relationship traversal."""
        try:
            driver = self._get_neo4j_driver()
            if not driver:
                return {"entities": [], "relationships": [], "expanded_context": []}

            query_entities = query_understanding.get("entities", [])
            if not query_entities:
                # Try basic entity extraction from query
                # TODO: Use NER or LLM for entity extraction
                return {"entities": [], "relationships": [], "expanded_context": []}

            with driver.session() as session:
                # Match query entities to graph nodes (1-2 hops)
                cypher = """
                MATCH (e:Entity)
                WHERE e.subscription_id = $sub_id
                  AND e.profile_id = $prof_id
                  AND toLower(e.name) IN $entity_names
                OPTIONAL MATCH (e)-[r]->(related:Entity)
                WHERE related.subscription_id = $sub_id
                  AND related.profile_id = $prof_id
                RETURN e.name as entity_name, e.type as entity_type,
                       type(r) as rel_type, r.predicate as predicate,
                       related.name as related_name, related.type as related_type
                LIMIT 50
                """
                result = session.run(cypher, {
                    "sub_id": subscription_id,
                    "prof_id": profile_id,
                    "entity_names": [e.lower() for e in query_entities]
                })

                entities = []
                relationships = []
                for record in result:
                    entities.append({
                        "name": record["entity_name"],
                        "type": record["entity_type"]
                    })
                    if record["related_name"]:
                        relationships.append({
                            "subject": record["entity_name"],
                            "predicate": record["predicate"] or record["rel_type"],
                            "object": record["related_name"],
                            "object_type": record["related_type"]
                        })

                return {
                    "entities": entities,
                    "relationships": relationships,
                    "expanded_context": []  # TODO: pull sections for matched entities
                }

        except Exception as e:
            logger.error(f"Neo4j search failed: {e}")
            return {"entities": [], "relationships": [], "expanded_context": []}

    def _metadata_search(self, profile_id: str, subscription_id: str,
                         query_understanding: dict) -> dict:
        """Layer 3: MongoDB document-level metadata."""
        try:
            db = self._get_mongo_db()
            if not db:
                return {"documents": []}

            # Fetch document metadata for the profile
            docs = list(db["documents"].find(
                {
                    "subscription_id": subscription_id,
                    "profile_id": profile_id,
                    "pipeline_status": "TRAINING_COMPLETED"
                },
                {
                    "document_id": 1, "source_file": 1,
                    "screening.summary": 1, "file_type": 1,
                    "_id": 0
                }
            ).limit(100))

            return {"documents": docs}

        except Exception as e:
            logger.error(f"MongoDB metadata search failed: {e}")
            return {"documents": []}

    def _assemble_context(self, results: dict, query: str, top_k: int) -> dict:
        """Merge results from all three layers into unified context."""
        qdrant_results = results.get("qdrant", {})
        kg_results = results.get("neo4j", {})
        mongo_results = results.get("mongodb", {})

        chunks = qdrant_results.get("chunks", [])

        # TODO: Cross-encoder rerank (BAAI/bge-reranker-base)
        # For now, trust Qdrant's scoring

        # Deduplicate overlapping chunks
        seen_texts = set()
        deduped_chunks = []
        for chunk in chunks:
            text_key = chunk.get("text", "")[:100]
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                deduped_chunks.append(chunk)

        # Trim to top_k
        deduped_chunks = deduped_chunks[:top_k]

        # Build document metadata lookup
        doc_meta = {}
        for doc in mongo_results.get("documents", []):
            doc_meta[doc.get("document_id")] = doc

        # Attach document metadata to chunks
        for chunk in deduped_chunks:
            doc_id = chunk.get("payload", {}).get("document_id")
            if doc_id and doc_id in doc_meta:
                chunk["document_meta"] = doc_meta[doc_id]

        return {
            "chunks": deduped_chunks,
            "kg_context": kg_results,
            "doc_metadata": doc_meta,
            "retrieval_stats": {
                "qdrant_hits": qdrant_results.get("total_hits", 0),
                "kg_entities": len(kg_results.get("entities", [])),
                "kg_relationships": len(kg_results.get("relationships", [])),
                "documents_in_profile": len(mongo_results.get("documents", []))
            }
        }

    def _get_qdrant_client(self):
        """Get or create Qdrant client."""
        if self._qdrant:
            return self._qdrant
        from qdrant_client import QdrantClient
        from src.api.config import Config
        self._qdrant = QdrantClient(url=Config.Qdrant.URL, api_key=Config.Qdrant.API)
        return self._qdrant

    def _get_neo4j_driver(self):
        """Get or create Neo4j driver."""
        if self._neo4j:
            return self._neo4j
        try:
            from neo4j import GraphDatabase
            from src.api.config import Config
            self._neo4j = GraphDatabase.driver(
                Config.Neo4j.URI,
                auth=(Config.Neo4j.USER, Config.Neo4j.PASSWORD)
            )
            return self._neo4j
        except Exception:
            return None

    def _get_mongo_db(self):
        """Get or create MongoDB connection."""
        if self._mongo_db:
            return self._mongo_db
        try:
            from pymongo import MongoClient
            from src.api.config import Config
            client = MongoClient(Config.MongoDB.URI)
            self._mongo_db = client[Config.MongoDB.DB]
            return self._mongo_db
        except Exception:
            return None

    def _embed_query(self, query: str) -> list:
        """Generate embedding for query text."""
        try:
            from src.embedding.model_loader import encode_with_fallback
            return encode_with_fallback(query)
        except Exception:
            # Fallback: try sentence-transformers directly
            from sentence_transformers import SentenceTransformer
            from src.api.config import Config
            model = SentenceTransformer(Config.Model.EMBEDDING_MODEL, device="cpu", local_files_only=True)
            return model.encode(query).tolist()


def _safe_flag_resolver():
    """Return the process-wide flag resolver or ``None`` if uninitialised.

    Phase 2 retrieval paths must not blow up when the flag resolver is not
    yet wired (pre-Phase-2 deploy, or unit tests that skip the init step);
    degrade to "flag off" semantics instead.
    """
    try:
        return get_flag_resolver()
    except RuntimeError:
        return None


def _sme_hit_to_dict(hit: Any) -> dict:
    """Normalise an SMERetrieval hit into the merge step's expected shape.

    :class:`src.retrieval.sme_retrieval.SMERetrieval.retrieve` already
    returns dicts; this helper is a thin pass-through that also handles
    dataclass-style hit objects some tests use. The Layer C items always
    carry ``kind='sme_artifact'`` so the merge step can flag them as
    ``sme_backed=True``.
    """
    if isinstance(hit, dict):
        out = dict(hit)
        out.setdefault("kind", "sme_artifact")
        if "text" not in out:
            out["text"] = out.get("narrative") or ""
        return out
    return {
        "kind": "sme_artifact",
        "text": getattr(hit, "narrative", "")
        or getattr(hit, "text", ""),
        "artifact_type": getattr(hit, "artifact_type", None),
        "confidence": getattr(hit, "confidence", None),
        "evidence": list(getattr(hit, "evidence", []) or []),
        "score": getattr(hit, "score", None),
    }


class UnifiedRetrieverKGClient:
    """Neo4j-backed KG client for :class:`UnifiedRetriever` Layer B.

    Exposes exactly the two methods :meth:`retrieve_layer_b` depends on:
    ``one_hop`` returns direct 1-hop neighbour edges from query entities (or
    from the full profile-scoped graph slice when ``entities`` is ``None``),
    and ``inferred_relations`` returns the Phase 2 ``INFERRED_RELATION``
    subset filtered by ``source='sme_synthesis'`` and a confidence floor.

    Every read enforces the hard ``(subscription_id, profile_id)`` tuple in
    the MATCH clause so the Phase 3 retrieval layer can never surface edges
    from another profile even if the caller is buggy. Cypher literals are
    parameter-bound — no string interpolation.
    """

    def __init__(self, driver: Any = None) -> None:
        self._driver = driver

    def one_hop(
        self,
        *,
        subscription_id: str,
        profile_id: str,
        entities: list[str] | None,
        top_k: int,
    ) -> list[dict]:
        driver = self._driver
        if driver is None:
            return []
        entity_names = [e.lower() for e in entities] if entities else []
        cypher = (
            "MATCH (a:Entity)-[r]->(b:Entity) "
            "WHERE a.subscription_id = $sub AND a.profile_id = $prof "
            "  AND b.subscription_id = $sub AND b.profile_id = $prof "
            "  AND ($entities IS NULL OR size($entities) = 0 OR toLower(a.name) IN $entities) "
            "  AND type(r) <> 'INFERRED_RELATION' "
            "RETURN a.node_id AS src, b.node_id AS dst, "
            "       type(r) AS type, "
            "       coalesce(r.evidence, []) AS evidence "
            "LIMIT $top_k"
        )
        try:
            with driver.session() as session:
                result = session.run(
                    cypher,
                    {
                        "sub": subscription_id,
                        "prof": profile_id,
                        "entities": entity_names or None,
                        "top_k": int(top_k),
                    },
                )
                return [
                    {
                        "src": rec["src"],
                        "dst": rec["dst"],
                        "type": rec["type"],
                        "evidence": list(rec["evidence"] or []),
                    }
                    for rec in result
                ]
        except Exception:
            logger.exception("one_hop KG read failed")
            return []

    def inferred_relations(
        self,
        *,
        subscription_id: str,
        profile_id: str,
        min_confidence: float,
        top_k: int,
    ) -> list[dict]:
        driver = self._driver
        if driver is None:
            return []
        cypher = (
            "MATCH (a)-[r:INFERRED_RELATION]->(b) "
            "WHERE r.source = 'sme_synthesis' "
            "  AND r.subscription_id = $sub AND r.profile_id = $prof "
            "  AND a.subscription_id = $sub AND a.profile_id = $prof "
            "  AND b.subscription_id = $sub AND b.profile_id = $prof "
            "  AND r.confidence >= $min_confidence "
            "RETURN a.node_id AS src, b.node_id AS dst, "
            "       r.relation_type AS relation_type, "
            "       r.confidence AS confidence, "
            "       r.evidence AS evidence, "
            "       r.inference_path AS inference_path "
            "ORDER BY r.confidence DESC LIMIT $top_k"
        )
        try:
            with driver.session() as session:
                result = session.run(
                    cypher,
                    {
                        "sub": subscription_id,
                        "prof": profile_id,
                        "min_confidence": float(min_confidence),
                        "top_k": int(top_k),
                    },
                )
                return [
                    {
                        "src": rec["src"],
                        "dst": rec["dst"],
                        "relation_type": rec["relation_type"],
                        "confidence": rec["confidence"],
                        "evidence": list(rec["evidence"] or []),
                        "inference_path": list(rec["inference_path"] or []),
                    }
                    for rec in result
                ]
        except Exception:
            logger.exception("inferred_relations KG read failed")
            return []
