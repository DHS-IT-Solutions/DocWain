"""Phase 2 — Deterministic plan execution with parallel step support."""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from src.query.planner import PlanStep, QueryPlan

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    """Result of executing a single plan step."""
    step_id: str
    action: str
    data: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class ExecutionResult:
    """Aggregated results of executing all plan steps."""
    step_results: Dict[str, StepResult] = field(default_factory=dict)
    success: bool = True
    duration_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Clients container (thin wrapper to unify access)
# ---------------------------------------------------------------------------

@dataclass
class ExecutionClients:
    """Holds references to backend clients needed by action handlers."""
    qdrant_client: Any = None
    neo4j_driver: Any = None
    mongo_db: Any = None
    embedder: Any = None
    profile_id: str = ""
    subscription_id: str = ""


# ---------------------------------------------------------------------------
# PlanExecutor
# ---------------------------------------------------------------------------

class PlanExecutor:
    """Execute a QueryPlan deterministically, parallelising independent steps."""

    def __init__(self, max_workers: int = 4):
        self._max_workers = max_workers

    def execute(self, plan: QueryPlan, clients: ExecutionClients) -> ExecutionResult:
        """Run all plan steps, respecting dependency ordering.

        Independent steps (no pending dependencies) are submitted in parallel.
        Steps whose action is ``generate`` are skipped here (handled in Phase 3).

        Args:
            plan: The QueryPlan from Phase 1.
            clients: Backend client references.

        Returns:
            ExecutionResult with per-step results.
        """
        t0 = time.monotonic()
        completed: Dict[str, StepResult] = {}
        pending = {s.id: s for s in plan.steps if s.action != "generate"}
        all_success = True

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            while pending:
                # Find steps whose dependencies are all satisfied
                ready = [
                    s for s in pending.values()
                    if all(dep in completed for dep in s.depends_on)
                ]
                if not ready:
                    # Circular or unsatisfiable dependency — break to avoid hang
                    logger.error(
                        "No executable steps remain; possible circular dependency. "
                        "Pending: %s",
                        list(pending.keys()),
                    )
                    for sid, step in pending.items():
                        completed[sid] = StepResult(
                            step_id=sid,
                            action=step.action,
                            error="Unresolvable dependency",
                        )
                    all_success = False
                    break

                futures = {
                    pool.submit(
                        self._execute_step, step, completed, clients
                    ): step
                    for step in ready
                }

                for future in as_completed(futures):
                    step = futures[future]
                    try:
                        result = future.result(timeout=60)
                    except Exception as exc:
                        logger.error("Step %s raised: %s", step.id, exc, exc_info=True)
                        result = StepResult(
                            step_id=step.id,
                            action=step.action,
                            error=str(exc),
                        )
                        all_success = False
                    completed[step.id] = result
                    pending.pop(step.id, None)

        duration = time.monotonic() - t0
        return ExecutionResult(
            step_results=completed,
            success=all_success,
            duration_seconds=round(duration, 3),
        )

    # ------------------------------------------------------------------
    # Step dispatcher
    # ------------------------------------------------------------------

    def _execute_step(
        self,
        step: PlanStep,
        completed: Dict[str, StepResult],
        clients: ExecutionClients,
    ) -> StepResult:
        """Route a step to the correct handler."""
        handler = self._get_handler(step.action)
        try:
            data = handler(step, completed, clients)
            return StepResult(step_id=step.id, action=step.action, data=data)
        except Exception as exc:
            logger.warning("Handler for %s/%s failed: %s", step.action, step.id, exc)
            return StepResult(step_id=step.id, action=step.action, error=str(exc))

    def _get_handler(self, action: str) -> Callable:
        dispatch: Dict[str, Callable] = {
            "search": self._handle_search,
            "knowledge_search": self._handle_knowledge_search,
            "kg_lookup": self._handle_kg_lookup,
            "kg_search": self._handle_kg_lookup,  # same handler, broader params
            "cross_reference": self._handle_cross_reference,
            "spreadsheet_query": self._handle_spreadsheet_query,
        }
        return dispatch.get(action, self._handle_unknown)

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _handle_search(
        self,
        step: PlanStep,
        completed: Dict[str, StepResult],
        clients: ExecutionClients,
    ) -> List[Dict[str, Any]]:
        """Qdrant vector search on the user document collection."""
        qdrant = clients.qdrant_client
        if qdrant is None:
            raise RuntimeError("Qdrant client not available")

        query_vector = self._embed(step.query, clients)
        collection = step.collection or clients.subscription_id

        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="profile_id",
                        match=MatchValue(value=clients.profile_id),
                    )
                ]
            )
            hits = qdrant.search(
                collection_name=collection,
                query_vector=("dense", query_vector),
                query_filter=search_filter,
                limit=step.top_k,
                with_payload=True,
            )
        except Exception as exc:
            logger.warning("Qdrant search failed for step %s: %s", step.id, exc)
            return []

        results: List[Dict[str, Any]] = []
        for hit in hits:
            payload = hit.payload or {}
            results.append({
                "content": payload.get("text", ""),
                "source": payload.get("source_file", payload.get("file_name", "")),
                "relevance_score": round(float(hit.score), 4),
                "page": payload.get("page"),
                "document_id": payload.get("document_id", ""),
                "section": payload.get("section_title", ""),
                "chunk_id": hit.id,
            })
        return results

    def _handle_knowledge_search(
        self,
        step: PlanStep,
        completed: Dict[str, StepResult],
        clients: ExecutionClients,
    ) -> List[Dict[str, Any]]:
        """Qdrant vector search on a knowledge pack collection."""
        qdrant = clients.qdrant_client
        if qdrant is None:
            raise RuntimeError("Qdrant client not available")

        query_vector = self._embed(step.query, clients)
        collection = step.collection or "knowledge_packs"

        try:
            hits = qdrant.search(
                collection_name=collection,
                query_vector=("dense", query_vector),
                limit=step.top_k,
                with_payload=True,
            )
        except Exception as exc:
            logger.warning("Knowledge search failed for step %s: %s", step.id, exc)
            return []

        results: List[Dict[str, Any]] = []
        for hit in hits:
            payload = hit.payload or {}
            results.append({
                "content": payload.get("text", ""),
                "source": payload.get("source", payload.get("pack_name", "knowledge_pack")),
                "relevance_score": round(float(hit.score), 4),
                "citation": payload.get("citation", ""),
                "domain": payload.get("domain", ""),
            })
        return results

    def _handle_kg_lookup(
        self,
        step: PlanStep,
        completed: Dict[str, StepResult],
        clients: ExecutionClients,
    ) -> List[Dict[str, Any]]:
        """Neo4j Cypher query for entity and relationship lookup."""
        driver = clients.neo4j_driver
        if driver is None:
            logger.info("Neo4j driver not available for step %s; skipping", step.id)
            return []

        # Extract entity names from step query or params
        entity_names = step.params.get("entities", [])
        if not entity_names:
            # Simple heuristic: use capitalised words from the query
            entity_names = [
                w for w in step.query.split()
                if w and w[0].isupper() and len(w) > 1
            ]

        if not entity_names:
            return []

        cypher = """
        MATCH (e:Entity)
        WHERE e.subscription_id = $sub_id
          AND e.profile_id = $prof_id
          AND toLower(e.name) IN $entity_names
        OPTIONAL MATCH (e)-[r]->(related:Entity)
        WHERE related.subscription_id = $sub_id
          AND related.profile_id = $prof_id
        RETURN e.name AS entity_name, e.type AS entity_type,
               type(r) AS rel_type, r.predicate AS predicate,
               related.name AS related_name, related.type AS related_type
        LIMIT 50
        """

        results: List[Dict[str, Any]] = []
        try:
            with driver.session() as session:
                records = session.run(cypher, {
                    "sub_id": clients.subscription_id,
                    "prof_id": clients.profile_id,
                    "entity_names": [n.lower() for n in entity_names],
                })
                for rec in records:
                    entry: Dict[str, Any] = {
                        "content": f"{rec['entity_name']} ({rec['entity_type'] or 'Entity'})",
                        "source": "knowledge_graph",
                        "relevance_score": 1.0,
                        "entity_name": rec["entity_name"],
                        "entity_type": rec["entity_type"],
                    }
                    if rec["related_name"]:
                        entry["relationship"] = {
                            "predicate": rec["predicate"] or rec["rel_type"],
                            "target": rec["related_name"],
                            "target_type": rec["related_type"],
                        }
                    results.append(entry)
        except Exception as exc:
            logger.warning("KG lookup failed for step %s: %s", step.id, exc)

        return results

    def _handle_cross_reference(
        self,
        step: PlanStep,
        completed: Dict[str, StepResult],
        clients: ExecutionClients,
    ) -> List[Dict[str, Any]]:
        """Compare results from two dependent steps for agreements/conflicts."""
        dep_ids = step.depends_on
        if len(dep_ids) < 2:
            return [{"content": "Insufficient dependencies for cross-reference", "source": "system", "relevance_score": 0.0}]

        set_a = completed.get(dep_ids[0])
        set_b = completed.get(dep_ids[1])
        if set_a is None or set_b is None:
            return [{"content": "Dependency results not available", "source": "system", "relevance_score": 0.0}]

        # Build text sets for simple overlap analysis
        texts_a = {d.get("content", "").strip().lower() for d in set_a.data if d.get("content")}
        texts_b = {d.get("content", "").strip().lower() for d in set_b.data if d.get("content")}

        # Find overlapping key phrases (sentence-level)
        agreements: List[str] = []
        conflicts: List[str] = []

        sources_a = {d.get("source", "") for d in set_a.data}
        sources_b = {d.get("source", "") for d in set_b.data}

        # Simple heuristic: if same content appears in both, it agrees
        overlap = texts_a & texts_b
        if overlap:
            for text in list(overlap)[:5]:
                agreements.append(text[:200])

        # If sources differ but texts don't overlap, flag potential conflict
        if not overlap and texts_a and texts_b:
            conflicts.append(
                f"No overlapping evidence between {dep_ids[0]} ({len(texts_a)} items) "
                f"and {dep_ids[1]} ({len(texts_b)} items)"
            )

        results: List[Dict[str, Any]] = []
        if agreements:
            results.append({
                "content": "Agreements found: " + "; ".join(agreements),
                "source": "cross_reference",
                "relevance_score": 0.9,
                "type": "agreement",
                "count": len(agreements),
            })
        if conflicts:
            results.append({
                "content": "Potential conflicts: " + "; ".join(conflicts),
                "source": "cross_reference",
                "relevance_score": 0.7,
                "type": "conflict",
                "count": len(conflicts),
            })
        if not results:
            results.append({
                "content": "Cross-reference completed; no clear agreements or conflicts detected.",
                "source": "cross_reference",
                "relevance_score": 0.5,
                "type": "neutral",
            })
        return results

    def _handle_spreadsheet_query(
        self,
        step: PlanStep,
        completed: Dict[str, StepResult],
        clients: ExecutionClients,
    ) -> List[Dict[str, Any]]:
        """Lookup structured data from MongoDB computed profiles."""
        db = clients.mongo_db
        if db is None:
            logger.info("MongoDB not available for step %s; skipping", step.id)
            return []

        try:
            query_filter: Dict[str, Any] = {
                "subscription_id": clients.subscription_id,
                "profile_id": clients.profile_id,
            }
            # Allow params to narrow the search
            if step.params.get("document_id"):
                query_filter["document_id"] = step.params["document_id"]

            docs = list(
                db["computed_profiles"].find(
                    query_filter,
                    {"_id": 0},
                ).limit(step.top_k)
            )

            results: List[Dict[str, Any]] = []
            for doc in docs:
                results.append({
                    "content": json.dumps(doc, default=str),
                    "source": doc.get("source_file", "spreadsheet"),
                    "relevance_score": 1.0,
                    "type": "spreadsheet",
                })
            return results
        except Exception as exc:
            logger.warning("Spreadsheet query failed for step %s: %s", step.id, exc)
            return []

    def _handle_unknown(
        self,
        step: PlanStep,
        completed: Dict[str, StepResult],
        clients: ExecutionClients,
    ) -> List[Dict[str, Any]]:
        logger.warning("Unknown action '%s' in step %s", step.action, step.id)
        return []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _embed(text: str, clients: ExecutionClients) -> List[float]:
        """Generate an embedding vector for the given text."""
        if clients.embedder is not None:
            if callable(clients.embedder):
                return clients.embedder(text)
            # Assume sentence-transformer-style .encode()
            vec = clients.embedder.encode(text)
            if hasattr(vec, "tolist"):
                return vec.tolist()
            return list(vec)

        # Fallback: try the project's embedding utility
        try:
            from src.embedding.model_loader import encode_with_fallback
            return encode_with_fallback(text)
        except Exception as exc:
            raise RuntimeError(f"No embedder available: {exc}") from exc


__all__ = ["StepResult", "ExecutionResult", "ExecutionClients", "PlanExecutor"]
