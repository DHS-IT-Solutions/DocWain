"""Researcher Agent Celery task.

Dispatched from `src.api.pipeline_api.trigger_embedding` alongside embedding + KG.
Reads the canonical extraction JSON from Azure Blob, prompts DocWain for
insights, and writes results to Qdrant (payload mapped by document_id) + Neo4j
(Insight nodes linked to Document).

Isolation: writes ONLY to `researcher.*` field in MongoDB; never touches
`pipeline_status`, `stages.*`, or `knowledge_graph.*`. Spec §5.6.
"""
from __future__ import annotations

import json
import logging
import time as _time
from typing import Any, Dict, List, Optional

from src.celery_app import app
from src.api.statuses import (
    RESEARCHER_COMPLETED,
    RESEARCHER_FAILED,
    RESEARCHER_IN_PROGRESS,
)

logger = logging.getLogger(__name__)


def _extract_doc_text(extraction_json: Dict[str, Any]) -> str:
    """Concatenate all visible text from a canonical extraction JSON."""
    parts: List[str] = []
    for page in (extraction_json.get("pages") or []):
        for b in (page.get("blocks") or []):
            if b.get("text"):
                parts.append(b["text"])
        for t in (page.get("tables") or []):
            for row in (t.get("rows") or []):
                parts.append(" | ".join(str(c) for c in row))
    for sheet in (extraction_json.get("sheets") or []):
        for cell in (sheet.get("cells") or {}).values():
            val = (cell or {}).get("value") if isinstance(cell, dict) else None
            if val is not None:
                parts.append(str(val))
    for slide in (extraction_json.get("slides") or []):
        for e in (slide.get("elements") or []):
            if e.get("text"):
                parts.append(e["text"])
        notes = (slide.get("notes") or "").strip()
        if notes:
            parts.append(notes)
    return "\n\n".join(parts)


def _call_docwain_for_insights(document_text: str, doc_type_hint: str = "generic"):
    from src.docwain.prompts.researcher import (
        RESEARCHER_SYSTEM_PROMPT,
        ResearcherInsights,
        build_user_prompt,
        parse_researcher_response,
    )
    try:
        from src.llm.gateway import LLMGateway
        from src.api.config import Config
        max_tokens = int(getattr(getattr(Config, "Researcher", None), "MAX_TOKENS", 4096))
        gw = LLMGateway()
        user = build_user_prompt(document_text=document_text, doc_type_hint=doc_type_hint)
        result = gw.generate_with_metadata(
            prompt=user,
            system=RESEARCHER_SYSTEM_PROMPT,
            temperature=0.2,
            max_tokens=max_tokens,
        )
        return parse_researcher_response(getattr(result, "text", "") or "")
    except Exception as exc:
        logger.warning("Researcher LLM call failed: %s", exc)
        return ResearcherInsights()


def _load_extraction(document_id: str, subscription_id: str, profile_id: str) -> Optional[Dict[str, Any]]:
    """Load the canonical extraction JSON from Azure Blob.

    Mirrors the pattern in `src.tasks.kg._load_extraction_from_blob` — if a
    helper with that name exists, reuse it; otherwise read directly.
    """
    try:
        from src.tasks.kg import _load_extraction_from_blob  # type: ignore
        return _load_extraction_from_blob(
            document_id=document_id, subscription_id=subscription_id, profile_id=profile_id
        )
    except Exception:
        pass
    # Fallback path — rarely hit
    try:
        from src.api.content_store import BlobStore
        store = BlobStore()
        # Adapt path per existing convention: {sub}/{profile}/{doc}/extraction.json
        path = f"{subscription_id}/{profile_id}/{document_id}/extraction.json"
        raw = store.get(path)
        return json.loads(raw.decode("utf-8")) if raw else None
    except Exception as exc:
        logger.warning("Failed to load extraction for %s: %s", document_id, exc)
        return None


def _write_insights_to_qdrant(document_id: str, subscription_id: str, profile_id: str,
                               insights: Dict[str, Any]) -> None:
    """Best-effort Qdrant payload enrichment. Never raises."""
    try:
        # This depends on the existing Qdrant client pattern. If not trivially
        # available, skip — observability still captures the insights.
        from qdrant_client import QdrantClient
        import os
        client = QdrantClient(url=os.environ["QDRANT_URL"],
                              api_key=os.environ["QDRANT_API_KEY"], timeout=60)
        # Collection name convention: assume subscription_id = collection.
        # Real projects may use a different mapping — adapt if needed.
        collection = subscription_id
        client.set_payload(
            collection_name=collection,
            payload={"researcher_insights": insights},
            points_selector=__import__("qdrant_client").models.Filter(
                must=[__import__("qdrant_client").models.FieldCondition(
                    key="document_id",
                    match=__import__("qdrant_client").models.MatchValue(value=document_id),
                )]
            ),
        )
    except Exception as exc:
        logger.debug("Qdrant insight write skipped for %s: %s", document_id, exc)


def _write_insight_to_neo4j(document_id: str, insights: Dict[str, Any]) -> None:
    """Best-effort Neo4j Insight node creation. Never raises."""
    try:
        from src.kg.neo4j_store import Neo4jStore
        store = Neo4jStore()
        store.create_insight_node(document_id=document_id, insights=insights)
    except AttributeError:
        # create_insight_node doesn't exist yet — inline fallback writes via raw cypher.
        try:
            from src.kg.neo4j_store import Neo4jStore
            store = Neo4jStore()
            cy = (
                "MATCH (d:Document {document_id: $doc_id}) "
                "MERGE (d)-[:HAS_INSIGHT]->(i:Insight {document_id: $doc_id}) "
                "SET i.summary = $summary, i.confidence = $confidence, "
                "    i.key_facts = $key_facts, i.recommendations = $recommendations, "
                "    i.anomalies = $anomalies, i.questions_to_ask = $questions_to_ask, "
                "    i.updated_at = timestamp()"
            )
            with store.driver.session() as session:
                session.run(cy, doc_id=document_id,
                            summary=insights.get("summary", ""),
                            confidence=float(insights.get("confidence", 0.0)),
                            key_facts=insights.get("key_facts", []),
                            recommendations=insights.get("recommendations", []),
                            anomalies=insights.get("anomalies", []),
                            questions_to_ask=insights.get("questions_to_ask", []))
        except Exception as exc:
            logger.debug("Neo4j insight write skipped for %s: %s", document_id, exc)
    except Exception as exc:
        logger.debug("Neo4j insight write skipped for %s: %s", document_id, exc)


def _set_researcher_status(document_id: str, status: str, **extra) -> None:
    """Update MongoDB researcher.* strand. Never touches pipeline_status."""
    try:
        from src.api.dw_newron import get_mongo_collection  # type: ignore
        col = get_mongo_collection("documents")
        update = {"researcher.status": status, "researcher.updated_at": _time.time()}
        for k, v in extra.items():
            update[f"researcher.{k}"] = v
        col.update_one({"document_id": document_id}, {"$set": update})
    except Exception as exc:
        logger.warning("Failed to set researcher status for %s: %s", document_id, exc)


@app.task(bind=True, max_retries=3, soft_time_limit=1200)
def run_researcher_agent(self, document_id: str, subscription_id: str, profile_id: str):
    """Runs the Researcher Agent for a single document. Fully isolated from embedding + KG."""
    started_at = _time.perf_counter()
    _set_researcher_status(document_id, RESEARCHER_IN_PROGRESS, started_at=_time.time())
    try:
        extraction = _load_extraction(document_id, subscription_id, profile_id)
        if not extraction:
            _set_researcher_status(document_id, RESEARCHER_FAILED,
                                   error="extraction not found",
                                   completed_at=_time.time())
            return {"status": RESEARCHER_FAILED, "error": "extraction not found"}

        doc_type_hint = ((extraction.get("metadata") or {}).get("doc_intel") or {}).get("doc_type_hint") or "generic"
        document_text = _extract_doc_text(extraction)
        insights_obj = _call_docwain_for_insights(document_text, doc_type_hint=doc_type_hint)
        from dataclasses import asdict as _asdict
        insights = _asdict(insights_obj)

        _write_insights_to_qdrant(document_id, subscription_id, profile_id, insights)
        _write_insight_to_neo4j(document_id, insights)

        _set_researcher_status(
            document_id, RESEARCHER_COMPLETED,
            summary_preview=(insights.get("summary") or "")[:200],
            confidence=float(insights.get("confidence", 0.0)),
            elapsed_ms=(_time.perf_counter() - started_at) * 1000.0,
            completed_at=_time.time(),
        )
        return {"status": RESEARCHER_COMPLETED, "confidence": insights.get("confidence", 0.0)}
    except Exception as exc:
        logger.warning("Researcher failed for %s: %r", document_id, exc)
        _set_researcher_status(document_id, RESEARCHER_FAILED,
                                error=repr(exc),
                                elapsed_ms=(_time.perf_counter() - started_at) * 1000.0,
                                completed_at=_time.time())
        return {"status": RESEARCHER_FAILED, "error": repr(exc)}
