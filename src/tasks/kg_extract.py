"""LLM-based knowledge extraction — Celery task version.

Moved off the extraction critical path in 2026-04-24: the inline version
in ``extraction_service._extract_from_connector`` cost 50-150s/doc at the
user's progress bar, because each section requires an LLM round trip
(entity/fact/relationship extraction + verification). Running it async
lets extraction return to the UI in ~40-80s while KG enrichment
completes in the background before screening is triggered.

Output lands in two places so downstream consumers don't need changes:

  1. Redis hot cache (via ``cache_document_knowledge``) — primary read
     path for retrieval / agent / intelligence layers.
  2. MongoDB ``knowledge_extraction`` sub-document — durability for the
     cases where Redis has been flushed or is unavailable. Also exposes
     counts through the pipeline-status endpoint.

The legacy pickle (``raw.content.kg_entities``) is NOT rewritten —
nothing actually reads it (verified before the move; only the inline
block wrote there). If that changes, add a re-pickle step here.
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from celery.exceptions import SoftTimeLimitExceeded

from src.celery_app import app
from src.api.content_store import load_extracted_pickle
from src.api.document_status import append_audit_log, update_document_fields


logger = logging.getLogger(__name__)


def _primary_text(content: Any) -> str:
    if isinstance(content, dict):
        return content.get("translated_text") or content.get("full_text") or ""
    if isinstance(content, str):
        return content
    return ""


def _sections_from_content(content: Any, full_text: str) -> List[Dict[str, Any]]:
    """Produce the list of sections the knowledge extractor runs on.

    Prefers already-chunked sections from extraction; falls back to 2 KB
    chunks to match the inline behaviour byte-for-byte.
    """
    sections: List[Dict[str, Any]] = []
    if isinstance(content, dict):
        for sec in content.get("sections") or []:
            if isinstance(sec, dict) and sec.get("text"):
                sections.append(sec)
    if sections:
        return sections

    chunk_size = 2000
    for i in range(0, len(full_text), chunk_size):
        sections.append({
            "text": full_text[i:i + chunk_size],
            "page": (i // chunk_size) + 1,
            "section_title": f"Section {(i // chunk_size) + 1}",
        })
    return sections


def _run_extraction_for_doc(
    document_id: str,
    subscription_id: Optional[str],
    profile_id: Optional[str],
) -> Dict[str, Any]:
    """Core work — also unit-testable without Celery."""
    from src.intelligence.knowledge_extractor import get_knowledge_extractor
    from src.intelligence.evidence_verifier import verify_knowledge_result
    from src.intelligence.hot_cache import cache_document_knowledge, recompute_profile_domain

    extractor = get_knowledge_extractor()
    payload = load_extracted_pickle(document_id)
    raw = (payload or {}).get("raw") if isinstance(payload, dict) else None
    if not raw:
        raise ValueError(f"pickle for {document_id} has no 'raw' section")

    aggregated: Dict[str, Any] = {
        "entities": [], "facts": [], "relationships": [], "claims": [],
        "summary": {}, "domain": "general",
    }

    for _fname, content in raw.items():
        full_text = _primary_text(content)
        if not full_text or len(full_text.strip()) < 100:
            continue

        sections = _sections_from_content(content, full_text)[:15]
        max_workers = min(len(sections), int(os.getenv("KG_EXTRACTION_MAX_WORKERS", "4")))

        def _extract_and_verify(sec: Dict[str, Any]):
            result = extractor.extract_section(
                text=sec.get("text", ""),
                page=sec.get("start_page", sec.get("page", 1)),
                section=sec.get("title", sec.get("section_title", "unknown")),
            )
            return verify_knowledge_result(result, sec.get("text", ""))

        section_results = []
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futs = {pool.submit(_extract_and_verify, s): s for s in sections}
            for fut in as_completed(futs):
                sec = futs[fut]
                try:
                    section_results.append(fut.result())
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "KG section failed for %s section=%s: %s",
                        document_id,
                        sec.get("title", sec.get("section_title", "unknown")),
                        exc,
                    )

        for result in section_results:
            for ent in result.entities:
                aggregated["entities"].append({
                    "name": ent.name, "type": ent.type,
                    "context": ent.context, "evidence": ent.evidence,
                    "confidence": ent.confidence, "location": ent.location,
                })
            for fact in result.facts:
                aggregated["facts"].append({
                    "statement": fact.statement, "evidence": fact.evidence,
                    "confidence": fact.confidence, "location": fact.location,
                })
            for rel in result.relationships:
                aggregated["relationships"].append({
                    "subject": rel.subject, "object": rel.object,
                    "relation": rel.relation, "evidence": rel.evidence,
                    "confidence": rel.confidence,
                })
            for claim in result.claims:
                aggregated["claims"].append({
                    "claim": claim.claim, "evidence": claim.evidence,
                    "confidence": claim.confidence,
                })

        summary = extractor.generate_document_summary(full_text)
        aggregated["summary"] = summary
        aggregated["domain"] = summary.get("domain", "general")

        try:
            from src.api.rag_state import get_app_state
            app_state = get_app_state()
            redis_client = getattr(app_state, "redis_client", None) if app_state else None
            if redis_client:
                cache_document_knowledge(
                    redis_client=redis_client,
                    profile_id=profile_id,
                    doc_id=document_id,
                    entities=aggregated["entities"],
                    facts=aggregated["facts"],
                    claims=aggregated["claims"],
                    relationships=aggregated["relationships"],
                    domain=summary.get("domain", "general"),
                    summary=summary.get("summary", ""),
                )
                recompute_profile_domain(redis_client, profile_id)
        except Exception as cache_err:  # noqa: BLE001
            logger.debug("Redis hot-cache write failed: %s", cache_err)

        # Only process the first doc in the masked_docs dict — matches the
        # inline behaviour (the loop has an implicit `break` after first).
        break

    return aggregated


@app.task(
    bind=True,
    name="src.tasks.kg_extract.run_knowledge_extraction",
    max_retries=1,
    soft_time_limit=1200,
)
def run_knowledge_extraction(self, document_id: str, subscription_id: str, profile_id: str):
    """Async version of the inline KG extraction block.

    Writes to Redis hot cache and Mongo ``knowledge_extraction`` field;
    never mutates extraction stage state. Safe to retry once on failure.
    """
    start = time.time()
    try:
        update_document_fields(
            document_id,
            {"knowledge_extraction.status": "IN_PROGRESS",
             "knowledge_extraction.started_at": start},
        )
        append_audit_log(document_id, "KG_EXTRACTION_STARTED",
                         celery_task_id=self.request.id)

        result = _run_extraction_for_doc(document_id, subscription_id, profile_id)
        duration = round(time.time() - start, 2)

        update_document_fields(document_id, {
            "knowledge_extraction.status": "COMPLETED",
            "knowledge_extraction.completed_at": time.time(),
            "knowledge_extraction.duration_seconds": duration,
            "knowledge_extraction.entities_count": len(result["entities"]),
            "knowledge_extraction.facts_count": len(result["facts"]),
            "knowledge_extraction.relationships_count": len(result["relationships"]),
            "knowledge_extraction.claims_count": len(result["claims"]),
            "knowledge_extraction.domain": result.get("domain", "general"),
        })
        append_audit_log(
            document_id, "KG_EXTRACTION_COMPLETED",
            entities=len(result["entities"]),
            facts=len(result["facts"]),
            relationships=len(result["relationships"]),
            claims=len(result["claims"]),
            duration_seconds=duration,
        )
        logger.info(
            "KG extraction async: doc=%s entities=%d facts=%d rels=%d claims=%d in %.1fs",
            document_id,
            len(result["entities"]), len(result["facts"]),
            len(result["relationships"]), len(result["claims"]),
            duration,
        )

    except SoftTimeLimitExceeded:
        update_document_fields(document_id, {
            "knowledge_extraction.status": "FAILED",
            "knowledge_extraction.completed_at": time.time(),
            "knowledge_extraction.error": "timeout",
        })
        append_audit_log(document_id, "KG_EXTRACTION_FAILED", error="timeout")
        logger.warning("KG extraction timed out for %s", document_id)

    except Exception as exc:  # noqa: BLE001
        update_document_fields(document_id, {
            "knowledge_extraction.status": "FAILED",
            "knowledge_extraction.completed_at": time.time(),
            "knowledge_extraction.error": str(exc)[:500],
        })
        append_audit_log(document_id, "KG_EXTRACTION_FAILED", error=str(exc)[:500])
        logger.warning("KG extraction failed for %s: %s", document_id, exc)
        try:
            self.retry(exc=exc, countdown=30)
        except Exception:
            pass
