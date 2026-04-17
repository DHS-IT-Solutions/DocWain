"""Single source of truth for async KG-ingest triggers.

Every extraction path — the canonical Celery task, the sync upload
fallback, and the legacy connector/embedding paths — should enqueue KG
updates through this module. Centralising the trigger prevents
double-ingest if multiple paths run on the same document, and it keeps
the "update KG in the background, don't block extraction" policy in one
place.

Two entry points, covering the two input shapes callers actually have:

- ``enqueue_from_extraction_result`` — new unified ExtractionEngine
  output. Used by ``src/tasks/extraction.py`` (the live Celery path).

- ``enqueue_from_legacy_payload`` — older ``payload_to_save`` dict plus
  ``DeepAnalysisResult``. Used by ``src/api/extraction_service.py`` and
  ``src/api/dataHandler.py``.

Both return immediately (daemon thread). KG failure is logged at ERROR
level and never propagates back to the caller — extraction completes
irrespective of KG state, per the canonical pipeline.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal — shared worker that does the actual enqueue
# ---------------------------------------------------------------------------


def _run_kg_ingest(
    *,
    document_id: str,
    subscription_id: str,
    profile_id: str,
    doc_name: str,
    texts: List[str],
    chunk_metadata: List[Dict[str, Any]],
    doc_metadata: Dict[str, Any],
    deep_entities: Optional[List[Dict[str, Any]]] = None,
    typed_relationships: Optional[List[Dict[str, Any]]] = None,
    temporal_spans: Optional[List[Dict[str, Any]]] = None,
    redis_client: Any = None,
) -> None:
    """Build a graph payload from the inputs and enqueue it. Never raises."""
    try:
        from src.kg.ingest import build_graph_payload, get_graph_ingest_queue

        if not texts:
            logger.info(
                "[KG-TRIGGER] skipped doc=%s: no text to ingest", document_id,
            )
            return

        graph_payload = build_graph_payload(
            embeddings_payload={
                "texts": texts,
                "chunk_metadata": chunk_metadata,
                "doc_metadata": doc_metadata,
            },
            subscription_id=str(subscription_id),
            profile_id=str(profile_id),
            document_id=str(document_id),
            doc_name=doc_name,
            deep_entities=deep_entities,
            typed_relationships=typed_relationships,
        )
        if graph_payload is None:
            logger.info(
                "[KG-TRIGGER] build_graph_payload returned None for doc=%s "
                "(KG disabled in config or no texts)",
                document_id,
            )
            return

        if temporal_spans:
            # build_graph_payload doesn't take temporal_spans directly; the
            # downstream ingest layer reads them from the payload object.
            graph_payload.temporal_spans = temporal_spans

        queue = get_graph_ingest_queue(redis_client)
        queue.enqueue(graph_payload)
        logger.info(
            "[KG-TRIGGER] enqueued doc=%s entities=%d text_chars=%d",
            document_id,
            len(deep_entities or []),
            sum(len(t) for t in texts),
        )
    except Exception:  # noqa: BLE001
        logger.exception(
            "[KG-TRIGGER] failed doc=%s — extraction not affected",
            document_id,
        )


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def enqueue_from_extraction_result(
    *,
    extraction_result: Any,
    source_file: str,
    async_: bool = True,
    redis_client: Any = None,
) -> None:
    """Trigger a KG update from a unified ``ExtractionResult``.

    ``extraction_result`` is the dataclass produced by
    ``src.extraction.engine.ExtractionEngine.extract``. Its Layer 1
    ``clean_text`` becomes the single KG "chunk"; Layer 2 entities are
    forwarded as ``deep_entities``.
    """
    text = (getattr(extraction_result, "clean_text", "") or "").strip()
    chunk_metadata = [{
        "chunk_id": f"{extraction_result.document_id}::extraction",
        "source_name": source_file,
    }] if text else []

    deep_entities: List[Dict[str, Any]] = []
    for e in (getattr(extraction_result, "entities", None) or []):
        if hasattr(e, "__dict__"):
            d = vars(e)
        elif isinstance(e, dict):
            d = e
        else:
            continue
        text_val = d.get("text") or d.get("name") or ""
        if not text_val:
            continue
        deep_entities.append({
            "text": str(text_val),
            "type": str(d.get("type") or "UNKNOWN"),
            "confidence": float(d.get("confidence") or 0.0),
            "source": str(d.get("source") or "v2"),
            "normalized_name": str(text_val).lower().strip(),
        })

    doc_metadata = {
        "document_type": (
            extraction_result.metadata.get("doc_type_detected", "generic")
            if hasattr(extraction_result, "metadata") else "generic"
        ),
        "doc_type": (
            extraction_result.metadata.get("doc_type_detected", "generic")
            if hasattr(extraction_result, "metadata") else "generic"
        ),
    }

    kwargs = dict(
        document_id=str(extraction_result.document_id),
        subscription_id=str(extraction_result.subscription_id),
        profile_id=str(extraction_result.profile_id),
        doc_name=source_file,
        texts=[text] if text else [],
        chunk_metadata=chunk_metadata,
        doc_metadata=doc_metadata,
        deep_entities=deep_entities,
        typed_relationships=None,
        temporal_spans=None,
        redis_client=redis_client,
    )

    if async_:
        threading.Thread(
            target=_run_kg_ingest, kwargs=kwargs,
            daemon=True, name="kg-trigger-result",
        ).start()
    else:
        _run_kg_ingest(**kwargs)


def enqueue_from_legacy_payload(
    *,
    document_id: str,
    subscription_id: str,
    profile_id: str,
    source_name: str,
    payload_to_save: Dict[str, Any],
    deep_result: Any = None,
    async_: bool = True,
    redis_client: Any = None,
) -> None:
    """Trigger a KG update from the legacy ``payload_to_save`` shape.

    ``payload_to_save`` is the dict-of-dicts that ``extraction_service``
    and ``dataHandler`` already build. Each entry keyed by filename has a
    ``full_text`` or ``text`` field. ``deep_result`` is the optional
    ``DeepAnalysisResult`` carrying entities / typed_relationships /
    temporal_spans.
    """
    texts: List[str] = []
    chunk_metadata: List[Dict[str, Any]] = []
    structured = (payload_to_save or {}).get("structured") or {}
    for fname, content in structured.items():
        if isinstance(content, dict):
            full_text = str(content.get("full_text") or content.get("text") or "")
            if full_text:
                texts.append(full_text)
                chunk_metadata.append({
                    "chunk_id": f"{document_id}::extraction::{fname}",
                    "source_name": fname,
                })

    deep_entities: Optional[List[Dict[str, Any]]] = None
    typed_relationships: Optional[List[Dict[str, Any]]] = None
    temporal_spans: Optional[List[Dict[str, Any]]] = None
    if deep_result is not None:
        try:
            deep_entities = [e.to_dict() for e in (deep_result.entities or [])]
        except Exception:  # noqa: BLE001
            deep_entities = None
        try:
            typed_relationships = list(deep_result.typed_relationships or [])
        except Exception:  # noqa: BLE001
            typed_relationships = None
        try:
            temporal_spans = list(deep_result.temporal_spans or [])
        except Exception:  # noqa: BLE001
            temporal_spans = None

    doc_classification = payload_to_save.get("document_classification") or {}
    doc_metadata = {
        "document_type": doc_classification.get("document_type", "generic"),
        "doc_type": doc_classification.get("domain", "generic"),
    }

    kwargs = dict(
        document_id=str(document_id),
        subscription_id=str(subscription_id),
        profile_id=str(profile_id),
        doc_name=source_name,
        texts=texts,
        chunk_metadata=chunk_metadata,
        doc_metadata=doc_metadata,
        deep_entities=deep_entities,
        typed_relationships=typed_relationships,
        temporal_spans=temporal_spans,
        redis_client=redis_client,
    )

    if async_:
        threading.Thread(
            target=_run_kg_ingest, kwargs=kwargs,
            daemon=True, name="kg-trigger-legacy",
        ).start()
    else:
        _run_kg_ingest(**kwargs)


def enqueue_from_embeddings(
    *,
    document_id: str,
    subscription_id: str,
    profile_id: str,
    doc_name: str,
    embeddings_payload: Dict[str, Any],
    doc_metadata: Optional[Dict[str, Any]] = None,
    async_: bool = True,
    redis_client: Any = None,
) -> None:
    """Trigger a KG update from an already-built ``embeddings_payload``.

    Used by the dataHandler embedding flow which builds the embeddings
    payload inline before enqueuing.
    """
    texts = list(embeddings_payload.get("texts") or [])
    chunk_metadata = list(embeddings_payload.get("chunk_metadata") or [])
    meta = doc_metadata or embeddings_payload.get("doc_metadata") or {}

    kwargs = dict(
        document_id=str(document_id),
        subscription_id=str(subscription_id),
        profile_id=str(profile_id),
        doc_name=doc_name,
        texts=texts,
        chunk_metadata=chunk_metadata,
        doc_metadata=meta,
        deep_entities=None,
        typed_relationships=None,
        temporal_spans=None,
        redis_client=redis_client,
    )

    if async_:
        threading.Thread(
            target=_run_kg_ingest, kwargs=kwargs,
            daemon=True, name="kg-trigger-embeddings",
        ).start()
    else:
        _run_kg_ingest(**kwargs)


__all__ = [
    "enqueue_from_extraction_result",
    "enqueue_from_legacy_payload",
    "enqueue_from_embeddings",
]
