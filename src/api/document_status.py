import json
from datetime import datetime
from src.utils.logging_utils import get_logger
import time
import traceback
from typing import Any, Dict, List, Optional

from pymongo import ReturnDocument
from bson import ObjectId

from src.api.config import Config
from src.api.statuses import (
    STATUS_UNDER_REVIEW,
    PIPELINE_UPLOADED,
    STAGE_PENDING,
    STAGE_IN_PROGRESS,
    STAGE_COMPLETED,
    STAGE_FAILED,
)

logger = get_logger(__name__)

_PROGRESS_TTL = 3600  # 1 hour
_PROGRESS_CHANNEL = "dw:training:events"

def emit_progress(
    document_id: str,
    stage: str,
    progress: float,
    detail: str = "",
    extra: dict = None,
):
    """Emit a training progress event to Redis for real-time SSE streaming."""
    try:
        from src.api.dw_newron import get_redis_client
        client = get_redis_client()
        if not client:
            return

        event = {
            "document_id": str(document_id),
            "stage": stage,
            "progress": round(min(max(progress, 0.0), 1.0), 3),
            "detail": detail,
            "timestamp": time.time(),
        }
        if extra:
            event["extra"] = extra

        payload = json.dumps(event)
        client.setex(f"dw:training:progress:{document_id}", _PROGRESS_TTL, payload)
        client.publish(_PROGRESS_CHANNEL, payload)
    except Exception:
        pass  # Best-effort, never block the pipeline

def get_training_progress(document_id: str) -> Optional[dict]:
    """Get the latest training progress from Redis (for polling)."""
    try:
        from src.api.dw_newron import get_redis_client
        client = get_redis_client()
        if not client:
            return None
        raw = client.get(f"dw:training:progress:{document_id}")
        return json.loads(raw) if raw else None
    except Exception:
        return None


def get_training_progress_batch(document_ids: List[str]) -> Dict[str, Optional[dict]]:
    """Fetch training progress for many documents in a single Redis MGET.

    The per-document `get_training_progress()` is N roundtrips to Azure Redis
    (~200ms each on Cosmos-backed deployments); this collapses them to one
    roundtrip for the polling endpoint where N is typically 5-50.
    Returns {doc_id: dict-or-None}.
    """
    if not document_ids:
        return {}
    try:
        from src.api.dw_newron import get_redis_client
        client = get_redis_client()
        if not client:
            return {doc_id: None for doc_id in document_ids}
        keys = [f"dw:training:progress:{doc_id}" for doc_id in document_ids]
        raws = client.mget(keys)
    except Exception:
        return {doc_id: None for doc_id in document_ids}
    out: Dict[str, Optional[dict]] = {}
    for doc_id, raw in zip(document_ids, raws):
        if not raw:
            out[doc_id] = None
            continue
        try:
            out[doc_id] = json.loads(raw)
        except Exception:
            out[doc_id] = None
    return out

_STATUS_LOG_TTL = 86400  # 24 hours


def emit_status_log(
    document_id: str,
    stage: str,
    step: str,
    detail: str,
    extra: dict = None,
) -> None:
    """Append a timestamped log entry to the document's status log in Redis.

    Each entry captures a discrete pipeline step with wall-clock time so the
    frontend can render a detailed timeline.
    """
    try:
        from src.api.dw_newron import get_redis_client
        client = get_redis_client()
        if not client:
            return
        entry = {
            "stage": stage,
            "step": step,
            "detail": detail,
            "timestamp": time.time(),
            "ts_iso": datetime.utcnow().isoformat() + "Z",
        }
        if extra:
            entry["extra"] = extra
        key = f"dw:status_log:{document_id}"
        client.rpush(key, json.dumps(entry))
        client.expire(key, _STATUS_LOG_TTL)
    except Exception:
        pass  # Best-effort, never block the pipeline


def get_status_logs(document_id: str) -> List[dict]:
    """Return the full ordered status log for a document.

    ``elapsed_since_stage_start`` is computed per-stage so extraction and
    embedding each start from 0, which is more useful for the UI than a
    single global counter.
    """
    try:
        from src.api.dw_newron import get_redis_client
        client = get_redis_client()
        if not client:
            return []
        raw_entries = client.lrange(f"dw:status_log:{document_id}", 0, -1)
        logs = []
        for raw in raw_entries:
            try:
                logs.append(json.loads(raw))
            except (json.JSONDecodeError, TypeError):
                pass
        # Compute elapsed per-stage (extraction and embedding each start from 0)
        stage_start: Dict[str, float] = {}
        for entry in logs:
            stage = entry.get("stage", "")
            ts = entry.get("timestamp", 0)
            if stage not in stage_start:
                stage_start[stage] = ts
            entry["elapsed_since_stage_start"] = round(ts - stage_start[stage], 2)
        return logs
    except Exception:
        return []


def clear_status_logs(document_id: str) -> None:
    """Clear status logs for a document (called at start of a new pipeline run)."""
    try:
        from src.api.dw_newron import get_redis_client
        client = get_redis_client()
        if client:
            client.delete(f"dw:status_log:{document_id}")
    except Exception:
        pass


def _progress_to_percent(progress: Optional[dict]) -> Optional[dict]:
    """Convert progress 0.0-1.0 to 0-100 scale for UI consumption."""
    if not progress:
        return progress
    result = dict(progress)
    if "progress" in result and isinstance(result["progress"], (int, float)):
        result["progress"] = round(result["progress"] * 100, 1)
    return result


# Maps document status to a deterministic pipeline progress % (0-100).
# Used when Redis real-time progress has expired.
_STATUS_TO_PROGRESS = {
    "UNDER_REVIEW": 0,
    "EXTRACTION_IN_PROGRESS": 10,
    "EXTRACTION_COMPLETED": 25,
    "EXTRACTION_FAILED": 0,
    "SCREENING_IN_PROGRESS": 30,
    "SCREENING_COMPLETED": 40,
    "TRAINING_STARTED": 45,
    "EMBEDDING_IN_PROGRESS": 50,
    "TRAINING_COMPLETED": 100,
    "TRAINING_FAILED": 0,
    "TRAINING_PARTIALLY_COMPLETED": 80,
    "TRAINING_BLOCKED_SECURITY": 0,
    "TRAINING_BLOCKED_CONFIDENTIAL": 0,
    "EXTRACTION_OR_CHUNKING_FAILED": 0,
    "EMBEDDING_FAILED": 0,
}


def _compute_document_progress(status: str, redis_progress: Optional[dict]) -> dict:
    """Build a unified progress object for a document.

    If Redis has a live value (from ``emit_progress``), use it and convert to
    0-100.  Otherwise derive a deterministic value from the document status so
    the UI always has something to show.
    """
    if redis_progress:
        result = dict(redis_progress)
        raw = result.get("progress", 0)
        if isinstance(raw, (int, float)):
            result["progress"] = round(raw * 100, 1)
        result["source"] = "live"
        return result

    pct = _STATUS_TO_PROGRESS.get(status, 0)
    return {
        "progress": pct,
        "stage": status.lower(),
        "detail": status.replace("_", " ").title(),
        "source": "derived",
    }


# Extraction-scoped progress: used by /api/extract/progress where the
# endpoint cares only about the extraction stage, not the full pipeline.
# EXTRACTION_COMPLETED is 100% here (extraction is done, even if screening
# and training are still ahead). Any downstream-stage status implies
# extraction finished earlier, so these also show 100%.
_EXTRACTION_PROGRESS_BY_STATUS = {
    "UNDER_REVIEW":                  0,
    "EXTRACTION_IN_PROGRESS":       50,
    "EXTRACTION_COMPLETED":        100,
    "EXTRACTION_FAILED":             0,
    "EXTRACTION_OR_CHUNKING_FAILED": 0,
    # Downstream stages — extraction is definitionally done.
    "SCREENING_IN_PROGRESS":       100,
    "SCREENING_COMPLETED":         100,
    "TRAINING_STARTED":            100,
    "EMBEDDING_IN_PROGRESS":       100,
    "TRAINING_COMPLETED":          100,
    "TRAINING_PARTIALLY_COMPLETED":100,
    "TRAINING_FAILED":             100,
    "EMBEDDING_FAILED":            100,
    "TRAINING_BLOCKED_SECURITY":   100,
    "TRAINING_BLOCKED_CONFIDENTIAL":100,
}


def _compute_extraction_progress(status: str, redis_progress: Optional[dict]) -> dict:
    """Extraction-scoped progress (used by /api/extract/progress).

    Shares the live-Redis path with ``_compute_document_progress`` but derives
    its deterministic value from the extraction-scoped map so that, for
    example, ``EXTRACTION_COMPLETED`` reads as 100% (extraction is done).
    """
    if redis_progress:
        result = dict(redis_progress)
        raw = result.get("progress", 0)
        if isinstance(raw, (int, float)):
            result["progress"] = round(raw * 100, 1)
        result["source"] = "live"
        return result

    pct = _EXTRACTION_PROGRESS_BY_STATUS.get(status, 0)
    return {
        "progress": pct,
        "stage": status.lower(),
        "detail": status.replace("_", " ").title(),
        "source": "derived",
    }


def _format_elapsed(seconds: Optional[float]) -> str:
    """Format elapsed seconds into a human-readable string."""
    if seconds is None or seconds <= 0:
        return "0s"
    if seconds < 60:
        return f"{int(seconds)}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes < 60:
        return f"{minutes}m {secs}s" if secs else f"{minutes}m"
    hours = int(minutes // 60)
    mins = minutes % 60
    return f"{hours}h {mins}m" if mins else f"{hours}h"


# Short TTL so polling (every 2-5s) hits cache most of the time without
# hiding active-upload progress behind stale data. Redis SETEX wants an
# int; PSETEX takes milliseconds if we ever want sub-second granularity.
_PROGRESS_CACHE_TTL_MS = 1500
_PROGRESS_CACHE_KEY = "dw:progress:extract:{profile_id}"

# Statuses that mean the document is still in the active upload/extraction
# flow. /api/extract/progress returns only these docs so the UI poller
# tracks the current batch, not every historical document in the profile.
_EXTRACT_PROGRESS_IN_FLIGHT_STATUSES = frozenset({
    "UNDER_REVIEW",
    "UPLOADED",
    "EXTRACTION_IN_PROGRESS",
})


def _progress_cache_get(profile_id: str) -> Optional[Dict[str, Any]]:
    try:
        from src.api.dw_newron import get_redis_client
        client = get_redis_client()
        if not client:
            return None
        raw = client.get(_PROGRESS_CACHE_KEY.format(profile_id=profile_id))
        if not raw:
            return None
        return json.loads(raw)
    except Exception:
        return None


def _progress_cache_set(profile_id: str, payload: Dict[str, Any]) -> None:
    try:
        from src.api.dw_newron import get_redis_client
        client = get_redis_client()
        if not client:
            return
        # PSETEX wants milliseconds; we want sub-second precision (1.5s).
        # Use default=str in json.dumps so stray datetime/ObjectId values
        # serialize instead of raising and aborting the cache write.
        client.psetex(
            _PROGRESS_CACHE_KEY.format(profile_id=profile_id),
            _PROGRESS_CACHE_TTL_MS,
            json.dumps(payload, default=str),
        )
    except Exception:
        pass


def get_profile_extraction_status(profile_id: str) -> Dict[str, Any]:
    """Get comprehensive extraction status for all documents in a profile.

    Wrapped in a short (1.5s) Redis cache — this endpoint is polled every
    few seconds by the UI, so a tiny TTL lets us service most calls from
    cache while still feeling live during an active upload. The TTL is
    smaller than the UI's typical 5s poll interval so the user never sees
    progress "stuck" because of stale cache.
    """
    cached = _progress_cache_get(profile_id)
    if cached is not None:
        return cached

    from src.utils.logging_utils import get_live_logs

    collection = get_documents_collection()
    if collection is None:
        empty = {"documents": [], "common_data": {
            "Overall_live_logs": [], "overall_progress": 0,
            "total_documents": 0, "uploaded": 0, "elapsed_time": "0s",
        }}
        _progress_cache_set(profile_id, empty)
        return empty

    # Pre-session ObjectId fix: docs may store the profile reference under
    # `profile` (ObjectId) or `profile_id`/`profileId` (string or ObjectId).
    prof_forms: List[Any] = [str(profile_id)]
    if ObjectId.is_valid(str(profile_id)):
        prof_forms.append(ObjectId(str(profile_id)))
    query = {
        "$or": [
            {"profile_id": {"$in": prof_forms}},
            {"profile":    {"$in": prof_forms}},
            {"profileId":  {"$in": prof_forms}},
        ]
    }
    docs = list(collection.find(
        query,
        {
            "document_id": 1, "status": 1,
            "name": 1, "source_file": 1,
            "extraction": 1, "screening": 1, "embedding": 1,
            "training_started_at": 1, "trained_at": 1,
            "created_at": 1, "updated_at": 1,
        },
    ).sort("updated_at", -1))

    now = time.time()
    result_docs = []
    progress_sum = 0.0
    earliest_start: Optional[float] = None
    latest_end: Optional[float] = None
    uploaded_count = 0

    # Batch Redis lookups: single MGET for all docs instead of N GETs.
    # On Azure-hosted Redis this collapses the polling endpoint's Redis
    # cost from N * RTT to 1 * RTT.
    all_doc_ids = [str(doc.get("document_id") or doc.get("_id")) for doc in docs]
    progress_by_doc = get_training_progress_batch(all_doc_ids)

    for doc in docs:
        doc_id = str(doc.get("document_id") or doc.get("_id"))
        status = doc.get("status", "UNKNOWN")

        # Only in-flight docs contribute to the exposed progress/elapsed
        # aggregates — otherwise historical completed docs push the overall
        # percentage toward 100 and drag the elapsed clock back to whenever
        # the oldest completed extraction started.
        if status not in _EXTRACT_PROGRESS_IN_FLIGHT_STATUSES:
            continue

        # Extraction-scoped: EXTRACTION_COMPLETED → 100% on this endpoint.
        progress = _compute_extraction_progress(status, progress_by_doc.get(doc_id))
        progress_sum += progress.get("progress", 0)

        # Track earliest start and latest end for elapsed time, scoped to
        # the in-flight batch only.
        extraction = doc.get("extraction") or {}
        embedding = doc.get("embedding") or {}
        start_at = extraction.get("started_at")
        if start_at and isinstance(start_at, (int, float)):
            if earliest_start is None or start_at < earliest_start:
                earliest_start = start_at

        end_at = (
            doc.get("trained_at")
            or embedding.get("completed_at")
            or extraction.get("completed_at")
        )
        if end_at and isinstance(end_at, (int, float)):
            if latest_end is None or end_at > latest_end:
                latest_end = end_at

        # ``uploaded`` here means "currently being extracted" (the narrower
        # sub-state of the in-flight set).
        if status == "EXTRACTION_IN_PROGRESS":
            uploaded_count += 1

        result_docs.append({
            "document_id": doc_id,
            "document_name": doc.get("source_file") or doc.get("name", ""),
            "progress": progress,
        })

    # Live logs: actual terminal output captured by RedisLogHandler
    live_logs = get_live_logs(profile_id)

    total_docs = len(result_docs)
    overall_progress = round(progress_sum / total_docs, 1) if total_docs else 0

    # Elapsed time: from earliest start to latest end (or now if still running)
    elapsed_seconds: Optional[float] = None
    if earliest_start:
        end_ref = latest_end or now
        elapsed_seconds = round(end_ref - earliest_start, 1)

    payload = {
        "documents": result_docs,
        "common_data": {
            "Overall_live_logs": live_logs,
            "overall_progress": overall_progress,
            "total_documents": total_docs,
            "uploaded": uploaded_count,
            "elapsed_time": _format_elapsed(elapsed_seconds),
        },
    }
    _progress_cache_set(profile_id, payload)
    return payload


def get_profile_training_status(profile_id: str) -> Dict[str, Any]:
    """Get comprehensive training (embedding) status for all documents in a profile."""
    from src.utils.logging_utils import get_live_logs

    collection = get_documents_collection()
    if collection is None:
        return {"documents": [], "common_data": {
            "Overall_live_logs": [], "overall_progress": 0,
            "total_documents": 0, "uploaded": 0, "elapsed_time": "0s",
        }}

    # Focus on documents that have reached or passed screening
    docs = list(collection.find(
        {
            "profile_id": profile_id,
            "status": {"$in": [
                "SCREENING_COMPLETED", "TRAINING_STARTED", "TRAINING_COMPLETED",
                "TRAINING_FAILED", "TRAINING_PARTIALLY_COMPLETED",
                "TRAINING_BLOCKED_SECURITY", "TRAINING_BLOCKED_CONFIDENTIAL",
                "EXTRACTION_OR_CHUNKING_FAILED", "EMBEDDING_FAILED",
            ]},
        },
        {
            "document_id": 1, "status": 1, "source_file": 1,
            "embedding": 1,
            "training_started_at": 1, "trained_at": 1,
            "created_at": 1, "updated_at": 1,
        },
    ).sort("updated_at", -1))

    now = time.time()
    result_docs = []
    progress_sum = 0.0
    earliest_start: Optional[float] = None
    latest_end: Optional[float] = None
    uploaded_count = 0

    for doc in docs:
        doc_id = str(doc.get("document_id") or doc.get("_id"))
        status = doc.get("status", "UNKNOWN")

        progress = _compute_document_progress(status, get_training_progress(doc_id))
        progress_sum += progress.get("progress", 0)

        # Track earliest start and latest end for elapsed time
        embedding = doc.get("embedding") or {}
        start_at = embedding.get("started_at") or doc.get("training_started_at")
        if start_at and isinstance(start_at, (int, float)):
            if earliest_start is None or start_at < earliest_start:
                earliest_start = start_at

        end_at = doc.get("trained_at") or embedding.get("completed_at")
        if end_at and isinstance(end_at, (int, float)):
            if latest_end is None or end_at > latest_end:
                latest_end = end_at

        uploaded_count += 1

        result_docs.append({
            "document_id": doc_id,
            "document_name": doc.get("source_file") or doc.get("name", ""),
            "progress": progress,
        })

    # Live logs: actual terminal output captured by RedisLogHandler
    live_logs = get_live_logs(profile_id)

    total_docs = len(result_docs)
    overall_progress = round(progress_sum / total_docs, 1) if total_docs else 0

    # Elapsed time: from earliest start to latest end (or now if still running)
    elapsed_seconds: Optional[float] = None
    if earliest_start:
        end_ref = latest_end or now
        elapsed_seconds = round(end_ref - earliest_start, 1)

    return {
        "documents": result_docs,
        "common_data": {
            "Overall_live_logs": live_logs,
            "overall_progress": overall_progress,
            "total_documents": total_docs,
            "uploaded": uploaded_count,
            "elapsed_time": _format_elapsed(elapsed_seconds),
        },
    }


_ZOMBIE_TIMEOUT_SECONDS = 1800  # 30 minutes
_EXTRACTION_ZOMBIE_TIMEOUT_SECONDS = 1200  # 20 minutes

def recover_zombie_documents(timeout_seconds: int = _ZOMBIE_TIMEOUT_SECONDS) -> int:
    """Auto-fail documents stuck in TRAINING_STARTED beyond timeout."""
    collection = get_documents_collection()
    cutoff = time.time() - timeout_seconds
    zombies = list(collection.find(
        {"status": "TRAINING_STARTED", "training_started_at": {"$lt": cutoff}},
        {"_id": 1, "document_id": 1, "training_started_at": 1},
    ))
    recovered = 0
    for doc in zombies:
        doc_id = str(doc.get("document_id") or doc.get("_id"))
        hours = (time.time() - (doc.get("training_started_at") or 0)) / 3600
        try:
            update_document_fields(doc_id, {
                "status": "TRAINING_FAILED",
                "training_error": f"Zombie recovery: stuck for {hours:.1f}h",
                "training_failed_at": time.time(),
                "error_summary": "zombie_timeout",
            })
            update_stage(doc_id, "embedding", {
                "status": "FAILED", "completed_at": time.time(),
                "reason": "zombie_timeout",
                "error": {"message": f"Process died — stuck for {hours:.1f}h, auto-recovered"},
            })
            emit_progress(doc_id, "failed", 0.0, f"Auto-recovered: stuck for {hours:.1f}h")
            recovered += 1
            logger.info("Recovered zombie document %s (stuck %.1fh)", doc_id, hours)
        except Exception:
            logger.warning("Failed to recover zombie %s", doc_id, exc_info=True)
    return recovered


def recover_zombie_extractions(timeout_seconds: int = _EXTRACTION_ZOMBIE_TIMEOUT_SECONDS) -> int:
    """Reset documents stuck in extraction IN_PROGRESS back to UNDER_REVIEW.

    When the server is killed during extraction, documents are left with
    ``status=UNDER_REVIEW`` and ``extraction.status=IN_PROGRESS`` forever.
    This function detects those zombies and resets extraction state so the
    next ``extract_documents()`` call will retry them.
    """
    collection = get_documents_collection()
    cutoff = time.time() - timeout_seconds
    zombies = list(collection.find(
        {
            "status": STATUS_UNDER_REVIEW,
            "extraction.status": "IN_PROGRESS",
            "extraction.started_at": {"$lt": cutoff},
        },
        {"_id": 1, "document_id": 1, "extraction.started_at": 1},
    ))
    recovered = 0
    for doc in zombies:
        doc_id = str(doc.get("document_id") or doc.get("_id"))
        started = (doc.get("extraction") or {}).get("started_at", 0)
        minutes = (time.time() - started) / 60 if started else 0
        try:
            update_stage(doc_id, "extraction", {
                "status": "PENDING",
                "completed_at": None,
                "error": None,
                "recovery_reason": f"zombie_reset: stuck in IN_PROGRESS for {minutes:.0f}min",
            })
            recovered += 1
            logger.info("Reset zombie extraction %s (stuck %.0fmin)", doc_id, minutes)
        except Exception:
            logger.warning("Failed to reset zombie extraction %s", doc_id, exc_info=True)
    return recovered

_MISSING = object()

def get_documents_collection():
    from src.api.dataHandler import db
    return db[Config.MongoDB.DOCUMENTS]

def get_screening_collection():
    from src.api.dataHandler import db
    return db["screening"]

def _doc_id_value(document_id: str):
    if ObjectId.is_valid(str(document_id)):
        return ObjectId(str(document_id))
    return str(document_id)

def _doc_filter(document_id: str) -> Dict[str, Any]:
    doc_id_str = str(document_id)
    candidates = [
        {"_id": doc_id_str},
        {"document_id": doc_id_str},
        {"documentId": doc_id_str},
        {"doc_id": doc_id_str},
        {"id": doc_id_str},
    ]
    if ObjectId.is_valid(str(document_id)):
        candidates.insert(0, {"_id": ObjectId(str(document_id))})
    return {"$or": candidates}

def init_document_record(
    document_id: str,
    subscription_id: Optional[str] = None,
    profile_id: Optional[str] = None,
    doc_type: Optional[str] = None,
    filename: Optional[str] = None,
    content_type: Optional[str] = None,
    size: Optional[int] = None,
    # New pipeline-schema parameters (with defaults for backward compat)
    source_file: Optional[str] = None,
    file_type: Optional[str] = None,
    content_size: Optional[int] = None,
    blob_url: Optional[str] = None,
    created_by: str = "system",
):
    """Create or update a document record with full pipeline schema.

    Backward compatible: existing callers using (doc_type, filename, size)
    continue to work.  New callers can use (source_file, file_type,
    content_size, blob_url, created_by) for the enriched schema.
    """
    now = time.time()
    utcnow = datetime.utcnow()

    # Resolve aliased parameters (new names take precedence)
    resolved_source_file = source_file or filename
    resolved_file_type = file_type or doc_type
    resolved_content_size = content_size if content_size is not None else size

    update: Dict[str, Any] = {"document_id": str(document_id), "updated_at": now}
    if subscription_id:
        update["subscription_id"] = str(subscription_id)
    if profile_id:
        update["profile_id"] = str(profile_id)
    if resolved_file_type:
        update["doc_type"] = str(resolved_file_type)
    if resolved_source_file:
        update["source_file"] = resolved_source_file
    if content_type:
        update["content_type"] = content_type
    if resolved_content_size is not None:
        update["content_size"] = int(resolved_content_size)
    if blob_url:
        update["blob_url"] = blob_url

    # Single-column status (per operator directive): status is the sole
    # source of truth; the legacy pipeline_status field is no longer written.
    # Initial state is UNDER_REVIEW to preserve /api/extract eligibility.
    update["created_by"] = created_by

    _pending_stage = lambda: {
        "status": STAGE_PENDING,
        "started_at": None,
        "completed_at": None,
        "celery_task_id": None,
        "error": None,
        "summary": None,
    }

    update["extraction"] = _pending_stage()
    update["screening"] = {**_pending_stage(), "plugins_run": None}
    update["knowledge_graph"] = {
        "status": STAGE_PENDING,
        "started_at": None,
        "completed_at": None,
        "node_count": 0,
        "edge_count": 0,
        "neo4j_subgraph_id": None,
    }
    update["embedding"] = _pending_stage()

    # Audit log
    update["audit_log"] = [{"action": "UPLOADED", "by": created_by, "at": utcnow}]

    collection = get_documents_collection()
    # Two-step: find existing doc first, then update with exact _id filter.
    existing = collection.find_one(_doc_filter(document_id), {"_id": 1})
    if existing:
        return collection.find_one_and_update(
            {"_id": existing["_id"]},
            {"$set": update},
            return_document=ReturnDocument.AFTER,
        )
    else:
        return collection.find_one_and_update(
            {"_id": _doc_id_value(document_id)},
            {
                "$set": update,
                "$setOnInsert": {
                    "created_at": now,
                    "_id": _doc_id_value(document_id),
                    "status": STATUS_UNDER_REVIEW,
                },
            },
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )

def _flatten(prefix: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in payload.items():
        target = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten(target, value))
        else:
            flat[target] = value
    return flat

def update_document_fields(document_id: str, fields: Dict[str, Any]):
    now = time.time()
    update = dict(fields)
    update["updated_at"] = now
    unset_fields: Dict[str, Any] = {}

    # Single-column status (per operator directive): legacy pipeline_status
    # sync removed. If a caller still passes pipeline_status through, drop it
    # rather than writing it to Mongo.
    update.pop("pipeline_status", None)

    # Enforce: error is either missing or an object (never null).
    error_value = update.pop("error", _MISSING)
    if error_value is not _MISSING:
        if error_value is None:
            unset_fields["error"] = ""
        elif isinstance(error_value, dict):
            update["error"] = error_value
        else:
            update["error"] = {"message": str(error_value)}

    for key, value in list(update.items()):
        if not key.endswith(".error"):
            continue
        update.pop(key)
        if value is None:
            unset_fields[key] = ""
        elif isinstance(value, dict):
            update[key] = value
        else:
            update[key] = {"message": str(value)}

    collection = get_documents_collection()
    update_ops: Dict[str, Any] = {
        "$set": update,
        "$setOnInsert": {"created_at": now, "_id": _doc_id_value(document_id)},
    }
    if unset_fields:
        update_ops["$unset"] = unset_fields

    # Two-step: find existing doc first, then update with exact _id filter.
    # Avoids $or + upsert which can create duplicates or silently fail on CosmosDB.
    existing = collection.find_one(_doc_filter(document_id), {"_id": 1})
    if existing:
        exact_filter = {"_id": existing["_id"]}
        # No upsert needed — document exists.
        update_ops.pop("$setOnInsert", None)
        result = collection.find_one_and_update(
            exact_filter,
            update_ops,
            return_document=ReturnDocument.AFTER,
        )
    else:
        # Document doesn't exist yet — insert with simple _id filter (no $or).
        simple_filter = {"_id": _doc_id_value(document_id)}
        result = collection.find_one_and_update(
            simple_filter,
            update_ops,
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )

    if result:
        new_status = result.get("status")
        result_id = result.get("_id")
        logger.info(
            "[STATUS_UPDATE] doc=%s _id=%s set_status=%s confirmed_status=%s",
            document_id, result_id,
            update.get("status", "N/A"), new_status,
        )
    return result

def update_stage(document_id: str, stage: str, patch: Optional[Dict[str, Any]] = None,
                 status: Optional[str] = None,
                 celery_task_id: Optional[str] = None,
                 error: object = _MISSING,
                 summary: Optional[Dict[str, Any]] = None,
                 blob_path: Optional[str] = None,
                 **extra):
    """Update a specific pipeline stage's status and metadata.

    Supports two calling conventions:
      - Legacy dict style:  update_stage(doc_id, "extraction", {"status": "IN_PROGRESS", ...})
      - New keyword style:  update_stage(doc_id, "extraction", status="IN_PROGRESS", celery_task_id="abc")

    When *patch* is provided (legacy style), keyword arguments are ignored.
    """
    now = time.time()

    if patch is not None:
        # --- Legacy dict-based path (backward compatible) ---
        patch_copy = dict(patch)
        error_value = patch_copy.pop("error", _MISSING)
        flat = _flatten(stage, patch_copy)
        flat["updated_at"] = now
    else:
        # --- New keyword-based path ---
        flat: Dict[str, Any] = {"updated_at": now}
        if status is not None:
            flat[f"{stage}.status"] = status
            if status == STAGE_IN_PROGRESS:
                flat[f"{stage}.started_at"] = now
            elif status in (STAGE_COMPLETED, STAGE_FAILED):
                flat[f"{stage}.completed_at"] = now
        if celery_task_id is not None:
            flat[f"{stage}.celery_task_id"] = celery_task_id
        if summary is not None:
            # Fold blob_path into the summary dict rather than setting both the
            # parent path ({stage}.summary) and a child path
            # ({stage}.summary.blob_path) in the same $set. MongoDB rejects
            # that as ConflictingUpdateOperators.
            if blob_path is not None:
                summary = {**summary, "blob_path": blob_path}
            flat[f"{stage}.summary"] = summary
        elif blob_path is not None:
            flat[f"{stage}.summary.blob_path"] = blob_path
        for key, value in extra.items():
            flat[f"{stage}.{key}"] = value
        error_value = error

    set_error: Dict[str, Any] = {}
    unset_error: Dict[str, Any] = {}
    if error_value is not _MISSING:
        error_path = f"{stage}.error"
        if error_value is None:
            unset_error[error_path] = ""
        elif isinstance(error_value, dict):
            set_error[error_path] = error_value
        else:
            set_error[error_path] = {"message": str(error_value)}

    collection = get_documents_collection()
    update_ops: Dict[str, Any] = {
        "$set": {**flat, **set_error},
        "$setOnInsert": {"created_at": now, "_id": _doc_id_value(document_id)},
    }
    if unset_error:
        update_ops["$unset"] = unset_error

    # Two-step: find existing doc first, then update with exact _id filter.
    existing = collection.find_one(_doc_filter(document_id), {"_id": 1})
    if existing:
        exact_filter = {"_id": existing["_id"]}
        update_ops.pop("$setOnInsert", None)
        return collection.find_one_and_update(
            exact_filter,
            update_ops,
            return_document=ReturnDocument.AFTER,
        )
    else:
        simple_filter = {"_id": _doc_id_value(document_id)}
        return collection.find_one_and_update(
            simple_filter,
            update_ops,
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )

def set_error(document_id: str, stage: str, exc: Exception):
    message = str(exc)
    trace = traceback.format_exc()
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    return update_stage(
        document_id,
        stage,
        {
            "status": "FAILED",
            "completed_at": time.time(),
            "error": {"message": message, "trace": trace, "code": code},
        },
    )

def append_audit_log(document_id: str, action: str, by: str = "system", **extra):
    """Append an entry to the document's audit log."""
    entry: Dict[str, Any] = {"action": action, "by": by, "at": datetime.utcnow()}
    entry.update(extra)
    collection = get_documents_collection()
    existing = collection.find_one(_doc_filter(document_id), {"_id": 1})
    if existing:
        collection.update_one(
            {"_id": existing["_id"]},
            {
                "$push": {"audit_log": entry},
                "$set": {"updated_at": time.time()},
            },
        )
    else:
        collection.update_one(
            {"_id": _doc_id_value(document_id)},
            {
                "$push": {"audit_log": entry},
                "$set": {"updated_at": time.time()},
            },
        )


def update_pipeline_status(document_id: str, pipeline_status: str):
    """Update the ``status`` field (single source of truth).

    Function name preserved for backward compatibility with existing
    callers (Celery tasks, API routes). Writes ONLY the ``status`` field
    and ``$unset``s any legacy ``pipeline_status`` field so the two never
    drift — per operator directive to collapse onto a single column.
    """
    collection = get_documents_collection()
    now = time.time()
    existing = collection.find_one(_doc_filter(document_id), {"_id": 1})
    update_ops = {
        "$set": {"status": pipeline_status, "updated_at": now},
        "$unset": {"pipeline_status": ""},
    }
    if existing:
        collection.update_one({"_id": existing["_id"]}, update_ops)
    else:
        collection.update_one(
            {"_id": _doc_id_value(document_id)}, update_ops
        )


def upsert_screening_report(
    run_id: str,
    document_id: str,
    endpoint: str,
    status: str,
    result: Optional[Dict[str, Any]],
    errors: Optional[list],
    warnings: Optional[list],
    options: Optional[Dict[str, Any]] = None,
    subscription_id: Optional[str] = None,
):
    now = time.time()
    update = {
        "run_id": run_id,
        "doc_id": str(document_id),
        "endpoint": endpoint,
        "status": status,
        "result": result,
        "errors": errors or [],
        "warnings": warnings or [],
        "options": options or {},
        "subscription_id": subscription_id,
        "updated_at": now,
    }
    collection = get_screening_collection()
    return collection.update_one(
        {"run_id": run_id, "doc_id": str(document_id), "endpoint": endpoint},
        {"$set": update, "$setOnInsert": {"created_at": now}},
        upsert=True,
    )

def get_document_record(document_id: str) -> Optional[Dict[str, Any]]:
    collection = get_documents_collection()
    return collection.find_one(_doc_filter(document_id))

_ERROR_NULL_PATHS = [
    "error",
    "embedding.error",
    "extraction.error",
    "understanding.error",
    "screening.error",
    "screening.security.error",
    "cleanup.error",
]

def normalize_error_fields(collection=None) -> Dict[str, Any]:
    """
    Ensure error fields are either missing or objects (never null).

    Choice: unset null error fields (do not replace with {}).
    """
    collection = collection or get_documents_collection()
    if collection is None:
        return {"updated": 0, "paths": list(_ERROR_NULL_PATHS), "skipped": True}

    updated = 0
    for path in _ERROR_NULL_PATHS:
        result = collection.update_many({path: None}, {"$unset": {path: ""}})
        updated += int(getattr(result, "modified_count", 0) or 0)

    return {"updated": updated, "paths": list(_ERROR_NULL_PATHS), "skipped": False}


# ---------------------------------------------------------------------------
# Control-plane profile helpers (SME synthesis gating).
# ---------------------------------------------------------------------------
_PROFILES_COLLECTION_NAME = "profiles"

# Guards update_profile_record: MongoDB is control plane only, so only
# these fields may be written via this path.
_CP_ALLOWED_PROFILE_KEYS = frozenset(
    {
        "profile_domain",
        "sme_synthesis_version",
        "sme_last_input_hash",
        "sme_redesign_enabled",
        "sme_last_run_id",
        "enable_sme_synthesis",
        "enable_sme_retrieval",
        "enable_kg_synthesized_edges",
    }
)


def _profiles_collection():
    """Return the Mongo profiles collection (control plane only)."""
    from src.api.dataHandler import db
    return db[_PROFILES_COLLECTION_NAME]


def count_incomplete_docs_in_profile(
    *, subscription_id: str, profile_id: str, exclude_document_id: str
) -> int:
    """Count docs in ``(subscription_id, profile_id)`` not yet at
    ``TRAINING_COMPLETED``, excluding ``exclude_document_id``.

    Used to gate last-doc-in-profile SME synthesis firing: when this count
    hits zero the last document is the one whose embedding just finished,
    so synthesis runs as the final training-stage step before
    ``PIPELINE_TRAINING_COMPLETED`` flips.
    """
    collection = get_documents_collection()
    if collection is None:
        return 0
    return int(
        collection.count_documents(
            {
                "subscription_id": subscription_id,
                "profile_id": profile_id,
                "document_id": {"$ne": exclude_document_id},
                "pipeline_status": {"$ne": "TRAINING_COMPLETED"},
            }
        )
    )


def get_profile_record(
    subscription_id: str, profile_id: str
) -> Optional[Dict[str, Any]]:
    """Return the profile record or ``None`` if not present."""
    collection = _profiles_collection()
    if collection is None:
        return None
    return collection.find_one(
        {"subscription_id": subscription_id, "profile_id": profile_id}
    )


def update_profile_record(
    subscription_id: str, profile_id: str, updates: Dict[str, Any]
) -> None:
    """Merge control-plane fields onto the profile record.

    Raises :class:`ValueError` when ``updates`` contains any key outside
    :data:`_CP_ALLOWED_PROFILE_KEYS` — preserving the
    "MongoDB = control plane only" invariant.

    Writes are upserts so the first-time synthesis run creates the profile
    record.
    """
    if not updates:
        return
    unknown = set(updates) - _CP_ALLOWED_PROFILE_KEYS
    if unknown:
        raise ValueError(
            "update_profile_record: only control-plane keys allowed; "
            f"got disallowed: {sorted(unknown)!r}"
        )
    collection = _profiles_collection()
    if collection is None:
        raise RuntimeError("profiles collection unavailable")
    collection.update_one(
        {"subscription_id": subscription_id, "profile_id": profile_id},
        {"$set": dict(updates)},
        upsert=True,
    )
