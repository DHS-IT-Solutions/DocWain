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


# Extraction-specific status → progress% mapping. Monotonic along the
# pipeline: once a document has moved past extraction, its contribution
# is 100% regardless of downstream stage. Pre-extraction (UNDER_REVIEW/
# UPLOADED) contributes 0, EXTRACTION_IN_PROGRESS contributes 50.
_EXTRACTION_PROGRESS_BY_STATUS = {
    "UNDER_REVIEW": 0,
    "UPLOADED": 0,
    "EXTRACTION_IN_PROGRESS": 50,
    "EXTRACTION_COMPLETED": 100,
    "EXTRACTION_FAILED": 0,
    "EXTRACTION_OR_CHUNKING_FAILED": 0,
    "SCREENING_IN_PROGRESS": 100,
    "SCREENING_COMPLETED": 100,
    "SCREENING_FAILED": 100,
    "TRAINING_STARTED": 100,
    "EMBEDDING_IN_PROGRESS": 100,
    "TRAINING_COMPLETED": 100,
    "TRAINING_PARTIALLY_COMPLETED": 100,
    "TRAINING_FAILED": 100,
    "TRAINING_BLOCKED_SECURITY": 0,
    "TRAINING_BLOCKED_CONFIDENTIAL": 0,
    "EMBEDDING_FAILED": 100,
    "AWAITING_REVIEW_1": 100,
    "AWAITING_REVIEW_2": 100,
    "REJECTED": 0,
    "PROCESSING_IN_PROGRESS": 100,
    "PROCESSING_COMPLETED": 100,
    "PROCESSING_FAILED": 100,
}

_UPLOAD_STATE_STATUSES = {"UNDER_REVIEW", "UPLOADED"}

_ACTIVE_EXTRACTION_STATUSES = {
    "UNDER_REVIEW",
    "UPLOADED",
    "EXTRACTION_IN_PROGRESS",
}

_EXTRACTION_FAILURE_STATUSES = {
    "EXTRACTION_FAILED",
    "EXTRACTION_OR_CHUNKING_FAILED",
}

# Documents uploaded within this many seconds of each other are treated as
# one batch. Wide enough to survive slow per-file uploads, narrow enough
# that a user returning after a break starts a fresh batch.
_BATCH_CLUSTER_WINDOW_SECONDS = 120


def _doc_status(doc: Dict[str, Any]) -> str:
    return doc.get("pipeline_status") or doc.get("status") or "UNKNOWN"


def _doc_created_at(doc: Dict[str, Any]) -> float:
    """Best-effort upload timestamp in epoch seconds.

    Different upload paths populate different fields:
    - ``init_document_record`` (Python)  → ``created_at`` (float epoch)
    - Node upload layer                   → ``createdAt`` (BSON datetime)
    - Legacy rows with neither            → fall back to ObjectId's
      generation time so clustering stays stable.
    """
    ts = doc.get("created_at")
    if isinstance(ts, (int, float)) and ts > 0:
        return float(ts)

    alt = doc.get("createdAt")
    if isinstance(alt, datetime):
        return alt.timestamp()
    if isinstance(alt, (int, float)) and alt > 0:
        return float(alt)

    oid = doc.get("_id")
    if isinstance(oid, ObjectId):
        return oid.generation_time.timestamp()
    return 0.0


def _doc_name(doc: Dict[str, Any]) -> str:
    """Prefer ``source_file`` (Python init path); fall back to ``name`` (Node path)."""
    return doc.get("source_file") or doc.get("name") or ""


def _identify_current_batch(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return the most recent upload batch from a list of profile documents.

    Two regimes:
    - If any doc is still actively extracting (UNDER_REVIEW / UPLOADED /
      EXTRACTION_IN_PROGRESS), anchor the batch at the earliest such doc's
      upload time and include everything uploaded within the cluster
      window before it — so freshly completed siblings stay in the view.
    - Otherwise walk the docs from most-recent backwards and stop at the
      first upload-time gap larger than the cluster window.
    """
    if not docs:
        return []

    sorted_docs = sorted(docs, key=_doc_created_at, reverse=True)

    active_docs = [d for d in sorted_docs if _doc_status(d) in _ACTIVE_EXTRACTION_STATUSES]
    if active_docs:
        earliest_active_ts = min(_doc_created_at(d) for d in active_docs)
        threshold = earliest_active_ts - _BATCH_CLUSTER_WINDOW_SECONDS
        return [d for d in sorted_docs if _doc_created_at(d) >= threshold]

    batch = [sorted_docs[0]]
    for i in range(1, len(sorted_docs)):
        prev_ts = _doc_created_at(sorted_docs[i - 1])
        curr_ts = _doc_created_at(sorted_docs[i])
        if prev_ts - curr_ts <= _BATCH_CLUSTER_WINDOW_SECONDS:
            batch.append(sorted_docs[i])
        else:
            break
    return batch


def get_profile_extraction_status(profile_id: str) -> Dict[str, Any]:
    """Get extraction progress for the current upload batch (UI-facing).

    Response shape::

        {
          "common_data": {
            "overall_live_logs": "No issues" | "<N document(s) failed extraction>",
            "elapsed_time": "23m 29s",
            "overall_progress": "47%",
            "total_documents": 5,
            "uploaded": 5,
          },
          "documents": [
            {"document_id": "...", "document_name": "..."},
            ...  # every document in the current upload batch
          ],
        }

    The endpoint reflects only the **current upload batch** — documents
    uploaded close together in time — so values don't regress when older
    profile documents (at various downstream stages) are mixed in.
    """
    collection = get_documents_collection()
    if collection is None:
        return {
            "common_data": {
                "overall_live_logs": [],
                "elapsed_time": "0s",
                "overall_progress": 0,
                "total_documents": 0,
                "uploaded": 0,
            },
            "documents": [],
        }

    # Project both snake_case (Python init path) and camelCase (Node upload
    # path) timestamp / name fields so batch detection works regardless of
    # which layer created the row.
    docs = list(collection.find(
        {"profile_id": profile_id},
        {
            "document_id": 1, "status": 1, "pipeline_status": 1,
            "source_file": 1, "name": 1, "extraction": 1,
            "created_at": 1, "createdAt": 1,
            "updated_at": 1, "updatedAt": 1,
        },
    ))

    batch = _identify_current_batch(docs)

    now = time.time()
    progress_sum = 0
    earliest_start: Optional[float] = None
    latest_end: Optional[float] = None
    batch_docs: List[Dict[str, Any]] = []
    failure_count = 0
    any_still_active = False

    for doc in batch:
        # Prefer pipeline_status (written by the tracked_stage decorator
        # and authoritative for the new pipeline). Fall back to the legacy
        # ``status`` field for older rows.
        status = _doc_status(doc)
        progress_sum += _EXTRACTION_PROGRESS_BY_STATUS.get(status, 0)

        # Upload-time anchors the user-facing "elapsed" clock — it reflects
        # "how long since I hit upload", not when Celery picked up the task.
        created_ts = _doc_created_at(doc)
        if created_ts > 0:
            if earliest_start is None or created_ts < earliest_start:
                earliest_start = created_ts

        extraction = doc.get("extraction") or {}
        end_at = extraction.get("completed_at")
        if isinstance(end_at, (int, float)):
            if latest_end is None or end_at > latest_end:
                latest_end = end_at

        if status in _EXTRACTION_FAILURE_STATUSES:
            failure_count += 1

        if status in _ACTIVE_EXTRACTION_STATUSES:
            any_still_active = True

        batch_docs.append({
            "document_id": str(doc.get("document_id") or doc.get("_id")),
            "document_name": _doc_name(doc),
        })

    batch_size = len(batch)
    overall_pct = int(progress_sum / batch_size) if batch_size else 0

    elapsed_seconds: Optional[float] = None
    if earliest_start:
        # While any doc is still extracting, tick against wall-clock; once the
        # batch finishes, freeze at the last completion so the UI stops moving.
        end_ref = now if any_still_active else (latest_end or now)
        elapsed_seconds = round(end_ref - earliest_start, 1)

    # ``overall_live_logs`` is deprecated — the field stays in the response
    # contract as an empty array so the UI keeps rendering without breaking.
    return {
        "common_data": {
            "overall_live_logs": [],
            "elapsed_time": _format_elapsed(elapsed_seconds),
            "overall_progress": overall_pct,
            "total_documents": batch_size,
            "uploaded": batch_size,
        },
        "documents": batch_docs,
    }


# Training-phase statuses the /train/progress endpoint considers.
_TRAINING_VISIBLE_STATUSES = {
    "SCREENING_COMPLETED", "TRAINING_STARTED", "TRAINING_COMPLETED",
    "TRAINING_FAILED", "TRAINING_PARTIALLY_COMPLETED",
    "TRAINING_BLOCKED_SECURITY", "TRAINING_BLOCKED_CONFIDENTIAL",
    "EXTRACTION_OR_CHUNKING_FAILED", "EMBEDDING_FAILED",
    "EMBEDDING_IN_PROGRESS",
}

_ACTIVE_TRAINING_STATUSES = {
    "SCREENING_COMPLETED",
    "TRAINING_STARTED",
    "EMBEDDING_IN_PROGRESS",
}

# Monotonic status → progress% for training, mirroring the extraction map.
_TRAINING_PROGRESS_BY_STATUS = {
    "SCREENING_COMPLETED": 0,
    "TRAINING_STARTED": 50,
    "EMBEDDING_IN_PROGRESS": 75,
    "TRAINING_COMPLETED": 100,
    "TRAINING_PARTIALLY_COMPLETED": 80,
    "TRAINING_FAILED": 0,
    "EMBEDDING_FAILED": 0,
    "TRAINING_BLOCKED_SECURITY": 0,
    "TRAINING_BLOCKED_CONFIDENTIAL": 0,
    "EXTRACTION_OR_CHUNKING_FAILED": 0,
}


def _doc_training_status(doc: Dict[str, Any]) -> str:
    """Pick the training-phase status for a doc.

    Prefer whichever of ``status`` / ``pipeline_status`` is already in the
    training-visible set — stale ``pipeline_status=EXTRACTION_COMPLETED``
    values are common in Mongo even after ``status=TRAINING_COMPLETED``.
    """
    status = doc.get("status") or ""
    pipeline_status = doc.get("pipeline_status") or ""
    if status in _TRAINING_VISIBLE_STATUSES:
        return status
    if pipeline_status in _TRAINING_VISIBLE_STATUSES:
        return pipeline_status
    return status or pipeline_status or "UNKNOWN"


def _doc_training_started_at(doc: Dict[str, Any]) -> float:
    """Return the training-start timestamp (epoch seconds).

    Priority:
    1. ``embedding.started_at`` (new pipeline-schema field)
    2. ``training_started_at`` (legacy field)
    3. upload time — so docs that are queued for training but not yet
       started still cluster together.
    """
    emb = doc.get("embedding") or {}
    ts = emb.get("started_at")
    if isinstance(ts, (int, float)) and ts > 0:
        return float(ts)
    ts = doc.get("training_started_at")
    if isinstance(ts, (int, float)) and ts > 0:
        return float(ts)
    return _doc_created_at(doc)


def _identify_current_training_batch(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return the most recent training batch via ``training_started_at`` clustering.

    Mirrors ``_identify_current_batch`` but uses the training-start timestamp
    so a batch = the set of docs the user triggered embedding for in one session,
    not necessarily the same docs as an earlier extraction batch.
    """
    if not docs:
        return []

    sorted_docs = sorted(docs, key=_doc_training_started_at, reverse=True)

    active_docs = [d for d in sorted_docs if _doc_training_status(d) in _ACTIVE_TRAINING_STATUSES]
    if active_docs:
        earliest_active_ts = min(_doc_training_started_at(d) for d in active_docs)
        threshold = earliest_active_ts - _BATCH_CLUSTER_WINDOW_SECONDS
        return [d for d in sorted_docs if _doc_training_started_at(d) >= threshold]

    batch = [sorted_docs[0]]
    for i in range(1, len(sorted_docs)):
        prev_ts = _doc_training_started_at(sorted_docs[i - 1])
        curr_ts = _doc_training_started_at(sorted_docs[i])
        if prev_ts - curr_ts <= _BATCH_CLUSTER_WINDOW_SECONDS:
            batch.append(sorted_docs[i])
        else:
            break
    return batch


def get_profile_training_status(profile_id: str) -> Dict[str, Any]:
    """Get training/embedding progress for the current training batch (UI-facing).

    Same response contract as ``get_profile_extraction_status`` — only a single
    training batch is reflected at a time, so values don't regress when older
    docs at downstream stages are mixed in. ``overall_live_logs`` is a
    deprecated placeholder that always returns ``[]``.
    """
    collection = get_documents_collection()
    if collection is None:
        return {
            "common_data": {
                "overall_live_logs": [],
                "elapsed_time": "0s",
                "overall_progress": 0,
                "total_documents": 0,
                "uploaded": 0,
            },
            "documents": [],
        }

    visible = list(_TRAINING_VISIBLE_STATUSES)
    docs = list(collection.find(
        {
            "profile_id": profile_id,
            "$or": [
                {"status": {"$in": visible}},
                {"pipeline_status": {"$in": visible}},
            ],
        },
        {
            "document_id": 1, "status": 1, "pipeline_status": 1,
            "source_file": 1, "name": 1, "embedding": 1,
            "training_started_at": 1, "trained_at": 1,
            "created_at": 1, "createdAt": 1,
            "updated_at": 1, "updatedAt": 1,
        },
    ))

    batch = _identify_current_training_batch(docs)

    now = time.time()
    progress_sum = 0
    earliest_start: Optional[float] = None
    latest_end: Optional[float] = None
    batch_docs: List[Dict[str, Any]] = []
    any_still_active = False

    for doc in batch:
        status = _doc_training_status(doc)
        progress_sum += _TRAINING_PROGRESS_BY_STATUS.get(status, 0)

        start_ts = _doc_training_started_at(doc)
        if start_ts > 0:
            if earliest_start is None or start_ts < earliest_start:
                earliest_start = start_ts

        embedding = doc.get("embedding") or {}
        end_at = doc.get("trained_at") or embedding.get("completed_at")
        if isinstance(end_at, (int, float)):
            if latest_end is None or end_at > latest_end:
                latest_end = end_at

        if status in _ACTIVE_TRAINING_STATUSES:
            any_still_active = True

        batch_docs.append({
            "document_id": str(doc.get("document_id") or doc.get("_id")),
            "document_name": _doc_name(doc),
        })

    batch_size = len(batch)
    overall_pct = int(progress_sum / batch_size) if batch_size else 0

    elapsed_seconds: Optional[float] = None
    if earliest_start:
        end_ref = now if any_still_active else (latest_end or now)
        elapsed_seconds = round(end_ref - earliest_start, 1)

    return {
        "common_data": {
            "overall_live_logs": [],
            "elapsed_time": _format_elapsed(elapsed_seconds),
            "overall_progress": overall_pct,
            "total_documents": batch_size,
            "uploaded": batch_size,
        },
        "documents": batch_docs,
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

    # Pipeline status and per-stage tracking
    update["pipeline_status"] = PIPELINE_UPLOADED
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

    # Keep pipeline_status aligned with status whenever a recognised
    # terminal status is written. Multiple call sites (screening_service,
    # embedding_service, extraction_service) transition status without
    # always setting pipeline_status; syncing here means all of them stay
    # consistent with the UI contract without duplicating the map. Do not
    # override an explicit pipeline_status the caller passed in.
    status_value = update.get("status")
    if status_value and "pipeline_status" not in update:
        _status_to_pipeline = {
            "EXTRACTION_COMPLETED": "EXTRACTION_COMPLETED",
            "EXTRACTION_FAILED": "EXTRACTION_FAILED",
            "SCREENING_COMPLETED": "SCREENING_COMPLETED",
            "EMBEDDING_COMPLETED": "TRAINING_COMPLETED",
            "EMBEDDING_FAILED": "EMBEDDING_FAILED",
            "TRAINING_COMPLETED": "TRAINING_COMPLETED",
            "TRAINING_FAILED": "EMBEDDING_FAILED",
        }
        mapped = _status_to_pipeline.get(str(status_value))
        if mapped:
            update["pipeline_status"] = mapped

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
    """Update the top-level pipeline status."""
    collection = get_documents_collection()
    now = time.time()
    existing = collection.find_one(_doc_filter(document_id), {"_id": 1})
    if existing:
        collection.update_one(
            {"_id": existing["_id"]},
            {"$set": {"pipeline_status": pipeline_status, "updated_at": now}},
        )
    else:
        collection.update_one(
            {"_id": _doc_id_value(document_id)},
            {"$set": {"pipeline_status": pipeline_status, "updated_at": now}},
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

# ---------------------------------------------------------------------------
# Control-plane helpers for SME Phase 2 (ERRATA §12)
#
# "MongoDB = control plane only" (project memory rule): these helpers read /
# write ONLY lightweight per-subscription and per-profile control fields. No
# document content, artifact content, or heavy payloads are ever persisted
# through this module — canonical artifacts live in Azure Blob, retrievable
# snippets in Qdrant, inferred edges in Neo4j.
#
# ``update_profile_record`` enforces an allowlist of control-plane-only keys
# so a misbehaving caller cannot sneak a ``narrative`` or payload blob into
# the profile record. Callers that need to persist heavy artifacts MUST use
# the Blob / Qdrant / Neo4j facade in ``src/intelligence/sme/storage.py``.
# ---------------------------------------------------------------------------
_SUBSCRIPTIONS_COLLECTION_NAME = "subscriptions"
_PROFILES_COLLECTION_NAME = "profiles"

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


def _documents_collection():
    """Return the Mongo documents collection, respecting a test patch."""
    return get_documents_collection()


def _subscriptions_collection():
    """Return the Mongo subscriptions collection.

    Separate helper so tests can patch this without touching the documents
    collection accessor.
    """
    from src.api.dataHandler import db

    return db[_SUBSCRIPTIONS_COLLECTION_NAME]


def _profiles_collection():
    """Return the Mongo profiles collection (control plane only)."""
    from src.api.dataHandler import db

    return db[_PROFILES_COLLECTION_NAME]


def count_incomplete_docs_in_profile(
    *, subscription_id: str, profile_id: str, exclude_document_id: str
) -> int:
    """Count docs in ``(subscription_id, profile_id)`` not yet at
    ``TRAINING_COMPLETED``, excluding ``exclude_document_id``.

    Used by Phase 2 Task 8 to gate last-doc-in-profile SME synthesis firing:
    when this count hits zero the last document is the one whose embedding
    just finished, so synthesis runs as the final training-stage step before
    ``PIPELINE_TRAINING_COMPLETED`` flips.
    """
    collection = _documents_collection()
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


def get_subscription_record(subscription_id: str) -> Optional[Dict[str, Any]]:
    """Return the subscription record or ``None`` if not present."""
    collection = _subscriptions_collection()
    if collection is None:
        return None
    return collection.find_one({"subscription_id": subscription_id})


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
