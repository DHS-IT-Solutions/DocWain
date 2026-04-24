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

# Active extraction roster (per profile) — populated at batch start so the
# progress endpoint knows "N docs are queued for this run" before each one
# has been touched by the sequential loop.
_ROSTER_TTL = 2 * 60 * 60  # 2 hours
_ROSTER_KEY_FMT = "dw:extraction:roster:{profile_id}"
# Subscription-level "batch starting" marker published the instant a batch
# lock is acquired, *before* the eligible-doc query. Plugs the race where
# the progress endpoint would otherwise fall through to the historical
# view and report stale "100%" for the second or two between the user
# clicking Extract and the per-profile roster being published.
_BATCH_MARKER_TTL = 30 * 60  # 30 min — longer than any realistic batch
_BATCH_MARKER_KEY_FMT = "dw:extraction:batch_start:{subscription_id}"


def _redis_client_safe():
    try:
        from src.api.dw_newron import get_redis_client
        return get_redis_client()
    except Exception:
        return None


def set_extraction_roster(profile_id: str, doc_ids: List[str], subscription_id: Optional[str] = None) -> None:
    """Publish the current extraction batch's doc roster for *profile_id*.

    Called at batch start so ``/api/extract/progress`` can compute totals
    against the real queue size instead of whatever subset has already been
    persisted to Mongo.
    """
    if not profile_id or not doc_ids:
        return
    client = _redis_client_safe()
    if not client:
        return
    try:
        payload = {
            "profile_id": profile_id,
            "subscription_id": subscription_id,
            "doc_ids": [str(x) for x in doc_ids],
            "total": len(doc_ids),
            "started_at": time.time(),
        }
        client.setex(_ROSTER_KEY_FMT.format(profile_id=profile_id), _ROSTER_TTL, json.dumps(payload))
    except Exception:
        pass


def get_extraction_roster(profile_id: str) -> Optional[Dict[str, Any]]:
    if not profile_id:
        return None
    client = _redis_client_safe()
    if not client:
        return None
    try:
        raw = client.get(_ROSTER_KEY_FMT.format(profile_id=profile_id))
        if not raw:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="ignore")
        return json.loads(raw)
    except Exception:
        return None


def clear_extraction_roster(profile_id: str) -> None:
    if not profile_id:
        return
    client = _redis_client_safe()
    if not client:
        return
    try:
        client.delete(_ROSTER_KEY_FMT.format(profile_id=profile_id))
    except Exception:
        pass


def mark_batch_starting(subscription_id: str) -> None:
    """Publish a subscription-level 'batch is about to run' marker.

    Called the instant a batch lock is acquired, before we know which
    profiles are involved. The progress endpoint consults this marker to
    avoid reporting a stale historical view while the eligibility query
    is still computing.
    """
    if not subscription_id:
        return
    client = _redis_client_safe()
    if not client:
        return
    try:
        payload = {"subscription_id": str(subscription_id), "started_at": time.time()}
        client.setex(
            _BATCH_MARKER_KEY_FMT.format(subscription_id=subscription_id),
            _BATCH_MARKER_TTL,
            json.dumps(payload),
        )
    except Exception:
        pass


def get_batch_starting_marker(subscription_id: str) -> Optional[Dict[str, Any]]:
    if not subscription_id:
        return None
    client = _redis_client_safe()
    if not client:
        return None
    try:
        raw = client.get(_BATCH_MARKER_KEY_FMT.format(subscription_id=subscription_id))
        if not raw:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="ignore")
        return json.loads(raw)
    except Exception:
        return None


def clear_batch_starting_marker(subscription_id: str) -> None:
    if not subscription_id:
        return
    client = _redis_client_safe()
    if not client:
        return
    try:
        client.delete(_BATCH_MARKER_KEY_FMT.format(subscription_id=subscription_id))
    except Exception:
        pass


def clear_orphan_batch_state_on_startup() -> Dict[str, int]:
    """Drop any batch-extraction state left behind by a previous process.

    When the server is SIGKILL'd or restarted mid-batch, the Redis batch
    lock (``docwain:batch_extraction:<sub>``), the batch-starting marker
    (``dw:extraction:batch_start:<sub>``), and the per-profile rosters
    (``dw:extraction:roster:<profile>``) all persist — the lock blocks
    the next trigger for up to 30 min, and the marker/roster can show a
    stale "starting" view on the progress endpoint.

    Called once at application startup, when by definition no batch can
    be running. Returns the count of each key class dropped for logging.
    """
    cleared = {"batch_locks": 0, "markers": 0, "rosters": 0}
    client = _redis_client_safe()
    if not client:
        return cleared
    try:
        for pattern, label in (
            ("docwain:batch_extraction:*", "batch_locks"),
            ("dw:extraction:batch_start:*", "markers"),
            ("dw:extraction:roster:*", "rosters"),
        ):
            keys = list(client.keys(pattern) or [])
            if not keys:
                continue
            try:
                client.delete(*keys)
            except Exception:
                for k in keys:
                    try:
                        client.delete(k)
                    except Exception:
                        pass
            cleared[label] = len(keys)
    except Exception:
        logger.debug("Orphan batch-state cleanup failed", exc_info=True)
    return cleared

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


# ---------------------------------------------------------------------------
# In-flight status sets — control which docs contribute to overall_progress.
# Completed, failed, and out-of-stage docs appear in the per-doc list but
# are excluded from aggregate counters so the progress bar reflects real work.
# ---------------------------------------------------------------------------

# Extraction endpoint: only these statuses are "in flight"
_EXTRACT_IN_FLIGHT_STATUSES = frozenset({
    "UPLOADED",
    "EXTRACTION_IN_PROGRESS",
})
_EXTRACT_COMPLETED_STATUSES = frozenset({
    "EXTRACTION_COMPLETED",
    "AWAITING_REVIEW_1",
    "UNDER_REVIEW",
})
_EXTRACT_FAILED_STATUSES = frozenset({
    "EXTRACTION_FAILED",
})

# Training endpoint: only these statuses are "in flight"
_TRAIN_IN_FLIGHT_STATUSES = frozenset({
    "EMBEDDING_IN_PROGRESS",
    "TRAINING_STARTED",
})
_TRAIN_COMPLETED_STATUSES = frozenset({
    "TRAINING_COMPLETED",
    "TRAINING_PARTIALLY_COMPLETED",
})
_TRAIN_FAILED_STATUSES = frozenset({
    "TRAINING_FAILED",
    "TRAINING_BLOCKED_SECURITY",
    "TRAINING_BLOCKED_CONFIDENTIAL",
    "EMBEDDING_FAILED",
    "EXTRACTION_OR_CHUNKING_FAILED",
})

# KG status strand — independent from pipeline_status per Plan 3 isolation.
_KG_IN_FLIGHT_STATUSES = frozenset({"KG_PENDING", "KG_IN_PROGRESS"})
_KG_COMPLETED_STATUSES = frozenset({"KG_COMPLETED"})
_KG_FAILED_STATUSES = frozenset({"KG_FAILED"})

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


_BATCH_WINDOW_SECONDS = 2 * 60 * 60  # 2h — docs started within this window are "same run"


def _is_extraction_in_flight(doc: Dict[str, Any]) -> bool:
    """True iff the doc is actively being extracted right now."""
    extraction = doc.get("extraction") or {}
    status = doc.get("status", "UNKNOWN")
    nested = (
        extraction.get("status") == "IN_PROGRESS"
        and extraction.get("started_at") is not None
        and not extraction.get("completed_at")
    )
    return nested or status in _EXTRACT_IN_FLIGHT_STATUSES


def get_profile_extraction_status(profile_id: str) -> Dict[str, Any]:
    """Get comprehensive extraction status for all documents in a profile.

    The returned ``documents`` list carries every document in the profile so
    the UI can render the full table with per-doc progress bars. The
    aggregate in ``common_data`` is strictly **doc-count based**:

        overall_progress = completed / (queued + in_flight + completed + failed) * 100

    100% therefore means *every* uploaded doc has completed extraction
    successfully and is ready for screening — never a sub-stage percentage.
    Failed docs stay in the denominator so a partially-failed batch cannot
    climb to 100% without retry.

    Source of truth for the batch total is the Redis roster published by
    ``extract_documents`` at batch start (``set_extraction_roster``). That
    way a 10-doc batch reports 10 total from the very first poll, not
    1 (currently extracting) + whatever the sequential loop has reached.
    """
    from src.utils.logging_utils import get_live_logs

    collection = get_documents_collection()
    if collection is None:
        return {"documents": [], "common_data": {
            "Overall_live_logs": [], "overall_progress": 0,
            "total_documents": 0, "queued": 0, "in_flight": 0, "completed": 0, "failed": 0,
            "elapsed_time": "0s",
        }}

    docs = list(collection.find(
        {"profile_id": profile_id},
        {
            "document_id": 1, "status": 1, "source_file": 1,
            "extraction": 1, "screening": 1, "embedding": 1,
            "training_started_at": 1, "trained_at": 1,
            "created_at": 1, "updated_at": 1,
        },
    ).sort("updated_at", -1))

    now = time.time()
    roster = get_extraction_roster(profile_id)

    # Build quick lookup from doc_id to its record for roster-based classification.
    doc_by_id: Dict[str, Dict[str, Any]] = {}
    for d in docs:
        key = str(d.get("document_id") or d.get("_id"))
        doc_by_id[key] = d

    result_docs = []
    batch_queued = 0
    batch_in_flight = 0
    batch_completed = 0
    batch_failed = 0
    batch_earliest_start: Optional[float] = None
    batch_latest_end: Optional[float] = None

    # --- ROSTER PATH (active batch) ----------------------------------------
    # If a roster is present, classify each roster doc_id against its Mongo
    # record relative to the roster.started_at cutoff. A doc is only
    # "completed for this run" if its extraction.completed_at >= cutoff.
    roster_ids: Optional[List[str]] = None
    roster_started_at: Optional[float] = None
    if roster and isinstance(roster, dict):
        ids = roster.get("doc_ids") or []
        if ids:
            roster_ids = [str(x) for x in ids]
            ts = roster.get("started_at")
            roster_started_at = float(ts) if isinstance(ts, (int, float)) else None

    if roster_ids:
        for doc_id in roster_ids:
            d = doc_by_id.get(doc_id) or {}
            extraction = d.get("extraction") or {}
            start_at = extraction.get("started_at") if isinstance(extraction.get("started_at"), (int, float)) else None
            end_at = extraction.get("completed_at") if isinstance(extraction.get("completed_at"), (int, float)) else None
            ext_status = extraction.get("status")
            status = d.get("status", "UNKNOWN")
            cutoff = roster_started_at or 0

            in_progress_now = ext_status == "IN_PROGRESS" and (start_at or 0) >= cutoff and not (end_at and end_at >= cutoff)
            completed_now = bool(end_at and end_at >= cutoff and ext_status == "COMPLETED")
            failed_now = status in _EXTRACT_FAILED_STATUSES and (end_at or 0) >= cutoff

            if completed_now:
                batch_completed += 1
            elif failed_now:
                batch_failed += 1
            elif in_progress_now:
                batch_in_flight += 1
            else:
                # Roster member that hasn't been touched yet by this run.
                batch_queued += 1

            if start_at is not None and start_at >= cutoff:
                if batch_earliest_start is None or start_at < batch_earliest_start:
                    batch_earliest_start = start_at
            if end_at is not None and end_at >= cutoff:
                if batch_latest_end is None or end_at > batch_latest_end:
                    batch_latest_end = end_at

    # Always build the per-doc list (covers every doc in the profile, not
    # just the batch — UI renders this as the file table).
    for doc in docs:
        doc_id = str(doc.get("document_id") or doc.get("_id"))
        status = doc.get("status", "UNKNOWN")
        extraction = doc.get("extraction") or {}
        start_at = extraction.get("started_at") if isinstance(extraction.get("started_at"), (int, float)) else None
        end_at = extraction.get("completed_at") if isinstance(extraction.get("completed_at"), (int, float)) else None

        ext_in_progress = (
            extraction.get("status") == "IN_PROGRESS"
            and start_at is not None
            and not end_at
        )

        progress = _compute_document_progress(status, get_training_progress(doc_id))
        per_doc_progress = progress.get("progress", 0)
        if ext_in_progress and progress.get("source") == "derived" and per_doc_progress <= 0:
            per_doc_progress = _STATUS_TO_PROGRESS.get("EXTRACTION_IN_PROGRESS", 10)
            progress = {
                "progress": per_doc_progress,
                "stage": "extraction_in_progress",
                "detail": "Extraction In Progress",
                "source": "derived",
            }

        result_docs.append({
            "document_id": doc_id,
            "document_name": doc.get("source_file", ""),
            "progress": progress,
        })

    # --- FALLBACK PATH (no active roster) ----------------------------------
    # Two sub-cases that must be distinguished to avoid reporting stale
    # "100%" numbers while a fresh batch is spinning up:
    #
    #   (a) A batch-starting marker exists for the owning subscription.
    #       Extraction has just been triggered but the per-profile roster
    #       hasn't been published yet (race window). Report a clean
    #       "starting" view: every doc is in_flight or queued, progress 0,
    #       elapsed anchored on the marker's started_at.
    #
    #   (b) No marker either — truly idle or long-finished. Report raw
    #       per-doc state, but cap elapsed to the last recorded
    #       extraction end (or 0 if nothing's ever run) so the timer
    #       doesn't pretend a run is in progress.
    marker: Optional[Dict[str, Any]] = None
    # Track started_at of currently-active (in-flight / queued) docs
    # separately so idle-path elapsed doesn't pull from 27-hour-old history.
    batch_earliest_active_start: Optional[float] = None
    if not roster_ids:
        # Try to find the owning subscription from any doc in the profile.
        subscription_id: Optional[str] = None
        for d in docs:
            sid = d.get("subscription_id")
            if sid:
                subscription_id = str(sid)
                break
        marker = get_batch_starting_marker(subscription_id) if subscription_id else None

        for doc in docs:
            status = doc.get("status", "UNKNOWN")
            extraction = doc.get("extraction") or {}
            start_at = extraction.get("started_at") if isinstance(extraction.get("started_at"), (int, float)) else None
            end_at = extraction.get("completed_at") if isinstance(extraction.get("completed_at"), (int, float)) else None

            ext_in_progress = (
                extraction.get("status") == "IN_PROGRESS"
                and start_at is not None
                and not end_at
            )

            classified_active = False
            if marker:
                # Batch is starting. Classify conservatively: anything not
                # yet completed-after-marker is queued or in-flight.
                marker_start = float(marker.get("started_at", 0) or 0)
                if end_at and end_at >= marker_start and extraction.get("status") == "COMPLETED":
                    batch_completed += 1
                elif status in _EXTRACT_FAILED_STATUSES and (end_at or 0) >= marker_start:
                    batch_failed += 1
                elif ext_in_progress:
                    batch_in_flight += 1
                    classified_active = True
                else:
                    # Not yet touched by this run → queued.
                    batch_queued += 1
                    classified_active = True
            else:
                # Truly idle view — raw per-doc snapshot with no "run" semantics.
                # UPLOADED is the 'waiting for extraction' queue; only count
                # docs in EXTRACTION_IN_PROGRESS / nested-IN_PROGRESS as in-flight
                # so the progress bar moves step-wise as each doc actually
                # starts extracting.
                if status == "UPLOADED" and not ext_in_progress:
                    batch_queued += 1
                    classified_active = True
                elif ext_in_progress or status == "EXTRACTION_IN_PROGRESS":
                    batch_in_flight += 1
                    classified_active = True
                elif status in _EXTRACT_FAILED_STATUSES:
                    batch_failed += 1
                else:
                    batch_completed += 1

            if start_at is not None:
                if batch_earliest_start is None or start_at < batch_earliest_start:
                    batch_earliest_start = start_at
                if classified_active and (
                    batch_earliest_active_start is None or start_at < batch_earliest_active_start
                ):
                    batch_earliest_active_start = start_at
            if end_at is not None:
                if batch_latest_end is None or end_at > batch_latest_end:
                    batch_latest_end = end_at

    live_logs = get_live_logs(profile_id)

    total_docs = batch_queued + batch_in_flight + batch_completed + batch_failed
    if total_docs == 0:
        total_docs = len(result_docs)

    # Doc-count-based progress. Always clamp to total_docs > 0 before divide.
    if total_docs > 0:
        overall_progress = round((batch_completed / total_docs) * 100, 1)
    else:
        overall_progress = 0.0

    # ------------------------------------------------------------------
    # Resolve a single, unambiguous state + elapsed.
    # Semantics (what the UI can render without interpretation):
    #   running    — at least one doc is actively being extracted
    #   starting   — a batch was just triggered (lock held, roster not
    #                yet published) — timer ticks from lock acquisition
    #   idle       — no extraction active. elapsed_time is "0s" so the
    #                UI doesn't flash a historical timer.
    # Progress ticks gradually: completed / total * 100. Never 100% at
    # kickoff — at t=0 every roster doc is queued and the ratio is 0%.
    # ------------------------------------------------------------------
    batch_has_work = bool(batch_queued or batch_in_flight)
    if roster_ids:
        state = "running" if batch_has_work else "completed"
        elapsed_start: Optional[float] = roster_started_at
    elif marker:
        state = "starting"
        elapsed_start = float(marker.get("started_at", 0) or 0) or None
    elif batch_has_work:
        # No batch artefact but docs are actively extracting (upload-per-
        # doc flow via Celery, or a stale zombie). Show running with
        # elapsed anchored to the earliest active doc's started_at.
        state = "running"
        elapsed_start = batch_earliest_active_start
    else:
        state = "idle"
        elapsed_start = None

    elapsed_seconds: Optional[float] = None
    if elapsed_start:
        elapsed_seconds = round(now - elapsed_start, 1)

    return {
        "documents": result_docs,
        "common_data": {
            "Overall_live_logs": live_logs,
            "state": state,
            "overall_progress": overall_progress,
            "total_documents": total_docs,
            "queued": batch_queued,
            "in_flight": batch_in_flight,
            "completed": batch_completed,
            "failed": batch_failed,
            "elapsed_time": _format_elapsed(elapsed_seconds),
        },
    }


def get_profile_training_status(profile_id: str) -> Dict[str, Any]:
    """Get comprehensive training (embedding) status for all documents in a profile."""
    from src.utils.logging_utils import get_live_logs

    collection = get_documents_collection()
    if collection is None:
        return {"documents": [], "common_data": {
            "Overall_live_logs": [], "overall_progress": 0,
            "total_documents": 0, "in_flight": 0, "completed": 0, "failed": 0,
            "kg": {"in_flight": 0, "completed": 0, "failed": 0, "pending": 0},
            "elapsed_time": "0s",
        }}

    # Fetch all docs for this profile so that KG status (a parallel strand) is
    # always visible, and terminal-failure filtering is done in Python rather
    # than in the Mongo query (avoids accidental exclusion of KG-active docs).
    docs = list(collection.find(
        {"profile_id": profile_id},
        {
            "document_id": 1, "status": 1, "source_file": 1,
            "embedding": 1, "knowledge_graph": 1,
            "training_started_at": 1, "trained_at": 1,
            "created_at": 1, "updated_at": 1,
        },
    ).sort("updated_at", -1))

    now = time.time()
    result_docs = []
    # Pipeline aggregate counters — only in-flight contributes to overall_progress.
    in_flight_count = 0
    completed_count = 0
    failed_count = 0
    progress_sum = 0.0
    # KG counters — tracked independently from pipeline status.
    kg_in_flight_count = 0
    kg_completed_count = 0
    kg_failed_count = 0
    kg_pending_count = 0
    earliest_start: Optional[float] = None
    latest_end: Optional[float] = None

    for doc in docs:
        doc_id = str(doc.get("document_id") or doc.get("_id"))
        status = doc.get("status", "UNKNOWN")

        progress = _compute_document_progress(status, get_training_progress(doc_id))
        per_doc_progress = progress.get("progress", 0)

        # Classify pipeline status — only in-flight docs feed overall_progress.
        if status in _TRAIN_IN_FLIGHT_STATUSES:
            in_flight_count += 1
            progress_sum += per_doc_progress
        elif status in _TRAIN_COMPLETED_STATUSES:
            completed_count += 1
        elif status in _TRAIN_FAILED_STATUSES:
            failed_count += 1
        # Other statuses (e.g., SCREENING_COMPLETED awaiting trigger) are visible
        # in the per-doc list but excluded from aggregates.

        # Classify KG status — independent strand per Plan 3.
        kg_field = (doc.get("knowledge_graph") or {})
        kg_status = kg_field.get("status") if isinstance(kg_field, dict) else None
        if kg_status is None:
            kg_pending_count += 1
        elif kg_status in _KG_IN_FLIGHT_STATUSES:
            kg_in_flight_count += 1
        elif kg_status in _KG_COMPLETED_STATUSES:
            kg_completed_count += 1
        elif kg_status in _KG_FAILED_STATUSES:
            kg_failed_count += 1
        else:
            kg_pending_count += 1  # unknown value -> pending bucket

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

        result_docs.append({
            "document_id": doc_id,
            "document_name": doc.get("source_file", ""),
            "progress": progress,
        })

    # Live logs: actual terminal output captured by RedisLogHandler
    live_logs = get_live_logs(profile_id)

    total_docs = len(result_docs)
    if in_flight_count:
        overall_progress = round(progress_sum / in_flight_count, 1)
    elif completed_count and not failed_count:
        overall_progress = 100.0
    else:
        overall_progress = 0.0

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
            "in_flight": in_flight_count,
            "completed": completed_count,
            "failed": failed_count,
            "kg": {
                "in_flight": kg_in_flight_count,
                "completed": kg_completed_count,
                "failed": kg_failed_count,
                "pending": kg_pending_count,
            },
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
    """Reset documents stuck in extraction IN_PROGRESS back to PENDING.

    When the server is killed during extraction (or an unhandled exception
    skips the final state write), documents are left with
    ``extraction.status=IN_PROGRESS`` and no ``completed_at`` forever. This
    function detects those zombies — regardless of the top-level ``status``
    — and resets ``extraction`` so ``/api/extract/progress`` stops showing
    a phantom in-flight doc and the next extraction run can retry them.
    """
    collection = get_documents_collection()
    cutoff = time.time() - timeout_seconds
    zombies = list(collection.find(
        {
            # Cover every top-level status where a stuck extraction may land:
            #   UNDER_REVIEW (re-extract of previously-reviewed doc),
            #   EXTRACTION_IN_PROGRESS (fresh run),
            #   UPLOADED (pre-review extraction interrupted).
            "extraction.status": "IN_PROGRESS",
            "extraction.started_at": {"$lt": cutoff},
            "$or": [
                {"extraction.completed_at": {"$exists": False}},
                {"extraction.completed_at": None},
            ],
        },
        {"_id": 1, "document_id": 1, "status": 1, "extraction.started_at": 1},
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
            logger.info(
                "Reset zombie extraction %s (status=%s, stuck %.0fmin)",
                doc_id, doc.get("status"), minutes,
            )
        except Exception:
            logger.warning("Failed to reset zombie extraction %s", doc_id, exc_info=True)
    return recovered


# ---------------------------------------------------------------------------
# Periodic zombie sweep
# ---------------------------------------------------------------------------
# Zombie recovery at startup catches docs stuck across restarts, but a doc
# that gets orphaned mid-run (e.g. the worker thread dies or the vLLM call
# hangs past its timeout) stays flagged ``IN_PROGRESS`` forever because
# nothing else runs the sweep. For UAT we need a periodic pass so the
# progress endpoint never shows a doc that no process is working on.
_SWEEP_INTERVAL_SECONDS = 5 * 60  # wake every 5 min
_SWEEP_EXTRACTION_TIMEOUT_SECONDS = 10 * 60  # 10 min is ample for a live run
_SWEEP_TRAINING_TIMEOUT_SECONDS = 30 * 60  # training is heavier; 30 min

_sweep_stop_event: Optional[Any] = None
_sweep_thread: Optional[Any] = None


def _zombie_sweep_loop(stop_event, interval: int, ext_timeout: int, train_timeout: int) -> None:
    logger.info(
        "Zombie sweep worker started (interval=%ds, extraction_timeout=%ds, training_timeout=%ds)",
        interval, ext_timeout, train_timeout,
    )
    # Initial short delay so we don't race the startup recovery pass.
    stop_event.wait(min(interval, 60))
    while not stop_event.is_set():
        try:
            ext_recovered = recover_zombie_extractions(timeout_seconds=ext_timeout)
            train_recovered = recover_zombie_documents(timeout_seconds=train_timeout)
            if ext_recovered or train_recovered:
                logger.info(
                    "Zombie sweep: reset %d extraction, %d training",
                    ext_recovered, train_recovered,
                )
        except Exception:  # noqa: BLE001
            logger.warning("Zombie sweep iteration failed", exc_info=True)
        stop_event.wait(interval)
    logger.info("Zombie sweep worker stopped")


def start_zombie_sweep_worker(
    interval: int = _SWEEP_INTERVAL_SECONDS,
    extraction_timeout: int = _SWEEP_EXTRACTION_TIMEOUT_SECONDS,
    training_timeout: int = _SWEEP_TRAINING_TIMEOUT_SECONDS,
) -> None:
    """Spin a daemon thread that periodically resets stuck extractions/trainings.

    Idempotent — safe to call multiple times; subsequent calls are no-ops
    while the worker is already running.
    """
    import threading

    global _sweep_stop_event, _sweep_thread
    if _sweep_thread is not None and _sweep_thread.is_alive():
        return
    _sweep_stop_event = threading.Event()
    _sweep_thread = threading.Thread(
        target=_zombie_sweep_loop,
        args=(_sweep_stop_event, interval, extraction_timeout, training_timeout),
        name="docwain-zombie-sweep",
        daemon=True,
    )
    _sweep_thread.start()


def stop_zombie_sweep_worker(timeout: float = 5.0) -> None:
    """Signal the sweep worker to stop and join briefly."""
    global _sweep_stop_event, _sweep_thread
    if _sweep_stop_event is not None:
        _sweep_stop_event.set()
    if _sweep_thread is not None:
        _sweep_thread.join(timeout=timeout)
    _sweep_thread = None
    _sweep_stop_event = None

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
            flat[f"{stage}.summary"] = summary
        if blob_path is not None:
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
