"""Pipeline API endpoints — UI-triggered document processing stages.

Phase 2 adds :func:`finalize_training_for_doc` — the terminal step of the
training stage. When the last document in a profile completes embedding and
the subscription has ``enable_sme_synthesis`` on, this runs
:class:`src.intelligence.sme.synthesizer.SMESynthesizer` as the final
internal step before ``PIPELINE_TRAINING_COMPLETED`` fires (no new status
string per spec invariant). Failure keeps the doc in its current status so
retry is idempotent.

Phase 2 user Task 13 adds an input-hash short-circuit: if the profile's
``sme_last_input_hash`` equals the current inputs' hash, synthesis is
skipped (idempotent re-run). On synthesis success the new hash + run_id
are persisted on the profile record so the next re-entry can short-circuit.
A :func:`invalidate_qa_index` Redis hook runs after the
``PIPELINE_TRAINING_COMPLETED`` flip to clear the Phase-3 Q&A fast-path
cache (ERRATA §13).
"""

import logging
import uuid
from typing import Any, Callable, Iterable, Optional

from fastapi import APIRouter, HTTPException

from src.api.document_status import (
    append_audit_log,
    count_incomplete_docs_in_profile,
    get_document_record,
    get_profile_record,
    update_pipeline_status,
    update_profile_record,
)
from src.api.statuses import (
    PIPELINE_EXTRACTION_COMPLETED,
    PIPELINE_SCREENING_COMPLETED,
    PIPELINE_TRAINING_COMPLETED,
)
from src.config.feature_flags import (
    ENABLE_SME_SYNTHESIS,
    get_flag_resolver,
)
from src.intelligence.sme.input_hash import (
    compute_input_hash_for_profile,
    input_hash_unchanged,
)
from src.tasks.screening import screen_document
from src.tasks.embedding import embed_document

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SME synthesis hook — injected factory. Apps that wire the SMESynthesizer at
# startup call :func:`register_sme_synthesizer_factory`; callers / tests that
# want to skip synthesis leave the factory unset (default), in which case
# :func:`finalize_training_for_doc` flips PIPELINE_TRAINING_COMPLETED
# directly. This keeps the control plane test-double-able without adding a
# hard dep on the synthesizer at import time.
# ---------------------------------------------------------------------------
_SMESynthesizerFactory = Callable[[], Any]
_sme_synth_factory: Optional[_SMESynthesizerFactory] = None


def register_sme_synthesizer_factory(
    factory: Optional[_SMESynthesizerFactory],
) -> None:
    """Install / clear the factory that produces an ``SMESynthesizer``-like
    object for :func:`finalize_training_for_doc` to call.

    The object returned by ``factory()`` must expose a ``run`` method with
    signature ``(subscription_id, profile_id, profile_domain,
    synthesis_version)`` returning a ``dict[str, int]`` of per-artifact
    accepted counts, matching :class:`SMESynthesizer.run`.

    Passing ``None`` clears the registration (used by tests).
    """
    global _sme_synth_factory
    _sme_synth_factory = factory


# ---------------------------------------------------------------------------
# Profile-chunk reader hook — used by the input-hash short-circuit to avoid
# re-synthesising unchanged inputs. Apps wire a reader at startup; tests
# install a callable returning in-memory chunks so the finalize flow can be
# exercised without talking to Qdrant/Mongo. Default ``None`` disables the
# short-circuit (synthesis always fires) which is the safe fallback when the
# reader is not yet wired.
# ---------------------------------------------------------------------------
_ProfileChunksReader = Callable[[str, str], Iterable[dict[str, Any]]]
_profile_chunks_reader: Optional[_ProfileChunksReader] = None


def register_profile_chunks_reader(
    reader: Optional[_ProfileChunksReader],
) -> None:
    """Install / clear the profile-chunks reader.

    ``reader`` takes ``(subscription_id, profile_id)`` and returns an
    iterable of dicts with ``doc_id`` / ``chunk_id`` / ``text`` keys. The
    return value is hashed (stable under reorder) and compared against
    the profile record's ``sme_last_input_hash``; equal hashes skip
    synthesis. Passing ``None`` clears the registration.
    """
    global _profile_chunks_reader
    _profile_chunks_reader = reader


# ---------------------------------------------------------------------------
# Adapter-loader hook for the input-hash computation. The synthesiser keeps
# its own loader reference; this hook exists so the pipeline layer can read
# ``adapter.version`` + ``adapter.content_hash`` without duplicating the
# synthesis wiring. Tests install a lambda returning a :class:`types.
# SimpleNamespace` with ``version`` / ``content_hash`` attributes.
# ---------------------------------------------------------------------------
_AdapterFingerprintReader = Callable[[str, str], tuple[str, str] | None]
_adapter_fingerprint_reader: Optional[_AdapterFingerprintReader] = None


def register_adapter_fingerprint_reader(
    reader: Optional[_AdapterFingerprintReader],
) -> None:
    """Install / clear the adapter-fingerprint reader.

    ``reader`` takes ``(subscription_id, profile_domain)`` and returns a
    tuple ``(version, content_hash)`` for the adapter that would be used
    for synthesis. Returning ``None`` disables the short-circuit.
    """
    global _adapter_fingerprint_reader
    _adapter_fingerprint_reader = reader


def _is_last_doc_in_profile(doc: dict) -> bool:
    """Return True when the given doc is the LAST remaining doc in its
    profile that has not yet hit ``PIPELINE_TRAINING_COMPLETED``.

    Uses ``count_incomplete_docs_in_profile`` which excludes the doc itself,
    so when the count returns 0 every other doc in the profile has already
    finished training.
    """
    subscription_id = doc.get("subscription_id")
    profile_id = doc.get("profile_id")
    document_id = doc.get("document_id")
    if not subscription_id or not profile_id or not document_id:
        return False
    return (
        count_incomplete_docs_in_profile(
            subscription_id=subscription_id,
            profile_id=profile_id,
            exclude_document_id=document_id,
        )
        == 0
    )


def _next_synthesis_version(subscription_id: str, profile_id: str) -> int:
    """Increment ``sme_synthesis_version`` on the profile record by 1."""
    rec = get_profile_record(subscription_id, profile_id) or {}
    try:
        return int(rec.get("sme_synthesis_version", 0)) + 1
    except (TypeError, ValueError):
        return 1


def finalize_training_for_doc(doc: dict) -> None:
    """Terminal step of the training stage.

    Strict order:

    1. If the document is NOT the last in its profile, flip status to
       ``PIPELINE_TRAINING_COMPLETED`` immediately (per-doc completion —
       synthesis fires once per profile, on the final doc).
    2. If ``enable_sme_synthesis`` is off for the subscription, flip status
       immediately. No synthesis.
    3. If no synthesizer factory is registered (legacy / tests) flip status
       immediately. Synthesis is effectively disabled.
    4. Otherwise run the synthesizer; on success, append an audit entry +
       update the profile record's ``sme_synthesis_version`` + flip status.
       On failure, append ``SME_SYNTHESIS_FAILED`` and re-raise — the caller
       (``src/tasks/embedding.py``) decides retry semantics; status does
       NOT advance.
    """
    document_id = doc.get("document_id")
    if not document_id:
        raise ValueError("finalize_training_for_doc: document_id required")
    subscription_id = doc.get("subscription_id")
    profile_id = doc.get("profile_id")

    if not _is_last_doc_in_profile(doc):
        update_pipeline_status(document_id, PIPELINE_TRAINING_COMPLETED)
        return

    if not subscription_id or not profile_id:
        # Incomplete metadata — legacy row. Flip status, skip synthesis.
        update_pipeline_status(document_id, PIPELINE_TRAINING_COMPLETED)
        return

    try:
        flag_on = get_flag_resolver().is_enabled(
            subscription_id, ENABLE_SME_SYNTHESIS
        )
    except RuntimeError:
        # Flag resolver not initialised (pre-Phase-2 deploy) — synthesis off.
        flag_on = False
    if not flag_on:
        update_pipeline_status(document_id, PIPELINE_TRAINING_COMPLETED)
        return

    if _sme_synth_factory is None:
        # No synthesizer wired at startup — synthesis effectively off. We
        # still flip status so the pipeline does not stall.
        update_pipeline_status(document_id, PIPELINE_TRAINING_COMPLETED)
        return

    profile_rec = get_profile_record(subscription_id, profile_id) or {}
    profile_domain = str(profile_rec.get("profile_domain", "generic"))

    # Incremental short-circuit: if we can build the input hash *and* it
    # matches the profile's last-stored hash, skip synthesis entirely. The
    # short-circuit is best-effort: any failure to compute the hash (missing
    # reader, adapter load error) falls through to real synthesis so we
    # never skip when the short-circuit state is indeterminate.
    current_input_hash = _safe_compute_input_hash(
        subscription_id=subscription_id,
        profile_id=profile_id,
        profile_domain=profile_domain,
    )
    if current_input_hash and input_hash_unchanged(
        subscription_id=subscription_id,
        profile_id=profile_id,
        current_hash=current_input_hash,
    ):
        append_audit_log(
            document_id,
            "SME_SYNTHESIS_SKIPPED_INPUT_UNCHANGED",
            subscription_id=subscription_id,
            profile_id=profile_id,
            profile_domain=profile_domain,
            input_hash=current_input_hash,
        )
        update_pipeline_status(document_id, PIPELINE_TRAINING_COMPLETED)
        _safe_invalidate_qa_index(
            subscription_id=subscription_id, profile_id=profile_id
        )
        return

    synthesis_version = _next_synthesis_version(subscription_id, profile_id)
    run_id = str(uuid.uuid4())

    try:
        synth = _sme_synth_factory()
        report = synth.run(
            subscription_id=subscription_id,
            profile_id=profile_id,
            profile_domain=profile_domain,
            synthesis_version=synthesis_version,
        )
    except Exception as exc:  # noqa: BLE001
        append_audit_log(
            document_id,
            "SME_SYNTHESIS_FAILED",
            subscription_id=subscription_id,
            profile_id=profile_id,
            profile_domain=profile_domain,
            error=str(exc),
        )
        logger.exception(
            "SME synthesis failed for %s/%s (doc=%s)",
            subscription_id,
            profile_id,
            document_id,
        )
        # Status stays at its pre-finalize value; caller decides retry.
        raise

    # Record success on the profile record (control-plane only, allowlisted
    # keys) + audit log, then flip status.
    updates: dict[str, Any] = {
        "sme_synthesis_version": synthesis_version,
        "profile_domain": profile_domain,
        "sme_last_run_id": run_id,
    }
    if current_input_hash:
        updates["sme_last_input_hash"] = current_input_hash
    try:
        update_profile_record(subscription_id, profile_id, updates)
    except Exception:  # noqa: BLE001
        # Best-effort: the pipeline advance should not block on a control-
        # plane write failure; re-synthesis picks up via the input hash.
        logger.warning(
            "update_profile_record failed after successful synthesis; "
            "pipeline will still advance",
            exc_info=True,
        )
    append_audit_log(
        document_id,
        "SME_SYNTHESIS_COMPLETED",
        subscription_id=subscription_id,
        profile_id=profile_id,
        synthesis_version=synthesis_version,
        run_id=run_id,
        input_hash=current_input_hash,
        counts=report if isinstance(report, dict) else None,
    )
    update_pipeline_status(document_id, PIPELINE_TRAINING_COMPLETED)
    _safe_invalidate_qa_index(
        subscription_id=subscription_id, profile_id=profile_id
    )


def _safe_compute_input_hash(
    *, subscription_id: str, profile_id: str, profile_domain: str
) -> str | None:
    """Best-effort input-hash computation. Returns ``None`` when any input
    is missing so the caller falls through to real synthesis."""
    reader = _profile_chunks_reader
    fp_reader = _adapter_fingerprint_reader
    if reader is None or fp_reader is None:
        return None
    try:
        chunks = list(reader(subscription_id, profile_id))
    except Exception:  # noqa: BLE001
        logger.debug("profile chunks reader failed", exc_info=True)
        return None
    try:
        fp = fp_reader(subscription_id, profile_domain)
    except Exception:  # noqa: BLE001
        logger.debug("adapter fingerprint reader failed", exc_info=True)
        return None
    if not fp:
        return None
    adapter_version, adapter_content_hash = fp
    try:
        return compute_input_hash_for_profile(
            subscription_id=subscription_id,
            profile_id=profile_id,
            chunks=chunks,
            adapter_version=adapter_version,
            adapter_content_hash=adapter_content_hash,
        )
    except Exception:  # noqa: BLE001
        logger.debug("compute_input_hash_for_profile failed", exc_info=True)
        return None


def invalidate_qa_index(*, subscription_id: str, profile_id: str) -> None:
    """Delete all ``qa_idx:{sub}:{prof}:*`` Redis entries for a profile.

    Called from :func:`finalize_training_for_doc` after
    ``PIPELINE_TRAINING_COMPLETED`` fires so Phase 3's ``QAFastPath.lookup``
    (ERRATA §13) never serves a stale answer after re-synthesis. Best-
    effort: Redis failure logs + continues — the pipeline status flip is
    already persisted and must not be undone.
    """
    try:
        from src.api.dw_newron import get_redis_client
        client = get_redis_client()
    except Exception:  # noqa: BLE001
        logger.debug("qa_idx invalidate: redis client unavailable", exc_info=True)
        return
    if client is None:
        return
    pattern = f"qa_idx:{subscription_id}:{profile_id}:*"
    try:
        for key in client.scan_iter(pattern):
            client.delete(key)
    except Exception:  # noqa: BLE001
        logger.warning(
            "qa_idx invalidate scan/delete failed for %s/%s",
            subscription_id,
            profile_id,
            exc_info=True,
        )


def _safe_invalidate_qa_index(*, subscription_id: str, profile_id: str) -> None:
    try:
        invalidate_qa_index(
            subscription_id=subscription_id, profile_id=profile_id
        )
    except Exception:  # noqa: BLE001
        logger.debug("qa_idx invalidation failed; ignoring", exc_info=True)

pipeline_router = APIRouter(prefix="/documents", tags=["pipeline"])


@pipeline_router.get("/{document_id}/status")
async def get_document_status(document_id: str):
    """Get pipeline status and per-stage summaries for a document."""
    record = get_document_record(document_id)
    if not record:
        raise HTTPException(status_code=404, detail="Document not found")

    return {
        "document_id": record.get("document_id"),
        "pipeline_status": record.get("pipeline_status"),
        "extraction": {
            "status": record.get("extraction", {}).get("status"),
            "summary": record.get("extraction", {}).get("summary"),
        },
        "screening": {
            "status": record.get("screening", {}).get("status"),
            "summary": record.get("screening", {}).get("summary"),
        },
        "knowledge_graph": {
            "status": record.get("knowledge_graph", {}).get("status"),
            "node_count": record.get("knowledge_graph", {}).get("node_count", 0),
            "edge_count": record.get("knowledge_graph", {}).get("edge_count", 0),
        },
        "embedding": {
            "status": record.get("embedding", {}).get("status"),
            "summary": record.get("embedding", {}).get("summary"),
        }
    }


@pipeline_router.get("/{document_id}/extraction")
async def get_extraction_summary(document_id: str):
    """Get extraction summary from MongoDB (for UI review)."""
    record = get_document_record(document_id)
    if not record:
        raise HTTPException(status_code=404, detail="Document not found")
    return {
        "document_id": document_id,
        "extraction": record.get("extraction", {})
    }


@pipeline_router.get("/{document_id}/extraction/detail")
async def get_extraction_detail(document_id: str):
    """Fetch full extraction JSON from Azure Blob."""
    record = get_document_record(document_id)
    if not record:
        raise HTTPException(status_code=404, detail="Document not found")

    blob_path = (record.get("extraction", {}).get("summary") or {}).get("blob_path")
    if not blob_path:
        raise HTTPException(status_code=404, detail="Extraction data not available")

    # TODO: Load from Azure Blob using blob_path
    raise HTTPException(status_code=501, detail="Azure Blob fetch not yet implemented")


@pipeline_router.post("/{document_id}/screen")
async def trigger_screening(document_id: str):
    """HITL trigger: user approved extraction, start screening."""
    record = get_document_record(document_id)
    if not record:
        raise HTTPException(status_code=404, detail="Document not found")

    # Accept either new pipeline_status or legacy status field
    pipeline_status = record.get("pipeline_status", "")
    legacy_status = record.get("status", "")
    extraction_stage = record.get("extraction", {}).get("status", "")
    extraction_done = (
        pipeline_status == PIPELINE_EXTRACTION_COMPLETED
        or legacy_status == "EXTRACTION_COMPLETED"
        or extraction_stage == "COMPLETED"
    )
    if not extraction_done:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot screen: extraction not complete "
                   f"(pipeline_status={pipeline_status}, status={legacy_status}, "
                   f"extraction.status={extraction_stage})"
        )

    subscription_id = record["subscription_id"]
    profile_id = record["profile_id"]

    # Try Celery, fallback to sync screening via existing gateway
    task_id = None
    mode = "queued"
    try:
        task = screen_document.delay(document_id, subscription_id, profile_id)
        task_id = task.id
    except Exception:
        mode = "sync"
        try:
            from src.api.extraction_service import _run_auto_screening
            _run_auto_screening(document_id, doc_type=record.get("doc_type"))
        except Exception as exc:
            logger.error("Sync screening failed for doc=%s: %s", document_id, exc)
            mode = "failed"

    append_audit_log(document_id, "SCREENING_TRIGGERED", by="user",
                    celery_task_id=task_id)

    return {"document_id": document_id, "status": "SCREENING_IN_PROGRESS",
            "task_id": task_id, "mode": mode}


@pipeline_router.get("/{document_id}/screening")
async def get_screening_summary(document_id: str):
    """Get screening summary from MongoDB (for UI review)."""
    record = get_document_record(document_id)
    if not record:
        raise HTTPException(status_code=404, detail="Document not found")
    return {
        "document_id": document_id,
        "screening": record.get("screening", {})
    }


@pipeline_router.get("/{document_id}/screening/detail")
async def get_screening_detail(document_id: str):
    """Fetch full screening report from Azure Blob."""
    record = get_document_record(document_id)
    if not record:
        raise HTTPException(status_code=404, detail="Document not found")

    blob_path = (record.get("screening", {}).get("summary") or {}).get("blob_path")
    if not blob_path:
        raise HTTPException(status_code=404, detail="Screening data not available")

    raise HTTPException(status_code=501, detail="Azure Blob fetch not yet implemented")


@pipeline_router.post("/{document_id}/embed")
async def trigger_embedding(document_id: str):
    """HITL trigger: user approved screening, start embedding."""
    record = get_document_record(document_id)
    if not record:
        raise HTTPException(status_code=404, detail="Document not found")

    # Accept either new pipeline_status or legacy status field
    pipeline_status = record.get("pipeline_status", "")
    legacy_status = record.get("status", "")
    screening_stage = record.get("screening", {}).get("status", "")
    screening_done = (
        pipeline_status == PIPELINE_SCREENING_COMPLETED
        or legacy_status == "SCREENING_COMPLETED"
        or screening_stage == "COMPLETED"
    )
    if not screening_done:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot embed: screening not complete "
                   f"(pipeline_status={pipeline_status}, screening.status={screening_stage})"
        )

    subscription_id = record["subscription_id"]
    profile_id = record["profile_id"]

    task = embed_document.delay(document_id, subscription_id, profile_id)
    append_audit_log(document_id, "EMBEDDING_TRIGGERED", by="user",
                    celery_task_id=task.id)

    return {"document_id": document_id, "status": "EMBEDDING_IN_PROGRESS",
            "task_id": task.id}


@pipeline_router.get("/{document_id}/kg/status")
async def get_kg_status(document_id: str):
    """Get KG build status (independent of pipeline)."""
    record = get_document_record(document_id)
    if not record:
        raise HTTPException(status_code=404, detail="Document not found")
    return {
        "document_id": document_id,
        "knowledge_graph": record.get("knowledge_graph", {})
    }
