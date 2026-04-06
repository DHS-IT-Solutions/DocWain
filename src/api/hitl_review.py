"""HITL (Human-in-the-Loop) review gate endpoints.

Provides approve/reject/reextract actions at two review gates:
  Gate 1: After extraction, before screening
  Gate 2: After screening, before processing (embed + KG)
"""

import time
from typing import Any, Dict

from src.api.document_status import update_document_fields, update_stage
from src.api.statuses import (
    PIPELINE_AWAITING_REVIEW_1,
    PIPELINE_AWAITING_REVIEW_2,
    PIPELINE_REJECTED,
    PIPELINE_SCREENING_IN_PROGRESS,
    PIPELINE_PROCESSING_IN_PROGRESS,
    PIPELINE_PROCESSING_FAILED,
    PIPELINE_SCREENING_FAILED,
    STATUS_UNDER_REVIEW,
)
from src.observability.logging import get_logger

logger = get_logger(__name__)


def _get_document_status(doc_id: str) -> str:
    """Query MongoDB for the current document status."""
    from src.api.document_status import get_document_record
    record = get_document_record(doc_id) or {}
    return record.get("status", "UNKNOWN")


def _trigger_screening(doc_id: str) -> None:
    """Update status to SCREENING_IN_PROGRESS and invoke screening."""
    update_document_fields(doc_id, {
        "status": PIPELINE_SCREENING_IN_PROGRESS,
        "screening_started_at": time.time(),
    })
    update_stage(doc_id, "screening", {
        "status": "IN_PROGRESS",
        "started_at": time.time(),
    })
    logger.info("Triggering screening for document %s", doc_id)
    try:
        from src.api.extraction_service import _run_auto_screening
        _run_auto_screening(doc_id)
    except Exception:
        logger.error("Screening failed for document %s", doc_id, exc_info=True)
        update_document_fields(doc_id, {
            "status": PIPELINE_SCREENING_FAILED,
        })
        update_stage(doc_id, "screening", {
            "status": "FAILED",
            "completed_at": time.time(),
        })


def _trigger_processing(doc_id: str) -> None:
    """Update status to PROCESSING_IN_PROGRESS and start processing.

    Currently falls through to the existing embedding pipeline.
    Phase 2b will add KG processing here.
    """
    update_document_fields(doc_id, {
        "status": PIPELINE_PROCESSING_IN_PROGRESS,
        "processing_started_at": time.time(),
    })
    logger.info("Triggering processing for document %s (placeholder — falls through to embedding)", doc_id)


def approve_review_gate_1(doc_id: str, reviewer: str) -> Dict[str, Any]:
    """Approve at gate 1: validates AWAITING_REVIEW_1, records reviewer, triggers screening."""
    current = _get_document_status(doc_id)
    if current != PIPELINE_AWAITING_REVIEW_1:
        logger.warning(
            "Cannot approve gate 1 for %s: expected AWAITING_REVIEW_1, got %s",
            doc_id, current,
        )
        return {
            "status": "error",
            "message": f"Document must be in AWAITING_REVIEW_1 to approve gate 1, currently: {current}",
        }
    update_document_fields(doc_id, {
        "review_1_approved_by": reviewer,
        "review_1_approved_at": time.time(),
    })
    _trigger_screening(doc_id)
    logger.info("Document %s approved at gate 1 by %s", doc_id, reviewer)
    return {"status": "approved", "gate": 1, "doc_id": doc_id, "reviewer": reviewer}


def approve_review_gate_2(doc_id: str, reviewer: str) -> Dict[str, Any]:
    """Approve at gate 2: validates AWAITING_REVIEW_2, records reviewer, triggers processing."""
    current = _get_document_status(doc_id)
    if current != PIPELINE_AWAITING_REVIEW_2:
        logger.warning(
            "Cannot approve gate 2 for %s: expected AWAITING_REVIEW_2, got %s",
            doc_id, current,
        )
        return {
            "status": "error",
            "message": f"Document must be in AWAITING_REVIEW_2 to approve gate 2, currently: {current}",
        }
    update_document_fields(doc_id, {
        "review_2_approved_by": reviewer,
        "review_2_approved_at": time.time(),
    })
    _trigger_processing(doc_id)
    logger.info("Document %s approved at gate 2 by %s", doc_id, reviewer)
    return {"status": "approved", "gate": 2, "doc_id": doc_id, "reviewer": reviewer}


def reject_document(doc_id: str, reviewer: str, reason: str) -> Dict[str, Any]:
    """Reject document at either review gate."""
    current = _get_document_status(doc_id)
    if current not in (PIPELINE_AWAITING_REVIEW_1, PIPELINE_AWAITING_REVIEW_2):
        logger.warning(
            "Cannot reject %s: expected AWAITING_REVIEW_1 or AWAITING_REVIEW_2, got %s",
            doc_id, current,
        )
        return {
            "status": "error",
            "message": f"Document must be in a review state to reject, currently: {current}",
        }
    update_document_fields(doc_id, {
        "status": PIPELINE_REJECTED,
        "rejected_by": reviewer,
        "rejected_at": time.time(),
        "rejection_reason": reason,
    })
    logger.info("Document %s rejected by %s: %s", doc_id, reviewer, reason)
    return {"status": "rejected", "doc_id": doc_id, "reviewer": reviewer, "reason": reason}


def request_reextraction(doc_id: str, reviewer: str, reason: str) -> Dict[str, Any]:
    """Request re-extraction: validates AWAITING_REVIEW_1, resets to UNDER_REVIEW."""
    current = _get_document_status(doc_id)
    if current != PIPELINE_AWAITING_REVIEW_1:
        logger.warning(
            "Cannot request reextraction for %s: expected AWAITING_REVIEW_1, got %s",
            doc_id, current,
        )
        return {
            "status": "error",
            "message": f"Document must be in AWAITING_REVIEW_1 to request reextraction, currently: {current}",
        }
    update_document_fields(doc_id, {
        "status": STATUS_UNDER_REVIEW,
        "reextraction_requested_by": reviewer,
        "reextraction_requested_at": time.time(),
        "reextraction_reason": reason,
    })
    update_stage(doc_id, "extraction", {
        "status": "PENDING",
        "started_at": None,
        "completed_at": None,
        "error": None,
    })
    logger.info("Document %s sent back for re-extraction by %s: %s", doc_id, reviewer, reason)
    return {"status": "reextraction_requested", "doc_id": doc_id, "reviewer": reviewer, "reason": reason}


def transition_to_awaiting_review_2(doc_id: str) -> None:
    """Transition document to AWAITING_REVIEW_2 after screening completes."""
    update_document_fields(doc_id, {
        "status": PIPELINE_AWAITING_REVIEW_2,
        "awaiting_review_2_at": time.time(),
    })
    update_stage(doc_id, "screening", {
        "status": "COMPLETED",
        "completed_at": time.time(),
        "awaiting_review": True,
    })
    logger.info("Document %s moved to AWAITING_REVIEW_2 (HITL gate 2)", doc_id)
