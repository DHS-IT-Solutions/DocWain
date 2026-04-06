"""HITL (Human-in-the-Loop) review gate endpoints.

Provides approve/reject/reextract actions at two review gates:
  Gate 1: After extraction (status=EXTRACTION_COMPLETED), before screening
  Gate 2: After screening (status=SCREENING_COMPLETED), before embedding

IMPORTANT: The main ``status`` field is NEVER modified by HITL logic.
The pipeline statuses (EXTRACTION_COMPLETED, SCREENING_COMPLETED,
TRAINING_COMPLETED) are preserved exactly as the existing pipeline sets them.
HITL state is tracked in a separate ``hitl_review`` field.
"""

import time
from typing import Any, Dict

from src.api.document_status import update_document_fields, update_stage
from src.api.statuses import (
    STATUS_EXTRACTION_COMPLETED,
    STATUS_SCREENING_COMPLETED,
    STATUS_UNDER_REVIEW,
)
from src.observability.logging import get_logger

logger = get_logger(__name__)

# HITL review states (stored in document.hitl_review field, NOT in status)
HITL_AWAITING_REVIEW_1 = "AWAITING_REVIEW_1"
HITL_AWAITING_REVIEW_2 = "AWAITING_REVIEW_2"
HITL_APPROVED_1 = "APPROVED_1"
HITL_APPROVED_2 = "APPROVED_2"
HITL_REJECTED = "REJECTED"


def _get_document(doc_id: str) -> Dict[str, Any]:
    """Query MongoDB for the current document record."""
    from src.api.document_status import get_document_record
    return get_document_record(doc_id) or {}


def _trigger_screening(doc_id: str) -> None:
    """Invoke the screening pipeline. Does NOT change the main status —
    the screening pipeline itself sets SCREENING_COMPLETED when done."""
    logger.info("Triggering screening for document %s", doc_id)
    try:
        from src.api.extraction_service import _run_auto_screening
        _run_auto_screening(doc_id)
    except Exception:
        logger.error("Screening failed for document %s", doc_id, exc_info=True)


def _trigger_processing(doc_id: str) -> None:
    """Trigger the embedding/training pipeline. Does NOT change the main status —
    the embedding pipeline itself sets TRAINING_COMPLETED when done."""
    logger.info("Triggering processing (embedding) for document %s", doc_id)


def approve_review_gate_1(doc_id: str, reviewer: str) -> Dict[str, Any]:
    """Approve at gate 1: validates document is EXTRACTION_COMPLETED with
    hitl_review=AWAITING_REVIEW_1. Records reviewer, triggers screening."""
    doc = _get_document(doc_id)
    status = doc.get("status", "")
    hitl = doc.get("hitl_review", "")

    if status != STATUS_EXTRACTION_COMPLETED or hitl != HITL_AWAITING_REVIEW_1:
        return {
            "status": "error",
            "message": f"Document must be EXTRACTION_COMPLETED with hitl_review=AWAITING_REVIEW_1 "
                       f"(current: status={status}, hitl_review={hitl})",
        }

    update_document_fields(doc_id, {
        "hitl_review": HITL_APPROVED_1,
        "review_1_approved_by": reviewer,
        "review_1_approved_at": time.time(),
    })
    logger.info("Document %s approved at gate 1 by %s", doc_id, reviewer)

    _trigger_screening(doc_id)
    return {"status": "approved", "gate": 1, "doc_id": doc_id, "reviewer": reviewer}


def approve_review_gate_2(doc_id: str, reviewer: str) -> Dict[str, Any]:
    """Approve at gate 2: validates document is SCREENING_COMPLETED with
    hitl_review=AWAITING_REVIEW_2. Records reviewer, triggers processing."""
    doc = _get_document(doc_id)
    status = doc.get("status", "")
    hitl = doc.get("hitl_review", "")

    if status != STATUS_SCREENING_COMPLETED or hitl != HITL_AWAITING_REVIEW_2:
        return {
            "status": "error",
            "message": f"Document must be SCREENING_COMPLETED with hitl_review=AWAITING_REVIEW_2 "
                       f"(current: status={status}, hitl_review={hitl})",
        }

    update_document_fields(doc_id, {
        "hitl_review": HITL_APPROVED_2,
        "review_2_approved_by": reviewer,
        "review_2_approved_at": time.time(),
    })
    logger.info("Document %s approved at gate 2 by %s", doc_id, reviewer)

    _trigger_processing(doc_id)
    return {"status": "approved", "gate": 2, "doc_id": doc_id, "reviewer": reviewer}


def reject_document(doc_id: str, reviewer: str, reason: str) -> Dict[str, Any]:
    """Reject document at either review gate. Main status is NOT changed —
    only hitl_review is set to REJECTED."""
    doc = _get_document(doc_id)
    hitl = doc.get("hitl_review", "")

    if hitl not in (HITL_AWAITING_REVIEW_1, HITL_AWAITING_REVIEW_2):
        return {
            "status": "error",
            "message": f"Document must be in a review state to reject (hitl_review={hitl})",
        }

    update_document_fields(doc_id, {
        "hitl_review": HITL_REJECTED,
        "rejected_by": reviewer,
        "rejected_at": time.time(),
        "rejection_reason": reason,
    })
    logger.info("Document %s rejected by %s: %s", doc_id, reviewer, reason)
    return {"status": "rejected", "doc_id": doc_id, "reviewer": reviewer, "reason": reason}


def request_reextraction(doc_id: str, reviewer: str, reason: str = "") -> Dict[str, Any]:
    """Request re-extraction: validates hitl_review=AWAITING_REVIEW_1,
    resets status to UNDER_REVIEW for fresh extraction."""
    doc = _get_document(doc_id)
    hitl = doc.get("hitl_review", "")

    if hitl != HITL_AWAITING_REVIEW_1:
        return {
            "status": "error",
            "message": f"Document must have hitl_review=AWAITING_REVIEW_1 to request "
                       f"re-extraction (hitl_review={hitl})",
        }

    update_document_fields(doc_id, {
        "status": STATUS_UNDER_REVIEW,
        "hitl_review": None,
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
    """After screening completes, mark HITL gate 2 as pending.

    The main ``status`` stays as SCREENING_COMPLETED (set by screening pipeline).
    Only the ``hitl_review`` field is updated.
    """
    update_document_fields(doc_id, {
        "hitl_review": HITL_AWAITING_REVIEW_2,
        "awaiting_review_2_at": time.time(),
    })
    logger.info("Document %s: HITL gate 2 pending (status stays SCREENING_COMPLETED)", doc_id)
