"""Tests for HITL review gate endpoints.

HITL review state is tracked in the ``hitl_review`` field, NOT in the main
``status`` field.  The main status (EXTRACTION_COMPLETED, SCREENING_COMPLETED,
TRAINING_COMPLETED) must never be overwritten by HITL logic.
"""

from unittest.mock import patch


def test_approve_review_1_transitions_to_screening():
    """Approving at gate 1 should trigger screening without changing main status."""
    from src.api.hitl_review import approve_review_gate_1, HITL_AWAITING_REVIEW_1
    from src.api.statuses import STATUS_EXTRACTION_COMPLETED

    with patch("src.api.hitl_review._get_document", return_value={
            "status": STATUS_EXTRACTION_COMPLETED, "hitl_review": HITL_AWAITING_REVIEW_1}), \
         patch("src.api.hitl_review.update_document_fields") as mock_update, \
         patch("src.api.hitl_review._trigger_screening") as mock_screen:

        result = approve_review_gate_1("doc_123", reviewer="muthu@docwain.com")
        assert result["status"] == "approved"
        mock_screen.assert_called_once_with("doc_123")
        # Main status must NOT be overwritten
        call_fields = mock_update.call_args[0][1]
        assert "status" not in call_fields
        assert call_fields["hitl_review"] == "APPROVED_1"


def test_approve_review_1_rejects_wrong_status():
    """Cannot approve if document is not EXTRACTION_COMPLETED."""
    from src.api.hitl_review import approve_review_gate_1

    with patch("src.api.hitl_review._get_document", return_value={
            "status": "EXTRACTION_IN_PROGRESS", "hitl_review": ""}):
        result = approve_review_gate_1("doc_123", reviewer="muthu@docwain.com")
        assert result["status"] == "error"


def test_reject_document():
    """Rejecting should set hitl_review=REJECTED, not change main status."""
    from src.api.hitl_review import reject_document, HITL_AWAITING_REVIEW_1, HITL_REJECTED

    with patch("src.api.hitl_review._get_document", return_value={
            "status": "EXTRACTION_COMPLETED", "hitl_review": HITL_AWAITING_REVIEW_1}), \
         patch("src.api.hitl_review.update_document_fields") as mock_update:

        result = reject_document("doc_123", reviewer="muthu@docwain.com", reason="Poor quality")
        assert result["status"] == "rejected"
        call_fields = mock_update.call_args[0][1]
        assert call_fields["hitl_review"] == HITL_REJECTED
        assert "status" not in call_fields


def test_approve_review_2_transitions_to_processing():
    """Approving at gate 2 should trigger processing without changing main status."""
    from src.api.hitl_review import approve_review_gate_2, HITL_AWAITING_REVIEW_2
    from src.api.statuses import STATUS_SCREENING_COMPLETED

    with patch("src.api.hitl_review._get_document", return_value={
            "status": STATUS_SCREENING_COMPLETED, "hitl_review": HITL_AWAITING_REVIEW_2}), \
         patch("src.api.hitl_review.update_document_fields") as mock_update, \
         patch("src.api.hitl_review._trigger_processing") as mock_proc:

        result = approve_review_gate_2("doc_123", reviewer="muthu@docwain.com")
        assert result["status"] == "approved"
        mock_proc.assert_called_once_with("doc_123")
        call_fields = mock_update.call_args[0][1]
        assert "status" not in call_fields


def test_screening_completion_transitions_to_awaiting_review_2():
    """After screening, hitl_review=AWAITING_REVIEW_2, main status stays SCREENING_COMPLETED."""
    from src.api.hitl_review import transition_to_awaiting_review_2, HITL_AWAITING_REVIEW_2

    with patch("src.api.hitl_review.update_document_fields") as mock_update:
        transition_to_awaiting_review_2("doc_456")
        call_fields = mock_update.call_args[0][1]
        assert call_fields["hitl_review"] == HITL_AWAITING_REVIEW_2
        assert "status" not in call_fields
