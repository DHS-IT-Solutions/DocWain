"""Tests for HITL review gate endpoints."""

from unittest.mock import patch


def test_approve_review_1_transitions_to_screening():
    from src.api.hitl_review import approve_review_gate_1
    from src.api.statuses import PIPELINE_AWAITING_REVIEW_1
    with patch("src.api.hitl_review._get_document_status", return_value=PIPELINE_AWAITING_REVIEW_1), \
         patch("src.api.hitl_review.update_document_fields"), \
         patch("src.api.hitl_review._trigger_screening") as mock_screen:
        result = approve_review_gate_1("doc_123", reviewer="muthu@docwain.com")
        assert result["status"] == "approved"
        mock_screen.assert_called_once_with("doc_123")


def test_approve_review_1_rejects_wrong_status():
    from src.api.hitl_review import approve_review_gate_1
    with patch("src.api.hitl_review._get_document_status", return_value="EXTRACTION_IN_PROGRESS"):
        result = approve_review_gate_1("doc_123", reviewer="muthu@docwain.com")
        assert result["status"] == "error"
        assert "AWAITING_REVIEW_1" in result["message"]


def test_reject_document():
    from src.api.hitl_review import reject_document
    from src.api.statuses import PIPELINE_AWAITING_REVIEW_1, PIPELINE_REJECTED
    with patch("src.api.hitl_review._get_document_status", return_value=PIPELINE_AWAITING_REVIEW_1), \
         patch("src.api.hitl_review.update_document_fields") as mock_update:
        result = reject_document("doc_123", reviewer="muthu@docwain.com", reason="Poor quality")
        assert result["status"] == "rejected"
        call_fields = mock_update.call_args[0][1]
        assert call_fields["status"] == PIPELINE_REJECTED


def test_approve_review_2_transitions_to_processing():
    from src.api.hitl_review import approve_review_gate_2
    from src.api.statuses import PIPELINE_AWAITING_REVIEW_2
    with patch("src.api.hitl_review._get_document_status", return_value=PIPELINE_AWAITING_REVIEW_2), \
         patch("src.api.hitl_review.update_document_fields"), \
         patch("src.api.hitl_review._trigger_processing") as mock_proc:
        result = approve_review_gate_2("doc_123", reviewer="muthu@docwain.com")
        assert result["status"] == "approved"
        mock_proc.assert_called_once_with("doc_123")


