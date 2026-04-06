def test_hitl_statuses_exist():
    from src.api import statuses
    assert statuses.PIPELINE_AWAITING_REVIEW_1 == "AWAITING_REVIEW_1"
    assert statuses.PIPELINE_AWAITING_REVIEW_2 == "AWAITING_REVIEW_2"
    assert statuses.PIPELINE_REJECTED == "REJECTED"
    assert statuses.PIPELINE_PROCESSING_IN_PROGRESS == "PROCESSING_IN_PROGRESS"
    assert statuses.PIPELINE_PROCESSING_COMPLETED == "PROCESSING_COMPLETED"
    assert statuses.PIPELINE_PROCESSING_FAILED == "PROCESSING_FAILED"
    assert "AWAITING_REVIEW_1" in statuses.ALL_STATUSES
    assert "AWAITING_REVIEW_2" in statuses.ALL_STATUSES
    assert "REJECTED" in statuses.ALL_STATUSES
    assert "PROCESSING_IN_PROGRESS" in statuses.ALL_STATUSES
    assert "PROCESSING_COMPLETED" in statuses.ALL_STATUSES
