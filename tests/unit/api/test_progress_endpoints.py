"""Tests for `/api/extract/progress` and `/api/train/progress` aggregate correctness.

Bug being fixed: endpoints aggregated `overall_progress` over ALL docs in the
profile (including terminal completed + failed), producing percentages that
didn't reflect the in-flight work operators actually care about.

Also: `/api/train/progress` should surface KG progress (parallel track added
in Plan 3) as a separate `kg` breakdown in the response.
"""
from unittest.mock import MagicMock

import pytest


def _fake_doc(document_id, status, **extra):
    """Build a minimal document dict matching what the Mongo projection returns."""
    base = {
        "document_id": document_id,
        "status": status,
        "source_file": f"{document_id}.pdf",
        "extraction": {},
        "screening": {},
        "embedding": {},
        "created_at": 1700000000,
        "updated_at": 1700000100,
    }
    base.update(extra)
    return base


@pytest.fixture
def mock_documents_collection(monkeypatch):
    """Provide a single mocked Mongo documents collection for all tests."""
    fake_collection = MagicMock()

    def _patch(docs):
        """Configure the fake collection to return `docs` on .find()."""
        # find() is chained with .sort() in the production code.
        # Return a MagicMock that supports .sort() and itself iterates docs.
        find_result = MagicMock()
        find_result.__iter__ = lambda self: iter(docs)
        find_result.sort.return_value = find_result
        fake_collection.find.return_value = find_result

    # Patch the module-level get_documents_collection in document_status
    import src.api.document_status as ds_mod
    monkeypatch.setattr(ds_mod, "get_documents_collection",
                        lambda *a, **kw: fake_collection, raising=False)

    # Also patch Redis-based training progress to return None (no live data)
    monkeypatch.setattr(ds_mod, "get_training_progress",
                        lambda *a, **kw: None, raising=False)

    # Patch get_live_logs to avoid Redis calls
    try:
        from src.utils import logging_utils
        monkeypatch.setattr(logging_utils, "get_live_logs",
                            lambda *a, **kw: [], raising=False)
    except Exception:
        pass

    return _patch


def test_extract_progress_excludes_completed_and_failed_from_aggregate(mock_documents_collection):
    """Only in-flight extraction docs should contribute to overall_progress."""
    from src.api.document_status import get_profile_extraction_status

    # 5 docs: 2 in-flight, 1 completed extraction, 1 failed, 1 already in training
    docs = [
        _fake_doc("d1", "EXTRACTION_IN_PROGRESS"),
        _fake_doc("d2", "UPLOADED"),
        _fake_doc("d3", "EXTRACTION_COMPLETED"),     # completed -> skip from aggregate
        _fake_doc("d4", "EXTRACTION_FAILED"),        # failed -> skip from aggregate
        _fake_doc("d5", "TRAINING_COMPLETED"),       # past stage -> skip
    ]
    mock_documents_collection(docs)

    result = get_profile_extraction_status("prof-1")

    common = result.get("common_data") or {}
    # in_flight field should reflect the 2 actively-extracting docs
    assert common.get("in_flight") == 2, f"in_flight={common.get('in_flight')}, expected 2 (d1 + d2)"

    # overall_progress must be computed over only the 2 in-flight docs, not all 5.
    # With UPLOADED at 5% and EXTRACTION_IN_PROGRESS at 10% from _STATUS_TO_PROGRESS,
    # the average should be in (0, 50) -- i.e. reflecting in-flight work specifically,
    # not dragged by completed=100 docs nor by stale untouched docs.
    overall = common.get("overall_progress", -1)
    assert 0 < overall < 50, f"overall_progress={overall}, expected between 0 and 50 for only in-flight docs"

    # Typo fix: total_documents (no longer toatal_documents)
    assert "total_documents" in common
    assert common["total_documents"] == 5  # total across all docs in profile
    assert "toatal_documents" not in common


def test_extract_progress_returns_breakdown_counters(mock_documents_collection):
    """Response should expose completed / failed / in_flight counts."""
    from src.api.document_status import get_profile_extraction_status

    docs = [
        _fake_doc("d1", "EXTRACTION_IN_PROGRESS"),
        _fake_doc("d2", "EXTRACTION_COMPLETED"),
        _fake_doc("d3", "EXTRACTION_COMPLETED"),
        _fake_doc("d4", "EXTRACTION_FAILED"),
    ]
    mock_documents_collection(docs)

    result = get_profile_extraction_status("prof-1")
    common = result.get("common_data") or {}
    assert common.get("in_flight") == 1
    assert common.get("completed") == 2
    assert common.get("failed") == 1


def test_train_progress_excludes_terminal_states_from_aggregate(mock_documents_collection):
    """Only in-flight training/embedding docs should contribute to overall_progress."""
    from src.api.document_status import get_profile_training_status

    docs = [
        _fake_doc("t1", "EMBEDDING_IN_PROGRESS"),    # in-flight
        _fake_doc("t2", "TRAINING_FAILED"),          # terminal failure -> skip
        _fake_doc("t3", "TRAINING_COMPLETED"),       # terminal success -> skip from in-flight aggregate
        _fake_doc("t4", "EMBEDDING_FAILED"),         # terminal failure -> skip
    ]
    mock_documents_collection(docs)

    result = get_profile_training_status("prof-1")
    common = result.get("common_data") or {}

    assert common.get("in_flight") == 1, f"in_flight={common.get('in_flight')}, expected 1 (t1 only)"
    overall = common.get("overall_progress", -1)
    # Only t1 (in-flight) contributes; it's at EMBEDDING_IN_PROGRESS which is a mid-% value.
    # Must not be dragged to ~25% by averaging with terminal-state docs.
    assert 0 < overall < 100, f"overall_progress={overall}"
    assert "total_documents" in common
    assert "toatal_documents" not in common


def test_train_progress_exposes_kg_breakdown(mock_documents_collection):
    """Response should have a `kg` breakdown from knowledge_graph.status per doc."""
    from src.api.document_status import get_profile_training_status

    docs = [
        _fake_doc("t1", "EMBEDDING_IN_PROGRESS",
                  knowledge_graph={"status": "KG_IN_PROGRESS"}),
        _fake_doc("t2", "TRAINING_COMPLETED",
                  knowledge_graph={"status": "KG_COMPLETED"}),
        _fake_doc("t3", "TRAINING_COMPLETED",
                  knowledge_graph={"status": "KG_FAILED"}),
        _fake_doc("t4", "TRAINING_COMPLETED"),   # no KG subdoc -> pending
    ]
    mock_documents_collection(docs)

    result = get_profile_training_status("prof-1")
    common = result.get("common_data") or {}

    kg = common.get("kg")
    assert kg is not None, "common_data.kg missing -- KG visibility required in train progress"
    assert kg.get("in_flight") == 1, f"KG in_flight={kg.get('in_flight')}, expected 1 (t1)"
    assert kg.get("completed") == 1, f"KG completed={kg.get('completed')}, expected 1 (t2)"
    assert kg.get("failed") == 1, f"KG failed={kg.get('failed')}, expected 1 (t3)"
    assert kg.get("pending") == 1, f"KG pending={kg.get('pending')}, expected 1 (t4 -- no KG data yet)"
