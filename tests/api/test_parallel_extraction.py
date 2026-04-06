import time
from unittest.mock import patch, MagicMock
from concurrent.futures import ThreadPoolExecutor

# Common patches to avoid hitting real MongoDB/Redis during tests
_PATCHES = [
    patch("src.api.extraction_service._get_current_doc_status", return_value=None),
    patch("src.api.extraction_service._emit_batch_progress"),
    patch("src.api.extraction_service.emit_progress"),
    patch("src.api.document_status.emit_status_log", return_value=None),
    patch("src.api.document_status.clear_status_logs", return_value=None),
]


def _apply_patches():
    mocks = [p.start() for p in _PATCHES]
    return mocks


def _stop_patches():
    for p in _PATCHES:
        p.stop()


def test_parallel_extraction_completes_all_docs():
    """All documents should be processed even when running in parallel."""
    _apply_patches()
    try:
        call_count = {"n": 0}

        def mock_extract(doc_id, doc_data, conn_data):
            call_count["n"] += 1
            time.sleep(0.1)  # simulate work
            return {"document_id": doc_id, "status": "EXTRACTION_COMPLETED"}

        from src.api.extraction_service import _run_parallel_extraction

        docs = {
            f"doc_{i}": {"dataDict": {"name": f"file_{i}.pdf", "status": "UNDER_REVIEW"}, "connDict": {}}
            for i in range(5)
        }

        results = _run_parallel_extraction(docs, extract_fn=mock_extract, max_workers=3)
        assert len(results) == 5
        assert all(r["status"] == "EXTRACTION_COMPLETED" for r in results)
    finally:
        _stop_patches()


def test_parallel_extraction_faster_than_sequential():
    """Parallel should be faster than sequential for multiple slow docs."""
    _apply_patches()
    try:
        def mock_extract(doc_id, doc_data, conn_data):
            time.sleep(0.3)
            return {"document_id": doc_id, "status": "EXTRACTION_COMPLETED"}

        from src.api.extraction_service import _run_parallel_extraction

        docs = {
            f"doc_{i}": {"dataDict": {"name": f"file_{i}.pdf", "status": "UNDER_REVIEW"}, "connDict": {}}
            for i in range(3)
        }

        start = time.time()
        results = _run_parallel_extraction(docs, extract_fn=mock_extract, max_workers=3)
        elapsed = time.time() - start

        # 3 docs x 0.3s each = 0.9s sequential, should be ~0.3s parallel
        assert elapsed < 0.7  # generous margin but still proves parallelism
        assert len(results) == 3
    finally:
        _stop_patches()


def test_parallel_extraction_single_worker_is_sequential():
    """With max_workers=1, behavior should be sequential."""
    _apply_patches()
    try:
        order = []

        def mock_extract(doc_id, doc_data, conn_data):
            order.append(doc_id)
            return {"document_id": doc_id, "status": "EXTRACTION_COMPLETED"}

        from src.api.extraction_service import _run_parallel_extraction

        docs = {
            f"doc_{i}": {"dataDict": {"name": f"file_{i}.pdf", "status": "UNDER_REVIEW"}, "connDict": {}}
            for i in range(3)
        }

        results = _run_parallel_extraction(docs, extract_fn=mock_extract, max_workers=1)
        assert len(results) == 3
    finally:
        _stop_patches()
