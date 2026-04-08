"""Tests for multi-document and batch processing in standalone_multi."""
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# process_batch
# ---------------------------------------------------------------------------

@patch("src.api.standalone_multi.process_document")
def test_process_batch_returns_per_file_results(mock_process):
    mock_process.return_value = {
        "request_id": "req-1", "status": "completed", "answer": "Answer",
        "confidence": 0.9, "sources": [], "structured_output": None,
        "usage": {"total_ms": 1000},
    }
    files = [
        {"filename": "doc1.pdf", "content": b"%PDF-fake1"},
        {"filename": "doc2.pdf", "content": b"%PDF-fake2"},
    ]
    from src.api.standalone_multi import process_batch
    result = process_batch(files, prompt="Summarize", mode="qa", subscription_id="sub-1")
    assert result["status"] == "completed"
    assert len(result["results"]) == 2
    assert result["summary"]["total"] == 2
    assert result["summary"]["completed"] == 2
    assert result["summary"]["failed"] == 0
    assert result["batch_id"].startswith("batch-")


@patch("src.api.standalone_multi.process_document")
def test_process_batch_handles_errors(mock_process):
    mock_process.side_effect = [
        {"request_id": "r1", "status": "completed", "answer": "OK", "confidence": 0.9,
         "sources": [], "structured_output": None, "usage": {"total_ms": 500}},
        Exception("Extraction failed"),
    ]
    files = [
        {"filename": "ok.pdf", "content": b"%PDF-ok"},
        {"filename": "bad.pdf", "content": b"garbage"},
    ]
    from src.api.standalone_multi import process_batch
    result = process_batch(files, prompt="Summarize", mode="qa", subscription_id="sub-1")
    assert result["summary"]["completed"] == 1
    assert result["summary"]["failed"] == 1
    # results are sorted to match input order — ok.pdf is index 0, bad.pdf is index 1
    assert result["results"][1]["status"] == "error"
    assert "Extraction failed" in result["results"][1]["error"]


@patch("src.api.standalone_multi.process_document")
def test_process_batch_result_order_matches_input(mock_process):
    """Results must be sorted to match the input file order."""
    answers = ["Answer for A", "Answer for B", "Answer for C"]
    call_count = {"n": 0}

    def side_effect(file_dict, **kwargs):
        idx = call_count["n"]
        call_count["n"] += 1
        return {
            "request_id": f"req-{idx}", "status": "completed",
            "answer": answers[idx], "confidence": 0.8,
            "sources": [], "structured_output": None,
            "usage": {"total_ms": 100},
        }

    mock_process.side_effect = side_effect
    files = [
        {"filename": "a.pdf", "content": b"%PDF-a"},
        {"filename": "b.pdf", "content": b"%PDF-b"},
        {"filename": "c.pdf", "content": b"%PDF-c"},
    ]
    from src.api.standalone_multi import process_batch
    result = process_batch(files, prompt="Q", mode="qa", subscription_id="sub-x")
    filenames = [r["filename"] for r in result["results"]]
    assert filenames == ["a.pdf", "b.pdf", "c.pdf"]


@patch("src.api.standalone_multi.process_document")
def test_process_batch_empty_files(mock_process):
    from src.api.standalone_multi import process_batch
    result = process_batch([], prompt="Q", mode="qa", subscription_id="sub-1")
    assert result["status"] == "completed"
    assert result["results"] == []
    assert result["summary"]["total"] == 0
    assert result["summary"]["completed"] == 0
    assert result["summary"]["failed"] == 0
    mock_process.assert_not_called()


@patch("src.api.standalone_multi.process_document")
def test_process_batch_usage_field_present(mock_process):
    mock_process.return_value = {
        "request_id": "r1", "status": "completed", "answer": "A",
        "confidence": 0.7, "sources": [], "structured_output": None,
        "usage": {"total_ms": 200},
    }
    from src.api.standalone_multi import process_batch
    result = process_batch(
        [{"filename": "f.pdf", "content": b"%PDF-f"}],
        prompt="Q", mode="qa", subscription_id="sub-1",
    )
    assert "usage" in result
    assert "total_ms" in result["usage"]


# ---------------------------------------------------------------------------
# process_multi_documents
# ---------------------------------------------------------------------------

@patch("src.api.standalone_multi.cleanup_collection")
@patch("src.api.standalone_multi.retrieve_and_generate")
@patch("src.api.standalone_multi.chunk_and_embed")
@patch("src.api.standalone_multi.run_intelligence")
@patch("src.api.standalone_multi.extract_from_bytes")
def test_process_multi_merges_results(
    mock_extract, mock_intel, mock_embed, mock_retrieve, mock_cleanup
):
    mock_extract.return_value = {"text": "extracted text", "pages": 2}
    mock_intel.return_value = {"entities": [], "summary": "short summary"}
    mock_embed.return_value = None  # side-effect only
    mock_retrieve.return_value = {
        "request_id": "multi-req-1",
        "status": "completed",
        "answer": "Cross-doc answer",
        "confidence": 0.85,
        "sources": [{"document": "doc1.pdf", "page": 1}],
        "structured_output": None,
        "usage": {"total_ms": 750},
    }

    files = [
        {"filename": "doc1.pdf", "content": b"%PDF-doc1"},
        {"filename": "doc2.pdf", "content": b"%PDF-doc2"},
    ]
    from src.api.standalone_multi import process_multi_documents
    result = process_multi_documents(
        files=files,
        document_ids=None,
        prompt="What are the key findings?",
        mode="qa",
        subscription_id="sub-multi",
    )

    assert result["status"] == "completed"
    assert result["answer"] == "Cross-doc answer"
    assert result["confidence"] == 0.85
    # cleanup must be called to remove the shared temp collection
    mock_cleanup.assert_called_once()


@patch("src.api.standalone_multi.cleanup_collection")
@patch("src.api.standalone_multi.retrieve_and_generate")
@patch("src.api.standalone_multi.chunk_and_embed")
@patch("src.api.standalone_multi.run_intelligence")
@patch("src.api.standalone_multi.extract_from_bytes")
def test_process_multi_cleanup_on_exception(
    mock_extract, mock_intel, mock_embed, mock_retrieve, mock_cleanup
):
    """cleanup_collection must be called even when retrieve_and_generate raises."""
    mock_extract.return_value = {"text": "text", "pages": 1}
    mock_intel.return_value = {}
    mock_embed.return_value = None
    mock_retrieve.side_effect = RuntimeError("retrieval failed")

    from src.api.standalone_multi import process_multi_documents
    with pytest.raises(RuntimeError, match="retrieval failed"):
        process_multi_documents(
            files=[{"filename": "f.pdf", "content": b"%PDF-f"}],
            document_ids=None,
            prompt="Q",
            mode="qa",
            subscription_id="sub-1",
        )

    mock_cleanup.assert_called_once()


def test_process_multi_requires_files_or_ids():
    """Must raise ValueError when neither files nor document_ids are provided."""
    from src.api.standalone_multi import process_multi_documents
    with pytest.raises(ValueError, match="files.*document_ids|document_ids.*files"):
        process_multi_documents(
            files=None,
            document_ids=None,
            prompt="Q",
            mode="qa",
            subscription_id="sub-1",
        )


@patch("src.api.standalone_multi.cleanup_collection")
@patch("src.api.standalone_multi.retrieve_and_generate")
@patch("src.api.standalone_multi.chunk_and_embed")
@patch("src.api.standalone_multi.run_intelligence")
@patch("src.api.standalone_multi.extract_from_bytes")
def test_process_multi_shared_collection_name(
    mock_extract, mock_intel, mock_embed, mock_retrieve, mock_cleanup
):
    """The shared collection name must start with 'dw_standalone_multi_'."""
    mock_extract.return_value = {"text": "text", "pages": 1}
    mock_intel.return_value = {}
    mock_embed.return_value = None
    mock_retrieve.return_value = {
        "request_id": "r", "status": "completed", "answer": "A",
        "confidence": 0.5, "sources": [], "structured_output": None,
        "usage": {"total_ms": 0},
    }

    from src.api.standalone_multi import process_multi_documents
    process_multi_documents(
        files=[{"filename": "x.pdf", "content": b"%PDF-x"}],
        document_ids=None,
        prompt="Q",
        mode="qa",
        subscription_id="sub-1",
    )

    # chunk_and_embed should have been called with a collection starting with the prefix
    assert mock_embed.called
    call_kwargs = mock_embed.call_args
    # collection name may be positional or keyword
    all_args = list(call_kwargs.args) + list(call_kwargs.kwargs.values())
    collection_names = [a for a in all_args if isinstance(a, str) and a.startswith("dw_standalone_multi_")]
    assert len(collection_names) == 1, (
        f"Expected one collection name starting with 'dw_standalone_multi_', got args={call_kwargs}"
    )
