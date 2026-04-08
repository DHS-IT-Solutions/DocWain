"""Multi-document and batch processing for the DocWain Standalone API.

Provides two public functions:
  - process_batch: processes multiple files independently with the same prompt
    using a thread pool.
  - process_multi_documents: cross-document Q&A by ingesting all documents into
    a shared temporary Qdrant collection and running retrieval + generation
    across all of them.
"""
from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Deferred imports from standalone_processor.
# Defined at module level so unit tests can patch them with
# @patch("src.api.standalone_multi.<name>").
# The try/except keeps the module importable even before standalone_processor
# is available (e.g. during test collection).
# ---------------------------------------------------------------------------
try:
    from src.api.standalone_processor import (  # noqa: PLC0415
        chunk_and_embed,
        cleanup_collection,
        extract_from_bytes,
        process_document,
        retrieve_and_generate,
        run_intelligence,
        build_structured_prompt,
        _parse_structured_response,
        _capture_learning_signal,
    )
except (ImportError, ModuleNotFoundError):
    # Stubs — replaced by @patch in tests; raise clearly if called in prod.
    def _missing(name: str):  # type: ignore[return]
        def _stub(*_a, **_kw):
            raise ImportError(
                f"src.api.standalone_processor is not available "
                f"(required symbol: {name})"
            )
        _stub.__name__ = name
        return _stub

    chunk_and_embed = _missing("chunk_and_embed")  # type: ignore[assignment]
    cleanup_collection = _missing("cleanup_collection")  # type: ignore[assignment]
    extract_from_bytes = _missing("extract_from_bytes")  # type: ignore[assignment]
    process_document = _missing("process_document")  # type: ignore[assignment]
    retrieve_and_generate = _missing("retrieve_and_generate")  # type: ignore[assignment]
    run_intelligence = _missing("run_intelligence")  # type: ignore[assignment]
    build_structured_prompt = _missing("build_structured_prompt")  # type: ignore[assignment]
    _parse_structured_response = _missing("_parse_structured_response")  # type: ignore[assignment]
    _capture_learning_signal = _missing("_capture_learning_signal")  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def process_batch(
    files: List[Dict[str, Any]],
    prompt: str,
    mode: str = "qa",
    subscription_id: str = "standalone",
    output_format: str = "json",
    template: Optional[str] = None,
) -> Dict[str, Any]:
    """Process multiple files independently with the same prompt.

    Each file is processed via :func:`process_document` from
    ``standalone_processor``.  Work is dispatched with a
    ``ThreadPoolExecutor(max_workers=min(len(files), 4))`` and results are
    sorted to match the original input order.

    Args:
        files: List of dicts with ``filename`` and ``content`` keys.
        prompt: The question / instruction to apply to every file.
        mode: Processing mode — ``"qa"``, ``"table"``, ``"entities"``, or
            ``"summary"``.
        subscription_id: Subscription / tenant identifier.
        output_format: One of ``"json"``, ``"markdown"``, ``"csv"``,
            ``"html"``.
        template: Optional named template to apply.

    Returns:
        A dict with keys ``batch_id``, ``status``, ``results``,
        ``summary``, and ``usage``.
    """
    batch_id = f"batch-{uuid.uuid4()}"

    if not files:
        return {
            "batch_id": batch_id,
            "status": "completed",
            "results": [],
            "summary": {"total": 0, "completed": 0, "failed": 0},
            "usage": {"total_ms": 0},
        }

    max_workers = min(len(files), 4)
    # Keyed by original index so we can restore order after concurrent execution.
    future_to_index: Dict[Any, int] = {}
    ordered_results: List[Optional[Dict[str, Any]]] = [None] * len(files)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, file_dict in enumerate(files):
            future = executor.submit(
                process_document,
                file_dict,
                prompt=prompt,
                mode=mode,
                subscription_id=subscription_id,
                output_format=output_format,
                template=template,
            )
            future_to_index[future] = idx

        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            filename = files[idx].get("filename", f"file_{idx}")
            try:
                res = future.result()
                ordered_results[idx] = {
                    "filename": filename,
                    "status": res.get("status", "completed"),
                    "answer": res.get("answer"),
                    "confidence": res.get("confidence", 0.0),
                    "structured_output": res.get("structured_output"),
                    "error": None,
                }
            except Exception as exc:  # noqa: BLE001
                ordered_results[idx] = {
                    "filename": filename,
                    "status": "error",
                    "answer": None,
                    "confidence": 0.0,
                    "structured_output": None,
                    "error": str(exc),
                }

    results = [r for r in ordered_results if r is not None]
    completed = sum(1 for r in results if r["status"] == "completed")
    failed = sum(1 for r in results if r["status"] == "error")

    return {
        "batch_id": batch_id,
        "status": "completed",
        "results": results,
        "summary": {
            "total": len(results),
            "completed": completed,
            "failed": failed,
        },
        "usage": {"total_ms": 0},
    }


# ---------------------------------------------------------------------------
# Cross-document (multi-doc) Q&A
# ---------------------------------------------------------------------------

def process_multi_documents(
    files: Optional[List[Dict[str, Any]]],
    document_ids: Optional[List[str]],
    prompt: str,
    mode: str = "qa",
    subscription_id: str = "standalone",
    template: Optional[str] = None,
) -> Dict[str, Any]:
    """Cross-document Q&A across multiple uploaded files and/or persisted docs.

    All documents are ingested into a shared temporary Qdrant collection named
    ``dw_standalone_multi_{request_id}``.  After retrieval and generation the
    collection is cleaned up.

    Args:
        files: Optional list of dicts with ``filename`` and ``content`` keys.
        document_ids: Optional list of previously persisted document IDs.
        prompt: The question / instruction to run across all documents.
        mode: Processing mode.
        subscription_id: Subscription / tenant identifier.
        template: Optional named template.

    Returns:
        A dict matching ``ProcessResponse`` fields.

    Raises:
        ValueError: When neither ``files`` nor ``document_ids`` are provided.
    """
    if not files and not document_ids:
        raise ValueError(
            "At least one of files or document_ids must be provided."
        )

    request_id = str(uuid.uuid4())
    collection_name = f"dw_standalone_multi_{request_id}"

    try:
        # ------------------------------------------------------------------
        # Ingest uploaded files into the shared collection
        # ------------------------------------------------------------------
        if files:
            for file_dict in files:
                content: bytes = file_dict["content"]
                filename: str = file_dict.get("filename", "unknown")

                extraction = extract_from_bytes(content, filename=filename)
                intelligence = run_intelligence(extraction, mode=mode)
                chunk_and_embed(
                    extraction,
                    intelligence,
                    collection_name,
                    document_id=filename,
                    subscription_id=subscription_id,
                )

        # ------------------------------------------------------------------
        # Retrieve and generate across the shared collection
        # ------------------------------------------------------------------
        result = retrieve_and_generate(
            prompt=prompt,
            collection_name=collection_name,
            mode=mode,
            subscription_id=subscription_id,
            template=template,
        )

    finally:
        cleanup_collection(collection_name)

    return result
