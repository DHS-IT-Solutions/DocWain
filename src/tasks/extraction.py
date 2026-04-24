"""Extraction pipeline Celery task."""

import json
import time

from celery.exceptions import SoftTimeLimitExceeded
from src.celery_app import app
from src.api.document_status import (
    get_document_record, update_stage, update_pipeline_status, append_audit_log
)
from src.api.statuses import (
    STAGE_IN_PROGRESS, STAGE_COMPLETED, STAGE_FAILED,
    PIPELINE_EXTRACTION_IN_PROGRESS, PIPELINE_EXTRACTION_COMPLETED,
    PIPELINE_EXTRACTION_FAILED
)
import logging

logger = logging.getLogger(__name__)


def _infer_format_hint(filename: str) -> str:
    import os
    _, ext = os.path.splitext(filename.lower())
    if ext == ".pdf":
        return "pdf_scanned"
    if ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        return "image"
    return "image"


def _get_redis_if_available():
    try:
        from src.api.dw_newron import get_redis_client
        return get_redis_client()
    except Exception:
        try:
            import redis
            return redis.Redis(host="localhost", port=6379, db=0,
                               decode_responses=True, socket_timeout=1.0)
        except Exception:
            return None


def _download_document_bytes(document_id: str, source_file: str) -> bytes:
    """Download raw document bytes from Azure Blob storage."""
    from src.api.blob_content_store import get_blob_client

    container = get_blob_client()
    blob_name = f"raw/{document_id}/{source_file}"
    blob_client = container.get_blob_client(blob_name)
    return blob_client.download_blob().readall()


def _extract_text_content(document_bytes: bytes, source_file: str,
                          content_type: str = "") -> str:
    """Use fileProcessor to extract basic text for the semantic pipeline."""
    from src.api.dataHandler import fileProcessor

    try:
        extracted = fileProcessor(document_bytes, source_file,
                                  content_type=content_type)
        # fileProcessor returns a dict keyed by filename/sheet
        # Combine all text values into a single string
        texts = []
        for key, value in extracted.items():
            if isinstance(value, str):
                texts.append(value)
            elif isinstance(value, dict):
                # ExtractedDocument or dict with text fields
                text = value.get("full_text") or value.get("text") or ""
                if text:
                    texts.append(str(text))
                else:
                    texts.append(str(value))
            elif isinstance(value, list):
                texts.append("\n".join(str(item) for item in value))
            else:
                # Dataclass or object with text attribute
                text = getattr(value, "full_text", None) or getattr(value, "text", None)
                if text:
                    texts.append(str(text))
                else:
                    texts.append(str(value))
        return "\n\n".join(texts)
    except Exception as exc:
        logger.warning("fileProcessor failed for %s: %s — proceeding with empty text",
                       source_file, exc)
        return ""


def _upload_extraction_json(subscription_id: str, profile_id: str,
                            document_id: str, result_dict: dict) -> str:
    """Upload full extraction result JSON to Azure Blob.

    Blob path: {subscription_id}/{profile_id}/{document_id}/extraction.json
    Returns the blob path.
    """
    from src.api.blob_content_store import get_blob_client
    from azure.storage.blob import ContentSettings

    container = get_blob_client()
    blob_path = f"{subscription_id}/{profile_id}/{document_id}/extraction.json"
    blob_client = container.get_blob_client(blob_path)

    payload = json.dumps(result_dict, default=str, ensure_ascii=False).encode("utf-8")
    blob_client.upload_blob(
        payload,
        overwrite=True,
        metadata={
            "docwain_artifact": "true",
            "document_id": document_id,
            "type": "extraction_result",
            "version": "v1",
        },
        content_settings=ContentSettings(content_type="application/json"),
    )
    logger.info("Uploaded extraction JSON: %s (%d bytes)", blob_path, len(payload))
    return blob_path


@app.task(bind=True, name="src.tasks.extraction.extract_document",
          max_retries=3, soft_time_limit=1500)
def extract_document(self, document_id: str, subscription_id: str,
                     profile_id: str):
    """Run three-model parallel extraction on an uploaded document.

    Models:
    - Triton: LayoutLM/DocFormer (structural)
    - Ollama DHS/DocWain:latest (semantic)
    - Ollama glm-ocr (vision)

    Results merged, stored to Azure Blob. Summary to MongoDB.
    """
    start_time = time.time()

    try:
        update_stage(document_id, "extraction", status=STAGE_IN_PROGRESS,
                     celery_task_id=self.request.id)
        update_pipeline_status(document_id, PIPELINE_EXTRACTION_IN_PROGRESS)
        append_audit_log(document_id, "EXTRACTION_STARTED",
                         celery_task_id=self.request.id)

        # 1. Get document record from MongoDB
        doc_record = get_document_record(document_id)
        if not doc_record:
            raise ValueError(f"Document record not found for {document_id}")

        source_file = doc_record.get("source_file") or doc_record.get("filename") or "document"
        file_type = doc_record.get("doc_type") or doc_record.get("file_type") or "pdf"
        content_type = doc_record.get("content_type") or ""

        # 2. Download document bytes from Azure Blob
        try:
            document_bytes = _download_document_bytes(document_id, source_file)
            logger.info("Downloaded document %s (%d bytes)", document_id, len(document_bytes))
        except Exception as exc:
            raise RuntimeError(f"Failed to download document from blob: {exc}") from exc

        # --- Plan 1: native-first dispatch ---
        try:
            from dataclasses import asdict as _dc_asdict
            from src.extraction.adapters.dispatcher import dispatch_native
            from src.extraction.adapters.errors import NotNativePathError
        except ImportError:
            dispatch_native = None  # type: ignore
            NotNativePathError = Exception  # type: ignore

        if dispatch_native is not None:
            try:
                _canonical = dispatch_native(document_bytes, filename=source_file, doc_id=document_id)
                _extraction_json = _dc_asdict(_canonical)
                # Normalize sheet cell tuple keys — dataclasses.asdict preserves tuple keys
                # which are not JSON-serialisable; stringify them.
                for _sheet in _extraction_json.get("sheets", []) or []:
                    _sheet["cells"] = {str(k): v for k, v in (_sheet.get("cells") or {}).items()}
                logger.info(
                    "[Plan1 native path] doc_id=%s format=%s pages=%d sheets=%d slides=%d",
                    document_id, _canonical.format,
                    len(_canonical.pages), len(_canonical.sheets), len(_canonical.slides),
                )
                _native_path_taken = True
                # Build result_dict as a direct ExtractionResult dict (no wrapper key).
                # subscription_id / profile_id are tenancy fields not on the dataclass;
                # merge them in as top-level siblings so the blob payload is self-describing.
                result_dict = {
                    **_extraction_json,
                    "subscription_id": subscription_id,
                    "profile_id": profile_id,
                }
                # Flatten tables for summary count across all pages/sheets/slides
                _table_count = (
                    sum(len(p.get("tables", [])) for p in _extraction_json.get("pages", []))
                    + sum(len(s.get("tables", [])) for s in _extraction_json.get("slides", []))
                )
                summary = {
                    "page_count": len(_canonical.pages) or len(_canonical.sheets) or len(_canonical.slides),
                    "entity_count": 0,
                    "section_count": 0,
                    "table_count": _table_count,
                    "doc_type_detected": _canonical.format,
                    "extraction_confidence": _canonical.metadata.coverage.verifier_score,
                    "models_used": ["native"],
                    "path_taken": "native",
                }
            except NotNativePathError as exc:
                logger.info("[Plan1] native path not applicable (%s); falling through to legacy engine", exc)
                _native_path_taken = False
            except Exception as exc:  # noqa: BLE001
                # Any failure in the native path must NOT break extraction — log and fall through.
                logger.warning("[Plan1] native path raised unexpected error: %r; falling through", exc)
                _native_path_taken = False
        else:
            _native_path_taken = False
        # --- end Plan 1 native-first dispatch ---

        # --- Plan 2: vision path (runs when native dispatch raised NotNativePathError) ---
        _vision_path_taken = False
        if not _native_path_taken:
            try:
                from dataclasses import asdict as _dc_asdict_vision
                from src.extraction.vision.orchestrator import extract_via_vision as _extract_via_vision

                _vision_result = _extract_via_vision(
                    document_bytes,
                    doc_id=document_id,
                    filename=source_file,
                    format_hint=_infer_format_hint(source_file),
                )
                _vision_dict = _dc_asdict_vision(_vision_result)
                # Normalize sheet cell tuple keys (vision path produces no sheets,
                # but the helper is defensive).
                for _sheet in _vision_dict.get("sheets", []) or []:
                    _sheet["cells"] = {str(k): v for k, v in (_sheet.get("cells") or {}).items()}
                result_dict = {
                    **_vision_dict,
                    "subscription_id": subscription_id,
                    "profile_id": profile_id,
                }
                _vision_table_count = sum(
                    len(p.get("tables", [])) for p in _vision_dict.get("pages", [])
                )
                summary = {
                    "page_count": len(_vision_result.pages),
                    "entity_count": 0,
                    "section_count": 0,
                    "table_count": _vision_table_count,
                    "doc_type_detected": _vision_result.format,
                    "extraction_confidence": _vision_result.metadata.coverage.verifier_score,
                    "models_used": ["vision"],
                    "path_taken": "vision",
                }
                _vision_path_taken = True
                logger.info(
                    "[Plan2 vision path] doc_id=%s pages=%d fallback_invocations=%d",
                    document_id,
                    len(_vision_result.pages),
                    len(_vision_result.metadata.coverage.fallback_invocations),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("[Plan2] vision path raised %r; falling through to legacy engine", exc)
                _vision_path_taken = False
        # --- end Plan 2 vision path ---

        if not _native_path_taken and not _vision_path_taken:
            # 3. Extract basic text using fileProcessor for the semantic pipeline
            text_content = _extract_text_content(document_bytes, source_file,
                                                 content_type=content_type)
            logger.info("Text extraction for %s: %d chars", document_id, len(text_content))

            # 4. Run ExtractionEngine.extract() with all three pipelines
            from src.extraction import ExtractionEngine

            engine = ExtractionEngine()
            result = engine.extract(
                document_id=document_id,
                subscription_id=subscription_id,
                profile_id=profile_id,
                document_bytes=document_bytes,
                file_type=file_type,
                text_content=text_content,
            )
            result_dict = result.to_dict()
            summary = result.to_summary()

        # 5. Store full extraction result JSON to Azure Blob
        blob_path = _upload_extraction_json(
            subscription_id, profile_id, document_id, result_dict
        )

        # 6. Store summary to MongoDB via update_stage()
        summary["blob_path"] = blob_path

        duration_seconds = round(time.time() - start_time, 2)
        summary["duration_seconds"] = duration_seconds

        update_stage(document_id, "extraction", status=STAGE_COMPLETED,
                     summary=summary, blob_path=blob_path, error=None)

        # 7. Update pipeline status to EXTRACTION_COMPLETED
        update_pipeline_status(document_id, PIPELINE_EXTRACTION_COMPLETED)
        append_audit_log(document_id, "EXTRACTION_COMPLETED",
                         duration_seconds=duration_seconds,
                         entity_count=summary.get("entity_count", 0),
                         table_count=summary.get("table_count", 0),
                         blob_path=blob_path)

        logger.info(
            "Extraction completed for %s in %.2fs: %d entities, %d tables, confidence=%.2f",
            document_id, duration_seconds,
            summary.get("entity_count", 0),
            summary.get("table_count", 0),
            summary.get("extraction_confidence", 0.0),
        )

        # Plan 2: per-extraction observability audit log.
        try:
            import time as _time
            from src.extraction.vision.observability import (
                ExtractionLogEntry,
                write_entry_if_redis,
            )
            _log_entry = ExtractionLogEntry(
                doc_id=document_id,
                format=str(result_dict.get("format", "unknown")),
                path_taken=str(result_dict.get("path_taken", "legacy")),
                timings_ms={},
                routing_decision=((result_dict.get("metadata") or {}).get("doc_intel") or {}),
                coverage_score=float(
                    ((result_dict.get("metadata") or {}).get("coverage") or {}).get(
                        "verifier_score", 1.0
                    )
                ),
                fallback_invocations=list(
                    ((result_dict.get("metadata") or {}).get("coverage") or {}).get(
                        "fallback_invocations", []
                    )
                ),
                human_review=False,
                completed_at=_time.time(),
            )
            write_entry_if_redis(redis_client=_get_redis_if_available(), entry=_log_entry)
        except Exception:
            logger.debug("observability log skipped", exc_info=True)

    except SoftTimeLimitExceeded:
        duration_seconds = round(time.time() - start_time, 2)
        error = {"message": "Extraction timed out", "code": "TIMEOUT"}
        update_stage(document_id, "extraction", status=STAGE_FAILED, error=error)
        update_pipeline_status(document_id, PIPELINE_EXTRACTION_FAILED)
        append_audit_log(document_id, "EXTRACTION_FAILED",
                         error="timeout", duration_seconds=duration_seconds)

    except Exception as exc:
        duration_seconds = round(time.time() - start_time, 2)
        error = {"message": str(exc), "code": "EXTRACTION_ERROR"}
        update_stage(document_id, "extraction", status=STAGE_FAILED, error=error)
        update_pipeline_status(document_id, PIPELINE_EXTRACTION_FAILED)
        append_audit_log(document_id, "EXTRACTION_FAILED",
                         error=str(exc), duration_seconds=duration_seconds)
        self.retry(exc=exc)
