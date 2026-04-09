"""FastAPI router for the DocWain Standalone API.

Provides one-shot document processing, batch processing, structured
extraction, document persistence, querying and usage audit endpoints
under the ``/v1/docwain`` prefix.
"""
from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile

from src.api.config import Config
from src.api.standalone_auth import require_api_key, track_document_processed, track_usage
from src.api.standalone_schemas import (
    ApiKeyCreateRequest,
    ApiKeyCreateResponse,
    AsyncAcceptedResponse,
    BatchResponse,
    DocumentStatusResponse,
    DocumentUploadResponse,
    ExtractResponse,
    ProcessResponse,
    ProcessedResultResponse,
    ProcessedResultsListResponse,
    TemplateInfo,
    TemplatesResponse,
    UsageResponse,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

standalone_router = APIRouter(prefix="/v1/docwain")

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

_VALID_MODES = {"qa", "table", "entities", "summary"}
_VALID_OUTPUT_FORMATS = {"json", "markdown", "csv", "html"}


def _validate_mode(mode: str) -> None:
    if mode not in _VALID_MODES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid mode {mode!r}. Must be one of {sorted(_VALID_MODES)}.",
        )


def _validate_output_format(fmt: str) -> None:
    if fmt not in _VALID_OUTPUT_FORMATS:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid output_format {fmt!r}. Must be one of {sorted(_VALID_OUTPUT_FORMATS)}.",
        )


def _validate_file_size(content: bytes) -> None:
    max_bytes = Config.Standalone.MAX_FILE_SIZE_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File size {len(content) / (1024 * 1024):.1f} MB exceeds "
                f"the {Config.Standalone.MAX_FILE_SIZE_MB} MB limit."
            ),
        )


# ---------------------------------------------------------------------------
# MongoDB helpers
# ---------------------------------------------------------------------------

def _get_db():
    """Return the MongoDB database used for standalone request records."""
    from pymongo import MongoClient

    client = MongoClient(Config.MongoDB.URI)
    return client[Config.MongoDB.DB]


def _save_document_record(
    document_id: str,
    name: Optional[str],
    subscription_id: str,
    key_hash: str,
    pages: int,
    document_type: str,
) -> None:
    """Insert a document record into the standalone_requests collection."""
    try:
        db = _get_db()
        collection = db[Config.Standalone.REQUESTS_COLLECTION]
        now = datetime.now(tz=timezone.utc).isoformat()
        collection.insert_one(
            {
                "document_id": document_id,
                "name": name or document_id,
                "subscription_id": subscription_id,
                "key_hash": key_hash,
                "pages": pages,
                "document_type": document_type,
                "status": "ready",
                "created_at": now,
                "ready_at": now,
            }
        )
    except Exception:
        logger.exception("_save_document_record failed for document_id=%s", document_id)


def _get_document_record(document_id: str, subscription_id: str) -> Optional[Dict[str, Any]]:
    """Return a document record from MongoDB, or None if not found."""
    try:
        db = _get_db()
        collection = db[Config.Standalone.REQUESTS_COLLECTION]
        return collection.find_one(
            {"document_id": document_id, "subscription_id": subscription_id},
            {"_id": 0},
        )
    except Exception:
        logger.exception("_get_document_record failed for document_id=%s", document_id)
        return None


def _log_request(
    api_key: Dict[str, Any],
    endpoint: str,
    mode: str,
    result: Dict[str, Any],
    filename: Optional[str] = None,
) -> None:
    """Store the full processing result in MongoDB for later retrieval."""
    try:
        db = _get_db()
        collection = db[Config.Standalone.REQUESTS_COLLECTION]
        collection.insert_one(
            {
                "type": "request_log",
                "key_hash": api_key.get("key_hash"),
                "subscription_id": api_key.get("subscription_id"),
                "endpoint": endpoint,
                "mode": mode,
                "filename": filename,
                "request_id": result.get("request_id") or result.get("batch_id"),
                "status": result.get("status"),
                "confidence": result.get("confidence"),
                "answer": result.get("answer"),
                "sources": result.get("sources"),
                "structured_output": result.get("structured_output"),
                "document_id": result.get("document_id"),
                "output_format": result.get("output_format", "json"),
                "usage": result.get("usage"),
                "logged_at": datetime.now(tz=timezone.utc).isoformat(),
            }
        )
    except Exception:
        logger.exception("_log_request failed for endpoint=%s", endpoint)


# ---------------------------------------------------------------------------
# Async dispatch helper
# ---------------------------------------------------------------------------

def _dispatch_async(
    callback_url: str,
    request_id: str,
    api_key: Dict[str, Any],
    fn_name: str,
    kwargs: Dict[str, Any],
) -> None:
    """Run a processing function in a background thread and POST the result.

    The result (or error) is delivered to *callback_url* via the standalone
    webhook module.
    """
    from src.api.standalone_webhook import deliver_webhook_async

    def _run() -> None:
        try:
            # Resolve the callable from standalone_processor / standalone_multi
            import importlib

            for mod_path in (
                "src.api.standalone_processor",
                "src.api.standalone_multi",
            ):
                mod = importlib.import_module(mod_path)
                fn = getattr(mod, fn_name, None)
                if fn is not None:
                    break
            else:
                raise ImportError(f"Function {fn_name!r} not found in standalone modules")

            result = fn(**kwargs)
            payload: Dict[str, Any] = {"request_id": request_id, "status": "completed", **result}
        except Exception as exc:  # noqa: BLE001
            logger.error("_dispatch_async fn=%s failed: %s", fn_name, exc)
            payload = {
                "request_id": request_id,
                "status": "error",
                "error": str(exc),
            }

        deliver_webhook_async(
            url=callback_url,
            payload=payload,
            request_id=request_id,
            key_hash=api_key.get("key_hash", ""),
        )

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@standalone_router.post("/process", response_model=ProcessResponse)
async def process(
    file: UploadFile,
    prompt: str = Form(...),
    mode: str = Form(default="qa"),
    output_format: str = Form(default="json"),
    persist: bool = Form(default=False),
    stream: bool = Form(default=False),
    template: Optional[str] = Form(default=None),
    confidence_threshold: float = Form(default=0.0),
    callback_url: Optional[str] = Form(default=None),
    api_key: Dict = Depends(require_api_key),
):
    """One-shot document processing."""
    _validate_mode(mode)
    _validate_output_format(output_format)

    content = await file.read()
    _validate_file_size(content)

    from src.api.standalone_templates import get_template

    tmpl = get_template(template) if template else None

    if callback_url and callback_url.strip().startswith("http"):
        request_id = str(uuid.uuid4())
        _dispatch_async(
            callback_url=callback_url,
            request_id=request_id,
            api_key=api_key,
            fn_name="process_document",
            kwargs={
                "content": content,
                "filename": file.filename or "upload",
                "prompt": prompt,
                "mode": mode,
                "subscription_id": api_key.get("subscription_id", "standalone"),
                "persist": persist,
                "template": tmpl,
                "confidence_threshold": confidence_threshold,
            },
        )
        return AsyncAcceptedResponse(request_id=request_id)

    from src.api.standalone_processor import process_document

    result = process_document(
        content=content,
        filename=file.filename or "upload",
        prompt=prompt,
        mode=mode,
        subscription_id=api_key.get("subscription_id", "standalone"),
        persist=persist,
        template=tmpl,
        confidence_threshold=confidence_threshold,
    )

    # Output format conversion
    if output_format != "json" and result.get("structured_output"):
        try:
            from src.api.standalone_output import convert_output

            converted = convert_output(
                result["structured_output"], mode, output_format
            )
            # convert_output returns str for non-json formats; wrap in dict for schema
            if isinstance(converted, str):
                result["structured_output"] = {"formatted": converted, "format": output_format}
            else:
                result["structured_output"] = converted
        except (ValueError, Exception) as exc:
            logger.warning("Output conversion failed: %s", exc)

    result["output_format"] = output_format

    track_usage(
        api_key["keys_collection"],
        api_key["key_hash"],
        endpoint="process",
        mode=mode,
    )
    _log_request(api_key, "process", mode, result, filename=file.filename)

    result["result_url"] = f"/api/v1/docwain/requests/{result['request_id']}"
    return ProcessResponse(**result)


@standalone_router.post("/process/multi", response_model=ProcessResponse)
async def process_multi(
    prompt: str = Form(...),
    files: List[UploadFile] = [],
    document_ids: Optional[str] = Form(default=None),
    mode: str = Form(default="qa"),
    output_format: str = Form(default="json"),
    callback_url: Optional[str] = Form(default=None),
    api_key: Dict = Depends(require_api_key),
):
    """Multi-document processing."""
    _validate_mode(mode)
    _validate_output_format(output_format)

    parsed_document_ids: Optional[List[str]] = None
    if document_ids:
        try:
            parsed_document_ids = json.loads(document_ids)
        except json.JSONDecodeError:
            raise HTTPException(status_code=422, detail="document_ids must be a valid JSON array string.")

    file_dicts: List[Dict[str, Any]] = []
    for f in files:
        content = await f.read()
        _validate_file_size(content)
        file_dicts.append({"filename": f.filename or "upload", "content": content})

    from src.api.standalone_multi import process_multi_documents

    kwargs: Dict[str, Any] = {
        "files": file_dicts or None,
        "document_ids": parsed_document_ids,
        "prompt": prompt,
        "mode": mode,
        "subscription_id": api_key.get("subscription_id", "standalone"),
    }

    if callback_url and callback_url.strip().startswith("http"):
        request_id = str(uuid.uuid4())
        _dispatch_async(
            callback_url=callback_url,
            request_id=request_id,
            api_key=api_key,
            fn_name="process_multi_documents",
            kwargs=kwargs,
        )
        return AsyncAcceptedResponse(request_id=request_id)

    result = process_multi_documents(**kwargs)
    result["output_format"] = output_format

    track_usage(
        api_key["keys_collection"],
        api_key["key_hash"],
        endpoint="process_multi",
        mode=mode,
    )
    _log_request(api_key, "process_multi", mode, result)

    return ProcessResponse(**result)


@standalone_router.post("/batch", response_model=BatchResponse)
async def batch(
    prompt: str = Form(...),
    files: List[UploadFile] = [],
    mode: str = Form(default="qa"),
    output_format: str = Form(default="json"),
    callback_url: Optional[str] = Form(default=None),
    api_key: Dict = Depends(require_api_key),
):
    """Bulk document processing."""
    _validate_mode(mode)
    _validate_output_format(output_format)

    if not callback_url and len(files) > Config.Standalone.MAX_BATCH_FILES:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Too many files: {len(files)}. Maximum is "
                f"{Config.Standalone.MAX_BATCH_FILES} without a callback_url."
            ),
        )

    file_dicts: List[Dict[str, Any]] = []
    for f in files:
        content = await f.read()
        _validate_file_size(content)
        file_dicts.append({"filename": f.filename or "upload", "content": content})

    from src.api.standalone_multi import process_batch

    kwargs: Dict[str, Any] = {
        "files": file_dicts,
        "prompt": prompt,
        "mode": mode,
        "subscription_id": api_key.get("subscription_id", "standalone"),
        "output_format": output_format,
    }

    if callback_url:
        batch_id = f"batch-{uuid.uuid4()}"
        _dispatch_async(
            callback_url=callback_url,
            request_id=batch_id,
            api_key=api_key,
            fn_name="process_batch",
            kwargs=kwargs,
        )
        return BatchResponse(batch_id=batch_id, status="processing")

    result = process_batch(**kwargs)

    track_usage(
        api_key["keys_collection"],
        api_key["key_hash"],
        endpoint="batch",
        mode=mode,
    )

    return BatchResponse(**result)


@standalone_router.post("/extract", response_model=ExtractResponse)
async def extract(
    file: UploadFile,
    mode: str = Form(...),
    prompt: Optional[str] = Form(default=None),
    output_format: str = Form(default="json"),
    template: Optional[str] = Form(default=None),
    api_key: Dict = Depends(require_api_key),
):
    """Structured extraction (table, entities, or summary)."""
    if mode not in {"table", "entities", "summary"}:
        raise HTTPException(
            status_code=422,
            detail="mode must be one of: table, entities, summary",
        )
    _validate_output_format(output_format)

    content = await file.read()
    _validate_file_size(content)

    from src.api.standalone_templates import get_template
    from src.api.standalone_processor import process_document

    tmpl = get_template(template) if template else None
    effective_prompt = prompt or f"Extract {mode} from this document."

    result = process_document(
        content=content,
        filename=file.filename or "upload",
        prompt=effective_prompt,
        mode=mode,
        subscription_id=api_key.get("subscription_id", "standalone"),
        persist=False,
        template=tmpl,
    )

    structured = result.get("structured_output")
    if output_format != "json" and structured:
        try:
            from src.api.standalone_output import convert_output

            converted = convert_output(structured, mode, output_format)
            if isinstance(converted, str):
                structured = {"formatted": converted, "format": output_format}
            else:
                structured = converted
        except (ValueError, Exception) as exc:
            logger.warning("Output conversion failed: %s", exc)

    track_usage(
        api_key["keys_collection"],
        api_key["key_hash"],
        endpoint="extract",
        mode=mode,
    )

    return ExtractResponse(
        request_id=result.get("request_id", str(uuid.uuid4())),
        mode=mode,
        result=structured,
        metadata={
            "output_format": output_format,
            "confidence": result.get("confidence", 0.0),
        },
    )


@standalone_router.post("/documents", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile,
    name: Optional[str] = Form(default=None),
    api_key: Dict = Depends(require_api_key),
):
    """Upload and persist a document for later querying."""
    content = await file.read()
    _validate_file_size(content)

    from src.api.standalone_processor import (
        chunk_and_embed,
        extract_from_bytes,
        run_intelligence,
    )

    document_id = str(uuid.uuid4())
    collection_name = f"standalone_{document_id.replace('-', '_')}"
    filename = file.filename or name or "upload"

    extracted = extract_from_bytes(content, filename)
    _intel = run_intelligence(extracted, document_id, filename=filename)
    chunk_count = chunk_and_embed(extracted, document_id, collection_name, filename=filename)

    pages = getattr(extracted, "pages", None) or 1
    document_type = getattr(extracted, "document_type", None) or "unknown"
    created_at = datetime.now(tz=timezone.utc).isoformat()

    _save_document_record(
        document_id=document_id,
        name=name or filename,
        subscription_id=api_key.get("subscription_id", "standalone"),
        key_hash=api_key.get("key_hash", ""),
        pages=pages,
        document_type=document_type,
    )

    track_document_processed(api_key["keys_collection"], api_key["key_hash"])

    return DocumentUploadResponse(
        document_id=document_id,
        name=name or filename,
        status="ready",
        created_at=created_at,
    )


@standalone_router.get("/documents/{doc_id}/status", response_model=DocumentStatusResponse)
async def document_status(
    doc_id: str,
    api_key: Dict = Depends(require_api_key),
):
    """Poll the status of a previously uploaded document."""
    record = _get_document_record(doc_id, api_key.get("subscription_id", "standalone"))
    if record is None:
        raise HTTPException(status_code=404, detail=f"Document {doc_id!r} not found.")

    return DocumentStatusResponse(
        document_id=doc_id,
        status=record.get("status", "unknown"),
        name=record.get("name"),
        pages=record.get("pages"),
        document_type=record.get("document_type"),
        created_at=record.get("created_at"),
        ready_at=record.get("ready_at"),
    )


@standalone_router.post("/query", response_model=ProcessResponse)
async def query(
    prompt: str = Form(...),
    document_id: Optional[str] = Form(default=None),
    document_ids: Optional[str] = Form(default=None),
    mode: str = Form(default="qa"),
    output_format: str = Form(default="json"),
    stream: bool = Form(default=False),
    confidence_threshold: float = Form(default=0.0),
    api_key: Dict = Depends(require_api_key),
):
    """Query one or more previously persisted documents."""
    _validate_mode(mode)
    _validate_output_format(output_format)

    subscription_id = api_key.get("subscription_id", "standalone")

    parsed_ids: Optional[List[str]] = None
    if document_ids:
        try:
            parsed_ids = json.loads(document_ids)
        except json.JSONDecodeError:
            raise HTTPException(status_code=422, detail="document_ids must be a valid JSON array string.")

    if parsed_ids and len(parsed_ids) > 1:
        from src.api.standalone_multi import process_multi_documents

        result = process_multi_documents(
            files=None,
            document_ids=parsed_ids,
            prompt=prompt,
            mode=mode,
            subscription_id=subscription_id,
        )
    else:
        effective_doc_id = document_id or (parsed_ids[0] if parsed_ids else None)
        if effective_doc_id is None:
            raise HTTPException(
                status_code=422,
                detail="Provide document_id or document_ids.",
            )

        from src.api.standalone_processor import query_persisted_document

        result = query_persisted_document(
            document_id=effective_doc_id,
            prompt=prompt,
            subscription_id=subscription_id,
            mode=mode,
            confidence_threshold=confidence_threshold,
        )

    result["output_format"] = output_format

    track_usage(
        api_key["keys_collection"],
        api_key["key_hash"],
        endpoint="query",
        mode=mode,
    )
    _log_request(api_key, "query", mode, result)

    return ProcessResponse(**result)


@standalone_router.get("/usage", response_model=UsageResponse)
async def usage(api_key: Dict = Depends(require_api_key)):
    """Return usage statistics and audit trail for the authenticated API key."""
    key_hash = api_key.get("key_hash", "")

    try:
        keys_collection = api_key["keys_collection"]
        key_doc = keys_collection.find_one({"key_hash": key_hash}, {"_id": 0}) or {}
    except Exception:
        key_doc = {}

    totals = {
        "total_requests": key_doc.get("total_requests", 0),
        "requests_today": key_doc.get("requests_today", 0),
        "documents_processed": key_doc.get("documents_processed", 0),
    }
    by_endpoint: Dict[str, int] = key_doc.get("by_endpoint", {})
    by_mode: Dict[str, int] = key_doc.get("by_mode", {})

    recent: List[Dict[str, Any]] = []
    try:
        db = _get_db()
        cursor = (
            db[Config.Standalone.REQUESTS_COLLECTION]
            .find({"key_hash": key_hash, "type": "request_log"}, {"_id": 0})
            .sort("logged_at", -1)
            .limit(20)
        )
        recent = list(cursor)
    except Exception:
        logger.exception("usage: failed to fetch recent requests for key_hash=%s", key_hash)

    return UsageResponse(
        api_key_name=api_key.get("name", ""),
        period="all_time",
        totals=totals,
        by_endpoint=by_endpoint,
        by_mode=by_mode,
        recent=recent,
    )


@standalone_router.get("/templates", response_model=TemplatesResponse)
async def list_templates():
    """List all available processing templates. No authentication required."""
    from src.api.standalone_templates import list_templates as _list_templates

    templates = [
        TemplateInfo(name=t.name, description=t.description, modes=t.modes)
        for t in _list_templates()
    ]
    return TemplatesResponse(templates=templates)


# ── GET /requests — list processed results ──────────────────────

@standalone_router.get("/requests", response_model=ProcessedResultsListResponse)
async def list_requests(
    limit: int = 20,
    offset: int = 0,
    mode: Optional[str] = None,
    api_key: Dict[str, Any] = Depends(require_api_key),
):
    """List all processed document results for this API key."""
    db = _get_db()
    col = db[Config.Standalone.REQUESTS_COLLECTION]

    query_filter = {
        "type": "request_log",
        "key_hash": api_key["key_hash"],
    }
    if mode:
        query_filter["mode"] = mode

    total = col.count_documents(query_filter)
    cursor = (
        col.find(query_filter, {"_id": 0})
        .sort("logged_at", -1)
        .skip(offset)
        .limit(limit)
    )

    results = []
    for doc in cursor:
        results.append(ProcessedResultResponse(
            request_id=doc.get("request_id"),
            endpoint=doc.get("endpoint"),
            mode=doc.get("mode"),
            filename=doc.get("filename"),
            status=doc.get("status"),
            answer=doc.get("answer"),
            sources=doc.get("sources"),
            structured_output=doc.get("structured_output"),
            confidence=doc.get("confidence"),
            document_id=doc.get("document_id"),
            output_format=doc.get("output_format"),
            usage=doc.get("usage"),
            logged_at=doc.get("logged_at"),
        ))

    return ProcessedResultsListResponse(results=results, total=total)


# ── GET /requests/{request_id} — get a specific result ──────────

@standalone_router.get("/requests/{request_id}", response_model=ProcessedResultResponse)
async def get_request(
    request_id: str,
    api_key: Dict[str, Any] = Depends(require_api_key),
):
    """Retrieve the full result of a previously processed document."""
    db = _get_db()
    col = db[Config.Standalone.REQUESTS_COLLECTION]

    doc = col.find_one(
        {"request_id": request_id, "key_hash": api_key["key_hash"]},
        {"_id": 0},
    )
    if doc is None:
        raise HTTPException(
            status_code=404,
            detail={"error": {"code": "NOT_FOUND", "message": f"Request {request_id} not found"}},
        )

    return ProcessedResultResponse(
        request_id=doc.get("request_id"),
        endpoint=doc.get("endpoint"),
        mode=doc.get("mode"),
        filename=doc.get("filename"),
        status=doc.get("status"),
        answer=doc.get("answer"),
        sources=doc.get("sources"),
        structured_output=doc.get("structured_output"),
        confidence=doc.get("confidence"),
        document_id=doc.get("document_id"),
        output_format=doc.get("output_format"),
        usage=doc.get("usage"),
        logged_at=doc.get("logged_at"),
    )


# ── POST /keys — generate a new API key ─────────────────────────

@standalone_router.post("/keys", response_model=ApiKeyCreateResponse)
async def create_api_key_endpoint(
    body: ApiKeyCreateRequest,
):
    """Generate a new API key. The raw key is returned only once — save it.

    No authentication required (this is how the first key gets created).
    In production, protect this endpoint via network-level controls or
    add a master secret header.
    """
    from pymongo import MongoClient
    from src.api.standalone_auth import generate_api_key

    raw_key, key_hash = generate_api_key()
    key_prefix = raw_key[:10] + "..."
    now = datetime.now(timezone.utc).isoformat()

    doc = {
        "key_hash": key_hash,
        "key_prefix": key_prefix,
        "name": body.name,
        "created_at": now,
        "active": True,
        "permissions": ["process", "extract", "batch", "query"],
        "usage": {
            "total_requests": 0,
            "last_used": None,
            "requests_today": 0,
            "documents_processed": 0,
        },
    }
    if body.subscription_id:
        doc["subscription_id"] = body.subscription_id

    client = MongoClient(Config.MongoDB.URI)
    db = client[Config.MongoDB.DB]
    col = db[Config.Standalone.API_KEYS_COLLECTION]
    col.insert_one(doc)

    return ApiKeyCreateResponse(
        api_key=raw_key,
        name=body.name,
        key_prefix=key_prefix,
        subscription_id=body.subscription_id,
        created_at=now,
    )
