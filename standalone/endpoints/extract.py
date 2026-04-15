import time
import uuid

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile

from standalone.dependencies import get_db, get_vllm_client
from standalone.auth import hash_api_key, validate_api_key
from standalone.file_reader import UnsupportedFileType, read_file_with_metadata
from standalone.output_formatter import format_output
from standalone.schemas import ExtractResponse, OutputFormat, ResponseMetadata
from standalone.vllm_client import VLLMClient

router = APIRouter()


@router.post("/api/v1/standalone/extract", response_model=ExtractResponse)
async def extract(
    file: UploadFile = File(...),
    output_format: OutputFormat = Form(...),
    prompt: str | None = Form(None),
    x_api_key: str | None = Header(None),
    db=Depends(get_db),
    vllm: VLLMClient = Depends(get_vllm_client),
):
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing X-Api-Key header")

    key_doc = await validate_api_key(x_api_key, db["api_keys"])
    if not key_doc:
        raise HTTPException(status_code=401, detail="Invalid API key")

    request_id = str(uuid.uuid4())
    start = time.time()

    data = await file.read()
    if len(data) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 50MB)")

    try:
        text, meta = read_file_with_metadata(file.filename or "unknown", data)
    except UnsupportedFileType as e:
        raise HTTPException(status_code=422, detail=str(e))

    raw_response = await vllm.extract(text, output_format.value, prompt)
    content = format_output(raw_response, output_format.value)

    elapsed_ms = int((time.time() - start) * 1000)

    # Log request (fire-and-forget)
    try:
        db.request_logs.insert_one({
            "request_id": request_id,
            "endpoint": "extract",
            "key_hash": hash_api_key(x_api_key),
            "filename": file.filename,
            "file_type": meta["file_type"],
            "output_format": output_format.value,
            "processing_time_ms": elapsed_ms,
            "timestamp": time.time(),
        })
    except Exception:
        pass  # Don't fail the request if logging fails

    return ExtractResponse(
        request_id=request_id,
        document_type=meta["file_type"],
        output_format=output_format.value,
        content=content,
        metadata=ResponseMetadata(
            pages=meta.get("pages", 1),
            processing_time_ms=elapsed_ms,
        ),
    )
