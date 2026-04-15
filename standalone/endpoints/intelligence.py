import json
import re
import time
import uuid

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile

from standalone.config import Config
from standalone.dependencies import get_db, get_vllm_client
from standalone.auth import hash_api_key, validate_api_key
from standalone.file_reader import UnsupportedFileType, read_file_with_metadata
from standalone.schemas import AnalysisType, IntelligenceResponse, ResponseMetadata
from standalone.vllm_client import VLLMClient

router = APIRouter()


@router.post("/api/v1/standalone/intelligence", response_model=IntelligenceResponse)
async def intelligence(
    file: UploadFile = File(...),
    analysis_type: AnalysisType = Form(AnalysisType.auto),
    prompt: str | None = Form(None),
    x_api_key: str | None = Header(None),
    db=Depends(get_db),
    vllm: VLLMClient = Depends(get_vllm_client),
):
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing X-Api-Key header")

    key_doc = validate_api_key(x_api_key, db["api_keys"])
    if not key_doc:
        raise HTTPException(status_code=401, detail="Invalid API key")

    request_id = str(uuid.uuid4())
    start = time.time()

    data = await file.read()
    max_bytes = Config.MAX_FILE_SIZE_MB * 1024 * 1024
    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail=f"File too large (max {Config.MAX_FILE_SIZE_MB}MB)")

    try:
        text, meta = read_file_with_metadata(file.filename or "unknown", data)
    except UnsupportedFileType as e:
        raise HTTPException(status_code=422, detail=str(e))

    raw_response = await vllm.analyze(text, analysis_type.value, prompt)

    # Parse insights from LLM response
    insights = _parse_insights(raw_response)

    elapsed_ms = int((time.time() - start) * 1000)

    # Log request and increment usage counter (fire-and-forget)
    key_hash = hash_api_key(x_api_key)
    try:
        db.request_logs.insert_one({
            "request_id": request_id,
            "endpoint": "intelligence",
            "key_hash": key_hash,
            "filename": file.filename,
            "file_type": meta["file_type"],
            "analysis_type": analysis_type.value,
            "processing_time_ms": elapsed_ms,
            "timestamp": time.time(),
        })
        db.api_keys.update_one({"key_hash": key_hash}, {"$inc": {"total_requests": 1}})
    except Exception:
        pass

    return IntelligenceResponse(
        request_id=request_id,
        document_type=meta["file_type"],
        analysis_type=analysis_type.value,
        insights=insights,
        metadata=ResponseMetadata(
            pages=meta.get("pages", 1),
            processing_time_ms=elapsed_ms,
        ),
    )


def _parse_insights(raw: str) -> dict:
    raw = raw.strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass

    match = re.search(r"```(?:json)?\s*\n(.*?)\n```", raw, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(1))
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass

    return {"summary": raw, "findings": [], "evidence": []}
