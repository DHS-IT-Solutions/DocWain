from typing import Any, Optional
from enum import Enum

from pydantic import BaseModel, Field


class OutputFormat(str, Enum):
    json = "json"
    csv = "csv"
    sections = "sections"
    flatfile = "flatfile"
    tables = "tables"


class AnalysisType(str, Enum):
    summary = "summary"
    key_facts = "key_facts"
    risk_assessment = "risk_assessment"
    recommendations = "recommendations"
    auto = "auto"


class ExtractRequest(BaseModel):
    output_format: OutputFormat
    prompt: Optional[str] = None


class IntelligenceRequest(BaseModel):
    analysis_type: AnalysisType = AnalysisType.auto
    prompt: Optional[str] = None


class ResponseMetadata(BaseModel):
    pages: int
    processing_time_ms: int


class ExtractResponse(BaseModel):
    request_id: str
    document_type: str
    output_format: str
    content: Any
    metadata: ResponseMetadata


class IntelligenceResponse(BaseModel):
    request_id: str
    document_type: str
    analysis_type: str
    insights: dict[str, Any]
    metadata: ResponseMetadata


class KeyCreateRequest(BaseModel):
    name: str = Field(..., min_length=1)


class KeyCreateResponse(BaseModel):
    key_id: str
    raw_key: str
    key_prefix: str
    name: str
    created_at: str


class KeyListItem(BaseModel):
    key_id: str
    key_prefix: str
    name: str
    created_at: str
    total_requests: int


class ErrorResponse(BaseModel):
    error: str
    request_id: Optional[str] = None
