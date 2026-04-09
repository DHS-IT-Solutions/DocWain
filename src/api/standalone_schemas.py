"""Pydantic request/response models for the DocWain Standalone API."""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared / embedded models
# ---------------------------------------------------------------------------


class SourceRef(BaseModel):
    """Reference to a source passage within a document."""

    page: Optional[int] = None
    section: Optional[str] = None
    confidence: Optional[float] = None
    document: Optional[str] = None
    document_id: Optional[str] = None


class UsageStats(BaseModel):
    """Timing breakdown for a single API request."""

    extraction_ms: int = 0
    intelligence_ms: int = 0
    retrieval_ms: int = 0
    generation_ms: int = 0
    total_ms: int = 0


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class ProcessRequest(BaseModel):
    """Request body for the /process endpoint."""

    prompt: str
    mode: Literal["qa", "table", "entities", "summary"] = "qa"
    output_format: Literal["json", "markdown", "csv", "html"] = "json"
    persist: bool = False
    stream: bool = False
    template: Optional[str] = None
    confidence_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    callback_url: Optional[str] = None


class MultiProcessRequest(BaseModel):
    """Request body for multi-document processing."""

    prompt: str
    document_ids: Optional[List[str]] = None
    mode: Literal["qa", "table", "entities", "summary"] = "qa"
    output_format: Literal["json", "markdown", "csv", "html"] = "json"
    callback_url: Optional[str] = None


class BatchRequest(BaseModel):
    """Request body for batch document processing."""

    prompt: str
    mode: Literal["qa", "table", "entities", "summary"] = "qa"
    output_format: Literal["json", "markdown", "csv", "html"] = "json"
    callback_url: Optional[str] = None


class ExtractRequest(BaseModel):
    """Request body for the /extract endpoint (mode is required)."""

    mode: Literal["table", "entities", "summary"]
    prompt: Optional[str] = None
    output_format: Literal["json", "markdown", "csv", "html"] = "json"
    template: Optional[str] = None


class QueryRequest(BaseModel):
    """Request body for the /query endpoint."""

    prompt: str
    document_id: Optional[str] = None
    document_ids: Optional[List[str]] = None
    mode: Literal["qa", "table", "entities", "summary"] = "qa"
    output_format: Literal["json", "markdown", "csv", "html"] = "json"
    stream: bool = False
    confidence_threshold: float = Field(default=0.0, ge=0.0, le=1.0)


class DocumentUploadRequest(BaseModel):
    """Optional metadata provided alongside a document upload."""

    name: Optional[str] = None


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class ProcessResponse(BaseModel):
    """Response body for synchronous processing requests."""

    request_id: str
    status: str = "completed"
    answer: Optional[str] = None
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    confidence: float = 0.0
    grounded: bool = False
    low_confidence: bool = False
    low_confidence_reasons: List[str] = Field(default_factory=list)
    structured_output: Optional[Dict[str, Any]] = None
    document_id: Optional[str] = None
    output_format: str = "json"
    partial_answer: Optional[str] = None
    result_url: Optional[str] = None
    usage: UsageStats = Field(default_factory=UsageStats)


class AsyncAcceptedResponse(BaseModel):
    """Response body when a request is accepted for async processing."""

    request_id: str
    status: str = "processing"
    poll_url: Optional[str] = None


class DocumentStatusResponse(BaseModel):
    """Status information for a single document."""

    document_id: str
    status: str
    name: Optional[str] = None
    pages: Optional[int] = None
    document_type: Optional[str] = None
    created_at: Optional[str] = None
    ready_at: Optional[str] = None


class DocumentUploadResponse(BaseModel):
    """Response body after a document upload is accepted."""

    document_id: str
    name: Optional[str] = None
    status: str = "processing"
    created_at: str


class BatchItemResult(BaseModel):
    """Result for a single file within a batch job."""

    filename: str
    status: str
    answer: Optional[str] = None
    confidence: float = 0.0
    structured_output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class BatchResponse(BaseModel):
    """Response body for batch processing jobs."""

    batch_id: str
    status: str
    results: List[BatchItemResult] = Field(default_factory=list)
    summary: Dict[str, int] = Field(default_factory=dict)
    usage: UsageStats = Field(default_factory=UsageStats)


class ExtractResponse(BaseModel):
    """Response body for the /extract endpoint."""

    request_id: str
    mode: str
    result: Any = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TemplateInfo(BaseModel):
    """Metadata about a processing template."""

    name: str
    description: str
    modes: List[str]


class TemplatesResponse(BaseModel):
    """List of available processing templates."""

    templates: List[TemplateInfo]


class UsageResponse(BaseModel):
    """API usage statistics for an API key over a given period."""

    api_key_name: str
    period: str
    totals: Dict[str, int]
    by_endpoint: Dict[str, int]
    by_mode: Dict[str, int]
    recent: List[Dict[str, Any]]


class ProcessedResultResponse(BaseModel):
    """A stored processing result."""

    request_id: Optional[str] = None
    endpoint: Optional[str] = None
    mode: Optional[str] = None
    filename: Optional[str] = None
    status: Optional[str] = None
    answer: Optional[str] = None
    sources: Optional[List[Dict[str, Any]]] = None
    structured_output: Optional[Dict[str, Any]] = None
    confidence: Optional[float] = None
    document_id: Optional[str] = None
    output_format: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    logged_at: Optional[str] = None


class ProcessedResultsListResponse(BaseModel):
    """List of stored processing results."""

    results: List[ProcessedResultResponse]
    total: int


class ApiKeyCreateRequest(BaseModel):
    """Request body for API key creation."""

    name: str
    subscription_id: Optional[str] = None


class ApiKeyCreateResponse(BaseModel):
    """Response after creating an API key. The raw key is shown only once."""

    api_key: str
    name: str
    key_prefix: str
    subscription_id: Optional[str] = None
    created_at: str


class StandaloneErrorDetail(BaseModel):
    """Detailed error information."""

    code: str
    message: str
    request_id: Optional[str] = None


class StandaloneErrorResponse(BaseModel):
    """Standard error envelope for standalone API errors."""

    error: StandaloneErrorDetail
