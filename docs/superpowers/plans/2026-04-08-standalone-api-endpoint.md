# Standalone API Endpoint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an authenticated standalone API endpoint that exposes all of DocWain's document intelligence (Q&A, table extraction, entity extraction, summaries) with multi-doc, batch, webhooks, templates, confidence gating, and finetune capture.

**Architecture:** New FastAPI router mounted at `/api/v1/docwain` on the existing app. API key auth via `X-API-Key` header with SHA-256 hashed keys in MongoDB. A standalone processor orchestrates the full pipeline (extract -> intelligence -> chunk -> embed -> retrieve -> reason -> generate) using existing modules. Output format conversion, webhook delivery, and prompt templates are separate focused modules.

**Tech Stack:** FastAPI, Pydantic v2, MongoDB (pymongo), Qdrant, existing DocWain modules (DocumentExtractor, SectionChunker, UnifiedRetriever, CoreAgent, LLMGateway, LearningSignalStore)

**Spec:** `docs/superpowers/specs/2026-04-08-standalone-api-endpoint-design.md`

---

### Task 1: Pydantic Schemas (`standalone_schemas.py`)

**Files:**
- Create: `src/api/standalone_schemas.py`
- Test: `tests/test_standalone_schemas.py`

- [ ] **Step 1: Write failing test for request/response models**

```python
# tests/test_standalone_schemas.py
import pytest
from pydantic import ValidationError


def test_process_request_defaults():
    from src.api.standalone_schemas import ProcessRequest
    r = ProcessRequest(prompt="Summarize this document")
    assert r.mode == "qa"
    assert r.output_format == "json"
    assert r.persist is False
    assert r.stream is False
    assert r.confidence_threshold == 0.0
    assert r.callback_url is None
    assert r.template is None


def test_process_request_validates_mode():
    from src.api.standalone_schemas import ProcessRequest
    with pytest.raises(ValidationError):
        ProcessRequest(prompt="test", mode="invalid_mode")


def test_process_response_shape():
    from src.api.standalone_schemas import ProcessResponse
    r = ProcessResponse(
        request_id="req-123",
        status="completed",
        answer="The revenue was $5M",
        sources=[{"page": 1, "section": "Financials"}],
        confidence=0.92,
        grounded=True,
    )
    assert r.low_confidence is False
    assert r.structured_output is None
    assert r.document_id is None


def test_batch_response_shape():
    from src.api.standalone_schemas import BatchResponse, BatchItemResult
    item = BatchItemResult(filename="doc.pdf", status="completed", answer="answer", confidence=0.9)
    r = BatchResponse(
        batch_id="batch-123",
        status="completed",
        results=[item],
        summary={"total": 1, "completed": 1, "failed": 0},
    )
    assert len(r.results) == 1


def test_extract_request_requires_mode():
    from src.api.standalone_schemas import ExtractRequest
    with pytest.raises(ValidationError):
        ExtractRequest(prompt="test")  # mode is required


def test_query_request_requires_document_id():
    from src.api.standalone_schemas import QueryRequest
    r = QueryRequest(prompt="What is revenue?", document_id="doc-123")
    assert r.document_id == "doc-123"


def test_usage_response():
    from src.api.standalone_schemas import UsageResponse
    r = UsageResponse(
        api_key_name="Partner X",
        period="2026-04-01 to 2026-04-08",
        totals={"requests": 100, "documents_processed": 10, "queries": 90},
        by_endpoint={"/process": 50, "/query": 50},
        by_mode={"qa": 80, "table": 20},
        recent=[],
    )
    assert r.api_key_name == "Partner X"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/test_standalone_schemas.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.api.standalone_schemas'`

- [ ] **Step 3: Implement all schemas**

```python
# src/api/standalone_schemas.py
"""Pydantic request/response models for the DocWain Standalone API."""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


# ── Request Models ──────────────────────────────────────────────

class ProcessRequest(BaseModel):
    """Fields parsed from multipart form alongside the uploaded file."""
    prompt: str
    mode: Literal["qa", "table", "entities", "summary"] = "qa"
    output_format: Literal["json", "markdown", "csv", "html"] = "json"
    persist: bool = False
    stream: bool = False
    template: Optional[str] = None
    confidence_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    callback_url: Optional[str] = None


class MultiProcessRequest(BaseModel):
    """Fields for multi-document processing."""
    prompt: str
    document_ids: Optional[List[str]] = None
    mode: Literal["qa", "table", "entities", "summary"] = "qa"
    output_format: Literal["json", "markdown", "csv", "html"] = "json"
    callback_url: Optional[str] = None


class BatchRequest(BaseModel):
    """Fields for batch processing."""
    prompt: str
    mode: Literal["qa", "table", "entities", "summary"] = "qa"
    output_format: Literal["json", "markdown", "csv", "html"] = "json"
    callback_url: Optional[str] = None


class ExtractRequest(BaseModel):
    """Fields for structured extraction."""
    mode: Literal["table", "entities", "summary"]
    prompt: Optional[str] = None
    output_format: Literal["json", "csv", "markdown", "html"] = "json"
    template: Optional[str] = None


class QueryRequest(BaseModel):
    """Query a previously persisted document."""
    prompt: str
    document_id: Optional[str] = None
    document_ids: Optional[List[str]] = None
    mode: Literal["qa", "table", "entities", "summary"] = "qa"
    output_format: Literal["json", "markdown", "csv", "html"] = "json"
    stream: bool = False
    confidence_threshold: float = Field(default=0.0, ge=0.0, le=1.0)


class DocumentUploadRequest(BaseModel):
    """Metadata for persisted document upload."""
    name: Optional[str] = None


# ── Response Models ─────────────────────────────────────────────

class SourceRef(BaseModel):
    page: Optional[int] = None
    section: Optional[str] = None
    confidence: Optional[float] = None
    document: Optional[str] = None
    document_id: Optional[str] = None


class UsageStats(BaseModel):
    extraction_ms: int = 0
    intelligence_ms: int = 0
    retrieval_ms: int = 0
    generation_ms: int = 0
    total_ms: int = 0


class ProcessResponse(BaseModel):
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
    usage: UsageStats = Field(default_factory=UsageStats)


class AsyncAcceptedResponse(BaseModel):
    request_id: str
    status: str = "processing"
    poll_url: Optional[str] = None


class DocumentStatusResponse(BaseModel):
    document_id: str
    status: str
    name: Optional[str] = None
    pages: Optional[int] = None
    document_type: Optional[str] = None
    created_at: Optional[str] = None
    ready_at: Optional[str] = None


class DocumentUploadResponse(BaseModel):
    document_id: str
    name: Optional[str] = None
    status: str = "processing"
    created_at: str


class BatchItemResult(BaseModel):
    filename: str
    status: str
    answer: Optional[str] = None
    confidence: float = 0.0
    structured_output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class BatchResponse(BaseModel):
    batch_id: str
    status: str
    results: List[BatchItemResult] = Field(default_factory=list)
    summary: Dict[str, int] = Field(default_factory=dict)
    usage: UsageStats = Field(default_factory=UsageStats)


class ExtractResponse(BaseModel):
    request_id: str
    mode: str
    result: Any = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TemplateInfo(BaseModel):
    name: str
    description: str
    modes: List[str]


class TemplatesResponse(BaseModel):
    templates: List[TemplateInfo]


class UsageResponse(BaseModel):
    api_key_name: str
    period: str
    totals: Dict[str, int]
    by_endpoint: Dict[str, int]
    by_mode: Dict[str, int]
    recent: List[Dict[str, Any]]


class StandaloneErrorDetail(BaseModel):
    code: str
    message: str
    request_id: Optional[str] = None


class StandaloneErrorResponse(BaseModel):
    error: StandaloneErrorDetail
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/test_standalone_schemas.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/api/standalone_schemas.py tests/test_standalone_schemas.py
git commit -m "feat(standalone): add Pydantic request/response schemas"
```

---

### Task 2: API Key Auth (`standalone_auth.py`)

**Files:**
- Create: `src/api/standalone_auth.py`
- Test: `tests/test_standalone_auth.py`

- [ ] **Step 1: Write failing tests for auth module**

```python
# tests/test_standalone_auth.py
import pytest
from unittest.mock import MagicMock, patch


def test_hash_api_key_deterministic():
    from src.api.standalone_auth import hash_api_key
    key = "dw_abc123def456"
    assert hash_api_key(key) == hash_api_key(key)
    assert len(hash_api_key(key)) == 64  # SHA-256 hex


def test_generate_api_key_format():
    from src.api.standalone_auth import generate_api_key
    raw_key, key_hash = generate_api_key()
    assert raw_key.startswith("dw_")
    assert len(raw_key) == 51  # "dw_" + 48 hex chars
    assert len(key_hash) == 64


def test_validate_api_key_success():
    from src.api.standalone_auth import validate_api_key_sync, hash_api_key
    raw_key = "dw_" + "a" * 48
    key_hash = hash_api_key(raw_key)
    mock_collection = MagicMock()
    mock_collection.find_one.return_value = {
        "key_hash": key_hash,
        "name": "Test Key",
        "subscription_id": "sub-123",
        "active": True,
        "permissions": ["process", "extract", "batch", "query"],
        "usage": {"total_requests": 0, "last_used": None},
    }
    result = validate_api_key_sync(raw_key, mock_collection)
    assert result is not None
    assert result["subscription_id"] == "sub-123"
    assert result["name"] == "Test Key"


def test_validate_api_key_inactive():
    from src.api.standalone_auth import validate_api_key_sync, hash_api_key
    raw_key = "dw_" + "b" * 48
    key_hash = hash_api_key(raw_key)
    mock_collection = MagicMock()
    mock_collection.find_one.return_value = {
        "key_hash": key_hash,
        "active": False,
    }
    result = validate_api_key_sync(raw_key, mock_collection)
    assert result is None


def test_validate_api_key_not_found():
    from src.api.standalone_auth import validate_api_key_sync
    mock_collection = MagicMock()
    mock_collection.find_one.return_value = None
    result = validate_api_key_sync("dw_nonexistent" + "0" * 37, mock_collection)
    assert result is None


def test_track_usage_increments():
    from src.api.standalone_auth import track_usage
    mock_collection = MagicMock()
    track_usage(mock_collection, "somehash", "/process", "qa")
    mock_collection.update_one.assert_called_once()
    call_args = mock_collection.update_one.call_args
    assert call_args[0][0] == {"key_hash": "somehash"}
    update = call_args[0][1]
    assert "$inc" in update
    assert "$set" in update
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/test_standalone_auth.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement auth module**

```python
# src/api/standalone_auth.py
"""API key authentication and usage tracking for Standalone API."""
from __future__ import annotations

import hashlib
import logging
import secrets
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import Depends, Header, HTTPException, Request

logger = logging.getLogger(__name__)


def hash_api_key(raw_key: str) -> str:
    """SHA-256 hash of the raw API key."""
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def generate_api_key() -> tuple[str, str]:
    """Generate a new API key. Returns (raw_key, key_hash)."""
    token = secrets.token_hex(24)  # 48 hex chars
    raw_key = f"dw_{token}"
    return raw_key, hash_api_key(raw_key)


def validate_api_key_sync(
    raw_key: str,
    keys_collection: Any,
) -> Optional[Dict[str, Any]]:
    """Validate an API key against MongoDB. Returns key doc or None."""
    key_hash = hash_api_key(raw_key)
    doc = keys_collection.find_one({"key_hash": key_hash})
    if doc is None:
        return None
    if not doc.get("active", False):
        return None
    return doc


def track_usage(
    keys_collection: Any,
    key_hash: str,
    endpoint: str,
    mode: str,
) -> None:
    """Increment usage counters (fire-and-forget)."""
    try:
        keys_collection.update_one(
            {"key_hash": key_hash},
            {
                "$inc": {
                    "usage.total_requests": 1,
                    "usage.requests_today": 1,
                    f"usage.by_endpoint.{endpoint.strip('/')}": 1,
                    f"usage.by_mode.{mode}": 1,
                },
                "$set": {
                    "usage.last_used": datetime.now(timezone.utc).isoformat(),
                },
            },
        )
    except Exception as exc:
        logger.debug("Failed to track usage: %s", exc)


def track_document_processed(keys_collection: Any, key_hash: str) -> None:
    """Increment document count."""
    try:
        keys_collection.update_one(
            {"key_hash": key_hash},
            {"$inc": {"usage.documents_processed": 1}},
        )
    except Exception as exc:
        logger.debug("Failed to track document: %s", exc)


def _get_keys_collection():
    """Get the api_keys MongoDB collection."""
    from pymongo import MongoClient
    from src.api.config import Config
    client = MongoClient(Config.MongoDB.URI)
    db = client[Config.MongoDB.DB]
    collection_name = getattr(Config, "Standalone", None)
    if collection_name and hasattr(collection_name, "API_KEYS_COLLECTION"):
        return db[collection_name.API_KEYS_COLLECTION]
    return db["api_keys"]


async def require_api_key(
    request: Request,
    x_api_key: Optional[str] = Header(None),
) -> Dict[str, Any]:
    """FastAPI dependency: validate X-API-Key header and inject key context."""
    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail={"error": {"code": "AUTH_INVALID", "message": "Missing X-API-Key header"}},
        )
    keys_collection = _get_keys_collection()
    key_doc = validate_api_key_sync(x_api_key, keys_collection)
    if key_doc is None:
        raise HTTPException(
            status_code=401,
            detail={"error": {"code": "AUTH_INVALID", "message": "Invalid or disabled API key"}},
        )
    # Attach key context for downstream use
    return {
        "key_hash": key_doc["key_hash"],
        "name": key_doc.get("name", ""),
        "subscription_id": key_doc["subscription_id"],
        "permissions": key_doc.get("permissions", []),
        "keys_collection": keys_collection,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/test_standalone_auth.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/api/standalone_auth.py tests/test_standalone_auth.py
git commit -m "feat(standalone): add API key auth with usage tracking"
```

---

### Task 3: Prompt Templates (`standalone_templates.py`)

**Files:**
- Create: `src/api/standalone_templates.py`
- Test: `tests/test_standalone_templates.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_standalone_templates.py
import pytest


def test_get_template_by_name():
    from src.api.standalone_templates import get_template
    t = get_template("invoice")
    assert t is not None
    assert t.name == "invoice"
    assert "table" in t.modes
    assert len(t.system_prompt) > 0


def test_get_template_unknown_returns_none():
    from src.api.standalone_templates import get_template
    assert get_template("nonexistent") is None


def test_list_templates():
    from src.api.standalone_templates import list_templates
    templates = list_templates()
    assert len(templates) >= 6
    names = {t.name for t in templates}
    assert "invoice" in names
    assert "contract_clauses" in names
    assert "resume" in names


def test_apply_template_enhances_prompt():
    from src.api.standalone_templates import get_template, apply_template
    t = get_template("invoice")
    result = apply_template(t, "Extract line items from this invoice")
    assert "Extract line items" in result["user_prompt"]
    assert len(result["system_prompt"]) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/test_standalone_templates.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement templates module**

```python
# src/api/standalone_templates.py
"""Pre-built prompt templates for common DocWain standalone API use cases."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PromptTemplate:
    name: str
    description: str
    modes: List[str]
    system_prompt: str
    extraction_hints: str
    output_schema: Dict[str, Any] = field(default_factory=dict)


_TEMPLATES: Dict[str, PromptTemplate] = {}


def _register(t: PromptTemplate) -> None:
    _TEMPLATES[t.name] = t


# ── Invoice ──
_register(PromptTemplate(
    name="invoice",
    description="Extract invoice fields: vendor, amounts, dates, line items",
    modes=["table", "entities"],
    system_prompt=(
        "You are a document intelligence system specialized in invoice processing. "
        "Extract all structured data from the invoice including: vendor name, invoice number, "
        "invoice date, due date, line items (description, quantity, unit price, amount), "
        "subtotal, tax, and total. Preserve exact monetary values and dates."
    ),
    extraction_hints="Focus on tabular line items, header fields, and totals. Preserve currency symbols.",
    output_schema={"fields": ["vendor", "invoice_number", "date", "due_date", "line_items", "subtotal", "tax", "total"]},
))

# ── Contract Clauses ──
_register(PromptTemplate(
    name="contract_clauses",
    description="Identify and extract contract clauses with risk assessment",
    modes=["entities", "summary"],
    system_prompt=(
        "You are a legal document analysis system. Identify all clauses in the contract, "
        "classify them by type (termination, liability, indemnification, confidentiality, IP, "
        "payment terms, force majeure, etc.), and flag any unusual or high-risk provisions. "
        "Quote exact clause text with section references."
    ),
    extraction_hints="Identify clause boundaries, section numbers, and cross-references.",
    output_schema={"fields": ["clause_type", "section_ref", "text", "risk_level", "notes"]},
))

# ── Compliance Checklist ──
_register(PromptTemplate(
    name="compliance_checklist",
    description="Check document against compliance requirements",
    modes=["qa", "entities"],
    system_prompt=(
        "You are a compliance review system. Analyze the document for regulatory compliance. "
        "Identify required disclosures, missing elements, and potential compliance gaps. "
        "Reference specific regulatory requirements where applicable. Be thorough and precise."
    ),
    extraction_hints="Look for regulatory references, required disclosures, and standard compliance elements.",
    output_schema={"fields": ["requirement", "status", "evidence", "section_ref", "notes"]},
))

# ── Medical Record ──
_register(PromptTemplate(
    name="medical_record",
    description="Extract patient info, diagnoses, medications, procedures",
    modes=["entities", "table"],
    system_prompt=(
        "You are a medical document intelligence system. Extract structured clinical data: "
        "patient demographics, diagnoses (with ICD codes if present), medications (name, dosage, "
        "frequency), procedures, lab results, vital signs, and care plan elements. "
        "Maintain clinical precision — do not paraphrase medical terminology."
    ),
    extraction_hints="Preserve medical codes, dosages, and clinical terminology exactly as written.",
    output_schema={"fields": ["patient_info", "diagnoses", "medications", "procedures", "lab_results", "vitals"]},
))

# ── Financial Report ──
_register(PromptTemplate(
    name="financial_report",
    description="Extract financial tables, KPIs, and executive summary",
    modes=["table", "summary"],
    system_prompt=(
        "You are a financial document analysis system. Extract all financial tables (income statement, "
        "balance sheet, cash flow), key performance indicators, period-over-period comparisons, "
        "and executive summary points. Preserve exact figures, percentages, and date ranges."
    ),
    extraction_hints="Focus on financial tables, charts data, KPI callouts, and year-over-year comparisons.",
    output_schema={"fields": ["tables", "kpis", "comparisons", "summary_points"]},
))

# ── Resume ──
_register(PromptTemplate(
    name="resume",
    description="Extract skills, experience, education, contact information",
    modes=["entities", "table"],
    system_prompt=(
        "You are a resume/CV analysis system. Extract structured data: contact information, "
        "professional summary, skills (technical and soft), work experience (company, title, dates, "
        "achievements), education (institution, degree, dates), certifications, and languages. "
        "Preserve dates and proper nouns exactly."
    ),
    extraction_hints="Identify section boundaries (experience, education, skills), extract date ranges, and list items.",
    output_schema={"fields": ["contact", "summary", "skills", "experience", "education", "certifications"]},
))


def get_template(name: str) -> Optional[PromptTemplate]:
    """Get a template by name. Returns None if not found."""
    return _TEMPLATES.get(name)


def list_templates() -> List[PromptTemplate]:
    """Return all registered templates."""
    return list(_TEMPLATES.values())


def apply_template(
    template: PromptTemplate,
    user_prompt: Optional[str] = None,
) -> Dict[str, str]:
    """Combine template with user prompt. User prompt always takes precedence."""
    system = template.system_prompt
    if template.extraction_hints:
        system += f"\n\nExtraction guidance: {template.extraction_hints}"

    final_user_prompt = user_prompt or f"Process this document using the {template.name} template."

    return {
        "system_prompt": system,
        "user_prompt": final_user_prompt,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/test_standalone_templates.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/api/standalone_templates.py tests/test_standalone_templates.py
git commit -m "feat(standalone): add prompt template registry with 6 templates"
```

---

### Task 4: Output Format Conversion (`standalone_output.py`)

**Files:**
- Create: `src/api/standalone_output.py`
- Test: `tests/test_standalone_output.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_standalone_output.py
import pytest


def test_convert_table_to_csv():
    from src.api.standalone_output import convert_output
    data = {
        "tables": [
            {"headers": ["Name", "Amount"], "rows": [["Widget", "$100"], ["Gadget", "$200"]], "page": 1}
        ]
    }
    result = convert_output(data, "table", "csv")
    assert "Name,Amount" in result
    assert "Widget" in result


def test_convert_table_to_markdown():
    from src.api.standalone_output import convert_output
    data = {
        "tables": [
            {"headers": ["Name", "Amount"], "rows": [["Widget", "$100"]], "page": 1}
        ]
    }
    result = convert_output(data, "table", "markdown")
    assert "| Name | Amount |" in result
    assert "| Widget | $100 |" in result


def test_convert_table_to_html():
    from src.api.standalone_output import convert_output
    data = {
        "tables": [
            {"headers": ["Name"], "rows": [["Widget"]], "page": 1}
        ]
    }
    result = convert_output(data, "table", "html")
    assert "<table>" in result
    assert "<th>Name</th>" in result


def test_convert_entities_to_csv():
    from src.api.standalone_output import convert_output
    data = {
        "entities": [
            {"text": "John Doe", "type": "PERSON", "page": 1, "confidence": 0.95}
        ]
    }
    result = convert_output(data, "entities", "csv")
    assert "John Doe" in result
    assert "PERSON" in result


def test_convert_summary_to_markdown():
    from src.api.standalone_output import convert_output
    data = {
        "sections": [
            {"title": "Overview", "summary": "This is the overview.", "key_points": ["Point 1"]}
        ]
    }
    result = convert_output(data, "summary", "markdown")
    assert "## Overview" in result
    assert "Point 1" in result


def test_json_passthrough():
    from src.api.standalone_output import convert_output
    data = {"tables": [{"headers": ["A"], "rows": [["1"]]}]}
    result = convert_output(data, "table", "json")
    assert result == data
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/test_standalone_output.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement output converter**

```python
# src/api/standalone_output.py
"""Output format conversion for the DocWain Standalone API."""
from __future__ import annotations

import csv
import io
from typing import Any, Dict, List, Union


def convert_output(
    data: Dict[str, Any],
    mode: str,
    output_format: str,
) -> Union[str, Dict[str, Any]]:
    """Convert structured output to the requested format.

    Returns:
        str for csv/markdown/html, dict for json (passthrough).
    """
    if output_format == "json":
        return data

    converters = {
        ("table", "csv"): _tables_to_csv,
        ("table", "markdown"): _tables_to_markdown,
        ("table", "html"): _tables_to_html,
        ("entities", "csv"): _entities_to_csv,
        ("entities", "markdown"): _entities_to_markdown,
        ("entities", "html"): _entities_to_html,
        ("summary", "csv"): _summary_to_csv,
        ("summary", "markdown"): _summary_to_markdown,
        ("summary", "html"): _summary_to_html,
    }
    fn = converters.get((mode, output_format))
    if fn is None:
        return data
    return fn(data)


# ── Table converters ────────────────────────────────────────────

def _tables_to_csv(data: Dict[str, Any]) -> str:
    parts = []
    for table in data.get("tables", []):
        buf = io.StringIO()
        writer = csv.writer(buf)
        headers = table.get("headers", [])
        if headers:
            writer.writerow(headers)
        for row in table.get("rows", []):
            writer.writerow(row)
        parts.append(buf.getvalue().strip())
    return "\n\n".join(parts)


def _tables_to_markdown(data: Dict[str, Any]) -> str:
    parts = []
    for table in data.get("tables", []):
        headers = table.get("headers", [])
        rows = table.get("rows", [])
        if not headers:
            continue
        header_line = "| " + " | ".join(str(h) for h in headers) + " |"
        sep_line = "| " + " | ".join("---" for _ in headers) + " |"
        lines = [header_line, sep_line]
        for row in rows:
            lines.append("| " + " | ".join(str(c) for c in row) + " |")
        caption = table.get("caption")
        if caption:
            lines.insert(0, f"**{caption}**\n")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def _tables_to_html(data: Dict[str, Any]) -> str:
    parts = []
    for table in data.get("tables", []):
        headers = table.get("headers", [])
        rows = table.get("rows", [])
        lines = ["<table>"]
        if headers:
            lines.append("<thead><tr>")
            for h in headers:
                lines.append(f"<th>{_esc(h)}</th>")
            lines.append("</tr></thead>")
        lines.append("<tbody>")
        for row in rows:
            lines.append("<tr>")
            for c in row:
                lines.append(f"<td>{_esc(c)}</td>")
            lines.append("</tr>")
        lines.append("</tbody></table>")
        parts.append("".join(lines))
    return "\n".join(parts)


# ── Entity converters ───────────────────────────────────────────

def _entities_to_csv(data: Dict[str, Any]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["text", "type", "page", "confidence"])
    for e in data.get("entities", []):
        writer.writerow([e.get("text", ""), e.get("type", ""), e.get("page", ""), e.get("confidence", "")])
    return buf.getvalue().strip()


def _entities_to_markdown(data: Dict[str, Any]) -> str:
    lines = ["## Entities\n"]
    for e in data.get("entities", []):
        lines.append(f"- **{e.get('text', '')}** ({e.get('type', '')}) — page {e.get('page', '?')}, confidence {e.get('confidence', '?')}")
    return "\n".join(lines)


def _entities_to_html(data: Dict[str, Any]) -> str:
    lines = ["<dl>"]
    for e in data.get("entities", []):
        lines.append(f"<dt>{_esc(e.get('text', ''))}</dt>")
        lines.append(f"<dd>Type: {_esc(e.get('type', ''))}, Page: {e.get('page', '?')}</dd>")
    lines.append("</dl>")
    return "".join(lines)


# ── Summary converters ──────────────────────────────────────────

def _summary_to_csv(data: Dict[str, Any]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["title", "summary", "key_points"])
    for s in data.get("sections", []):
        writer.writerow([s.get("title", ""), s.get("summary", ""), "; ".join(s.get("key_points", []))])
    return buf.getvalue().strip()


def _summary_to_markdown(data: Dict[str, Any]) -> str:
    parts = []
    for s in data.get("sections", []):
        title = s.get("title", "Section")
        parts.append(f"## {title}\n")
        parts.append(s.get("summary", ""))
        kps = s.get("key_points", [])
        if kps:
            parts.append("\n**Key Points:**")
            for kp in kps:
                parts.append(f"- {kp}")
    return "\n".join(parts)


def _summary_to_html(data: Dict[str, Any]) -> str:
    parts = []
    for s in data.get("sections", []):
        parts.append(f"<section><h2>{_esc(s.get('title', ''))}</h2>")
        parts.append(f"<p>{_esc(s.get('summary', ''))}</p>")
        kps = s.get("key_points", [])
        if kps:
            parts.append("<ul>")
            for kp in kps:
                parts.append(f"<li>{_esc(kp)}</li>")
            parts.append("</ul>")
        parts.append("</section>")
    return "".join(parts)


def _esc(val: Any) -> str:
    """Minimal HTML escaping."""
    s = str(val)
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/test_standalone_output.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/api/standalone_output.py tests/test_standalone_output.py
git commit -m "feat(standalone): add output format conversion (json/csv/markdown/html)"
```

---

### Task 5: Webhook Delivery (`standalone_webhook.py`)

**Files:**
- Create: `src/api/standalone_webhook.py`
- Test: `tests/test_standalone_webhook.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_standalone_webhook.py
import pytest
from unittest.mock import patch, MagicMock
import json


def test_compute_signature():
    from src.api.standalone_webhook import compute_signature
    body = b'{"answer": "test"}'
    secret = "abc123"
    sig = compute_signature(body, secret)
    assert len(sig) == 64  # HMAC-SHA256 hex
    # Deterministic
    assert sig == compute_signature(body, secret)


def test_compute_signature_different_secrets():
    from src.api.standalone_webhook import compute_signature
    body = b'{"answer": "test"}'
    sig1 = compute_signature(body, "secret1")
    sig2 = compute_signature(body, "secret2")
    assert sig1 != sig2


def test_deliver_webhook_success():
    from src.api.standalone_webhook import deliver_webhook
    with patch("src.api.standalone_webhook.requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_post.return_value = mock_resp
        result = deliver_webhook(
            url="https://example.com/hook",
            payload={"answer": "test"},
            request_id="req-123",
            key_hash="abc",
        )
        assert result is True
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert "X-DocWain-Request-Id" in call_kwargs.kwargs.get("headers", call_kwargs[1].get("headers", {}))


def test_deliver_webhook_retries_on_failure():
    from src.api.standalone_webhook import deliver_webhook
    with patch("src.api.standalone_webhook.requests.post") as mock_post:
        mock_resp_fail = MagicMock()
        mock_resp_fail.status_code = 500
        mock_resp_ok = MagicMock()
        mock_resp_ok.status_code = 200
        mock_post.side_effect = [mock_resp_fail, mock_resp_ok]
        result = deliver_webhook(
            url="https://example.com/hook",
            payload={"answer": "test"},
            request_id="req-123",
            key_hash="abc",
            max_retries=3,
            backoff_base=0.01,  # fast for tests
        )
        assert result is True
        assert mock_post.call_count == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/test_standalone_webhook.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement webhook delivery**

```python
# src/api/standalone_webhook.py
"""Webhook callback delivery for the DocWain Standalone API."""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Optional

import requests

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="dw-webhook")


def compute_signature(body: bytes, secret: str) -> str:
    """HMAC-SHA256 signature of the body using the secret."""
    return hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()


def deliver_webhook(
    url: str,
    payload: Dict[str, Any],
    request_id: str,
    key_hash: str,
    max_retries: int = 3,
    backoff_base: float = 1.0,
) -> bool:
    """POST payload to callback URL with HMAC signature and retries.

    Returns True if delivery succeeded, False otherwise.
    """
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    signature = compute_signature(body, key_hash)
    headers = {
        "Content-Type": "application/json",
        "X-DocWain-Request-Id": request_id,
        "X-DocWain-Signature": signature,
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(url, data=body, headers=headers, timeout=30)
            if 200 <= resp.status_code < 300:
                logger.info("[WEBHOOK] Delivered %s to %s (attempt %d)", request_id, url, attempt + 1)
                return True
            logger.warning(
                "[WEBHOOK] %s returned %d (attempt %d/%d)",
                url, resp.status_code, attempt + 1, max_retries,
            )
        except Exception as exc:
            logger.warning("[WEBHOOK] %s failed (attempt %d/%d): %s", url, attempt + 1, max_retries, exc)

        if attempt < max_retries - 1:
            time.sleep(backoff_base * (5 ** attempt))

    logger.error("[WEBHOOK] Exhausted retries for %s (request %s)", url, request_id)
    return False


def deliver_webhook_async(
    url: str,
    payload: Dict[str, Any],
    request_id: str,
    key_hash: str,
    on_complete: Optional[Callable[[bool], None]] = None,
) -> None:
    """Submit webhook delivery to background thread pool."""
    def _task():
        success = deliver_webhook(url, payload, request_id, key_hash)
        if on_complete:
            on_complete(success)

    _executor.submit(_task)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/test_standalone_webhook.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/api/standalone_webhook.py tests/test_standalone_webhook.py
git commit -m "feat(standalone): add webhook delivery with HMAC signing and retries"
```

---

### Task 6: Core Processor (`standalone_processor.py`)

This is the heart of the standalone API — orchestrates the full DocWain pipeline.

**Files:**
- Create: `src/api/standalone_processor.py`
- Test: `tests/test_standalone_processor.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_standalone_processor.py
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import dataclass


def _make_extracted_doc():
    """Create a minimal ExtractedDocument for testing."""
    from src.api.pipeline_models import ExtractedDocument, Section, ChunkCandidate
    return ExtractedDocument(
        full_text="Revenue was $5M in Q1 2026.",
        sections=[Section(section_id="s1", title="Financials", level=1, start_page=1, end_page=1, text="Revenue was $5M in Q1 2026.")],
        tables=[],
        figures=[],
        chunk_candidates=[ChunkCandidate(text="Revenue was $5M in Q1 2026.", page=1, section_title="Financials", section_id="s1")],
    )


@patch("src.api.standalone_processor._get_document_extractor")
def test_extract_from_bytes_pdf(mock_get_extractor):
    from src.api.standalone_processor import extract_from_bytes
    mock_extractor = MagicMock()
    mock_extractor.extract_text_from_pdf.return_value = _make_extracted_doc()
    mock_get_extractor.return_value = mock_extractor

    result = extract_from_bytes(b"%PDF-fake", "report.pdf")
    assert result.full_text == "Revenue was $5M in Q1 2026."
    mock_extractor.extract_text_from_pdf.assert_called_once()


@patch("src.api.standalone_processor._get_document_extractor")
def test_extract_from_bytes_docx(mock_get_extractor):
    from src.api.standalone_processor import extract_from_bytes
    mock_extractor = MagicMock()
    mock_extractor.extract_text_from_docx.return_value = _make_extracted_doc()
    mock_get_extractor.return_value = mock_extractor

    result = extract_from_bytes(b"PK-fake", "report.docx")
    assert result is not None
    mock_extractor.extract_text_from_docx.assert_called_once()


def test_detect_file_type():
    from src.api.standalone_processor import detect_file_type
    assert detect_file_type("report.pdf", b"%PDF-1.5") == "pdf"
    assert detect_file_type("data.docx", b"PK") == "docx"
    assert detect_file_type("slides.pptx", b"PK") == "pptx"
    assert detect_file_type("image.png", b"\x89PNG") == "image"
    assert detect_file_type("image.jpg", b"\xff\xd8\xff") == "image"
    assert detect_file_type("data.csv", b"a,b,c") == "csv"
    assert detect_file_type("notes.txt", b"Hello") == "txt"


def test_build_structured_output_for_table_mode():
    from src.api.standalone_processor import build_structured_prompt
    prompt_parts = build_structured_prompt(
        mode="table",
        user_prompt="Extract all tables",
        document_text="Revenue $5M, Cost $3M",
        template=None,
    )
    assert "table" in prompt_parts["system_prompt"].lower() or "tabular" in prompt_parts["system_prompt"].lower()
    assert "Extract all tables" in prompt_parts["user_prompt"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/test_standalone_processor.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement the core processor**

```python
# src/api/standalone_processor.py
"""Core processing orchestrator for the DocWain Standalone API.

Runs the full pipeline: extract -> intelligence -> chunk -> embed -> retrieve -> reason -> generate.
Uses the same modules as the main app for intelligence parity.
"""
from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.api.config import Config
from src.api.pipeline_models import ExtractedDocument

logger = logging.getLogger(__name__)

# ── File type detection ─────────────────────────────────────────

_EXT_MAP = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".doc": "docx",
    ".pptx": "pptx",
    ".ppt": "pptx",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".tif": "image",
    ".tiff": "image",
    ".bmp": "image",
    ".webp": "image",
    ".csv": "csv",
    ".xlsx": "excel",
    ".xls": "excel",
    ".txt": "txt",
    ".md": "txt",
    ".rtf": "txt",
}


def detect_file_type(filename: str, content_bytes: bytes) -> str:
    """Detect file type from extension and magic bytes."""
    ext = Path(filename).suffix.lower()
    if ext in _EXT_MAP:
        return _EXT_MAP[ext]
    # Magic byte fallback
    if content_bytes[:4] == b"%PDF":
        return "pdf"
    if content_bytes[:2] == b"PK":
        return "docx"  # Could be docx/pptx/xlsx — extension wins
    if content_bytes[:4] == b"\x89PNG":
        return "image"
    if content_bytes[:3] == b"\xff\xd8\xff":
        return "image"
    return "txt"


# ── Extraction ──────────────────────────────────────────────────

_extractor = None


def _get_document_extractor():
    """Lazy singleton for DocumentExtractor."""
    global _extractor
    if _extractor is None:
        from src.api.dw_document_extractor import DocumentExtractor
        _extractor = DocumentExtractor()
    return _extractor


def extract_from_bytes(
    content: bytes,
    filename: str,
) -> ExtractedDocument:
    """Extract structured content from raw file bytes."""
    file_type = detect_file_type(filename, content)
    extractor = _get_document_extractor()

    if file_type == "pdf":
        return extractor.extract_text_from_pdf(content, filename=filename)
    elif file_type == "docx":
        return extractor.extract_text_from_docx(content, filename=filename)
    elif file_type == "pptx":
        return extractor.extract_text_from_pptx(content, filename=filename)
    elif file_type == "image":
        return extractor.extract_text_from_image(content, filename=filename)
    elif file_type in ("csv", "excel"):
        import pandas as pd
        import io
        if file_type == "csv":
            df = pd.read_csv(io.BytesIO(content))
        else:
            df = pd.read_excel(io.BytesIO(content))
        return extractor.extract_dataframe(df, filename=filename)
    else:
        return extractor.extract_text_from_txt(content, filename=filename)


# ── Intelligence pipeline ───────────────────────────────────────

def run_intelligence(
    extracted: ExtractedDocument,
    document_id: str,
) -> Dict[str, Any]:
    """Run V2 intelligence pipeline: classification, entities, summarization.

    Returns intelligence metadata dict.
    """
    intel_meta = {}
    try:
        from src.intelligence.integration import process_document_intelligence
        intel_result = process_document_intelligence(
            document_id=document_id,
            extracted_doc=extracted,
        )
        if isinstance(intel_result, dict):
            intel_meta = intel_result
    except ImportError:
        logger.debug("[STANDALONE] Intelligence integration not available")
    except Exception as exc:
        logger.warning("[STANDALONE] Intelligence pipeline error: %s", exc)
    return intel_meta


# ── Chunking + Embedding ────────────────────────────────────────

def chunk_and_embed(
    extracted: ExtractedDocument,
    document_id: str,
    collection_name: str,
) -> int:
    """Chunk the document and index vectors in Qdrant.

    Returns the number of chunks indexed.
    """
    from src.embedding.chunking.section_chunker import SectionChunker
    from src.embedding.model_loader import get_embedding_model
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct

    chunker = SectionChunker()
    chunks = chunker.chunk(extracted)

    if not chunks:
        # Fallback: treat full text as one chunk
        chunks = [{"text": extracted.full_text, "metadata": {"page": 1, "section": "full"}}]

    embedder, dim = get_embedding_model()
    texts = [c["text"] if isinstance(c, dict) else c.text for c in chunks]

    vectors = embedder.encode(texts, show_progress_bar=False, normalize_embeddings=True)

    qdrant = QdrantClient(url=Config.Qdrant.URL, api_key=Config.Qdrant.API)

    # Create collection
    try:
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
    except Exception:
        pass  # Collection may already exist

    # Upsert points
    points = []
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        meta = chunk.get("metadata", {}) if isinstance(chunk, dict) else {}
        text = chunk["text"] if isinstance(chunk, dict) else chunk.text
        points.append(PointStruct(
            id=i,
            vector=vector.tolist(),
            payload={
                "text": text,
                "document_id": document_id,
                "chunk_index": i,
                "page": meta.get("page", 1),
                "section": meta.get("section", ""),
                **meta,
            },
        ))

    if points:
        qdrant.upsert(collection_name=collection_name, points=points)

    return len(points)


def cleanup_collection(collection_name: str) -> None:
    """Delete a temporary Qdrant collection."""
    try:
        from qdrant_client import QdrantClient
        qdrant = QdrantClient(url=Config.Qdrant.URL, api_key=Config.Qdrant.API)
        qdrant.delete_collection(collection_name)
    except Exception as exc:
        logger.debug("[STANDALONE] Failed to cleanup collection %s: %s", collection_name, exc)


# ── Retrieval + Generation ──────────────────────────────────────

def retrieve_and_generate(
    query: str,
    collection_name: str,
    subscription_id: str,
    profile_id: str = "standalone",
    document_id: Optional[str] = None,
    system_prompt: Optional[str] = None,
    debug: bool = False,
) -> Dict[str, Any]:
    """Retrieve relevant chunks and generate answer using full CoreAgent pipeline.

    Uses the same execution router as the main app for intelligence parity.
    """
    from src.execution.router import execute_request
    from types import SimpleNamespace

    # Build a request-like object matching what execute_request expects
    request = SimpleNamespace(
        query=query,
        subscription_id=subscription_id,
        profile_id=profile_id,
        user_id="standalone-api",
        document_id=document_id,
        agent_name=None,
    )
    ctx = SimpleNamespace(
        query=query,
        subscription_id=subscription_id,
        profile_id=profile_id,
        user_id="standalone-api",
        session_id=None,
        collection_name=collection_name,
    )
    session_state = SimpleNamespace()

    result = execute_request(request, session_state, ctx, debug=debug)
    return result.answer if hasattr(result, "answer") else {}


# ── Structured mode prompts ─────────────────────────────────────

_MODE_SYSTEM_PROMPTS = {
    "qa": (
        "You are DocWain, a document intelligence system. Answer the user's question "
        "based solely on the provided document content. Cite specific pages and sections. "
        "If the information is not in the document, say so."
    ),
    "table": (
        "You are DocWain, a document intelligence system specialized in tabular data extraction. "
        "Extract all relevant tables from the document. Return each table as a JSON object with "
        "'headers' (list of column names), 'rows' (list of row arrays), 'page' (source page number), "
        "and 'caption' (table description). Wrap all tables in a JSON object: {\"tables\": [...]}. "
        "Preserve exact values, numbers, and currency symbols."
    ),
    "entities": (
        "You are DocWain, a document intelligence system specialized in entity extraction. "
        "Extract all named entities from the document. Return each entity as a JSON object with "
        "'text' (entity text), 'type' (PERSON, ORGANIZATION, LOCATION, DATE, MONEY, PRODUCT, etc.), "
        "'page' (source page), and 'confidence' (0.0-1.0). "
        "Wrap all entities in: {\"entities\": [...]}."
    ),
    "summary": (
        "You are DocWain, a document intelligence system specialized in summarization. "
        "Produce a structured summary of the document. Return each section as a JSON object with "
        "'title' (section heading), 'summary' (concise summary paragraph), and 'key_points' "
        "(list of bullet points). Wrap in: {\"sections\": [...]}."
    ),
}


def build_structured_prompt(
    mode: str,
    user_prompt: str,
    document_text: str,
    template: Optional[Any] = None,
) -> Dict[str, str]:
    """Build system + user prompt for the given mode, optionally enhanced by a template."""
    if template:
        from src.api.standalone_templates import apply_template
        parts = apply_template(template, user_prompt)
        return parts

    system = _MODE_SYSTEM_PROMPTS.get(mode, _MODE_SYSTEM_PROMPTS["qa"])
    return {
        "system_prompt": system,
        "user_prompt": user_prompt,
    }


# ── Full pipeline orchestrator ──────────────────────────────────

def process_document(
    content: bytes,
    filename: str,
    prompt: str,
    mode: str = "qa",
    subscription_id: str = "standalone",
    persist: bool = False,
    template: Optional[Any] = None,
    confidence_threshold: float = 0.0,
) -> Dict[str, Any]:
    """Run the full DocWain pipeline on a single document.

    Returns a dict matching ProcessResponse fields.
    """
    request_id = f"req-{uuid.uuid4().hex[:12]}"
    document_id = f"doc-{uuid.uuid4().hex[:12]}"
    timings: Dict[str, int] = {}

    # 1. Extract
    t0 = time.time()
    extracted = extract_from_bytes(content, filename)
    timings["extraction_ms"] = int((time.time() - t0) * 1000)

    # 2. Intelligence
    t0 = time.time()
    intel_meta = run_intelligence(extracted, document_id)
    timings["intelligence_ms"] = int((time.time() - t0) * 1000)

    # 3. Chunk + Embed
    collection_name = f"dw_standalone_{request_id}" if not persist else f"dw_standalone_{subscription_id}_{document_id}"
    t0 = time.time()
    num_chunks = chunk_and_embed(extracted, document_id, collection_name)
    timings["retrieval_ms"] = int((time.time() - t0) * 1000)  # includes embedding time

    # 4. Build prompt
    prompt_parts = build_structured_prompt(mode, prompt, extracted.full_text, template)

    # 5. Retrieve + Generate
    t0 = time.time()
    combined_query = prompt_parts["user_prompt"]
    if prompt_parts.get("system_prompt"):
        combined_query = prompt_parts["system_prompt"] + "\n\n" + combined_query

    answer = retrieve_and_generate(
        query=combined_query,
        collection_name=collection_name,
        subscription_id=subscription_id,
        document_id=document_id,
    )
    timings["generation_ms"] = int((time.time() - t0) * 1000)
    timings["total_ms"] = sum(timings.values())

    # 6. Extract structured output for non-qa modes
    structured_output = None
    if mode != "qa":
        structured_output = _parse_structured_response(answer.get("response", ""), mode)

    # 7. Confidence gating
    confidence = answer.get("metadata", {}).get("confidence", 0.0)
    if not isinstance(confidence, (int, float)):
        confidence = 0.0
    grounded = answer.get("grounded", False)
    low_confidence = confidence_threshold > 0 and confidence < confidence_threshold
    low_confidence_reasons = []
    if low_confidence:
        low_confidence_reasons = _build_low_confidence_reasons(answer, confidence)

    # 8. Finetune capture
    _capture_learning_signal(
        query=prompt,
        context=extracted.full_text[:2000],
        answer_text=answer.get("response", ""),
        sources=answer.get("sources", []),
        confidence=confidence,
        mode=mode,
        template_name=getattr(template, "name", None) if template else None,
        request_id=request_id,
    )

    # 9. Cleanup temp collection (unless persisting)
    if not persist:
        cleanup_collection(collection_name)

    return {
        "request_id": request_id,
        "status": "completed",
        "answer": None if low_confidence else answer.get("response", ""),
        "sources": answer.get("sources", []),
        "confidence": confidence,
        "grounded": grounded,
        "low_confidence": low_confidence,
        "low_confidence_reasons": low_confidence_reasons,
        "structured_output": structured_output,
        "document_id": document_id if persist else None,
        "partial_answer": answer.get("response", "") if low_confidence else None,
        "usage": timings,
    }


def query_persisted_document(
    document_id: str,
    prompt: str,
    subscription_id: str,
    mode: str = "qa",
    confidence_threshold: float = 0.0,
) -> Dict[str, Any]:
    """Query a previously persisted and indexed document."""
    request_id = f"req-{uuid.uuid4().hex[:12]}"
    collection_name = f"dw_standalone_{subscription_id}_{document_id}"
    timings: Dict[str, int] = {}

    prompt_parts = build_structured_prompt(mode, prompt, "", None)
    combined_query = prompt_parts["system_prompt"] + "\n\n" + prompt_parts["user_prompt"]

    t0 = time.time()
    answer = retrieve_and_generate(
        query=combined_query,
        collection_name=collection_name,
        subscription_id=subscription_id,
        document_id=document_id,
    )
    timings["generation_ms"] = int((time.time() - t0) * 1000)
    timings["total_ms"] = timings["generation_ms"]

    confidence = answer.get("metadata", {}).get("confidence", 0.0)
    if not isinstance(confidence, (int, float)):
        confidence = 0.0
    grounded = answer.get("grounded", False)
    low_confidence = confidence_threshold > 0 and confidence < confidence_threshold

    structured_output = None
    if mode != "qa":
        structured_output = _parse_structured_response(answer.get("response", ""), mode)

    _capture_learning_signal(
        query=prompt,
        context="",
        answer_text=answer.get("response", ""),
        sources=answer.get("sources", []),
        confidence=confidence,
        mode=mode,
        request_id=request_id,
    )

    return {
        "request_id": request_id,
        "status": "completed",
        "answer": None if low_confidence else answer.get("response", ""),
        "sources": answer.get("sources", []),
        "confidence": confidence,
        "grounded": grounded,
        "low_confidence": low_confidence,
        "low_confidence_reasons": _build_low_confidence_reasons(answer, confidence) if low_confidence else [],
        "structured_output": structured_output,
        "document_id": document_id,
        "partial_answer": answer.get("response", "") if low_confidence else None,
        "usage": timings,
    }


# ── Helpers ─────────────────────────────────────────────────────

def _parse_structured_response(response_text: str, mode: str) -> Optional[Dict[str, Any]]:
    """Try to parse JSON structured output from LLM response."""
    import json
    if not response_text:
        return None
    # Try to find JSON block in response
    try:
        # Look for ```json blocks first
        import re
        json_match = re.search(r"```json\s*(.*?)```", response_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))
        # Try direct parse
        return json.loads(response_text)
    except (json.JSONDecodeError, ValueError):
        pass
    # Try to find { ... } block
    try:
        start = response_text.index("{")
        end = response_text.rindex("}") + 1
        return json.loads(response_text[start:end])
    except (ValueError, json.JSONDecodeError):
        return None


def _build_low_confidence_reasons(
    answer: Dict[str, Any],
    confidence: float,
) -> List[str]:
    """Generate human-readable low-confidence reasons."""
    reasons = []
    if confidence < 0.3:
        reasons.append("Very low confidence — document content may not be relevant to the query")
    elif confidence < 0.5:
        reasons.append("Low retrieval relevance — query topic may not be covered in the document")
    if not answer.get("grounded", False):
        reasons.append("Response could not be grounded in source content")
    if not answer.get("context_found", True):
        reasons.append("No relevant context found in the document")
    meta = answer.get("metadata", {})
    if meta.get("ocr_quality") and meta["ocr_quality"] < 0.75:
        reasons.append(f"Document OCR quality is poor (estimated accuracy: {int(meta['ocr_quality'] * 100)}%)")
    if not reasons:
        reasons.append(f"Confidence ({confidence:.2f}) is below the requested threshold")
    return reasons


def _capture_learning_signal(
    query: str,
    context: str,
    answer_text: str,
    sources: list,
    confidence: float,
    mode: str,
    template_name: Optional[str] = None,
    request_id: str = "",
) -> None:
    """Record to finetune buffer — same as main app."""
    try:
        from src.api.learning_signals import LearningSignalStore
        store = LearningSignalStore()
        metadata = {
            "source": "standalone_api",
            "mode": mode,
            "template": template_name,
            "request_id": request_id,
        }
        if confidence >= 0.7:
            store.record_high_quality(
                query=query,
                context=context,
                answer=answer_text,
                sources=sources,
                metadata=metadata,
            )
        else:
            store.record_low_confidence(
                query=query,
                context=context,
                answer=answer_text,
                reason=f"confidence={confidence:.2f}",
                metadata=metadata,
            )
    except Exception as exc:
        logger.debug("[STANDALONE] Failed to capture learning signal: %s", exc)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/test_standalone_processor.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/api/standalone_processor.py tests/test_standalone_processor.py
git commit -m "feat(standalone): add core processor orchestrating full DocWain pipeline"
```

---

### Task 7: Multi-Document & Batch Processing (`standalone_multi.py`)

**Files:**
- Create: `src/api/standalone_multi.py`
- Test: `tests/test_standalone_multi.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_standalone_multi.py
import pytest
from unittest.mock import patch, MagicMock


@patch("src.api.standalone_multi.process_document")
def test_process_batch_returns_per_file_results(mock_process):
    from src.api.standalone_multi import process_batch
    mock_process.return_value = {
        "request_id": "req-1",
        "status": "completed",
        "answer": "Answer",
        "confidence": 0.9,
        "sources": [],
        "structured_output": None,
        "usage": {"total_ms": 1000},
    }
    files = [
        {"filename": "doc1.pdf", "content": b"%PDF-fake1"},
        {"filename": "doc2.pdf", "content": b"%PDF-fake2"},
    ]
    result = process_batch(files, prompt="Summarize", mode="qa", subscription_id="sub-1")
    assert result["status"] == "completed"
    assert len(result["results"]) == 2
    assert result["summary"]["total"] == 2
    assert result["summary"]["completed"] == 2


@patch("src.api.standalone_multi.process_document")
def test_process_batch_handles_errors(mock_process):
    from src.api.standalone_multi import process_batch
    mock_process.side_effect = [
        {"request_id": "r1", "status": "completed", "answer": "OK", "confidence": 0.9, "sources": [], "structured_output": None, "usage": {"total_ms": 500}},
        Exception("Extraction failed"),
    ]
    files = [
        {"filename": "ok.pdf", "content": b"%PDF-ok"},
        {"filename": "bad.pdf", "content": b"garbage"},
    ]
    result = process_batch(files, prompt="Summarize", mode="qa", subscription_id="sub-1")
    assert result["summary"]["completed"] == 1
    assert result["summary"]["failed"] == 1
    assert result["results"][1]["status"] == "error"


@patch("src.api.standalone_multi.process_document")
def test_process_multi_merges_results(mock_process):
    from src.api.standalone_multi import process_multi_documents
    mock_process.return_value = {
        "request_id": "r1",
        "status": "completed",
        "answer": "Cross-doc answer",
        "confidence": 0.85,
        "sources": [{"page": 1, "document": "doc1.pdf"}],
        "structured_output": None,
        "usage": {"total_ms": 2000},
    }
    files = [
        {"filename": "doc1.pdf", "content": b"%PDF-1"},
        {"filename": "doc2.pdf", "content": b"%PDF-2"},
    ]
    result = process_multi_documents(
        files=files,
        document_ids=None,
        prompt="Compare these documents",
        mode="qa",
        subscription_id="sub-1",
    )
    assert result["status"] == "completed"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/test_standalone_multi.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement multi-doc and batch processing**

```python
# src/api/standalone_multi.py
"""Multi-document and batch processing for the DocWain Standalone API."""
from __future__ import annotations

import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from src.api.standalone_processor import (
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

logger = logging.getLogger(__name__)


def process_batch(
    files: List[Dict[str, Any]],
    prompt: str,
    mode: str = "qa",
    subscription_id: str = "standalone",
    output_format: str = "json",
    template: Optional[Any] = None,
) -> Dict[str, Any]:
    """Process multiple files independently with the same prompt.

    Each file is processed in its own thread. Results are collected per file.
    """
    batch_id = f"batch-{uuid.uuid4().hex[:12]}"
    results = []
    total_ms = 0

    def _process_one(file_info: Dict[str, Any]) -> Dict[str, Any]:
        try:
            result = process_document(
                content=file_info["content"],
                filename=file_info["filename"],
                prompt=prompt,
                mode=mode,
                subscription_id=subscription_id,
                template=template,
            )
            return {
                "filename": file_info["filename"],
                "status": "completed",
                "answer": result.get("answer"),
                "confidence": result.get("confidence", 0.0),
                "structured_output": result.get("structured_output"),
            }
        except Exception as exc:
            logger.warning("[BATCH] Failed to process %s: %s", file_info["filename"], exc)
            return {
                "filename": file_info["filename"],
                "status": "error",
                "error": str(exc),
            }

    with ThreadPoolExecutor(max_workers=min(len(files), 4)) as pool:
        futures = {pool.submit(_process_one, f): f for f in files}
        for future in as_completed(futures):
            results.append(future.result())

    # Sort results to match input order
    file_order = {f["filename"]: i for i, f in enumerate(files)}
    results.sort(key=lambda r: file_order.get(r["filename"], 999))

    completed = sum(1 for r in results if r["status"] == "completed")
    failed = sum(1 for r in results if r["status"] == "error")

    return {
        "batch_id": batch_id,
        "status": "completed",
        "results": results,
        "summary": {"total": len(files), "completed": completed, "failed": failed},
        "usage": {"total_ms": 0},
    }


def process_multi_documents(
    files: Optional[List[Dict[str, Any]]],
    document_ids: Optional[List[str]],
    prompt: str,
    mode: str = "qa",
    subscription_id: str = "standalone",
    template: Optional[Any] = None,
) -> Dict[str, Any]:
    """Process multiple documents for cross-document Q&A.

    Extracts and embeds all docs into a shared temporary collection,
    then runs retrieval + generation across all of them.
    """
    request_id = f"req-{uuid.uuid4().hex[:12]}"
    shared_collection = f"dw_standalone_multi_{request_id}"
    timings: Dict[str, int] = {}
    all_sources_meta: List[Dict[str, Any]] = []

    t0 = time.time()

    # Extract and embed uploaded files
    if files:
        for file_info in files:
            doc_id = f"doc-{uuid.uuid4().hex[:8]}"
            extracted = extract_from_bytes(file_info["content"], file_info["filename"])
            run_intelligence(extracted, doc_id)
            chunk_and_embed(extracted, doc_id, shared_collection)
            all_sources_meta.append({"document_id": doc_id, "filename": file_info["filename"]})

    # Add persisted document IDs to the same collection
    if document_ids:
        for doc_id in document_ids:
            # Persisted docs are already in their own collection — we query across both
            all_sources_meta.append({"document_id": doc_id, "filename": doc_id})

    timings["extraction_ms"] = int((time.time() - t0) * 1000)

    # Build prompt
    prompt_parts = build_structured_prompt(mode, prompt, "", template)
    combined_query = prompt_parts["system_prompt"] + "\n\n" + prompt_parts["user_prompt"]

    # Retrieve + generate across shared collection
    t0 = time.time()
    answer = retrieve_and_generate(
        query=combined_query,
        collection_name=shared_collection,
        subscription_id=subscription_id,
    )
    timings["generation_ms"] = int((time.time() - t0) * 1000)
    timings["total_ms"] = sum(timings.values())

    confidence = answer.get("metadata", {}).get("confidence", 0.0)
    if not isinstance(confidence, (int, float)):
        confidence = 0.0

    structured_output = None
    if mode != "qa":
        structured_output = _parse_structured_response(answer.get("response", ""), mode)

    _capture_learning_signal(
        query=prompt,
        context="multi-document",
        answer_text=answer.get("response", ""),
        sources=answer.get("sources", []),
        confidence=confidence,
        mode=mode,
        request_id=request_id,
    )

    # Cleanup
    cleanup_collection(shared_collection)

    return {
        "request_id": request_id,
        "status": "completed",
        "answer": answer.get("response", ""),
        "sources": answer.get("sources", []),
        "confidence": confidence,
        "grounded": answer.get("grounded", False),
        "low_confidence": False,
        "low_confidence_reasons": [],
        "structured_output": structured_output,
        "document_id": None,
        "usage": timings,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/test_standalone_multi.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/api/standalone_multi.py tests/test_standalone_multi.py
git commit -m "feat(standalone): add multi-document and batch processing"
```

---

### Task 8: Config + Router + Mount (`standalone_api.py`)

**Files:**
- Create: `src/api/standalone_api.py`
- Modify: `src/api/config.py:198+`
- Modify: `src/main.py:209`
- Test: `tests/test_standalone_api.py`

- [ ] **Step 1: Write failing integration tests**

```python
# tests/test_standalone_api.py
import pytest
from unittest.mock import patch, MagicMock


def test_standalone_router_exists():
    from src.api.standalone_api import standalone_router
    assert standalone_router is not None
    assert standalone_router.prefix == "/v1/docwain"


def test_templates_endpoint():
    from src.api.standalone_api import standalone_router
    routes = [r.path for r in standalone_router.routes]
    assert "/templates" in routes or any("/templates" in r for r in routes)


def test_process_endpoint_registered():
    from src.api.standalone_api import standalone_router
    routes = [r.path for r in standalone_router.routes]
    assert "/process" in routes or any("/process" in r for r in routes)


def test_config_standalone_section():
    from src.api.config import Config
    assert hasattr(Config, "Standalone")
    assert hasattr(Config.Standalone, "ENABLED")
    assert hasattr(Config.Standalone, "MAX_BATCH_FILES")
    assert hasattr(Config.Standalone, "MAX_FILE_SIZE_MB")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/test_standalone_api.py -v`
Expected: FAIL — `ModuleNotFoundError` or `AttributeError`

- [ ] **Step 3: Add Config.Standalone to config.py**

Add after the `Redis` class (around line 199) in `src/api/config.py`:

```python
    class Standalone:
        ENABLED = os.getenv("DOCWAIN_STANDALONE_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
        TEMP_COLLECTION_TTL = int(os.getenv("STANDALONE_TEMP_TTL", "3600"))
        MAX_BATCH_FILES = int(os.getenv("STANDALONE_MAX_BATCH", "10"))
        MAX_FILE_SIZE_MB = int(os.getenv("STANDALONE_MAX_FILE_MB", "50"))
        WEBHOOK_MAX_WORKERS = int(os.getenv("STANDALONE_WEBHOOK_WORKERS", "4"))
        WEBHOOK_MAX_RETRIES = int(os.getenv("STANDALONE_WEBHOOK_RETRIES", "3"))
        API_KEYS_COLLECTION = os.getenv("STANDALONE_KEYS_COLLECTION", "api_keys")
        REQUESTS_COLLECTION = os.getenv("STANDALONE_REQUESTS_COLLECTION", "standalone_requests")
```

- [ ] **Step 4: Create the router**

```python
# src/api/standalone_api.py
"""DocWain Standalone API — authenticated endpoints for document intelligence."""
from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from src.api.config import Config
from src.api.standalone_auth import require_api_key, track_usage, track_document_processed
from src.api.standalone_schemas import (
    AsyncAcceptedResponse,
    BatchResponse,
    DocumentStatusResponse,
    DocumentUploadResponse,
    ExtractResponse,
    ProcessResponse,
    StandaloneErrorResponse,
    TemplateInfo,
    TemplatesResponse,
    UsageResponse,
)

logger = logging.getLogger(__name__)

standalone_router = APIRouter(prefix="/v1/docwain")


# ── POST /process — one-shot document processing ───────────────

@standalone_router.post("/process", response_model=ProcessResponse)
async def process_document_endpoint(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    mode: str = Form("qa"),
    output_format: str = Form("json"),
    persist: bool = Form(False),
    stream: bool = Form(False),
    template: Optional[str] = Form(None),
    confidence_threshold: float = Form(0.0),
    callback_url: Optional[str] = Form(None),
    api_key: Dict[str, Any] = Depends(require_api_key),
):
    """One-shot: send document + prompt, get answer."""
    _validate_mode(mode)
    _validate_output_format(output_format)
    content = await file.read()
    _validate_file_size(content)

    # Resolve template
    tmpl = None
    if template:
        from src.api.standalone_templates import get_template
        tmpl = get_template(template)
        if tmpl is None:
            raise HTTPException(status_code=422, detail={"error": {"code": "TEMPLATE_NOT_FOUND", "message": f"Template '{template}' not found"}})

    # Async webhook mode
    if callback_url:
        request_id = f"req-{uuid.uuid4().hex[:12]}"
        _dispatch_async(
            callback_url=callback_url,
            request_id=request_id,
            api_key=api_key,
            fn_name="process",
            kwargs=dict(content=content, filename=file.filename, prompt=prompt, mode=mode,
                       subscription_id=api_key["subscription_id"], persist=persist, template=tmpl,
                       confidence_threshold=confidence_threshold),
        )
        track_usage(api_key["keys_collection"], api_key["key_hash"], "/process", mode)
        return JSONResponse(
            status_code=202,
            content=AsyncAcceptedResponse(request_id=request_id, status="processing").model_dump(),
        )

    # Synchronous processing
    from src.api.standalone_processor import process_document as do_process
    result = do_process(
        content=content,
        filename=file.filename or "uploaded_file",
        prompt=prompt,
        mode=mode,
        subscription_id=api_key["subscription_id"],
        persist=persist,
        template=tmpl,
        confidence_threshold=confidence_threshold,
    )

    # Output format conversion
    if output_format != "json" and result.get("structured_output"):
        from src.api.standalone_output import convert_output
        result["structured_output"] = convert_output(result["structured_output"], mode, output_format)

    result["output_format"] = output_format
    track_usage(api_key["keys_collection"], api_key["key_hash"], "/process", mode)
    if persist:
        track_document_processed(api_key["keys_collection"], api_key["key_hash"])

    _log_request(api_key, "/process", mode, result)
    return ProcessResponse(**result)


# ── POST /process/multi — multi-document processing ────────────

@standalone_router.post("/process/multi", response_model=ProcessResponse)
async def process_multi_endpoint(
    prompt: str = Form(...),
    files: List[UploadFile] = File(None),
    document_ids: Optional[str] = Form(None),
    mode: str = Form("qa"),
    output_format: str = Form("json"),
    callback_url: Optional[str] = Form(None),
    api_key: Dict[str, Any] = Depends(require_api_key),
):
    """Multi-document processing: cross-document Q&A and comparison."""
    _validate_mode(mode)
    import json as json_mod

    file_list = []
    if files:
        for f in files:
            content = await f.read()
            _validate_file_size(content)
            file_list.append({"filename": f.filename or "file", "content": content})

    doc_ids = None
    if document_ids:
        try:
            doc_ids = json_mod.loads(document_ids)
        except Exception:
            doc_ids = [document_ids]

    if not file_list and not doc_ids:
        raise HTTPException(status_code=422, detail={"error": {"code": "VALIDATION_ERROR", "message": "Provide files or document_ids"}})

    from src.api.standalone_multi import process_multi_documents
    result = process_multi_documents(
        files=file_list or None,
        document_ids=doc_ids,
        prompt=prompt,
        mode=mode,
        subscription_id=api_key["subscription_id"],
    )

    if output_format != "json" and result.get("structured_output"):
        from src.api.standalone_output import convert_output
        result["structured_output"] = convert_output(result["structured_output"], mode, output_format)

    result["output_format"] = output_format
    track_usage(api_key["keys_collection"], api_key["key_hash"], "/process/multi", mode)
    _log_request(api_key, "/process/multi", mode, result)
    return ProcessResponse(**result)


# ── POST /batch — bulk processing ──────────────────────────────

@standalone_router.post("/batch", response_model=BatchResponse)
async def batch_endpoint(
    prompt: str = Form(...),
    files: List[UploadFile] = File(...),
    mode: str = Form("qa"),
    output_format: str = Form("json"),
    callback_url: Optional[str] = Form(None),
    api_key: Dict[str, Any] = Depends(require_api_key),
):
    """Bulk: same prompt applied to many files independently."""
    _validate_mode(mode)
    max_batch = Config.Standalone.MAX_BATCH_FILES

    if len(files) > max_batch and not callback_url:
        raise HTTPException(
            status_code=413,
            detail={"error": {"code": "BATCH_TOO_LARGE", "message": f"Max {max_batch} files without callback_url"}},
        )

    file_list = []
    for f in files:
        content = await f.read()
        _validate_file_size(content)
        file_list.append({"filename": f.filename or "file", "content": content})

    from src.api.standalone_multi import process_batch
    result = process_batch(
        files=file_list,
        prompt=prompt,
        mode=mode,
        subscription_id=api_key["subscription_id"],
        output_format=output_format,
    )

    track_usage(api_key["keys_collection"], api_key["key_hash"], "/batch", mode)
    _log_request(api_key, "/batch", mode, result)
    return BatchResponse(**result)


# ── POST /extract — structured extraction ───────────────────────

@standalone_router.post("/extract", response_model=ExtractResponse)
async def extract_endpoint(
    file: UploadFile = File(...),
    mode: str = Form(...),
    prompt: Optional[str] = Form(None),
    output_format: str = Form("json"),
    template: Optional[str] = Form(None),
    api_key: Dict[str, Any] = Depends(require_api_key),
):
    """Structured extraction: tables, entities, or summaries."""
    if mode not in ("table", "entities", "summary"):
        raise HTTPException(status_code=422, detail={"error": {"code": "VALIDATION_ERROR", "message": "mode must be table, entities, or summary"}})

    content = await file.read()
    _validate_file_size(content)

    tmpl = None
    if template:
        from src.api.standalone_templates import get_template
        tmpl = get_template(template)

    from src.api.standalone_processor import process_document as do_process
    result = do_process(
        content=content,
        filename=file.filename or "file",
        prompt=prompt or f"Extract all {mode} data from this document",
        mode=mode,
        subscription_id=api_key["subscription_id"],
        template=tmpl,
    )

    structured = result.get("structured_output")
    if output_format != "json" and structured:
        from src.api.standalone_output import convert_output
        structured = convert_output(structured, mode, output_format)

    track_usage(api_key["keys_collection"], api_key["key_hash"], "/extract", mode)

    return ExtractResponse(
        request_id=result["request_id"],
        mode=mode,
        result=structured,
        metadata={
            "extraction_ms": result.get("usage", {}).get("extraction_ms", 0),
            "intelligence_ms": result.get("usage", {}).get("intelligence_ms", 0),
        },
    )


# ── POST /documents — persist a document ───────────────────────

@standalone_router.post("/documents", response_model=DocumentUploadResponse)
async def upload_document_endpoint(
    file: UploadFile = File(...),
    name: Optional[str] = Form(None),
    api_key: Dict[str, Any] = Depends(require_api_key),
):
    """Upload and persist a document for repeated queries."""
    content = await file.read()
    _validate_file_size(content)

    from src.api.standalone_processor import extract_from_bytes, run_intelligence, chunk_and_embed
    import uuid as uuid_mod

    document_id = f"doc-{uuid_mod.uuid4().hex[:12]}"
    subscription_id = api_key["subscription_id"]
    collection_name = f"dw_standalone_{subscription_id}_{document_id}"

    # Process synchronously (extract + embed)
    extracted = extract_from_bytes(content, file.filename or "file")
    run_intelligence(extracted, document_id)
    chunk_and_embed(extracted, document_id, collection_name)

    # Record in MongoDB
    _save_document_record(
        document_id=document_id,
        name=name or file.filename,
        subscription_id=subscription_id,
        key_hash=api_key["key_hash"],
        pages=len(extracted.sections) or 1,
        document_type=extracted.doc_type,
    )

    track_usage(api_key["keys_collection"], api_key["key_hash"], "/documents", "upload")
    track_document_processed(api_key["keys_collection"], api_key["key_hash"])

    return DocumentUploadResponse(
        document_id=document_id,
        name=name or file.filename,
        status="ready",
        created_at=datetime.now(timezone.utc).isoformat(),
    )


# ── GET /documents/{doc_id}/status ──────────────────────────────

@standalone_router.get("/documents/{doc_id}/status", response_model=DocumentStatusResponse)
async def document_status_endpoint(
    doc_id: str,
    api_key: Dict[str, Any] = Depends(require_api_key),
):
    """Check document processing status."""
    doc = _get_document_record(doc_id, api_key["subscription_id"])
    if doc is None:
        raise HTTPException(status_code=404, detail={"error": {"code": "DOCUMENT_NOT_FOUND", "message": f"Document {doc_id} not found"}})
    return DocumentStatusResponse(
        document_id=doc_id,
        status=doc.get("status", "unknown"),
        name=doc.get("name"),
        pages=doc.get("pages"),
        document_type=doc.get("document_type"),
        created_at=doc.get("created_at"),
        ready_at=doc.get("ready_at"),
    )


# ── POST /query — query persisted document ──────────────────────

@standalone_router.post("/query", response_model=ProcessResponse)
async def query_endpoint(
    prompt: str = Form(...),
    document_id: Optional[str] = Form(None),
    document_ids: Optional[str] = Form(None),
    mode: str = Form("qa"),
    output_format: str = Form("json"),
    stream: bool = Form(False),
    confidence_threshold: float = Form(0.0),
    api_key: Dict[str, Any] = Depends(require_api_key),
):
    """Query previously persisted documents."""
    import json as json_mod
    _validate_mode(mode)

    # Handle multiple document IDs
    doc_ids = []
    if document_id:
        doc_ids.append(document_id)
    if document_ids:
        try:
            doc_ids.extend(json_mod.loads(document_ids))
        except Exception:
            doc_ids.append(document_ids)

    if not doc_ids:
        raise HTTPException(status_code=422, detail={"error": {"code": "VALIDATION_ERROR", "message": "Provide document_id or document_ids"}})

    if len(doc_ids) == 1:
        from src.api.standalone_processor import query_persisted_document
        result = query_persisted_document(
            document_id=doc_ids[0],
            prompt=prompt,
            subscription_id=api_key["subscription_id"],
            mode=mode,
            confidence_threshold=confidence_threshold,
        )
    else:
        from src.api.standalone_multi import process_multi_documents
        result = process_multi_documents(
            files=None,
            document_ids=doc_ids,
            prompt=prompt,
            mode=mode,
            subscription_id=api_key["subscription_id"],
        )

    if output_format != "json" and result.get("structured_output"):
        from src.api.standalone_output import convert_output
        result["structured_output"] = convert_output(result["structured_output"], mode, output_format)

    result["output_format"] = output_format
    track_usage(api_key["keys_collection"], api_key["key_hash"], "/query", mode)
    _log_request(api_key, "/query", mode, result)
    return ProcessResponse(**result)


# ── GET /usage — audit trail ────────────────────────────────────

@standalone_router.get("/usage", response_model=UsageResponse)
async def usage_endpoint(
    api_key: Dict[str, Any] = Depends(require_api_key),
):
    """Get usage statistics and audit trail for this API key."""
    from pymongo import MongoClient
    client = MongoClient(Config.MongoDB.URI)
    db = client[Config.MongoDB.DB]
    requests_col = db[Config.Standalone.REQUESTS_COLLECTION]

    recent = list(
        requests_col.find(
            {"api_key_hash": api_key["key_hash"]},
            {"_id": 0, "request_id": 1, "endpoint": 1, "mode": 1, "timestamp": 1, "latency_ms": 1},
        )
        .sort("timestamp", -1)
        .limit(20)
    )

    # Get key usage from keys collection
    keys_col = api_key["keys_collection"]
    key_doc = keys_col.find_one({"key_hash": api_key["key_hash"]})
    usage = key_doc.get("usage", {}) if key_doc else {}

    return UsageResponse(
        api_key_name=api_key["name"],
        period="all time",
        totals={
            "requests": usage.get("total_requests", 0),
            "documents_processed": usage.get("documents_processed", 0),
            "queries": usage.get("total_requests", 0),
        },
        by_endpoint=usage.get("by_endpoint", {}),
        by_mode=usage.get("by_mode", {}),
        recent=recent,
    )


# ── GET /templates — list templates ─────────────────────────────

@standalone_router.get("/templates", response_model=TemplatesResponse)
async def templates_endpoint():
    """List available prompt templates."""
    from src.api.standalone_templates import list_templates
    templates = list_templates()
    return TemplatesResponse(
        templates=[
            TemplateInfo(name=t.name, description=t.description, modes=t.modes)
            for t in templates
        ]
    )


# ── Helpers ─────────────────────────────────────────────────────

def _validate_mode(mode: str) -> None:
    if mode not in ("qa", "table", "entities", "summary"):
        raise HTTPException(status_code=422, detail={"error": {"code": "VALIDATION_ERROR", "message": "mode must be qa, table, entities, or summary"}})


def _validate_output_format(fmt: str) -> None:
    if fmt not in ("json", "markdown", "csv", "html"):
        raise HTTPException(status_code=422, detail={"error": {"code": "VALIDATION_ERROR", "message": "output_format must be json, markdown, csv, or html"}})


def _validate_file_size(content: bytes) -> None:
    max_mb = Config.Standalone.MAX_FILE_SIZE_MB
    if len(content) > max_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail={"error": {"code": "FILE_TOO_LARGE", "message": f"File exceeds {max_mb}MB limit"}})


def _dispatch_async(
    callback_url: str,
    request_id: str,
    api_key: Dict[str, Any],
    fn_name: str,
    kwargs: Dict[str, Any],
) -> None:
    """Run processing in background and deliver result via webhook."""
    from src.api.standalone_webhook import deliver_webhook_async
    from concurrent.futures import ThreadPoolExecutor
    import threading

    def _bg():
        try:
            from src.api.standalone_processor import process_document as do_process
            result = do_process(**kwargs)
            result["request_id"] = request_id
            deliver_webhook_async(callback_url, result, request_id, api_key["key_hash"])
            _log_request(api_key, f"/{fn_name}", kwargs.get("mode", "qa"), result)
        except Exception as exc:
            logger.error("[STANDALONE] Async processing failed: %s", exc)
            deliver_webhook_async(
                callback_url,
                {"request_id": request_id, "status": "error", "error": str(exc)},
                request_id,
                api_key["key_hash"],
            )

    threading.Thread(target=_bg, daemon=True).start()


def _save_document_record(
    document_id: str,
    name: str,
    subscription_id: str,
    key_hash: str,
    pages: int,
    document_type: Optional[str],
) -> None:
    """Save document metadata to MongoDB."""
    try:
        from pymongo import MongoClient
        client = MongoClient(Config.MongoDB.URI)
        db = client[Config.MongoDB.DB]
        col = db[Config.Standalone.REQUESTS_COLLECTION]
        col.insert_one({
            "document_id": document_id,
            "type": "document",
            "name": name,
            "subscription_id": subscription_id,
            "api_key_hash": key_hash,
            "status": "ready",
            "pages": pages,
            "document_type": document_type,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "ready_at": datetime.now(timezone.utc).isoformat(),
        })
    except Exception as exc:
        logger.warning("[STANDALONE] Failed to save document record: %s", exc)


def _get_document_record(document_id: str, subscription_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve document metadata from MongoDB."""
    try:
        from pymongo import MongoClient
        client = MongoClient(Config.MongoDB.URI)
        db = client[Config.MongoDB.DB]
        col = db[Config.Standalone.REQUESTS_COLLECTION]
        return col.find_one({"document_id": document_id, "subscription_id": subscription_id, "type": "document"})
    except Exception:
        return None


def _log_request(
    api_key: Dict[str, Any],
    endpoint: str,
    mode: str,
    result: Dict[str, Any],
) -> None:
    """Log request to standalone_requests collection for audit trail."""
    try:
        from pymongo import MongoClient
        client = MongoClient(Config.MongoDB.URI)
        db = client[Config.MongoDB.DB]
        col = db[Config.Standalone.REQUESTS_COLLECTION]
        col.insert_one({
            "type": "request",
            "request_id": result.get("request_id") or result.get("batch_id", ""),
            "api_key_hash": api_key["key_hash"],
            "endpoint": endpoint,
            "mode": mode,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "latency_ms": result.get("usage", {}).get("total_ms", 0),
            "status": result.get("status", "unknown"),
            "confidence": result.get("confidence"),
        })
    except Exception as exc:
        logger.debug("[STANDALONE] Failed to log request: %s", exc)
```

- [ ] **Step 5: Mount the router in main.py**

Add after the `agents_router` include (around line 209 in `src/main.py`):

```python
try:
    from src.api.standalone_api import standalone_router
    api_router.include_router(standalone_router, tags=["Standalone API"])
except ImportError:
    pass
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/test_standalone_api.py -v`
Expected: All 4 tests PASS

- [ ] **Step 7: Commit**

```bash
git add src/api/standalone_api.py src/api/config.py src/main.py tests/test_standalone_api.py
git commit -m "feat(standalone): add FastAPI router with all endpoints, mount on main app"
```

---

### Task 9: API Key Management Script

**Files:**
- Create: `scripts/manage_api_keys.py`

- [ ] **Step 1: Create the management script**

```python
#!/usr/bin/env python3
"""CLI tool to manage DocWain Standalone API keys.

Usage:
    python scripts/manage_api_keys.py create --name "Partner X" --subscription-id sub-123
    python scripts/manage_api_keys.py list
    python scripts/manage_api_keys.py revoke --prefix dw_abc123
    python scripts/manage_api_keys.py reset-usage --prefix dw_abc123
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pymongo import MongoClient
from src.api.config import Config
from src.api.standalone_auth import generate_api_key


def get_collection():
    client = MongoClient(Config.MongoDB.URI)
    db = client[Config.MongoDB.DB]
    return db[getattr(Config.Standalone, "API_KEYS_COLLECTION", "api_keys")]


def cmd_create(args):
    raw_key, key_hash = generate_api_key()
    col = get_collection()
    col.insert_one({
        "key_hash": key_hash,
        "key_prefix": raw_key[:10] + "...",
        "name": args.name,
        "subscription_id": args.subscription_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "active": True,
        "permissions": ["process", "extract", "batch", "query"],
        "usage": {
            "total_requests": 0,
            "last_used": None,
            "requests_today": 0,
            "documents_processed": 0,
        },
    })
    print(f"API Key created successfully!")
    print(f"  Name:            {args.name}")
    print(f"  Subscription:    {args.subscription_id}")
    print(f"  Key:             {raw_key}")
    print(f"  Prefix:          {raw_key[:10]}...")
    print()
    print("  SAVE THIS KEY — it cannot be retrieved again.")


def cmd_list(args):
    col = get_collection()
    keys = col.find({}, {"_id": 0, "key_hash": 0})
    print(f"{'Name':<25} {'Prefix':<15} {'Subscription':<20} {'Active':<8} {'Requests':<10} {'Last Used'}")
    print("-" * 100)
    for k in keys:
        usage = k.get("usage", {})
        print(
            f"{k.get('name', '?'):<25} "
            f"{k.get('key_prefix', '?'):<15} "
            f"{k.get('subscription_id', '?'):<20} "
            f"{str(k.get('active', '?')):<8} "
            f"{usage.get('total_requests', 0):<10} "
            f"{usage.get('last_used', 'never')}"
        )


def cmd_revoke(args):
    col = get_collection()
    result = col.update_one(
        {"key_prefix": {"$regex": f"^{args.prefix}"}},
        {"$set": {"active": False}},
    )
    if result.modified_count:
        print(f"Revoked key with prefix {args.prefix}")
    else:
        print(f"No active key found with prefix {args.prefix}")


def cmd_reset_usage(args):
    col = get_collection()
    result = col.update_one(
        {"key_prefix": {"$regex": f"^{args.prefix}"}},
        {"$set": {
            "usage.total_requests": 0,
            "usage.requests_today": 0,
            "usage.documents_processed": 0,
            "usage.last_used": None,
            "usage.by_endpoint": {},
            "usage.by_mode": {},
        }},
    )
    if result.modified_count:
        print(f"Reset usage for key with prefix {args.prefix}")
    else:
        print(f"No key found with prefix {args.prefix}")


def main():
    parser = argparse.ArgumentParser(description="Manage DocWain Standalone API keys")
    sub = parser.add_subparsers(dest="command")

    create_p = sub.add_parser("create", help="Create a new API key")
    create_p.add_argument("--name", required=True, help="Friendly name for the key")
    create_p.add_argument("--subscription-id", required=True, help="Subscription ID to scope the key to")

    sub.add_parser("list", help="List all API keys")

    revoke_p = sub.add_parser("revoke", help="Revoke an API key")
    revoke_p.add_argument("--prefix", required=True, help="Key prefix (e.g., dw_abc123)")

    reset_p = sub.add_parser("reset-usage", help="Reset usage counters")
    reset_p.add_argument("--prefix", required=True, help="Key prefix")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    {"create": cmd_create, "list": cmd_list, "revoke": cmd_revoke, "reset-usage": cmd_reset_usage}[args.command](args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test the script runs**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python scripts/manage_api_keys.py --help`
Expected: Shows usage help text

- [ ] **Step 3: Commit**

```bash
git add scripts/manage_api_keys.py
git commit -m "feat(standalone): add API key management CLI script"
```

---

### Task 10: Integration Test

**Files:**
- Create: `tests/test_standalone_integration.py`

- [ ] **Step 1: Write integration test that validates the full router**

```python
# tests/test_standalone_integration.py
"""Integration tests for the Standalone API router using FastAPI TestClient."""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from io import BytesIO


@pytest.fixture
def api_key_doc():
    return {
        "key_hash": "testhash",
        "name": "Test Key",
        "subscription_id": "test-sub",
        "active": True,
        "permissions": ["process", "extract", "batch", "query"],
        "usage": {"total_requests": 0, "last_used": None},
    }


@pytest.fixture
def client(api_key_doc):
    """Create a test client with mocked auth."""
    from src.api.standalone_api import standalone_router
    from src.api.standalone_auth import require_api_key
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(standalone_router)

    async def mock_auth():
        return {
            "key_hash": api_key_doc["key_hash"],
            "name": api_key_doc["name"],
            "subscription_id": api_key_doc["subscription_id"],
            "permissions": api_key_doc["permissions"],
            "keys_collection": MagicMock(),
        }

    app.dependency_overrides[require_api_key] = mock_auth
    return TestClient(app)


def test_templates_endpoint(client):
    resp = client.get("/v1/docwain/templates")
    assert resp.status_code == 200
    data = resp.json()
    assert "templates" in data
    assert len(data["templates"]) >= 6


@patch("src.api.standalone_api._log_request")
@patch("src.api.standalone_processor.process_document")
def test_process_endpoint(mock_process, mock_log, client):
    mock_process.return_value = {
        "request_id": "req-test",
        "status": "completed",
        "answer": "The revenue was $5M",
        "sources": [{"page": 1}],
        "confidence": 0.92,
        "grounded": True,
        "low_confidence": False,
        "low_confidence_reasons": [],
        "structured_output": None,
        "document_id": None,
        "partial_answer": None,
        "usage": {"extraction_ms": 100, "intelligence_ms": 50, "retrieval_ms": 50, "generation_ms": 200, "total_ms": 400},
    }
    resp = client.post(
        "/v1/docwain/process",
        data={"prompt": "What is the revenue?", "mode": "qa"},
        files={"file": ("test.pdf", b"%PDF-1.5 fake", "application/pdf")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "The revenue was $5M"
    assert data["confidence"] == 0.92


def test_process_requires_auth():
    """Without auth override, should fail."""
    from src.api.standalone_api import standalone_router
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(standalone_router)
    raw_client = TestClient(app)
    resp = raw_client.post(
        "/v1/docwain/process",
        data={"prompt": "test"},
        files={"file": ("test.pdf", b"fake", "application/pdf")},
    )
    assert resp.status_code == 401


def test_extract_endpoint_requires_mode(client):
    resp = client.post(
        "/v1/docwain/extract",
        data={"prompt": "test"},
        files={"file": ("test.pdf", b"fake", "application/pdf")},
    )
    assert resp.status_code == 422
```

- [ ] **Step 2: Run integration tests**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/test_standalone_integration.py -v`
Expected: All 4 tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_standalone_integration.py
git commit -m "test(standalone): add integration tests for router endpoints"
```

---

### Task 11: Final Verification

- [ ] **Step 1: Run all standalone tests together**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -m pytest tests/test_standalone_*.py -v`
Expected: All tests PASS

- [ ] **Step 2: Verify router mounts correctly**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -c "from src.api.standalone_api import standalone_router; print('Router OK:', standalone_router.prefix); print('Routes:', [r.path for r in standalone_router.routes])"`
Expected: Shows prefix `/v1/docwain` and all route paths

- [ ] **Step 3: Verify config loads**

Run: `cd /home/ubuntu/PycharmProjects/DocWain && python -c "from src.api.config import Config; print('Standalone enabled:', Config.Standalone.ENABLED); print('Max batch:', Config.Standalone.MAX_BATCH_FILES)"`
Expected: Shows `True` and `10`

- [ ] **Step 4: Commit all changes**

```bash
git add -A
git commit -m "feat(standalone): complete standalone API endpoint with auth, multi-doc, batch, templates, webhooks"
```
