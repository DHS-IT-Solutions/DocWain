# DocWain Standalone API Endpoint — Design Spec

**Date:** 2026-04-08
**Status:** Approved

## Overview

A standalone, authenticated API endpoint that exposes all of DocWain's document intelligence capabilities as a single product feature. Consumers send documents + prompts and receive processed results — Q&A answers, structured table extraction, entity extraction, and summaries — with the same intelligence pipeline as the main app.

## Goals

- Expose DocWain's full intelligence as an authenticated API
- Support one-shot processing (document + prompt in one call) and persistent documents for repeated queries
- Structured extraction modes: tables, entities, summaries with output format control
- Multi-document cross-referencing and batch processing
- Webhook callbacks for async processing
- Pre-built prompt templates for common use cases
- Confidence-gated responses for enterprise reliability
- Full finetune signal capture (identical to main app)
- Usage tracking and audit trail per API key

## Non-Goals

- User management UI (keys managed via CLI script)
- Billing integration (tracked, not enforced)
- Knowledge graph write operations (read-only KG augmentation during retrieval)
- Conversation session management (stateless per request; persistent docs provide continuity)

## Authentication

**Mechanism:** API key via `X-API-Key` header.

**Storage:** MongoDB collection `api_keys`:
```json
{
  "key_hash": "<sha256 hex>",
  "key_prefix": "dw_...",
  "name": "Partner X",
  "subscription_id": "sub-123",
  "created_at": "2026-04-08T00:00:00Z",
  "active": true,
  "permissions": ["process", "extract", "batch", "query"],
  "usage": {
    "total_requests": 0,
    "last_used": null,
    "requests_today": 0,
    "documents_processed": 0
  }
}
```

Keys are SHA-256 hashed at rest. The raw key is shown once at creation time. Prefix `dw_` followed by 48 hex chars for easy identification.

**Validation flow:**
1. Extract `X-API-Key` header
2. Hash it, look up in `api_keys` collection
3. Verify `active == true`
4. Inject `subscription_id` from key record into request context
5. Increment usage counters (async, non-blocking)

## Endpoints

### POST `/api/v1/docwain/process`

One-shot: send a single document + prompt, get answer back.

**Request:** `multipart/form-data`
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| file | UploadFile | yes | PDF, DOCX, PPTX, image |
| prompt | string | yes | What to do with the document |
| mode | string | no | `qa` (default), `table`, `entities`, `summary` |
| output_format | string | no | `json` (default), `markdown`, `csv`, `html` |
| persist | boolean | no | Save document for reuse (default: false) |
| stream | boolean | no | SSE streaming response (default: false) |
| template | string | no | Pre-built prompt template name |
| confidence_threshold | float | no | Min confidence; below returns low_confidence flag (default: 0.0) |
| callback_url | string | no | Webhook URL for async result delivery |

**Response:**
```json
{
  "request_id": "req-uuid",
  "status": "completed",
  "answer": "...",
  "sources": [{"page": 1, "section": "...", "confidence": 0.95}],
  "confidence": 0.92,
  "grounded": true,
  "low_confidence": false,
  "low_confidence_reasons": [],
  "structured_output": null,
  "document_id": null,
  "output_format": "json",
  "usage": {
    "extraction_ms": 1200,
    "intelligence_ms": 400,
    "retrieval_ms": 300,
    "generation_ms": 800,
    "total_ms": 2700
  }
}
```

When `mode` is `table`, `entities`, or `summary`, `structured_output` is populated:
- **table**: `{"tables": [{"headers": [...], "rows": [[...]], "page": 1, "caption": "..."}]}`
- **entities**: `{"entities": [{"text": "...", "type": "PERSON", "page": 1, "confidence": 0.9}]}`
- **summary**: `{"sections": [{"title": "...", "summary": "...", "key_points": [...]}]}`

When `callback_url` is provided:
- Returns immediately: `{"request_id": "req-uuid", "status": "processing", "poll_url": "/api/v1/docwain/requests/{req_id}/status"}`
- POSTs full response to `callback_url` when done
- Includes `X-DocWain-Signature` header (HMAC-SHA256 of body using API key hash as secret) for webhook verification

### POST `/api/v1/docwain/process/multi`

Multi-document processing: cross-document Q&A, comparison, merged extraction.

**Request:** `multipart/form-data`
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| files | List[UploadFile] | yes* | Multiple documents |
| document_ids | string (JSON list) | yes* | Previously persisted doc IDs |
| prompt | string | yes | Cross-document query |
| mode | string | no | Same as `/process` |
| output_format | string | no | Same as `/process` |
| callback_url | string | no | Webhook URL |

*At least one of `files` or `document_ids` required. Can mix both.

**Processing:** Each document is extracted and embedded independently, then retrieval runs across all document collections simultaneously. The LLM receives merged context with per-document source attribution.

**Response:** Same shape as `/process`, with sources tagged by document:
```json
{
  "sources": [
    {"document": "contract_a.pdf", "document_id": "doc-1", "page": 3, "section": "..."},
    {"document": "contract_b.pdf", "document_id": "doc-2", "page": 7, "section": "..."}
  ]
}
```

### POST `/api/v1/docwain/batch`

Bulk processing: same prompt applied to many files independently.

**Request:** `multipart/form-data`
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| files | List[UploadFile] | yes | Multiple documents |
| prompt | string | yes | Applied to each document |
| mode | string | no | Same as `/process` |
| output_format | string | no | Same as `/process` |
| callback_url | string | no | Webhook for completion |

**Response:**
```json
{
  "batch_id": "batch-uuid",
  "status": "completed",
  "results": [
    {"filename": "doc1.pdf", "status": "completed", "answer": "...", "confidence": 0.9, ...},
    {"filename": "doc2.pdf", "status": "completed", "answer": "...", "confidence": 0.85, ...},
    {"filename": "doc3.pdf", "status": "error", "error": "Unsupported format"}
  ],
  "summary": {"total": 3, "completed": 2, "failed": 1},
  "usage": {"total_ms": 8500}
}
```

Batch always uses `callback_url` if provided. Without it, processes synchronously (capped at 10 files; larger batches require callback).

### POST `/api/v1/docwain/extract`

Structured extraction: focused on pulling structured data from documents.

**Request:** `multipart/form-data`
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| file | UploadFile | yes | Document to extract from |
| mode | string | yes | `table`, `entities`, `summary` |
| prompt | string | no | Optional extraction guidance |
| output_format | string | no | `json` (default), `csv`, `markdown`, `html` |
| template | string | no | Template name (e.g., `invoice`, `contract_clauses`) |

**Response:**
```json
{
  "request_id": "req-uuid",
  "mode": "table",
  "result": {...},
  "metadata": {
    "pages": 12,
    "document_type": "financial_report",
    "extraction_ms": 1500,
    "intelligence_ms": 400
  }
}
```

### POST `/api/v1/docwain/documents`

Upload and persist a document for repeated queries.

**Request:** `multipart/form-data`
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| file | UploadFile | yes | Document to persist |
| name | string | no | Friendly name |

**Response:**
```json
{
  "document_id": "doc-uuid",
  "name": "Q1 Report",
  "status": "processing",
  "created_at": "2026-04-08T12:00:00Z"
}
```

### GET `/api/v1/docwain/documents/{doc_id}/status`

Poll document processing status.

**Response:**
```json
{
  "document_id": "doc-uuid",
  "status": "ready",
  "name": "Q1 Report",
  "pages": 24,
  "document_type": "financial_report",
  "created_at": "2026-04-08T12:00:00Z",
  "ready_at": "2026-04-08T12:00:45Z"
}
```

Status values: `processing`, `ready`, `error`.

### POST `/api/v1/docwain/query`

Query a previously persisted document.

**Request:** `application/json`
```json
{
  "document_id": "doc-uuid",
  "prompt": "What were Q1 revenues?",
  "mode": "qa",
  "output_format": "json",
  "stream": false,
  "confidence_threshold": 0.7
}
```

Also supports `document_ids: [...]` for multi-doc queries against persisted documents.

**Response:** Same shape as `/process`.

### GET `/api/v1/docwain/usage`

Audit trail and usage statistics.

**Response:**
```json
{
  "api_key_name": "Partner X",
  "period": "2026-04-01 to 2026-04-08",
  "totals": {
    "requests": 1247,
    "documents_processed": 89,
    "queries": 1158
  },
  "by_endpoint": {
    "/process": 450,
    "/query": 600,
    "/extract": 150,
    "/batch": 47
  },
  "by_mode": {
    "qa": 800,
    "table": 200,
    "entities": 150,
    "summary": 97
  },
  "recent": [
    {"request_id": "req-xyz", "endpoint": "/process", "mode": "qa", "timestamp": "...", "latency_ms": 2700}
  ]
}
```

### GET `/api/v1/docwain/templates`

List available prompt templates.

**Response:**
```json
{
  "templates": [
    {"name": "invoice", "description": "Extract invoice fields: vendor, amounts, dates, line items", "modes": ["table", "entities"]},
    {"name": "contract_clauses", "description": "Identify and extract contract clauses with risk assessment", "modes": ["entities", "summary"]},
    {"name": "compliance_checklist", "description": "Check document against compliance requirements", "modes": ["qa", "entities"]},
    {"name": "medical_record", "description": "Extract patient info, diagnoses, medications, procedures", "modes": ["entities", "table"]},
    {"name": "financial_report", "description": "Extract financial tables, KPIs, and executive summary", "modes": ["table", "summary"]},
    {"name": "resume", "description": "Extract skills, experience, education, contact information", "modes": ["entities", "table"]}
  ]
}
```

## Processing Pipeline (Intelligence Parity)

The standalone endpoint uses the exact same pipeline as the main app. No shortcuts, no simplified paths.

### One-shot `/process` flow:
```
1. Auth (API key → subscription_id)
2. Extract (DocumentExtractor — layout-aware, OCR, table detection)
3. Intelligence (V2 pipeline):
   - Document classification (DocumentClassifier)
   - Entity extraction (EntityExtractor)
   - Section summarization (Summarizer)
   - Key facts identification
4. Structured extraction (StructuredExtractionEngine — if mode != qa)
5. Chunk (SectionChunker — same chunk_size, overlap as main app)
6. Embed (BGE-large-en-v1.5 — same model as main app)
7. Index in temporary Qdrant collection (prefix: `dw_standalone_{request_id}`)
8. Retrieve (UnifiedRetriever — hybrid dense+sparse, same TOPK settings)
9. Rerank (cross-encoder — same model as main app)
10. KG augmentation (if available — read-only)
11. Reason (CoreAgent or FastPath — same routing as main app)
12. Generate (vLLM/Ollama via LLM gateway — same model)
13. Citation verification (CitationVerifier)
14. Confidence scoring (ConfidenceScorer)
15. Confidence gate check (if confidence < threshold → low_confidence response)
16. Finetune capture (LearningSignalStore — high_quality.jsonl + finetune_buffer.jsonl)
17. Cleanup temporary Qdrant collection (unless persist=true)
```

### Finetune capture:
Every request records to LearningSignalStore:
- `record_high_quality()` for confident responses (same threshold as main app)
- `record_low_confidence()` for below-threshold responses
- `record_failure()` for processing errors
- Metadata includes: `source: "standalone_api"`, `api_key_name`, `mode`, `template` (if used)

### Persistent document flow:
Steps 1-6 run once at upload time. Vectors stored in permanent Qdrant collection `dw_standalone_{subscription_id}`. Subsequent `/query` calls skip to step 8.

## Output Format Conversion

The `standalone_output.py` module converts structured results to requested formats:

- **json**: Native response (default)
- **markdown**: Tables as markdown tables, entities as bullet lists, summaries as headers + paragraphs
- **csv**: Tables as CSV strings (one per table), entities as CSV rows
- **html**: Tables as `<table>` elements, entities as definition lists, summaries as semantic HTML

## Prompt Templates

Stored in `standalone_templates.py` as a registry. Each template defines:
```python
@dataclass
class PromptTemplate:
    name: str
    description: str
    modes: List[str]
    system_prompt: str      # Injected as system context
    extraction_hints: str   # Guides structured extraction
    output_schema: dict     # Expected output structure for validation
```

Initial templates: `invoice`, `contract_clauses`, `compliance_checklist`, `medical_record`, `financial_report`, `resume`.

Templates are additive — they enhance the user's prompt, not replace it. The user's `prompt` field always takes precedence.

## Webhook Delivery

When `callback_url` is provided:
1. Request is accepted immediately with `status: "processing"` and `request_id`
2. Processing runs in a background thread (ThreadPoolExecutor, max 4 concurrent)
3. On completion, POST full response JSON to `callback_url`
4. Include headers:
   - `X-DocWain-Request-Id: {request_id}`
   - `X-DocWain-Signature: {hmac_sha256(body, api_key_hash)}`
   - `Content-Type: application/json`
5. Retry up to 3 times with exponential backoff (1s, 5s, 25s)
6. Result also stored in MongoDB `standalone_requests` collection for polling via status endpoint

## Confidence Gating

When `confidence_threshold` is set and the response confidence falls below it:
```json
{
  "request_id": "req-uuid",
  "status": "completed",
  "answer": null,
  "low_confidence": true,
  "low_confidence_reasons": [
    "Document OCR quality is poor (estimated accuracy: 72%)",
    "Query topic not found in document content",
    "Conflicting information found across sections"
  ],
  "confidence": 0.45,
  "grounded": false,
  "partial_answer": "Based on limited evidence: ...",
  "usage": {...}
}
```

Reasons are generated by the confidence scorer analyzing: OCR quality metrics, retrieval scores, citation verification results, and entity coverage.

## File Structure

### New files:
| File | Purpose |
|------|---------|
| `src/api/standalone_api.py` | FastAPI router, all endpoint definitions, request validation |
| `src/api/standalone_auth.py` | API key hashing, validation dependency, usage tracking |
| `src/api/standalone_processor.py` | Core orchestration: extract → intelligence → embed → retrieve → generate |
| `src/api/standalone_multi.py` | Multi-document and batch processing logic |
| `src/api/standalone_templates.py` | Template registry with pre-built prompt templates |
| `src/api/standalone_schemas.py` | All Pydantic request/response models |
| `src/api/standalone_output.py` | Output format conversion (JSON/markdown/CSV/HTML) |
| `src/api/standalone_webhook.py` | Webhook callback delivery with HMAC signing and retry |
| `scripts/manage_api_keys.py` | CLI script to create/revoke/list API keys |

### Existing files modified (minimal):
| File | Change |
|------|--------|
| `src/main.py` | Add `from src.api.standalone_api import standalone_router` and `api_router.include_router(standalone_router, tags=["Standalone API"])` |
| `src/api/config.py` | Add `class Standalone` config section |

### Config additions (`Config.Standalone`):
```python
class Standalone:
    ENABLED = os.getenv("DOCWAIN_STANDALONE_ENABLED", "true")
    TEMP_COLLECTION_TTL = int(os.getenv("STANDALONE_TEMP_TTL", "3600"))  # 1 hour
    MAX_BATCH_FILES = int(os.getenv("STANDALONE_MAX_BATCH", "10"))
    MAX_FILE_SIZE_MB = int(os.getenv("STANDALONE_MAX_FILE_MB", "50"))
    WEBHOOK_MAX_WORKERS = int(os.getenv("STANDALONE_WEBHOOK_WORKERS", "4"))
    WEBHOOK_MAX_RETRIES = int(os.getenv("STANDALONE_WEBHOOK_RETRIES", "3"))
    API_KEYS_COLLECTION = os.getenv("STANDALONE_KEYS_COLLECTION", "api_keys")
    REQUESTS_COLLECTION = os.getenv("STANDALONE_REQUESTS_COLLECTION", "standalone_requests")
```

## Error Handling

All errors return consistent JSON:
```json
{
  "error": {
    "code": "EXTRACTION_FAILED",
    "message": "Failed to extract document: unsupported format .xyz",
    "request_id": "req-uuid"
  }
}
```

Error codes: `AUTH_INVALID`, `AUTH_DISABLED`, `EXTRACTION_FAILED`, `PROCESSING_FAILED`, `DOCUMENT_NOT_FOUND`, `DOCUMENT_NOT_READY`, `FILE_TOO_LARGE`, `BATCH_TOO_LARGE`, `UNSUPPORTED_FORMAT`, `TEMPLATE_NOT_FOUND`, `WEBHOOK_INVALID`, `INTERNAL_ERROR`.

HTTP status mapping: 401 for auth errors, 404 for not found, 413 for size limits, 422 for validation, 500 for internal errors.

## Usage Tracking

Every request updates the API key's usage counters in MongoDB (async, fire-and-forget):
- `total_requests` incremented
- `last_used` timestamp updated
- `requests_today` incremented (reset daily)
- `documents_processed` incremented per unique document

Detailed request log stored in `standalone_requests` collection:
```json
{
  "request_id": "req-uuid",
  "api_key_hash": "...",
  "endpoint": "/process",
  "mode": "qa",
  "template": null,
  "timestamp": "2026-04-08T12:00:00Z",
  "latency_ms": 2700,
  "status": "completed",
  "confidence": 0.92,
  "file_count": 1,
  "file_sizes": [1048576]
}
```
