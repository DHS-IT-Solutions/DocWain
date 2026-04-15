# DocWain Standalone Service Redesign

**Date:** 2026-04-15
**Status:** Approved

## Overview

Redesign the DocWain standalone API from a tightly-coupled router inside the main app into a fully independent service. The new standalone service runs as its own process on port 8400, talks directly to vLLM for all document processing, and has zero code imports from the main app.

## Goals

1. **Process isolation** — Standalone runs as a separate systemd service; crashes, load, or restarts don't affect the main app.
2. **Code isolation** — Zero imports from `src/`. Standalone is a self-contained FastAPI app in `standalone/` at the project root.
3. **Simplification** — Reduce from 13 endpoints to 3 endpoint groups: extraction, intelligence, and admin key management.
4. **vLLM-native** — All LLM work (OCR, extraction, intelligence) goes through the vLLM-served DocWain model over HTTP. The standalone service is a thin API layer.

## Non-Goals

- Shared Python library between main app and standalone (over-engineering for this scope).
- Self-service API key generation (enterprise use requires admin-provisioned keys).
- Vector storage or RAG in the standalone service (vLLM handles everything in a single pass).

---

## Architecture

### Directory Structure

```
standalone/
├── __init__.py
├── __main__.py          # uvicorn entry point
├── app.py               # FastAPI app setup
├── config.py            # Standalone-specific config (env vars)
├── auth.py              # API key validation + admin key management
├── endpoints/
│   ├── extract.py       # POST /api/v1/standalone/extract
│   ├── intelligence.py  # POST /api/v1/standalone/intelligence
│   └── keys.py          # Admin key CRUD
├── vllm_client.py       # Async HTTP client to vLLM
├── file_reader.py       # Minimal file-to-content conversion
├── schemas.py           # Pydantic request/response models
└── output_formatter.py  # Convert LLM responses to JSON/CSV/sections/flatfile
```

### Service Characteristics

- **Port:** 8400
- **Process:** Independent uvicorn process via systemd
- **Dependencies:** MongoDB (own database), vLLM (HTTP), no main app dependency
- **Zero imports from `src/`**

---

## Endpoints

All endpoints under `/api/v1/standalone`.

### 1. Extract — `POST /api/v1/standalone/extract`

Extracts document content into structured formats, enhanced by DocWain's LLM intelligence (entity recognition, classification, summarization alongside structural output).

**Request:**
- `file` (UploadFile) — the document
- `output_format` (enum: `json`, `csv`, `sections`, `flatfile`, `tables`) — desired output structure
- `prompt` (optional string) — additional extraction guidance (e.g., "focus on financial line items")

**Flow:**
1. `file_reader.py` converts the file to a vLLM-consumable format (PDF pages as base64 images, DOCX/Excel as text)
2. `vllm_client.extract()` sends content with extraction-focused prompt including output_format instruction and any user prompt
3. `output_formatter.py` parses the model response into the requested structured format
4. Returns structured result + metadata

**Response:**
```json
{
  "request_id": "uuid",
  "document_type": "invoice",
  "output_format": "json",
  "content": { ... },
  "metadata": {
    "pages": 3,
    "processing_time_ms": 1200
  }
}
```

### 2. Intelligence — `POST /api/v1/standalone/intelligence`

Higher-level analytical understanding: summaries, key facts, risk assessments, comparisons, recommendations.

**Request:**
- `file` (UploadFile) — the document
- `prompt` (optional string) — specific analysis request (e.g., "what are the compliance risks in this contract?")
- `analysis_type` (optional enum: `summary`, `key_facts`, `risk_assessment`, `recommendations`, `auto`) — defaults to `auto` (model decides)

**Flow:**
1. File converted via `file_reader.py`
2. `vllm_client.analyze()` sends content with intelligence-focused prompt including analysis_type and user prompt
3. Model returns analytical insights grounded in document content
4. Response includes analysis plus supporting evidence/citations

**Response:**
```json
{
  "request_id": "uuid",
  "document_type": "contract",
  "analysis_type": "risk_assessment",
  "insights": {
    "summary": "...",
    "findings": [...],
    "evidence": [...]
  },
  "metadata": {
    "pages": 12,
    "processing_time_ms": 3400
  }
}
```

### 3. Keys — Admin-Only Key Management

Protected by `X-Admin-Secret` header checked against `ADMIN_SECRET` env var.

- `POST /admin/keys` — Create a new API key. Returns raw key once (`dw_sa_` + 48 hex). Stores SHA-256 hash in MongoDB.
- `GET /admin/keys` — List active keys (prefix + name only, never the raw key).
- `DELETE /admin/keys/{key_id}` — Revoke a key.

---

## vLLM Client

Async HTTP client (`httpx`) that talks to vLLM's OpenAI-compatible API.

**Configuration:**
- `VLLM_BASE_URL` — e.g., `http://localhost:8001/v1`
- `VLLM_MODEL_NAME` — defaults to `DocWain`
- `VLLM_TIMEOUT` — request timeout in seconds (default 120)

**Two call patterns:**
- `extract(content, output_format, prompt)` — extraction system prompt + user content
- `analyze(content, analysis_type, prompt)` — intelligence system prompt + user content

Sends multimodal messages when content includes images (PDF pages), text-only messages for text documents. System prompts are minimal and instruct the model on output structure only — domain knowledge lives in the model weights.

---

## File Reader

Minimal file-to-content conversion. No intelligence, no chunking, no embedding.

| Input Type | Conversion | Output to vLLM |
|---|---|---|
| PDF | Render pages as images (PyMuPDF) | Base64 images in multimodal message |
| DOCX | python-docx text extraction | Plain text |
| Excel/CSV | openpyxl/csv to text table | Plain text |
| Images (PNG/JPG) | Pass through | Base64 image |
| Plain text/JSON | Pass through | Plain text |

The DocWain model handles OCR, layout understanding, and table detection through its vision capabilities.

---

## Authentication

**Two layers:**
- **Admin auth** — `X-Admin-Secret` header for key management endpoints only.
- **API key auth** — `X-Api-Key` header for extract and intelligence endpoints. SHA-256 hashed, looked up in `docwain_standalone.api_keys` collection.

**Key format:** `dw_sa_` + 48 hex characters (prefix distinguishes from main app keys).

**Usage tracking:** Each request logs to MongoDB: endpoint, timestamp, processing time, document type. Simple counters on the key document track total requests.

---

## Configuration

All env-var driven, no shared config with the main app:

```
STANDALONE_PORT=8400
VLLM_BASE_URL=http://localhost:8001/v1
VLLM_MODEL_NAME=DocWain
VLLM_TIMEOUT=120
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB=docwain_standalone
ADMIN_SECRET=<required>
MAX_FILE_SIZE_MB=50
LOG_LEVEL=INFO
```

---

## Infrastructure

### Systemd Service

New unit `docwain-standalone.service`:
- Runs on port 8400
- Independent of main app, vLLM management, and Teams services
- Depends only on MongoDB being reachable and at least one vLLM instance running
- Auto-restart on failure

### MongoDB

- **Database:** `docwain_standalone` (separate from main app's database)
- **Collections:**
  - `api_keys` — key hashes, names, permissions, usage counters
  - `request_logs` — audit trail per request

---

## Cleanup: What Gets Removed from the Main App

Once the standalone service is live:
- Delete `src/api/standalone_api.py`
- Delete `src/api/standalone_processor.py`
- Delete `src/api/standalone_multi.py`
- Delete `src/api/standalone_auth.py`
- Delete `src/api/standalone_templates.py`
- Delete `src/api/standalone_output.py`
- Delete `src/api/standalone_webhook.py`
- Delete `src/api/standalone_schemas.py`
- Remove `standalone_router` registration from `src/main.py`
- Remove `Config.Standalone` section from `src/api/config.py`
- Clean up dead imports

---

## Summary of All Three DocWain Sections

| Section | Role | Isolation | Port |
|---|---|---|---|
| **Main App** | Web-based document intelligence platform with full pipeline (extraction, embedding, RAG, KG, screening) | Primary service | 8000 |
| **Teams App** | Teams plugin — captures documents from Teams, provides insights. Own data namespace, shares compute | Separate systemd service, data-isolated | 8300 |
| **Standalone** | Thin API layer for extraction + document intelligence. Calls vLLM directly, zero main app coupling | Separate systemd service, fully independent | 8400 |
