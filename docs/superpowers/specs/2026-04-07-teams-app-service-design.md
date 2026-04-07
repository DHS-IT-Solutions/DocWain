# DocWain Teams App — Standalone Service Design

**Date:** 2026-04-07
**Status:** Approved
**Approach:** Standalone service with shared library imports (Approach B)

## Overview

A separate systemd service (`docwain-teams`) that handles Microsoft Teams bot interactions — document ingestion with auto-triggered pipeline, progress tracking via Adaptive Cards, and query proxying to the main app. Fully isolated from the main app at the process, data, and code level.

## Architecture

### Directory Layout

```
DocWain/
├── src/                          ← Main app (read-only dependency, never modified)
├── teams_app/                    ← Teams service (all new code here)
│   ├── main.py                   ← FastAPI entry point, own lifespan
│   ├── config.py                 ← Teams-specific config (extends src.api.config)
│   ├── bot/
│   │   ├── handler.py            ← Bot Framework message handler
│   │   ├── attachments.py        ← File download (Teams uploads + OneDrive/SharePoint)
│   │   └── cards.py              ← Adaptive Card templates (progress, response, onboarding)
│   ├── pipeline/
│   │   ├── orchestrator.py       ← Auto-trigger pipeline (extraction→screening→embedding)
│   │   ├── fast_path.py          ← Tiered routing: express vs full
│   │   └── workers.py            ← Concurrent document processing (asyncio + semaphore)
│   ├── storage/
│   │   ├── namespace.py          ← Collection/prefix namespacing logic
│   │   └── tenant.py             ← Auto-provisioning from AAD identity
│   ├── proxy/
│   │   └── query_proxy.py        ← HTTP proxy to main app /api/ask (SSE streaming)
│   ├── signals/
│   │   └── capture.py            ← Learning signal capture for finetuning
│   └── models.py                 ← Teams-specific data models
├── deploy/
│   ├── docwain-app.service       ← Existing
│   └── docwain-teams.service     ← New
```

### Service Characteristics

| Property | Value |
|----------|-------|
| Port | 8300 (configurable via `TEAMS_APP_PORT`) |
| Systemd unit | `docwain-teams.service` |
| Log identifier | `docwain-teams` |
| Startup time | ~10-15s |
| Depends on | `docwain-app.service` (soft dependency — starts without it) |
| Shared virtualenv | Yes (same `.venv` as main app) |

### Lifespan — What Gets Loaded

**Loaded:**
1. Embedding model (SentenceTransformer)
2. Qdrant client
3. MongoDB client
4. Redis client
5. Bot Framework adapter

**Not loaded (handled by main app):**
- EnterpriseRAGSystem
- LLMGateway
- NLU engine, intent classifier, domain router
- Cross-encoder reranker
- spaCy models
- Vision OCR

## Document Ingestion

### Input Sources

1. **Teams chat attachments** — files sent directly in a Teams chat/channel with the bot. Downloaded from Bot Framework CDN.
2. **OneDrive/SharePoint links** — shared as URLs in chat. Downloaded once via Microsoft Graph API. Users can type "refresh <filename>" to re-fetch and re-process.

### Auto-Triggered Pipeline

Unlike the main app (where screening and embedding are UI-triggered), the Teams app auto-triggers the full pipeline on upload:

```
Upload → Download → Triage → Extraction → Screening → Embedding → [KG]
```

All stages run automatically. No user intervention needed.

### Tiered Fast Path

| Category | File Types | Pipeline | What's Skipped | Target Time |
|----------|-----------|----------|---------------|-------------|
| Express | `.txt`, `.md`, `.csv`, `.xlsx`, `.json`, `.xml`, `.html` | Native parse → screen → embed | OCR, layout analysis, section summaries, KG | < 10s |
| Full | `.pdf`, `.docx`, `.pptx`, images | Complete extraction → screen → embed → KG | Nothing | 30-120s |
| Full (auto-escalation) | Any file where express extraction yields < 50 chars | Re-processes from scratch via full pipeline | — | 30-120s |

**Express path optimizations:**
- Uses `src.extraction.native_parsers` directly — no `DocumentExtractor` overhead
- Smaller chunks (1024 tokens vs 2048)
- Skips cross-encoder reranking during embedding quality evaluation
- Skips section summary vectors (`DWX_SECTION_SUMMARY_VECTORS`)

**Concurrency:**
- Multiple documents process concurrently via `asyncio.Semaphore`-bounded worker pool
- Default concurrency: 3 documents simultaneously (configurable)
- Pipeline stages run sequentially per document

### Shared Module Imports (Read-Only)

The Teams service imports these from `src/` without modification:

- `src.extraction.native_parsers` — express file parsing
- `src.api.dw_document_extractor.DocumentExtractor` — full extraction
- `src.screening.engine` — security/compliance screening
- `src.api.embedding_service` — embedding functions
- `src.api.pii_masking` — PII detection
- `src.kg.neo4j_store` — KG storage (full path only)

## Progress Tracking

### Single Adaptive Card, Updated In-Place

One card per document, updated via `activity.update` at each stage transition. No message spam — the user sees one card that evolves.

**During processing:**
```
┌──────────────────────────────────────────────┐
│  quarterly_report.pdf                        │
│  Express pipeline · 2.3 MB                   │
│                                              │
│  Done  Downloaded                            │
│  Done  Extracted — 12 sections, 3 tables     │
│  Done  Screening passed                      │
│  ...   Embedding... (45 chunks)              │
│  -     Knowledge Graph                       │
│                                              │
│  ░░░░░░░░░░░░░░░░▓▓▓▓  75%                  │
└──────────────────────────────────────────────┘
```

**On completion:**
```
┌──────────────────────────────────────────────┐
│  Done  quarterly_report.pdf — Ready          │
│  12 sections · 3 tables · 45 chunks          │
│  Pipeline: express · Completed in 8s         │
│                                              │
│  Ask me anything about this document.        │
│                                              │
│  [Refresh]                                   │
└──────────────────────────────────────────────┘
```

**Rules:**
- Updates happen only on stage transitions (5-6 updates total per document)
- Multiple files uploaded at once get one summary card with per-file status
- Failed/blocked documents show inline with reason and a Retry button

## Namespacing & Data Isolation

### Prefix-Based Isolation on Shared Backends

| Backend | Main App Pattern | Teams App Pattern |
|---------|-----------------|-------------------|
| MongoDB | `documents` collection | `teams_documents` collection |
| Qdrant | `sub_{subscription_id}` | `teams_{tenant_id}` |
| Neo4j | `subscription_id` property | `teams_tenant_id` property |
| Redis | `session:{user_id}` | `teams:session:{aad_user_id}` |
| Azure Blob | `documents/{subscription_id}/` | `teams/{tenant_id}/` |

Cross-contamination is impossible — namespace prefixes enforce separation.

### Tenant Auto-Provisioning

No registration step. On first message from a new AAD tenant:

1. Extract `tenant_id` + `user_id` from Teams activity context
2. Check `teams_tenants` collection for `tenant_id`
   - Exists: load config, proceed
   - Missing: auto-provision:
     - Create `teams_tenants` record
     - Create Qdrant collection `teams_{tenant_id}`
     - Reply with welcome message
3. Check `teams_users` collection for `user_id`
   - Exists: load record
   - Missing: create user record

**`teams_tenants` document:**
```json
{
    "tenant_id": "aad_tenant_abc",
    "display_name": "Contoso Ltd",
    "qdrant_collection": "teams_aad_tenant_abc",
    "settings": {
        "kg_enabled": true,
        "max_documents": 1000,
        "express_pipeline": true
    },
    "created_at": "...",
    "document_count": 0
}
```

**`teams_documents` document:**
```json
{
    "document_id": "...",
    "tenant_id": "aad_tenant_abc",
    "user_id": "aad_user_123",
    "source": "attachment | onedrive",
    "source_url": "...",
    "filename": "report.pdf",
    "pipeline": "express | full",
    "status": "downloading | extracting | screening | embedding | kg_building | ready | failed | blocked",
    "progress": {
        "extraction": {"status": "completed", "sections": 12, "tables": 3},
        "screening": {"status": "completed", "pii_masked": true},
        "embedding": {"status": "in_progress", "chunks": 45},
        "kg": {"status": "pending"}
    },
    "teams_message_id": "...",
    "teams_conversation_id": "...",
    "created_at": "...",
    "completed_at": "..."
}
```

## Query Proxy

### HTTP Proxy to Main App

The Teams service does not run its own RAG pipeline. All queries are proxied to the main app:

```
POST http://localhost:8000/api/ask
Headers:
  x-source: teams
  x-teams-tenant: aad_tenant_abc
Body:
  query: "what are the Q3 revenue figures?"
  user_id: "teams_aad_user_123"
  subscription_id: "teams_aad_tenant_abc"
  stream: true
  profile_id: null
```

### Streaming to Teams (Chunked Updates)

Teams doesn't support true SSE streaming. The proxy uses chunked message updates:

1. Collect SSE tokens from main app
2. After first ~100 tokens (or first sentence boundary), send initial Teams reply
3. Update that reply every ~200 additional tokens
4. Final update with complete response + sources card

First reply appears within ~1s of LLM generation starting.

### No-Documents Fallback

If `context_found: false` from main app:
- **Tenant has documents:** Forward response as-is (general LLM answer)
- **Tenant has no documents:** Append onboarding nudge — "I don't have any documents to search yet. Send me a file to get started!"

### Error Handling

If the main app is unreachable, reply: "I'm having trouble right now. Please try again in a moment." Document ingestion continues to work independently.

## Learning Signal Capture

### Finetuning Data Collection

Every query/response pair is captured for the finetuning pipeline:

```json
{
    "query": "what are Q3 revenue figures?",
    "response": "...",
    "sources": ["..."],
    "grounded": true,
    "context_found": true,
    "source": "teams",
    "tenant_id": "aad_tenant_abc",
    "pipeline": "express",
    "latency_ms": 1230,
    "timestamp": "..."
}
```

**User feedback via response card buttons:**
- Thumbs up → `src/outputs/learning_signals/high_quality.jsonl` with `"signal": "positive"`
- Thumbs down → `src/outputs/learning_signals/finetune_buffer.jsonl` with `"signal": "negative"`
- No feedback → `src/outputs/learning_signals/finetune_buffer.jsonl` with `"signal": "implicit"`

**Rules:**
- Same JSONL format as main app signals — finetuning pipeline is source-agnostic
- `"source": "teams"` field enables filtering/weighting Teams interactions
- No customer document content in signals — only query/response pairs
- Thumbs up/down on every response card — low friction

## Isolation Guarantees

### Code Isolation
- All Teams-specific code in `teams_app/` — never in `src/`
- Imports from `src/` are read-only consumers of specific functions
- If Teams needs different behavior, it wraps/adapts locally — never patches `src/`
- Existing `src/teams/` module stays untouched

### Data Isolation
- Separate MongoDB collections (`teams_*` prefix)
- Separate Qdrant collections (`teams_*` prefix)
- Separate Neo4j node properties (`teams_tenant_id`)
- Separate Redis key prefix (`teams:`)
- Separate Blob storage prefix (`teams/`)

### Process Isolation
- Separate systemd service, PID, port
- Independent restart: `sudo systemctl restart docwain-teams`
- Own log stream: `journalctl -u docwain-teams`
- Crash does not affect main app

### Resource Isolation
- Embedding model loaded independently (~1.5GB, CPU-only)
- No shared GPU usage
- At scale, can offload embedding to main app via HTTP

### Shared Module Change Impact
- Breaking changes in imported `src/` function signatures would require Teams service updates
- Mitigated by importing specific functions, not entire modules

## Systemd Service

```ini
[Unit]
Description=DocWain Teams Bot Service
After=network.target docwain-app.service
Wants=docwain-app.service

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/PycharmProjects/DocWain
ExecStart=/home/ubuntu/PycharmProjects/DocWain/.venv/bin/python -m teams_app.main
Restart=on-failure
RestartSec=5
TimeoutStartSec=60
TimeoutStopSec=30
StandardOutput=journal
SyslogIdentifier=docwain-teams

[Install]
WantedBy=multi-user.target
```

## Health Check

`GET :8300/health` returns:
- Qdrant connectivity
- MongoDB connectivity
- Redis connectivity
- Main app reachability (`localhost:8000/api/health`)

## Scale Considerations

Designed for small team initially, grows to org-wide:
- Start with 3 concurrent document workers, increase as needed
- Qdrant collections per tenant — scales naturally
- If embedding becomes a bottleneck, offload to main app HTTP or dedicated embedding service
- If Teams message volume grows, add Redis-backed task queue (Celery) replacing asyncio workers
