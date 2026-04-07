# DocWain Teams App — Standalone Service Design

**Date:** 2026-04-07
**Status:** Approved
**Approach:** Standalone service with shared library imports (Approach B)

## Overview

A separate systemd service (`docwain-teams`) that handles Microsoft Teams bot interactions — document ingestion with auto-triggered pipeline, progress tracking via Adaptive Cards, and query proxying to the main app. Fully isolated from the main app at the process, data, and code level.

## Azure Infrastructure (Current)

| Resource | Name | Resource Group | Details |
|----------|------|---------------|---------|
| Azure Bot Service | `dhs-docwain-bot` | `rg-docwain-dev` | F0 SKU, SingleTenant |
| API Management | `dhs-docwain-api` | `rg-docwain-dev` | UK South, TLS termination |
| App ID | `384893f8-3cd6-4b9d-bdfe-038c64387a43` | — | Matches manifest.json |
| Tenant | `13a1a520-d90b-4bfe-ada0-2be3d1f3c582` | — | dhsit.co.uk |
| Subscription | `249bb11f-9b6e-4c0e-a844-500d627b80b3` | — | Microsoft Azure Sponsorship-DocWain |
| Enabled Channels | webchat, directline, msteams | — | All active |

### Current Request Flow

```
Teams Client
    │
    ▼
Azure Bot Service (dhs-docwain-bot)
  Endpoint: https://dhs-docwain-api.azure-api.net/teams/messages
    │
    ▼
API Management (dhs-docwain-api.azure-api.net)
  API: docwain_api (path: /)
  Backend: http://4.213.139.185:8000/
    │
    ▼
Main App (port 8000)
  Route: POST /teams/messages → DocWainTeamsBot
```

### New Request Flow (After Migration)

The Azure Bot Service endpoint stays the same. APIM routes `/teams/messages` to the new Teams service on port 8300 instead of the main app on port 8000.

```
Teams Client
    │
    ▼
Azure Bot Service (dhs-docwain-bot)
  Endpoint: https://dhs-docwain-api.azure-api.net/teams/messages
    │
    ▼
API Management (dhs-docwain-api.azure-api.net)
  Route: /teams/* → http://4.213.139.185:8300/
  Route: /* (everything else) → http://4.213.139.185:8000/
    │
    ├─ Teams traffic → Teams Service (port 8300)
    └─ All other traffic → Main App (port 8000)
```

### APIM Route Update (Deployment Step)

```bash
# Add/update APIM operation to route /teams/* to port 8300
az apim api operation create \
  --resource-group rg-docwain-dev \
  --service-name dhs-docwain-api \
  --api-id docwain-api \
  --url-template "/teams/*" \
  --method POST \
  --display-name "Teams Bot Messages" \
  --operation-id teams-messages

# Set backend for this operation to port 8300
az apim api operation update ...  # Policy to override backend URL
```

This is automated via `teams_app/deploy.py` which wraps the Azure CLI/SDK calls.

### Existing Teams Code (src/teams/)

The main app already has a mature Teams integration at `src/teams/` with:
- `bot_app.py` — DocWainTeamsBot (TeamsActivityHandler), BotFrameworkAdapter
- `pipeline.py` — 3-stage pipeline (Identify → Screen → Embed) with progress cards
- `logic.py` — TeamsChatService for RAG queries
- `state.py` — Redis-backed TeamsStateStore
- `attachments.py` — File download and Document Intelligence
- `tools.py` — TeamsToolRouter for Adaptive Card actions
- `teams_storage.py` — MongoDB document storage
- `insights.py` — Proactive insights
- `cards/` — 16 Adaptive Card JSON templates

The new `teams_app/` service extracts and extends this code. The existing `src/teams/` module stays untouched and the `/teams/messages` route in `src/main.py` can be deprecated once the standalone service is verified working.

### Dependencies Installed

```
botbuilder-core==4.17.0
botbuilder-schema==4.17.0
botframework-connector==4.17.0
botframework-streaming==4.17.0
azure-mgmt-botservice==2.0.0
azure-mgmt-resource==25.0.0
azure-identity==1.25.3
azure-storage-blob==12.28.0
msgraph-sdk==1.55.0          ← NEW (OneDrive/SharePoint file download)
```

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

## Deployment & APIM Routing

### Deployment Script (`teams_app/deploy.py`)

Automates the APIM route update to split Teams traffic from main app traffic:

1. Create a new APIM backend pointing to `http://4.213.139.185:8300`
2. Add an APIM policy on `/teams/*` operations to route to the Teams backend
3. Verify the route is active by hitting the health endpoint through APIM
4. Optionally roll back by removing the policy (traffic falls back to main app)

### Deployment Steps

```bash
# 1. Install and start the Teams service
sudo cp deploy/docwain-teams.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable docwain-teams
sudo systemctl start docwain-teams

# 2. Verify it's healthy
curl http://localhost:8300/health

# 3. Update APIM routing (Teams traffic → port 8300)
python -m teams_app.deploy route-teams

# 4. Verify end-to-end
python -m teams_app.deploy verify

# 5. (Optional) Rollback — route Teams back to main app
python -m teams_app.deploy rollback
```

### Migration from src/teams/

Once the standalone service is verified:
1. APIM routes Teams traffic to port 8300 — main app stops receiving Teams messages
2. The `/teams/messages` route in `src/main.py` becomes dead code
3. Remove it in a future cleanup — no rush, it's harmless

## Scale Considerations

Designed for small team initially, grows to org-wide:
- Start with 3 concurrent document workers, increase as needed
- Qdrant collections per tenant — scales naturally
- If embedding becomes a bottleneck, offload to main app HTTP or dedicated embedding service
- If Teams message volume grows, add Redis-backed task queue (Celery) replacing asyncio workers
