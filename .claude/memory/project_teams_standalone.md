---
name: Teams Standalone Service
description: Standalone Teams bot service (port 8300) — fully self-contained, no main app dependency for any operation
type: project
---

## DocWain Teams App — Standalone Service

Separate systemd service (`docwain-teams`, port 8300) that handles Microsoft Teams bot interactions independently from the main DocWain app.

**Why:** Teams needs auto-triggered pipeline (upload → extract → screen → embed → analyze), fast processing, and must never interfere with the main app.

**How to apply:** All Teams-specific code lives in `teams_app/`. Never modify `src/` for Teams-only needs. The service shares the same `.venv` but runs its own process.

### Architecture

- **Service:** `deploy/docwain-teams.service` → `python -m teams_app.main` on port 8300
- **APIM routing:** `dhs-docwain-api` routes `POST /teams/messages` to port 8300 via operation-level policy on `teamschat` operation
- **Azure Bot:** `dhs-docwain-bot` in `rg-docwain-dev`, endpoint stays at `https://dhs-docwain-api.azure-api.net/teams/messages`

### Key Design Decisions

1. **No `/api/ask` proxy** — Teams service does NOT call the main app. It searches Qdrant directly and generates responses via its own LLM gateway (Ollama Cloud).
2. **No `train_on_document`** — Main app's `train_on_document` triggers 12+ cloud LLM calls and is too slow. Teams uses `teams_app/pipeline/embedder.py` which does chunk → encode → Qdrant upsert with zero LLM calls.
3. **Auto-clear on upload** — Each new file upload deletes the old Qdrant collection so queries focus on the freshly uploaded document only.
4. **LLM-powered intelligence** — After embedding, `teams_app/pipeline/intelligence.py` sends document text to Ollama Cloud to generate: doc type, summary, key entities, and 5 smart questions shown as `messageBack` buttons on the intelligence card.
5. **messageBack buttons** — Card action buttons use `msteams.messageBack` type (not `Action.Submit`) to avoid Teams "Something went wrong" errors. Clicks appear as regular user messages.
6. **Plain text responses** — Query responses sent as plain text (not Adaptive Cards) because Teams renders markdown natively and Adaptive Card TextBlocks don't handle long Reasoner output well. Tables converted to bullet lists via `_format_for_teams()`.
7. **Production Reasoner** — Queries use `src/generation/reasoner.Reasoner` with the production expert system prompt, hybrid retrieval (dense + keyword fallback), and cross-encoder reranking when available.

### Pipeline Flow (5 steps)

```
Upload → Download → Extract → Screen → Embed → LLM Intelligence Analysis
```

Progress shown as a single Adaptive Card updated in-place at each stage.

### Key Files

```
teams_app/
├── main.py                     — FastAPI app, /teams/messages + /health
├── lifespan.py                 — Loads Qdrant, MongoDB, Redis, Bot adapter
├── config.py                   — TeamsAppConfig (port, concurrency, etc.)
├── bot/handler.py              — StandaloneTeamsBot (extends DocWainTeamsBot)
├── pipeline/
│   ├── embedder.py             — LLM-free embedding (chunk → encode → Qdrant)
│   ├── intelligence.py         — LLM doc analysis (summary, entities, 5 questions)
│   ├── cards.py                — Adaptive Card builders (progress, intelligence, error)
│   ├── fast_path.py            — Express vs full pipeline classification
│   └── workers.py              — Semaphore-bounded concurrent workers
├── proxy/
│   ├── query_handler.py        — Direct Qdrant search + Reasoner generation
│   └── query_proxy.py          — (legacy, unused — was /api/ask proxy)
├── storage/
│   ├── namespace.py            — teams_* prefix isolation
│   └── tenant.py               — AAD tenant auto-provisioning
├── signals/capture.py          — Learning signal capture for finetuning
├── graph/onedrive.py           — OneDrive/SharePoint download via Graph API
└── deploy.py                   — APIM route-teams / rollback / verify / status
```

### Commands

```bash
sudo systemctl restart docwain-teams          # Restart
journalctl -u docwain-teams -f                # Live logs
python -m teams_app.deploy route-teams        # APIM → port 8300
python -m teams_app.deploy rollback           # APIM → main app
python -m teams_app.deploy verify             # Health check
```

### User Commands in Teams Chat

- Type any question → searches documents, generates grounded response
- Upload a file → auto-processes with progress card + intelligence report
- `clear all` / `reset` / `clear documents` → deletes all embeddings, fresh start
- `help` → usage guide

### Known Constraints

- Embedding model `BAAI/bge-m3` can't load (HuggingFace unreachable from this server). Falls back to `BAAI/bge-large-en-v1.5` which is cached locally.
- Each new upload clears old documents. If user wants multi-document queries, they need to upload all files together (or this behavior needs to change).
