---
name: Teams App Must Be Fully Isolated
description: Teams service must never call main app APIs or modify src/ — completely self-contained pipeline
type: feedback
---

Teams app must be fully self-contained — no calls to the main app's `/api/ask` or any other endpoint.

**Why:** Using `/api/ask` interferes with the main app's resources and creates coupling. The user explicitly rejected the proxy approach after seeing it in action.

**How to apply:**
- Teams service queries Qdrant directly via `TeamsQueryHandler` and generates responses with its own LLM gateway
- Teams embedding uses `teams_app/pipeline/embedder.py` (not `src/api/dataHandler.train_on_document` which triggers 12+ slow LLM calls)
- All Teams-specific code goes in `teams_app/`, never modify `src/` for Teams-only features
- When `src/` functions are imported (e.g., `fileProcessor`, `_run_security_screening`), they are read-only consumers — never patched
