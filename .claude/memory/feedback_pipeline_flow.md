---
name: DocWain Pipeline Flow Rules
description: HITL-gated three-stage pipeline. Upload auto-triggers extraction; screening and training are HITL-triggered only. KG belongs in training, not screening.
type: feedback
originSessionId: 93168fda-607e-4c51-b06c-5b5e0f18a6b1
---
Pipeline is HITL-gated with exactly ONE auto-trigger: upload → extraction.

1. Upload → Extraction: automatic (no separate Extract button).
2. Extraction → Screening: HITL review + manual trigger.
3. Screening → Training/Embedding: HITL review + manual trigger.

During training, three tracks run IN PARALLEL: embedding (Qdrant dense+sparse), Document Intelligence build, and KG update (Neo4j).

**Why:** HITL quality gates at two points (post-extraction, post-screening) are non-negotiable. Human must approve each stage before the next. User reaffirmed this on 2026-04-17.

**How to apply:**
- Never add auto-dispatch between HITL-gated stages. Never bypass human review.
- KG build belongs in the TRAINING stage, not the screening stage. Earlier memory claiming "Screening auto-triggers KG building" is stale. Code currently triggers KG in multiple places (extraction_service.py, embedding_service.py, dataHandler.py) — this is a gap vs the canonical flow, not a feature.
- See `project_docwain_pipeline.md` for the full canonical flow.
