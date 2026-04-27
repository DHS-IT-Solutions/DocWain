---
name: DocWain Document Processing Pipeline
description: Three-stage HITL pipeline — upload/extract → screen → train/embed. Domain-agnostic, works for any doc type. User's canonical description 2026-04-17.
type: project
originSessionId: 93168fda-607e-4c51-b06c-5b5e0f18a6b1
---
DocWain is a domain-agnostic document intelligence product. Users upload documents of ANY type (PDF, image, Excel, Word, email, scanned docs, forms, multi-column layouts) and DocWain builds queryable intelligence over them. Three HITL-gated stages.

## Stage 1 — Upload + Extraction (only auto-triggered stage)

- UI handles upload.
- Upload auto-triggers extraction (no separate Extract button).
- Extraction requirements (user's words): "analyze the documents in detail and accurately capture the contents ... documents may be of any format ... efficiently processed and its contents are captured including clear OCR and layout aware extraction. This is the source hence this needs to be 100% efficient and accurate."
- MongoDB status: `UPLOADED` → `EXTRACTION_IN_PROGRESS` → `EXTRACTION_COMPLETED` | `EXTRACTION_FAILED`

## Stage 2 — HITL review → Screening (HITL-triggered)

- HITL reviews extracted content, selects documents for screening.
- HITL manually triggers screening with a screening TYPE.
- Screening is type-specific:
  - Security → scan for PII + confidential information, highlight findings
  - Resume → validate cert names + educational institutions via internet
  - Additional types per plugin architecture
- MongoDB status: `SCREENING_IN_PROGRESS` → `SCREENING_COMPLETED` | `SCREENING_FAILED`

## Stage 3 — HITL review → Training/Embedding (HITL-triggered)

- HITL reviews screening reports, manually triggers training/embedding.
- During training, THREE tracks run IN PARALLEL:
  1. Embeddings — extracted content → dense + sparse vectors → Qdrant (for RAG)
  2. Document Intelligence — per-doc summaries, key_facts, key_values, entities, answerable topics → stored for fast retrieval-time consumption
  3. Knowledge Graph update — entities + relationships → Neo4j
- MongoDB status: `EMBEDDING_IN_PROGRESS` → `TRAINING_COMPLETED` | `EMBEDDING_FAILED`
- Additional terminal states: `TRAINING_BLOCKED_SECURITY`, `TRAINING_BLOCKED_CONFIDENTIAL`, `TRAINING_PARTIALLY_COMPLETED`

## Post-pipeline — Retrieval (runtime)

- User queries run against the pre-built intelligence.
- Retrieval should be fast because ALL intelligence (embeddings + DocIntel + KG) is pre-computed at ingestion — query time only reads and combines.

## Guiding architectural principles

- **Compute-once-at-ingest, read-many-at-query.** Any heavy intelligence computation belongs in the training stage, not at query time.
- **Domain-agnostic excellence.** Extraction, screening, and prompts must work generically — no hardcoded branches for "invoice" vs "resume" vs "contract" in the core path. Per-domain behavior comes from learned profiles, not hardcoded rules.
- **HITL gates are non-negotiable.** User approves each stage transition after stage 1.
- **MongoDB status values are an immutable contract with the UI.** See feedback_mongo_status_stability.md.

## Known code-vs-spec gaps (as of 2026-04-17)

- Extraction engine (`src/extraction/engine.py`) wires 4 parallel engines but 3 are TODO stubs (`structural.py`, `semantic.py`, `vision.py` all `return {}`). Only `v2_extractor.py` (Ollama) actually runs. Layout-aware extraction for forms / multi-column PDFs / scanned images is therefore degraded against the "any format, layout aware, 100% accurate" requirement.
- KG build is triggered in BOTH `src/api/extraction_service.py:221-262` AND `src/api/embedding_service.py:240-256` and `src/api/dataHandler.py:1697, 1711-1737`. This contradicts the user's canonical flow where KG belongs cleanly in the training stage. Likely a legacy from an earlier pipeline design.
- Screening plugin implementations in `src/screening/plugins/` are largely TODO stubs (risk_analyzer, pii_detector, etc.). Resume-specific internet validation for educational institutions is not obviously implemented.
- Training-stage parallelism: in `src/api/embedding_service.py`, DocIntel's `_upsert_doc_intelligence` is called sequentially inside the embedding flow, not in a parallel track.

**Why this memory:** user's 2026-04-17 clarification. Previous memory notes (pipeline flow, KG trigger timing) are partially stale — this supersedes.

**How to apply:** Any new work in this project is judged against this pipeline. Flag any proposal that reintroduces auto-dispatch between HITL stages, changes MongoDB status values, or defers intelligence computation to query time.
