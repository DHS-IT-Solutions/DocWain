---
name: Enterprise Architecture Redesign
description: Major architecture redesign approved 2026-03-15 — new pipeline, Celery workers, plugin screening, KG-enhanced RAG
type: project
---

Full architecture redesign approved on 2026-03-15. Design document: `docs/plans/2026-03-15-docwain-enterprise-architecture-design.md`

Key decisions:
- Pipeline: Upload (auto-extract) → HITL → Screen (auto-KG) → HITL → Embed
- Extraction: LayoutLM/DocFormer (Triton) + qwen3:14b (Ollama) + glm-ocr, merged
- Queue: Celery + Redis, isolated workers per stage
- Screening: Plugin architecture, security plugins always mandatory
- KG: Neo4j, builds async after screening, never blocks pipeline
- Embedding: Single dense+sparse per chunk, enriched payloads
- Retrieval: 3-layer (Qdrant + Neo4j + MongoDB) + cross-encoder rerank
- Storage: MongoDB = control plane only, Azure Blob = all document content
- Deployment: Monolith API + isolated Celery workers
- /api/askStream deprecated, use /api/ask with stream flag

**Why:** Month of incremental changes yielded minimal accuracy improvement. This is a foundational rebuild for enterprise-grade accuracy.

**How to apply:** All implementation work must follow this design document. No shortcuts on extraction quality or HITL gates.
