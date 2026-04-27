---
name: Storage Separation Rules
description: MongoDB is control plane only — NEVER store document content in MongoDB, use Azure Blob and Qdrant
type: feedback
---

MongoDB must NEVER store document content. It is the control plane only:
- MongoDB: pipeline state, UI-facing summaries, audit logs, blob_path references
- Azure Blob: raw files, extraction results (JSON), screening reports — all document content
- Qdrant: searchable chunks with vectors and enriched payloads
- Neo4j: knowledge graph nodes and edges

**Why:** Overloading MongoDB with document content causes performance issues and mixes concerns.

**How to apply:** Any extraction, screening, or document output goes to Azure Blob. MongoDB only gets summaries with a blob_path pointer.
