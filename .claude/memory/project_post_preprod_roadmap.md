---
name: Post-preprod_v01 Roadmap (updated sequencing)
description: 2026-04-23 resequenced plan after backend-quality audit and researcher-agent vision — extraction accuracy first, serving reversed to vLLM-primary
type: project
originSessionId: dc7597b6-0d4a-464a-8305-e7a3b998992a
---

On 2026-04-23, prod was reverted to `preprod_v01` after quality regressions on `main`. That day, after the backend-quality audit and the researcher-agent product-direction discussion, Muthu resequenced the plan. This memory supersedes the original 5-item ordering.

## Current ordering (execute in this order, one step at a time)

1. **preprod_v01 as baseline** — done. Prod runs here. Main is quarantined.

2. **Extraction accuracy overhaul — TOP PRIORITY.** Goal: accurately extract any uploaded document "without taking much time," targeting near-100% accuracy with no trade-off against accuracy for speed. Parallelize across documents (not across engines within a doc) to improve throughput. First deliverable is a research + design spec, not code.

   **Document Intelligence is an ML/DL model that lives INSIDE the extraction layer (per Muthu 2026-04-23 clarification).** It is NOT a post-extraction artifact, a summarizer, or a separate training-stage track. Its responsibilities:
   - **Understand the document before and during extraction** — layout type, document type/domain, section structure, table/form regions, reading order, field patterns, context (what this document is *about*). Runs as a learned model, not a rule set.
   - **Drive extraction strategy** — its understanding configures the main extractor: form-like doc → field extraction; narrative → section-aware chunking; scan with complex layout → layout-aware OCR; and so on.
   - **Improve over time through a learning loop** — patterns observed across extractions feed back into DocIntel's training set. Cached templates and model updates live as artifacts in Azure Blob per `feedback_adapter_yaml_blob.md`.
   - **Does not persist a separate DocIntel artifact after extraction.** Once extraction completes, its contribution is embedded in the *quality* of the extracted content. Summaries/entities/key_facts/analysis that used to be "DocIntel output" are produced by the **Researcher Agent** in the training stage instead (see `project_researcher_agent_vision.md`).

   This supersedes the prior framing where DocIntel was a training-stage artifact read at query time.

   See `project_researcher_agent_vision.md` for why extraction accuracy is the load-bearing layer for all downstream research.

3. **RAG + Researcher Agent as one integrated workstream.** The researcher runs as a fourth parallel track in the training stage (alongside embeddings, DocIntel, KG), performs domain-aware deep analysis (medical / expenses / contracts / resumes via plugin-shaped adapter YAMLs in Azure Blob), and **persists insights to both Qdrant (mapped to document_id) and Neo4j**. Refresh cadence: incremental update when new documents land, periodic weekend whole-set refresh. Query-time is lookup-only of precomputed insights + chunks. See `project_researcher_agent_vision.md`.

4. **KG ingestion as a background service during training and embedding.** Remove the current KG-build triggers from `src/api/extraction_service.py`, `src/api/embedding_service.py`, and `src/api/dataHandler.py` (per `feedback_pipeline_flow.md`), consolidate into a single async task that kicks off after HITL screening approval as part of the training stage. Must not block user-facing latency.

5. **Serving stack — REVERSED from audit recommendation.** User decision 2026-04-23: **vLLM local primary, Ollama Cloud 397B as fallback.** Overrides the 2026-04-23 audit verdict which recommended Cloud-primary based on quality. The trade-off: local vLLM 14B is measurably less intelligent than Cloud 397B (see `project_audit_2026_04_23_backend_quality.md` Q9/Q10), but user-perceived latency is the priority. Mitigate the intelligence gap at the engineering layer (prompting, retrieval, Researcher Agent precomputed insights) rather than at the model layer. Do NOT route to Ollama local — quality failures too severe.

6. **teams_app cherry-pick + standalone rebuild — LAST.** Deferred after extraction, researcher, KG move, and serving work. Low-urgency surfaces; don't let them block the foundational rebuild.

## Why this sequence
- **Extraction accuracy is foundational** to every downstream layer — insights, RAG, researcher, KG all inherit any extraction error. Getting this right first prevents compounding failures later.
- **Researcher depends on accurate extraction** — doing deep medical/expense analysis on wrong numbers is worse than doing nothing.
- **Users should not feel latency** — vLLM-primary with Researcher precomputation is the latency design; the intelligence gap is an engineering problem to solve in the RAG+Researcher layer, not a serving problem.
- **teams_app is a user surface** — it doesn't block the platform. Deferring it doesn't block anything.

## How to apply
- Never start an item without: a fresh baseline, an eval that will score the change, and explicit scope agreement.
- **Item 2 (extraction) is research-first.** Expect a design spec that maps current extraction path → where it loses accuracy/time → what replaces it. Don't start rewriting `src/extraction/` engines until the design is approved.
- **Item 3 (RAG + researcher) is cross-cutting** — schema changes in Qdrant payload and Neo4j ontology, a new adapter framework, and new training-stage orchestration. Must have its own design spec before implementation.
- **Item 5 (serving swap) requires wiring vLLM** as the gateway primary — currently `src/llm/gateway.py` on preprod_v01 wires Ollama Cloud as primary and never instantiates the vLLM client. This is a code change, not just config.
- Respect all existing rules: HITL gates, MongoDB status immutability, storage separation (Blob / Mongo / Qdrant / Neo4j), no wall-clock timeouts, engineering-first-training-last, no customer data in training, no Claude attribution in commits.

## Previously in this memory (2026-04-23 original)
The original 5-point list had extraction and RAG as items 3 and 4 (similar), serving as a full "vLLM vs Ollama" audit (#5), and teams_app cherry-pick as item 2. The audit is done; serving was decided (vLLM primary, reversing the audit's Cloud recommendation for latency reasons); teams_app was pushed last. Sequencing above reflects the post-audit state.
