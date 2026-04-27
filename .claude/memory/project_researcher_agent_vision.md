---
name: DocWain as Document Research Agent (not just Q&A)
description: 2026-04-23 product-direction upgrade — DocWain must proactively research documents and generate domain-aware, actionable insights at ingestion; Q&A is a surface, not the product
type: project
originSessionId: dc7597b6-0d4a-464a-8305-e7a3b998992a
---
On 2026-04-23, Muthu set the next-generation product direction for DocWain:

**DocWain is not a Q&A product. It is a document research agent.** When a user uploads a set of documents, a **backend Researcher Agent** performs all the analysis that set of documents supports — and surfaces *insights*, not just answers, whenever the user interacts.

### Concrete examples he gave
- **Medical documents uploaded** → in-depth clinical / prognostic research; summaries of the patient's condition; risk factors; recommended follow-ups; anomalies across records. Surfaced as proactive findings, not only as answers to questions.
- **Expense documents uploaded** → the backend researcher performs cost-structure analysis and **proactively suggests how to improve cost efficiency** — trend detection, duplicate vendors, category drift, benchmark comparisons. The user shouldn't have to *ask* "where can I save money"; DocWain should already know.

His framing: **"more than AI, this product should be AGI."** Taken as aspirational for agentic, reasoning-heavy, proactive behavior — not literal AGI. The grounded meaning: persistent background research, domain-adaptive analysis, prepared insights, and the ability to act on behalf of the user on the content they've provided.

### Where this sits in the pipeline

Ingestion-time work (all precomputed, zero heavy compute at query time):
1. **Extraction** — auto-triggered after upload. **Primary foundation. Extraction accuracy is the single most important quality gate** because every downstream layer depends on it. A Document Intelligence ML/DL model lives INSIDE this stage and drives extraction quality (layout detection, doc-type classification, pattern recognition, context understanding). DocIntel does not produce a separate persisted artifact — its effect is embedded in the accuracy of the extracted content. Per `project_post_preprod_roadmap.md` item 2.
2. **HITL review** → Screening (HITL-triggered).
3. **HITL review** → Training stage. This is where:
   - Embeddings (dense + sparse) → Qdrant
   - **Knowledge Graph ingestion** — runs ONLY HERE as a backend service, not in extraction and not in screening. Reaffirms the canonical pipeline. (Current code triggers KG in 3 places: `src/api/extraction_service.py`, `src/api/embedding_service.py`, `src/api/dataHandler.py` — this is a gap to fix.)
   - **Researcher Agent** — runs here, in parallel with embeddings and KG. Performs domain-aware deep analysis (medical reasoning, expense analysis, contract risk review, whatever the doc set supports) and persists outputs to BOTH Qdrant (payload mapped by document_id) AND Neo4j (as insight-typed nodes/edges). Refresh cadence: **incremental update** when new documents land, **periodic weekend whole-set refresh** for cross-doc re-analysis. The Researcher Agent subsumes what the old "DocIntel training-stage track" used to produce (summaries, key facts, answerable topics), plus the new deep-analysis work.

Query-time work: **lookup only**. Surface the precomputed insights, KG, DocIntel, and chunks; reason over them; answer.

### Why
- Extraction is the load-bearing layer. "These level of analysis can be performed only if the extracted content is accurate. So the primary goal should be extraction accuracy" — Muthu 2026-04-23.
- Background research + precomputed insights are the only way to deliver proactive, high-quality answers at fast query time. Reinforces existing `feedback_intelligence_precompute.md` rule.
- Keeps KG firmly in the training stage (already a rule per `feedback_pipeline_flow.md`). The canonical pipeline stands; the Researcher Agent is added alongside existing intelligence tracks, not a replacement.
- Agentic, plugin-shaped analysis keeps DocWain domain-agnostic by default (see `feedback_domain_extensibility.md`) — medical / expenses / resumes / contracts / HR etc. are adapters, not hard-coded branches. Generic adapter is the safe default; domain adapters are auto-detected and overridable.

### How to apply
- **Extraction accuracy is the top priority.** Any roadmap sequencing that skips or deprioritizes it is wrong. This reorders the 5-point roadmap: item 3 (extraction overhaul) is now the foundation, not just a speed fix.
- **The Researcher Agent belongs inside the training stage**, running in parallel with embeddings, DocIntel, KG. Never at query time, never in extraction, never in screening.
- **KG ingestion must be removed from extraction_service / embedding_service / dataHandler** and consolidated into the training stage trigger as a backend service after screening approval. Matches the canonical pipeline.
- **Domain-aware research is plugin-shaped** — drop-in adapter YAMLs (living in Azure Blob per `feedback_adapter_yaml_blob.md`) describe how to analyze medical vs expenses vs contracts vs resumes. A `generic` adapter always works. Never a closed enum.
- **Insights are a first-class persistent artifact** alongside embeddings/DocIntel/KG — stored, queryable, and surfaced at query time without re-running analysis.
- **Query-time performance must stay fast** — because the researcher ran during ingestion, query-time is purely lookup + generation. Consistent with `feedback_no_timeouts.md` and `feedback_intelligence_precompute.md`.
- **Before implementation:** this is a design-level change. Requires its own spec before code. Cross-cuts extraction, training stage orchestration, adapter YAML schema, retrieval, and the Reasoner. Do not start implementing piece-meal in an RAG batch.

### Sequencing implication

The 2026-04-23 5-point roadmap stays valid but gains explicit ordering rationale:

1. preprod_v01 = baseline (done)
2. Cherry-pick `teams_app/` + rebuild `standalone/` (unblocked; low risk; keeps the user-facing surfaces working)
3. **Extraction accuracy overhaul** — foundation for everything downstream. Profile first, redesign second.
4. **RAG + Researcher Agent** — both together, not sequentially. The researcher IS the intelligence layer; RAG is its query-time delivery surface. Needs its own design spec.
5. Serving decision — audited 2026-04-23; recommendation is "stay on Cloud 397B, raise `num_predict` budget, keep vLLM local as fallback, do NOT use Ollama local for user-facing traffic." Already decided; revisit only if cost/latency forces a change.

The Researcher Agent spec should be written **before** the RAG batch begins, because it determines what data exists at query time and therefore what the RAG layer is retrieving against.
