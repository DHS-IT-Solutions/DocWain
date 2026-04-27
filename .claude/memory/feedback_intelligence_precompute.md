---
name: Intelligence Precomputed at Ingestion, Consumed at Query
description: All query-time intelligence is precomputed at ingestion. As of 2026-04-23 the "DocIntel as training-stage artifact" framing is superseded — the Researcher Agent fills that role; DocIntel is an in-extraction ML/DL model.
type: feedback
originSessionId: 93168fda-607e-4c51-b06c-5b5e0f18a6b1
---
DocWain must build ALL intelligence that retrieval will later consume during ingestion. Query time is strictly a lookup-and-combine path. No heavy compute at query time.

## Current (2026-04-23 onwards) — what is built where

**In the extraction stage:**
- **Document Intelligence (DocIntel)** — an ML/DL model that understands layout, doc type, section structure, patterns, context. Runs during extraction, drives extraction quality. **Does NOT persist a separate DocIntel artifact post-extraction.** Its contribution is embedded in the quality of the extracted content. Improves over time via a learning loop. See `project_post_preprod_roadmap.md` item 2.

**In the training stage (HITL-approved, runs in parallel):**
- Dense + sparse embeddings → Qdrant
- **Researcher Agent** — domain-aware deep analysis (medical / expenses / contracts / resumes etc. via plugin-shaped adapters). Outputs → Qdrant payload mapped by document_id + Neo4j insight-typed nodes/edges. Subsumes what the old "DocIntel training-stage track" (summaries, key_facts, key_values, entities, answerable topics) used to produce, plus deeper inferential work. See `project_researcher_agent_vision.md`.
- Knowledge Graph (entities + relationships) → Neo4j, running as a background service.

**Read at query time:**
- Vector similarity (dense + sparse, RRF-fused)
- Researcher Agent insights via chunk payload (Qdrant) and graph traversal (Neo4j)
- KG traversal for entity/relationship lookups

## Deprecated framing (do not reintroduce)

The previous framing that described DocIntel as a training-stage track (summaries, key_facts, key_values, entities, answerable topics stored in Qdrant payload + Mongo pointers) is **superseded**. Any code or design that:
- Runs a standalone "DocIntel build" step after extraction, OR
- Stores DocIntel outputs separately from extraction/researcher/KG, OR
- Expects `doc_context` at query time to contain a DocIntel-typed artifact

...is working from the old model. The equivalent query-time context now comes from the Researcher Agent's outputs.

**Why (2026-04-17 original):** This is the only way to deliver fast, intelligent responses over large collections. User: "during this whole pipeline the intelligence should be built up completely and efficiently, during the retrieval it will be faster to process the intelligence gathered."

**Why (2026-04-23 update):** DocIntel was conflated with two separate concerns — (a) making extraction smarter, and (b) providing query-time intelligence. Separating them cleanly: DocIntel does (a) inside extraction; the Researcher Agent does (b) during training. See `project_researcher_agent_vision.md` and `project_post_preprod_roadmap.md` item 2.

## How to apply

- When adding a new type of *in-extraction* understanding (better layout detection, new doc-type classifier, new pattern matcher), wire it through DocIntel in the extraction stage.
- When adding a new type of *query-time* intelligence (cross-doc summaries, answerability scoring, domain insights, suggestions), build it in the Researcher Agent in the training stage and persist to Qdrant + Neo4j.
- Retrieval code should only READ pre-computed artifacts (chunks, researcher insights, KG), never compute them inline.
- If existing query-time code is computing something expensive (LLM calls beyond the final generation, graph walks, re-extraction), that's a sign the precomputation step is missing — move the work to training stage (Researcher Agent) or extraction (DocIntel), depending on whether the computation needs cross-doc reasoning or per-doc understanding.
