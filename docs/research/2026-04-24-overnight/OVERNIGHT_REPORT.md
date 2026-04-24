# DocWain Overnight Research + Implementation Report

**Date:** 2026-04-24 → 2026-04-25
**Branch:** `preprod_v02`
**Scope:** UAT readiness. Move DocWain from Q&A portal toward an intelligent document-analysis platform. Safe additive changes only; no UI contract breaks; each behaviour change feature-flagged for rollback.

---

## 1. Executive summary

Three parallel research agents audited the RAG pipeline, the Researcher Agent, and the multi-document analysis surface. A live query battery exercised 12 representative prompts against a 43-doc financial profile. Findings converged on one sentence:

**The plumbing for intelligent analysis already exists — the API surface to reach it from the UI does not.**

DocWain already writes, at ingestion time, a rich knowledge base per profile:
- Redis hot-cache: entities + facts + claims + relationships + document summaries (`src/intelligence/hot_cache.py`)
- Neo4j KG: document/section/chunk/entity graph with `MENTIONS` + `SIMILAR_TO` + `RELATED_TO` + timeline edges (`src/kg/neo4j_store.py:135-363`)
- Mongo `profile_intelligence`: per-doc briefs + cross-document analysis (`src/intelligence/profile_intelligence.py:390-421`)
- Mongo `researcher.*` strand: per-doc Researcher Agent output (`src/tasks/researcher.py:178-213`)
- Mongo `doc_intelligence`: summaries + answerable_topics + key_facts + entities

What was missing: no endpoint exposed any of this to the UI. The Q&A endpoint (`/api/ask`) treated every query as a retrieval-and-generate round, never aggregating across the corpus it had already analysed. The Researcher Agent wrote insights to Mongo/Qdrant/Neo4j but the UI had no way to read them back.

**Commits shipped this session:**

| Commit | Change |
|---|---|
| `0ad02b8` | New endpoints: `GET /api/documents/{id}/researcher` + `GET /api/profiles/{id}/insights`. Researcher task now persists full insight payload on Mongo. `get_prevalent_entities` helper on hot-cache. Document status response includes a `researcher` strand. |
| `1a0e2c2` | Extended thinking on analysis-intent queries (compare/summarize/overview/investigate/aggregate/analyze). Cap `get_live_logs` read at 200 entries (dropped progress endpoint latency 7 s → 1.3 s). |

**Live impact verified:** profile insights endpoint returns real aggregations in <1 s; progress endpoint now sub-second. Extended thinking produces more structured, risk-aware summaries on analysis queries.

---

## 2. Research findings (three parallel audits)

### 2.1 RAG pipeline — `src/agent/core_agent.py`, `src/generation/`

**System is answer-centric, not analysis-centric.** Every `/api/ask` call runs the same shape: retrieve → rerank → compose prompt → single-shot LLM.

- Retrieval is solid (Qdrant dense + keyword F1 + cross-encoder rerank at `src/retrieval/reranker.py:35-92`; dynamic top-K per task; intent-aware synonyms).
- System prompt is rigorous on grounding (zero-fabrication enforced at `src/generation/prompts.py:13-75`).
- **Gaps flagged:** no cross-document correlation scoring, no anomaly detection, no temporal pattern extraction, no contradictions check. KG is integrated as a retrieval hint but cross-doc entity traversal is stub. Feedback is captured (`src/api/learning_signals.py`) but never fed back into re-ranking within the session.
- **Extended thinking was hardcoded off for vLLM** at `src/agent/core_agent.py:602` — addressed in `1a0e2c2`.

### 2.2 Researcher Agent — `src/tasks/researcher.py`

**Scaffold complete, surface missing.**

- Fires post-embedding via `src/api/pipeline_api.py::trigger_embedding` (line 192-205). Weekly profile refresh beat schedule behind `DOCWAIN_RESEARCHER_WEEKEND_REFRESH_ENABLED`.
- Reads extraction JSON from Azure Blob. Output shape (`ResearcherInsights` dataclass at `src/docwain/prompts/researcher.py:42-49`):
  ```
  summary · key_facts · entities · recommendations
  anomalies · questions_to_ask · confidence
  ```
- Writes: Mongo `researcher.*`, Qdrant payload `researcher_insights`, Neo4j `(doc)-[:HAS_INSIGHT]->(Insight)`.
- **Pre-fix:** only `summary_preview` was on Mongo; the UI could not read the rich payload back.
- **Pre-fix:** `GET /api/documents/{id}/status` was silent about the researcher strand; UI had no signal that insights were ready.
- Both fixed in `0ad02b8`.

### 2.3 Multi-document analysis — `src/intelligence/*`, `src/kg/*`, `src/rag_v3/corpus_analytics.py`

**The plumbing exists in abundance:**

- `src/intelligence/cross_doc.py:40-100` — version chain + entity overlap (Jaccard ≥ 0.15) across documents, persisted to Neo4j.
- `src/intelligence/cross_doc.py:204-250` — doc similarity via Qdrant cosine ≥ 0.85, `SIMILAR_TO` edges written.
- `src/intelligence/hot_cache.py:32-140` — entity, fact, claim, relationship, domain cache written at ingestion; per-profile aggregation keyed.
- `src/rag_v3/corpus_analytics.py:141-224` — answers "how many invoices?", "total amount?", "average X?" deterministically from chunk scans.
- `src/rag_v3/comparator.py:229-363` — document ranking by field strength.
- `src/kg/neo4j_store.py:251-345` — entity probes, `RELATED_TO`, timeline nodes.
- `src/tools/insights.py:176-236` — domain-specific anomaly detection (invoice / legal / medical).

**What was missing:** a single endpoint that returns the consolidated cross-document view. Fixed in `0ad02b8` with `GET /api/profiles/{id}/insights` — one request aggregates dominant domain + prevalent entities bucketed by type + top relationships + every Researcher Agent anomaly/recommendation/question across the corpus. Zero LLM round trips.

---

## 3. Live query battery (12 prompts, real profile)

**Profile:** `69e8b9e4af9231725f583f29` — 43 documents (financial domain: invoices, POs, quotes across Apparel / Furniture / Hospo / Clean domains).

Full JSON in `docs/research/2026-04-24-overnight/live_query_battery{,_part2}.json`.

| # | Query | Time | Grounded | What worked | What didn't |
|---|---|---|---|---|---|
| factual-1 | What vendors appear in the invoices? | 26 s | ✓ | Listed 4 vendors correctly. | — |
| factual-2 | Total value of all invoices? | 35 s | ✗ | Returned a total + breakdown. | Not grounded — LLM invented aggregate. |
| factual-3 | Who issued PO5? | 17 s | ✗ | Correctly said "PO5 not found, but these PO numbers exist: …". | — |
| insight-1 | Summarise key themes across documents. | 43 s | ✗ | Structured domain+entity overview. | Entity list felt generic. |
| insight-2 | Top three vendors by frequency. | 25 s | ✗ | Returned vendor names + which docs mention them. | Frequency counts off. |
| insight-3 | Any anomalies or inconsistencies? | 36 s | ✓ | "No anomalies detected." | **Concerning — it should have flagged duplicate PO numbers and mismatched amounts. System is not actually running anomaly detection, it's just claiming there are none.** |
| insight-4 | What % are Apparel vs Furniture vs Hospo? | 41 s | ✗ | "The documents do not specify these categories." | **Wrong — filenames carry the category (`PO8_Apparel_*`).** System never considers filename as a signal. |
| cross-1 | Compare Apparel vs Furniture invoices. | 30 s | ✗ | Returned a comparison table with org + amount. | Picked one doc per category; missed aggregation. |
| cross-2 | Docs referencing the same PO number? | 26 s | ✗ | Table of doc → PO number, correctly identified shared PO 5206556 across 2 docs. | — |
| temporal-1 | Order invoices chronologically, identify gaps. | 37 s | ✗ | Table by date + amount; caught some dates. | No gap analysis. |
| analysis-1 | Supplier risk review highlights. | 37 s | ✗ | Structured risk categories. | Heuristic-level, no numbers. |
| analysis-2 | Executive summary for weekly briefing. | 24 s | ✗ | Structured exec summary with entities + time range. | Fabricated "24 documents" — profile actually has 43. |

**Quality signals from this battery:**

- **Latency median 30 s** per `/api/ask` call — acceptable for analysis but too slow for rapid iterative use.
- **Grounded ratio 2 / 12 = 17 %** — most answers synthesise beyond the literal evidence. The grounding check regex-matches numbers in the output against the evidence block; any LLM-invented aggregate trips it.
- **Anomaly blind spot (insight-3)** is the most dangerous finding. The system hallucinates "no anomalies" rather than admit it isn't running an anomaly check. **The new `/api/profiles/{id}/insights` endpoint surfaces the Researcher Agent's `anomalies` bucket as a structured field, so the UI can render actual anomalies instead of trusting the LLM to assert their absence.**
- **Filename/metadata blind spot (insight-4)** — LLM wasn't given the filenames as a separate signal. Worth a prompt-engineering follow-up: inject source_file into each evidence chunk header.

**After extended-thinking enabled** (`1a0e2c2`): a similar "summarise key themes + flag risks" query returned structured output (`## Key Themes` / `## Risk Areas` / `### Key Takeaway`) with named entities and doc counts. Latency ~38 s (no thinking was ~25 s). Trade-off is worth it for analysis tasks.

---

## 4. Performance audit

Windowed over last 6 hours of app logs. Request-completion latencies by endpoint:

| Endpoint | n | Avg | Notes |
|---|---|---|---|
| `/api/ask` | 16 | 31 616 ms | Analysis-dominated. |
| `/api/documents/embed` | 1 | 759 043 ms | 12.6 min for a 30-doc embed — dominated by KG inline + doc-intelligence-per-doc. The KG async move (commit `87ddf21`) + vLLM routing (`2340050`) should cut this in half on the next run. |
| `/api/extract` | 5 | 656 334 ms | Batch of N docs; per-doc median ≈ 107 s after the earlier optimizations. |
| `/api/extract/progress` | 15 | **7 091 ms → 1 300 ms** | UI polls this every ~5 s. The uncapped `get_live_logs` Redis read was scanning thousands of entries per poll. Commit `1a0e2c2` caps at 200 entries. Verified 1.3 s after fix. |
| `/api/gateway/screen` | 1 | 72 318 ms | Single-shot screen. |
| `/api/profiles/{id}/intelligence` | 1 | 1 237 ms | Mongo fetch, unchanged. |

**Other performance wins that landed earlier this week but are worth restating:**

- All doc-processing LLM calls now route through the vLLM gateway instead of spinning up a second in-process Ollama client on an already-saturated GPU (commit `0f7cadb`, `2340050`). No more 300 s Ollama timeout retries interrupting embedding or KG extraction.
- KG extraction moved off the extraction critical path to Celery `kg_queue` (commit `87ddf21`). Extraction wall-clock drops from ~150 s to ~60-80 s per doc; KG enrichment completes in the background before screening is clicked.
- Orphaned batch state cleared on every app startup (`9e958cd`) so a SIGKILL mid-batch doesn't lock the subscription for 30 minutes.
- `/api/extract/progress` now strictly scoped to the current run (commit `ea6c294`, `3407b6c`) — historical docs never bleed into the progress bar; endpoint now conforms exactly to the UI contract (`total_documents`, `uploaded`, `overall_progress`, `elapsed_time`).

**GPU snapshot:** 74 122 / 80 000 MiB used (mostly vLLM at 0.90 util). Embedding model forced to CPU on low-memory events. Possible future win: drop vLLM `--gpu-memory-utilization` to 0.78 to give embedding headroom back on GPU. Not changed tonight (needs a clean window to restart vLLM).

---

## 5. Changes shipped (this session only)

### 5.1 New API endpoints

#### `GET /api/documents/{document_id}/researcher`
Returns the full Researcher Agent insight payload for a single document.
```
{
  document_id, status,
  summary_preview, confidence,
  insights: {
    summary, key_facts[], entities[], recommendations[],
    anomalies[], questions_to_ask[], confidence
  },
  started_at, completed_at, updated_at, elapsed_ms, error
}
```
Defined at `src/api/pipeline_api.py::get_document_researcher`.

#### `GET /api/profiles/{profile_id}/insights`
Corpus-level aggregation for the UI's intended "Insights" tab. Zero LLM calls. Sub-second on a populated profile.
```
{
  profile_id, dominant_domain,
  document_counts: { total, by_status: {...} },
  prevalent_entities: { by_type: { identifier:[…], person:[…], … }, total },
  top_relationships: [ {subject, object, relation_type, confidence} ],
  researcher: {
    enabled,
    summaries: [{doc_id, name, confidence, summary}],
    anomalies: [{doc_id, name, confidence, anomaly}],
    recommendations: [{doc_id, name, confidence, recommendation}],
    questions_to_ask: [{doc_id, name, confidence, question}],
    docs_with_insights
  }
}
```
Query params: `min_doc_count` (default 2), `limit_entities` (default 30), `include_researcher` (default true).
Defined at `src/api/profile_intelligence_api.py::get_profile_insights`.

**Live-verified on the 43-doc test profile:**
- `dominant_domain = financial`
- 43 total docs; 18 SCREENING_COMPLETED + 25 TRAINING_COMPLETED
- 10 entity-type buckets populated; examples: `"invoice number"` in 15 docs, `"nancy doe"` in 8 docs, `"coffee bliss suppliers"` in 7 docs, `"charleston blue seat sofa"` in 7 docs
- `researcher.*` counts are 0 for this profile because the docs were processed before the task was updated to persist the full insights dict. New extractions will fill these in.

#### Augmented `GET /api/documents/{id}/status`
Adds a `researcher` strand alongside extraction/screening/embedding/kg so the UI can detect "Insights ready" in a single existing poll.

### 5.2 Researcher task — full-insight persistence

`src/tasks/researcher.py::run_researcher_agent` now writes the full `ResearcherInsights` dict to Mongo on completion (not just `summary_preview`). Enables the new `/researcher` endpoint to serve a single Mongo fetch with no Qdrant/Neo4j dependency.

### 5.3 Extended thinking on analysis-intent queries

`src/agent/core_agent.py:598-615` — thinking was hardcoded off for vLLM. Now enabled when:
- `DOCWAIN_EXTENDED_THINKING` env var is not `false`
- Backend supports it (vllm/ollama/gemini/openai/azure)
- `task_type ∈ {compare, summarize, overview, investigate, aggregate, analyze}`

Lookup/extract queries stay fast. Analysis queries get deeper reasoning.

### 5.4 Progress-endpoint perf cap

`src/utils/logging_utils.py::get_live_logs` capped at 200 most-recent entries (was unbounded). The `/api/extract/progress` endpoint polls this every ~5 s; previously scanned the whole Redis list each time. Latency 7 s → 1.3 s.

### 5.5 New hot-cache helper

`src/intelligence/hot_cache.py::get_prevalent_entities(redis, profile_id, min_doc_count, limit, entity_type)` — pure deterministic scan over the entity cache written at ingestion. Backs the insights endpoint.

---

## 6. Gaps identified but NOT fixed this session

Tagged with estimated effort. Listed in priority order.

### A. Anomaly detection actually runs (HIGH — insight-3 battery failure is dangerous)
Today the LLM says "no anomalies" whether or not it looked. Options:
1. Surface Researcher Agent's `anomalies` field in the `/api/ask` response metadata (1-2 h).
2. Pre-compute deterministic anomalies: duplicate PO numbers, mismatched vendor↔amount, out-of-range dates (1 day).
3. Add anomaly-specific retrieval step that scans hot-cache facts/claims for contradictions (3-5 days).

### B. Cross-doc aggregation ("% by category", "top-N by total") (HIGH)
`insight-4` and `insight-2` showed this is the product's biggest quality gap. Simplest path: extend `src/rag_v3/corpus_analytics.py` to GROUP BY entity type or metadata field. ~50 LoC. The LLM would get back deterministic numbers to narrate. 1 day.

### C. Inject filename + doc_type into evidence headers (MEDIUM)
`insight-4` failed because the LLM couldn't see the filenames. Change `build_reason_prompt` to include `source_file` / `doc_type` as an evidence-chunk prefix. Small, safe. 2 h.

### D. Researcher Agent run on the existing corpus (MEDIUM)
Existing 43-doc profile has `researcher.*` counts = 0 because the task hasn't run on those docs. Either (a) trigger a one-shot backfill Celery task, or (b) add a manual re-run endpoint so the UI can force it. 2 h.

### E. Live query confidence → session-level re-ranking (MEDIUM)
Feedback data already captured. No current path back into the same session's retrieval. A simple token-level attention weight on the reranker based on `grounded=False` signals would lift grounding quality. 1-2 days.

### F. Thinking traces surfaced to the UI (LOW)
Thinking text is returned in `metadata.thinking` today but not exposed on the response. Adding a collapsed "reasoning trace" panel would help UAT reviewers understand why the model answered a certain way. 4 h.

### G. vLLM GPU headroom for embedding (LOW — ops)
Drop `--gpu-memory-utilization` from 0.90 → 0.78 on `docwain-vllm-fast.service` to free ~1 GB for the embedding model. Needs a clean restart window.

---

## 7. Commit log (this session, chronological)

```
0ad02b8  insights: expose Researcher Agent + corpus-level insights for UAT
1a0e2c2  intelligence + perf: extended thinking for analysis, cap live_logs read
```

Plus earlier commits this week that set the foundation:
```
3407b6c  extract: append uploaded doc to profile roster
ea6c294  fix(extract-progress): match UI contract — add 'uploaded' field, scope to current run only
2340050  embedding/training: route IntelligenceRouter + domain agents through vLLM gateway
0f7cadb  embedding/training: route doc-processing LLM calls through vLLM gateway
87ddf21  extract: move LLM knowledge extraction to async Celery task
9e958cd  fix(extract): clear orphaned batch locks/markers/rosters on startup
0ea6b07  fix(extract-progress): eliminate race window + fix zombie-elapsed drift
```

---

## 8. What to test during UAT

1. **Insights tab (new):** `GET /api/profiles/{profile_id}/insights` on any profile with ingested docs. Expect sub-second response and rich entity / status / researcher aggregations.
2. **Per-doc insights (new):** `GET /api/documents/{doc_id}/researcher` after embedding. Full Researcher Agent payload should return once the task has run on that doc.
3. **Status endpoint:** `GET /api/documents/{id}/status` now carries a `researcher` strand.
4. **Extended thinking:** ask `/api/ask` an analysis question like "compare X vs Y" or "summarise key themes and flag risks" — expect a more structured response (headers, risk buckets, named entities) at the cost of ~10-15 s extra latency.
5. **Progress endpoint responsiveness:** during an extraction batch, the UI's Insights polling should feel noticeably snappier (was 7 s, now 1.3 s).

---

## 9. Artifacts for later review

All saved under `docs/research/2026-04-24-overnight/`:

- `OVERNIGHT_REPORT.md` — this file
- `live_query_battery.json` — raw JSON of the first 7 query responses (factual + insight-1..4 + cross-1..2 interrupted)
- `live_query_battery_part2.json` — remaining 5 queries (cross-1..2 + temporal-1 + analysis-1..2) re-run after the app restart
- Raw text output of both batteries in `/tmp/live_query_battery*.log` (not committed to repo)

---

*End of report.*
