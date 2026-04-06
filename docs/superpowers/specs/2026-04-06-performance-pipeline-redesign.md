# DocWain Performance & Pipeline Redesign

**Date:** 2026-04-06
**Status:** Approved
**Scope:** Performance optimization (latency, accuracy, throughput, ingestion), document processing pipeline redesign with HITL, analytical intelligence

---

## 1. Current Bottleneck Map

### 1.1 Latency (Current: 5-15s per query, Target: sub-2s simple, streaming complex)

Typical query flows through this sequential chain:

```
Planner (1.5s LLM) ──┐
                      ├── parallel ── Pre-fetch retrieval (2s Qdrant)
Intent analysis (LLM) ┘
         │
    Reranking (0.6s, cross-encoder on CPU, BLOCKS)
         │
    RepairLoop (if LOW quality: +2-4s per iteration, up to 2 rounds)
         │
    Doc intelligence scroll (0.5s, sequential double Qdrant call)
         │
    KG enrichment (0.3-0.5s, Redis + Neo4j, BLOCKS)
         │
    Reason (3-5s LLM, only this stage streams)
```

Key findings:
- 3-4 sequential LLM calls per query
- 4-6 sequential DB round-trips
- Cross-encoder reranking blocks on CPU (~0.6s, up to 10 pairs)
- RepairLoop adds 2-4s when quality is LOW (full re-retrieval cycle)
- Streaming only exists for the final REASON step
- No query-result cache, no embedding cache

### 1.2 Accuracy

- **Embedding model**: `bge-large-en-v1.5` — general-purpose, not domain-tuned, causes retrieval misses
- **Chunking**: Section-aware but no overlap — boundary content lost
- **Multi-document**: No explicit cross-document reasoning step
- **Hallucination**: Grounding relies on LLM self-check (weak)
- **Extraction**: Table/form errors propagate through RAG chain uncorrected

### 1.3 Throughput

- Embedding model loaded in-process as singleton, single-threaded
- No request queuing or backpressure
- Cross-encoder always loaded regardless of need
- No batching across concurrent queries

### 1.4 Document Processing Speed

- Sequential extraction pipeline, no page-level parallelism
- Heavy compute steps (OCR, layout, schema detection) run serially
- No priority queue — large docs block small docs
- Active bug: 42M-char CSV blocking extraction pipeline (as of 2026-04-06)

---

## 2. Approach: Parallel Tracks

Three independent workstreams that touch different code paths:

- **Track 1 — Latency & Throughput**: Query routing, caching, parallelization
- **Track 2 — Accuracy & Quality**: Eval harness, embeddings, chunking, verification
- **Track 3 — Ingestion Speed**: Parallel processing, priority queues, smart routing

Plus cross-cutting redesigns:
- **Document Processing Pipeline** with HITL gates
- **Pre-computed Knowledge** as query-time enrichment
- **Analytical Intelligence** for insight generation

---

## 3. Track 1 — Latency & Throughput

### 3.1 Query Classifier (Entry Point)

Lightweight, zero-LLM classifier at the top of the execution router:

```
Request arrives
    │
    ▼
QueryClassifier (rules + NLU, <50ms)
    │
    ├── SIMPLE (lookup, factoid, single-entity)
    │     → FastPath: embed → search → top-3 rerank → stream reason
    │     → Target: sub-2s
    │
    ├── COMPLEX (multi-doc, compare, aggregate, investigate)
    │     → FullPath: parallelized pipeline with streaming
    │     → Target: first token <3s, full response <10s
    │
    ├── ANALYTICAL (suggest, improve, recommend, analyze trends)
    │     → AnalyticalPath: broad retrieval + full enrichment + structured output
    │     → Target: first token <3s, full response <15s
    │
    └── CONVERSATIONAL (greeting, follow-up, clarification)
          → DirectPath: no retrieval, LLM-only with chat history
          → Target: sub-1s
```

Classification signals (no LLM needed):
- Intent rules from `src/nlp/intent_rules.py`
- Entity count from `src/nlp/query_entity_extractor.py`
- Query length, keyword patterns, conversation history
- Analytical signals: "suggest", "improve", "recommend", "analyze", "identify risks", "based on the data"
- Falls back to COMPLEX if uncertain

### 3.2 Fast Path Pipeline

For SIMPLE queries — strip non-essential stages:

| Stage | Current | Fast Path | Savings |
|-------|---------|-----------|---------|
| Planner LLM call | 1.5s | Skip — use raw query + entity extraction | -1.5s |
| Intent analysis LLM | Parallel but heavy | Use rule-based classifier result | -0s (already parallel) |
| Retrieval | 2s | Same, but single collection only | -0.5s |
| Reranking | 0.6s CPU | Top-3 only, GPU or lighter model | -0.4s |
| RepairLoop | 2-4s if triggered | Skip entirely | -2-4s |
| KG enrichment | 0.3-0.5s | Skip for simple lookups | -0.3s |
| Reason LLM | 3-5s | Shorter prompt, lower max_tokens, stream | -1-2s |
| **Total** | **5-15s** | **~1.5-2.5s** | |

### 3.3 Parallelized Complex Path

For COMPLEX and ANALYTICAL queries — same stages, maximally concurrent:

```
┌─── Planner (LLM) ────────────────┐
├─── Pre-fetch retrieval (Qdrant) ──┤  all parallel
├─── KG context lookup (Neo4j) ─────┤
└─── Doc intelligence (Qdrant x2) ──┘
              │
              ▼ (merge results)
     Reranking (GPU, top-10)
              │
              ▼
     Stream Reason (LLM) ← starts as soon as evidence ready
              │
     RepairLoop ONLY if confidence < threshold
     (runs as background re-generation, not blocking)
```

RepairLoop becomes async — user gets first answer via streaming, repair runs in background and pushes refined answer if quality improves.

### 3.4 Caching Layer

Three caches in Redis:

| Cache | Key | TTL | Purpose |
|-------|-----|-----|---------|
| **Query embedding** | hash(query_text) → vector | 1h | Avoid re-encoding identical queries |
| **Search results** | hash(query_vector + collection) → chunk IDs + scores | 5m | Avoid Qdrant round-trip |
| **Full response** | hash(query_text + profile_id) → response | 5m | Instant response for repeated queries |

Full response cache is opt-in per tenant (some may want real-time freshness).

### 3.5 Reranker Optimization

- Move cross-encoder to GPU (shares vLLM GPU during inference idle time)
- Or replace with ColBERTv2 late-interaction model (10x faster, comparable accuracy)
- Reduce candidate set: top-5 for SIMPLE, top-10 for COMPLEX

### 3.6 Embedding Service Extraction

Move embedding from in-process singleton to a separate service:
- Enables batching across concurrent requests
- Independent scaling (CPU pod pool for embeddings, GPU pods for LLM)
- Prepares for embedding model upgrade in Track 2

### 3.7 Extraction Pipeline Hardening

**Problem 1: Giant documents block the pipeline**

| Fix | What | Impact |
|-----|------|--------|
| Per-document timeout | Hard 5-minute wall clock per document, configurable by doc type | Unblocks queue |
| Document size gate | Pre-check extracted text size. If >5MB raw text, split into chunks before downstream processing | Prevents memory/time explosion |
| Parallel document processing | Process batch documents concurrently (ThreadPool, max 3), not sequentially | 1 stuck doc doesn't block others |
| Dead document detector | Background watchdog: if a document stays in `extracting` >10 minutes, mark failed, move to next | Auto-recovery |
| Progress heartbeat | Each extraction stage emits heartbeat to Redis. Watchdog checks heartbeat staleness | Detects stuck-in-stage |

**Problem 2: No backpressure for oversized documents**

Size gating happens within the auto-extract phase (before HITL Gate 1):

```
Upload triggers auto-extract
    │
    ▼
Size + type pre-check (<50ms)
    │
    ├── Small doc (<1MB text): normal extraction pipeline
    ├── Medium doc (1-5MB text): chunked extraction (split → extract chunks → merge)
    ├── Large doc (>5MB text): background extraction (async, user notified on completion)
    └── Oversized (>20MB text): reject with user guidance
    │
    ▼
All paths converge at EXTRACTION_COMPLETED → AWAITING_REVIEW_1
```

**Problem 3: Duplicate request rejection without recovery**

- Stale lock detector: if extraction lock held >15 minutes, auto-release
- Return current extraction status and ETA in rejection response
- Force re-extract endpoint that clears lock and restarts

**Problem 4: Neo4j is down, adding latency to every query**

- Circuit breaker on Neo4j client: after 3 consecutive failures, skip KG enrichment for 5 minutes, then retry
- Log clear warning when circuit opens
- Graceful degradation: queries work without KG context

---

## 4. Track 2 — Accuracy & Quality

### 4.1 Evaluation Harness (Build First)

Test bank structure:
```
eval/
  test_bank.json          # Q&A pairs with ground truth
  eval_runner.py          # Automated evaluation
  eval_report.py          # Score tracking over time
```

Each test case contains:
- Query text, profile/subscription context
- Expected source documents (ground truth retrieval)
- Expected answer (or key facts that must appear)
- Negative facts (must NOT appear — hallucination detection)
- Query category: `simple_lookup | multi_doc | extraction | table | comparison | analytical`

Scoring dimensions:

| Metric | What It Measures | How |
|--------|-----------------|-----|
| Retrieval Recall@k | Do right chunks appear in top-k? | Compare retrieved chunk IDs to ground truth |
| Retrieval Precision@k | Are irrelevant chunks polluting context? | Ratio of relevant to total retrieved |
| Answer Faithfulness | Is every claim grounded in source? | LLM-as-judge: for each claim, is there a supporting chunk? |
| Answer Completeness | Does the answer cover all expected facts? | Fact checklist against ground truth |
| Hallucination Rate | Does the answer contain fabricated content? | LLM-as-judge + negative fact check |
| Table/Form Accuracy | Are structured data values correctly extracted? | Exact match on key-value pairs |

Build 50-100 test cases from real user queries (anonymized). Run eval before and after every Track 2 change.

### 4.2 Embedding Model Upgrade

Staged upgrade with eval harness validation:

| Stage | Model | Why |
|-------|-------|-----|
| 1. Baseline | Current bge-large | Establish Retrieval Recall@10 baseline |
| 2. Drop-in upgrade | bge-large → bge-m3 | Multi-granularity (dense + sparse + colbert in one model) |
| 3. Domain fine-tune | Fine-tune bge-m3 on DocWain corpus | Train on (query, positive_chunk, hard_negative) triplets |
| 4. Evaluate | Run eval harness, compare | Only keep if measurable improvement |

Re-embedding strategy:
- New model writes to shadow Qdrant collections (e.g., `profile_123_v2`)
- Eval harness validates against both old and new
- Atomic swap once validated, keep old for 7 days as rollback

### 4.3 Chunking: Parent-Child with Overlap

```
Document
  └── Parent chunk (1500 tokens, semantic section boundary)
        ├── Child chunk A (300 tokens, overlap 50 tokens with B)
        ├── Child chunk B (300 tokens, overlap 50 tokens with A,C)
        └── Child chunk C (300 tokens, overlap 50 tokens with B)
```

Retrieval behavior:
1. Search against child chunks (fine-grained matching)
2. When child retrieved, expand to parent chunk for context
3. Deduplicate: if 2+ children from same parent, use parent once

Parent chunks stored as `parent_text` payload field on child points in Qdrant.

### 4.4 Cross-Document Reasoning

KG-guided cross-document retrieval:

```
User query: "How do the protocols differ between Site A and Site B?"
    │
    ▼
1. Standard retrieval → finds chunks from Site A docs
2. Entity extraction from retrieved chunks → ["Site A", "protocol"]
3. KG traversal → finds related entities → ["Site B protocol"]
4. Targeted retrieval → search specifically for Site B chunks
5. Merge both result sets → synthesizer gets full picture
```

- Only triggers when query intent is `compare`, `aggregate`, or `investigate`
- Max 2 expansion hops, max 8 additional chunks
- Depends on Neo4j circuit breaker (3.7) for graceful skip

### 4.5 Citation Verification

Separate verification pass after Reasoner:

```
Reasoner output (draft answer with inline citations)
    │
    ▼
Citation Verifier (fast model via vLLM)
    │
    ├── For each claim: does cited chunk support it?
    │     ├── YES → keep claim + citation
    │     ├── PARTIAL → flag as "inferred"
    │     └── NO → remove claim, flag as ungrounded
    │
    ▼
Verified answer with confidence score
```

- Uses fast model (docwain-fast) — checking, not generating
- Runs only on COMPLEX and ANALYTICAL queries
- Adds ~0.5-1s — acceptable for complex queries
- Can run parallel with streaming: stream draft, append verification badges

### 4.6 Table/Form Extraction Validation

```
Raw extraction (existing deep_analyzer)
    │
    ▼
Table Validator
    ├── Structure check: rows/cols align? Headers detected?
    ├── Value check: numbers parseable? Dates valid?
    ├── Cross-reference: totals match sum of line items?
    │
    ├── PASS → embed normally
    ├── WARN → embed with quality flag, retriever deprioritizes
    └── FAIL → re-extract with alternate strategy (vision model / native parser)
```

---

## 5. Track 3 — Ingestion Speed

### 5.1 Parallel Page Processing

```
Document arrives
    │
    ▼
Splitter (fast, <1s)
    ├── Page 1  ──┐
    ├── Page 2  ──┤
    ├── Page 3  ──┼── ThreadPoolExecutor (max_workers=4)
    ├── ...     ──┤
    └── Page N  ──┘
         │
         ▼
    Merger (reassemble, preserve ordering, handle cross-page content)
         │
         ▼
    Chunking + Embedding
```

- Max 4 concurrent pages per document
- Each page worker has 60s timeout
- Cross-page content (spanning tables, split sentences) handled in merger

### 5.2 Priority Queue

Weighted priority queue replacing FIFO:

| Factor | Weight | Logic |
|--------|--------|-------|
| Document size | 40% | Smaller docs higher priority |
| Document type | 30% | Structured (CSV, Excel) > unstructured (PDF) |
| Wait time | 20% | Docs waiting >5 min get priority boost |
| Tenant tier | 10% | Premium tenants get slight boost |

- `heapq`-based priority queue
- Express lane (<10 pages) with dedicated worker
- Bulk lane (>10 pages) cannot starve express lane

### 5.3 Smart Extraction Routing

```
Document arrives → Type + Size classifier (<50ms)
    │
    ├── CSV/Excel → Native parser (no OCR, no vision) → ~2s
    ├── Plain text/Markdown → Direct chunking → <1s
    ├── PDF <5 pages → Standard extraction → ~10-15s
    ├── PDF 5-50 pages → Parallel page extraction → ~30-60s
    ├── PDF >50 pages → Background bulk pipeline → async
    ├── Image → Vision model extraction → ~5s
    └── Word/DOCX → python-docx parser + layout → ~3-5s
```

### 5.4 Background Re-embedding

For embedding model upgrades (Track 2):
- Runs as background job during GPU idle time
- One collection at a time, oldest first, max 100 docs/hour
- Shadow collection strategy: write to `collection_v2`, atomic swap when complete
- Progress tracked in Redis: `reembed:progress:{collection_id}`

### 5.5 Extraction Pipeline Observability

Per-stage timing emitted to structured log + Redis:

```
stages = [
    "blob_download",      # Azure blob fetch
    "type_detection",     # File type classification
    "raw_extraction",     # OCR / parser / vision
    "sanitization",       # PII masking, text cleanup
    "deep_analysis",      # Schema detection, entity extraction
    "chunking",           # Section-aware splitting
    "embedding",          # Vector encoding
    "qdrant_upsert",      # Vector store write
]
```

Per-document extraction timeline enables targeted optimization.

---

## 6. Dependency Map & Execution Order

```
Track 1 (Latency)              Track 2 (Accuracy)           Track 3 (Ingestion)
──────────────────             ──────────────────           ───────────────────
3.1 Query Classifier           4.1 Eval Harness ◄────────── MUST BE FIRST
3.2 Fast Path                  4.2 Embedding Upgrade ──────► 5.4 Re-embedding
3.3 Parallel Complex Path      4.3 Parent-Child Chunks
3.4 Caching Layer              4.4 Cross-Doc Reasoning
3.5 Reranker Optimization      4.5 Citation Verification
3.6 Embedding Service          4.6 Table Validation ◄─────► 5.3 Smart Routing
3.7 Extraction Hardening ◄──────────────────────────────── 5.1-5.5 (all)
```

Execution order:

| Phase | Items | Why First |
|-------|-------|-----------|
| Phase 0 | 3.7 Extraction Hardening + Neo4j circuit breaker | Unblock stuck pipeline NOW |
| Phase 1 | 4.1 Eval Harness | Cannot measure improvement without it |
| Phase 2a | 3.1 Query Classifier + 3.2 Fast Path | Biggest latency win, lowest risk |
| Phase 2b | 5.3 Smart Extraction Routing + 5.1 Parallel Pages + 5.2 Priority Queue | Fix ingestion speed |
| Phase 3a | 3.3 Parallel Complex Path + 3.4 Caching | Second latency tier |
| Phase 3b | 4.3 Parent-Child Chunks + 4.2 Embedding Upgrade | Measure against eval harness |
| Phase 4 | 4.4 Cross-Doc Reasoning + 4.5 Citation Verification | Accuracy refinements |
| Phase 5 | 3.5 Reranker + 3.6 Embedding Service + 4.6 Table Validation | Scale and polish |

Phase 0 is immediate. Phases 2a and 2b run in parallel. Each phase validates against eval harness before merging.

---

## 7. Document Processing Pipeline with HITL

### 7.1 Pipeline Flow

```
Upload ──► Auto-Extract ──► ✋ HITL Review ──► Screen ──► ✋ HITL Review ──► Embed+KG+Intel ──► Complete
```

Each stage writes status to MongoDB. Each HITL gate waits for explicit human "proceed" from UI. No stage runs until prior stage completes and human approves.

### 7.2 Status Machine

```
UPLOADED
    │ (automatic)
    ▼
EXTRACTION_IN_PROGRESS → EXTRACTION_COMPLETED  ───or───  EXTRACTION_FAILED
                              │
                              ▼
                         AWAITING_REVIEW_1  ◄── human sees extraction results
                              │
                              │  ✋ human clicks "Approve & Screen"
                              │  (or "Reject" → REJECTED, or "Re-extract")
                              ▼
                    SCREENING_IN_PROGRESS → SCREENING_COMPLETED  ───or───  SCREENING_FAILED
                                                │
                                                ▼
                                           AWAITING_REVIEW_2  ◄── human sees screening results
                                                │
                                                │  ✋ human clicks "Approve & Process"
                                                │  (or "Reject" → REJECTED)
                                                ▼
                                      PROCESSING_IN_PROGRESS → PROCESSING_COMPLETED
                                      (embedding + KG + intelligence)
```

HITL rejection paths:
- At either gate: Approve, Reject (dead-end with reason), or Re-extract (back to extraction)
- Rejected documents marked REJECTED — never enter embedding or KG

### 7.3 Knowledge Gathering During Processing (The Heavy Phase)

Once human approves at HITL Gate 2, the PROCESSING phase does ALL knowledge work:

```
PROCESSING_IN_PROGRESS
    │
    ├──► Stage 1: Deep Analysis (sequential — everything else depends on this)
    │      ├── Entity extraction (people, orgs, dates, amounts, locations)
    │      ├── Relationship extraction (who → what → when)
    │      ├── Temporal analysis (date ranges, periods, durations, gaps)
    │      ├── Numerical analysis (totals, averages, trends, anomalies)
    │      ├── Table/form structure extraction + validation
    │      └── Document metadata (type, language, domain, sections)
    │
    ├──► Stage 2: Cross-Document Analysis (parallel with Stage 3)
    │      ├── Link entities to existing KG entities (dedup, merge, resolve)
    │      ├── Detect temporal relationships across documents
    │      ├── Detect contradictions across documents
    │      ├── Detect trends across documents
    │      └── Compute document similarity scores to neighbors
    │
    ├──► Stage 3: Embedding Generation (parallel with Stage 2)
    │      ├── Parent-child chunking (section 4.3)
    │      ├── Embed child chunks → Qdrant
    │      ├── Store parent text as payload metadata
    │      └── Quality validation on chunks before upsert
    │
    ├──► Stage 4: KG Population (after Stage 1 + 2)
    │      ├── Upsert entities to Neo4j with properties
    │      ├── Upsert relationships with evidence pointers
    │      ├── Upsert temporal edges (BEFORE, AFTER, OVERLAPS, CONTINUES, GAP_OF)
    │      ├── Upsert cross-document edges (CONTRADICTS, SUPERSEDES, CORROBORATES)
    │      └── Update entity co-occurrence scores
    │
    └──► Stage 5: Pre-computed Intelligence (after all above)
           ├── Document summary (stored, not generated at query time)
           ├── Key facts list (extractive, with source locations)
           ├── Temporal timeline (ordered events with dates)
           ├── Entity profiles (per-entity summary across this + related docs)
           ├── Anomaly flags (outliers, contradictions, missing data)
           └── Collection-level insights update (aggregates across profile)
```

### 7.4 Pre-computed Knowledge Store

For each document, after PROCESSING completes, stored in MongoDB `document_intelligence`:

```json
{
  "document_id": "...",
  "profile_id": "...",
  "subscription_id": "...",

  "entities": [
    { "name": "Acme Corp", "type": "ORG", "mentions": 14, "first_seen": "page 1", "confidence": 0.95 }
  ],
  "relationships": [
    { "subject": "Acme Corp", "predicate": "employs", "object": "John Smith", "evidence": "chunk_id_123" }
  ],

  "temporal": {
    "date_range": { "earliest": "2024-01-01", "latest": "2025-06-30" },
    "periods": ["Q1 2024", "Q2 2024"],
    "events_timeline": [
      { "date": "2024-03-15", "event": "Contract signed", "source": "page 3, para 2" }
    ],
    "gaps": [
      { "from": "2024-06-01", "to": "2024-09-01", "note": "No data for Q3 2024" }
    ]
  },

  "numerical": {
    "key_figures": [
      { "label": "Total Revenue", "value": 1500000, "currency": "USD", "period": "FY2024" }
    ],
    "trends": [
      { "metric": "revenue", "direction": "increasing", "rate": "+20%/quarter" }
    ],
    "anomalies": [
      { "metric": "expense_ratio", "value": 0.92, "expected_range": [0.3, 0.7], "severity": "high" }
    ]
  },

  "cross_doc": {
    "continues": ["doc_id_prev_quarter"],
    "contradicts": [{ "doc_id": "doc_xyz", "field": "salary", "this_value": "80K", "other_value": "85K" }],
    "supersedes": [],
    "related_by_entity": ["doc_id_1", "doc_id_2"]
  },

  "summary": "This document is a Q2 2024 financial report for Acme Corp...",
  "key_facts": [
    { "fact": "Revenue grew 20% QoQ to $1.5M", "source": "page 2, table 1", "confidence": 0.98 }
  ],

  "processed_at": "2026-04-06T12:00:00Z",
  "processing_duration_seconds": 45,
  "model_version": "docwain-v2-active"
}
```

Neo4j stores: entity nodes, relationship edges, temporal edges, cross-document edges.
Qdrant stores: child chunks with parent text payload, entity tags, document-level embeddings.

### 7.5 KG Loading at Server Startup

```
Server starts
    │
    ▼
1. Initialize Neo4j connection pool
    │
    ▼
2. Load KG index into memory cache:
    ├── Entity catalog: all entity names + types + document counts
    ├── Relationship schema: known predicate types + frequency
    └── Collection-level stats: entities per profile, temporal range per profile
    │
    ▼
3. Warm Qdrant collection caches:
    ├── List all collections, verify health
    └── Pre-fetch collection metadata (point counts, vector dimensions)
    │
    ▼
4. Health gate: if Neo4j unreachable, log warning + enable circuit breaker
```

Cache refreshes every 15 minutes via background task. New documents processed between refreshes handled via ingest queue (writes to Neo4j directly).

### 7.6 Query-Time Behavior (Lightweight Assembly Only)

With all knowledge pre-computed, query time becomes:

```
User query arrives
    ▼
Query Classifier (rules, <50ms)
    ▼
Embed query → Search Qdrant (child chunks)
    ▼
Expand to parent chunks + fetch document_intelligence from MongoDB
    ▼
KG lookup: traverse entity neighbors (from in-memory catalog, 1 Neo4j hop)
    ▼
Assemble context:
    ├── Retrieved chunks (with parent context)
    ├── Pre-computed temporal analysis (if query involves time)
    ├── Pre-computed numerical analysis (if query involves numbers)
    ├── Pre-computed cross-doc links (if multi-doc query)
    ├── Pre-computed key facts from related documents
    └── KG entity context (relationships, co-occurrences)
    ▼
Reason + Stream response (single LLM call with rich context)
```

What query time does NOT do:
- No entity extraction (pre-computed)
- No temporal analysis (pre-computed)
- No cross-document comparison (pre-computed)
- No KG ingestion (done during processing)
- No embedding generation (done during processing)
- No document summarization (pre-computed)

---

## 8. Pre-computed Knowledge as Context Enrichment

### 8.1 Enrichment Injection into Reasoner

Pre-computed knowledge is context enrichment, NOT the answer source. The Reasoner reads actual source chunks and generates a fresh response grounded in evidence.

```
REASONER PROMPT STRUCTURE:

PRIMARY (what the answer is grounded in):
    ├── Retrieved source chunks (verbatim from docs)
    ├── Parent chunk context (expanded boundaries)
    └── KG entity relationships (evidence-linked)

ENRICHMENT (additional signal, NOT answer source):
    ├── Temporal context
    ├── Numerical context
    ├── Cross-doc context
    └── Anomaly flags

INSTRUCTION:
    "Generate your response from the source chunks.
     Use enrichment context to inform your reasoning
     but cite only primary sources. If enrichment
     contradicts sources, trust the sources."
```

### 8.2 Enrichment Rules

| Rule | Why |
|------|-----|
| Enrichment is always labeled separately in the prompt | LLM knows primary vs supplementary |
| Enrichment never appears in citations | User sees source documents, not computed metadata |
| If enrichment contradicts source chunks, source wins | Pre-computed data could be stale |
| Enrichment is optional — query works without it | Graceful degradation |
| Enrichment is scoped to query intent | Temporal only for time queries, numerical only for number queries |

---

## 9. Document Intelligence — Accurate Extraction with Holistic Context

### 9.1 Multi-Layer Extraction Architecture

```
LAYER 1: CONTENT FIDELITY
    Goal: Extract EXACTLY what's in the document
    ├── Text extraction (OCR / native parser)
    ├── Table extraction (structure-preserving: rows, columns, headers, merged cells, typed values)
    ├── Form field extraction (key-value pairs)
    ├── Image/chart description (vision model)
    └── Metadata (author, dates, page count)
    Validation: round-trip check — can we reconstruct the document's information?

LAYER 2: STRUCTURAL UNDERSTANDING
    Goal: Understand HOW the document is organized
    ├── Section hierarchy (H1 → H2 → H3 → body)
    ├── Logical flow (intro → analysis → conclusion)
    ├── Table-to-text links ("As shown in Table 3")
    ├── Cross-references ("See Section 4.2")
    ├── Footnote resolution
    └── Page-spanning content stitching
    Output: document_structure tree for parent-child chunking

LAYER 3: SEMANTIC INTELLIGENCE
    Goal: Understand WHAT the document means
    ├── Entity extraction (NER + domain-specific)
    ├── Relationship extraction (who did what to whom, when, where)
    ├── Temporal analysis (date ranges, event sequencing, duration, gaps)
    ├── Numerical analysis (key figures, aggregates, anomaly detection)
    ├── Sentiment/tone (where relevant)
    └── Document intent classification (report, contract, invoice, letter)

LAYER 4: HOLISTIC CONTEXT
    Goal: Understand how this document fits in the broader knowledge base
    ├── Cross-document entity resolution ("John Smith" = "J. Smith")
    ├── Temporal continuity detection (Doc A=Q1, Doc B=Q2 → series)
    ├── Value change detection (same field, different values → classify as update/correction/conflict)
    ├── Coverage analysis (topics/periods covered vs gaps)
    ├── Supersession detection (newer doc replaces older)
    └── Collection-level insight refresh
```

### 9.2 Extraction Validation Gate

| Layer | Validation | Fail Action |
|-------|-----------|-------------|
| Content Fidelity | Character coverage >95%, tables consistent, numbers parseable | Re-extract with alternate strategy |
| Structural | Cross-references resolve, no orphaned footnotes, section tree acyclic | Flag gaps, proceed with partial |
| Semantic | Entity confidence >0.7, relationships have evidence, temporal ranges valid | Low-confidence flagged, surfaces in HITL review |
| Holistic | Entity resolution confidence >0.8, value changes have both source refs | Low-confidence cross-doc links marked tentative |

Validation results from Layers 1-2 (Content Fidelity, Structural) surface at HITL Gate 1 (post-extraction review) — reviewer sees extraction quality and can request re-extraction. Validation results from Layers 3-4 (Semantic, Holistic) surface at HITL Gate 2 (post-screening review) — reviewer sees entity confidence, cross-doc links, and flagged issues before approving processing.

### 9.3 CSV/Excel Special Path

Structured data uses a completely different extraction path:

```
CSV / Excel detected
    │
    ▼
Native parser (pandas / openpyxl) — NOT OCR
    │
    ▼
Schema detection:
    ├── Column types (string, numeric, date, categorical)
    ├── Header row identification
    ├── Sheet enumeration (Excel)
    └── Row count, null distribution
    │
    ▼
Intelligent sampling (if >10K rows):
    ├── Statistical profile per column (min, max, mean, stddev, distribution)
    ├── Sample representative rows (head + tail + random + outliers)
    ├── Group-by aggregates on categorical columns
    └── Time-series detection on date columns
    │
    ▼
Store: full data in Azure Blob, statistical profile + sample in MongoDB,
       profile embeddings in Qdrant (NOT raw CSV text)
```

Key principle: Don't embed raw structured data. Embed the statistical profile and representative samples.

---

## 10. Analytical Intelligence — Insight Generation

### 10.1 Capability Levels

```
LEVEL 1 — Retrieval:    "What does the document say?"
LEVEL 2 — Synthesis:    "What does this mean across documents?"
LEVEL 3 — Analysis:     "What patterns, risks, and opportunities exist?"
LEVEL 4 — Advisory:     "What should you do about it?"
```

DocWain currently operates at Levels 1-2. This section adds Levels 3-4.

### 10.2 Analytical Reasoner

When query intent is ANALYTICAL, the Reasoner uses a specialized prompt:

```
REASONER PROMPT (ANALYTICAL MODE):

SOURCE DATA:
    [retrieved chunks with actual numbers]

COMPUTED CONTEXT:
    [trends, anomalies, cross-doc changes from pre-computed intelligence]

TASK: Analyze this data and provide:
    1. Key observations (grounded in data)
    2. Pattern analysis (what the data shows)
    3. Risk identification (anomalies, concerning trends)
    4. Actionable recommendations

RULES:
    - Every observation must cite source data
    - Recommendations must follow from evidence
    - Clearly separate facts from analysis
    - Flag uncertainty ("data suggests" vs "data confirms")
```

### 10.3 Domain-Aware Analysis Templates

Analysis adapts to document domain (detected during extraction):

**Financial:**
- Trend analysis: direction, rate, inflection points
- Ratio analysis: key ratios vs benchmarks/history
- Variance analysis: actual vs expected/budget/prior period
- Risk flags: anomalies warranting attention
- Improvement areas: data-backed suggestions with options

**Healthcare:**
- Protocol comparison across documents
- Outcome tracking over time periods
- Compliance gaps vs guideline documents
- Risk indicators for adverse outcomes

**Legal:**
- Clause comparison across contract versions
- Obligation tracking: deadlines, deliverables, penalties
- Risk exposure: unfavorable terms, missing protections
- Precedent matching from other documents

**HR:**
- Compensation benchmarking across documents/periods
- Skill gap analysis across candidate pool
- Attrition patterns by tenure, role, department
- Policy compliance checks

### 10.4 How Pre-computed Knowledge Powers Analysis

```
Pre-computed (doc processing)          →    Used at query time
──────────────────────────────              ────────────────────
Temporal: date ranges, gaps            →    "No data for Q3 — investigate"
Numerical: revenue +20%/Q             →    "Growth decelerating: Q1 +25%, Q2 +20%, Q3 +12%"
Anomalies: expense ratio 0.92         →    "OpEx consuming 92% of revenue — norm is 60-70%"
Cross-doc: salary 80K→85K             →    "Comp increased 6.25% YoY — below market median"
Entity relationships                   →    "Key person risk: 3 projects depend on 1 person"
```

Without pre-computed knowledge: model crunches data in real-time — slow, incomplete, misses cross-doc patterns.
With pre-computed knowledge: model focuses on reasoning and recommendations.

### 10.5 Confidence and Guardrails

| Guardrail | Implementation |
|-----------|---------------|
| Fact-analysis separation | Response enforces sections: "Key Data Points" (cited) → "Analysis" (reasoned) → "Recommendations" (advisory) |
| Confidence signaling | Each recommendation tagged: HIGH (strong data), MEDIUM (data suggests), LOW (limited data) |
| Data sufficiency check | If <3 data points for a trend, state "insufficient data" rather than hallucinate a pattern |
| Domain boundary | Frames as "the data suggests" not "you should" — never gives medical/legal/financial advice |
| Citation density | Minimum 1 citation per 2 claims. Below this threshold → flag as speculative |
| Contradiction surfacing | If data contradicts itself, surface both sides rather than picking one |

---

## Appendix: Immediate Action — Stuck Extraction Fix

As of 2026-04-06, extraction is stuck on subscription `67fde0754e36c00b14cea7f5`:
- Document `df_raw (3).csv` produced 42,142,724 chars of extracted text
- Pipeline stuck after sanitization — no progress on docs 2-5
- Neo4j is down — repeated routing errors adding latency to all queries
- Duplicate extraction requests being rejected with no recovery path

Immediate fixes (Phase 0):
1. Kill or timeout the stuck extraction
2. Implement per-document timeout (5 min hard limit)
3. Add CSV/Excel native parser path (bypass OCR/deep analysis for structured files)
4. Add Neo4j circuit breaker
5. Add stale extraction lock auto-release (>15 min)
