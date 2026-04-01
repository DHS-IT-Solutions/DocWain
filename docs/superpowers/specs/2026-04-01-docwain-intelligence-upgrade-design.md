# DocWain Intelligence Upgrade — Design Specification

**Date:** 2026-04-01
**Status:** Approved
**Hardware:** NVIDIA A100-SXM4-80GB (80GB VRAM, CUDA 12.4)
**Base Model:** Vision-grafted Qwen3-14B (SigLIP-SO400M + ProjectionMLP + Qwen3-14B)

## Overview

Two-pillar upgrade to make DocWain a highly intelligent enterprise document platform:

1. **Model Intelligence** — Complete V2 4-phase training with intelligence-first enhancements (CoT, curriculum learning, DPO, self-verification, insight generation), then retarget daily fine-tune loop to the resulting 14B model.
2. **Document Processing** — Fundamental pipeline upgrade across extraction (4-engine ensemble + intelligent merger), KG (LLM-driven entity/relationship extraction with ontology), embeddings (hybrid 3-signal retrieval), and visualization (model-native chart generation with insights).

**Core constraint:** Fine-tuning trains on metadata, patterns, and Q&A pairs ONLY. Never on raw document content. Document data lives in the retrieval layer (embeddings, KG).

---

## Pillar 1: V2 Model Training Pipeline

### Phase 1: Vision-Language Alignment (Exists — No Changes)

- **Status:** Implemented in `src/finetune/v2/train_phase1.py`
- Trains ProjectionMLP only (vision & text frozen)
- Input: Image-caption pairs (LLaVA-Pretrain style)
- Loss: Cosine similarity alignment
- QA gates: cosine_sim >= 0.60, caption_bleu >= 0.15

### Phase 2: Document Intelligence SFT (Implement — Currently Stubbed)

**File:** `src/finetune/v2/train_phase2.py`

**Engine:** Unsloth + TRL SFTTrainer on A100 80GB, full-precision (no quantization).

**Trainable parameters:**
- ProjectionMLP (unfrozen from Phase 1)
- LoRA adapters on Qwen3-14B text layers (r=64, alpha=128, target: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)
- SigLIP vision encoder: frozen

**Intelligence enhancements:**

1. **Chain-of-Thought training:** Every training example includes a `<think>...</think>` block before the response. The model learns to reason before answering — self-correcting during inference.

2. **Curriculum learning — 4 stages:**
   - Stage 1 (epochs 1-2): Clean, well-formatted documents — baseline competence
   - Stage 2 (epochs 3-4): Noisy scans, skewed images, faded text — robustness
   - Stage 3 (epochs 5-6): Complex layouts (nested tables, multi-column, mixed languages) — expertise
   - Stage 4 (epochs 7-8): Adversarial cases (contradictory docs, missing pages, ambiguous fields) — resilience

3. **Dataset mix (weighted sampling):**
   - Table understanding: 40% (DocVQA table subsets, PubTables-1M, SynthTabNet)
   - Layout comprehension: 25% (PubLayNet, DocLayNet, DUDE)
   - OCR correction: 20% (SynthDoc with injected OCR errors + corrections)
   - Cross-document reasoning: 15% (multi-doc QA pairs from Natural Questions, HotpotQA adapted to doc format)

4. **Training config:**
   - Batch size: 4 (gradient accumulation: 8, effective batch: 32)
   - Learning rate: 2e-5 with cosine scheduler, warmup 10%
   - Max sequence length: 4096 tokens
   - Mixed precision: bf16
   - Checkpointing: every 500 steps + end of each curriculum stage

**QA gates:** docvqa_accuracy >= 0.75, table_f1 >= 0.80, layout_map >= 0.70

### Phase 2.5: Contrastive Preference Training — DPO (New)

**File:** `src/finetune/v2/train_phase2_5_dpo.py`

**Engine:** TRL DPOTrainer on A100 80GB.

**Purpose:** Teach the model what bad extraction looks like, not just what good looks like. Dramatically reduces hallucination.

**Trainable:** LoRA adapters (same as Phase 2, loaded from Phase 2 checkpoint)

**Data construction:**
- For each training document, generate 2 extractions:
  - **Chosen:** Complete, structured, accurate extraction with correct `<think>` reasoning
  - **Rejected:** Extraction with realistic errors — missing fields, hallucinated values, broken table structure, incorrect reasoning
- Sources: Programmatically corrupted versions of Phase 2 ground truth + V2 model's own mistakes (rejection sampling: run Phase 2 model, collect outputs below quality threshold)

**Training config:**
- Beta: 0.1 (standard DPO temperature)
- Batch size: 2 (gradient accumulation: 16, effective batch: 32)
- Learning rate: 5e-6 (lower than SFT — fine adjustment)
- Epochs: 3
- Max prompt length: 2048, max response length: 2048

**QA gates:** hallucination_rate <= 0.05 (on held-out validation set), extraction_f1 improvement >= 5% over Phase 2 checkpoint

### Phase 3: Tool-Calling SFT (Implement — Currently Stubbed)

**File:** `src/finetune/v2/train_phase3.py`

**Engine:** Unsloth + TRL SFTTrainer on A100 80GB.

**Trainable:** LoRA adapters only (projection frozen from Phase 2.5).

**9 core tools** (schemas in `src/finetune/v2/tool_schemas.py`):
1. `ocr_extract` — Extract text from image region
2. `layout_extract` — Parse document layout structure
3. `extract_table` — Extract and structure table data
4. `extract_entities` — Named entity extraction with typing
5. `context_understand` — Deep contextual analysis of a passage
6. `cross_reference` — Cross-document reference resolution
7. `search_documents` — Semantic search across document repository
8. `summarize_section` — Section-level summarization
9. `visualize_data` — Generate visualization directive from data

**Intelligence enhancements:**

1. **Self-verification training:** Model learns to chain tool calls — after extraction, it re-invokes `cross_reference` on its own output vs the source chunk, compares, and self-corrects. No new tool needed; this is a learned multi-step tool-use pattern.

2. **Confidence calibration:** Every tool call output includes calibrated confidence with reasoning:
   - "High confidence: value appears in both header and line items"
   - "Low confidence: OCR quality poor in this region, two possible readings"

3. **Visualization-aware responses:** Model learns to emit `<viz>` directives when data is chart-worthy:
   ```
   <viz type="bar" title="Revenue by Quarter" x="[Q1,Q2,Q3,Q4]" y="[120,145,132,178]" unit="$K" />
   ```

**Dataset mix:**
- Synthetic traces from `tool_data_generator.py`: 40%
- ToolBench (adapted to DocWain tool schemas): 25%
- Gorilla (function-calling benchmarks): 20%
- NexusRaven (tool selection reasoning): 15%

**Training config:**
- Batch size: 4 (gradient accumulation: 8)
- Learning rate: 1e-5
- Max sequence length: 4096
- Epochs: 5

**QA gates:** tool_accuracy >= 0.85, arg_correctness >= 0.90, false_positive_rate <= 0.10

### Phase 3.5: Insight Generation SFT (New)

**File:** `src/finetune/v2/train_phase3_5_insights.py`

**Purpose:** Train the model to proactively surface insights, not just answer questions.

**Trainable:** LoRA adapters (loaded from Phase 3 checkpoint).

**Insight categories:**
1. **Pattern Recognition:** "Across 12 invoices, delivery is consistently 5-7 days late"
2. **Anomaly Detection:** "This contract's liability cap is 10x lower than comparable contracts"
3. **Trend Analysis:** "Employee turnover increased 40% over last 3 quarterly reports"
4. **Comparative Analysis:** "Version 2 removes the arbitration clause present in Version 1"
5. **Gap Analysis:** "This compliance report covers 8 of 12 required sections"

**Dataset construction:**
- Synthetic: Generate document sets with planted patterns/anomalies/trends, create Q&A pairs where the model identifies them
- Real-world: Curate examples from public financial reports, legal filings, compliance documents where insights are annotatable
- Each example includes `<think>` reasoning explaining how the insight was derived

**Training config:**
- Batch size: 4 (gradient accumulation: 8)
- Learning rate: 1e-5
- Epochs: 4
- Includes visualization directives in responses where appropriate

**QA gates:** insight_precision >= 0.80 (flagged insights are actually real), insight_recall >= 0.60 (model catches most planted patterns)

### Phase 4: Merge & Promote (Exists — Minor Updates)

**File:** `src/finetune/v2/merge_promote.py`

**Updates needed:**
- Merge sequence: Phase 2 LoRA + Phase 2.5 DPO LoRA + Phase 3 LoRA + Phase 3.5 LoRA into base
- Quantize to GGUF Q4_K_M (A100 80GB can serve full precision, but GGUF for Ollama distribution)
- Regression tests expanded: V1 baseline + reasoning quality + tool accuracy + insight quality + confidence calibration
- Min 90% pass rate on regression suite
- Promote to `docwain:v2` on Ollama, then `docwain:latest` after 24h soak

### Daily Fine-Tune Loop (Post-V2)

**File:** `src/finetune/agentic_orchestrator.py` (retarget)

**Changes:**
- Base model: `docwain:v2` (14B) instead of `qwen3:8b`
- **Data policy enforcement:** Hard filter — reject any training pair containing raw document content. Allow only: metadata, extraction patterns, Q&A pairs, user feedback signals.
- **Feedback signals collected:**
  - Low-confidence queries (model uncertainty)
  - Grounding failures (hallucination corrections)
  - User corrections (explicit feedback)
  - Visualization engagement (did user interact with charts?)
  - Tool-call failures (wrong tool selected, incorrect arguments)
- **Reasoning preservation:** Every daily training batch includes `<think>` block examples — never train the model to skip reasoning
- **Calibration maintenance:** Include confidence calibration examples in every batch
- **Regression gate:** Daily model must pass V2 baseline on reasoning quality, tool accuracy, and insight precision — not just answer accuracy

---

## Pillar 2: Document Processing Pipeline

### 2A: Extraction Pipeline

#### Adaptive Document Triage (New — Pre-Extraction)

**File:** `src/extraction/triage.py`

Lightweight classifier analyzes document before extraction to route intelligently:

| Document Type | Primary Engines | Skip |
|---|---|---|
| Clean digital PDF | Structural + Semantic | Heavy OCR |
| Scanned document | Vision (GLM-OCR) + V2 + Structural | - |
| Handwritten form | V2 (vision reasoning) + OCR with preprocessing | Structural |
| Mixed (digital + scans) | Per-page routing — different engines per page | - |
| Spreadsheet/table-heavy | Structural + Semantic + V2 | - |

**Implementation:**
- Feature extraction: DPI, text-layer presence, image noise score, page count, file type
- Classification: Lightweight gradient-boosted tree (XGBoost) trained on document metadata — not a heavy model
- Output: `TriageResult` with engine weights per page, preprocessing directives, confidence
- Triage metadata propagates to extraction merger as prior confidence weights

#### Pre-Processing Intelligence (New)

**File:** `src/extraction/preprocessor.py`

Applied selectively based on triage output:

1. **Adaptive image enhancement:** Deskew (Hough transform), denoise (bilateral filter), contrast (CLAHE) — only for pages flagged as degraded
2. **Page classification:** Cover, TOC, body, appendix, form, table — via lightweight CNN or rule-based heuristics on layout features
3. **Language detection per region:** fastText langdetect on text blocks — routes to appropriate OCR language pack
4. **Resolution upscaling:** Real-ESRGAN lightweight (x2) for sub-150 DPI scan regions before OCR

#### V2 as Fourth Extraction Engine (New)

**File:** `src/extraction/v2_extractor.py`

After V2 training completes, it joins the extraction ensemble:

- **Input:** Document page image + any available text layer
- **Output:** Structured extraction with `<think>` reasoning
- **Unique capability:** Sees the document AND reasons about what it sees
  - Understands that a box in the top-right is a date field
  - Recognizes indented text as a sub-clause
  - Identifies signature blocks, stamps, watermarks
  - Detects mathematical inconsistencies in tables
- **Confidence scores:** Calibrated from Phase 3 training — the merger can trust them
- **Prompt template:** Includes document type from triage, specific extraction instructions per page type

#### Intelligent Extraction Merger (Upgrade)

**File:** `src/extraction/merger.py` (rewrite)

Current: flat confidence boost (+0.1) on cross-model agreement.
New: weighted agreement with conflict resolution.

**Weighted agreement matrix:**

| Engine | Structured Forms | Free Text | Scanned Docs | Tables | Complex Layout |
|---|---|---|---|---|---|
| LayoutLM (Structural) | 0.9 | 0.4 | 0.3 | 0.8 | 0.7 |
| Qwen3 (Semantic) | 0.5 | 0.9 | 0.5 | 0.6 | 0.6 |
| GLM-OCR (Vision) | 0.6 | 0.5 | 0.9 | 0.5 | 0.5 |
| V2 (Vision+Reasoning) | 0.8 | 0.8 | 0.8 | 0.9 | 0.9 |

**Conflict resolution:**
- When engines disagree on a field: re-extract that specific field with focused V2 prompting
- When entity types conflict: use surrounding context + KG prior knowledge to adjudicate
- When table structures differ: prefer the engine with highest table weight for this doc type

**Output:** `MergedExtractionResult` with per-field confidence, engine attribution, conflict log, quality scorecard.

#### Post-Extraction Validation (New)

**File:** `src/extraction/validator.py`

1. **Self-consistency:** Do extracted values add up? Dates chronological? Cross-references valid?
2. **Schema validation:** For known document types, validate against expected schema. Flag missing required fields.
3. **Re-extraction loop:** If validation fails for specific fields, trigger focused re-extraction with adjusted engine weights (max 2 retries).
4. **HITL queue:** Fields below confidence threshold (0.7) get flagged for human review. Stored in MongoDB with extraction context for reviewer.

### 2B: Knowledge Graph Pipeline

#### LLM-Driven Entity & Relationship Extraction (Replace Regex+spaCy)

**File:** `src/kg/llm_entity_extractor.py` (new, replaces primary path in `entity_extractor.py`)

- V2 model extracts entities AND typed relationships from document chunks
- Output schema:
  ```json
  {
    "entities": [
      {"name": "John Smith", "type": "PERSON", "aliases": ["J. Smith"], "confidence": 0.95}
    ],
    "relationships": [
      {"source": "John Smith", "target": "Acme Corp", "type": "signatory_of",
       "evidence": "signed by John Smith on behalf of Acme Corporation",
       "confidence": 0.92, "temporal_bounds": {"effective": "2025-01-15"}}
    ]
  }
  ```
- **Validation layer:** Regex + spaCy remain as cross-check. If LLM says "John Smith" is PERSON but spaCy disagrees, flag for review.
- V2's `<think>` reasoning produces auditable extraction chains.

#### Hierarchical Entity Resolution (New)

**File:** `src/kg/entity_resolver.py`

- **Alias resolution:** "John Smith", "J. Smith", "Mr. Smith (CEO)" -> single canonical entity with alias list
- **Cross-document linking:** Same entity in multiple documents gets merged node in Neo4j
- **Temporal awareness:** Relationships carry time bounds — "CEO during 2024-2025", "CFO from 2025"
- **Algorithm:** Fuzzy name matching (rapidfuzz, threshold 0.85) + type consistency + co-occurrence in same documents
- **Confidence propagation:** High-confidence mention in Document A boosts low-confidence fuzzy match in Document B

#### Domain Ontology (New)

**File:** `src/kg/ontology.py`

Typed relationship schemas per domain:

- **Legal:** party_to, signatory_of, governed_by, amends, supersedes, terminates, effective_from, expires_on
- **Financial:** invoiced_by, paid_to, line_item_of, totals_to, billed_on, due_on
- **HR:** employed_by, reports_to, holds_certification, worked_during, role_of
- **Medical:** diagnosed_with, prescribed, treated_by, allergic_to, admitted_on
- **Generic:** related_to, mentioned_in, part_of, located_at

Ontology is loaded at startup and used by V2 extraction prompts + Neo4j schema validation.

#### Incremental KG Enrichment (Upgrade)

**File:** `src/kg/ingest.py` (upgrade)

- Current: one-shot ingestion per document
- New: progressive enrichment
  - New document entity matched to existing KG entity -> merge, add new relationships, update temporal bounds
  - **Periodic cross-document inference:** After every 10 document ingestions, run inference pass:
    - Co-occurrence analysis: entities appearing in same documents but no explicit relationship -> suggest relationship
    - Transitive inference: A reports_to B, B reports_to C -> A indirectly_reports_to C
  - Inference results stored with `source: "inferred"` flag and lower confidence (0.6 base)

#### KG Quality & Completeness Scoring (New)

**File:** `src/kg/quality.py`

- **Entity completeness score:** Percentage of expected attributes populated (name-only = 20%, full profile = 100%)
- **Relationship evidence score:** Number of corroborating documents (single mention = low, 3+ docs = high)
- **Gap detection:** "We know X is CEO of Y, but no contract links them — possible missing document"
- Scores stored as Neo4j node/edge properties, accessible during retrieval for response confidence framing

### 2C: Hybrid Embedding Pipeline

#### Sparse Embeddings — SPLADE v3 (Implement Stub)

**File:** `src/embedding/sparse.py` (new)

- Model: `naver/splade-v3` (or latest available)
- Generates learned sparse representations — superior to raw BM25 for keyword matching
- Stored in Qdrant as named sparse vectors alongside dense vectors
- Catches exact terminology: contract numbers, medical codes, policy IDs, proper nouns

#### V2 Model Embeddings (New)

**File:** `src/embedding/v2_embeddings.py` (new)

- Extract hidden-state embeddings from V2's penultimate transformer layer
- Pool: mean pooling over non-padding tokens -> 5120-dim vector, projected down to 1024-dim via trained linear layer
- These embeddings capture DocWain-specific domain understanding
- Complementary to BGE: BGE handles general semantics, V2 handles domain-specific concept equivalence

#### Three-Signal Retrieval Fusion (New)

**File:** `src/retrieval/fusion.py` (new)

- **Signal 1:** BGE dense (1024-dim) — general semantic similarity
- **Signal 2:** SPLADE sparse — keyword/terminology matching
- **Signal 3:** V2 semantic (1024-dim) — domain-tuned understanding
- **Fusion:** Reciprocal Rank Fusion (RRF) with learned weights per signal
  - Default weights: BGE 0.4, SPLADE 0.3, V2 0.3
  - Weights tunable per query type (e.g., exact lookup -> boost SPLADE, conceptual -> boost V2)
- Cross-encoder reranker (existing) applied after fusion on top-K candidates

#### Context-Aware Chunking (Upgrade)

**File:** `src/embedding/chunking/semantic_chunker.py` (new)

- V2 model identifies semantic boundaries: "this paragraph concludes pricing, next starts payment terms"
- **Hierarchical chunks:** Document -> Section -> Paragraph, embedded at each level
  - Section-level: matches broad queries ("summarize the contract")
  - Paragraph-level: matches specific queries ("what is the liability cap?")
- Parent-child relationships stored in Qdrant payload metadata (`parent_chunk_id`, `level`)
- Never split mid-table, mid-list, or mid-argument

#### KG-Enriched Embedding Text (New)

**File:** `src/embedding/kg_enrichment.py` (new)

- Before embedding, prepend KG context to chunk text:
  - Raw: "The vendor delivered 500 units on March 15"
  - Enriched: "[Doc: Invoice #4521] [Vendor: Acme Corp] [Project: Alpha] The vendor delivered 500 units on March 15"
- Queries like "What did Acme Corp deliver?" match even when chunk never mentions "Acme Corp" by name
- KG context fetched from Neo4j at embedding time, cached in Redis

#### Embedding Quality Feedback Loop (New)

**File:** `src/embedding/feedback.py` (new)

- Track retrieval outcomes: correct chunk retrieved vs missed
- **Hard negative mining:** Retrieved-but-irrelevant chunks become training negatives
- **Adaptive re-embedding:** When KG enriches entity resolutions, re-embed affected chunks
- **Metrics:** MRR, recall@10, precision@10 per collection — tracked in MongoDB, surfaced in health endpoints

### 2D: Visualization & Insight Generation

#### Model-Native Visualization (Upgrade)

**File:** `src/visualization/enhancer.py` (upgrade)

- V2 emits `<viz>` directives during generation (trained in Phase 3)
- Directive schema:
  ```
  <viz type="bar|line|pie|scatter|heatmap" title="..." x="[...]" y="[...]" unit="..." annotations="[...]" />
  ```
- Response pipeline parses directives, renders via Plotly (web) or matplotlib (Teams/email)
- Auto-detection remains as fallback for non-V2 model responses
- `<viz>` directives stripped from final text response after rendering

#### Insight Engine Integration (New)

**File:** `src/visualization/insights.py` (new)

- Routes V2 model's insight outputs to appropriate visualization:
  - Pattern recognition -> summary card with supporting data table
  - Anomaly detection -> highlighted comparison chart with normal range shading
  - Trend analysis -> line chart with trend line and projection
  - Comparative analysis -> side-by-side bar chart or diff table
  - Gap analysis -> checklist visualization with coverage percentage
- Each insight carries: category, severity (info/warning/critical), evidence sources, confidence

#### Interactive Plotly (Upgrade)

- Default to Plotly HTML for web responses: pan, zoom, hover details
- Drill-down capability: click bar -> see underlying data points with source document links
- **Source linking:** Every data point carries provenance (document, page, extraction confidence)
- Teams/email: static PNG with data summary text (existing behavior)

#### Multi-Document Dashboard Responses (New)

**File:** `src/visualization/dashboard.py` (new)

- When query spans multiple documents, compose a mini-dashboard:
  - Data table + chart + timeline + risk flags as structured sections
- V2 decides dashboard composition based on data shape and query intent
- Layout: Markdown sections with embedded `<viz>` directives, rendered as cohesive response

### 2E: GPU Configuration & Memory Management

#### GPU Detection Module (Implement)

**File:** `src/utils/gpu.py` (new — currently missing)

```python
class GPUConfig:
    name: str                    # "NVIDIA A100-SXM4-80GB"
    vram_mb: int                 # 81920
    cuda_version: str            # "12.4"
    is_high_memory: bool         # True (>= 40GB)
    available: bool              # True
    use_4bit_quantization: bool  # False (A100 80GB doesn't need it)
    recommended_embedding_batch_size: int  # 256
    recommended_training_batch_size: int   # 4
    max_concurrent_models: int   # 3 (V2 + BGE + SPLADE fit in 80GB)
```

- Auto-detect via `torch.cuda.get_device_properties()`
- Tiered configs: A100-80GB (full precision), A100-40GB (selective quantization), T4-16GB (aggressive quantization), CPU (fallback)

#### VRAM Memory Manager (New)

**File:** `src/utils/vram_manager.py` (new)

- Dynamic model loading/offloading based on current task:
  - **Document processing mode:** V2 + extraction models loaded, embedding models offloaded
  - **Query answering mode:** V2 + embedding + reranker loaded, extraction models offloaded
  - **Training mode:** V2 exclusive access, everything else offloaded
- **Priority system:** Inference > Embedding > Training
- **Memory budget:** Track VRAM per model, refuse to load if would exceed 90% VRAM
- **Graceful degradation:** Under VRAM pressure, fall back to quantized inference (not OOM crash)
- Model load/unload events logged for observability

---

## Data Flow — End to End

```
Document Upload
    |
    v
[Adaptive Triage] --> engine weights, preprocessing directives
    |
    v
[Pre-Processing] --> deskew, denoise, upscale (selective)
    |
    v
[4-Engine Parallel Extraction]
    |-- LayoutLM (structural)
    |-- Qwen3-14B (semantic)
    |-- GLM-OCR (vision/OCR)
    |-- V2 Model (vision + reasoning + <think>)
    |
    v
[Intelligent Merger] --> weighted agreement, conflict resolution
    |
    v
[Post-Extraction Validation] --> self-consistency, schema check
    |                               |
    | (pass)                   (fail, retry <= 2)
    v                               |
[HITL Gate] <-----------------------+
    |
    v (approved)
[Screening] --> security plugins (mandatory), domain plugins
    |
    v
[LLM Entity + Relationship Extraction] --> V2 model
    |
    v
[Entity Resolution] --> alias merging, cross-doc linking
    |
    v
[Neo4j KG Ingest] --> typed relationships, temporal bounds, ontology
    |
    v
[KG-Enriched Chunking] --> semantic boundaries, hierarchical levels
    |
    v
[3-Signal Embedding]
    |-- BGE dense (1024-dim)
    |-- SPLADE sparse
    |-- V2 semantic (1024-dim)
    |
    v
[Qdrant Ingest] --> dense + sparse + V2 vectors, enriched payloads
```

```
User Query
    |
    v
[Intent Analysis] --> query type, domain, complexity
    |
    v
[3-Signal Retrieval] --> BGE + SPLADE + V2, RRF fusion
    |
    v
[Cross-Encoder Rerank]
    |
    v
[KG Context Augmentation] --> entity relationships, evidence scores
    |
    v
[V2 Model Reasoning] --> <think> chain, tool calls, self-verification
    |
    v
[Insight Generation] --> patterns, anomalies, trends, comparisons
    |
    v
[Visualization Directives] --> <viz> tags for chart-worthy data
    |
    v
[Response Composition] --> text + charts + source links + confidence
    |
    v
[Render] --> Plotly HTML (web) / PNG (Teams) / Markdown (API)
```

---

## Training Data Policy

**Allowed in fine-tuning:**
- Document metadata (titles, types, dates, categories)
- Extraction patterns (field schemas, layout patterns, entity type distributions)
- Question-answer pairs (user queries + validated responses)
- User feedback signals (corrections, low-confidence flags, grounding failures)
- Tool-call traces (correct tool selection + arguments)
- Insight examples (pattern/anomaly/trend examples with reasoning)

**Never allowed in fine-tuning:**
- Raw document text content
- Document embeddings or vectors
- PII or sensitive fields (even from metadata)
- Unvalidated/unreviewed extraction outputs

---

## Key Files Summary

### New Files
| File | Purpose |
|---|---|
| `src/finetune/v2/train_phase2_5_dpo.py` | DPO contrastive preference training |
| `src/finetune/v2/train_phase3_5_insights.py` | Insight generation SFT |
| `src/extraction/triage.py` | Adaptive document triage |
| `src/extraction/preprocessor.py` | Pre-processing intelligence |
| `src/extraction/v2_extractor.py` | V2 as fourth extraction engine |
| `src/extraction/validator.py` | Post-extraction validation |
| `src/kg/llm_entity_extractor.py` | LLM-driven entity/relationship extraction |
| `src/kg/entity_resolver.py` | Hierarchical entity resolution |
| `src/kg/ontology.py` | Domain relationship ontology |
| `src/kg/quality.py` | KG quality & completeness scoring |
| `src/embedding/sparse.py` | SPLADE sparse embeddings |
| `src/embedding/v2_embeddings.py` | V2 model semantic embeddings |
| `src/embedding/chunking/semantic_chunker.py` | Context-aware semantic chunking |
| `src/embedding/kg_enrichment.py` | KG-enriched embedding text |
| `src/embedding/feedback.py` | Embedding quality feedback loop |
| `src/retrieval/fusion.py` | Three-signal retrieval fusion (RRF) |
| `src/visualization/insights.py` | Insight engine integration |
| `src/visualization/dashboard.py` | Multi-document dashboard responses |
| `src/utils/gpu.py` | GPU detection & configuration |
| `src/utils/vram_manager.py` | VRAM memory manager |

### Modified Files
| File | Change |
|---|---|
| `src/finetune/v2/train_phase2.py` | Implement training loop with CoT + curriculum |
| `src/finetune/v2/train_phase3.py` | Implement training loop with self-verification + confidence + viz |
| `src/finetune/v2/pipeline.py` | Add Phase 2.5 and 3.5 to pipeline orchestration |
| `src/finetune/v2/merge_promote.py` | Update merge sequence for 4 LoRA adapters |
| `src/finetune/agentic_orchestrator.py` | Retarget to V2 14B, enforce data policy |
| `src/extraction/merger.py` | Rewrite with weighted agreement + conflict resolution |
| `src/extraction/engine.py` | Add V2 extractor to ensemble, integrate triage |
| `src/kg/ingest.py` | Incremental enrichment, cross-doc inference |
| `src/kg/entity_extractor.py` | LLM-primary with regex+spaCy validation |
| `src/embedding/pipeline/qdrant_ingestion.py` | Add sparse + V2 vectors |
| `src/embedding/enhanced_embedding.py` | Integrate KG enrichment |
| `src/embedding/orchestrator.py` | Add V2 embeddings, feedback loop |
| `src/embedding/chunking/` | Add semantic chunker option |
| `src/visualization/enhancer.py` | Parse V2 `<viz>` directives, Plotly upgrade |
| `src/retrieval/` | Integrate 3-signal fusion |

---

## Success Criteria

| Metric | Target | Current Baseline |
|---|---|---|
| Document extraction F1 | >= 0.90 | ~0.70 (estimated) |
| Table extraction F1 | >= 0.85 | ~0.65 (estimated) |
| Hallucination rate | <= 0.05 | ~0.15 (estimated) |
| Retrieval MRR@10 | >= 0.80 | ~0.60 (estimated) |
| KG entity completeness | >= 0.75 | ~0.40 (estimated) |
| Insight precision | >= 0.80 | N/A (new capability) |
| Tool-call accuracy | >= 0.85 | N/A (new capability) |
| V2 regression vs V1 | >= 90% pass | N/A |
| Confidence calibration | ECE <= 0.10 | N/A (new capability) |
