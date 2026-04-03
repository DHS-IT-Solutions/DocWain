# DocWain V2 LLM Architecture — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement dual-model vLLM serving, Plan-Execute-Generate query pipeline, Profile Intelligence Builder, region-aware Domain Knowledge Pack system, proactive alerts, and retrieval upgrades — integrating with the existing DocWain RAG v2 pipeline.

**Architecture:** Two vLLM instances (Qwen3.5-14B fast path + Qwen3.5-27B smart path) with LLM-based intent routing. Query pipeline uses Plan→Execute→Generate+Verify pattern. Profile Intelligence Builder pre-computes structured knowledge during document processing. Knowledge Packs provide region-aware authoritative domain content.

**Tech Stack:** vLLM (FP8 + EAGLE3 + prefix caching), FastAPI, Qdrant, Neo4j, MongoDB, BGE-M3, Recharts/Nivo

**Spec:** `docs/superpowers/specs/2026-04-03-docwain-v2-llm-architecture-design.md`

---

## Phase 1: Serving Layer + Query Pipeline Core

### Task 1: vLLM Manager — Dual Instance Configuration

**Files:**
- Create: `src/serving/__init__.py`
- Create: `src/serving/config.py`
- Create: `src/serving/vllm_manager.py`

- [ ] **Step 1: Create serving config**

```python
# src/serving/config.py
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class VLLMInstanceConfig:
    model: str
    instance_name: str
    port: int
    dtype: str = "fp8"
    kv_cache_dtype: str = "fp8"
    max_model_len: int = 16384
    gpu_memory_utilization: float = 0.45
    enable_prefix_caching: bool = True
    enable_chunked_prefill: bool = False
    speculative_model: Optional[str] = None
    guided_decoding_backend: str = "xgrammar"
    tensor_parallel_size: int = 1

FAST_PATH_CONFIG = VLLMInstanceConfig(
    model="unsloth/Qwen3-14B",  # or Qwen3.5-14B when available
    instance_name="docwain-fast",
    port=8100,
    gpu_memory_utilization=0.25,
    max_model_len=8192,
    speculative_model="RedHatAI/Qwen3-14B-speculator.eagle3",
)

SMART_PATH_CONFIG = VLLMInstanceConfig(
    model="unsloth/Qwen3.5-27B",
    instance_name="docwain-smart",
    port=8200,
    gpu_memory_utilization=0.50,
    max_model_len=32768,
    enable_chunked_prefill=True,
    speculative_model=None,  # add when EAGLE3 head available for 27B
)
```

- [ ] **Step 2: Create vLLM manager**

Manages start/stop/health of both vLLM instances via subprocess. Provides `query_fast()` and `query_smart()` convenience methods using the OpenAI-compatible API.

- [ ] **Step 3: Commit**

```bash
git add -f src/serving/
git commit -m "feat: add vLLM dual-instance serving layer config and manager"
```

---

### Task 2: Intent Router — LLM-Based Query Classification

**Files:**
- Create: `src/serving/model_router.py`

- [ ] **Step 1: Implement intent router**

Uses the 14B (fast path) to classify every incoming query into intent + complexity + route. Returns structured JSON via guided_json. If 14B is unavailable, falls back to keyword-based heuristics.

Key function: `route_query(query, profile_context) -> RouterResult`

RouterResult contains: intent, complexity, route (fast|smart), requires_kg, requires_visualization.

- [ ] **Step 2: Commit**

```bash
git add -f src/serving/model_router.py
git commit -m "feat: add LLM-based intent router for fast/smart path classification"
```

---

### Task 3: Fast Path Handler

**Files:**
- Create: `src/serving/fast_path.py`

- [ ] **Step 1: Implement fast path**

Handles simple queries (greetings, identity, single-fact lookups) using the 14B model directly. For lookups that need retrieval, uses a simplified single-step retrieval (Qdrant search + direct generation, no planning/verification).

Key function: `handle_fast_path(query, router_result, profile_context) -> ResponsePayload`

- [ ] **Step 2: Commit**

```bash
git add -f src/serving/fast_path.py
git commit -m "feat: add fast path handler for simple queries via 14B"
```

---

### Task 4: Query Planner — Phase 1 (27B Plan Generation)

**Files:**
- Create: `src/query/__init__.py`
- Create: `src/query/planner.py`

- [ ] **Step 1: Implement query planner**

Sends query + profile_context to 27B with guided_json to produce a structured execution plan. The plan schema defines available actions (search, knowledge_search, kg_lookup, cross_reference, spreadsheet_query, generate) with dependencies between steps.

Key function: `plan_query(query, profile_context, router_result) -> QueryPlan`

QueryPlan is a dataclass with: intent, complexity, steps (list of PlanStep), requires_kg, requires_visualization, domain_pack, region.

- [ ] **Step 2: Commit**

```bash
git add -f src/query/
git commit -m "feat: add Phase 1 query planner — 27B generates execution plans"
```

---

### Task 5: Pipeline Executor — Phase 2 (Deterministic Execution)

**Files:**
- Create: `src/query/executor.py`

- [ ] **Step 1: Implement pipeline executor**

Executes plan steps with parallel execution of independent steps (using asyncio or ThreadPoolExecutor). Maps each action to the appropriate backend:
- `search` → existing Qdrant retriever (`src/ask/retriever.py`)
- `kg_lookup` → existing Neo4j queries (`src/kg/`)
- `cross_reference` → retrieve + compare logic
- `spreadsheet_query` → structured data lookup from MongoDB computed profiles

Key function: `execute_plan(plan, clients) -> ExecutionResult`

ExecutionResult contains per-step results and assembled context.

- [ ] **Step 2: Commit**

```bash
git add -f src/query/executor.py
git commit -m "feat: add Phase 2 pipeline executor — parallel deterministic plan execution"
```

---

### Task 6: Context Assembler

**Files:**
- Create: `src/query/context_assembler.py`

- [ ] **Step 1: Implement context assembler**

Takes ExecutionResult and builds the structured context blocks for Phase 3:
- `<profile_context>` from pre-computed intelligence
- `<retrieved_evidence>` from search results
- `<kg_context>` from KG lookups
- `<knowledge_pack_context>` from domain knowledge searches
- `<cross_reference_results>` from cross-referencing
- `<spreadsheet_data>` from spreadsheet queries

Key function: `assemble_context(execution_result, profile_intelligence) -> str`

- [ ] **Step 2: Commit**

```bash
git add -f src/query/context_assembler.py
git commit -m "feat: add context assembler for Phase 3 structured input"
```

---

### Task 7: Response Generator + Verifier — Phase 3

**Files:**
- Create: `src/query/generator.py`
- Create: `src/query/confidence.py`

- [ ] **Step 1: Implement response generator**

Sends assembled context + query to 27B for final response generation. Parses output into three sections: `<response>`, `<chart_spec>`, `<alerts>`. Uses guided_json for chart_spec and alerts when present.

Key function: `generate_response(query, context, router_result) -> GeneratedResponse`

GeneratedResponse contains: response_text, chart_spec (optional dict), alerts (optional list), confidence, sources.

- [ ] **Step 2: Implement confidence verifier**

Self-verification: checks if generated response has sufficient evidence grounding. If confidence < 0.7 and re-retrieves < 2, returns a refined query for re-execution.

Key function: `verify_confidence(response, context) -> VerificationResult`

VerificationResult: passed (bool), confidence (float), refined_query (optional str).

- [ ] **Step 3: Commit**

```bash
git add -f src/query/generator.py src/query/confidence.py
git commit -m "feat: add Phase 3 response generator with self-verification loop"
```

---

### Task 8: Unified Query Pipeline — Orchestrator

**Files:**
- Create: `src/query/pipeline.py`

- [ ] **Step 1: Implement the orchestrator**

Ties together all components: Router → (Fast Path | Plan → Execute → Generate+Verify). Handles the re-retrieval loop (max 2). Returns the final ResponsePayload compatible with existing API response format.

Key function: `run_query_pipeline(query, profile_id, subscription_id, clients) -> ResponsePayload`

- [ ] **Step 2: Commit**

```bash
git add -f src/query/pipeline.py
git commit -m "feat: add unified query pipeline orchestrator — route, plan, execute, generate"
```

---

### Task 9: API Integration — Wire Into Existing /ask Endpoint

**Files:**
- Modify: `src/api/dw_newron.py`
- Modify: `src/main.py`

- [ ] **Step 1: Add V2 pipeline flag and integration**

Add `Config.QUERY_V2.ENABLED` flag. When enabled, route through `src/query/pipeline.py` instead of the existing RAG v2 pipeline. Extend `AnswerPayload` to include `chart_spec` and `alerts` fields. Existing pipeline remains as fallback.

- [ ] **Step 2: Commit**

```bash
git add -f src/api/dw_newron.py src/main.py
git commit -m "feat: integrate V2 query pipeline into /ask endpoint with feature flag"
```

---

## Phase 2: Intelligence Layer

### Task 10: Profile Intelligence Builder

**Files:**
- Create: `src/intelligence/__init__.py`
- Create: `src/intelligence/profile_builder.py`
- Create: `src/intelligence/computed_profiles.py`
- Create: `src/intelligence/collection_insights.py`

- [ ] **Step 1: Implement profile builder**

Runs after document embedding. Queries KG for all entities/relationships in a profile, computes structured profiles per entity, generates collection-level insights (distributions, patterns, gaps), stores results in MongoDB.

Domain-specific logic for: HR (candidate profiles + role-fit), Finance (vendor analysis + spend patterns), Legal (obligation tracking + risk scoring), Logistics (inventory + supplier dependency), Medical (patient summaries + interaction checks).

- [ ] **Step 2: Hook into document processing pipeline**

Add post-embed hook in the extraction/embedding pipeline that triggers Profile Intelligence Builder for the affected profile.

- [ ] **Step 3: Commit**

```bash
git add -f src/intelligence/
git commit -m "feat: add Profile Intelligence Builder — pre-computed domain profiles and insights"
```

---

### Task 11: Alert Generator

**Files:**
- Create: `src/intelligence/alert_generator.py`

- [ ] **Step 1: Implement alert detection and formatting**

Parses `<alerts>` from model response. Also provides `generate_alerts_from_context()` for the scheduled analysis agent — scans computed profiles for threshold breaches, upcoming deadlines, gaps. Domain-specific alert rules.

- [ ] **Step 2: Commit**

```bash
git add -f src/intelligence/alert_generator.py
git commit -m "feat: add alert generator — in-response and scheduled alerting"
```

---

### Task 12: Scheduled Analysis Agent

**Files:**
- Create: `src/agents/__init__.py`
- Create: `src/agents/scheduled_analysis.py`
- Create: `src/agents/alert_digest.py`

- [ ] **Step 1: Implement scheduled analysis**

Background agent (Celery task or cron job) that runs per-profile on a configurable schedule. Sends meta-query through the smart path pipeline, collects alerts, stores digest in MongoDB, triggers notification delivery.

- [ ] **Step 2: Commit**

```bash
git add -f src/agents/
git commit -m "feat: add scheduled analysis agent and alert digest delivery"
```

---

## Phase 3: Domain Knowledge Packs

### Task 13: Knowledge Pack Framework

**Files:**
- Create: `src/knowledge_packs/__init__.py`
- Create: `src/knowledge_packs/base.py`
- Create: `src/knowledge_packs/registry.py`
- Create: `src/knowledge_packs/updater.py`

- [ ] **Step 1: Implement pack interfaces and registry**

`KnowledgePackScraper` and `KnowledgePackParser` abstract base classes. `PackRegistry` maps (domain, region) → pack implementation. `PackUpdater` handles monthly refresh jobs.

- [ ] **Step 2: Commit**

```bash
git add -f src/knowledge_packs/
git commit -m "feat: add knowledge pack framework — pluggable region-aware domain knowledge"
```

---

### Task 14: NICE Clinical Guidelines Pack

**Files:**
- Create: `src/knowledge_packs/packs/clinical/__init__.py`
- Create: `src/knowledge_packs/packs/clinical/nice_scraper.py`
- Create: `src/knowledge_packs/packs/clinical/nice_parser.py`
- Create: `src/knowledge_packs/packs/clinical/config.yaml`

- [ ] **Step 1: Implement NICE scraper and parser**

Scrapes published guidance from nice.org.uk. Parses into structured sections (recommendations, evidence, quality statements). Indexes into `knowledge_nice` Qdrant collection.

- [ ] **Step 2: Commit**

```bash
git add -f src/knowledge_packs/packs/
git commit -m "feat: add NICE clinical guidelines knowledge pack — UK medical domain"
```

---

## Phase 4: Retrieval Upgrades

### Task 15: BGE-M3 Hybrid Retriever

**Files:**
- Create: `src/retrieval/bgem3_retriever.py`

- [ ] **Step 1: Implement BGE-M3 hybrid retrieval**

Replaces BGE-large with BGE-M3 for combined dense + sparse retrieval in a single model. Returns both dense vectors and sparse token weights for Qdrant hybrid search.

- [ ] **Step 2: Commit**

```bash
git add -f src/retrieval/bgem3_retriever.py
git commit -m "feat: add BGE-M3 hybrid dense+sparse retriever"
```

---

### Task 16: CRAG — Corrective RAG Evaluator

**Files:**
- Create: `src/retrieval/crag_evaluator.py`

- [ ] **Step 1: Implement retrieval quality evaluator**

Scores retrieved chunks for relevance before they're assembled into context. Low-quality chunks discarded. If overall quality is below threshold, triggers re-retrieval with refined query.

- [ ] **Step 2: Commit**

```bash
git add -f src/retrieval/crag_evaluator.py
git commit -m "feat: add CRAG evaluator — corrective retrieval with quality scoring"
```

---

## Phase 5: Integration + Live Testing

### Task 17: End-to-End Integration

**Files:**
- Modify: `src/query/pipeline.py`
- Create: `scripts/test_e2e.py`

- [ ] **Step 1: Wire all components together**

Connect query pipeline with Profile Intelligence, Knowledge Packs, Alert system, and CRAG retrieval. Ensure the full flow works: /ask → route → plan → execute (with knowledge packs + KG + CRAG) → generate (with charts + alerts) → deliver.

- [ ] **Step 2: Create E2E test script**

20 test scenarios covering all capabilities:
- Greetings and identity (fast path)
- Simple document lookups (fast path)
- Multi-document analysis (smart path)
- Excel/CSV queries (smart path + spreadsheet)
- KG-dependent queries (smart path + KG)
- Chart generation requests (smart path + chart_spec)
- Medical queries with NICE guidelines (knowledge pack)
- Alert-triggering scenarios (logistics, HR, finance)
- Cross-document contradiction detection
- Candidate ranking (HR profile intelligence)

- [ ] **Step 3: Run tests and fix issues iteratively**

- [ ] **Step 4: Commit**

```bash
git add -f scripts/test_e2e.py src/query/pipeline.py
git commit -m "feat: end-to-end integration and test suite for V2 architecture"
```

---

### Task 18: Start DocWain Server + Live User Testing

- [ ] **Step 1: Start the DocWain server with V2 pipeline enabled**

```bash
export QUERY_V2_ENABLED=true
python -m src.main
```

- [ ] **Step 2: Run live user-based tests**

Upload test documents (resumes, invoices, contracts, spreadsheets, medical records) through the UI. Test every capability end-to-end via the actual API.

- [ ] **Step 3: Document all issues, fix, and re-test**

- [ ] **Step 4: Generate final test report**

---
