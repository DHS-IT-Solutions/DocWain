# DocWain V2 LLM Architecture — Design Specification

## Overview

Complete architecture redesign of DocWain's LLM serving, query pipeline, and knowledge systems. Dual-model serving via vLLM (Qwen3.5-14B fast path + Qwen3.5-27B smart path), two-phase query pipeline (Plan then Execute), profile intelligence builder, region-aware domain knowledge packs, and proactive alerting.

**Base models:** Qwen3.5-14B (router + fast path) + Qwen3.5-27B (planner + smart path). Note: if Qwen3.5-14B is not available, use Qwen3-14B for the fast path — same tokenizer family, fully compatible.
**Serving:** Two isolated vLLM instances on A100-80GB, FP8 quantization
**Legacy:** Ollama for archived model versions
**Hardware:** NVIDIA A100-SXM4-80GB

## Goals

- **Intelligence:** 27B handles complex reasoning, cross-document synthesis, KG-augmented analysis, and visualization — GPT-class document intelligence
- **Speed:** 14B handles simple queries with sub-second latency. EAGLE3 speculative decoding + prefix caching recover throughput on 27B path
- **Domain expertise:** Pluggable, region-aware knowledge packs bring authoritative domain knowledge (NICE, CDC, FCA, legislation) into model reasoning
- **Proactive intelligence:** Model surfaces actionable alerts (stock depletion, contract renewal, candidate gaps, medication interactions) both in-response and via scheduled digest
- **Zero regression:** Document processing pipeline is completely unchanged. All changes are query-side and serving-side only

## Non-Goals

- Changing the document processing pipeline (Upload → Extract → HITL → Screen → KG → HITL → Embed)
- Multi-GPU or multi-node serving (single A100-80GB target)
- Real-time streaming knowledge pack updates (monthly batch is sufficient)
- Training separate models per domain (one model, multiple LoRA/knowledge packs)

---

## Architecture

### Serving Layer — Dual vLLM Instances

Two isolated vLLM processes on the same A100-80GB:

| Instance | Model | VRAM | Role | Optimizations |
|----------|-------|------|------|---------------|
| vLLM-1 | Qwen3.5-14B (FP8) | ~20 GB | Intent router, fast path generation | EAGLE3 speculator, prefix caching, FP8 KV cache |
| vLLM-2 | Qwen3.5-27B (FP8) | ~35 GB | Query planner, smart path generation + verification | EAGLE3 speculator, prefix caching, FP8 KV cache, chunked prefill, guided_json |

Total VRAM: ~55 GB used, ~25 GB headroom for KV cache and concurrent requests.

Legacy model versions (V1, older iterations) remain on Ollama as GGUF archives for rollback.

**vLLM configuration for both instances:**

```
--dtype fp8
--kv-cache-dtype fp8
--enable-prefix-caching
--enable-chunked-prefill          # Instance 2 only
--speculative-model <eagle3>      # Per-model speculator
--guided-decoding-backend xgrammar
```

### Query Pipeline — Plan then Execute

```
User Query
    │
    ▼
┌─────────────────┐
│  Intent Router   │  14B classifies intent + complexity
│  (vLLM-1, 14B)  │  Output: {intent, complexity, route: fast|smart}
└────────┬────────┘
         │
    ┌────┴─────┐
    │          │
    ▼          ▼
 FAST PATH   SMART PATH
    │          │
    │     ┌────┴────────┐
    │     │ Phase 1:     │  27B produces execution plan
    │     │ PLAN (27B)   │  {steps: [search, kg_lookup, ...]}
    │     └────┬────────┘
    │          │
    │     ┌────┴────────┐
    │     │ Phase 2:     │  Deterministic execution
    │     │ EXECUTE      │  Qdrant, Neo4j, spreadsheets,
    │     │ (no LLM)     │  knowledge packs — parallel where possible
    │     └────┬────────┘
    │          │
    │     ┌────┴────────┐
    │     │ Phase 3:     │  27B generates response + chart_spec
    │     │ GENERATE +   │  + alerts. Self-verifies confidence.
    │     │ VERIFY (27B) │  Re-retrieves if confidence < threshold
    │     └────┬────────┘  (max 2 re-retrieve loops)
    │          │
    └────┬─────┘
         │
         ▼
  Response Delivery
  (markdown + chart_spec + alerts + sources)
```

### Intent Router (14B, LLM-based)

The 14B model classifies every incoming query. No regex — the LLM understands paraphrasing, multilingual queries, and ambiguous intent.

**Router prompt:**

```
Classify this query. Return JSON only.
- intent: greeting|identity|lookup|list|count|summarize|compare|analyze|investigate|extract|generate|aggregate|rank|timeline|visualize
- complexity: low|medium|high
- route: fast|smart
- requires_kg: true|false
- requires_visualization: true|false

Rules:
- greeting, identity → fast
- lookup, list, count (single doc) → fast
- Everything else → smart
- Any multi-document, analytical, comparative, or visualization query → smart
```

**Output (~50 tokens, <200ms):**

```json
{"intent": "compare", "complexity": "high", "route": "smart", "requires_kg": true, "requires_visualization": true}
```

If routed to fast path, the 14B already has the query loaded and responds immediately — zero wasted work.

### Phase 1 — Query Planner (27B)

The 27B produces a structured execution plan using `guided_json` for guaranteed valid output.

**Available plan actions:**

| Action | Executor | What it does |
|--------|----------|-------------|
| `search` | Qdrant (BGE-M3 hybrid) | Semantic + sparse vector search in user documents |
| `knowledge_search` | Qdrant (knowledge pack collection) | Search domain knowledge (NICE, CDC, legislation) |
| `kg_lookup` | Neo4j Cypher | Entity/relationship traversal |
| `kg_search` | Neo4j full-text | Find entities matching description |
| `cross_reference` | Retrieve + compare | Verify claims across sources |
| `spreadsheet_query` | Structured data lookup | Query Excel/CSV data |
| `generate` | 27B Phase 3 | Final response generation |

**Plan JSON schema:**

```json
{
  "intent": "string",
  "complexity": "low|medium|high",
  "steps": [
    {
      "id": "step_1",
      "action": "search|knowledge_search|kg_lookup|kg_search|cross_reference|spreadsheet_query|generate",
      "query": "string",
      "collection": "user_docs|knowledge_nice|knowledge_who|...",
      "top_k": "number (default 10)",
      "depends_on": ["step_id", "..."],
      "params": {}
    }
  ],
  "requires_kg": "boolean",
  "requires_visualization": "auto|true|false",
  "domain_pack": "clinical|legislation|financial|null",
  "region": "UK|US|EU|AU|IN|*"
}
```

### Phase 2 — Pipeline Executor (deterministic, no LLM)

Executes the plan steps. Analyzes dependencies and runs independent steps in parallel.

```
Plan steps:
  1. search("inventory levels")           ─┐
  2. search("purchase orders pending")     ├── parallel (no dependencies)
  3. kg_lookup("all_suppliers")            ─┘
  4. cross_reference(step_1, step_2, step_3)  ── sequential (depends on 1,2,3)
  5. generate(...)                            ── sequential (depends on 4)
```

**Context assembly for Phase 3:**

The executor assembles all results into a structured context block:

```
<profile_context>
[Pre-computed domain metadata + collection insights from Profile Intelligence Builder]
</profile_context>

<retrieved_evidence>
[Chunk 1] Source: filename.pdf, p.N, Relevance: 0.94
Content...
[Chunk 2] ...
</retrieved_evidence>

<kg_context>
entities:
  - id: E1, name: "...", type: ..., doc_sources: [...]
relationships:
  - E1 --[RELATION]--> E2, ...
</kg_context>

<knowledge_pack_context>
Source: NICE NG136, Section 1.4.2
"First-line treatment for hypertension..."
</knowledge_pack_context>

<cross_reference_results>
- CONFLICT: ...
- AGREEMENT: ...
</cross_reference_results>

<spreadsheet_data>
[Structured tabular results]
</spreadsheet_data>

<user_query>
The original user question
</user_query>
```

**Retrieval improvements integrated into Phase 2:**

- **BGE-M3 hybrid retrieval:** Dense + sparse in one model, replaces BGE-large
- **CRAG (Corrective RAG):** Retrieved chunks scored for relevance before assembly. Low-quality chunks discarded, re-retrieval triggered with refined query if needed
- **RAPTOR (tree-based):** Hierarchical chunk summaries for long documents (100+ pages). Cross-chunk reasoning without full context stuffing

### Phase 3 — Generate + Verify (27B)

The 27B receives the assembled context and generates the final response.

**Output format (three sections):**

```
<response>
[Markdown-formatted answer with evidence grounding and citations]
</response>

<chart_spec>
[JSON chart specification — only when visualization adds value]
{"charts": [{
  "id": "chart_1",
  "type": "bar|line|pie|scatter|heatmap",
  "title": "string",
  "subtitle": "string",
  "x": {"label": "string", "values": ["..."]},
  "series": [{"name": "string", "values": [numbers], "color": "string|null"}],
  "unit": "string",
  "annotations": [{"point": "string", "text": "string"}],
  "source": "string"
}], "layout": "single|side_by_side|stacked"}
</chart_spec>

<alerts>
[JSON array — only when actionable items detected]
[{
  "severity": "critical|warning|info",
  "category": "string",
  "title": "string",
  "detail": "string",
  "action": "string",
  "source": "string"
}]
</alerts>
```

**Self-verification loop:**

After generating, the 27B evaluates its own confidence:
- If confidence < 0.7 and re-retrieve count < 2: refine search query based on what was missing, trigger Phase 2 re-execution, regenerate
- If confidence >= 0.7 or max re-retrieves reached: deliver response
- Confidence is grounded in evidence coverage: "Did I find supporting data for each claim?"

**Chart_spec generation triggers:**
- Explicit request: always generate
- Auto-detect: response contains 3+ comparable numeric values, trend data, percentage breakdowns, or ranked data
- Suppress: greetings, short answers, single values, non-numeric responses, gap/error responses

**Alert generation triggers:**
- Threshold breaches detected in data (stock levels, expiry dates, contract deadlines)
- Cross-document contradictions or inconsistencies
- Missing required information (gaps in compliance, incomplete candidate profiles)
- Severity determined by urgency and business impact

---

## Document Processing Pipeline — Intelligence Layer

The existing pipeline is **completely unchanged**. A new Intelligence Builder stage is appended after Embed:

```
Upload → Extract → HITL → Screen → KG → HITL → Embed → Profile Intelligence Builder
```

### Profile Intelligence Builder

Runs after embedding, produces pre-computed intelligence stored in MongoDB (metadata only, no document content):

```json
{
  "profile_id": "string",
  "profile_type": "hr_recruitment|finance|legal|logistics|medical|generic",
  "document_count": 50,
  "last_updated": "ISO timestamp",

  "entities_summary": {
    "total_entities": 245,
    "by_type": {"Person": 50, "Skill": 142, "Certification": 38, "Company": 87}
  },

  "computed_profiles": [
    {
      "entity_id": "E_priya_sharma",
      "type": "candidate",
      "name": "Priya Sharma",
      "structured_data": {
        "total_experience_years": 8,
        "skills": ["Kubernetes", "AWS", "Terraform"],
        "certifications": ["AWS SA", "CKA"],
        "role_fit_scores": {"Senior DevOps": 0.92, "Cloud Architect": 0.85}
      },
      "doc_sources": ["resume_priya_sharma.pdf"]
    }
  ],

  "collection_insights": {
    "skill_distribution": {"Kubernetes": 34, "AWS": 28, "Python": 45},
    "patterns": ["72% candidates have cloud certification"],
    "gaps": ["No HashiCorp Terraform certified candidates"],
    "anomalies": []
  },

  "domain_metadata": {
    "detected_domain": "hr_recruitment",
    "document_types": {"resume": 48, "job_description": 2},
    "analysis_templates": ["candidate_ranking", "skill_gap", "diversity_analysis"]
  }
}
```

**Domain-specific computed profiles:**

| Domain | Computed Profiles | Collection Insights |
|--------|-------------------|---------------------|
| HR | Candidates: skills, certs, experience, role-fit scores | Skill distribution, experience curves, cert gaps |
| Finance | Invoices: amounts, vendors, terms, line items | Spend by vendor, payment trends, anomalies |
| Legal | Contracts: parties, clauses, obligations, key dates | Obligation timeline, risk scores, renewal calendar |
| Logistics | Products: stock levels, suppliers, lead times, expiry | Reorder alerts, expiry calendar, supplier dependency |
| Medical | Patients: diagnoses, medications, providers, history | Treatment patterns, medication interactions, care gaps |

**Storage:**
- MongoDB: computed profiles, collection insights, domain metadata (control plane only)
- Neo4j KG: enriched with computed relationships (role-fit scores, risk scores, dependency chains)
- Qdrant: enriched chunk payloads include entity context from computed profiles

**Profile context injection at query time:**

The system prompt is dynamically enriched with profile intelligence:

```
<profile_context>
Domain: hr_recruitment
Documents: 50 (48 resumes, 2 job descriptions)
Candidates: 50
Key skills: Kubernetes (34), Python (45), AWS (28)
Experience range: 0-15 years (avg 5.8)
Certification coverage: 72% have cloud certs
Notable gaps: No HashiCorp Terraform certified candidates
Available analyses: candidate_ranking, skill_gap, diversity_analysis
</profile_context>
```

---

## Domain Knowledge Pack System

### Architecture

Pluggable, region-aware system for integrating authoritative domain knowledge.

**Pack structure:**

```
Knowledge Pack
├── Scraper       — KnowledgePackScraper interface: fetch content from source
├── Parser        — KnowledgePackParser interface: structure raw content
├── Training Data — SFT examples: domain reasoning patterns, citation formats
├── RAG Collection— Qdrant collection with latest content
├── Update Job    — Cron schedule for content refresh
├── Domain Config — Terminology, relationship types, alert rules, citation format
└── Validation    — Quality checks for scrape/parse accuracy
```

**Region-aware pack registry:**

```
Knowledge Pack: Clinical Guidelines
├── Region: UK  → NICE (nice.org.uk)
├── Region: US  → CDC + FDA + AMA guidelines
├── Region: EU  → EMA + ECDC guidelines
├── Region: AU  → NHMRC + TGA guidelines
├── Region: IN  → ICMR + CDSCO guidelines
└── Region: *   → WHO guidelines (global fallback)

Knowledge Pack: Legislation
├── Region: UK  → legislation.gov.uk
├── Region: US  → congress.gov + CFR
├── Region: EU  → eur-lex.europa.eu
└── Region: *   → UN treaty database (fallback)

Knowledge Pack: Financial Regulation
├── Region: UK  → FCA handbook
├── Region: US  → SEC + FINRA
├── Region: EU  → ESMA + MiFID II
└── Region: *   → Basel Committee (fallback)

Knowledge Pack: Employment Law
├── Region: UK  → ACAS + CIPD + Employment Rights Act
├── Region: US  → EEOC + DOL + FLSA
├── Region: EU  → EU Employment Directive
└── Region: *   → ILO conventions (fallback)
```

### Integration Model: Hybrid (fine-tune framework + RAG specifics)

- **Fine-tune once:** SFT training data teaches the model domain reasoning patterns, terminology, decision frameworks, and citation formats. The model "thinks like a clinician" or "thinks like a lawyer."
- **RAG always:** Specific guidelines, numbers, dates, recommendations are retrieved from the knowledge pack Qdrant collection. Always current, never stale.
- **Monthly update:** Scraper checks for new/updated/withdrawn content. Qdrant collection updated. No model retraining needed for content updates.

### Region Configuration

Region is a tenant-level setting:

```json
{
  "subscription_id": "sub_123",
  "region": "UK",
  "active_packs": ["clinical", "legislation", "financial"],
  "pack_overrides": {
    "clinical": ["nice", "who"]
  }
}
```

At query time, the 27B planner knows the tenant's region and includes the appropriate knowledge pack collections in its search steps.

### First Pack: NICE Clinical Guidelines

| Attribute | Value |
|-----------|-------|
| Source | nice.org.uk/guidance/published |
| Content types | Clinical guidelines (NG), Technology appraisals (TA), Quality standards (QS), Pathways |
| Estimated volume | ~2,000+ documents |
| Scrape method | NICE API + HTML parsing |
| Update schedule | Monthly (1st of each month) |
| SFT training examples | ~3,000 (clinical reasoning, citation format, recommendation interpretation) |
| Qdrant collection | `knowledge_nice` |
| Citation format | `[NICE NG136, Section 1.4.2, Recommendation 1.4.2.3]` |

### Adding a New Pack

1. Implement `KnowledgePackScraper` — URL patterns, auth, pagination
2. Implement `KnowledgePackParser` — raw content → structured sections/recommendations
3. Generate SFT training data — domain reasoning examples using scraped content as templates
4. Run one fine-tuning iteration to absorb domain reasoning patterns
5. Register pack config: region, Qdrant collection name, update schedule, citation format

No pipeline changes, no architecture changes.

---

## Proactive Intelligence — Alerts System

### In-Response Alerts

Built into Phase 3 generation. When assembled context contains data triggering alert conditions, the 27B surfaces them in the `<alerts>` block.

**Severity levels:**
- `critical` — requires immediate action, business impact if ignored
- `warning` — needs attention within days/weeks
- `info` — worth noting, no urgency

**Alert categories per domain:**

| Domain | Alert Types |
|--------|------------|
| Logistics | stock_depletion, expiry_approaching, supplier_delay, reorder_needed |
| HR | certification_gap, missing_reference, experience_mismatch, diversity_flag |
| Finance | payment_overdue, budget_exceeded, anomalous_transaction, contract_renewal |
| Legal | obligation_deadline, clause_conflict, compliance_gap, amendment_needed |
| Medical | medication_interaction, guideline_deviation, follow_up_overdue, contraindication |

### Scheduled Analysis Agent

Background agent running on configurable schedule per profile:

- **Trigger:** Cron (configurable — daily, weekly)
- **Process:** For each active profile, send meta-query to 27B smart path: "Analyze all documents for items requiring attention"
- **Output:** Alert digest stored in MongoDB, delivered via Teams / email / UI banner
- **Isolation:** Runs per-profile, respects profile isolation (non-negotiable)
- **Uses same pipeline:** 27B smart path with Phase 1-3 — no separate alert logic

---

## Efficiency Optimizations

### Tier 1: Deploy Immediately

| Technique | Impact | Configuration |
|-----------|--------|---------------|
| FP8 model weights | Better quality than Q4_K_M, fits both models on A100 | `--dtype fp8` |
| FP8 KV cache | 2x concurrent users in same VRAM | `--kv-cache-dtype fp8` |
| Prefix caching (APC) | Eliminates system prompt reprocessing | `--enable-prefix-caching` |
| Guided JSON (XGrammar) | Guaranteed valid chart_spec/alerts JSON | `guided_json=<schema>` per request |
| Chunked prefill | Long document contexts don't block short requests | `--enable-chunked-prefill` |

### Tier 2: High Impact

| Technique | Impact | Implementation |
|-----------|--------|---------------|
| EAGLE3 speculative decoding | 1.5-2x throughput, zero quality loss | Download `RedHatAI/Qwen3-14B-speculator.eagle3`, configure in vLLM |
| BGE-M3 hybrid retrieval | Better accuracy, dense + sparse in one model | Replace BGE-large, single model handles both |
| CRAG (Corrective RAG) | Fewer hallucinations — bad chunks filtered before generation | Add retrieval evaluator scoring step |
| RAPTOR tree retrieval | Cross-chunk reasoning for 100+ page documents | Hierarchical chunk summaries |

### Tier 3: Architecture

| Technique | Impact | Implementation |
|-----------|--------|---------------|
| Multi-LoRA serving | Per-tenant or per-domain specialization | vLLM multi-LoRA with hot-swap |

---

## Model Versioning

```
vLLM (production):
  DHS/DocWain:v2-14b-fp8    ← vLLM Instance 1 (router + fast path)
  DHS/DocWain:v2-27b-fp8    ← vLLM Instance 2 (planner + smart path)

Ollama (legacy/archive):
  DHS/DocWain:v1             ← Frozen original (Qwen3-14B Q4_K_M)
  DHS/DocWain:v2-14b-gguf    ← GGUF archive of V2 14B
  DHS/DocWain:v2-27b-gguf    ← GGUF archive of V2 27B
```

Production models on vLLM, all legacy/rollback versions on Ollama.

---

## File Structure

### New Files

```
src/serving/
├── vllm_manager.py            # Start/stop/configure dual vLLM instances
├── model_router.py            # Intent-based routing (14B classification)
├── fast_path.py               # Fast path handler (14B direct response)
└── config.py                  # vLLM instance configs, ports, VRAM allocation

src/query/
├── planner.py                 # Phase 1: 27B query plan generation
├── executor.py                # Phase 2: deterministic plan execution (parallel steps)
├── generator.py               # Phase 3: 27B response generation + verification
├── context_assembler.py       # Assemble retrieved evidence + KG + knowledge into context
└── confidence.py              # Self-verification and re-retrieval logic

src/intelligence/
├── profile_builder.py         # Profile Intelligence Builder (post-embed)
├── computed_profiles.py       # Domain-specific profile computation
├── collection_insights.py     # Collection-level pattern detection
└── alert_generator.py         # In-response alert detection and formatting

src/knowledge_packs/
├── __init__.py
├── registry.py                # Pack registry with region routing
├── base.py                    # KnowledgePackScraper + KnowledgePackParser interfaces
├── updater.py                 # Monthly update cron job
├── packs/
│   ├── clinical/
│   │   ├── nice_scraper.py    # UK NICE guidelines scraper
│   │   ├── nice_parser.py     # NICE content parser
│   │   └── config.yaml        # Collection name, citation format, schedule
│   ├── legislation/
│   │   ├── uk_scraper.py
│   │   └── config.yaml
│   └── financial/
│       ├── fca_scraper.py
│       └── config.yaml

src/retrieval/
├── bgem3_retriever.py         # BGE-M3 hybrid dense+sparse retrieval
├── crag_evaluator.py          # Corrective RAG: score retrieved chunks
└── raptor.py                  # RAPTOR tree-based hierarchical retrieval

src/agents/
├── scheduled_analysis.py      # Background scheduled analysis agent
└── alert_digest.py            # Alert digest generation and delivery
```

### Modified Files

```
src/ask/pipeline.py            # Refactored to use new query pipeline (Plan→Execute→Generate)
src/generation/prompts.py      # Updated for profile_context injection + knowledge pack context
src/extraction/engine.py       # Profile Intelligence Builder hook after embed stage
```

### UI Changes (docwain-ui repo, develop branch)

```
src/components/
├── Chart/
│   ├── ChartRenderer.tsx      # Render chart_spec JSON via Recharts/Nivo
│   ├── ChartCard.tsx          # White card container with source attribution
│   └── chartUtils.ts          # Type definitions, color themes, spec validation
├── Alerts/
│   ├── AlertBanner.tsx        # Alert display in message bubble
│   ├── AlertDigest.tsx        # Scheduled digest view
│   └── alertUtils.ts          # Severity styling, category icons
```

---

## Success Criteria

| Metric | V1 Baseline | V2 Target |
|--------|------------|-----------|
| Fast path latency (greeting/lookup) | ~2s | < 500ms |
| Smart path latency (complex analysis) | ~5-8s | < 4s (with EAGLE3 + caching) |
| Reasoning depth (judge score 1-5) | ~2.5 | >= 4.0 |
| Hallucination rate | ~0.15 | <= 0.05 |
| Chart_spec JSON validity | N/A | 100% (guided_json) |
| Alert precision | N/A | >= 0.85 |
| Retrieval accuracy (CRAG-filtered) | ~0.70 | >= 0.90 |
| Document extraction F1 | ~0.70 | >= 0.90 |
| V1 regression pass rate | N/A | >= 90% |
| Knowledge pack citation accuracy | N/A | >= 0.90 |
| Concurrent users (A100-80GB) | ~5-10 | >= 20-30 |

---

## Constraints

- Document processing pipeline is never modified
- Profile isolation is non-negotiable — no data leakage across profiles/subscriptions
- All training data is synthetic — zero customer document content
- Knowledge pack content is publicly available authoritative sources only
- Single A100-80GB target — architecture must fit within 80GB VRAM
- MongoDB is control plane only — no document content stored
- Region configuration is tenant-level, not per-query
