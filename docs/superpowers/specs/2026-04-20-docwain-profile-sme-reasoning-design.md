# DocWain Profile-SME Reasoning Layer — Design Spec

**Status:** Draft, awaiting user review
**Author/owner:** Muthu Subramanian G
**Date:** 2026-04-20
**Sub-project:** A (of a decomposition: A Profile-SME reasoning, B Insight & analytics, C URL ingestion & crawling, D Long-form generation, E Reasoning-aware eval, F Training data & model capability)
**Related memories:** engineering-first/model-last; domain-agnostic & open-set; no timeouts; adapter YAMLs in Azure Blob; intelligence precomputed at ingestion; MongoDB = control plane only; LLM prompt code paths; unified model; V5 failure lessons

---

## 1. Context and motivation

DocWain today is architecturally a retrieval-augmented chat system (UNDERSTAND → RETRIEVE → REASON → COMPOSE in `src/agent/core_agent.py`) backed by a fine-tuned Qwen3-14B (V2 in production), Qdrant vectors, a Neo4j KG, and Redis caches. The architecture is more sophisticated than a plain RAG (65+ modules under `src/intelligence/`), but at query time the Neo4j KG is used as retrieval *hints* rather than a reasoning substrate (`src/kg/retrieval.py:56-102`, 1–2 hop expansion only), profile-level intelligence is *precomputed factual aggregation* (`src/intelligence/profile_builder.py`, `collection_insights.py`) rather than live cross-document synthesis, and no SME-role persona is injected into generation. The RAGAS baseline in `tests/ragas_metrics.json` captures the outcome precisely: `context_recall 0.802` (good — retrieval finds the right evidence), `answer_faithfulness 0.514` (weak — reasoning over evidence drifts), `hallucination_rate 0.0` (strong — must preserve).

The product vision is that DocWain acts as a domain subject-matter expert per profile — a finance profile reasons like a CFO-advisor (trend and recommendation analysis across Q1/Q2/Q3), a troubleshooting profile reasons like a support engineer (symptom → cause → fix with diagnostic flow), etc. — across any set of documents, with grounded synthesis and actionable recommendations rather than fact-dumps.

This spec describes the **engineering-first** redesign that moves DocWain from extractive RAG to pre-reasoned SME synthesis, without retraining the model. Model retraining is explicitly deferred to a later sub-project (F), triggered only when captured patterns warrant it.

## 2. Scope

### In scope
- Extend the training stage with a final **SME synthesis** step that produces five artifact types per profile
- Implement **hybrid dense+sparse retrieval** with cross-encoder reranking, parallel across four layers (chunks / KG / SME artifacts / ephemeral URL)
- Add three new query intents (`diagnose`, `analyze`, `recommend`) with rich-default response shape, compact auto-fallback, and explicit compact-on-command override
- Inject **domain-SME personas** loaded from adapter YAMLs stored in Azure Blob, with plugin extensibility for new domains
- Handle **URL-as-prompt** as an ephemeral in-memory source fetched in parallel with profile retrieval (no persistence)
- Preserve existing 0.0 hallucination rate through four layered grounding checks
- Ship six new reasoning-quality metrics alongside existing RAGAS
- Instrument ingest and query paths with trace stores that feed future pattern mining

### Out of scope (future sub-projects)
- B: Insight & analytics engine (proactive alerting, dashboards, scheduled reports)
- C: URL ingestion as persistent profile documents; scheduled/depth-controlled crawling
- D: Long-form report generation beyond the rich-mode cap
- E: Full reasoning-aware evaluation harness (LLM-judge infra, per-adapter dashboards, regression history UI)
- F: Targeted SFT/DPO data generation and fine-tuning — only triggered when patterns stabilize

### Non-goals within A
- Domain-specialized artifact builders (IT troubleshooting decision trees, legal obligation registers) — the five common-core artifacts adapt content per domain via adapter, but ship with a uniform shape
- Agentic multi-step query loop with tool iteration
- Automatic fine-tune trigger
- Refactor of `profile_builder.py`, `collection_insights.py`, `profile_intelligence.py` — these coexist; SME artifacts are additive
- Changes to authentication, authorization, tenancy, or profile creation UX

## 3. Architectural invariants

1. **Query-path latency invariant.** Query time performs *zero* new LLM calls on the hot path. The two existing LLM calls (intent classification in UNDERSTAND, generation in REASON) remain. URL-in-prompt may trigger one additional supplementary LLM call, only when URL content arrives after the primary response has begun streaming.
2. **Ingest is where intelligence is built.** All cross-document reasoning, synthesis, and inference materialization happen in the training stage. Query time is retrieval + composition.
3. **Pipeline status immutability.** No new `pipeline_status` string is introduced. SME synthesis runs as the final internal step of the training stage; `PIPELINE_TRAINING_COMPLETED` fires only after SME synthesis passes.
4. **Profile isolation.** Every artifact and retrieval layer is keyed by `(subscription_id, profile_id)` with hard filters. No cross-profile data leakage.
5. **Storage separation.** MongoDB holds control-plane only (pointers, status, `profile_domain`). Azure Blob holds document content and canonical SME artifacts and adapter YAMLs and traces. Qdrant holds vectors (chunks + SME artifact snippets). Neo4j holds the graph including synthesized edges. Redis holds short-lived cache only.
6. **Domain openness.** Zero domain names are hard-coded in core code. All domain behavior loads at runtime from adapter YAML files in Azure Blob. A `generic.yaml` adapter is the required always-works fallback.
7. **Grounding fail-closed.** `SMEVerifier` drops any artifact item that fails verification; dropped items are logged for pattern capture but never persisted and never retrieved. No ungrounded content reaches the user.
8. **Zero timeouts on internal steps.** No wall-clock abort on retrieval, reasoning, synthesis, or composition. Only external I/O (URL fetch) has per-operation safety timeouts. Latency is managed through efficiency levers, not cutoffs.
9. **Response-formatting authority.** All response-shape logic lives in `src/generation/prompts.py` and adapter YAMLs. `src/intelligence/generator.py` is not modified for formatting.

## 4. Architecture overview

### Ingest path

```
upload → EXTRACTION (auto)                                [unchanged]
      ↓
HITL SCREENING                                            [unchanged]
      ↓
HITL EMBEDDING (Qdrant; now with sparse vectors alongside dense)   [existing + sparse re-index]
      ↓
TRAINING STAGE                                            [extended]
  ├── KG build (Neo4j)                                    [existing]
  └── SME SYNTHESIS (NEW, final step)
      src/intelligence/sme/synthesizer.py
        ├── DomainResolver            (auto-detect + admin override)
        ├── SMEDossierBuilder         → Dossier artifact
        ├── InsightIndexBuilder       → Insight Index artifact
        ├── ComparativeRegisterBuilder → Comparative Register artifact
        ├── KGMultiHopMaterializer    → synthesized edges in Neo4j
        ├── RecommendationBankBuilder → Recommendation Bank artifact
        └── SMEVerifier               → drops unverifiable items, fail-closed
      ↓
PIPELINE_TRAINING_COMPLETED                               [same string; fires only after SME synthesis passes]
```

### Query path (latency-preserving)

```
POST /api/ask  [src/main.py:249]
      ↓
execution/router.py: classify SIMPLE vs COMPLEX; session load
      ↓
CoreAgent.handle()
  ├── UNDERSTAND   — intent + entities + format_hint + URL detection
  │                  (extended classifier includes new intents diagnose/analyze/recommend)
  ├── URL FETCH   — kicked off at time 0 in parallel, if URL in query (SSRF-safe, safety-timed)
  ├── RETRIEVE   — parallel four layers → merge → cross-encoder rerank → MMR diversity
  │     Layer A: chunks          (hybrid dense+sparse via Qdrant + RRF)
  │     Layer B: KG hints        (Neo4j 1-hop + pre-materialized synthesized edges)
  │     Layer C: SME artifacts   (hybrid dense+sparse against SME collection, type-filterable)
  │     Layer D: ephemeral URL   (only if URL detected; merged when ready)
  ├── REASON     — single LLM call; richer pack + adapter persona; streaming
  └── COMPOSE    — rich-default / compact-auto / compact-override shape; existing citation_verifier runs
```

## 5. Domain detection and adapters

### Domain detection
- Extended classifier in `src/intelligence/profile_builder.py` aggregates per-document domain classifications and picks the dominant domain as `profile_domain`
- Persisted on the MongoDB profile record as `profile_domain` field (control-plane data, allowed by rule)
- Admin override via `PATCH /profiles/{profile_id}` with a `profile_domain` body field
- Unknown or ambiguous → `generic`; system always functions

### Adapter YAML schema (abbreviated)
```yaml
domain: <string>         # must match the profile_domain this adapter is invoked for
version: <semver string>
persona:
  role: <string>         # e.g. "senior financial analyst advising the C-suite"
  voice: <string>        # e.g. "direct, quantitative, hedged"
  grounding_rules: [<string>, ...]
dossier:
  section_weights:       # weights sum to 1.0; determines builder focus
    <section_name>: <float>
  prompt_template: <blob-relative path, e.g. "prompts/finance_dossier.md">
insight_detectors: [{type, rule, params}, ...]
comparison_axes: [{name, dimension, unit?}, ...]
kg_inference_rules: [{pattern, produces, confidence_floor, max_hops}, ...]
recommendation_frames: [{frame, template, requires: {insight_types}}, ...]
response_persona_prompts:
  diagnose: <blob-relative path>
  analyze:  <blob-relative path>
  recommend:<blob-relative path>
retrieval_caps:
  max_pack_tokens:
    analyze: 6000
    diagnose: 5000
    recommend: 4500
    investigate: 8000
    # smaller intents inherit from generic
output_caps:
  analyze: 1200
  diagnose: 1500
  recommend: 1000
  investigate: 2000
```

### Adapter storage (all YAMLs and templates in Azure Blob)

```
{blob_container}/
  sme_adapters/
    global/
      generic.yaml                (required; always-safe fallback)
      finance.yaml, legal.yaml, hr.yaml, medical.yaml, it_support.yaml
      prompts/*.md
    subscription/{subscription_id}/
      <domain>.yaml               (optional per-subscription override)
      prompts/*.md
```

Resolution order for `(subscription_id, profile_domain)`: subscription-specific → global → `generic`.

### Adapter loader
- `src/intelligence/sme/adapter_loader.py`
- First-use fetch per `(subscription_id, domain)` key; in-memory cache with 5-minute TTL
- Admin API `POST /admin/sme-adapters/invalidate` for force-reload
- Every load logs adapter `version` + content hash; every synthesis trace + query trace records which adapter version was in effect
- Bootstrap: deployment playbook uploads `deploy/sme_adapters/defaults/` → `blob:sme_adapters/global/` on first install; code ships NO hardcoded YAMLs under `src/`
- Failure handling: Blob fetch fails → last-cached version. If no cache and Blob is unreachable, the service loads a minimal embedded `generic` adapter shipped under `deploy/sme_adapters/last_resort/generic.yaml` and flags health status `degraded`. This embedded copy is the bootstrap safety net only; the production load path always goes through Blob. The `generic` adapter must always succeed; the system never refuses to answer because an adapter is missing.

## 6. The five SME artifacts

Every item in every artifact carries a provenance chain (`doc_ids + chunk_ids`) and `confidence` score; unverifiable items never persist.

| Artifact | Purpose | Indexed | Key fields |
|---|---|---|---|
| **SME Dossier** | Domain-aware synthesized overview of the profile, SME-voiced | Blob (canonical) + Qdrant per-section (retrievable) | `section`, `narrative`, `evidence[]`, `entity_refs[]`, `confidence` |
| **Insight Index** | Typed analytical outputs from the profile | Qdrant (narrative-embedded, type-filterable) | `type ∈ {trend, anomaly, gap, risk, opportunity, conflict}`, `narrative`, `evidence[]`, `confidence`, `domain_tags[]`, `temporal_scope?`, `entity_refs[]` |
| **Comparative Register** | Cross-doc deltas, conflicts, timeline, corroboration | Qdrant (type-filterable) | `type ∈ {delta, conflict, timeline, corroboration}`, `axis`, `compared_items[] (≥2 docs)`, `analysis`, `resolution?`, `evidence[]` |
| **KG Multi-Hop Materialization** | Pre-computed 3–5 hop inferences as new Neo4j edges | Neo4j (generic `INFERRED_RELATION` edge with `relation_type` property) | `relation_type`, `confidence`, `evidence[]`, `inference_path`, `synthesis_version`, `adapter_version`, `generated_at` |
| **Recommendation Bank** | Grounded, actionable recommendations tied to insights | Qdrant (recommendation-embedded) | `recommendation`, `rationale`, `linked_insights[]`, `estimated_impact` (quantitative if evidence supports, else qualitative + labeled), `assumptions[]`, `caveats[]`, `evidence[]`, `confidence`, `domain_tags[]` |

### SMEVerifier contract (fail-closed)

Runs after each artifact builder, before persistence. Five checks:

1. **Evidence presence.** Every claim cites ≥1 `(doc_id, chunk_id)` from the profile's content.
2. **Evidence validity.** Cited chunks exist in Qdrant; item's quoted/summarized text is substantively present (lexical + semantic similarity threshold).
3. **Inference provenance.** Inferred items carry an `inference_path` traceable to grounded evidence within N hops (adapter-configured).
4. **Confidence calibration.** `confidence > 0.8` requires ≥2 independent evidence sources; otherwise rolled back to ≤0.6.
5. **Contradiction check.** No item contradicts a higher-confidence item in the same batch without a `conflict` annotation.

Failed items → logged to synthesis trace → dropped.

## 7. Query-time retrieval architecture

### Zero LLM calls in retrieval
All retrieval stages (query expansion, entity extraction, intent classification, rerank) are non-LLM or reuse the existing intent-classifier LLM call.

### Stages

| Stage | What | Typical wall-clock |
|---|---|---|
| **Stage 0 — Query expansion** (non-LLM) | Entity extraction, temporal parsing, intent + format_hint classification (existing call), synonym expansion from adapter lexicon, URL detection | 10–30 ms |
| **Stage 1 — Parallel retrieval** | A chunks (hybrid + RRF), B KG (Neo4j 1-hop + synthesized edges), C SME artifacts, D ephemeral URL (if present, truly parallel) | wall-clock = max of layers; 150–250 ms without URL; URL bound by external fetch |
| **Stage 2 — Merge + rerank** | Union candidates; cross-encoder rerank top-40 → top-10; MMR for diversity | 100–200 ms |
| **Stage 3 — Budget-aware pack assembly** | Enforce adapter `max_pack_tokens`; drop lowest-confidence first; semantic dedup; SME artifacts compressed to key-claims + refs | 5–20 ms |
| **Stage 4 — Reasoner** (single LLM call, streaming) | Pack → response, streaming | TTFT 500–900 ms; completion varies with output length |

### Six efficiency mechanisms

| # | Mechanism | Effect |
|---|---|---|
| 1 | Hybrid dense+sparse with RRF (Qdrant native) | Semantic recall + keyword precision; smaller, stronger candidate pool |
| 2 | Cross-encoder reranker always on | Top-N has real relevance; tighter pack, shorter prompt, faster prefill |
| 3 | KG multi-hop is pre-materialized | Query time always 1-hop, bounded |
| 4 | QA cache on the fast path | Cache hit → return pre-grounded answer, skip Reasoner entirely |
| 5 | Intent-aware layer gating | Simple intents skip Layer B/C entirely |
| 6 | Redis retrieval cache keyed by `(sub, prof, query_fingerprint)` | Near-duplicate queries skip Stages 1–2 |

### Adaptive top-K per layer

| Query shape | Layer A | Layer B | Layer C |
|---|---|---|---|
| Trivial lookup | 5 | off | off |
| Structured extract | 10 | off | 0–2 |
| Compare / summarize | 12 | 5 | 5 |
| Analyze / diagnose / recommend / investigate | 15 | 10 | 10 |

Adapter-tunable per domain.

### URL-as-prompt handling

- SSRF-safe fetcher in `src/tools/url_fetcher.py`: blocks localhost, RFC1918, link-local, `file://`, private redirects; HTTP/HTTPS only
- Safety limits (external I/O only): fetch 15 s, extract 30 s, size 10 MB — all configurable per subscription via adapter
- Two cases, auto-selected:
  - **Supplementary**: profile retrieval yields strong evidence → Reasoner starts streaming immediately using profile pack; URL runs in parallel; URL chunks include in pack if ready before Stage 3, else buffer and emit supplementary analysis section after primary response ends (one additional LLM call, URL queries only, late-arrival only)
  - **Primary**: profile retrieval yields weak evidence and URL is query-dominant → Reasoner waits for URL fetch before starting; honest-compact fallback if fetch fails
- Ephemeral throughout: URL chunks never persisted to Qdrant, Neo4j, or Blob; session-scoped only

## 8. Prompts, response shape, grounding

### New intents (added to `src/generation/prompts.py` TASK_FORMATS)

| Intent | Triggered by | Pack emphasis | Default shape |
|---|---|---|---|
| `diagnose` | troubleshooting phrasing, symptom→cause→fix queries | Recommendation Bank + troubleshooting-tagged Insights + diagnostic chunks | Rich: symptom → ranked candidate causes → fix steps → caveats |
| `analyze` | "analyze/assess/evaluate", pattern/trend phrasing | Insight Index + Comparative Register + Dossier + chunks | Rich: exec summary → observations → patterns → interpretation → caveats |
| `recommend` | "what should we do", "how do we improve" | Recommendation Bank primary + linked Insights + evidence | Rich: top 3–5 recommendations with rationale + evidence + estimated impact + assumptions + caveats |

### Shape resolution

```
explicit compact override in query    → compact (today's templates)
trivial intents (lookup, single-field extract, greeting, identity, count) → compact (today)
analytical intents (analyze, diagnose, recommend, investigate)            → rich
borderline (compare, summarize, aggregate, list)                          → rich IF SME artifacts meaningfully contributed to pack, else honest-compact
```

### Rich template skeleton

```
## {exec_summary}                           ← streams first (headline)
## Analysis                                  ← evidence-grounded narrative
## {Recommendations|Causes|Patterns|Findings} ← intent-specific section
## Assumptions & caveats                     ← explicit
## Evidence                                  ← dedup'd sources (existing composer.py)
```

### Domain persona injection

Adapter's persona block is prepended to the system prompt for rich-mode responses. All rich-mode prompt templates and persona prompts live in Azure Blob under `sme_adapters/.../prompts/` and load through the adapter loader. No domain names in core code.

### Grounding at query time — four choke points

1. **Ingest-time SMEVerifier** (Section 6): artifacts without valid evidence are dropped before they reach retrieval
2. **Retrieval filtering**: profile_id hard filter; confidence thresholds on synthesized edges; only verified artifacts retrievable
3. **Query-time prompt discipline**: pack items carry provenance; prompt mandates inline citations; adapter persona rules prohibit silent extrapolation
4. **Post-generation verification**: existing `citation_verifier.py` runs; new pass verifies recommendation-intent responses tie recommendations to bank items or exposed reasoning; failing claims dropped with candid "Note: {N} claim(s) could not be verified" appendix

Invariant: a claim with no provenance never reaches the user. 0.0 hallucination rate preserved as regression gate.

## 9. Storage map

| Data | System | Location |
|---|---|---|
| Profile record (incl. `profile_domain`, SME status) | MongoDB | existing collection |
| Pipeline document status | MongoDB | existing |
| Chunks + dense + sparse embeddings | Qdrant | `{subscription_id}` (existing collection naming preserved; sparse vectors added to existing payloads via re-index) |
| SME artifact retrievable snippets | Qdrant | `sme_artifacts_{subscription_id}` (NEW collection naming) |
| SME artifact canonical full text | Azure Blob | `sme_artifacts/{sub}/{prof}/{artifact_type}/{version}.json` |
| Original KG edges | Neo4j | existing ontology |
| Synthesized KG edges | Neo4j | generic `INFERRED_RELATION` edge type on existing nodes, rich properties incl. `relation_type` |
| Adapter YAMLs | Azure Blob | `sme_adapters/global/` + `sme_adapters/subscription/{sub}/` |
| Adapter prompt templates | Azure Blob | `sme_adapters/global/prompts/` + per-sub overrides |
| Synthesis traces | Azure Blob | `sme_traces/synthesis/{sub}/{prof}/{synthesis_id}.jsonl` |
| Query traces | Azure Blob | `sme_traces/queries/{sub}/{prof}/{YYYY-MM-DD}/{query_id}.jsonl` |
| Retrieval cache | Redis | `dwx:retrieval:{sub}:{prof}:{query_fingerprint}` (TTL 5 min) |
| Eval sets + metrics | Repo | `tests/sme_evalset_v1/`, `tests/sme_metrics_daily.json` |

### Neo4j synthesized edge scheme

One generic edge type, rich properties — keeps schema stable while adapters define arbitrary inference types:

```cypher
(a)-[:INFERRED_RELATION {
    source: 'sme_synthesis',
    relation_type: 'indirectly_funds',     -- adapter-defined string
    confidence: 0.72,
    evidence: ['doc_123#chunk_7', 'doc_145#chunk_2'],
    inference_path: [{from,edge,to}, ...],
    synthesis_version: 3,
    adapter_version: '1.2.0',
    generated_at: '2026-04-20T14:30:00Z',
    subscription_id: 'sub_xyz',
    profile_id: 'prof_finance_1'
}]->(c)
```

Confidence-threshold filtered at query. Scheduled cleanup prunes edges below a rolling confidence floor.

## 10. Measurement framework

### Baselines captured before any change ships
- Existing RAGAS on current V2: `faithfulness 0.514`, `hallucination 0.0`, `context_recall 0.802`, `grounding_bypass 0.0`
- Per-intent latency distributions (TTFT, total p50/p95/p99) on current production path
- Human-rated SME score on ~100 queries per major domain (6 domains × 100 = 600 queries), 1–5 scale

### New reasoning metrics (supplement RAGAS, run nightly)

| Metric | Measures | Pass threshold |
|---|---|---|
| `recommendation_groundedness` | Every recommendation traces to Recommendation Bank or exposed reasoning | ≥ 0.95 |
| `cross_doc_integration_rate` | Analytical responses cite ≥2 distinct documents | ≥ 0.70 |
| `insight_novelty` | Analytical claims exceed single-doc facts | ≥ 0.40 |
| `sme_persona_consistency` | LLM-judge 0–5 rating of voice matching adapter persona | ≥ 4.0 avg |
| `verified_removal_rate` | % of responses with zero citation-verifier drops (safety drops should be rare, not never) | ≥ 0.85 |
| `sme_artifact_hit_rate` | Analytical queries retrieve ≥1 SME artifact | ≥ 0.90 |

### Launch gate for sub-project A (all must hold)
- RAGAS: `faithfulness ≥ 0.80` (up from 0.514), `hallucination = 0.0`, `context_recall ≥ 0.80`, `grounding_bypass = 0.0`
- All 6 new metrics at or above threshold
- Human-rated SME average ≥ 4.0/5.0 improvement over baseline on matched query set
- Zero regressions on existing suite
- Latency regression check (non-blocking flag): p95 TTFT on complex queries ≤ 1.5 s; p95 total ≤ 15 s on baseline hardware — if exceeded, investigate; do not auto-block

## 11. Pattern-capture instrumentation

### Ingest-time trace
`sme_traces/synthesis/{sub}/{prof}/{synthesis_id}.jsonl` — per run: synthesis metadata, adapter version + content hash, per-builder input/output/LLM calls/timings, verifier drops with reasons, per-item provenance.

### Query-time trace
`sme_traces/queries/{sub}/{prof}/{YYYY-MM-DD}/{query_id}.jsonl` — per query: intent, format_hint, adapter version, retrieval layers + top results + rerank order + cache hit/miss, pack composition + tokens, Reasoner prompt hash + response, citation verifier drops, per-stage timing, URL details if any, user feedback from existing `feedback_tracker.py`.

### Pattern mining
Monthly batch via `scripts/mine_sme_patterns.py`:
- Clusters successful patterns (high-rated, cleanly-cited, high artifact utilization)
- Clusters failure patterns (low-rated, verifier drops, honest-compact fallbacks, recurring failing query shapes)
- Outputs `analytics/sme_patterns_{YYYY-MM}.md` for human review

### The bridge to future training
When a failure pattern stabilizes across ≥2 months of data with sufficient volume, it becomes a candidate for sub-project F training-data generation. Success patterns from the same data define what "right" looks like. Only then is F triggered.

## 12. Rollout phases (each a measurement gate, not a calendar)

### Phase 0 — Baselines + eval harness (prerequisite)
Capture RAGAS, latency distributions, human-rated SME scores on 600-query eval set. Build `tests/sme_evalset_v1/`. Implement the 6 new metrics as measurement tooling. **No production changes.**

### Phase 1 — Infrastructure, dark-launched
Adapter loader + Blob storage + admin API; default YAMLs uploaded. `SMEVerifier`, trace stores, five artifact builders (generic default). Qdrant re-index to add sparse vectors (per-sub rolling, dense-only fallback during migration). Cross-encoder reranker wired in. Synthesis runs on sandbox subscription only.

### Phase 2 — SME synthesis in training stage, retrieval behind flag
SME synthesis runs for HITL opt-in subscriptions. Retrieval layers 2 and 3 implemented but gated by `enable_sme_retrieval`. Prompts unchanged. **Gate:** artifact quality holds on real profiles.

### Phase 3 — Query-time SME retrieval on (responses still compact)
Flag flipped for opt-in subscriptions. Retrieval pulls all four layers. Prompts still compact. **Gate:** RAGAS faithfulness 0.514 → ≥0.80 on eval set; hallucination unchanged; context recall unchanged. Proves pre-reasoned artifacts improve answers before touching response shape.

### Phase 4 — Rich-mode responses + three new intents
`prompts.py` extensions live; `diagnose` / `analyze` / `recommend`; rich-default templates; compact override; domain persona injection. **Gate:** all 6 new metrics ≥ threshold; human-rated SME ≥4.0 avg improvement; zero regressions.

### Phase 5 — URL-as-prompt live
SSRF-safe fetcher; ephemeral source; parallel crawl with supplementary/primary cases. Per-subscription flag. **Gate:** URL fetch fail rate <3% on curated URLs; no SSRF bypass; URL-supplementary latency ≈ URL-less + fetch time.

### Phase 6 — Pattern mining + monthly review loop
First `mine_sme_patterns` run 30 days after Phase 4 wide rollout. Findings in `analytics/sme_patterns_{YYYY-MM}.md`. Review closes the engineering-first → pattern-capture loop and defines trigger conditions for sub-project F.

### Rollback
Every phase has a feature flag or flag-equivalent reverting to the prior phase's behavior. No phase introduces a data-destroying change. Synthesized edges are confidence-taggable for bulk-hide (not delete) on rollback.

## 13. Risks and mitigations

| Risk | Mitigation |
|---|---|
| SME synthesis balloons training-stage wall-clock on large profiles | Adapter `max_synthesis_tokens` + per-builder budgets; **incremental synthesis** — re-synthesize only artifacts affected by newly-approved docs; builders persist streaming |
| Over-synthesis produces low-quality artifacts that pollute retrieval | SMEVerifier fail-closed; trace-level drop-rate monitoring; adapter-tunable thresholds; eval-set gate blocks deploy on regression |
| Adapter YAML drift (broken admin uploads) | Schema validation on upload; dry-run synthesis on a sample profile before commit; adapter-version logging in every trace |
| Rich-mode regresses simple-query UX | Honest-compact fallback + auto-compact for trivial intents + explicit compact override |
| URL-as-prompt data-leak risk | SSRF blocks; per-sub domain allowlist; audit trail; size cap; safety timeouts |
| Neo4j edge count explodes from multi-hop materialization | Adapter `kg_inference_rules` depth cap (3–5); confidence-floor persistence; scheduled cleanup |
| 65+ intelligence modules cause integration complexity | Implementation plan audits candidate reuse (`reasoning_engine`, `cross_doc`, `answerability_index`, `evidence_verifier`, `verifier`, `hallucination_corrector`, `confidence_scorer`, `qa_generator`) before committing to reuse vs. wrap vs. replace |
| `intelligence/generator.py` accidentally gets formatting code | Memory rule flagged explicitly in spec acceptance criteria; code-review gate |
| QA cache serves stale answers after ingest | Cache invalidated on `PIPELINE_TRAINING_COMPLETED` transition; per-synthesis-version stamping |
| Cross-subscription data leak | Hard `(subscription_id, profile_id)` filter at every retrieval layer; SMEVerifier refuses artifacts if subscription boundary can't be asserted; integration-test gate for cross-sub query rejection |

## 14. Open questions (deferred to implementation plan)

1. Existing-module reuse audit: which of the 8 candidate modules are library-reused, rewritten, or deprecated. Settled in the implementation plan after a read-through.
2. Cross-encoder reranker model choice (MiniLM / BGE / trained DocWain reranker) and its validation.
3. Adapter YAML schema breaking-change migration playbook.
4. Per-artifact LLM-call budget defaults — Phase 1 calibration on sandbox corpus fixes the concrete numbers.
5. Incremental synthesis strategy — Phase 1 prototype informs Phase 2 defaults.
6. `sme_persona_consistency` LLM-judge — local small model vs. gateway call. Decided when the metric is implemented in Phase 0.
7. Synthesis-failure mode on low-quality corpora: mark-and-fire vs. retry; operator-configurable; default = mark + fire, do not block production.

## 15. File touch list

### New files
```
src/intelligence/sme/__init__.py
src/intelligence/sme/synthesizer.py
src/intelligence/sme/dossier_builder.py
src/intelligence/sme/insight_index_builder.py
src/intelligence/sme/comparative_register_builder.py
src/intelligence/sme/kg_materializer.py
src/intelligence/sme/recommendation_bank_builder.py
src/intelligence/sme/adapter_loader.py
src/intelligence/sme/storage.py              # Blob + Qdrant + Neo4j persistence for artifacts
src/intelligence/sme/verifier.py             # SMEVerifier
src/intelligence/sme/trace.py                # synthesis + query trace writers
src/retrieval/sme_retrieval.py               # query-time SME artifact retrieval layer
src/retrieval/hybrid_search.py               # dense+sparse RRF fusion helper
src/retrieval/reranker.py                    # cross-encoder wiring (may replace partial existing)
src/tools/url_fetcher.py                     # SSRF-safe HTTP fetcher
src/tools/url_ephemeral_source.py            # URL-in-prompt ephemeral pipeline
scripts/mine_sme_patterns.py                 # monthly pattern mining job
scripts/reindex_qdrant_sparse.py             # one-time sparse vector re-index
deploy/sme_adapters/defaults/                # default YAMLs + prompts (uploaded to Blob at deploy)
tests/sme_evalset_v1/                        # versioned eval set
tests/intelligence/sme/                      # per-builder unit + integration tests
```

### Modified files
```
src/generation/prompts.py                    # add diagnose/analyze/recommend, rich templates, compact override, persona injection
src/agent/core_agent.py                      # wire URL-as-prompt + SME retrieval + enriched pack
src/execution/router.py                      # compact-command detection, URL-primary vs supplementary routing
src/serving/model_router.py                  # extend intent classifier with new labels + format_hint
src/retrieval/unified_retriever.py           # implement layers 2 (KG) and 3 (SME artifacts)
src/intelligence/profile_builder.py          # extend domain detection; persist profile_domain
src/api/profiles_api.py                      # PATCH profile_domain; admin adapter CRUD endpoints
src/api/pipeline_api.py                      # training stage gains SME synthesis as final link before TRAINING_COMPLETED
src/kg/retrieval.py                          # include synthesized edges with confidence filter
```

### Coexists, unchanged
```
src/intelligence/profile_builder.py          # (existing outputs preserved; new fields added)
src/intelligence/collection_insights.py
src/intelligence/profile_intelligence.py
src/intelligence/generator.py                # explicitly NOT modified for formatting (memory rule)
```

## 16. Acceptance criteria

- All architectural invariants (Section 3) hold under integration tests
- All launch-gate conditions (Section 10) pass on the eval set
- No `pipeline_status` string added; `PIPELINE_TRAINING_COMPLETED` semantics preserved
- `generic.yaml` adapter produces functioning SME responses on a profile with no recognized domain
- Cross-subscription query isolation test passes (attempt to retrieve another subscription's artifacts rejected)
- URL-as-prompt passes SSRF test suite (private-IP, redirect-to-private, oversize, slow-loris)
- Adapter hot-swap demonstrated: update `finance.yaml` in Blob, admin invalidate, next query uses new version; old synthesis traces still reference old version
- Rollback demonstrated per phase

---

*End of design spec. Implementation plan to be produced next via the writing-plans skill.*
