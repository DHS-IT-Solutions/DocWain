# DocWain Insights Portal — Design Specification

- **Date:** 2026-04-25
- **Branch target:** `preprod_v03` (created off `preprod_v02` after the implementation plan is approved)
- **Status:** Draft for user review
- **Predecessor specs:**
  - `2026-04-20-docwain-profile-sme-reasoning-design.md` — SME persona scaffolding (predates this; partially superseded)
  - `2026-04-23-extraction-accuracy-design.md` — extraction overhaul (foundation; referenced, not redesigned here)
  - `2026-04-24-kg-training-stage-background-design.md` — KG consolidation (referenced; this spec writes to KG via the same training-stage entry)
  - `2026-04-24-unified-docwain-engineering-layer-design.md` — Researcher Agent v1 (this spec extends it to v2)

## 0. One-line summary

DocWain shifts from a Q&A response shape to a **document-grounded, domain-aware, proactive insights portal**. All intelligence is precomputed at ingestion and served via lookup-only endpoints. Domain behavior is plugin-shaped (Azure Blob YAMLs); a generic adapter always works. Every insight cites doc evidence. Big-bang ship, flag-by-flag enable; no regression to existing flow when all flags are off.

## 1. Goals

1. Make DocWain feel like a **research portal**, not a chatbot. The user opens the product and sees prepared insights about their documents — they don't have to ask first.
2. Cover all 29 capabilities marked MUST in the 2026-04-25 brainstorming session — typed insights, agentic actions, supplementary world-knowledge, multi-trigger proactivity, auto-domain handling, extraction foundation, dashboard surface.
3. Domain-agnostic by default; insurance, medical, procurement, HR, legal, banking, finance, expenses, contracts, resumes, etc. all work via the same plugin-shaped adapter framework. Generic adapter must produce useful output on unknown domains.
4. Every output is **document-grounded with citation evidence**. World knowledge is supplementary (used to interpret), never primary (used to fabricate).
5. Agentic action layer — DocWain composes artifacts, drafts forms, schedules follow-ups — but never takes irreversible external action without explicit user confirmation.
6. **Single-flag revertible per capability.** Big-bang merge is acceptable; big-bang enablement is not. Each capability lights up only after its eval gate passes.

## 2. Non-Goals

- Replacing `/api/ask`. Chat remains a first-class surface; insights enrich it, they don't displace it.
- Building unsanctioned external lookups. World knowledge is limited to per-adapter sanctioned KBs declared in the YAML.
- Re-architecting extraction. Extraction overhaul is a separate workstream (`2026-04-23-extraction-accuracy-design.md`); this spec consumes its output.
- Re-architecting the chat reasoner. Response composition stays in `src/generation/prompts.py` per `feedback_prompt_paths.md`; we add an injection point for proactive insights, not rewrite the reasoner.
- Customer data in any adapter or KB training. All adapter examples are synthetic per `feedback_no_customer_data_training.md`.
- Adapting the Teams app or other downstream surfaces in this spec. Teams remains isolated (`feedback_teams_isolation.md`); it can later consume the new endpoints, out of scope here.

## 3. Hard Constraints

| # | Constraint | Source |
|---|---|---|
| C1 | With **all feature flags off**, behavior on `preprod_v03` is byte-identical to `preprod_v02`. | User direction 2026-04-25 |
| C2 | **No query-time latency increase.** p50 and p95 of `/api/ask` unchanged or better with all insight-injection flags on. | User direction; `feedback_no_timeouts.md` |
| C3 | **No ingestion-latency increase to user-blocking gates.** Time-to-extraction-completion and time-to-HITL-screening-ready unchanged. New work runs on isolated queues, post-screening. | User direction 2026-04-25 |
| C4 | **Per-capability feature flag + eval gate** for all 29 capabilities. Single-flag revertible. | `feedback_intelligence_rag_zero_error.md` |
| C5 | **Document-grounded.** Insights with zero doc-evidence spans are rejected at the persistence layer. | User direction |
| C6 | **Domain-agnostic + open-set.** No closed enum of domains in core. Generic adapter must handle unknown domains. | `feedback_domain_extensibility.md` |
| C7 | **Adapters and referenced templates live in Azure Blob.** Hot-swap, per-subscription override. | `feedback_adapter_yaml_blob.md` |
| C8 | **Storage separation.** Insights persist to Qdrant payload + Neo4j Insight nodes + Mongo control-plane index only. No document content in Mongo. | `feedback_storage_separation.md` |
| C9 | **No customer data in training of any model or adapter.** | `feedback_no_customer_data_training.md` |
| C10 | **No Claude / Anthropic / Co-Authored-By in commits, code, or docs.** | `feedback_no_claude_attribution.md` |
| C11 | **HITL gates honored.** Researcher v2 runs only after screening approval. Continuous-refresh respects the same gate. | `feedback_pipeline_flow.md` |
| C12 | **MongoDB pipeline_status strings immutable.** New researcher v2 statuses use new field names; existing statuses untouched. | `feedback_mongo_status_stability.md` |

## 4. Layered Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Layer 5 — Surface                                                      │
│   /api/ask (proactive injection)                                        │
│   /api/profiles/{id}/insights         /api/profiles/{id}/actions        │
│   /api/profiles/{id}/insights/{iid}   /api/profiles/{id}/actions/.../execute │
│   /api/profiles/{id}/visualizations   /api/profiles/{id}/artifacts      │
│   /api/profiles/{id}/refresh-status                                     │
└──────────────────────────────────▲──────────────────────────────────────┘
                                   │ (lookup only — no compute)
┌──────────────────────────────────┴──────────────────────────────────────┐
│  Layer 4 — Insight Store                                                │
│   Qdrant payload  ←→  Neo4j Insight nodes  ←→  Mongo control-plane index│
└──────────────────────────────────▲──────────────────────────────────────┘
                                   │
┌──────────────────────────────────┴──────────────────────────────────────┐
│  Layer 3 — Continuous-Refresh Layer                                     │
│   on-upload incremental  /  scheduled weekly  /  watchlist signals      │
│   (Celery: researcher_refresh_queue, low priority)                      │
└──────────────────────────────────▲──────────────────────────────────────┘
                                   │
┌──────────────────────────────────┴──────────────────────────────────────┐
│  Layer 2 — Researcher Agent v2                                          │
│   detect domain → load adapter → run typed-insight passes → emit        │
│                                                                         │
│   Insight types: anomaly · gap · comparison · scenario · trend          │
│                  recommendation · conflict · projection · next-action   │
│                                                                         │
│   Plus: agentic action suggestions (artifact / form / alert / plan)     │
└──────────────────────────────────▲──────────────────────────────────────┘
                                   │
┌──────────────────────────────────┴──────────────────────────────────────┐
│  Layer 1 — Domain Adapter Framework                                     │
│   YAML in Azure Blob: sme_adapters/global/{domain}.yaml                 │
│                       sme_adapters/subscription/{sub_id}/{domain}.yaml  │
│   Adapter declares: insight prompts per type, sanctioned KBs,           │
│   watchlists, action templates, visualization choices.                  │
│   Generic adapter is the always-safe default.                           │
└──────────────────────────────────▲──────────────────────────────────────┘
                                   │
┌──────────────────────────────────┴──────────────────────────────────────┐
│  Layer 0 — Foundation (existing / separate workstreams)                 │
│   Extraction + DocIntel (in-extraction ML/DL) — already in roadmap      │
│   Embeddings → Qdrant                                                   │
│   Knowledge Graph → Neo4j (training-stage background per kg spec)       │
└─────────────────────────────────────────────────────────────────────────┘
```

**Layer responsibilities are independent.** Each layer has a defined input/output contract; consumers do not read internals. A change to one layer's internals (e.g., a different LLM in Researcher v2) does not require changes elsewhere.

## 5. Data Model

### 5.1 Insight schema (canonical type)

```jsonc
{
  "insight_id": "uuid",
  "profile_id": "string",
  "subscription_id": "string",
  "document_ids": ["string", "..."],            // 1+ docs this insight is grounded in
  "domain": "string",                            // adapter id used; "generic" allowed
  "insight_type": "anomaly|gap|comparison|scenario|trend|recommendation|conflict|projection|next_action",
  "headline": "string (≤ 25 words)",
  "body": "string (≤ 600 chars, markdown allowed)",
  "evidence_doc_spans": [                        // REQUIRED, ≥1 entry; reject if empty
    { "document_id": "string", "page": 0, "char_start": 0, "char_end": 0, "quote": "string" }
  ],
  "external_kb_refs": [                          // optional; sanctioned KBs only
    { "kb_id": "string", "ref": "string", "label": "string" }
  ],
  "confidence": 0.0,                             // 0..1
  "severity": "info|notice|warn|critical",       // for sorting / dashboard pinning
  "suggested_actions": ["action_id", "..."],     // links to Layer-2 emitted actions
  "tags": ["string"],
  "created_at": "iso8601",
  "refreshed_at": "iso8601",
  "stale": false,
  "adapter_version": "string",                   // hash of YAML used
  "feature_flags": ["string"]                    // flags this insight depends on for visibility
}
```

**Persistence rule.** An insight with `evidence_doc_spans = []` is rejected at the writer (`InsightStore.write`). This is a hard constraint, not a warning.

**Storage:**
- **Qdrant:** the insight is also embedded (headline + body) and stored as a row in the `insights` collection with payload `{insight_id, profile_id, subscription_id, document_ids, insight_type, severity, tags}`. Used for retrieval-time lookup.
- **Neo4j:** `(:Insight {insight_id, headline, severity, insight_type})` linked to `(:Document)` via `[:GROUNDED_IN]` and to `(:Profile)` via `[:OF_PROFILE]`.
- **Mongo (control plane):** `insights_index` collection — `{profile_id, insight_id, insight_type, severity, refreshed_at, stale}` only. No content. Used for fast list queries on the dashboard endpoint.

### 5.2 Action schema

```jsonc
{
  "action_id": "uuid",
  "profile_id": "string",
  "subscription_id": "string",
  "domain": "string",
  "action_type": "artifact|form_fill|alert|plan|reminder",
  "title": "string",
  "description": "string",
  "preview": "string (≤ 1KB; what the action would produce)",
  "requires_confirmation": true,                 // default true; only side-effect-free actions can be false
  "produces_artifact": false,
  "artifact_template": "string|null",            // pointer to Blob template
  "input_schema": { /* JSON schema of inputs */ },
  "executed_at": "iso8601|null",
  "execution_status": "pending|executed|failed|cancelled",
  "audit_trail": [{"actor":"...", "at":"...", "what":"..."}]
}
```

**Action constraint.** Any action that produces side-effects outside DocWain (sending email, scheduling external calendar, etc.) **MUST** have `requires_confirmation: true` and **MUST** route through a human-confirmable execute step. v1 of this spec ships only side-effect-free actions (artifact generation, form pre-fill, in-system reminders). External side-effect actions are declared in adapter but disabled at the execute layer behind a separate flag.

### 5.3 Adapter YAML schema

```yaml
# sme_adapters/global/{domain}.yaml
name: insurance                          # adapter id
version: "1.0"
description: "Insurance policies, claims, coverage analysis"
applies_when:                            # auto-detect rules
  domain_classifier_labels: [insurance, policy]
  doc_type_hints: [policy_document, certificate_of_insurance, claim_form]
  keyword_evidence_min: 3
  keywords: [policyholder, deductible, premium, sum insured, coverage, exclusion]

researcher:                              # one entry per insight type the adapter wants emitted
  insight_types:
    anomaly:
      prompt_template: "prompts/insurance_anomaly.md"
      enabled: true
    gap:
      prompt_template: "prompts/insurance_gap.md"
      enabled: true
    comparison:
      prompt_template: "prompts/insurance_comparison.md"
      enabled: true
      requires_min_docs: 2
    scenario:
      prompt_template: "prompts/insurance_scenario.md"
      enabled: true
    recommendation:
      prompt_template: "prompts/insurance_recommendation.md"
      enabled: true
    # ... trend / conflict / projection / next_action / ...

knowledge:
  sanctioned_kbs:
    - kb_id: insurance_taxonomy_v1
      ref: "blob://kbs/insurance_taxonomy_v1.json"
      describes: "Common policy types, coverage categories, exclusion classes"
  citation_rule: "doc_grounded_first"     # insights must cite ≥1 doc span; KB refs optional

watchlists:
  - id: renewal_due
    description: "Policy renewal date within 60 days"
    eval: "expr:doc.policy_end_date - now < 60d"
    fires_insight_type: next_action

actions:
  - action_id: generate_coverage_summary
    title: "Generate coverage summary PDF"
    action_type: artifact
    artifact_template: "templates/insurance_coverage_summary.md"
    requires_confirmation: false           # safe; just produces a PDF
  - action_id: draft_claim_letter
    title: "Draft claim letter"
    action_type: artifact
    artifact_template: "templates/insurance_claim_letter.md"
    requires_confirmation: false

visualizations:
  - viz_id: coverage_comparison_table
    insight_types: [comparison]
  - viz_id: premium_timeline
    insight_types: [trend]
```

The **generic adapter** has the same shape, with insight_type prompts that work on any domain (no domain-specific reasoning). It is the always-safe fallback.

### 5.4 Knowledge schema

KBs are declared as static blob references in the adapter. v1 ships read-only JSON KBs. KBs are versioned (`kb_id` includes version); upgrading a KB is a new file + adapter update, not in-place edit. KB content is never used to fabricate facts; only to interpret document content (e.g., mapping ICD-10 codes that appear in the doc to human-readable conditions).

## 6. Domain Adapter Framework

### 6.1 Resolution order

1. `sme_adapters/subscription/{subscription_id}/{domain}.yaml` (per-tenant override)
2. `sme_adapters/global/{domain}.yaml`
3. `sme_adapters/global/generic.yaml` (always succeeds)

### 6.2 Auto-detection

Per document, the existing `domain_classifier.py` returns a label + confidence. The adapter framework uses:
- If `confidence >= 0.7` → use that domain's adapter.
- If `confidence < 0.7` → fall back to generic.
- Multiple high-confidence labels → multi-domain document; researcher v2 runs the union of insight passes from each adapter, deduplicated by insight_type.

### 6.3 Multi-domain profiles

A profile may have docs in different domains (e.g., medical + insurance + tax). Researcher v2 runs per-document with the doc's adapter. Profile-level cross-doc passes (comparison, conflict, trend, projection) run with the **profile's dominant domain adapter** for prompts but consider all docs in evidence. If multi-domain (no clear dominant), profile-level passes use the generic adapter.

### 6.4 Hot-reload

`AdapterStore` caches loaded YAMLs in-memory with 5-min TTL. Admin API `POST /admin/adapters/invalidate?domain=X` flushes a domain. Each load logs `{adapter_id, version, content_hash}` for audit.

### 6.5 Failure mode

If Blob fetch fails: serve the last cached version. If no cache: fall back to `generic`. Generic must always work. Surface the failure on `/health/adapters`.

## 7. Researcher Agent v2

### 7.1 Loop

Per document, post-screening:

1. **Detect domain** — call `domain_classifier`, choose adapter.
2. **Load adapter** — `AdapterStore.get(domain)` (cached).
3. **For each insight_type enabled in the adapter**, run a typed insight pass:
    - Build prompt from `insight_types.{type}.prompt_template` + extracted document text.
    - Send to vLLM gateway (consistent with `feedback_prompt_paths.md` — researcher path uses its own prompts module, separate from chat reasoner).
    - Parse structured response → list of insights.
    - Validate each insight: `evidence_doc_spans` non-empty, citation references real spans in the doc.
    - Reject any insight failing validation, log to `researcher_rejected_insights` (Mongo, sample for audit).
4. **Emit suggested actions** — adapter's `actions` list scored against detected insights; surface those whose `applies_when` predicates match.
5. **Persist** to Insight Store (Layer 4).

Profile-level passes (comparison, conflict, trend, projection) run **after all per-doc passes complete** for that profile. They consume per-doc insights + chunks as evidence.

### 7.2 Idempotency

Each insight has a deterministic `dedup_key` derived from `(profile_id, document_ids[], insight_type, headline_hash)`. On re-run, identical insights upsert (no duplicates). Adapter version change forces re-run for the affected `insight_type`.

### 7.3 Isolation

Researcher v2 writes ONLY to:
- `researcher_v2.*` field in Mongo per-doc record (status, last_run, adapter_version)
- `insights_index` Mongo collection
- `insights` Qdrant collection
- `:Insight` Neo4j nodes

It NEVER touches `pipeline_status`, `stages.*`, or existing `researcher.*` (v1 field) — keeping `feedback_mongo_status_stability.md` honored. v1 researcher payload is migrated to v2 in a one-time backfill task; both can coexist behind flags during rollout.

### 7.4 Latency budget per document

- Per-doc insight passes: **30s soft cap** (no wall-clock kill — see `feedback_no_timeouts.md`); LLM calls budget allocated per insight_type.
- Profile-level passes: **2 min soft cap** for profiles up to 50 docs.
- Both run on `researcher_v2_queue` Celery, isolated from `extraction_queue`, `embedding_queue`, `kg_queue`.

## 8. Knowledge Layer

- KBs declared per adapter (Section 5.3).
- A `KnowledgeProvider` interface loads a KB from Blob, exposes lookup methods (`lookup(term)`, `interpret(value)`).
- Researcher v2 prompts may reference `{{kb.lookup(...)}}` template directives, resolved before LLM call.
- KB references in an insight populate `external_kb_refs[]`.
- **Hard rule (citation-required).** An insight cannot have `external_kb_refs` without also having `evidence_doc_spans`. Doc evidence is required; KB refs are augmentation.
- **Hard rule (separation, OQ1).** External references are **cited, not mixed into document-content claims**. The `body` field of an insight contains only statements derivable from `evidence_doc_spans`. KB references appear only in the `external_kb_refs[]` metadata array, intended for the consumer to render distinctly (e.g., "References" footer), never interleaved with body text. Every insight-type prompt explicitly instructs the model to keep KB-derived interpretation out of the body. The InsightStore writer rejects any insight whose body text references KB content the doc-spans don't support.
- KBs are static, not dynamic. v1 ships small bundled KBs (insurance taxonomy, ICD-10 subset, common HR policies, common procurement terms). Larger / live KBs (drug-interactions, market-policy-prices) are out of scope for v1; declared as future adapter extensions.

## 9. Continuous Re-analysis (Layer 3)

### 9.1 Three triggers

1. **On-upload incremental.** When a new doc lands and is screened, only insights with `document_ids` containing the new doc, plus profile-level passes that depend on the new doc's domain, refresh. Other insights are untouched.
2. **Scheduled weekly cross-profile pass.** A Celery beat task runs weekly per active profile, replays profile-level passes with the latest data. Active = profile had a query or upload in the last 90 days.
3. **Watchlist signals.** Each adapter declares watchlists (Section 5.3). A scheduler evaluates watchlists nightly per profile. Fires emit `next_action` insights and optionally Layer-2 actions.

### 9.2 Queue isolation

All refresh work runs on `researcher_refresh_queue`, separate from user-facing queues. Worker concurrency: 2 (configurable). Priority: low. If GPU/CPU is busy serving user traffic, refresh waits.

### 9.3 Staleness

Insights have `stale: bool` flag. A scheduled job marks stale = true when underlying docs change but the insight has not yet re-run. Surface layer surfaces stale insights with a "refreshing..." indicator; never silently shows outdated data without flagging it.

## 10. Surface Layer — Endpoints

All endpoints are flag-gated. Return 404 (with `Feature not enabled`) when flag is off. All read-only except `actions/.../execute`.

### 10.1 `GET /profiles/{profile_id}/insights`

**Query params:**
- `insight_type` (optional, comma-sep)
- `severity` (optional, comma-sep)
- `domain` (optional)
- `since` (optional, iso8601 — only insights refreshed after)
- `limit` (default 50, max 200), `offset`

**Response:**
```json
{
  "profile_id": "...",
  "total": 0,
  "stale_count": 0,
  "insights": [/* Insight objects, summary form: insight_id, headline, type, severity, refreshed_at */],
  "domains_present": ["medical", "insurance"],
  "last_refresh": "iso8601"
}
```

Backed by Mongo `insights_index` for the list (fast); detail fetch is a separate endpoint.

### 10.2 `GET /profiles/{profile_id}/insights/{insight_id}`

Returns the full Insight object (Section 5.1).

### 10.3 `GET /profiles/{profile_id}/actions`

Lists agentic actions available for the profile. Returns Action objects in summary form (id, title, type, requires_confirmation, preview).

### 10.4 `POST /profiles/{profile_id}/actions/{action_id}/execute`

Body:
```json
{ "inputs": { /* matches action's input_schema */ }, "confirmed": true }
```

If `requires_confirmation: true` and `confirmed != true` → 400 with the preview. Otherwise execute, return artifact ref (Blob URL) or status.

### 10.5 `GET /profiles/{profile_id}/visualizations`

Lists visualization specs per profile, each with viz_id, type, source insight_ids, and a JSON data payload ready for the frontend. v1 ships: timeline, comparison_table, trend_chart. Visualization data is precomputed at insight write time.

### 10.6 `GET /profiles/{profile_id}/artifacts`

Lists generated artifacts (from past `actions/.../execute` runs). Each entry: `{artifact_id, action_id, blob_url, generated_at, expires_at}`.

### 10.7 `GET /profiles/{profile_id}/refresh-status`

Returns the state of continuous-refresh work for the profile: last on-upload refresh, last scheduled run, pending watchlist evaluations, stale-insight count.

### 10.8 `/api/ask` proactive injection

Existing endpoint, surface enhanced. After the reasoner produces its answer to the user's question, a post-step pulls top-N insights for the profile (filtered by relevance to the asked entities/topics) and appends a "Related findings" section. The reasoner is NOT replaced. The proactive section is rendered in `src/generation/prompts.py` (per `feedback_prompt_paths.md`); the insight retrieval is lookup-only against `insights_index`.

**Always-on once the flag is enabled (OQ4).** No per-user opt-out in v1. Goal is maximum proactive injection — every answer carries proactive findings whenever relevant insights exist for the profile, subject only to the 50ms / no-LLM injection budget (Section 13.2). Severity filtering (`notice`+) and relevance filtering (must match query entities or topic) are quality guards; they reduce *what* gets injected but never disable the injection step itself.

Behind flag `INSIGHTS_PROACTIVE_INJECTION` — when off, `/api/ask` returns the existing response unchanged.

## 11. Agentic Action Layer

### 11.1 Contract

An action is declared in adapter YAML, registered at startup. The execution path:

1. Caller hits `POST /profiles/{id}/actions/{action_id}/execute` with `{inputs, confirmed}`.
2. Action runner validates `inputs` against `input_schema`.
3. If `requires_confirmation` and not confirmed → return preview.
4. Otherwise, action handler runs:
    - **artifact**: render template with profile + insights + inputs as context, produce PDF/MD, upload to Blob.
    - **form_fill**: same as artifact but produces a structured form (PDF/JSON).
    - **alert**: write a watchlist alert to insights index.
    - **plan**: produce a checklist artifact + persist as `next_action` insights.
    - **reminder**: schedule via existing Celery beat (registers a one-time fire at a date).
5. Audit-log every execution to `actions_audit` Mongo collection.

### 11.2 Sandboxing

v1 actions are pure-function (no external side-effects). External-side-effect actions (email, calendar) are declared but disabled at the runner behind a separate flag `ACTIONS_EXTERNAL_SIDE_EFFECTS_ENABLED` — off by default. When eventually enabled, they go through a 2-step confirmation flow.

### 11.3 Idempotency

Each action execution gets an `execution_id`. Re-execution with the same `execution_id` (within 1h) returns the cached result. Prevents accidental double-fire from UI retries.

## 12. Extraction + DocIntel — Foundation Reference

This spec **consumes** the extraction overhaul output; it does not redesign it. The extraction overhaul is described in `2026-04-23-extraction-accuracy-design.md`. Key relevant outputs that this spec requires:

- Canonical extraction JSON in Azure Blob, structured (pages, blocks, tables, sheets, slides) — already exists.
- Per-doc `domain_classifier` label + confidence — already exists.
- DocIntel (in-extraction ML model) drives extraction quality but produces no separate persisted artifact (per `feedback_intelligence_precompute.md`).

If extraction quality is insufficient for a domain (e.g., medical scans), the corresponding adapter's eval gate will fail. We do not attempt to compensate at the researcher layer.

## 13. Latency Discipline

This section is **load-bearing for C2 + C3** (no query-time latency increase, no ingestion-blocking latency increase). Every design choice in this spec is checked against these rules:

### 13.1 What runs where

| Work | Queue | Triggers ingestion-blocking gate? |
|---|---|---|
| Extraction | `extraction_queue` (existing) | Yes (gates HITL screening) |
| Embedding | `embedding_queue` (existing) | Yes (gates queryability) |
| KG build | `kg_queue` (per `2026-04-24-kg-training-stage-background-design.md`) | No (background) |
| **Researcher Agent v2 — per-doc** | `researcher_v2_queue` (new) | **No (background, post-screening)** |
| **Researcher Agent v2 — profile-level** | `researcher_v2_queue` (same) | **No** |
| **Continuous-refresh — incremental** | `researcher_refresh_queue` (new) | **No** |
| **Continuous-refresh — scheduled / watchlist** | `researcher_refresh_queue` (same) | **No** |
| **Action execution** | `actions_queue` (new) | **No (user-initiated; not in upload path)** |
| **Insight retrieval at /api/ask** | inline lookup, no compute | Query-time path; must be ≤ 50ms p95 |

**Document is queryable as soon as embeddings complete.** Researcher v2 is additive: insights become available when they're ready, not as a precondition for query.

### 13.2 Query-time budget

`/api/ask` p95 with all flags on must equal `/api/ask` p95 with all flags off, ±5%. The proactive-injection step has a hard budget: **50ms** for insight lookup + **0 LLM calls** beyond what the reasoner already makes. Insight content is appended to the prompt context if budget allows; if not, skipped silently.

### 13.3 Ingestion-time budget

Time from upload-complete to HITL-screening-eligible is unchanged. This is enforced because researcher v2 starts AFTER screening approval, on its own queue. Screening eligibility depends only on extraction completion (existing behavior).

Time from screening approval to "first insight visible" — new metric, measured but not gated. Target p50 ≤ 2 min for a 5-doc profile.

### 13.4 No new synchronous network calls in critical path

- Adapter loads are cached (5-min TTL).
- KB lookups are local file reads (Blob fetched once at adapter load).
- LLM calls are queued, never inline in a user-facing handler.
- `/api/ask` injection reads from Mongo `insights_index` only — no Qdrant query in the injection step (injection uses entities already extracted by the existing query-understanding step to filter the index).

### 13.5 Latency tests in CI

A new perf test suite runs in CI:
- `tests/perf/api_ask_latency.py` — 100 runs, asserts p95 unchanged with flags on vs. off.
- `tests/perf/upload_to_screening_eligible.py` — synthetic upload, asserts time-to-screening-eligible unchanged.
- `tests/perf/insight_lookup_p95.py` — 1000 runs, asserts ≤ 50ms.

CI fails if any regresses by > 5%.

## 14. Per-Capability Feature Flags + Eval Gates

All 29 MUST capabilities, each with its own flag and gate. Big-bang merge, flag-by-flag enable.

### 14.1 Flag naming convention

`INSIGHTS_<area>_<capability>_ENABLED`. Stored in `src/api/config.py` with defaults `false`. Hot-reloadable via existing config admin path.

### 14.2 Gate definition rules

Each gate must be **mechanical** (pass/fail by script, no human judgment). Each capability has:
- A test fixture: synthetic profile + synthetic docs + expected output shape.
- A scoring rubric: precision/recall/coverage on a small held-out fixture.
- A baseline (measured before code lands; preprod_v02 is the baseline).
- A pass threshold (≥ baseline + delta, or ≥ absolute floor).

### 14.3 Capability table

| # | Capability (from brainstorm) | Flag | Gate (mechanical) |
|---|---|---|---|
| A1 | Anomaly / risk detection | `INSIGHTS_TYPE_ANOMALY_ENABLED` | ≥ 0.7 precision on 50 synthetic anomaly fixtures across 3+ domains |
| A2 | Gap analysis | `INSIGHTS_TYPE_GAP_ENABLED` | ≥ 0.6 recall on 30 synthetic gap fixtures |
| A3 | Cross-doc comparison | `INSIGHTS_TYPE_COMPARISON_ENABLED` | ≥ 0.7 precision on 20 multi-doc comparison fixtures |
| A4 | Scenario reasoning | `INSIGHTS_TYPE_SCENARIO_ENABLED` | ≥ 0.6 plausibility (rubric-scored) on 30 scenario fixtures |
| A5 | Trend / timeline | `INSIGHTS_TYPE_TREND_ENABLED` | ≥ 0.8 directionally-correct on 20 time-series fixtures |
| A6 | Recommendation | `INSIGHTS_TYPE_RECOMMENDATION_ENABLED` | ≥ 0.7 actionability rubric on 40 fixtures |
| A7 | Conflict detection | `INSIGHTS_TYPE_CONFLICT_ENABLED` | ≥ 0.8 precision on 20 fixtures with planted conflicts |
| A8 | Projection | `INSIGHTS_TYPE_PROJECTION_ENABLED` | within 15% of ground truth on 20 numeric fixtures |
| B9 | Generate structured artifact | `ACTIONS_ARTIFACT_ENABLED` | template renders cleanly on 10 fixtures, output passes JSON/MD validators |
| B10 | Pre-fill forms | `ACTIONS_FORM_FILL_ENABLED` | ≥ 90% field-fill accuracy on 10 fixture forms |
| B11 | Compose follow-up plan | `ACTIONS_PLAN_ENABLED` | rubric ≥ 0.7 on 10 fixtures |
| B12 | Schedule reminders / alerts | `ACTIONS_REMINDER_ENABLED` | reminder fires at scheduled time in 10/10 integration runs |
| C13 | Domain knowledge bases | `KB_BUNDLED_ENABLED` | KB load + lookup test passes; insights cite KB refs in ≥ 5 fixtures |
| C14 | External benchmark / market data | `KB_EXTERNAL_ENABLED` | **OFF for v1** — declared, not implemented |
| C15 | Citation model | `INSIGHTS_CITATION_ENFORCEMENT_ENABLED` | 100% of written insights have ≥ 1 doc-span; injection test of zero-citation insight is rejected |
| D16 | On-upload research | `REFRESH_ON_UPLOAD_ENABLED` | new doc lands → researcher kicked within 30s in 10/10 integration runs |
| D17 | On-query proactive insight | `INSIGHTS_PROACTIVE_INJECTION` | 20 fixture queries — injected section non-empty when relevant insights exist; latency C2 holds |
| D18 | Scheduled weekly re-analysis | `REFRESH_SCHEDULED_ENABLED` | beat fires for active profile, new insights upsert |
| D19 | On-change incremental refresh | `REFRESH_INCREMENTAL_ENABLED` | refresh touches only affected insights — verified by diff |
| D20 | Watchlist signals | `WATCHLIST_ENABLED` | 5 watchlists fire on synthetic data; non-firing conditions don't fire |
| E21 | Auto-detect domain | `ADAPTER_AUTO_DETECT_ENABLED` | classifier ≥ 0.85 accuracy on 100 mixed-domain fixtures |
| E22 | Plug-in adapters in Blob | `ADAPTER_BLOB_LOADING_ENABLED` | hot-swap test: new domain YAML uploaded → researcher uses it within TTL |
| E23 | Generic adapter always works | `ADAPTER_GENERIC_FALLBACK_ENABLED` | unknown-domain fixture → generic adapter produces ≥ 3 valid insights |
| F24 | DocIntel-driven extraction | (separate spec; flag tracked there) | (gate per extraction spec) |
| F25 | Per-doc-type extraction strategies | (separate spec) | (gate per extraction spec) |
| F26 | Continuous improvement loop | (separate spec) | (gate per extraction spec) |
| G27 | Visualizations | `VIZ_ENABLED` | viz endpoint returns valid JSON for 5 fixture profiles spanning 3+ domains |
| G28 | Structured artifacts | (covered by B9; same flag) | (same gate) |
| G29 | Insights dashboard endpoint | `INSIGHTS_DASHBOARD_ENABLED` | endpoint returns within p95 < 500ms over 100 runs on 50-doc profile |

**29 capabilities total.** F24/F25/F26 belong to the extraction spec; this spec depends on them but does not own their flags. The remaining 26 capabilities are owned here, behind 25 flags (G28 *Structured artifacts* shares the `ACTIONS_ARTIFACT_ENABLED` flag from B9 because they're the same code path).

### 14.4 Enablement order

A defensible order to enable flags in production:

1. **C15** (citation enforcement) — must be on before any other insight type
2. **E21, E22, E23** (adapter framework + auto-detect + generic fallback)
3. **A1, A6, A7** (anomaly, recommendation, conflict — highest value, easiest to evaluate)
4. **A2, A3, A4, A5, A8** (rest of insight types)
5. **C13** (bundled KBs)
6. **D16** (on-upload research)
7. **G29, G27** (dashboard, viz)
8. **D17** (proactive injection in /api/ask) — last, because it touches user-facing path
9. **D19, D18, D20** (incremental + scheduled + watchlist)
10. **B9, B10, B11, B12** (action types)

C14 and external-side-effect actions remain off in v1.

## 15. Existing Flow Preservation

### 15.1 The "all flags off" guarantee

A regression test suite asserts that with every flag off:
- `/api/ask` response is byte-identical to `preprod_v02` for a 50-fixture set.
- Upload → screening eligible time matches `preprod_v02` for 20 fixture documents within p95.
- All existing endpoints return identical responses on a 100-call replay test.
- No new Mongo collections are written to (only when respective flags fire).

Test: `tests/regression/all_flags_off.py`. CI gate: must pass before merge.

### 15.2 Per-flag enablement test

Each flag has a test that asserts: with only that flag on, behavior changes only in the documented way (e.g., a new endpoint returns 200 instead of 404). Existing endpoints unchanged.

## 16. Cost + Storage Budgets

| Resource | Current (preprod_v02) | After v1 enabled | Notes |
|---|---|---|---|
| Qdrant payload size | X | X + ~30% | Insights collection roughly 0.3× document collection size |
| Neo4j node count | Y | Y + ~5× per profile | Insight nodes per doc; insights deduped + stale-cleaned |
| Mongo collections | n | n + 4 | `insights_index`, `actions_audit`, `researcher_rejected_insights`, `watchlist_state` |
| GPU utilization (training queue) | Z | Z + 20% | Researcher v2 LLM calls; isolated queue |
| GPU utilization (user-facing) | W | W (unchanged) | Strict isolation per Layer 13 |

**Cap:** if Researcher v2 GPU utilization exceeds 30% sustained, queue concurrency drops automatically (autoscale rule). User-facing inference always wins.

## 17. Rollout / Migration

### 17.1 Branch + merge strategy

- New long-lived branch `preprod_v03` cut off `preprod_v02` at the time the implementation plan is approved (mirrors the `preprod_v02` pattern in `project_preprod_v02_branch.md`).
- The 26 capabilities owned by this spec are implemented as PR-sized chunks (one per sub-project in Section 19), each PR landing on `preprod_v03` directly. No feature branches off `preprod_v03` unless a chunk needs more than ~600 LOC.
- Each PR includes its capability's eval-gate test and is required to pass that gate plus the all-flags-off regression suite (Section 15.1) before merge.
- Once all 26 gates pass and `preprod_v03` is stable for one cycle, `preprod_v03` becomes the next production baseline (replacing `preprod_v02`), at which point production flags are enabled in the order in Section 14.4, one at a time, with a monitoring window per flag.

### 17.2 Backfill of existing profiles

Once `RESEARCHER_V2_ENABLED` is on, a one-time backfill task replays researcher v2 for all existing profiles, on `researcher_v2_queue`, low priority. Backfill is interruptible and idempotent (Section 7.2).

### 17.3 Coexistence with researcher v1

v1 (`src/tasks/researcher.py`) keeps running until v2 reaches feature parity behind `RESEARCHER_V1_ENABLED` (default true). Once v2 is fully enabled in prod and stable for 2 weeks, v1 is deprecated and removed in a follow-up PR.

### 17.4 Rollback

Any flag can be flipped off independently. Each flag's effect is fully reversible (no destructive schema changes). Removed insights remain in Qdrant/Neo4j (cheap to keep, reusable on re-enable).

## 18. V5 Lesson Mapping

`feedback_v5_failure_lessons.md` enumerates 8 hard rules from the V5 failure. Mapping how this design honors each:

| V5 rule (paraphrased) | How honored here |
|---|---|
| Gate distillation on teacher identity | Each capability has its own mechanical gate. No promotion of any flag without its gate passing. |
| Validate scorers first | All gate scorers (Section 14.3) implemented + validated against synthetic baseline before any capability code lands. |
| Don't advance on failed gates | Section 14.4 enablement order is strictly sequential per flag; flag does not enable until its gate passes. |
| Synthetic-data-only for training | This spec adds no model training. Adapter examples are synthetic. |
| Pipeline isolation | New queues `researcher_v2_queue`, `researcher_refresh_queue`, `actions_queue` isolate from existing. |
| Single-flag revert | Each capability is single-flag. Flag off = behavior reverts. |
| Measure before change | Baselines for all 29 capabilities measured before code lands. |
| No "fix in next PR" | Each capability is tested + gated within its own PR. No ungated lands. |

## 19. Sub-Project Decomposition (within big-bang)

Even though this is big-bang merge, the implementation plan (next step) divides the work into PR-sized sub-projects:

1. **SP-A — Adapter Framework** (`src/intelligence/adapters/`, AdapterStore, generic adapter, auto-detect wiring)
2. **SP-B — Insight Schema + Store** (`src/intelligence/insights/`, InsightStore, validators, citation enforcement)
3. **SP-C — Researcher Agent v2** (`src/tasks/researcher_v2.py`, `src/intelligence/researcher_v2/`, all 9 insight types, prompts module)
4. **SP-D — Knowledge Layer** (`src/intelligence/knowledge/`, KnowledgeProvider, bundled KBs)
5. **SP-E — Continuous Refresh** (`src/tasks/researcher_refresh_v2.py`, beat schedules, watchlist evaluator)
6. **SP-F — Surface Endpoints** (new endpoints in `src/api/insights_api.py`, `actions_api.py`, `visualizations_api.py`, `artifacts_api.py`)
7. **SP-G — /api/ask injection** (modification in `src/generation/prompts.py` + retrieval helper)
8. **SP-H — Agentic Action Layer** (`src/intelligence/actions/`, action runner, audit, sandbox)
9. **SP-I — Visualizations** (`src/intelligence/visualizations/`, viz spec generation at insight-write time)
10. **SP-J — Per-capability flags + eval harness** (`src/api/config.py`, `tests/insights_eval/`)
11. **SP-K — Regression + perf tests** (`tests/regression/`, `tests/perf/`)
12. **SP-L — Backfill + migration tooling** (`scripts/insights_backfill.py`, runbook)

Each SP has its own gate; SPs A→J in dependency order; K and L touched throughout.

## 20. Risks + Open Questions

### 20.1 Known risks

- **Insight quality bounded by extraction quality.** If extraction overhaul lags, some adapters' gates may fail. Mitigation: extraction overhaul is a parallel workstream; this spec ships generic + 3-domain proof, then waits for extraction overhaul before enabling more.
- **LLM cost.** Researcher v2 makes more LLM calls than v1 (one per insight type vs. one per doc). Mitigation: Section 16 cap; vLLM-primary serving; insights cached.
- **Adapter sprawl.** Adding 5 domains × 9 insight types = 45 prompts. Mitigation: shared prompt scaffolding; insight_types may inherit a generic template and override only the domain-specific cues.
- **Multi-domain profiles edge cases.** A profile with 3 domains may produce duplicate / overlapping insights. Mitigation: dedup by `dedup_key` in Section 7.2; profile-level passes use dominant or generic adapter.
- **Silent injection in `/api/ask`.** Bad injection content could degrade chat quality. Mitigation: injection content is severity-filtered (only `notice`+ severity); off behind flag; per-call injection budget (50ms, no LLM); A/B-able.

### 20.2 Open questions — resolved 2026-04-25

- **OQ1 — RESOLVED.** External references may be cited via `external_kb_refs[]` but **must not be mixed into document-content claims**. The body of an insight contains only document-grounded statements; KB references are separate metadata for the consumer to render distinctly (e.g., as a "References" footer, not interleaved with the body). The LLM prompt for every insight type explicitly instructs separation. The writer enforces structural separation: if the body text contains content not derivable from `evidence_doc_spans`, the insight is rejected.
- **OQ2 — RESOLVED.** Yes — v1 ships side-effect-free actions only (artifacts, form-fill, plans, in-system reminders). External-side-effect actions (email send, calendar push) are declared in adapter but disabled at runner behind `ACTIONS_EXTERNAL_SIDE_EFFECTS_ENABLED` (off in v1).
- **OQ3 — RESOLVED.** Dashboard endpoint shape in Section 10.1 stands as drafted.
- **OQ4 — RESOLVED.** `/api/ask` proactive injection is **always-on once the flag is enabled**. No per-user opt-out in v1. Goal is maximum proactive injection — every answer carries proactive findings whenever relevant insights exist, subject only to the 50ms / no-LLM injection budget (Section 13.2). Severity filtering (`notice`+) is preserved as a quality guard, not as an opt-out.
- **OQ5 — RESOLVED.** Watchlist cadence stays nightly across all adapter declarations. No watchlist requires higher frequency in v1.

### 20.3 Out of scope for v1

- External market-data lookups (`C14`).
- External-side-effect actions (email, calendar push).
- Live KBs (`C13` only ships static bundled KBs).
- Frontend / dashboard UI build (this spec ships endpoints; UI is a follow-on).
- Teams app integration (separate workstream).
- Multi-tenant adapter override UI (Blob upload by ops only in v1).
