# Intelligence Layer + RAG Re-integration — Design Spec

**Date:** 2026-04-21
**Owner:** Muthu
**Status:** Approved in brainstorming; pending spec review before implementation plan
**Supersedes (partial):** `2026-04-01-docwain-intelligence-upgrade-design.md`, `2026-04-07-latency-and-expert-intelligence-design.md` (for the parts that discussed fast/smart split and layered RAG)
**Related memory:** `feedback_unified_model`, `feedback_measure_before_change`, `feedback_intelligence_precompute`, `feedback_prompt_paths`, `feedback_engineering_first_model_last`, `feedback_adapter_yaml_blob`, `feedback_v5_failure_lessons`

## 1. Context

Yesterday (2026-04-20) the intelligence layer was producing accurate, grounded, persona-consistent responses. Today (2026-04-21) responses are:

- Factually wrong / hallucinated
- Generic / ignoring the document context
- Missing DocWain persona and expected formatting

Between yesterday and today, two commits landed on `main`:

- `c849b04` — doc pipeline working end-to-end (status/progress/extraction/embedding)
- `b0c7211` — fix(pipeline): extraction latency + qdrant payload schema + progress uploaded-count

The qdrant payload-schema change in `b0c7211` is the primary suspect for the grounding/retrieval regression: the write side emits new payload keys, the read side (filters in `src/retrieval/*`, `src/intelligence/retrieval.py`, profile-query builders) may still reference the old keys.

Separately, the SME + 4-layer RAG workstream that was built on branch `sme-work-backup-2026-04-21` (50+ commits across phases 0–6) has not been integrated into `main`. The branch contains:

- Baseline eval harness with 10 metrics (phase 0)
- Flag-gated hybrid dense+sparse retrieval, cross-encoder reranker, adapter YAMLs, storage/trace writers, fail-closed verifier (phase 1)
- Dossier/insight/comparative/recbank synthesis, KG inferred-relation materializer, retrieval layers B & C (phase 2)
- 4-layer parallel orchestration, merge + MMR, pack assembly, adaptive top-K, Redis fast-path (phase 3)
- Extended intent classifier, rich prompt templates, adapter resolver wiring, grounding post-pass (phase 4)
- SSRF-safe URL adapter leg with ephemeral fetch+extract+chunk+embed (phase 5)
- Monthly pattern mining: clustering, report composer, stabilization-score trigger, systemd timer (phase 6)

There are also fast/smart split remnants in `src/serving/` (module `fast_path.py`, model name `docwain-fast`, docstrings referencing "14B fast" / "27B smart", default `route_taken="smart"`) despite the standing rule that DocWain is a unified model.

## 2. Goals

1. **Restore accuracy.** The "yesterday-good" quality returns.
2. **Unify the model path.** One DocWain model, one serving code path, no fast/smart split anywhere in live code.
3. **Re-integrate RAG.** The SME+RAG work from the SME branch lands on main in phased batches, every layer behind a feature flag, default OFF in prod until eval parity is proven.
4. **Protect the document-processing pipeline.** The recent pipeline fixes (extraction/embedding/qdrant write/mongo status) are untouched by any batch in this workstream.

## 3. Non-goals

- Model retraining / fine-tuning — this is an engineering-layer project per the "engineering first, model training last" rule.
- Changes to the write path (extraction, embedding, qdrant indexing, mongo status transitions).
- Cross-profile or tenant changes beyond what existing tests cover.
- Changes to Teams standalone service (fully isolated per its own memory rule).

## 4. Architecture

### 4.1 Unified model surface

One `VLLMManager` instance pointing at a single `docwain` model. No fast/smart distinction anywhere: docstrings, metric labels, default values, route-taken labels, config files.

`FastPathHandler` as a standalone module is deleted. Its two responsibilities — "no-retrieval for greeting/identity" and "thin retrieval for lookup/list/count" — become two small functions inside a single `IntelligenceHandler` that always uses the same model and the same generation prompts.

`IntentRouter` stays. Intent still drives retrieval strategy (KG expansion, viz, top-K) — just not model selection.

All generation prompts go through `src/generation/prompts.py` (per `feedback_prompt_paths` memory). `src/intelligence/generator.py` stops owning response formatting.

### 4.2 RAG stack (behind flags)

```
query
  │
  ▼
IntentRouter ── intent, complexity, requires_kg, requires_viz
  │
  ▼
RetrievalPlanner ── per-intent adaptive top-K, which layers to fire
  │
  ▼
4-layer parallel retrieval (each flag-gated):
  A. Dense+Sparse RRF over Qdrant       (flag: retrieval.hybrid)
  B. INFERRED_RELATION over Neo4j        (flag: sme.layer_b)
  C. SME artifact pack (dossier/insights/recs) (flag: sme.layer_c)
  D. Ephemeral URL leg                   (flag: sme.url_adapter)
  │
  ▼
Merge + cross-encoder blend + MMR         (flag: retrieval.rerank)
  │
  ▼
Pack assembly (adapter-budget-aware)      (flag: sme.rich_pack)
  │
  ▼
Grounding verifier (5-check, fail-closed)
  │
  ▼
Reasoner (src/generation) — single unified DocWain model
  │
  ▼
Response composer (citations, chart_spec, alerts)
```

Every flag defaults OFF on first merge. Dev flips on after eval parity; prod stays OFF until the whole stack is eval-green. Turning a flag off mid-incident is a single Redis/config write — no deploy.

### 4.3 Domain adapter plane

Adapter YAMLs live in Azure Blob (per `feedback_adapter_yaml_blob`): global default + per-subscription overrides, TTL-cached, hot-swappable, last-resort generic fallback. The reasoner reads adapter shape + grounding rules at query time, not at startup.

### 4.4 What stays the same

The document-processing pipeline (upload → extraction → embedding → qdrant write → mongo status) is untouched. No batch modifies:

- `src/api/extraction_service.py`, `src/api/extraction_pipeline_api.py`, `src/api/dw_document_extractor.py`
- `src/api/embedding_service.py`, `src/embed/`, `src/embedding/`
- `src/api/qdrant_indexes.py`, `src/api/qdrant_setup.py`, `src/api/vector_store.py`
- Celery worker definitions, systemd units for the pipeline
- Mongo status field names (immutable per the status-stability rule)

Only three SME cherry-picks touch `src/api/pipeline_api.py`, and only as additive hooks (invalidation on status flip, idempotent hash short-circuit, finalize-training gate).

## 5. Batch Plan

Eight batches, each an independent PR onto `main`. No batch merges until the previous has passed its eval gate (Section 6). SME branch is the source of truth; commits get cherry-picked, never force-merged.

| # | Name | Source | What lands | Flag state on merge |
|---|------|--------|------------|---------------------|
| **0** | Unified-model + qdrant-audit + regression fix | *new work on main* | Delete `src/serving/fast_path.py`; collapse into one `IntelligenceHandler`; rename `docwain-fast` → `docwain`; audit `src/retrieval/retriever.py` filter keys against the new qdrant payload schema from `b0c7211`; repoint generation prompt path through `src/generation/prompts.py` only | n/a (no feature flags yet) |
| **1** | Phase0 — eval baseline | SME phase 0 | 10 metrics, CLI orchestrator, JSONL result store, human-rating CSV export/import, runbook, fixed query set, baseline snapshot | n/a (eval offline-only) |
| **2** | Phase1 — scaffolds | SME phase 1 | Adapter YAML + Blob loader, admin endpoints, 5 skeleton builders, ABC + orchestrator skeleton, trace writers, verifier, flag resolver, hybrid RRF helper + cross-encoder reranker (flag-gated OFF), rolling sparse re-index script | all SME flags default **OFF** |
| **3** | Phase2 — synthesis | SME phase 2 | Dossier, insight index, comparative, KG INFERRED_RELATION materializer, recbank, incremental, retrieve_layer_b + retrieve_layer_c, control-plane helpers, finalize-gate, qa_idx emission + invalidation, real-builder e2e | flags still **OFF** |
| **4** | Phase3 — orchestration + cache | SME phase 3 | 4-layer parallel orchestration, merge + cross-encoder blend + MMR, pack assembly, adaptive top-K, Redis fast-path, intent gating, e2e integration tests | flags **OFF** in prod; **ON** in dev after eval parity |
| **5** | Phase4 — agent wiring | SME phase 4 | Extended intent labels + format_hint + URL detection, rich prompt templates + PackSummary, adapter resolver in agent, grounding post-pass, rich-mode consumer tests | flags **OFF** in prod |
| **6** | Phase5 — URL adapter leg | SME phase 5 | SSRF-safe URL fetcher, ephemeral fetch+extract+chunk+embed, case selector, citation annotation, parallel URL leg in core agent | URL feature **OFF** in prod |
| **7** | Phase6 — monthly pattern mining | SME phase 6 | Schema + trace loader + fingerprint + clustering, report composer, stabilization-score trigger, orchestrator + CLI, systemd monthly timer | monthly timer **disabled** on merge; enable via separate ops PR |

Batch 0 is not a cherry-pick — it is new work. Batches 1–7 are cherry-picks with conflict resolution forward onto main's current pipeline code.

## 6. Eval Gate & Rollback Contract

### 6.1 Per-batch exit criteria (blocking before merge)

| Batch | Required to pass before merge |
|---|---|
| **0** | Grep gate clean; qdrant retrieval integration test green; 10-query smoke test non-empty+grounded+DocWain-persona; all existing pipeline tests unchanged-green; `git diff --stat` touches zero files from the do-not-touch list in Section 4.4 |
| **1** | Baseline snapshot committed to `eval_results/baseline-YYYY-MM-DD.jsonl`; 10 metrics produce values on the canned query set; re-running the harness on the same inputs produces results within tolerance (see §6.1.1 below) |
| **2** | All new code flag-gated OFF; re-running the Batch 1 baseline harness produces results **within tolerance** of the baseline (proves the new code is inert when flags are OFF); unit tests for adapter loader, flag resolver, verifier, retrieval helpers all green |
| **3** | Same as Batch 2: flag-OFF baseline within tolerance; running the harness with `sme.layer_b=on` or `sme.layer_c=on` in a dev profile produces a scorecard delta doc (captured, not pass/fail) |
| **4** | Flag-OFF within tolerance. With `retrieval.hybrid=on` + `retrieval.rerank=on`: groundedness ≥ baseline − 0.02, hit-rate ≥ baseline − 0.02, latency p95 ≤ 1.5× baseline. Failure merges with flags off and becomes a follow-up issue — does not block merge, does block prod flag-on |
| **5** | Flag-OFF within tolerance; `format_hint` unit tests green; rich-mode consumer e2e smoke green |
| **6** | Flag-OFF within tolerance; SSRF block-list honored in malicious-URL unit test; URL leg off by default |
| **7** | Flag-OFF within tolerance; systemd timer **inactive** on merge; dry-run of mining CLI against small fixture produces report without errors |

The "flag-OFF within tolerance" gate is the single most important invariant. No matter how much code lands, the user experience cannot regress from what Batch 0 shipped unless a flag is flipped.

### 6.1.1 Tolerance definition

"Within tolerance" means:
- **Deterministic metrics** (hit-rate, verified-removal-rate, cross-doc-integration-rate, latency p50/p95/p99, RAGAS lexical components): **byte-identical** numbers. Any drift is a fail.
- **LLM-judge metrics** (groundedness, persona-consistency, recommendation-groundedness, insight-novelty): **|delta| ≤ 0.02** on a 0–1 scale, **and** the same LLM-judge model + temperature=0 + fixed seed on the harness run. Larger drift is a fail.
- **Path-equivalence check** (additional to numbers): in flag-OFF mode, the retrieval call trace must be equivalent to baseline — same number of qdrant calls, same filter shape, no new Redis gets, no new KG queries. A structural trace diff is captured per batch.

Without the trace check, two batches could produce identical numbers by coincidence while silently changing code paths. The trace check closes that gap.

### 6.2 Flag taxonomy (all land in Batch 2, used by 3–7)

```
retrieval.hybrid          # dense+sparse RRF + cross-encoder rerank (Layer A enhanced)
retrieval.rerank          # cross-encoder blend in the merge step
sme.layer_b               # Neo4j INFERRED_RELATION retrieval
sme.layer_c               # SME artifact pack (dossier/insights/recs)
sme.rich_pack             # adapter-budget-aware pack assembly + rich prompts
sme.url_adapter           # ephemeral URL leg
sme.monthly_mining_timer  # systemd timer enable (ops PR only)
```

Stored in the `MutableFlagStore` from phase 1 (Redis-backed, admin endpoints for PATCH/GET). Defaults: all **OFF in prod**; dev/staging can flip per profile.

### 6.3 Rollback contract

- **Any flag-gated regression:** flip the flag off via admin endpoint → rollback in seconds, no deploy.
- **Batch 0 regression (no flag possible):** `git revert` the Batch 0 commit; pipeline write path is untouched so no data rollback.
- **A cherry-pick conflict resolution turned out wrong:** each batch is one PR → one revert = one phase removed cleanly.
- **Safety rail:** every PR description includes the exact `git revert <sha>` command that would undo it, and the list of flag writes that would achieve the same effect without a revert.

### 6.4 Operational discipline

1. Every batch is merged from its own branch named `batch-N-<description>` — never pushed to `main` directly.
2. Every PR has an eval-gate section in the description with the command that reproduces the pass/fail.
3. No two batches merge on the same day. One batch per day minimum, so any regression has exactly one suspect.
4. Batch 0 gets a canary: deploy to a single profile (the owner's) first, observe for 1 hour of real queries, then roll to prod.
5. Implementation agent prepares each PR and hands off; the owner keeps the merge button.

## 7. Batch 0 — Detailed Scope

### 7.1 Unified-model collapse

- Delete `src/serving/fast_path.py` — the module, not just the class.
- Rename in `src/serving/config.py` and `src/serving/vllm_manager.py`: `docwain-fast` → `docwain`. Also the default `model=` arg on `VLLMManager.__init__`.
- Rewrite `src/query/pipeline.py`: remove the `_is_fast_path` / `_handle_fast_path` branch, remove the `try/except ImportError` fast_path import guard, replace `route_taken: str = "smart"` default with `route_taken: str = "intelligence"`. One unified path: Route → Plan → Execute → Assemble → Generate+Verify.
- Scrub docstrings in `src/serving/model_router.py` — replace "27B smart" / "14B fast" language with "unified DocWain model".
- Greeting/identity stays a special case (no retrieval) but inside the unified handler, not a separate module.
- Grep gate before merge: `grep -riE "fast.path|docwain.fast|smart.model|14B|27B" src/ docs/superpowers/ deploy/ systemd/` returns zero hits from live code. Historical references in `docs/superpowers/specs/` are allowed.

### 7.2 Qdrant payload-schema audit

The commit `b0c7211` changed the write-side payload schema. Read-side audit steps:

1. List every caller of `qdrant_client.scroll`, `qdrant_client.search`, and `qdrant_client.query_points` inside `src/` excluding the pipeline write path.
2. For each, enumerate the payload field names used in `Filter`, `must`, `should`, `match`, `match_any`, `FieldCondition`.
3. Diff that set against the names the write path currently emits after `b0c7211`.
4. Fix every mismatch in `src/retrieval/retriever.py`, `src/retrieval/filter_builder.py`, `src/retrieval/profile_query.py`, `src/retrieval/bgem3_retriever.py`, `src/intelligence/retrieval.py`. Do **not** change the write path.
5. Add a new integration test: ingest a fixture doc through the live pipeline, query it through `src/query/pipeline.py`, assert retrieval finds it. This test is the regression guard going forward and runs on every subsequent batch.

### 7.3 Generation-prompt repoint

- Any call to `src/intelligence/generator.py` that still formats responses inline gets repointed to `src/generation/prompts.py`. The reasoner (`src/generation/reasoner.py`) is the only place that calls the LLM for user-visible text.
- DocWain identity/formatting comes from `build_system_prompt` in `src/generation/prompts.py`. No per-intent inline system prompts in the serving layer (the two strings `_GREETING_SYSTEM = "You are DocWain."` survive as a fallback only).

### 7.4 Do-not-touch list for Batch 0

- `src/api/pipeline_api.py` (control plane)
- `src/api/extraction_service.py`, `extraction_pipeline_api.py`, `dw_document_extractor.py`
- `src/api/embedding_service.py`, `src/embed/`, `src/embedding/`
- `src/api/qdrant_indexes.py`, `qdrant_setup.py`, `vector_store.py`
- Celery worker definitions, systemd units for the pipeline
- Mongo status field names
- Any `src/intelligence_v2/` file (revisited in Batch 2 as SME grafts on or around it)

### 7.5 Batch 0 exit criteria

1. Grep gate passes (no `docwain-fast` / `fast path` / `smart path` / `14B` / `27B` model refs in live code).
2. New retrieval integration test passes against the pipeline's current payload schema.
3. A smoke test of 10 canned queries (one per major intent) returns non-empty, grounded, DocWain-persona responses.
4. Existing document-processing integration tests pass unchanged.
5. Canary deploy to the owner's profile: 1 hour of real queries, human-judged "feels like yesterday again" check.

## 8. Testing Strategy

### 8.1 Plane 1 — Unit tests (CI on every PR)

- Batch 0 additions: retrieval filter builder round-trip test (write payload → build filter from current schema → assert match); grep-gate check as a pytest that greps source and fails on forbidden tokens; prompt-path test that asserts no `src/intelligence/generator.py` function returns user-visible text.
- Batches 1–7: each cherry-picked commit brings its own unit tests — run as-is. Conflict resolutions get a brief test added that pins the resolved behavior.
- Rule: no batch merges with red unit tests. No `-k "not broken"` skips.

### 8.2 Plane 2 — Pipeline-isolation regression tests (every PR)

- Existing document-processing integration tests (upload → extract → embed → qdrant write → mongo status transitions) run unchanged on every batch.
- New test in Batch 0: "roundtrip" — ingest a fixture, query through `src/query/pipeline.py`, assert grounded response with correct `file_name` in sources. Runs on every subsequent batch.
- Rule: pipeline tests must be byte-identical green across all batches — no "known flaky" allowances.

### 8.3 Plane 3 — Eval harness (blocking on Batch 0+1, gate per Section 6.1)

- Batch 1 establishes a fixed query set (~50 queries across intents, domains, edge cases) committed to `eval_sets/` — never modified after Batch 1 merges so numbers are comparable across batches.
- Baseline snapshot from Batch 1 lives at `eval_results/baseline-YYYY-MM-DD.jsonl` and is immutable.
- Every subsequent batch runs the harness twice: once with all SME flags OFF (must match baseline exactly), once with its own phase's flags ON in a dev profile (captured as a delta doc).
- Delta doc is markdown, committed alongside the PR, named `eval_deltas/batch-N-<phase>.md`.

### 8.4 Canary plane (Batch 0 only)

- After Batch 0 merges, deploy to the owner's profile first. Run 1 hour of real queries. Watch: groundedness, persona presence, non-empty rate, latency p95, error rate in logs. Then flip to all of prod.
- Canary gate is human-judged — the "does it feel like yesterday again" check.

### 8.5 What this design explicitly does not test

- Model accuracy for its own sake. Model training/retraining is out of scope for all 8 batches.
- Cross-profile consistency beyond what existing tests cover.

### 8.6 CI changes required

- Add a `pytest` marker `@pytest.mark.pipeline_isolation` and a CI job that runs only those tests with the pipeline containers up.
- Add a nightly job that runs the eval harness on current `main` with all flags off and fails if numbers drift from the committed baseline. This is the long-term guard.

## 9. Current test-suite baseline (pre-Batch-0)

Captured 2026-04-21:

- 5443 tests collected, **97 collection errors** — dominated by `TelemetryStore`, `jellyfish`, `get_qdrant_client` imports. These fail because the test files reference SME-branch code that isn't on `main` yet. Not regressions; pre-existing since the SME work diverged.
- Full-suite pass/fail snapshot captured at `/tmp/docwain_test_baseline.log` (committed copy added to `eval_results/pre-batch-0-test-baseline.txt` in Batch 0 itself).

The collection errors will self-resolve as Batches 1–3 bring the SME modules onto main. No separate cleanup batch is required.

## 10. Open questions / risks

- **Adapter YAML hosting.** The spec assumes Azure Blob per the standing rule. If the subscription-override store is empty today, Batch 2 ships only the generic default; per-subscription overrides are an ops follow-up.
- **Redis dependency.** The feature-flag store (phase 1) and retrieval cache (phase 3) both depend on Redis. If Redis is down, flag resolver must fail-closed (flags treated as OFF) and cache must degrade to always-miss. Both behaviors are already in the SME-branch code; they need to be verified in Batch 2 review.
- **vLLM availability during training.** VLLMManager falls back to Ollama Cloud during training-mode. The eval harness must run during a serving-mode window to produce comparable numbers; running it during training-mode taints the baseline. Batch 1 PR must document this and add a refusal-to-run check.
- **`src/intelligence_v2/` overlap.** That parallel stack exists on main and has some overlap with SME phase 2 synthesis. Batch 2 review should decide whether SME replaces, augments, or quarantines it. Resolution deferred to the Batch 2 implementation plan.

## 11. Out-of-scope follow-ups

- Ops PR to enable the monthly pattern-mining systemd timer (after Batch 7).
- Ops PR to flip production flags on, one at a time, after eval parity for each layer.
- A future `intelligence_v2` reconciliation PR if Batch 2 review decides a merge/deprecation is needed.
- Any model retraining or fine-tuning work — tracked separately in the V2/V5/V6 workstreams.
