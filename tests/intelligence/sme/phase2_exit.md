# Phase 2 complete — SME synthesis landed on `main`

Executed: 2026-04-21

Base: commit `9b0ceed` (phase1(sme-integration): sandbox end-to-end plumbing).
Final: see `git log --oneline 9b0ceed..HEAD`.

## Self-review appendix

Every numbered task in the user's execution order ran to a green commit:

| # | Task | Status |
|---|------|--------|
| 10 | Layer B (`retrieve_layer_b`) extended on `UnifiedRetriever` with `INFERRED_RELATION` gated by two flags. | done |
| 11 | Layer C `SMERetrieval` module with per-subscription Qdrant + hard profile filter. | done |
| 12 | Flag-gating end-to-end: Layer B requires master + SME retrieval + synthesized edges; Layer C requires master + SME retrieval. Per-subscription overrides via `FlagStore`; default OFF. | done |
| 13 | Incremental synthesis via `src/intelligence/sme/input_hash.py`; `finalize_training_for_doc` short-circuits on unchanged input hash and persists new hash + run_id on success. | done |
| 14 | Q&A cache-index emission (`emit_qa_index`, `qa_index_fingerprint`) + `invalidate_qa_index` on `PIPELINE_TRAINING_COMPLETED`. | done |
| 15 | End-to-end integration test with real builders + fake LLM producing persisted artifacts, verified edges, trace lifecycle, and non-zero `sme_artifact_hit_rate`. | done |
| 16 | Phase 0 eval-harness wrapper (`scripts/sme_eval/run_sandbox.py`) with dry-run + snapshot assertions; today's dry-run snapshot committed. | done |
| 17 | `scripts/phase2_exit_check.sh` — 9 automated gates, all passing. | done |
| 18 | Public API re-exported via `src/intelligence/sme/__init__.py`; no TODO/FIXME introduced in Phase 2 diff; this marker committed. | done |

## Phase 2 invariants verified

* No Claude / Anthropic / Co-Authored-By references in Phase 2 diff.
* No `datetime.utcnow()` introduced in Phase 2 diff.
* No new `pipeline_status` strings.
* No internal wall-clock timeouts on synthesis paths.
* `src/intelligence/generator.py` untouched.
* All 8 SME feature flags default OFF globally.
* Profile isolation hard at every new retrieval / storage surface.
* MongoDB control-plane writes gated by `_CP_ALLOWED_PROFILE_KEYS`
  (no document content, no training data).

## Phase 3 handoff

Phase 3 retrieval layer can rely on:

* `UnifiedRetriever.retrieve_layer_b(query, subscription_id, profile_id, top_k, include_inferred=True, inferred_confidence_floor=0.6)` — canonical Layer B helper (ERRATA §7).
* `SMERetrieval(qdrant, embedder).retrieve(query=, subscription_id=, profile_id=, artifact_types=None, top_k=10)` — Layer C (ERRATA §8).
* `finalize_training_for_doc` emits `SME_SYNTHESIS_COMPLETED` / `SME_SYNTHESIS_SKIPPED_INPUT_UNCHANGED` / `SME_SYNTHESIS_FAILED` audit events with `run_id` + `input_hash` — Phase 3 observability can subscribe to these.
* QA fast-path keys: `qa_idx:{sub}:{prof}:{fingerprint}` with fingerprint = SHA-256 of whitespace-normalized lowercased question (`qa_index_fingerprint`).
* Flag-off everywhere; Phase 3 opens the per-subscription override once a launch gate passes.
