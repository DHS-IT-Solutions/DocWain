# Phase 3 Preflight Audit — 2026-04-20

Executed before any Phase 3 code ships. Confirms every Phase 1 + Phase 2
surface that Phase 3 Tasks 2-7 consume. All checks are import + attribute
introspection only — no network I/O, no Blob / Qdrant / Neo4j calls.

## Result: PASS (11/11 checks)

| # | Check | Status | Notes |
|---|-------|--------|-------|
| 1 | `AdapterLoader.load` + `get_adapter_loader` importable, both callable | PASS | ERRATA §1 canonical; `.get` retained as alias |
| 2 | `SMEFeatureFlags.is_enabled` + 8 flag constants (`SME_REDESIGN_ENABLED`, `ENABLE_SME_SYNTHESIS`, `ENABLE_SME_RETRIEVAL`, `ENABLE_KG_SYNTHESIZED_EDGES`, `ENABLE_RICH_MODE`, `ENABLE_URL_AS_PROMPT`, `ENABLE_HYBRID_RETRIEVAL`, `ENABLE_CROSS_ENCODER_RERANK`) exported with canonical names | PASS | ERRATA §4 |
| 3 | `UnifiedRetriever.retrieve_layer_b` present on class | PASS | ERRATA §7; Phase 2 Task surface |
| 4 | `SMERetrieval.retrieve` present on class (NOT `SMERetrievalLayer`) | PASS | ERRATA §8 canonical rename |
| 5 | `HybridSearcher.search` callable | PASS | Phase 1 surface |
| 6 | `CrossEncoderReranker.rerank` callable | PASS | Phase 1 surface (default `cross-encoder/ms-marco-MiniLM-L-6-v2`) |
| 7 | `src.intelligence.qa_generator.qa_index_fingerprint` + `emit_qa_index` present | PASS | ERRATA §13; Phase 2 emission wired |
| 8 | `src.intelligence.sme` package re-exports full Phase 1+2 surface | PASS | `AdapterLoader`, `Adapter`, `ArtifactItem`, `EvidenceRef`, `SMEArtifactStorage`, `StorageDeps`, `SMESynthesizer`, `SMEVerifier`, `SynthesisTraceWriter`, `compute_input_hash`, all builders |
| 9 | `src.api.pipeline_api.finalize_training_for_doc` + `invalidate_qa_index` callable | PASS | ERRATA §13 invalidation hook; Phase 2 Task 10 integration point |
| 10 | `src.api.sme_admin_api.build_router` + `AdapterAdminDeps` importable | PASS | ERRATA §19 canonical filename (was `admin_sme_api.py`) |
| 11 | `src/agent/core_agent.py` exists + `CoreAgent.handle` defined | PASS | Phase 3 Task 3 + Task 7 integration point — modify in place |

## Prior-phase test baseline
`python -m pytest tests/intelligence/sme tests/api tests/retrieval tests/config tests/scripts` →
**284 passed, 4 warnings** (torchao deprecations, unrelated).

## Gaps / follow-ups
- No Phase 1 `set_subscription_override` helper yet in `feature_flags.py`. Task 2
  of Phase 3 adds it (flag-store mutator + monotonic `flag_set_version` counter)
  because Phase 1's `FlagStore` Protocol is read-only today. This is an expected
  Phase 3 addition, not a blocker.
- `src.cache.redis_store` module is referenced by the plan but the canonical
  redis client accessor in this repo is `src.api.dw_newron.get_redis_client`
  (same path used by `qa_generator._resolve_redis_client`). Phase 3 code
  follows that convention.

## Go / no-go
All 11 gates green — proceed to Task 2.
