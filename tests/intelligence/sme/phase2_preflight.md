# SME Phase 2 Preflight Audit

Date: 2026-04-20
Purpose: verify Phase 1 surfaces are present and match the ERRATA-canonical shape
before Phase 2 tasks 2-9 start.

## Phase 1 modules verified

All imports succeed from a fresh interpreter:

- `src.intelligence.sme.adapter_schema`: `Adapter`, `Persona`, `DossierConfig`,
  `InsightDetector`, `ComparisonAxis`, `KGInferenceRule`, `RecommendationFrame`,
  `ResponsePersonaPrompts`, `RetrievalCaps`, `OutputCaps`.
- `src.intelligence.sme.adapter_loader`: `AdapterLoader` with canonical method
  `.load(sub_id, domain)` (ERRATA §1), `get_adapter_loader()` / `init_adapter_loader(...)`
  module-level singleton factory.
- `src.intelligence.sme.artifact_models`: `ArtifactItem`, `EvidenceRef`. Unified
  `ArtifactItem.artifact_type` slugs frozen at
  `("dossier", "insight", "comparison", "kg_edge", "recommendation")`.
- `src.intelligence.sme.verifier`: `SMEVerifier(chunk_store, max_inference_hops)`,
  `.verify(item, ctx)`, `.verify_batch(items, ctx)`, `VerifierContext`, `Verdict`.
- `src.intelligence.sme.trace`: `SynthesisTraceWriter`, `QueryTraceWriter`; both
  expose `.append(event)` (canonical) and `.record` alias (ERRATA §5).
- `src.intelligence.sme.storage`: `SMEArtifactStorage`, `StorageDeps(blob, qdrant,
  neo4j, embedder)` (ERRATA §2 + §20) with facade methods `put_snippet`,
  `put_canonical`, `put_manifest`, `persist_items`.
- `src.intelligence.sme.synthesizer`: `SMESynthesizer`, `SynthesizerDeps`,
  `ArtifactBuilder` protocol.
- `src.intelligence.sme.builders.*`: `_base.ArtifactBuilder` ABC plus five
  skeleton builders with frozen `artifact_type` slugs — `SMEDossierBuilder`
  (`dossier`), `InsightIndexBuilder` (`insight`), `ComparativeRegisterBuilder`
  (`comparison`), `KGMultiHopMaterializer` (`kg_edge`),
  `RecommendationBankBuilder` (`recommendation`).
- `src.config.feature_flags`: all 8 `Final[str]` constants plus `SMEFeatureFlags`,
  `get_flag_resolver()`, `init_flag_resolver(*, store)` (ERRATA §4).
- `src.retrieval.hybrid_search.HybridSearcher` and
  `src.retrieval.reranker.CrossEncoderReranker` exist; Phase 2 tasks 1-9 do not
  touch these but Task 1 confirms availability for Phase 3.

## Baseline test count

Phase 1 test baseline prior to Phase 2 Tasks 1-9:

- `tests/intelligence/sme`: 71 tests
- `tests/api`: 14 tests
- `tests/config`: 10 tests
- `tests/retrieval`: 15 tests

All pass. Phase 2 Tasks 2-9 extend this surface without rewriting any Phase 1
file outside the exact modifications each task's plan entry names.

## Gaps and notes for downstream tasks

- `src/api/document_status.py` today ships `append_audit_log`,
  `update_pipeline_status`, `get_document_record`, `get_documents_collection`,
  `update_document_fields`, `update_stage`. The four helpers Task 9 adds
  (`count_incomplete_docs_in_profile`, `get_subscription_record`,
  `get_profile_record`, `update_profile_record`) are not present yet.
- `src/api/pipeline_api.py` today has no `finalize_training_for_doc` helper;
  Task 8 adds it.
- The unified `ArtifactItem` contract on Phase 1 uses `text` + `evidence:
  list[EvidenceRef]` + `confidence` + `inference_path` + `metadata` rather than
  per-type schema classes. Phase 2 builders produce these `ArtifactItem`
  instances directly — the per-type schemas described in the plan Task 2 are
  folded into the unified model via `metadata` (ERRATA §3/§6: every type exposes
  `.text` on the unified contract). No parallel `artifact_schemas.py` is
  introduced.
- `SynthesizerDeps.builders` is an ordered `dict[str, ArtifactBuilder]` keyed
  on the artifact-type slug; iteration order encodes the execution order
  `dossier → insight → comparison → kg_edge → recommendation`.

## Verdict

Phase 1 surfaces are intact and match ERRATA §§1-6, §12 (needs additions),
§15, §20. Phase 2 Tasks 2-9 can proceed.
