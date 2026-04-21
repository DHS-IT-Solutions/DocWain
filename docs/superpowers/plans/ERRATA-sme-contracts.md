# DocWain SME Plans — Cross-Plan Contract Errata

**Purpose:** document the canonical shared contracts that all Phase 1-6 plans must align on, and list the drift found during validation that must be reconciled during execution. Before any task with `# Modify` touching a shared module runs, verify the signatures here.

**Authority:** the design spec at `docs/superpowers/specs/2026-04-20-docwain-profile-sme-reasoning-design.md` defines behavior; this errata resolves naming choices where plans diverged.

**Resolution principle:** Phase 1 is the producer of shared modules and owns their signatures. Phases 2-6 are consumers and MUST be corrected to call Phase 1's actual surface. Where Phase 1's naming is non-ideal, the fix lands in Phase 1 itself before Phase 2 runs.

---

## 1. AdapterLoader

**Canonical name:** `src/intelligence/sme/adapter_loader.py`

**Canonical method:** `AdapterLoader.load(subscription_id: str, profile_domain: str) -> Adapter` (NOT `.get()`).

**Canonical metadata access:** after `load()`, the caller can obtain content hash and version via the returned `Adapter`'s built-in `meta` field (or equivalent). Phase 1 must expose `adapter.content_hash` and `adapter.version` as attributes on the loaded `Adapter` model, not only in a separate cache dict.

**Canonical factory:** `get_adapter_loader()` module-level singleton for non-FastAPI callers. FastAPI lifespan wires the same instance into `app.state`.

**Drift:**
- Phase 1 plan (`2026-04-20-docwain-sme-phase1-infrastructure.md`) ships `.get()` and keeps hash in internal cache. **Fix in Phase 1:** rename `get` → `load`; add `content_hash` + `version` fields to `Adapter` model; add module-level `get_adapter_loader()`.
- Phase 2/3/4 plans use `.load()` — leave as-is once Phase 1 is fixed.

---

## 2. SMEArtifactStorage

**Canonical name:** `src/intelligence/sme/storage.py`, class `SMEArtifactStorage`.

**Canonical API:** three facade methods wrapping one internal implementation.

```python
class SMEArtifactStorage:
    def put_snippet(self, subscription_id: str, profile_id: str, item: ArtifactItem, *, synthesis_version: int) -> None:
        """Write retrievable snippet to Qdrant sme_artifacts_{subscription_id}."""
    def put_canonical(self, subscription_id: str, profile_id: str, artifact_type: str, items: list[ArtifactItem], *, synthesis_version: int) -> None:
        """Write canonical JSON to Azure Blob at sme_artifacts/{sub}/{prof}/{artifact_type}/{version}.json."""
    def put_manifest(self, subscription_id: str, profile_id: str, manifest: SynthesisManifest) -> None:
        """Write run manifest pointing to the above."""
    def persist_items(self, subscription_id: str, profile_id: str, artifact_type: str, items: list[ArtifactItem], *, synthesis_version: int) -> None:
        """Convenience wrapper: calls put_snippet for each + put_canonical for batch."""
```

**Drift:**
- Phase 1 plan ships only `persist_items` / `delete_version`. **Fix in Phase 1:** add the three facade methods; keep `persist_items` as a convenience wrapper.
- Phase 2 plan uses a class called `ArtifactStorage`. **Fix in Phase 2 Task 1:** rename constructor call to `SMEArtifactStorage(StorageDeps(...))` per Phase 1.

---

## 3. SMEVerifier

**Canonical class:** `src/intelligence/sme/verifier.py`, class `SMEVerifier`.

**Canonical constructor:** `SMEVerifier(chunk_store: VerifierChunkStore, max_inference_hops: int = 3)`.

**Canonical method:** `SMEVerifier.verify(item: ArtifactItem, ctx: VerifierContext) -> Verdict` for single items, `SMEVerifier.verify_batch(items: list[ArtifactItem], ctx: VerifierContext) -> list[Verdict]` for batches.

**Canonical item contract:** every `ArtifactItem` has a `text: str` field that the verifier uses for similarity checks, plus `evidence: list[EvidenceRef]`, `confidence: float`, and `inference_path: Optional[list]`.

**Drift:**
- Phase 2 produces heterogeneous item types (`DossierSection.narrative`, `InferredEdge.relation_type`, etc.) that lack a unified `.text` field. **Fix in Phase 2:** give every artifact type a `.text` property returning the substantively-checkable string (e.g., `DossierSection.text = self.narrative`). The verifier operates on `.text`; per-type attributes remain for downstream consumers.
- Phase 2 calls `verifier.verify(items, profile_ctx=ctx)` plural. **Fix in Phase 2:** use `verify_batch(items, ctx)` for batch calls.

---

## 4. Feature flags

**Canonical module:** `src/config/feature_flags.py`.

**Canonical class:** `SMEFeatureFlags`.

**Canonical method:** `SMEFeatureFlags.is_enabled(subscription_id: str, flag: FlagName) -> bool` with master-gate precedence.

**Canonical flag names (exactly 8):** `sme_redesign_enabled` (master), `enable_sme_synthesis`, `enable_sme_retrieval`, `enable_kg_synthesized_edges`, `enable_rich_mode`, `enable_url_as_prompt`, `enable_hybrid_retrieval`, `enable_cross_encoder_rerank`.

**Canonical constants:** `src/config/feature_flags.py` exports string constants for each flag (`SME_REDESIGN_ENABLED`, `ENABLE_SME_RETRIEVAL`, ...).

**Canonical singleton:** `get_flag_resolver()` returns the process-wide resolver instance.

**Drift:**
- Phase 1 plan does not export flag constants. **Fix in Phase 1:** add module-level `Final[str]` constants for each flag name.
- Phase 1 plan does not expose `get_flag_resolver()`. **Fix in Phase 1:** add module-level factory.
- Phase 2 plan creates `src/intelligence/sme/flags.py` with `flag_enabled()`. **Fix in Phase 2:** import from `src.config.feature_flags` instead; delete the parallel module.
- Phase 3 plan uses `FeatureFlagResolver.resolve()`. **Fix in Phase 3:** rename all references to `SMEFeatureFlags.is_enabled()`.
- Phase 3 Task 3 has broken flag-gating expression `flags.get(...) is not None` (always True). **Fix in Phase 3:** change to `flags.is_enabled(subscription_id, ENABLE_SME_RETRIEVAL)`.
- Phase 4 plan creates parallel `src/config/features.py` with `is_rich_mode_enabled`. **Fix in Phase 4:** replace with `SMEFeatureFlags(...).is_enabled(sub, ENABLE_RICH_MODE)`.

---

## 5. Trace writers

**Canonical classes:** `src/intelligence/sme/trace.py`, `SynthesisTraceWriter` and `QueryTraceWriter`.

**Canonical method:** `.append(entry: dict) -> None` on both writers.

**Drift:**
- Phase 1 plan uses `.record()`. **Fix in Phase 1:** rename `record` → `append` (underlying `TraceBlobAppender` already uses `.append()`, so match).

---

## 6. Artifact builders

**Canonical module layout:** `src/intelligence/sme/builders/{dossier,insight_index,comparative_register,kg_materializer,recommendation_bank}.py` (a package).

**Canonical base class:** `src/intelligence/sme/builders/_base.py` defines `ArtifactBuilder` ABC.

**Canonical unified item type:** `src/intelligence/sme/artifact_models.py` defines `ArtifactItem` + `EvidenceRef`. Per-artifact-type schemas live alongside (not in a separate `artifact_schemas.py`).

**Drift:**
- Phase 2 plan uses `src/intelligence/sme/{dossier,...}_builder.py` at package root. **Fix in Phase 2:** align with Phase 1's `builders/` subpackage (paths + imports).
- Phase 2 plan ships `artifact_schemas.py` duplicating `artifact_models.py`. **Fix in Phase 2:** either merge or explicitly deprecate `artifact_models.py`.

---

## 7. Retrieval layer B (KG) helper

**Canonical class:** `src/retrieval/unified_retriever.py`, method `UnifiedRetriever.retrieve_layer_b(query, subscription_id, profile_id, top_k, include_inferred=True, inferred_confidence_floor=0.6) -> list[dict]`.

**Drift:**
- Phase 3 plan calls `self._kg.retrieve_context(...)` directly with different kwargs. **Fix in Phase 3:** route through `retrieve_layer_b()` per Phase 2's published surface.

---

## 8. SME retrieval layer (Layer C)

**Canonical class:** `src/retrieval/sme_retrieval.py`, class `SMERetrieval` (NOT `SMERetrievalLayer`).

**Canonical method:** `.retrieve(query, subscription_id, profile_id, top_k, artifact_types: list[str] | None = None) -> list[dict]`.

**Drift:**
- Phase 3 plan uses class name `SMERetrievalLayer`. **Fix in Phase 3:** rename all references to `SMERetrieval`.

---

## 9. ClassifiedQuery

**Canonical dataclass:** `src/serving/model_router.py`, `ClassifiedQuery(frozen=True)`.

**Canonical fields:** `query_text: str`, `intent: Intent`, `format_hint: FormatHint`, `entities: tuple[str, ...]`, `urls: tuple[str, ...]`.

**Drift:**
- Phase 4 plan omits `query_text`. **Fix in Phase 4 Task 2:** add it to the dataclass and populate it in `classify_query()`.

---

## 10. PackSummary

**Canonical dataclass:** defined in Phase 4, in `src/generation/pack_summary.py`.

**Canonical fields:** `total_chunks: int`, `distinct_docs: int`, `has_sme_artifacts: bool`, `bank_entries: tuple[dict, ...]`, `evidence_items: tuple[PackedItem, ...]`, `insights: tuple[PackedItem, ...]`.

**Canonical factory:** `PackSummary.from_packed_items(items: list[PackedItem]) -> PackSummary` that filters by `metadata["artifact_type"]`.

**Drift:**
- Phase 4 plan's `PackSummary` lacks `bank_entries`, `evidence_items`, `insights`, and the factory. **Fix in Phase 4 Task 6:** extend the dataclass and add the factory with full unit tests.

---

## 11. PackedItem

**Canonical dataclass:** Phase 3, `src/retrieval/types.py`.

**Canonical fields:** `text: str`, `provenance: tuple[tuple[str, str], ...]` (doc_id, chunk_id pairs), `layer: Literal["a","b","c","d"]`, `confidence: float`, `rerank_score: float`, `sme_backed: bool`, `metadata: dict` (keys include `artifact_type`, `relation_type`).

**Drift:**
- Phase 3 plan does not set `sme_backed=True` for Layer B synthesized-edge items. **Fix in Phase 3 merge step:** when Layer B item's `kind == "kg_inferred"`, set `sme_backed=True`.
- Phase 3 `bundle.degraded_layers` is double-appended. **Fix in Phase 3:** remove the single-char append; keep the full-name append.

---

## 12. Pipeline helpers in `document_status.py`

Phase 2 assumes the following helpers exist in `src/api/document_status.py`; only `append_audit_log` exists today. Add as part of Phase 2:

- `count_incomplete_docs_in_profile(subscription_id, profile_id) -> int`
- `get_subscription_record(subscription_id) -> dict | None`
- `get_profile_record(subscription_id, profile_id) -> dict | None`
- `update_profile_record(subscription_id, profile_id, updates: dict) -> None`

**Fix in Phase 2:** add a task that creates these helpers with unit tests before Task 10 (training-stage integration).

---

## 13. QA cache index

Phase 3's `QAFastPath.lookup` reads `qa_idx:{sub}:{prof}:{fingerprint}` from Redis, but no plan task creates this index.

**Fix in Phase 2:** add a task in the synthesis pipeline that emits the `qa_idx:` entries when `qa_generator.py` produces Q&A pairs. Invalidation on `PIPELINE_TRAINING_COMPLETED` transition must be explicitly tested.

---

## 14. Retrieval-cache invalidation

Phase 3 requires Redis retrieval cache invalidation on `PIPELINE_TRAINING_COMPLETED` transition, but the hook is hand-waved.

**Fix in Phase 3:** add a concrete task that wires `invalidate_profile()` into the completion branch of `src/api/pipeline_api.py`, with integration test that verifies the cache is cleared when the status transitions.

---

## 15. KG Cypher literal injection

Phase 2's `KGMultiHopMaterializer` interpolates adapter-provided `rule['pattern']` directly into Cypher via f-string.

**Fix in Phase 2:** validate `pattern` against an allowlist of edge types at adapter load time (Phase 1 `Adapter` model should enforce this); reject anything outside the allowlist before the builder runs.

---

## 16. Phase 0 plan — direct fixes applied

1. Task 15 run-time mutation: use `q.model_copy(update={"subscription_id": ..., "profile_id": ...})` instead of in-place assignment.
2. Task 16 placeholder violation: split into six sub-tasks (one per domain) with concrete first-5-queries-per-intent examples.
3. Minor: replace `datetime.utcnow()` with `datetime.now(timezone.utc)`.
4. Minor: add `scale_max` field to `MetricResult` for the 0-5-scale `sme_persona_consistency` metric.
5. Minor: `verified_removal_rate` returns `None` when `metadata.citation_verifier_dropped` field is absent (not 1.0), excluded from denominator.
6. Minor: fixture file overrides via `tests/sme_evalset_v1/fixtures/test_profiles.local.yaml` (gitignored) to avoid dirty-diff workflow.
7. Minor: TTFT capture noted as "deferred to Phase 4" in Task 4 schema documentation.
8. Minor: RAGAS wrapper docstring clarified — reimplements rather than imports from legacy `scripts/ragas_evaluator.py`.

---

## 17. Phase 5 plan — direct fixes applied

1. Six bare `pass` test bodies filled in with real assertions.
2. DNS rebinding: connect-by-pinned-IP + Host-header strategy documented in Task 11 with custom httpx transport; previously the pin was done pre-connect only.
3. `extract_timeout_s` wired into `_extract()` call.
4. Robots.txt body streamed with cap (not `resp.content`).
5. PDF content-type rejected by default in `accept_content_types` (PDF extraction deferred to later phase).

---

## 18. Phase 6 plan — direct fixes applied

1. Redis client import corrected from non-existent `src.utils.redis_client` to `src.utils.redis_cache` (or the actual path — verify in implementation).
2. `datetime.utcnow()` → `datetime.now(timezone.utc)`.
3. Systemd enable runbook added: `systemctl enable --now docwain-sme-pattern-mining.timer`.

---

## 19. Admin router canonical filename

**Canonical:** `src/api/sme_admin_api.py` (Phase 1 Task 4 creates this).

**Drift:** Phase 3 plan originally used `src/api/admin_sme_api.py` (filename swap). **Fixed in Phase 3** on 2026-04-21 — renamed all 5 references to `sme_admin_api.py`.

## 20. StorageDeps shape reconciliation

**Canonical:** `StorageDeps(blob: BlobStore, qdrant: QdrantBridge, neo4j: Neo4jBridge, embedder: object | None = None)`.

Phase 1 ships the four-field dataclass with `embedder` optional. Phase 2 extends it — embedder becomes required at `put_snippet` call sites (it's needed to compute vectors at write time). The `KGMultiHopMaterializer` receives `neo4j` directly, not via StorageDeps, so `neo4j` on StorageDeps is reserved but unused by the storage module itself.

**Fix applied 2026-04-21:** both Phase 1 and Phase 2 StorageDeps updated to match this canonical shape.

## 21. Phase 5 uses Phase 1's feature-flag module (extension of §4)

**Canonical:** Phase 5 is a pure consumer of `src.config.feature_flags` — no parallel module under `src/intelligence/sme/feature_flags.py`.

**Drift:** Phase 5 plan originally shipped its own `src/intelligence/sme/feature_flags.py` with `FeatureFlagResolver` + `UrlAsPromptFlag` constant. This contradicts ERRATA §4 (Phase 1 owns the canonical feature-flag surface).

**Fix applied 2026-04-21:**
- Phase 5 Task 2 is now an audit-only task that greps `src/config/feature_flags.py` for `ENABLE_URL_AS_PROMPT`, `SMEFeatureFlags`, `get_flag_resolver` — no new file, no code.
- Consumer imports throughout Phase 5 updated: `from src.config.feature_flags import SMEFeatureFlags, ENABLE_URL_AS_PROMPT, get_flag_resolver`.
- All `.resolve(...)` calls replaced with `.is_enabled(...)`.
- `UrlAsPromptFlag` constant references replaced with `ENABLE_URL_AS_PROMPT`.
- Phase 5 file count dropped by one (no `feature_flags.py` under `src/intelligence/sme/`).

## 22. SMERetrieval rename reached Phase 5 (extension of §8)

**Fix applied 2026-04-21:** Phase 5's sole reference to `SMERetrievalLayer` renamed to `SMERetrieval` to match Phase 3 canonical.

## Execution order

When executing these plans, follow this order:
1. **Fix Phase 1 first** per items 1, 2, 4, 5, 6 above (rename methods, add constants, add facade methods, add `get_adapter_loader`, add unified `ArtifactItem.text`).
2. **Fix Phase 2** per items 3, 6, 12, 13, 15 above (verifier signature, builder paths, document_status helpers, QA index emission, Cypher allowlist).
3. **Fix Phase 3** per items 4, 7, 8, 11, 14 above (flag API, Layer B routing, SMERetrieval naming, PackedItem enrichment, cache invalidation).
4. **Fix Phase 4** per items 4, 9, 10 above (flag API consumer, `query_text` field, `PackSummary` extensions).
5. Phase 5 direct fixes as per item 17.
6. Phase 6 direct fixes as per item 18.

Each phase's Task 1 (preflight audit) must grep for the drift patterns listed here and fail the task if any appear.
