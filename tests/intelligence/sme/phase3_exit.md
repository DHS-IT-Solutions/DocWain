# Phase 3 exit marker — 2026-04-21

Phase 3 (SME retrieval layer) is complete. This marker file is the exit
sentinel Task 14 checks for; its mere presence (plus a clean
`scripts/phase3_exit_check.sh`) is what unblocks Phase 4.

## Scope delivered

- **Task 8 — Redis retrieval cache** (`src/retrieval/retrieval_cache.py`)
  - Key layout `dwx:retrieval:{sub}:{prof}:{query_fp}:{flag_set_version}`
  - 5-minute TTL (`ttl_seconds=300`)
  - Profile-scoped invalidation via `scan_iter` + `delete`
  - Wired into `src/api/pipeline_api.py::_safe_invalidate_qa_index`
    so every `PIPELINE_TRAINING_COMPLETED` transition evicts the
    matching `(sub, prof)` entries
- **Task 9 — Intent-aware layer gating**
  (`src/retrieval/intent_gating.py`)
  - Simple intents (greeting, identity, lookup, count) skip B + C
  - Borderline + analytical intents run all three layers
  - Integrated into `retrieve_four_layers` via `gate=` kwarg
- **Task 10 — CoreAgent wiring**
  (`src/agent/core_agent.py::_build_sme_pack`)
  - Cache lookup → four-layer retrieval → merge → rerank (CE flag
    gated) → MMR → PackAssembler → `doc_context["sme_pack"]`
  - Phase 4 rich-mode consumers read `doc_context["sme_pack"]` to
    detect presence of SME artifacts
- **Task 11 — End-to-end integration tests**
  (`tests/agent/test_phase3_end_to_end.py`)
  - Flag OFF → no SME items in pack
  - Flag ON → SME-backed items in pack
  - Cache hit skips retrieval
  - `bump_flag_set_version()` invalidates via key change
  - Pipeline-complete hook evicts cache
- **Task 12 — Phase 0 eval harness snapshot**
  (`tests/sme_metrics_phase3_sandbox_2026-04-21.json`)
  - Dry-run snapshot (live API not available in dev env)
  - Surface verification via unit + integration tests documented
- **Task 13 — Top-K tuning**
  (`docs/superpowers/specs/phase3_top_k_defaults.md`)
  - No tuning applied — deferred pending live eval signal
  - Canonical defaults in `src/retrieval/top_k.py` unchanged
- **Task 14 — Phase 3 exit checklist script**
  (`scripts/phase3_exit_check.sh`)

## Contract handed to Phase 4

Phase 4 consumes `doc_context["sme_pack"]: list[PackedItem]` where
each `PackedItem` (frozen dataclass from `src.retrieval.types`) carries:

```
PackedItem(
    text: str,                         # chunk text OR compressed SME
    provenance: tuple[tuple[doc_id, chunk_id], ...],
    layer: Literal["a", "b", "c", "d"],
    confidence: float,
    rerank_score: float,
    sme_backed: bool,                  # True for Layer C OR Layer B kg_inferred
    metadata: dict,
)
```

Phase 4's rich-mode shape resolver uses `sme_backed=True` + `metadata
["artifact_type"]` to pick rich over compact templates.

## Invariants preserved

- `src/generation/prompts.py` untouched
- `src/intelligence/generator.py` untouched
- `ENABLE_SME_RETRIEVAL` default still `False` (per-subscription opt-in)
- No new `pipeline_status` strings
- No internal wall-clock timeouts on retrieval hot path
- Profile isolation enforced at every retrieval + cache operation
- MongoDB remains control-plane only

## Next step

Phase 4 (rich-mode response synthesis) can begin. The retrieval surface
is frozen; Phase 4 only adds response-shape logic.
