# DocWain SME Phase 3 — Query-Time SME Retrieval, Compact Responses

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Flip `enable_sme_retrieval` for opt-in subscriptions, wire all four retrieval layers (A chunks, B KG, C SME artifacts, D URL placeholder), add a cross-encoder reranker with MMR diversity, assemble a budget-aware pack per adapter, and gate simple intents out of Layer B/C — **without touching response shape**. The proof point is that pre-reasoned SME artifacts raise RAGAS `answer_faithfulness` from the 0.514 Phase 0 baseline to ≥ 0.80 while leaving `hallucination_rate = 0.0` and `context_recall ≥ 0.80` unchanged. Response formatting (rich templates, new intents, personas) is strictly Phase 4 territory; Phase 3 responses look exactly like today's to the end user.

**Architecture:** A new orchestration layer lives in `src/retrieval/unified_retriever.py` (extended, not replaced) and a thin coordinator `src/retrieval/pack_assembler.py`. Four layer fetchers run in parallel via a single `ThreadPoolExecutor` (max_workers=4). Layer A is the existing dense-or-hybrid Qdrant path (hybrid already landed in Phase 1 under `enable_hybrid_retrieval`). Layer B is `src/kg/retrieval.py` extended to include synthesized edges with a confidence floor (Phase 2 landed the edges; Phase 3 turns on the query-time filter flip). Layer C is `src/retrieval/sme_retrieval.py` from Phase 2, now **called from the agent** instead of being dark. Layer D is a stub that returns an empty result for all queries — Phase 5 owns URL fetch. Results from all four layers flow through a merge step that deduplicates on `(doc_id, chunk_id)`-equivalent keys, then a cross-encoder reranker (`src/retrieval/reranker.py` already exists — extended with the SME-aware score blend), then MMR diversity selection with adapter-tunable lambda, then `PackAssembler` enforces the per-intent `max_pack_tokens` budget from the adapter and drops lowest-confidence items first. A QA-cache fast path short-circuits the entire retrieval and reasoning pipeline when the query fingerprint hits a pre-grounded Q&A pair from `src/intelligence/qa_generator.py`. A Redis retrieval cache keyed by `(subscription_id, profile_id, query_fingerprint, flag_set_version)` memoizes the merged pack for five minutes so near-duplicate queries skip Stages 1–2 entirely. Intent-aware layer gating — driven off the intent field that `src/agent/intent.py` produces — turns Layer B and Layer C **off** for `greeting`, `identity`, `lookup`, `count`, and `extract` intents, where SME artifacts add latency but no recall. Three feature flags (`enable_sme_retrieval`, `enable_hybrid_retrieval`, `enable_cross_encoder_rerank`) default ON under the Phase 1 master `sme_redesign_enabled` flag; `enable_sme_retrieval` remains the one that must be flipped per-subscription to actually pull Layer C.

**Tech Stack:** Python 3.12, existing `qdrant-client` (dense + sparse via hybrid helper from Phase 1), existing `neo4j` driver (Phase 2's `INFERRED_RELATION` edges), existing `sentence-transformers` cross-encoder (already loaded once at app startup in `src/api/rag_state.py`), existing Redis (`src/cache/redis_store.py`), existing `SMEFeatureFlags` from Phase 1 (`src/config/feature_flags.py`), existing `AdapterLoader` from Phase 1 (`src/intelligence/sme/adapter_loader.py`), existing `SMERetrieval` from Phase 2 (`src/retrieval/sme_retrieval.py`), existing `QAGenerator` from `src/intelligence/qa_generator.py`, `pytest` + `pytest-asyncio` for tests, `hypothesis` for property-based pack-budget tests.

**Related spec:** `docs/superpowers/specs/2026-04-20-docwain-profile-sme-reasoning-design.md` Sections 7 (query-time retrieval architecture — fully implemented here), 8 (grounding — only the retrieval-filter + pack provenance parts; prompt work is Phase 4), 10 (measurement — Phase 3 launch gate), 12 (rollout — Phase 3 gate: faithfulness 0.514 → ≥ 0.80, hallucination unchanged, context_recall ≥ 0.80).

**Prior-phase contracts referenced by Phase 3:**
- Phase 1 — `SMEFeatureFlags.is_enabled(sub_id, flag)` (from `src.config.feature_flags`, singleton via `get_flag_resolver()`), `AdapterLoader.load(sub_id, profile_domain)` (singleton via `get_adapter_loader()`), Qdrant sparse-vector re-index complete, embedded cross-encoder available as `app_state.reranker`.
- Phase 2 — `SMERetrieval.retrieve(query, sub_id, profile_id, top_k, artifact_types=None)` returns `List[SMEArtifactHit]`; `UnifiedRetriever.retrieve_layer_b(query, subscription_id, profile_id, top_k, include_inferred=True, inferred_confidence_floor=0.6)` is the canonical KG helper; KG multi-hop materializer has populated `INFERRED_RELATION` edges with a `confidence` property that clients filter by.
- Phase 0 — `tests/sme_evalset_v1/` plus `scripts/sme_eval/run_baseline.py` plus `tests/sme_metrics_baseline_YYYY-MM-DD.json` committed. Phase 3 re-runs the same harness against the flag-ON deployment to produce `tests/sme_metrics_phase3_YYYY-MM-DD.json`.

**Memory rules that constrain this plan (hard):**
- **No Claude attribution** — no Claude / Anthropic / Co-Authored-By strings in commits, code, or docs.
- **No internal timeouts** — parallel-layer retrieval has no wall-clock abort. `as_completed` is used without `timeout=` (the existing `timeout=30` call in `unified_retriever.py:59` is removed in Task 3). The only timeout allowed is the existing httpx safety timeout around URL fetch — not relevant to Phase 3.
- **MongoDB = control plane only** — Phase 3 reads `profile_domain` and per-subscription flag values from Mongo. No document content touches Mongo.
- **Response formatting lives in `src/generation/prompts.py`** — Phase 3 does **not** modify `prompts.py` at all. Any temptation to reshape the prompt from the retrieval side is deferred to Phase 4. The pack handed to the Reasoner stays the same shape as today with additional items, not a new schema.
- **No new `pipeline_status` strings** — Phase 3 is query-path only. Training stage is untouched.
- **Profile isolation hard** — every retrieval layer, the Redis cache key, the QA-cache key, and the adapter lookup all include `(subscription_id, profile_id)`. Tests assert cross-subscription rejection.
- **Engineering-first** — no model retraining. No new LLM call on the hot path. Cross-encoder reranker is a local sentence-transformers model, not a gateway call.
- **Intelligence precomputed at ingestion** — Layer C only reads pre-built SME artifacts; Layer B only reads pre-materialized synthesized edges. No live synthesis at query time.

---

## File structure

```
src/retrieval/
├── unified_retriever.py                 [MODIFIED — 4-layer parallel orchestration]
├── reranker.py                          [MODIFIED — cross-encoder blend + SME-aware weighting]
├── pack_assembler.py                    [NEW — budget-aware pack assembly]
├── mmr.py                               [NEW — maximal marginal relevance]
├── sme_retrieval.py                     [EXISTING — Phase 2; called from here now]
├── retrieval_cache.py                   [NEW — Redis retrieval-pack cache]
├── qa_fast_path.py                      [NEW — QA-cache hit short-circuit]
├── intent_gating.py                     [NEW — simple-intent layer-gating policy]
└── hybrid_search.py                     [EXISTING — Phase 1; Layer A dense+sparse]

src/agent/
└── core_agent.py                        [MODIFIED — wire QA fast path + 4-layer orchestration]

src/kg/
└── retrieval.py                         [MODIFIED — include INFERRED_RELATION with confidence floor]

src/api/
└── admin_sme_api.py                     [MODIFIED — add per-sub enable_sme_retrieval flip endpoint]

src/config/
└── feature_flags.py                     [EXISTING — Phase 1; Phase 3 adds 2 flag names + monotonic flag-set version]

tests/retrieval/                         [existing dir]
├── test_unified_retriever_four_layer.py [NEW]
├── test_unified_retriever_gating.py     [NEW]
├── test_unified_retriever_cache.py      [NEW]
├── test_reranker_cross_encoder.py       [NEW]
├── test_reranker_mmr.py                 [NEW]
├── test_pack_assembler_budget.py        [NEW]
├── test_pack_assembler_drop_order.py    [NEW]
├── test_qa_fast_path.py                 [NEW]
├── test_retrieval_cache.py              [NEW]
├── test_intent_gating.py                [NEW]
└── test_hybrid_flag_off.py              [NEW]

tests/agent/
├── test_core_agent_sme_retrieval.py     [NEW — flag ON, profile has artifacts]
├── test_core_agent_sme_retrieval_off.py [NEW — flag OFF fallback]
├── test_core_agent_qa_short_circuit.py  [NEW]
└── test_core_agent_isolation.py         [NEW — cross-sub rejection]

tests/kg/
└── test_retrieval_with_inferred_edges.py [NEW]

tests/api/
└── test_admin_sme_flag_flip.py          [NEW]

tests/sme_evalset_v1/                    [existing — Phase 0]
└── (re-used as-is; no file changes)

tests/sme_metrics_phase3_YYYY-MM-DD.json [NEW — committed after Task 12 run]
```

Each file does one thing. `pack_assembler.py` does not know about HTTP or the LLM gateway; `mmr.py` is a pure function over scored items; `intent_gating.py` is a stateless policy object. Tests are partitioned so the cross-encoder rerank can be swapped (MiniLM → BGE → trained reranker) without touching other test files.

---

## Task 1 — Audit Phase 1 + Phase 2 prerequisites

**Files:**
- Audit only: `src/config/feature_flags.py`, `src/intelligence/sme/adapter_loader.py`, `src/retrieval/hybrid_search.py`, `src/retrieval/sme_retrieval.py`, `src/kg/retrieval.py`, `src/retrieval/unified_retriever.py`, `src/api/rag_state.py`, `scripts/reindex_qdrant_sparse.py`.

Purpose: before a single line of Phase 3 code ships, confirm the prior-phase surfaces Phase 3 depends on are actually in place in the branch, at the shape the plan assumes. Phase 3 fails silently and nastily if Layer C is not retrievable or if the cross-encoder is not cached on the app state.

- [ ] **Step 1: Verify `SMEFeatureFlags` surface**

Run: `grep -n "class SMEFeatureFlags" src/config/feature_flags.py` and confirm the canonical surface exists. If any item below is missing, STOP and file a Phase 1 bug — do not proceed.

Required:
- `SMEFeatureFlags.is_enabled(subscription_id: str, flag: str) -> bool` (master-gating precedence)
- `get_flag_resolver() -> SMEFeatureFlags` module-level singleton factory
- An override entry point that can be used by the admin endpoint in Task 2 (e.g. `set_subscription_override(subscription_id, flag, value, actor)` — exact name TBD by Phase 1; audit confirms it exists and is callable)
- Known flag constants (exact string values per ERRATA §4): `SME_REDESIGN_ENABLED`, `ENABLE_SME_SYNTHESIS`, `ENABLE_SME_RETRIEVAL`, `ENABLE_KG_SYNTHESIZED_EDGES`, `ENABLE_RICH_MODE`, `ENABLE_URL_AS_PROMPT`, `ENABLE_HYBRID_RETRIEVAL`, `ENABLE_CROSS_ENCODER_RERANK`.

If any flag constant is missing from `src.config.feature_flags`: stop and file a Phase 1 bug (Phase 1 is responsible for the full 8-flag set).

- [ ] **Step 2: Verify `AdapterLoader` produces `retrieval_caps`**

```bash
python -c "from src.intelligence.sme.adapter_loader import get_adapter_loader; \
  a = get_adapter_loader().load('sub_probe', 'generic'); \
  print('retrieval_caps:', a.retrieval_caps); \
  print('max_pack_tokens:', a.retrieval_caps.get('max_pack_tokens', {}))"
```

Expected: prints the adapter's per-intent `max_pack_tokens` map for `analyze`, `diagnose`, `recommend`, `investigate` (and inherits `generic` for smaller intents). If the output is empty or the key is missing, update the `generic.yaml` in Blob per the spec Section 5 example, invalidate the adapter cache, retry. This is a blocker.

- [ ] **Step 3: Verify `SMERetrieval` is callable against a seeded sandbox profile**

```bash
python -c "
from src.retrieval.sme_retrieval import SMERetrieval
layer = SMERetrieval()
hits = layer.retrieve(
    query='summarize revenue trends',
    subscription_id='REPLACE_WITH_SANDBOX_SUB',
    profile_id='REPLACE_WITH_SANDBOX_FINANCE',
    top_k=5,
)
print('sme_hits:', len(hits))
for h in hits[:2]:
    print(h.artifact_type, h.confidence, h.evidence)
"
```

Expected: ≥ 1 hit on the sandbox finance profile (Phase 2 synthesized artifacts should be persisted in Qdrant under `sme_artifacts_{sub}`). If zero hits, the Phase 2 synthesis did not persist artifacts — raise with Phase 2 owner, do not proceed.

- [ ] **Step 4: Verify Neo4j synthesized edges exist**

```bash
python -c "
from src.kg.neo4j_store import get_neo4j_driver
drv = get_neo4j_driver()
with drv.session() as s:
    r = s.run('MATCH ()-[r:INFERRED_RELATION {source: \\'sme_synthesis\\'}]->() RETURN count(r) AS n').single()
    print('inferred_edges:', r['n'])
"
```

Expected: a non-zero count on the sandbox subscription. Zero means Phase 2 KG multi-hop materializer didn't run — block.

- [ ] **Step 5: Verify cross-encoder is loaded on app state**

```bash
python -c "
from src.api.rag_state import get_app_state
s = get_app_state()
print('reranker:', type(s.reranker).__name__ if s and getattr(s, 'reranker', None) else None)
"
```

Expected: prints a class name like `CrossEncoder`, not `None`. If `None`: confirm `src/api/app_lifespan.py` loads the model at startup; Phase 1 was supposed to make this non-optional. Block.

- [ ] **Step 6: Verify sparse re-index completed on sandbox subscription**

```bash
python -c "
from qdrant_client import QdrantClient
from src.api.config import Config
c = QdrantClient(url=Config.Qdrant.URL, api_key=Config.Qdrant.API)
info = c.get_collection('REPLACE_WITH_SANDBOX_SUB')
vecs = info.config.params.vectors
print('dense:', 'dense' in vecs if isinstance(vecs, dict) else 'single')
print('sparse:', list((info.config.params.sparse_vectors or {}).keys()))
"
```

Expected: `dense` named vector AND at least one sparse vector name (`bm25` per Phase 1). If sparse is missing, `enable_hybrid_retrieval=true` will raise at runtime — Task 10 includes a dense-only fallback, but sparse should be present on sandbox.

- [ ] **Step 7: Write the audit report**

Create `tests/retrieval/phase3_preflight_report_YYYY-MM-DD.md` with one line per audit step above: PASS / FAIL / N/A. Commit alongside Task 2.

- [ ] **Step 8: Commit**

Once all 6 audit steps PASS on the sandbox subscription:

```bash
git add tests/retrieval/phase3_preflight_report_*.md
git commit -m "phase3(sme-preflight): phase 1 + phase 2 dependencies verified"
```

---

## Task 2 — Admin endpoint to flip `enable_sme_retrieval` per subscription

**Files:**
- Modify: `src/api/admin_sme_api.py`
- Create: `tests/api/test_admin_sme_flag_flip.py`

Phase 1 shipped the generic flag resolver; Phase 2 shipped a sandbox-only `enable_sme_synthesis` flip. Phase 3 needs an operator-callable endpoint to flip `enable_sme_retrieval` per opt-in subscription with audit logging. The endpoint reuses the existing flag-flip handler; Phase 3 only adds route wiring, a validator for the specific flag names Phase 3 introduces, and an audit-log call.

- [ ] **Step 1: Write the failing tests**

Create `tests/api/test_admin_sme_flag_flip.py`:

```python
"""Admin endpoint tests for Phase 3 SME retrieval flag."""
import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.config.feature_flags import (
    get_flag_resolver, ENABLE_SME_RETRIEVAL, SME_REDESIGN_ENABLED,
)
from src.config import feature_flags as _ff_mod


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def _reset_flag_store(monkeypatch):
    # Reset the in-memory override store and the monotonic version counter.
    # The admin endpoint calls set_subscription_override which bumps the
    # module-level counter (see Task 10).
    from tests.helpers.flag_store import reset_test_flag_store  # helper shipped with Phase 1
    reset_test_flag_store()
    yield
    reset_test_flag_store()


def test_flip_enable_sme_retrieval_on(client, admin_auth_headers):
    # Master must be on for the dependent flag to resolve True end-to-end.
    resp_master = client.post(
        f"/admin/sme/flags/{SME_REDESIGN_ENABLED}",
        json={"subscription_id": "sub_fin_1", "value": True, "reason": "phase3 rollout"},
        headers=admin_auth_headers,
    )
    assert resp_master.status_code == 200
    resp = client.post(
        f"/admin/sme/flags/{ENABLE_SME_RETRIEVAL}",
        json={"subscription_id": "sub_fin_1", "value": True, "reason": "opt-in fin"},
        headers=admin_auth_headers,
    )
    assert resp.status_code == 200
    assert resp.json()["flag"] == ENABLE_SME_RETRIEVAL
    assert resp.json()["value"] is True
    assert get_flag_resolver().is_enabled("sub_fin_1", ENABLE_SME_RETRIEVAL) is True


def test_flip_rejects_unknown_flag(client, admin_auth_headers):
    resp = client.post(
        "/admin/sme/flags/made_up_flag",
        json={"subscription_id": "sub_fin_1", "value": True},
        headers=admin_auth_headers,
    )
    assert resp.status_code == 400
    assert "unknown flag" in resp.json()["detail"].lower()


def test_flip_requires_admin_auth(client):
    resp = client.post(
        f"/admin/sme/flags/{ENABLE_SME_RETRIEVAL}",
        json={"subscription_id": "sub_fin_1", "value": True},
    )
    assert resp.status_code in (401, 403)


def test_flip_writes_audit_log(client, admin_auth_headers, captured_audit):
    client.post(
        f"/admin/sme/flags/{ENABLE_SME_RETRIEVAL}",
        json={"subscription_id": "sub_fin_1", "value": True, "reason": "opt-in"},
        headers=admin_auth_headers,
    )
    assert any(
        e["flag"] == ENABLE_SME_RETRIEVAL and e["new_value"] is True
        for e in captured_audit
    )


def test_flip_off_disables_layer_c(client, admin_auth_headers):
    # Seed master ON + flag ON via the admin endpoint, then flip flag OFF.
    for flag in (SME_REDESIGN_ENABLED, ENABLE_SME_RETRIEVAL):
        client.post(
            f"/admin/sme/flags/{flag}",
            json={"subscription_id": "sub_fin_1", "value": True, "reason": "seed"},
            headers=admin_auth_headers,
        )
    resp = client.post(
        f"/admin/sme/flags/{ENABLE_SME_RETRIEVAL}",
        json={"subscription_id": "sub_fin_1", "value": False, "reason": "rollback"},
        headers=admin_auth_headers,
    )
    assert resp.status_code == 200
    assert get_flag_resolver().is_enabled("sub_fin_1", ENABLE_SME_RETRIEVAL) is False
```

Run: `pytest tests/api/test_admin_sme_flag_flip.py -v`
Expected: FAIL (route not wired yet).

- [ ] **Step 2: Wire the route**

Modify `src/api/admin_sme_api.py`. Find the existing router and add:

```python
# src/api/admin_sme_api.py  (additions only; surrounding module untouched)
from fastapi import HTTPException
from src.config.feature_flags import (
    get_flag_resolver, set_subscription_override, bump_flag_set_version,
    SME_REDESIGN_ENABLED, ENABLE_SME_SYNTHESIS, ENABLE_SME_RETRIEVAL,
    ENABLE_KG_SYNTHESIZED_EDGES, ENABLE_RICH_MODE, ENABLE_URL_AS_PROMPT,
    ENABLE_HYBRID_RETRIEVAL, ENABLE_CROSS_ENCODER_RERANK,
)

# Phase 3 owns the retrieval-side flips. Master + dependent toggles remain
# Phase 2 territory but are whitelisted here too so a single endpoint serves
# both phases.
_PHASE3_FLIPPABLE = {
    SME_REDESIGN_ENABLED,
    ENABLE_SME_RETRIEVAL,
    ENABLE_KG_SYNTHESIZED_EDGES,
    ENABLE_HYBRID_RETRIEVAL,
    ENABLE_CROSS_ENCODER_RERANK,
}


@router.post("/flags/{flag_name}")
def flip_flag(flag_name: str, body: FlipBody, admin=Depends(require_admin_auth)):
    if flag_name not in _PHASE3_FLIPPABLE:
        raise HTTPException(status_code=400, detail=f"unknown flag: {flag_name}")
    resolver = get_flag_resolver()
    prior = resolver.is_enabled(body.subscription_id, flag_name)
    # Phase 1's override helper persists the override to the flag store and
    # bumps the monotonic flag_set_version counter Task 10 reads.
    set_subscription_override(
        subscription_id=body.subscription_id,
        flag=flag_name,
        value=body.value,
        actor=admin.email,
    )
    bump_flag_set_version()  # idempotent; invalidates retrieval cache keys
    _audit_log({
        "kind": "sme_flag_flip",
        "flag": flag_name,
        "subscription_id": body.subscription_id,
        "prior_value": prior,
        "new_value": body.value,
        "reason": body.reason,
        "actor": admin.email,
    })
    return {"flag": flag_name, "subscription_id": body.subscription_id,
            "value": body.value, "prior_value": prior}
```

`set_subscription_override` and `bump_flag_set_version` are Phase 1 exports (canonical per ERRATA §4). If missing at audit time, stop and file a Phase 1 bug; Phase 3 must not define a parallel override path.

- [ ] **Step 3: Run tests green**

`pytest tests/api/test_admin_sme_flag_flip.py -v` → PASS on all five tests.

- [ ] **Step 4: Commit**

```bash
git add src/api/admin_sme_api.py src/config/feature_flags.py \
        tests/api/test_admin_sme_flag_flip.py
git commit -m "phase3(sme-flags): admin flip endpoint for enable_sme_retrieval"
```

---

## Task 3 — Four-layer parallel retrieval orchestration in `unified_retriever.py`

**Files:**
- Modify: `src/retrieval/unified_retriever.py`
- Create: `tests/retrieval/test_unified_retriever_four_layer.py`

The core of Phase 3. Today's `UnifiedRetriever.retrieve()` runs three layers (Qdrant, Neo4j, Mongo metadata). Phase 3 re-casts those as **Layer A (chunks)**, **Layer B (KG)**, adds **Layer C (SME artifacts)**, and stubs **Layer D (URL placeholder, always-empty in Phase 3)**. The Mongo-metadata layer that existed before becomes part of the Layer A payload filter build and is no longer its own concurrent leg.

- [ ] **Step 1: Write the failing tests**

Create `tests/retrieval/test_unified_retriever_four_layer.py`:

```python
"""Four-layer parallel retrieval — Phase 3 core contract."""
from unittest.mock import MagicMock, patch

import pytest

from src.retrieval.unified_retriever import UnifiedRetriever, RetrievalBundle
from src.config.feature_flags import ENABLE_SME_RETRIEVAL, SME_REDESIGN_ENABLED


@pytest.fixture
def _all_flags_on(monkeypatch):
    """Stub get_flag_resolver() to return a fake that reports everything on."""
    from src.config import feature_flags as ff
    class _AllOn:
        def is_enabled(self, *_a, **_kw): return True
    monkeypatch.setattr(ff, "get_flag_resolver", lambda: _AllOn())
    yield


@pytest.fixture
def _sme_flag_off(monkeypatch):
    from src.config import feature_flags as ff
    class _Resolver:
        def is_enabled(self, sub, flag):
            return flag != ENABLE_SME_RETRIEVAL
    monkeypatch.setattr(ff, "get_flag_resolver", lambda: _Resolver())
    yield


@pytest.fixture
def fake_qdrant():
    c = MagicMock()
    c.search.return_value = [
        MagicMock(id="ch_1", score=0.91, payload={"text": "Q3 revenue up 12%",
                                                   "doc_id": "d1", "chunk_id": "ch_1",
                                                   "profile_id": "p1"}),
        MagicMock(id="ch_2", score=0.88, payload={"text": "gross margin",
                                                   "doc_id": "d2", "chunk_id": "ch_2",
                                                   "profile_id": "p1"}),
    ]
    return c


@pytest.fixture
def fake_kg():
    # Phase 2 KGRetrievalClient surface: one_hop + inferred_relations.
    k = MagicMock()
    k.one_hop.return_value = [
        {"from": "n1", "to": "n2", "relation_type": "cites", "confidence": 0.9},
    ]
    k.inferred_relations.return_value = [
        {"from": "n1", "to": "n3", "relation_type": "correlates_with", "confidence": 0.82},
    ]
    return k


@pytest.fixture
def fake_sme():
    s = MagicMock()
    s.retrieve.return_value = [
        MagicMock(artifact_type="dossier", narrative="Q3 narrative",
                  confidence=0.9, evidence=["d1#ch_1"], score=0.85),
    ]
    return s


def test_all_four_layers_invoked(_all_flags_on, fake_qdrant, fake_kg, fake_sme):
    ur = UnifiedRetriever(kg_client=fake_kg, qdrant=fake_qdrant, sme=fake_sme)
    bundle = ur.retrieve(
        query="revenue trend",
        subscription_id="s1", profile_id="p1",
        query_understanding={"intent": "analyze"},
        flags={},  # resolver is the source of truth; flags dict is legacy
    )
    assert isinstance(bundle, RetrievalBundle)
    assert bundle.layer_a_chunks is not None
    assert bundle.layer_b_kg is not None
    assert bundle.layer_c_sme is not None
    assert bundle.layer_d_url == []
    fake_qdrant.search.assert_called()
    # retrieve_layer_b drives one_hop + inferred_relations under the hood.
    fake_kg.one_hop.assert_called()
    fake_sme.retrieve.assert_called()


def test_layers_run_in_parallel(_all_flags_on, fake_qdrant, fake_kg, fake_sme):
    import time
    def slow_qdrant(*a, **kw): time.sleep(0.15); return fake_qdrant.search.return_value
    def slow_kg(*a, **kw): time.sleep(0.15); return fake_kg.one_hop.return_value
    def slow_sme(*a, **kw): time.sleep(0.15); return fake_sme.retrieve.return_value
    fake_qdrant.search.side_effect = slow_qdrant
    fake_kg.one_hop.side_effect = slow_kg
    fake_sme.retrieve.side_effect = slow_sme
    ur = UnifiedRetriever(kg_client=fake_kg, qdrant=fake_qdrant, sme=fake_sme)
    t0 = time.perf_counter()
    ur.retrieve(query="x", subscription_id="s", profile_id="p",
                query_understanding={"intent": "analyze"},
                flags={})
    elapsed = time.perf_counter() - t0
    # three slow legs × 0.15s serial = 0.45s. parallel must be <= ~0.30s.
    assert elapsed < 0.32, f"layers ran serially: {elapsed:.2f}s"


def test_layer_c_skipped_when_flag_off(_sme_flag_off, fake_qdrant, fake_kg, fake_sme):
    ur = UnifiedRetriever(kg_client=fake_kg, qdrant=fake_qdrant, sme=fake_sme)
    bundle = ur.retrieve(
        query="x", subscription_id="s", profile_id="p",
        query_understanding={"intent": "analyze"},
        flags={},
    )
    fake_sme.retrieve.assert_not_called()
    assert bundle.layer_c_sme == []


def test_layer_failure_does_not_kill_others(_all_flags_on, fake_qdrant, fake_kg, fake_sme):
    fake_sme.retrieve.side_effect = RuntimeError("qdrant down")
    ur = UnifiedRetriever(kg_client=fake_kg, qdrant=fake_qdrant, sme=fake_sme)
    bundle = ur.retrieve(
        query="x", subscription_id="s", profile_id="p",
        query_understanding={"intent": "analyze"},
        flags={},
    )
    assert bundle.layer_a_chunks is not None
    assert bundle.layer_b_kg is not None
    assert bundle.layer_c_sme == []  # degraded gracefully
    # Fix per ERRATA §11: only the full layer name is appended (no 'c' stub).
    assert "layer_c" in bundle.degraded_layers
    assert "c" not in bundle.degraded_layers


def test_profile_isolation_filter_on_every_layer(_all_flags_on, fake_qdrant, fake_kg, fake_sme):
    ur = UnifiedRetriever(kg_client=fake_kg, qdrant=fake_qdrant, sme=fake_sme)
    ur.retrieve(query="x", subscription_id="s1", profile_id="p_a",
                query_understanding={"intent": "analyze"},
                flags={})
    # Qdrant call must include profile_id filter
    q_kwargs = fake_qdrant.search.call_args.kwargs
    # check the filter includes the profile_id key somewhere
    assert "p_a" in str(q_kwargs)
    # SME call must include subscription_id AND profile_id
    s_kwargs = fake_sme.retrieve.call_args.kwargs
    assert s_kwargs.get("subscription_id") == "s1"
    assert s_kwargs.get("profile_id") == "p_a"
    # KG call likewise (via retrieve_layer_b → one_hop)
    k_kwargs = fake_kg.one_hop.call_args.kwargs
    assert k_kwargs.get("subscription_id") == "s1"
    assert k_kwargs.get("profile_id") == "p_a"


def test_no_internal_timeout(_all_flags_on, fake_qdrant, fake_kg, fake_sme):
    """as_completed must NOT be called with a timeout argument (memory rule)."""
    import concurrent.futures as cf
    original_as_completed = cf.as_completed
    captured = {}

    def spy_as_completed(*a, **kw):
        captured["kw"] = kw
        captured["args"] = a
        return original_as_completed(*a, **kw)

    with patch("src.retrieval.unified_retriever.as_completed", side_effect=spy_as_completed):
        ur = UnifiedRetriever(kg_client=fake_kg, qdrant=fake_qdrant, sme=fake_sme)
        ur.retrieve(query="x", subscription_id="s", profile_id="p",
                    query_understanding={"intent": "analyze"},
                    flags={})
    assert "timeout" not in captured["kw"]
```

Run: `pytest tests/retrieval/test_unified_retriever_four_layer.py -v` — all six expected to fail.

- [ ] **Step 2: Rewrite `UnifiedRetriever` with four layers**

Replace the body of `src/retrieval/unified_retriever.py`:

```python
"""Four-layer unified retriever — Phase 3.

Layer A: Qdrant chunks (dense, or dense+sparse RRF under enable_hybrid_retrieval)
Layer B: Neo4j KG (includes synthesized INFERRED_RELATION edges when flag ON)
Layer C: SME artifacts (pre-reasoned, behind enable_sme_retrieval)
Layer D: Ephemeral URL (Phase 5 wires the fetch; Phase 3 returns empty)

No internal timeouts. Layer failures degrade gracefully, never abort the bundle.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RetrievalBundle:
    """Raw per-layer results plus telemetry; consumed by merge + rerank.

    layer_b_kg is a flat List[Dict] of edge dicts as returned by
    retrieve_layer_b (ERRATA §7), not the legacy {nodes, edges} graph shape.
    Each dict carries a ``kind`` key of "kg_direct" or "kg_inferred".
    """
    layer_a_chunks: List[Dict[str, Any]] = field(default_factory=list)
    layer_b_kg: List[Dict[str, Any]] = field(default_factory=list)
    layer_c_sme: List[Dict[str, Any]] = field(default_factory=list)
    layer_d_url: List[Dict[str, Any]] = field(default_factory=list)
    degraded_layers: List[str] = field(default_factory=list)
    per_layer_ms: Dict[str, float] = field(default_factory=dict)


class UnifiedRetriever:
    """Phase 3 four-layer orchestrator. Constructor matches Phase 2's
    `UnifiedRetriever(kg_client, qdrant, sme)` surface with optional Phase 3
    additions (`hybrid`, `intent_gate`)."""

    def __init__(self, kg_client=None, qdrant=None, sme=None, *,
                 hybrid=None, intent_gate=None):
        self._qdrant = qdrant
        self._kg = kg_client
        self._sme = sme
        self._hybrid = hybrid
        self._gate = intent_gate  # src.retrieval.intent_gating.IntentGate

    def retrieve(self, *, query: str, subscription_id: str, profile_id: str,
                 query_understanding: Dict[str, Any],
                 flags: Dict[str, bool],
                 top_k_overrides: Optional[Dict[str, int]] = None) -> RetrievalBundle:
        import time
        intent = (query_understanding or {}).get("intent", "lookup")
        gate = self._gate.decide(intent) if self._gate else _default_gate(intent)
        top_k = _resolve_top_k(intent, top_k_overrides)

        from src.config.feature_flags import (
            get_flag_resolver,
            ENABLE_SME_RETRIEVAL, ENABLE_HYBRID_RETRIEVAL,
            ENABLE_KG_SYNTHESIZED_EDGES,
        )
        resolver = get_flag_resolver()

        bundle = RetrievalBundle()
        jobs = {}
        with ThreadPoolExecutor(max_workers=4, thread_name_prefix="dw-retrieve") as ex:
            if gate.run_a:
                jobs[ex.submit(self._layer_a, query, profile_id, subscription_id,
                                query_understanding, top_k.a, flags)] = ("layer_a", time.perf_counter())
            # Layer B (KG) runs whenever the gate says so — it is independent of
            # the SME retrieval master flag (KG direct-edges ship without SME).
            # Synthesized edges are filtered inside retrieve_layer_b by the
            # enable_kg_synthesized_edges flag.
            if gate.run_b:
                jobs[ex.submit(self._layer_b, query, profile_id, subscription_id,
                                query_understanding, top_k.b, flags)] = ("layer_b", time.perf_counter())
            # Layer C (SME artifacts) only runs when enable_sme_retrieval is on
            # for this subscription.
            if gate.run_c and resolver.is_enabled(subscription_id, ENABLE_SME_RETRIEVAL):
                jobs[ex.submit(self._layer_c, query, profile_id, subscription_id,
                                query_understanding, top_k.c)] = ("layer_c", time.perf_counter())
            # Layer D is a placeholder in Phase 3 — Phase 5 wires URL fetch.
            jobs[ex.submit(self._layer_d_placeholder)] = ("layer_d", time.perf_counter())

            # NO TIMEOUT on as_completed — memory rule, no internal wall-clock aborts.
            for fut in as_completed(jobs):
                name, t0 = jobs[fut]
                try:
                    result = fut.result()
                    setattr(bundle, f"{name}_" + ("chunks" if name == "layer_a"
                                                  else "kg" if name == "layer_b"
                                                  else "sme" if name == "layer_c"
                                                  else "url"), result)
                except Exception as e:
                    logger.warning("%s degraded: %s", name, e)
                    bundle.degraded_layers.append(name)
                bundle.per_layer_ms[name] = (time.perf_counter() - t0) * 1000.0

        return bundle

    # --- layer implementations -------------------------------------------

    def _layer_a(self, query, profile_id, subscription_id, qu, k, flags):
        from src.config.feature_flags import get_flag_resolver, ENABLE_HYBRID_RETRIEVAL
        hybrid_on = get_flag_resolver().is_enabled(subscription_id, ENABLE_HYBRID_RETRIEVAL)
        if hybrid_on and self._hybrid is not None:
            return self._hybrid.search(query=query, subscription_id=subscription_id,
                                       profile_id=profile_id,
                                       query_understanding=qu, top_k=k)
        return _dense_only_search(self._qdrant, query, subscription_id, profile_id, qu, k)

    def _layer_b(self, query, profile_id, subscription_id, qu, k, flags):
        # Canonical KG routing per ERRATA §7 and Phase 2's published surface:
        # UnifiedRetriever.retrieve_layer_b handles direct + inferred edges and
        # the enable_kg_synthesized_edges flag is read inside that helper.
        # The 0.6 floor is the canonical default from Phase 2
        # `inferred_edge_confidence_floor()`; adapter-tunable.
        return self.retrieve_layer_b(
            query=query, subscription_id=subscription_id, profile_id=profile_id,
            top_k=k, include_inferred=True, inferred_confidence_floor=0.6,
        )

    def _layer_c(self, query, profile_id, subscription_id, qu, k):
        hits = self._sme.retrieve(
            query=query, subscription_id=subscription_id, profile_id=profile_id,
            top_k=k, artifact_types=None,
        )
        return [_sme_hit_to_dict(h) for h in hits]

    def _layer_d_placeholder(self):
        # Phase 5 replaces this with the URL-ephemeral pipeline.
        return []
```

`retrieve_layer_b` is the Phase 2 helper on `UnifiedRetriever` (ERRATA §7). It filters `INFERRED_RELATION` edges by `subscription_id`, `profile_id`, and `min_confidence` when `enable_kg_synthesized_edges` is on for the subscription, and always returns direct edges.

Helpers `_resolve_top_k`, `_default_gate`, `_dense_only_search`, `_sme_hit_to_dict` live at module bottom (~25 lines total; trivial). The `top_k` table follows spec Section 7 — Task 6 tunes the concrete values.

- [ ] **Step 3: Run the failing tests green**

`pytest tests/retrieval/test_unified_retriever_four_layer.py -v` → PASS all six.

- [ ] **Step 4: Commit**

```bash
git add src/retrieval/unified_retriever.py tests/retrieval/test_unified_retriever_four_layer.py
git commit -m "phase3(sme-retrieval): 4-layer parallel orchestration, no timeouts"
```

---

## Task 4 — Merge, cross-encoder rerank, MMR diversity

**Files:**
- Modify: `src/retrieval/reranker.py`
- Create: `src/retrieval/mmr.py`
- Create: `tests/retrieval/test_reranker_cross_encoder.py`
- Create: `tests/retrieval/test_reranker_mmr.py`

This is the second load-bearing piece. After all layers return, Phase 3 must:
1. **Merge** the four layer outputs into a single candidate list, deduplicating on an equivalence key so a chunk that surfaces in both Layer A and Layer C's evidence refs isn't double-counted.
2. **Cross-encoder rerank** the top-40 merged candidates into a top-10 by semantic relevance to the query.
3. **MMR** over that top-10 to trade relevance against diversity with adapter-configurable lambda (default 0.7).

- [ ] **Step 1: Write failing tests — cross-encoder + MMR**

Create `tests/retrieval/test_reranker_cross_encoder.py`:

```python
"""Cross-encoder rerank + SME-aware score blend."""
from unittest.mock import MagicMock
import pytest
from src.retrieval.reranker import rerank_merged_candidates


def _cand(text, layer, score, confidence=0.9, doc_id="d", chunk_id="c"):
    return {"text": text, "layer": layer, "raw_score": score,
            "confidence": confidence, "doc_id": doc_id, "chunk_id": chunk_id}


def test_cross_encoder_reshuffles_top_k():
    ce = MagicMock()
    # Reverse order: CE thinks candidate with lowest raw_score is actually best.
    ce.predict.return_value = [0.1, 0.4, 0.9]
    cands = [
        _cand("irrelevant", "layer_a", 0.95, doc_id="d1", chunk_id="c1"),
        _cand("middling",   "layer_a", 0.80, doc_id="d2", chunk_id="c2"),
        _cand("very relevant", "layer_a", 0.60, doc_id="d3", chunk_id="c3"),
    ]
    ranked = rerank_merged_candidates(
        query="x", candidates=cands, cross_encoder=ce, top_k=3,
    )
    assert ranked[0]["chunk_id"] == "c3"  # CE picked this
    assert ranked[2]["chunk_id"] == "c1"


def test_cross_encoder_bypassed_when_flag_off():
    ce = MagicMock()
    cands = [_cand("a", "layer_a", 0.9, chunk_id="c1"),
             _cand("b", "layer_a", 0.8, chunk_id="c2")]
    ranked = rerank_merged_candidates(
        query="x", candidates=cands, cross_encoder=ce, top_k=2,
        enable_cross_encoder=False,
    )
    ce.predict.assert_not_called()
    assert [r["chunk_id"] for r in ranked] == ["c1", "c2"]


def test_sme_layer_gets_bonus_on_analytical_intent():
    ce = MagicMock()
    ce.predict.return_value = [0.7, 0.7]  # tied on CE
    cands = [
        _cand("chunk", "layer_a", 0.9, chunk_id="a1"),
        _cand("dossier", "layer_c", 0.7, confidence=0.92, chunk_id="s1"),
    ]
    ranked = rerank_merged_candidates(
        query="analyze trends", candidates=cands, cross_encoder=ce,
        top_k=2, intent="analyze",
    )
    assert ranked[0]["chunk_id"] == "s1"


def test_merge_dedups_on_evidence_key():
    from src.retrieval.reranker import merge_layers
    a = [{"layer": "layer_a", "doc_id": "d1", "chunk_id": "c1", "text": "t", "raw_score": 0.8}]
    c = [{"layer": "layer_c", "doc_id": "d1", "chunk_id": "c1", "text": "t",
          "raw_score": 0.6, "artifact_type": "dossier"}]
    merged = merge_layers(a=a, b=[], c=c, d=[])
    assert len(merged) == 1
    # When Layer A and C have the same evidence, merged item keeps Layer C flag
    # so the pack assembler can mark it as SME-backed.
    assert merged[0].get("sme_backed") is True


def test_merge_marks_kg_inferred_as_sme_backed():
    """Per ERRATA §11: Layer B items with kind=='kg_inferred' are SME-backed."""
    from src.retrieval.reranker import merge_layers
    b_direct = {"layer": "layer_b", "kind": "kg_direct",
                "from": "n1", "to": "n2", "relation_type": "cites",
                "confidence": 0.95, "text": "n1 cites n2"}
    b_inferred = {"layer": "layer_b", "kind": "kg_inferred",
                  "from": "n1", "to": "n3", "relation_type": "correlates_with",
                  "confidence": 0.82, "text": "n1 correlates_with n3"}
    merged = merge_layers(a=[], b=[b_direct, b_inferred], c=[], d=[])
    by_relation = {m["relation_type"]: m for m in merged}
    assert by_relation["cites"].get("sme_backed") is not True
    assert by_relation["correlates_with"].get("sme_backed") is True
```

Create `tests/retrieval/test_reranker_mmr.py`:

```python
"""MMR diversity selection."""
import numpy as np
import pytest
from src.retrieval.mmr import mmr_select


def test_mmr_picks_diverse_when_lambda_low():
    # Two clusters of near-identical embeddings; MMR should span both.
    items = [
        {"id": "a1", "embedding": [1.0, 0.0], "score": 0.95},
        {"id": "a2", "embedding": [0.99, 0.01], "score": 0.93},
        {"id": "b1", "embedding": [0.0, 1.0], "score": 0.90},
    ]
    picks = mmr_select(items=items, top_k=2, lam=0.3)
    ids = [p["id"] for p in picks]
    assert "a1" in ids and "b1" in ids


def test_mmr_picks_highest_score_when_lambda_one():
    items = [
        {"id": "a1", "embedding": [1.0, 0.0], "score": 0.99},
        {"id": "a2", "embedding": [0.99, 0.01], "score": 0.98},
        {"id": "b1", "embedding": [0.0, 1.0], "score": 0.50},
    ]
    picks = mmr_select(items=items, top_k=2, lam=1.0)
    ids = [p["id"] for p in picks]
    assert ids == ["a1", "a2"]


def test_mmr_handles_missing_embeddings():
    items = [
        {"id": "a1", "score": 0.9},   # no embedding → degraded to no-op diversity
        {"id": "a2", "score": 0.8},
    ]
    picks = mmr_select(items=items, top_k=2, lam=0.5)
    assert len(picks) == 2
```

Run both test files — all failing.

- [ ] **Step 2: Implement `merge_layers` + `rerank_merged_candidates`**

Append to `src/retrieval/reranker.py` (keep the existing `rerank_chunks` function; Phase 3 calls the new one):

```python
# src/retrieval/reranker.py  (Phase 3 additions)
from typing import Any, Dict, Iterable, List, Optional

_SME_INTENT_BONUS = {"analyze": 0.08, "diagnose": 0.08, "recommend": 0.10,
                     "investigate": 0.05}


def merge_layers(*, a: List[Dict], b: List[Dict], c: List[Dict],
                 d: List[Dict]) -> List[Dict]:
    """Union layer outputs, dedup on (doc_id, chunk_id) keeping SME flag on overlap.

    Per ERRATA §11: Layer C items AND Layer B items with ``kind == "kg_inferred"``
    are SME-backed (both represent pre-reasoned synthesis output). Layer B direct
    edges are NOT SME-backed.
    """
    seen: Dict[tuple, Dict] = {}
    for layer_name, items in (("layer_a", a), ("layer_b", b), ("layer_c", c), ("layer_d", d)):
        for it in items or []:
            it = dict(it)
            it.setdefault("layer", layer_name)
            is_sme_source = (
                layer_name == "layer_c"
                or (layer_name == "layer_b" and it.get("kind") == "kg_inferred")
            )
            key = (it.get("doc_id"), it.get("chunk_id"))
            if key in seen and key != (None, None):
                # If same evidence appears via an SME source, flag the existing item.
                if is_sme_source:
                    seen[key]["sme_backed"] = True
                continue
            if is_sme_source:
                it["sme_backed"] = True
            seen[key if key != (None, None) else (layer_name, id(it))] = it
    return list(seen.values())


def rerank_merged_candidates(
    *, query: str, candidates: List[Dict], cross_encoder: Any = None,
    top_k: int = 10, intent: str = "lookup",
    enable_cross_encoder: bool = True,
) -> List[Dict]:
    """Blend raw layer score, CE relevance, and SME-intent bonus."""
    if not candidates:
        return []
    if not (enable_cross_encoder and cross_encoder is not None):
        return sorted(candidates, key=lambda c: c.get("raw_score", 0.0),
                      reverse=True)[:top_k]
    pool = sorted(candidates, key=lambda c: c.get("raw_score", 0.0),
                  reverse=True)[: max(40, top_k * 4)]
    pairs = [(query, (c.get("text") or c.get("narrative") or "")[:1600]) for c in pool]
    ce_scores = cross_encoder.predict(pairs)
    bonus = _SME_INTENT_BONUS.get(intent, 0.0)
    for c, ce in zip(pool, ce_scores):
        blended = 0.6 * float(ce) + 0.3 * c.get("raw_score", 0.0) \
                  + 0.1 * c.get("confidence", 0.5)
        if c.get("layer") == "layer_c" or c.get("sme_backed"):
            blended += bonus
        c["rerank_score"] = blended
    return sorted(pool, key=lambda c: c["rerank_score"], reverse=True)[:top_k]
```

- [ ] **Step 3: Implement `mmr.py`**

Create `src/retrieval/mmr.py`:

```python
"""Maximal marginal relevance selection."""
from __future__ import annotations
import math
from typing import Any, Dict, List


def _cos(a, b) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def mmr_select(*, items: List[Dict[str, Any]], top_k: int,
               lam: float = 0.7) -> List[Dict[str, Any]]:
    """Pick top_k with trade-off between score and diversity (lam=1 all score)."""
    if not items:
        return []
    have_emb = all("embedding" in it for it in items)
    if not have_emb or top_k >= len(items):
        return sorted(items, key=lambda x: x.get("score", 0.0), reverse=True)[:top_k]
    selected: List[Dict] = []
    remaining = sorted(items, key=lambda x: x.get("score", 0.0), reverse=True)
    while remaining and len(selected) < top_k:
        best, best_score = None, -math.inf
        for cand in remaining:
            rel = cand.get("score", 0.0)
            div = 0.0
            if selected:
                div = max(_cos(cand["embedding"], s["embedding"]) for s in selected)
            mmr = lam * rel - (1.0 - lam) * div
            if mmr > best_score:
                best, best_score = cand, mmr
        selected.append(best)
        remaining = [r for r in remaining if r is not best]
    return selected
```

- [ ] **Step 4: Run tests green**

`pytest tests/retrieval/test_reranker_cross_encoder.py tests/retrieval/test_reranker_mmr.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/retrieval/reranker.py src/retrieval/mmr.py \
        tests/retrieval/test_reranker_cross_encoder.py \
        tests/retrieval/test_reranker_mmr.py
git commit -m "phase3(sme-rerank): cross-encoder blend + MMR + layer merge"
```

---

## Task 5 — Budget-aware pack assembly

**Files:**
- Create: `src/retrieval/pack_assembler.py`
- Create: `tests/retrieval/test_pack_assembler_budget.py`
- Create: `tests/retrieval/test_pack_assembler_drop_order.py`

Third load-bearing piece. The reranked + MMR-selected candidates must be packed under the adapter's per-intent `max_pack_tokens` budget. Overflow is **dropped lowest-confidence first**, Layer C SME artifacts are **compressed to key-claims + evidence refs** rather than included as full narratives, and a semantic dedup pass removes near-duplicate items that the merge step missed (e.g., two Layer A chunks that paraphrase the same fact).

- [ ] **Step 1: Write failing tests**

Create `tests/retrieval/test_pack_assembler_budget.py`:

```python
"""Pack assembler enforces adapter max_pack_tokens."""
from types import SimpleNamespace
import pytest
from src.retrieval.pack_assembler import PackAssembler, PackedItem


def _adapter(caps):
    return SimpleNamespace(retrieval_caps={"max_pack_tokens": caps})


def _item(text, conf=0.9, layer="layer_a", tokens=None):
    it = {"text": text, "confidence": conf, "layer": layer,
          "doc_id": "d1", "chunk_id": "c1", "rerank_score": 0.8}
    if tokens is not None:
        it["_tokens"] = tokens
    return it


def test_pack_under_budget_keeps_all():
    ad = _adapter({"analyze": 1000})
    pa = PackAssembler(adapter=ad)
    items = [_item("short a", tokens=50), _item("short b", tokens=50)]
    pack = pa.assemble(items=items, intent="analyze")
    assert len(pack) == 2


def test_pack_over_budget_drops_lowest_confidence_first():
    ad = _adapter({"analyze": 200})
    pa = PackAssembler(adapter=ad)
    items = [
        _item("a", conf=0.95, tokens=120),
        _item("b", conf=0.50, tokens=120),  # should be dropped
        _item("c", conf=0.80, tokens=60),
    ]
    pack = pa.assemble(items=items, intent="analyze")
    ids = [p.text for p in pack]
    assert "a" in ids and "c" in ids
    assert "b" not in ids


def test_layer_c_items_compressed_not_full_narrative():
    ad = _adapter({"analyze": 4000})
    long_narrative = "narrative " * 500  # ~1000 tokens
    pa = PackAssembler(adapter=ad)
    it = {
        "layer": "layer_c", "artifact_type": "dossier",
        "narrative": long_narrative,
        "key_claims": ["Q3 revenue up 12%", "margin expanded 2pts"],
        "evidence": ["d1#c1", "d2#c2"],
        "confidence": 0.9,
        "rerank_score": 0.8,
    }
    pack = pa.assemble(items=[it], intent="analyze")
    assert len(pack) == 1
    # Compressed form keeps key_claims not the whole narrative.
    assert "Q3 revenue" in pack[0].text
    assert len(pack[0].text) < len(long_narrative)


def test_provenance_preserved_on_every_item():
    ad = _adapter({"analyze": 1000})
    pa = PackAssembler(adapter=ad)
    items = [_item("a", tokens=40)]
    pack = pa.assemble(items=items, intent="analyze")
    assert pack[0].provenance == [("d1", "c1")]


def test_unknown_intent_falls_back_to_generic_cap():
    ad = SimpleNamespace(retrieval_caps={"max_pack_tokens": {"generic": 500}})
    pa = PackAssembler(adapter=ad)
    pack = pa.assemble(
        items=[_item("x", tokens=200), _item("y", tokens=200), _item("z", tokens=200)],
        intent="exotic",
    )
    # 500-token budget fits 2 of 3.
    assert len(pack) == 2
```

Create `tests/retrieval/test_pack_assembler_drop_order.py`:

```python
"""Drop order is deterministic: lowest confidence, then lowest rerank_score."""
from types import SimpleNamespace
from src.retrieval.pack_assembler import PackAssembler


def test_drop_order_is_stable():
    ad = SimpleNamespace(retrieval_caps={"max_pack_tokens": {"analyze": 100}})
    pa = PackAssembler(adapter=ad)
    items = [
        {"text": "hi", "_tokens": 60, "confidence": 0.5, "rerank_score": 0.9,
         "doc_id": "d", "chunk_id": "c1"},
        {"text": "hj", "_tokens": 60, "confidence": 0.5, "rerank_score": 0.3,
         "doc_id": "d", "chunk_id": "c2"},
        {"text": "hk", "_tokens": 60, "confidence": 0.9, "rerank_score": 0.4,
         "doc_id": "d", "chunk_id": "c3"},
    ]
    pack = pa.assemble(items=items, intent="analyze")
    # Budget fits one. Highest confidence wins; rerank breaks ties lower down.
    assert len(pack) == 1
    assert pack[0].text == "hk"
```

Run — all failing.

- [ ] **Step 2: Implement `pack_assembler.py`**

```python
"""Budget-aware pack assembly with adapter-driven per-intent caps."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


def _approx_tokens(text: str) -> int:
    # 4 chars ≈ 1 token heuristic; good enough for budget gating.
    return max(1, len(text or "") // 4)


@dataclass
class PackedItem:
    text: str
    provenance: List[Tuple[str, str]]
    layer: str
    confidence: float
    rerank_score: float = 0.0
    sme_backed: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class PackAssembler:
    """Assembles a token-budgeted pack from reranked retrieval candidates."""

    def __init__(self, adapter):
        caps = (adapter.retrieval_caps or {}).get("max_pack_tokens", {})
        self._caps = caps
        self._default = caps.get("generic", 4000)

    def _budget_for(self, intent: str) -> int:
        return int(self._caps.get(intent, self._default))

    def _compress_sme(self, it: Dict[str, Any]) -> str:
        claims = it.get("key_claims") or []
        if not claims and it.get("narrative"):
            claims = [it["narrative"][:400]]
        ev = it.get("evidence") or []
        body = " | ".join(claims[:8])
        refs = " ".join(f"[{e}]" for e in ev[:6])
        a_type = it.get("artifact_type", "sme")
        return f"[SME/{a_type}] {body} {refs}".strip()

    def _expand_item(self, it: Dict[str, Any]) -> PackedItem:
        layer = it.get("layer", "layer_a")
        if layer == "layer_c":
            text = self._compress_sme(it)
        else:
            text = it.get("text") or it.get("narrative") or ""
        prov = []
        if it.get("doc_id") and it.get("chunk_id"):
            prov.append((it["doc_id"], it["chunk_id"]))
        for e in it.get("evidence", []) or []:
            if "#" in e:
                d, c = e.split("#", 1); prov.append((d, c))
        return PackedItem(
            text=text, provenance=prov, layer=layer,
            confidence=float(it.get("confidence", 0.5)),
            rerank_score=float(it.get("rerank_score", 0.0)),
            sme_backed=bool(it.get("sme_backed") or layer == "layer_c"),
            metadata={"artifact_type": it.get("artifact_type"),
                      "relation_type": it.get("relation_type")},
        )

    def assemble(self, *, items: List[Dict[str, Any]], intent: str) -> List[PackedItem]:
        if not items:
            return []
        budget = self._budget_for(intent)
        # Expand first so SME compression shortens tokens before gating.
        expanded: List[Tuple[PackedItem, int]] = []
        for raw in items:
            it = self._expand_item(raw)
            tokens = raw.get("_tokens", _approx_tokens(it.text))
            expanded.append((it, int(tokens)))
        # Sort by rank of (confidence desc, rerank_score desc) so drop cuts tail.
        expanded.sort(key=lambda t: (t[0].confidence, t[0].rerank_score), reverse=True)
        picked: List[PackedItem] = []
        used = 0
        for it, tk in expanded:
            if used + tk > budget:
                continue
            picked.append(it)
            used += tk
        return _semantic_dedup(picked)


def _semantic_dedup(items: List[PackedItem]) -> List[PackedItem]:
    """Drop near-duplicate texts; prefers SME-backed, then higher confidence."""
    kept: List[PackedItem] = []
    for it in items:
        dup = False
        for k in kept:
            if _overlap(it.text, k.text) > 0.85:
                dup = True
                break
        if not dup:
            kept.append(it)
    return kept


def _overlap(a: str, b: str) -> float:
    ta, tb = set(a.lower().split()), set(b.lower().split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta), len(tb))
```

- [ ] **Step 3: Green tests**

`pytest tests/retrieval/test_pack_assembler_budget.py tests/retrieval/test_pack_assembler_drop_order.py -v` → PASS.

- [ ] **Step 4: Commit**

```bash
git add src/retrieval/pack_assembler.py tests/retrieval/test_pack_assembler_budget.py \
        tests/retrieval/test_pack_assembler_drop_order.py
git commit -m "phase3(sme-pack): budget-aware pack assembly + SME compression"
```

---

## Task 6 — Adaptive top-K per intent/complexity

**Files:**
- Modify: `src/retrieval/unified_retriever.py` (`_resolve_top_k` helper)
- Create: `tests/retrieval/test_unified_retriever_top_k.py`

Spec Section 7 fixes the nominal top-K table per query shape. Phase 3 implements the table and exposes it as a dataclass so Task 13 can tune it off eval data without touching the orchestrator.

- [ ] **Step 1: Write failing tests**

Create `tests/retrieval/test_unified_retriever_top_k.py`:

```python
"""Top-K resolution per intent."""
import pytest
from src.retrieval.unified_retriever import _resolve_top_k, LayerTopK


@pytest.mark.parametrize("intent,expected", [
    ("lookup",      LayerTopK(a=5, b=0, c=0)),
    ("extract",     LayerTopK(a=10, b=0, c=2)),
    ("compare",     LayerTopK(a=12, b=5, c=5)),
    ("summarize",   LayerTopK(a=12, b=5, c=5)),
    ("analyze",     LayerTopK(a=15, b=10, c=10)),
    ("diagnose",    LayerTopK(a=15, b=10, c=10)),
    ("recommend",   LayerTopK(a=15, b=10, c=10)),
    ("investigate", LayerTopK(a=15, b=10, c=10)),
])
def test_top_k_table_matches_spec(intent, expected):
    got = _resolve_top_k(intent, overrides=None)
    assert got == expected


def test_unknown_intent_falls_back_to_lookup_table():
    assert _resolve_top_k("exotic", overrides=None) == LayerTopK(a=5, b=0, c=0)


def test_overrides_win():
    got = _resolve_top_k("analyze", overrides={"a": 20, "c": 15})
    assert got.a == 20 and got.c == 15 and got.b == 10  # b unchanged
```

- [ ] **Step 2: Implement**

Append to `src/retrieval/unified_retriever.py`:

```python
@dataclass(frozen=True)
class LayerTopK:
    a: int
    b: int
    c: int


_TOP_K_TABLE: Dict[str, LayerTopK] = {
    "lookup":      LayerTopK(5, 0, 0),
    "greeting":    LayerTopK(0, 0, 0),
    "identity":    LayerTopK(0, 0, 0),
    "count":       LayerTopK(5, 0, 0),
    "extract":     LayerTopK(10, 0, 2),
    "list":        LayerTopK(10, 0, 2),
    "aggregate":   LayerTopK(10, 0, 2),
    "compare":     LayerTopK(12, 5, 5),
    "summarize":   LayerTopK(12, 5, 5),
    "overview":    LayerTopK(12, 5, 5),
    "analyze":     LayerTopK(15, 10, 10),
    "diagnose":    LayerTopK(15, 10, 10),
    "recommend":   LayerTopK(15, 10, 10),
    "investigate": LayerTopK(15, 10, 10),
}


def _resolve_top_k(intent: str, overrides: Optional[Dict[str, int]]) -> LayerTopK:
    base = _TOP_K_TABLE.get(intent, _TOP_K_TABLE["lookup"])
    if not overrides:
        return base
    return LayerTopK(a=overrides.get("a", base.a),
                     b=overrides.get("b", base.b),
                     c=overrides.get("c", base.c))
```

- [ ] **Step 3: Green + Commit**

```bash
pytest tests/retrieval/test_unified_retriever_top_k.py -v
git add src/retrieval/unified_retriever.py tests/retrieval/test_unified_retriever_top_k.py
git commit -m "phase3(sme-topk): adaptive per-intent top-K table"
```

---

## Task 7 — QA-cache fast path

**Files:**
- Create: `src/retrieval/qa_fast_path.py`
- Create: `tests/retrieval/test_qa_fast_path.py`
- Modify: `src/agent/core_agent.py` (1-call integration; Task 11 handles the bulk)

Phase 1 already emits Q&A pairs into Redis at ingestion (`src/intelligence/qa_generator.py` → `to_redis_format`). Phase 3 turns that cache into a **pre-grounded-answer fast path** that short-circuits the entire retrieval + reasoning pipeline when a near-exact query match exists.

**Index prerequisite (cross-ref ERRATA §13):** the `qa_idx:{sub}:{prof}:{fingerprint}` keys that `QAFastPath.lookup` reads are emitted by Phase 2's `qa_generator.py` when Q&A pairs are produced. Phase 3 **assumes** these keys exist. If they do not (Phase 2 Task wasn't implemented, or a profile was synthesized before the emission was added), Phase 3 must populate them once as a backfill — see Step 1.5 below.

- [ ] **Step 1: Write failing tests**

```python
# tests/retrieval/test_qa_fast_path.py
"""QA-cache fast path short-circuits the Reasoner when a pre-grounded answer exists."""
from unittest.mock import MagicMock
import pytest
from src.retrieval.qa_fast_path import QAFastPath, QAFastPathHit


def test_returns_hit_on_exact_query_fingerprint():
    redis = MagicMock()
    redis.get.return_value = (
        b'{"question": "What is our Q3 revenue?", '
        b'"answer": "Q3 revenue was $12.4M.", '
        b'"confidence": 0.95, "source_section_id": "sec_7", '
        b'"source_entities": ["Q3", "revenue"]}'
    )
    fp = QAFastPath(redis_client=redis, min_confidence=0.85)
    hit = fp.lookup(query="What is our Q3 revenue?",
                    subscription_id="s1", profile_id="p1")
    assert isinstance(hit, QAFastPathHit)
    assert hit.answer == "Q3 revenue was $12.4M."
    assert hit.confidence >= 0.85


def test_miss_returns_none():
    redis = MagicMock()
    redis.get.return_value = None
    fp = QAFastPath(redis_client=redis, min_confidence=0.85)
    assert fp.lookup(query="q", subscription_id="s", profile_id="p") is None


def test_skips_low_confidence_hits():
    redis = MagicMock()
    redis.get.return_value = b'{"question": "q", "answer": "a", "confidence": 0.5}'
    fp = QAFastPath(redis_client=redis, min_confidence=0.85)
    assert fp.lookup(query="q", subscription_id="s", profile_id="p") is None


def test_key_is_per_profile():
    redis = MagicMock()
    redis.get.return_value = None
    fp = QAFastPath(redis_client=redis)
    fp.lookup(query="q", subscription_id="s1", profile_id="p1")
    key_seen = redis.get.call_args.args[0].decode() if isinstance(
        redis.get.call_args.args[0], bytes) else redis.get.call_args.args[0]
    assert "s1" in key_seen and "p1" in key_seen
```

- [ ] **Step 1.5: Verify the `qa_idx:` index exists; backfill if missing**

Before the fast path can return anything, the index Phase 2 should have emitted must exist. Probe the sandbox:

```bash
python -c "
from src.cache.redis_store import get_redis_client
r = get_redis_client()
n = sum(1 for _ in r.scan_iter(match='qa_idx:REPLACE_SUB:*'))
print('qa_idx entries:', n)
"
```

If `0`, run this one-shot backfill (do not commit — it is an operator step):

```bash
python -c "
import json
from src.cache.redis_store import get_redis_client
from src.retrieval.qa_fast_path import _fingerprint
r = get_redis_client()
for qa_key in r.scan_iter(match='qa:*:*'):
    raw = r.get(qa_key)
    if not raw: continue
    qa = json.loads(raw)
    sub = qa.get('subscription_id'); prof = qa.get('profile_id')
    q = qa.get('question'); qa_id = qa.get('qa_id') or qa_key.decode().split(':')[-1]
    if not (sub and prof and q): continue
    idx_key = f'qa_idx:{sub}:{prof}:{_fingerprint(q)}'
    r.set(idx_key, json.dumps({**qa, 'qa_id': qa_id}))
print('backfill complete')
"
```

Long-term the emission lives in Phase 2's `qa_generator.py` (ERRATA §13). This backfill is for profiles synthesized before the emission was wired.

- [ ] **Step 2: Implement `qa_fast_path.py`**

```python
"""Short-circuit the hot path when a pre-grounded Q&A pair exists in Redis.

Reuses the qa:{doc}:{qa_id} entries written by src.intelligence.qa_generator.
A separate index qa_idx:{sub}:{prof}:{fingerprint} → qa_id lets us hit by
query fingerprint without scanning.
"""
from __future__ import annotations
import hashlib
import json
import logging
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


def _fingerprint(text: str) -> str:
    return hashlib.sha256(text.strip().lower().encode()).hexdigest()[:20]


@dataclass
class QAFastPathHit:
    answer: str
    confidence: float
    source_section_id: Optional[str]
    source_entities: List[str]
    qa_id: str


class QAFastPath:
    def __init__(self, redis_client, min_confidence: float = 0.85):
        self._redis = redis_client
        self._min = min_confidence

    def lookup(self, *, query: str, subscription_id: str,
               profile_id: str) -> Optional[QAFastPathHit]:
        if self._redis is None:
            return None
        idx_key = f"qa_idx:{subscription_id}:{profile_id}:{_fingerprint(query)}"
        try:
            raw = self._redis.get(idx_key)
            if raw is None:
                return None
            if isinstance(raw, (bytes, bytearray)):
                raw = raw.decode("utf-8")
            data = json.loads(raw)
            if float(data.get("confidence", 0)) < self._min:
                return None
            return QAFastPathHit(
                answer=data["answer"],
                confidence=float(data["confidence"]),
                source_section_id=data.get("source_section_id"),
                source_entities=data.get("source_entities", []),
                qa_id=data.get("qa_id", ""),
            )
        except Exception:
            logger.debug("qa_fast_path lookup failed", exc_info=True)
            return None
```

- [ ] **Step 3: Wire call site in `core_agent.py`**

Task 11 threads this through the agent; here, just add the import and a stubbed call at the top of `CoreAgent.handle()` protected by a flag so earlier tests don't regress. The integration tests are in Task 11.

- [ ] **Step 4: Commit**

```bash
pytest tests/retrieval/test_qa_fast_path.py -v
git add src/retrieval/qa_fast_path.py tests/retrieval/test_qa_fast_path.py
git commit -m "phase3(sme-qa-cache): qa fast-path short-circuit"
```

---

## Task 8 — Redis retrieval cache

**Files:**
- Create: `src/retrieval/retrieval_cache.py`
- Create: `tests/retrieval/test_retrieval_cache.py`

Spec Section 7 mechanism #6: memoize the merged retrieval pack for 5 minutes so near-duplicate queries don't re-hit Qdrant+Neo4j+SME. Key is `(sub, prof, query_fingerprint, flag_set_version)` so a flag flip invalidates naturally.

- [ ] **Step 1: Write failing tests**

```python
# tests/retrieval/test_retrieval_cache.py
from unittest.mock import MagicMock
import pytest
from src.retrieval.retrieval_cache import RetrievalCache


def test_set_then_get_roundtrips():
    r = MagicMock()
    store = {}
    r.setex.side_effect = lambda k, ttl, v: store.__setitem__(k, v)
    r.get.side_effect = lambda k: store.get(k)
    cache = RetrievalCache(redis_client=r, ttl_seconds=300)
    pack = [{"text": "a", "layer": "layer_a", "confidence": 0.9}]
    cache.set(subscription_id="s", profile_id="p", query="q",
              flag_set_version="v1", pack=pack)
    got = cache.get(subscription_id="s", profile_id="p", query="q",
                    flag_set_version="v1")
    assert got == pack


def test_miss_on_different_flag_version():
    r = MagicMock()
    store = {}
    r.setex.side_effect = lambda k, ttl, v: store.__setitem__(k, v)
    r.get.side_effect = lambda k: store.get(k)
    cache = RetrievalCache(redis_client=r)
    cache.set(subscription_id="s", profile_id="p", query="q",
              flag_set_version="v1", pack=[{"text": "x"}])
    assert cache.get(subscription_id="s", profile_id="p", query="q",
                     flag_set_version="v2") is None


def test_miss_on_different_profile():
    r = MagicMock(); store = {}
    r.setex.side_effect = lambda k, ttl, v: store.__setitem__(k, v)
    r.get.side_effect = lambda k: store.get(k)
    cache = RetrievalCache(redis_client=r)
    cache.set(subscription_id="s", profile_id="p1", query="q",
              flag_set_version="v1", pack=[{"text": "x"}])
    assert cache.get(subscription_id="s", profile_id="p2", query="q",
                     flag_set_version="v1") is None


def test_get_returns_none_when_redis_unavailable():
    cache = RetrievalCache(redis_client=None)
    assert cache.get(subscription_id="s", profile_id="p", query="q",
                     flag_set_version="v") is None


def test_invalidate_on_training_complete():
    r = MagicMock()
    r.scan_iter.return_value = [b"dwx:retrieval:s:p:xxx:v1"]
    cache = RetrievalCache(redis_client=r)
    cache.invalidate_profile(subscription_id="s", profile_id="p")
    r.delete.assert_called()
```

- [ ] **Step 2: Implement**

```python
# src/retrieval/retrieval_cache.py
"""Redis-backed retrieval-pack cache. TTL 5m. Key includes flag-set version
so any flag flip invalidates naturally.
"""
from __future__ import annotations
import hashlib
import json
import logging
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


def _fp(text: str) -> str:
    return hashlib.sha256(text.strip().lower().encode()).hexdigest()[:20]


class RetrievalCache:
    PREFIX = "dwx:retrieval"

    def __init__(self, redis_client=None, ttl_seconds: int = 300):
        self._redis = redis_client
        self._ttl = ttl_seconds

    def _key(self, *, subscription_id: str, profile_id: str,
             query: str, flag_set_version: str) -> str:
        return (f"{self.PREFIX}:{subscription_id}:{profile_id}:"
                f"{_fp(query)}:{flag_set_version}")

    def get(self, **kw) -> Optional[List[dict]]:
        if self._redis is None:
            return None
        try:
            raw = self._redis.get(self._key(**kw))
            if raw is None:
                return None
            if isinstance(raw, (bytes, bytearray)):
                raw = raw.decode("utf-8")
            return json.loads(raw)
        except Exception:
            logger.debug("retrieval_cache.get failed", exc_info=True)
            return None

    def set(self, *, subscription_id: str, profile_id: str,
            query: str, flag_set_version: str,
            pack: List[Any]) -> None:
        if self._redis is None:
            return
        try:
            key = self._key(subscription_id=subscription_id, profile_id=profile_id,
                             query=query, flag_set_version=flag_set_version)
            self._redis.setex(key, self._ttl, json.dumps(pack, default=str))
        except Exception:
            logger.debug("retrieval_cache.set failed", exc_info=True)

    def invalidate_profile(self, *, subscription_id: str, profile_id: str) -> None:
        if self._redis is None:
            return
        try:
            pattern = f"{self.PREFIX}:{subscription_id}:{profile_id}:*"
            for k in self._redis.scan_iter(match=pattern):
                self._redis.delete(k)
        except Exception:
            logger.debug("retrieval_cache.invalidate failed", exc_info=True)
```

- [ ] **Step 3: Green (cache unit tests) + Commit the cache module**

```bash
pytest tests/retrieval/test_retrieval_cache.py -v
git add src/retrieval/retrieval_cache.py tests/retrieval/test_retrieval_cache.py
git commit -m "phase3(sme-cache): redis retrieval cache + per-profile invalidate"
```

---

## Task 8.5 — Wire retrieval-cache invalidation into `PIPELINE_TRAINING_COMPLETED`

**Files:**
- Modify: `src/api/pipeline_api.py`
- Create: `tests/api/test_pipeline_cache_invalidation.py`

Per ERRATA §14 the previous Task 8 hand-waved the cache-invalidation hook. This task spells it out: the retrieval cache is keyed by `(sub, prof, query_fp, flag_set_version)`, and on `PIPELINE_TRAINING_COMPLETED` transitions new SME artifacts become visible. Any cached pack for that profile is now stale and MUST be evicted before the next query.

Memory-rule check: `pipeline_status` strings are immutable — this task only reads `PIPELINE_TRAINING_COMPLETED`, never renames or adds values.

- [ ] **Step 1: Locate the completion branch**

```bash
grep -n "PIPELINE_TRAINING_COMPLETED" src/api/pipeline_api.py
```

Expected: at least one branch where the status transition is written. If the constant lives in `src/api/document_status.py` under a different spelling, update the import accordingly.

- [ ] **Step 2: Write the failing integration test FIRST**

Create `tests/api/test_pipeline_cache_invalidation.py`:

```python
"""PIPELINE_TRAINING_COMPLETED must evict the retrieval cache for the profile."""
from unittest.mock import MagicMock, patch

import pytest


def test_training_complete_invalidates_retrieval_cache(monkeypatch):
    from src.api import pipeline_api
    from src.retrieval.retrieval_cache import RetrievalCache

    fake_redis = MagicMock()
    fake_redis.scan_iter.return_value = [
        b"dwx:retrieval:sub_A:prof_X:abc123:v1",
        b"dwx:retrieval:sub_A:prof_X:def456:v1",
    ]
    cache = RetrievalCache(redis_client=fake_redis)
    monkeypatch.setattr(pipeline_api, "get_retrieval_cache", lambda: cache)

    # Drive the handler that fires on the status transition.
    pipeline_api._on_pipeline_training_completed(
        subscription_id="sub_A", profile_id="prof_X",
    )
    assert fake_redis.delete.called
    deleted_keys = [c.args[0] for c in fake_redis.delete.call_args_list]
    assert all(b"sub_A:prof_X" in k for k in deleted_keys)


def test_training_complete_skips_when_cache_unavailable(monkeypatch):
    """If Redis is down, invalidation must not raise."""
    from src.api import pipeline_api
    from src.retrieval.retrieval_cache import RetrievalCache

    monkeypatch.setattr(pipeline_api, "get_retrieval_cache",
                        lambda: RetrievalCache(redis_client=None))
    # Must not raise.
    pipeline_api._on_pipeline_training_completed(
        subscription_id="sub_A", profile_id="prof_X",
    )
```

Run: `pytest tests/api/test_pipeline_cache_invalidation.py -v` → FAIL (hook not wired).

- [ ] **Step 3: Wire the hook**

In `src/api/pipeline_api.py`, next to the completion branch:

```python
from src.retrieval.retrieval_cache import RetrievalCache

_retrieval_cache_singleton: RetrievalCache | None = None


def get_retrieval_cache() -> RetrievalCache:
    global _retrieval_cache_singleton
    if _retrieval_cache_singleton is None:
        from src.cache.redis_store import get_redis_client
        _retrieval_cache_singleton = RetrievalCache(redis_client=get_redis_client())
    return _retrieval_cache_singleton


def _on_pipeline_training_completed(*, subscription_id: str, profile_id: str) -> None:
    """Invoked from the status-transition branch when pipeline_status enters
    PIPELINE_TRAINING_COMPLETED. Evicts cached retrieval packs for the profile
    so the next query re-retrieves against freshly-materialized SME artifacts.
    """
    try:
        get_retrieval_cache().invalidate_profile(
            subscription_id=subscription_id, profile_id=profile_id,
        )
    except Exception:
        # Cache invalidation is best-effort; log via the existing logger
        # but never block the training-completion transition itself.
        import logging
        logging.getLogger(__name__).warning(
            "retrieval_cache.invalidate_profile failed", exc_info=True,
        )
```

Then find the existing completion branch (where `pipeline_status` is written to `PIPELINE_TRAINING_COMPLETED`) and add one line immediately after the Mongo write:

```python
_on_pipeline_training_completed(
    subscription_id=subscription_id, profile_id=profile_id,
)
```

- [ ] **Step 4: Green + Commit**

```bash
pytest tests/api/test_pipeline_cache_invalidation.py -v
git add src/api/pipeline_api.py tests/api/test_pipeline_cache_invalidation.py
git commit -m "phase3(sme-cache): invalidate retrieval cache on training complete"
```

---

## Task 9 — Intent-aware layer gating

**Files:**
- Create: `src/retrieval/intent_gating.py`
- Create: `tests/retrieval/test_intent_gating.py`

Spec Section 7 mechanism #5. Simple intents (`greeting`, `identity`, `lookup`, `count`, `extract`) skip Layer B and Layer C entirely — those layers add latency and pack tokens but provide no lift for single-fact lookups. Borderline intents (`compare`, `summarize`) run all three layers because SME artifacts can inform multi-doc synthesis. Analytical intents (`analyze`, `diagnose`, `recommend`, `investigate`) always run B + C.

- [ ] **Step 1: Failing tests**

```python
# tests/retrieval/test_intent_gating.py
import pytest
from src.retrieval.intent_gating import IntentGate, GateDecision


@pytest.mark.parametrize("intent,expected", [
    ("greeting",    GateDecision(run_a=False, run_b=False, run_c=False)),
    ("identity",    GateDecision(run_a=False, run_b=False, run_c=False)),
    ("lookup",      GateDecision(run_a=True,  run_b=False, run_c=False)),
    ("count",       GateDecision(run_a=True,  run_b=False, run_c=False)),
    ("extract",     GateDecision(run_a=True,  run_b=False, run_c=True)),
    ("list",        GateDecision(run_a=True,  run_b=False, run_c=True)),
    ("compare",     GateDecision(run_a=True,  run_b=True,  run_c=True)),
    ("summarize",   GateDecision(run_a=True,  run_b=True,  run_c=True)),
    ("analyze",     GateDecision(run_a=True,  run_b=True,  run_c=True)),
    ("diagnose",    GateDecision(run_a=True,  run_b=True,  run_c=True)),
    ("recommend",   GateDecision(run_a=True,  run_b=True,  run_c=True)),
    ("investigate", GateDecision(run_a=True,  run_b=True,  run_c=True)),
    ("aggregate",   GateDecision(run_a=True,  run_b=False, run_c=True)),
])
def test_gate_table(intent, expected):
    assert IntentGate().decide(intent) == expected


def test_unknown_intent_is_conservative():
    # Unknown intents should default to "run everything" so the pack isn't
    # starved — easier to over-retrieve and let the pack assembler trim
    # than to miss evidence.
    d = IntentGate().decide("exotic_new_intent")
    assert d.run_a and d.run_b and d.run_c


def test_override_closes_b_c_for_user_compact_request():
    gate = IntentGate()
    d = gate.decide("analyze", user_requested_compact=True)
    assert d.run_a and not d.run_b and not d.run_c
```

- [ ] **Step 2: Implement**

```python
"""Intent-aware per-layer gating.

Table mirrors spec Section 7 (adaptive top-K per layer):
- simple (greeting/identity/lookup/count): A only (B+C off)
- extract/list/aggregate: A + C (B off — KG adds little for single-row lookups)
- borderline (compare/summarize/overview): all three
- analytical (analyze/diagnose/recommend/investigate): all three
Unknown intents: conservative — run everything.
User-requested compact mode shuts B+C off regardless of intent.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class GateDecision:
    run_a: bool
    run_b: bool
    run_c: bool


_GATE_TABLE = {
    "greeting":    GateDecision(False, False, False),
    "identity":    GateDecision(False, False, False),
    "lookup":      GateDecision(True,  False, False),
    "count":       GateDecision(True,  False, False),
    "extract":     GateDecision(True,  False, True),
    "list":        GateDecision(True,  False, True),
    "aggregate":   GateDecision(True,  False, True),
    "compare":     GateDecision(True,  True,  True),
    "summarize":   GateDecision(True,  True,  True),
    "overview":    GateDecision(True,  True,  True),
    "analyze":     GateDecision(True,  True,  True),
    "diagnose":    GateDecision(True,  True,  True),
    "recommend":   GateDecision(True,  True,  True),
    "investigate": GateDecision(True,  True,  True),
}


class IntentGate:
    def decide(self, intent: str, user_requested_compact: bool = False) -> GateDecision:
        base = _GATE_TABLE.get(intent, GateDecision(True, True, True))
        if user_requested_compact:
            return GateDecision(run_a=base.run_a, run_b=False, run_c=False)
        return base
```

- [ ] **Step 3: Green + Commit**

```bash
pytest tests/retrieval/test_intent_gating.py -v
git add src/retrieval/intent_gating.py tests/retrieval/test_intent_gating.py
git commit -m "phase3(sme-gating): intent-aware per-layer gating"
```

---

## Task 10 — Flag wiring: `enable_hybrid_retrieval` + `enable_cross_encoder_rerank` + flag-set versioning

**Files:**
- Modify: `src/config/feature_flags.py`
- Modify: `src/retrieval/unified_retriever.py` (no change — retriever reads resolver directly; confirm)
- Modify: `src/agent/core_agent.py`
- Create: `tests/retrieval/test_hybrid_flag_off.py`
- Create: `tests/config/test_flag_set_version.py`

Phase 1 ships all 8 flag constants (ERRATA §4). This task confirms the two infrastructure-independent flags (`ENABLE_HYBRID_RETRIEVAL`, `ENABLE_CROSS_ENCODER_RERANK`) are present and adds a **monotonic flag-set version counter** that the retrieval cache key consumes. The counter bumps on any `set_subscription_override` call; retrieval-cache entries keyed off the old version become unreachable and eventually TTL-expire.

Per ERRATA §4, `SMEFeatureFlags` does not expose a `version_for(sub)` method. Phase 3 adds a simpler primitive: a module-level monotonic counter `_flag_set_version_counter` incremented from the override setter. `flag_set_version()` returns its current string form. This is *subscription-agnostic* — any override flip invalidates the cache globally. That is acceptable because override flips are rare and the alternative (per-sub counter with bookkeeping) adds cost for no win.

- [ ] **Step 1: Confirm Phase 1 constants exist; add flag-set version counter**

Verify in `src/config/feature_flags.py`:

```python
# These are owned by Phase 1 per ERRATA §4 — Phase 3 only confirms presence.
# SME_REDESIGN_ENABLED, ENABLE_SME_SYNTHESIS, ENABLE_SME_RETRIEVAL,
# ENABLE_KG_SYNTHESIZED_EDGES, ENABLE_RICH_MODE, ENABLE_URL_AS_PROMPT,
# ENABLE_HYBRID_RETRIEVAL, ENABLE_CROSS_ENCODER_RERANK
```

Add (if not already present — Phase 1's plan owns these):

```python
# --- Monotonic flag-set version counter (Phase 3) -------------------------
_flag_set_version_counter: int = 0


def flag_set_version() -> str:
    """Return current flag-set version as a short string.

    Consumers (retrieval cache key) treat this as opaque. It bumps on any
    set_subscription_override() call so cache entries keyed off the old
    version are naturally bypassed until TTL expiry.
    """
    return f"v{_flag_set_version_counter}"


def bump_flag_set_version() -> None:
    """Increment the counter. Called from set_subscription_override and from
    admin flip endpoints; idempotent to call directly from tests."""
    global _flag_set_version_counter
    _flag_set_version_counter += 1
```

Phase 1's `set_subscription_override(subscription_id, flag, value, actor)` MUST call `bump_flag_set_version()` internally so callers do not need to remember to. If Phase 1 missed this, add a one-line patch here and reference ERRATA §4.

- [ ] **Step 2: Test that hybrid OFF falls back to dense-only**

```python
# tests/retrieval/test_hybrid_flag_off.py
from unittest.mock import MagicMock

from src.retrieval.unified_retriever import UnifiedRetriever
from src.config.feature_flags import ENABLE_HYBRID_RETRIEVAL


def test_hybrid_flag_off_skips_sparse_path(monkeypatch):
    from src.config import feature_flags as ff
    class _HybridOff:
        def is_enabled(self, sub, flag):
            return flag != ENABLE_HYBRID_RETRIEVAL
    monkeypatch.setattr(ff, "get_flag_resolver", lambda: _HybridOff())

    qd = MagicMock(); qd.search.return_value = []
    hy = MagicMock()
    ur = UnifiedRetriever(kg_client=MagicMock(), qdrant=qd, sme=MagicMock(), hybrid=hy)
    ur.retrieve(query="q", subscription_id="s", profile_id="p",
                query_understanding={"intent": "lookup"},
                flags={})
    hy.search.assert_not_called()
    qd.search.assert_called()  # fell back to dense
```

- [ ] **Step 2.5: Test the flag-set version counter**

```python
# tests/config/test_flag_set_version.py
def test_flag_set_version_bumps_on_override():
    from src.config.feature_flags import (
        flag_set_version, bump_flag_set_version,
    )
    before = flag_set_version()
    bump_flag_set_version()
    after = flag_set_version()
    assert before != after
```

- [ ] **Step 3: Wire `core_agent.py` to resolve flags and cache-key by version**

In `CoreAgent.handle()`, right before calling `retrieve()`, compute:

```python
from src.config.feature_flags import (
    get_flag_resolver, flag_set_version,
    ENABLE_SME_RETRIEVAL, ENABLE_KG_SYNTHESIZED_EDGES,
    ENABLE_HYBRID_RETRIEVAL, ENABLE_CROSS_ENCODER_RERANK,
)

resolver = get_flag_resolver()
flags = {
    "enable_sme_retrieval":        resolver.is_enabled(subscription_id, ENABLE_SME_RETRIEVAL),
    "enable_kg_synthesized_edges": resolver.is_enabled(subscription_id, ENABLE_KG_SYNTHESIZED_EDGES),
    "enable_hybrid_retrieval":     resolver.is_enabled(subscription_id, ENABLE_HYBRID_RETRIEVAL),
    "enable_cross_encoder_rerank": resolver.is_enabled(subscription_id, ENABLE_CROSS_ENCODER_RERANK),
}
version = flag_set_version()  # bumps on any override flip
```

Note: `flags` is still populated for downstream convenience (pack assembler, reranker read specific keys). The retriever itself calls `resolver.is_enabled` directly; no need to thread a dict through every layer.

- [ ] **Step 4: Commit**

```bash
pytest tests/retrieval/test_hybrid_flag_off.py tests/config/test_flag_set_version.py -v
git add src/config/feature_flags.py src/agent/core_agent.py \
        tests/retrieval/test_hybrid_flag_off.py \
        tests/config/test_flag_set_version.py
git commit -m "phase3(sme-flags): hybrid+CE flag gates + flag-set version counter"
```

---

## Task 11 — Integration: `core_agent.py` four-layer wiring

**Files:**
- Modify: `src/agent/core_agent.py`
- Create: `tests/agent/test_core_agent_sme_retrieval.py`
- Create: `tests/agent/test_core_agent_sme_retrieval_off.py`
- Create: `tests/agent/test_core_agent_qa_short_circuit.py`
- Create: `tests/agent/test_core_agent_isolation.py`

This is the wiring point. All the machinery from Tasks 3–10 comes together inside `CoreAgent.handle()`. The agent's request flow becomes:

1. **QA fast-path** lookup — if hit, return pre-grounded answer, skip everything else.
2. **UNDERSTAND** (existing intent classifier LLM call — unchanged).
3. Compute **flag set** + **flag_set_version** (Task 10).
4. **Retrieval cache** lookup by `(sub, prof, query_fingerprint, flag_set_version)`. If hit → skip to step 7.
5. **UnifiedRetriever.retrieve()** — four-layer parallel.
6. **Merge → rerank → MMR → PackAssembler.assemble()**, then write the pack to the retrieval cache.
7. **Reason** — single LLM call, unchanged prompt surface, richer pack.
8. **Compose + cite** — unchanged path. Task 11 verifies no change in response shape.

- [ ] **Step 1: Write the four integration tests**

Skeleton for `tests/agent/test_core_agent_sme_retrieval.py`:

```python
"""Flag ON, profile has artifacts → SME layer content lands in the pack."""
from unittest.mock import MagicMock, patch
import pytest

from src.config.feature_flags import ENABLE_SME_RETRIEVAL


@pytest.fixture
def agent_with_fakes(monkeypatch):
    from src.agent.core_agent import CoreAgent
    a = CoreAgent.__new__(CoreAgent)  # bypass heavy __init__ in test
    a._llm = MagicMock()
    a._llm.generate.return_value = MagicMock(text="Q3 revenue grew 12%.", sources=[])
    a._qdrant = MagicMock(); a._qdrant.search.return_value = []
    a._kg = MagicMock()
    # Phase 2 KGRetrievalClient surface — retrieve_layer_b drives these two.
    a._kg.one_hop.return_value = []
    a._kg.inferred_relations.return_value = []
    a._sme = MagicMock()
    a._sme.retrieve.return_value = [
        MagicMock(artifact_type="dossier", narrative="Q3 up 12%",
                  confidence=0.92, evidence=["d1#c1"], score=0.85)
    ]
    # Patch get_flag_resolver everywhere it might be imported so tests are
    # independent of the real MongoDB-backed store.
    class _AllOn:
        def is_enabled(self, sub, flag): return True
    from src.config import feature_flags as ff
    monkeypatch.setattr(ff, "get_flag_resolver", lambda: _AllOn())
    return a


def test_analyze_query_pulls_layer_c(agent_with_fakes):
    a = agent_with_fakes
    resp = a.handle(query="Analyze Q3 revenue trend",
                    subscription_id="s1", profile_id="p_fin")
    a._sme.retrieve.assert_called()
    # The Reasoner prompt must have received a pack containing Layer C text.
    reason_args = a._llm.generate.call_args
    prompt_pack = reason_args.kwargs.get("pack") or reason_args.args[-1]
    assert any("[SME/dossier]" in p.text for p in prompt_pack)


def test_lookup_query_skips_layer_c(agent_with_fakes):
    a = agent_with_fakes
    a.handle(query="What is the Q3 revenue number?",
             subscription_id="s1", profile_id="p_fin")
    a._sme.retrieve.assert_not_called()
```

Skeleton for `tests/agent/test_core_agent_sme_retrieval_off.py`:

```python
"""Flag OFF → Layer C never called; response shape is identical to baseline."""
from src.config.feature_flags import ENABLE_SME_RETRIEVAL


def test_layer_c_off(agent_with_fakes, monkeypatch):
    a = agent_with_fakes
    from src.config import feature_flags as ff
    class _SMEoff:
        def is_enabled(self, sub, flag):
            return flag != ENABLE_SME_RETRIEVAL
    monkeypatch.setattr(ff, "get_flag_resolver", lambda: _SMEoff())

    a.handle(query="Analyze Q3 revenue trend",
             subscription_id="s1", profile_id="p_fin")
    a._sme.retrieve.assert_not_called()
```

Skeleton for `tests/agent/test_core_agent_qa_short_circuit.py`:

```python
"""QA-cache hit short-circuits the rest of the pipeline."""
def test_qa_hit_skips_retrieval(agent_with_fakes, monkeypatch):
    a = agent_with_fakes
    import src.agent.core_agent as mod
    from unittest.mock import MagicMock
    hit = MagicMock(answer="Q3 was $12.4M.", confidence=0.95,
                    source_section_id="sec_7",
                    source_entities=["Q3"], qa_id="abc")
    monkeypatch.setattr(mod, "_qa_fast_path_lookup", lambda **kw: hit)
    resp = a.handle(query="What is our Q3 revenue?",
                    subscription_id="s1", profile_id="p_fin")
    a._sme.retrieve.assert_not_called()
    a._qdrant.search.assert_not_called()
    assert "$12.4M" in resp.text
```

Skeleton for `tests/agent/test_core_agent_isolation.py`:

```python
"""Cross-subscription retrieval MUST be rejected at every layer."""
def test_subscription_id_forwarded_to_layers(agent_with_fakes):
    a = agent_with_fakes
    a.handle(query="x", subscription_id="sub_A", profile_id="p")
    assert a._sme.retrieve.call_args.kwargs["subscription_id"] == "sub_A"
    # retrieve_layer_b drives one_hop; assert the subscription_id is forwarded.
    assert a._kg.one_hop.call_args.kwargs["subscription_id"] == "sub_A"
    # Qdrant profile_id filter must be present in the search call.
    q_call = a._qdrant.search.call_args
    assert "p" in str(q_call)
```

- [ ] **Step 2: Wire `CoreAgent.handle()`**

The integration is scoped — no new LLM calls, prompt surface unchanged, pack schema unchanged. Sketch:

```python
# src/agent/core_agent.py  (Phase 3 addition points — not full replacement)
from src.retrieval.qa_fast_path import QAFastPath
from src.retrieval.retrieval_cache import RetrievalCache
from src.retrieval.unified_retriever import UnifiedRetriever
from src.retrieval.reranker import merge_layers, rerank_merged_candidates
from src.retrieval.pack_assembler import PackAssembler
from src.retrieval.mmr import mmr_select
from src.retrieval.intent_gating import IntentGate
from src.intelligence.sme.adapter_loader import get_adapter_loader
from src.config.feature_flags import (
    get_flag_resolver, flag_set_version,
    ENABLE_SME_RETRIEVAL, ENABLE_KG_SYNTHESIZED_EDGES,
    ENABLE_HYBRID_RETRIEVAL, ENABLE_CROSS_ENCODER_RERANK,
)


def handle(self, *, query, subscription_id, profile_id, **kw):
    resolver = get_flag_resolver()
    flags = {
        "enable_sme_retrieval":
            resolver.is_enabled(subscription_id, ENABLE_SME_RETRIEVAL),
        "enable_kg_synthesized_edges":
            resolver.is_enabled(subscription_id, ENABLE_KG_SYNTHESIZED_EDGES),
        "enable_hybrid_retrieval":
            resolver.is_enabled(subscription_id, ENABLE_HYBRID_RETRIEVAL),
        "enable_cross_encoder_rerank":
            resolver.is_enabled(subscription_id, ENABLE_CROSS_ENCODER_RERANK),
    }
    version = flag_set_version()  # bumps on any override flip; opaque key fragment

    qa_hit = self._qa_fast_path.lookup(query=query, subscription_id=subscription_id,
                                        profile_id=profile_id)
    if qa_hit is not None:
        return self._compose_qa_hit(qa_hit, query)

    qu = self._understand(query)  # existing LLM call
    cache_key_args = dict(subscription_id=subscription_id, profile_id=profile_id,
                          query=query, flag_set_version=version)
    cached_pack = self._retrieval_cache.get(**cache_key_args)
    if cached_pack is None:
        bundle = self._retriever.retrieve(
            query=query, subscription_id=subscription_id, profile_id=profile_id,
            query_understanding=qu, flags=flags,
        )
        # Layer B is now a flat list of edge dicts (via retrieve_layer_b), not
        # the legacy {"nodes", "edges"} shape — merge_layers consumes dicts.
        merged = merge_layers(a=bundle.layer_a_chunks, b=bundle.layer_b_kg or [],
                              c=bundle.layer_c_sme, d=bundle.layer_d_url)
        reranked = rerank_merged_candidates(
            query=query, candidates=merged, cross_encoder=self._ce,
            top_k=10, intent=qu.get("intent", "lookup"),
            enable_cross_encoder=flags["enable_cross_encoder_rerank"],
        )
        diverse = mmr_select(items=reranked, top_k=10, lam=0.7)
        adapter = get_adapter_loader().load(subscription_id,
                                            qu.get("profile_domain") or "generic")
        pack = self._pack_assembler_for(adapter).assemble(items=diverse, intent=qu["intent"])
        self._retrieval_cache.set(pack=[p.__dict__ for p in pack], **cache_key_args)
    else:
        pack = [self._packed_item_from_dict(d) for d in cached_pack]

    reasoner_result = self._llm.generate(query=query, pack=pack, intent=qu["intent"])
    return self._compose(reasoner_result, pack, intent=qu["intent"])
```

`bundle.layer_b_kg` is typed as `List[Dict]` in the updated Task 3 implementation (it comes from `retrieve_layer_b`, a flat list of direct + inferred edge dicts). If Phase 2 returns the legacy `{"nodes": ..., "edges": ...}` shape instead, update `retrieve_layer_b` to normalize before returning — do not normalize here.

The `_compose_qa_hit`, `_compose`, `_packed_item_from_dict`, `_resolve_flags`, `_flag_set_version`, and `_pack_assembler_for` helpers are thin adapters (~5 lines each). The existing `composer.py` and `citation_verifier.py` are called unchanged.

- [ ] **Step 3: Green all four tests**

`pytest tests/agent/test_core_agent_sme_retrieval.py tests/agent/test_core_agent_sme_retrieval_off.py tests/agent/test_core_agent_qa_short_circuit.py tests/agent/test_core_agent_isolation.py -v` → PASS.

- [ ] **Step 4: Commit**

```bash
git add src/agent/core_agent.py tests/agent/test_core_agent_sme_retrieval.py \
        tests/agent/test_core_agent_sme_retrieval_off.py \
        tests/agent/test_core_agent_qa_short_circuit.py \
        tests/agent/test_core_agent_isolation.py
git commit -m "phase3(sme-agent): wire 4-layer retrieval + qa fast path + cache"
```

---

## Task 12 — Run Phase 0 eval harness on flag-ON deployment

**Files:**
- Create: `tests/sme_metrics_phase3_YYYY-MM-DD.json` (committed)
- Create: `tests/sme_results_phase3_YYYY-MM-DD.jsonl` (committed)
- Create: `tests/sme_phase3_gate_report_YYYY-MM-DD.md` (committed)

This is the proof-point task. The entire purpose of Phase 3 is to demonstrate that pre-reasoned SME artifacts move `faithfulness 0.514 → ≥ 0.80` without destabilizing other metrics. The Phase 0 harness is re-used verbatim; only the deployment state changes.

- [ ] **Step 1: Preconditions**

- Preflight audit (Task 1) PASS.
- All unit + integration tests (Tasks 2–11) green on main.
- Sandbox subscription opted in via Task 2 admin endpoint: `enable_sme_retrieval=true`.
- Sandbox has completed Phase 2 synthesis for every profile in `tests/sme_evalset_v1/fixtures/test_profiles.yaml`.

Verify with:

```bash
python -c "
from src.config.feature_flags import get_flag_resolver, ENABLE_SME_RETRIEVAL
import yaml
fixt = yaml.safe_load(open('tests/sme_evalset_v1/fixtures/test_profiles.yaml'))
sub = fixt['test_subscription']['subscription_id']
print('enable_sme_retrieval:', get_flag_resolver().is_enabled(sub, ENABLE_SME_RETRIEVAL))
"
```

Expected: `True`. If `False`, flip via the admin endpoint before running.

- [ ] **Step 2: Re-run the Phase 0 baseline harness**

```bash
python -m scripts.sme_eval.run_baseline \
    --eval-dir tests/sme_evalset_v1/queries \
    --fixtures tests/sme_evalset_v1/fixtures/test_profiles.yaml \
    --out tests/sme_metrics_phase3_$(date +%Y-%m-%d).json \
    --results-jsonl tests/sme_results_phase3_$(date +%Y-%m-%d).jsonl
```

- [ ] **Step 3: Compare Phase 3 against Phase 0 baseline**

Write `tests/sme_phase3_gate_report_YYYY-MM-DD.md`:

```markdown
# Phase 3 gate report — $(date +%Y-%m-%d)

## Launch-gate metrics (must all pass)

| Metric                    | Phase 0 baseline | Phase 3 result | Pass threshold       | Pass? |
|---------------------------|------------------|----------------|----------------------|-------|
| answer_faithfulness       | 0.514            | <PASTE>        | ≥ 0.80               |       |
| hallucination_rate        | 0.0              | <PASTE>        | ≤ 0.02 (no rise)     |       |
| context_recall            | 0.802            | <PASTE>        | ≥ 0.80               |       |
| sme_artifact_hit_rate     | 0.0              | <PASTE>        | ≥ 0.90 on analytical |       |

## Non-blocking deltas (informational)

| Metric                        | Phase 0 | Phase 3 | Direction |
|-------------------------------|---------|---------|-----------|
| cross_doc_integration_rate    |         |         |           |
| insight_novelty               |         |         |           |
| verified_removal_rate         |         |         |           |
| recommendation_groundedness   |         |         |           |
| sme_persona_consistency       |         |         | (Phase 4) |

## Latency

p95 TTFT per intent, Phase 0 vs Phase 3:
- analyze:
- diagnose:
- recommend:
- investigate:

## Per-domain SME artifact hit rate

- finance:
- legal:
- hr:
- medical:
- it_support:
- generic:

## Verdict

- [ ] PASS — proceed to Phase 4
- [ ] FAIL — loop into Task 13 (tuning) before re-evaluating
```

- [ ] **Step 4: Commit the Phase 3 snapshot**

```bash
git add tests/sme_metrics_phase3_*.json tests/sme_results_phase3_*.jsonl \
        tests/sme_phase3_gate_report_*.md
git commit -m "phase3(sme-eval): flag-on measurement snapshot"
git tag -a sme-phase3-v1 -m "Phase 3 retrieval live; measurement frozen"
```

- [ ] **Step 5: If any gate fails → Task 13**

Do **not** proceed to Phase 4 with a failed gate. The point of Phase 3 is that pre-reasoned artifacts alone close the faithfulness gap. If they don't, prompts aren't the fix.

---

## Task 13 — Tune top-K + pack budgets based on Phase 3 eval data

**Files:**
- Modify: `src/retrieval/unified_retriever.py` (`_TOP_K_TABLE`)
- Modify: `deploy/sme_adapters/defaults/generic.yaml` (and per-domain YAMLs if data warrants)
- Create: `tests/retrieval/test_top_k_tuning_regression.py`

This task only runs if Task 12 gate fails OR if the pass is thin (faithfulness in the 0.80–0.82 band). It uses the per-query results JSONL from Task 12 to identify where more/less retrieval helps.

- [ ] **Step 1: Per-query analysis**

```bash
python -c "
import json, collections
from pathlib import Path
res = [json.loads(l) for l in open(
    sorted(Path('tests').glob('sme_results_phase3_*.jsonl'))[-1])]
by_intent = collections.defaultdict(list)
for r in res:
    by_intent[r['query']['intent']].append(r)
for intent, rs in by_intent.items():
    faith = [r['metrics']['answer_faithfulness'] for r in rs
             if 'answer_faithfulness' in r.get('metrics', {})]
    hits = [r['metrics'].get('sme_artifact_hit_rate', 0) for r in rs]
    print(f'{intent}: n={len(rs)} faith_avg={sum(faith)/max(len(faith),1):.3f} '
          f'sme_hit={sum(hits)/max(len(hits),1):.3f}')
"
```

Expected: see which intent has the weakest faithfulness AND lowest SME hit rate. That intent's top-K for Layer C should go up (max +5). Any intent at the budget ceiling whose faithfulness saturates should have its pack budget raised in the adapter YAML (max +1000 tokens per intent).

- [ ] **Step 2: Edit the table or YAML based on data, not vibes**

Only two knobs allowed in this task:
1. `_TOP_K_TABLE` Layer C value (up/down by up to 5).
2. Adapter `retrieval_caps.max_pack_tokens[intent]` (up/down by up to 1000).

Any other change is out of scope for Phase 3 and must go through Phase 4 or a spec amendment.

- [ ] **Step 3: Regression test that the change doesn't break gating**

```python
# tests/retrieval/test_top_k_tuning_regression.py
from src.retrieval.unified_retriever import _resolve_top_k, _TOP_K_TABLE


def test_no_intent_has_zero_for_all_layers_except_greet_identity():
    for intent, k in _TOP_K_TABLE.items():
        if intent in ("greeting", "identity"):
            continue
        assert k.a > 0, f"{intent} has zero Layer A top-K"


def test_analytical_intents_still_pull_sme():
    for intent in ("analyze", "diagnose", "recommend", "investigate"):
        assert _TOP_K_TABLE[intent].c > 0
```

- [ ] **Step 4: Re-run Task 12, commit the tuned snapshot**

```bash
python -m scripts.sme_eval.run_baseline \
    --eval-dir tests/sme_evalset_v1/queries \
    --fixtures tests/sme_evalset_v1/fixtures/test_profiles.yaml \
    --out tests/sme_metrics_phase3_tuned_$(date +%Y-%m-%d).json \
    --results-jsonl tests/sme_results_phase3_tuned_$(date +%Y-%m-%d).jsonl

git add src/retrieval/unified_retriever.py deploy/sme_adapters/defaults/ \
        tests/retrieval/test_top_k_tuning_regression.py \
        tests/sme_metrics_phase3_tuned_*.json tests/sme_results_phase3_tuned_*.jsonl
git commit -m "phase3(sme-tune): top-K + pack budgets tuned from eval data"
```

- [ ] **Step 5: If second iteration still fails, escalate per spec 13.4**

Two full tuning iterations that still miss the gate → **full rollback trigger** per design spec Section 13.4. Do not silently lower the threshold to "pass." Write the post-mortem (`analytics/sme_rollback_YYYY-MM-DD.md` per Section 13.3 step 6), flip `sme_redesign_enabled=false` on affected subscriptions, preserve all data, and feed learnings into sub-project F.

---

## Task 14 — Phase 3 exit checklist

**File:**
- Create: `docs/superpowers/plans/2026-04-20-docwain-sme-phase3-exit-checklist.md`

This task is a literal checklist ritual: a second engineer walks through it and ticks each box. No Phase 4 work starts until every box is ticked.

### Exit checklist

Measurement (Task 12 or Task 13 final run):
- [ ] `answer_faithfulness ≥ 0.80` on the 600-query eval set
- [ ] `hallucination_rate ≤ 0.02` (ideally still `0.0`)
- [ ] `context_recall ≥ 0.80` (no regression from Phase 0's 0.802)
- [ ] `sme_artifact_hit_rate ≥ 0.90` on `analyze/diagnose/recommend/investigate` queries
- [ ] Per-intent p95 TTFT on `analyze` within 20% of Phase 0 baseline (non-blocking; investigate only)
- [ ] Gate report markdown signed off by a second engineer

Code (git log sanity):
- [ ] 14 commits scoped `phase3(sme-X): ...`
- [ ] `src/generation/prompts.py` **UNCHANGED** across the phase (`git log --oneline src/generation/prompts.py` shows no commit newer than the Phase 4 placeholder point)
- [ ] No new `pipeline_status` strings introduced
- [ ] No Claude / Anthropic references in any commit, code, or doc
- [ ] No internal wall-clock timeouts (`grep -rn "timeout=" src/retrieval/unified_retriever.py` shows none on `as_completed`)

Tests:
- [ ] `pytest tests/retrieval/ -v` all green
- [ ] `pytest tests/agent/test_core_agent_sme_*.py -v` all green
- [ ] `pytest tests/agent/test_core_agent_qa_short_circuit.py -v` green
- [ ] `pytest tests/agent/test_core_agent_isolation.py -v` green
- [ ] `pytest tests/api/test_admin_sme_flag_flip.py -v` green

Rollback demonstrated:
- [ ] Flipping `enable_sme_retrieval=false` on an opted-in subscription returns retrieval to the Phase 0 baseline shape within 1 minute (flag cache TTL expires; next query hits dense-only + KG)
- [ ] `enable_hybrid_retrieval=false` falls back to dense-only without error
- [ ] `enable_cross_encoder_rerank=false` bypasses the reranker without error

Isolation:
- [ ] Cross-subscription probe: query against `sub_A` with `profile_id` from `sub_B` returns empty + logs a rejection (integration test passes)
- [ ] `generic` adapter profile returns a valid (if shallow) pack — no adapter crash

Operational:
- [ ] Retrieval cache invalidated on `PIPELINE_TRAINING_COMPLETED` (Task 8 hook wired)
- [ ] Redis retrieval cache key collisions: stress test with 10k distinct queries at the same fingerprint prefix → no wrong-profile reads
- [ ] Per-layer latency telemetry captured in `bundle.per_layer_ms` and surfaced in query traces (Phase 2 trace writer consumes this)

Documentation:
- [ ] `deploy/sme_adapters/defaults/generic.yaml` documents the `retrieval_caps.max_pack_tokens` field
- [ ] This plan (`docs/superpowers/plans/2026-04-20-docwain-sme-phase3-retrieval.md`) committed and referenced in the Phase 4 plan's "prior-phase contracts" section

Once every box is ticked:

```bash
git add docs/superpowers/plans/2026-04-20-docwain-sme-phase3-exit-checklist.md
git commit -m "phase3(sme-exit): exit checklist complete — phase 4 unblocked"
```

---

## Self-review

### Spec coverage

Every item in Sections 7, 8 (grounding-filter + provenance only), 10 (Phase 3 gate), and 12 (Phase 3 rollout entry) maps to a task above:

- Stage 1 parallel retrieval, four layers → Task 3
- Stage 2 merge + cross-encoder rerank + MMR → Task 4
- Stage 3 budget-aware pack assembly → Task 5
- Six efficiency mechanisms: hybrid (Task 10), cross-encoder always on (Tasks 4 + 10), KG pre-materialized (inherited from Phase 2 + Task 3 Layer B flag), QA cache fast path (Task 7), intent-aware layer gating (Task 9), Redis retrieval cache (Task 8)
- Adaptive top-K per layer → Task 6 + tuning loop in Task 13
- URL-as-prompt → Layer D placeholder only in this phase (stub in Task 3); Phase 5 wires fetch
- Grounding filters at retrieval (Section 8 choke point 2): profile_id hard filter covered in Tasks 3 + 11 isolation tests
- Grounding choke points 3 + 4: untouched in Phase 3 — those are Phase 4
- Measurement gate → Task 12 (primary) + Task 13 (tuning) + Task 14 (exit)

14 tasks — inside the 12–16 budget. Code blocks kept ≤ 40 Python lines per implementation step.

### Memory rules applied

- No Claude / Anthropic / Co-Authored-By anywhere in the plan.
- No internal timeouts — Task 3 explicitly removes the existing `timeout=30` from `as_completed`, and Task 3 Step 1 includes a test that asserts this.
- MongoDB = control plane — Mongo is never written to by retrieval code; flag resolver reads Mongo, nothing else.
- `src/generation/prompts.py` is untouched across all 14 tasks. Exit checklist (Task 14) asserts this by git-log.
- No new `pipeline_status` strings — Phase 3 is query-path only.
- Profile isolation — every retrieval call, cache key, and QA-fast-path key includes `(subscription_id, profile_id)`; Task 11 has a dedicated isolation test.
- Engineering-first — zero retraining, zero new gateway LLM calls on the hot path, cross-encoder is a local model already loaded at startup.

### Placeholder scan

Every `REPLACE_WITH_...` placeholder is an operator-supplied sandbox ID in shell commands; no placeholder is in code files.

### Type consistency

`RetrievalBundle` and `PackedItem` are defined once each; `GateDecision` and `LayerTopK` are frozen dataclasses. `QAFastPathHit` fields match the existing composer signature exactly. Each new file is created exactly once; modified files are flagged "Modify" in the task header.

### What this plan deliberately does NOT do

Touch `prompts.py` (Phase 4), introduce new intents (Phase 4), wire URL fetch (Phase 5), change response shape, retrain the model, or add a new hot-path LLM call.

### Evidence pack shape handed to Phase 4

Phase 4 consumes a `List[PackedItem]` where each item has:

```python
PackedItem(
    text: str,                         # Layer A chunk text OR compressed SME (“[SME/dossier] ... [d1#c1]”)
    provenance: List[Tuple[doc_id, chunk_id]],
    layer: str,                        # layer_a | layer_b | layer_c | layer_d
    confidence: float,
    rerank_score: float,
    sme_backed: bool,                  # True for Layer C items, for Layer B items with kind=="kg_inferred", or for Layer A items that overlap such SME sources (ERRATA §11)
    metadata: dict,                    # {artifact_type: "dossier|insight|comparative|recommendation",
                                       #  relation_type: "...indirectly_funds..."}
)
```

This is what Phase 4's rich templates and persona injection will format. Phase 4 does not need to re-retrieve; it only transforms this pack into rich Markdown sections. The `sme_backed` flag is the signal for Phase 4's shape-resolution logic to pick rich over compact when SME artifacts meaningfully contributed.

### Retrieval-call signatures handed to Phase 4

```python
UnifiedRetriever.retrieve(
    *, query: str, subscription_id: str, profile_id: str,
    query_understanding: dict, flags: dict, top_k_overrides: dict | None
) -> RetrievalBundle
```

```python
merge_layers(*, a, b, c, d) -> List[dict]
rerank_merged_candidates(*, query, candidates, cross_encoder, top_k, intent, enable_cross_encoder) -> List[dict]
mmr_select(*, items, top_k, lam) -> List[dict]
PackAssembler(adapter).assemble(*, items, intent) -> List[PackedItem]
```

Phase 4 adds response shape only; every retrieval surface above stays fixed.

---

## Appendix — ERRATA alignment (2026-04-21)

Applied ERRATA §§1, 4, 7, 8, 11, 14 on 2026-04-21. Specifically:

- **§1 (AdapterLoader)** — confirmed `AdapterLoader.load(sub_id, profile_domain)` usage throughout; `get_adapter_loader()` factory import standardized to `src.intelligence.sme.adapter_loader`.
- **§4 (Feature flags)** — replaced all `FeatureFlagResolver` references with `SMEFeatureFlags`; replaced `.resolve()` with `.is_enabled()`; replaced `src.intelligence.sme.feature_flags` imports with `src.config.feature_flags`; all string-literal flag names replaced with imported constants (`SME_REDESIGN_ENABLED`, `ENABLE_SME_RETRIEVAL`, `ENABLE_KG_SYNTHESIZED_EDGES`, `ENABLE_RICH_MODE`, `ENABLE_URL_AS_PROMPT`, `ENABLE_HYBRID_RETRIEVAL`, `ENABLE_CROSS_ENCODER_RERANK`). Methods `resolve_explicit`, `resolve_all`, `set`, `version_for`, `clear_cache`, `resolve_with_defaults` removed or replaced; flag-set versioning implemented as a module-level monotonic counter (`flag_set_version()` / `bump_flag_set_version()`) rather than a per-subscription resolver method. The broken flag-gating expression `flags.get("enable_sme_retrieval", False) is not None` was fixed to `get_flag_resolver().is_enabled(subscription_id, ENABLE_SME_RETRIEVAL)`.
- **§7 (KG Layer B routing)** — replaced `self._kg.retrieve_context(...)` with `self.retrieve_layer_b(...)` per Phase 2's canonical helper; `include_inferred=True, inferred_confidence_floor=0.6` match Phase 2's `inferred_edge_confidence_floor()` default.
- **§8 (SMERetrieval class name)** — all `SMERetrievalLayer` references renamed to `SMERetrieval`.
- **§11 (PackedItem sme_backed + degraded_layers)** — `merge_layers` now sets `sme_backed=True` for Layer B items with `kind == "kg_inferred"` in addition to all Layer C items; `bundle.degraded_layers` double-append bug (appending both `'c'` and `'layer_c'`) fixed to append only the full-name form.
- **§14 (cache invalidation hook)** — hand-waved cache invalidation step promoted to a concrete `Task 8.5` that (a) locates the `PIPELINE_TRAINING_COMPLETED` branch in `src/api/pipeline_api.py`, (b) adds a call to `retrieval_cache.invalidate_profile(sub, prof)`, and (c) ships a failing-test-first integration test verifying the cache is cleared when the status transitions.

Cross-references: ERRATA §13 (Phase 2 emits `qa_idx:{sub}:{prof}:{fingerprint}` keys) acknowledged in Task 7; inline backfill provided for legacy profiles.
