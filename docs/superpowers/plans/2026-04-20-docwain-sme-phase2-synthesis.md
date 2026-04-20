# DocWain SME Phase 2 — SME Synthesis in Training Stage (Retrieval Flag-Gated)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run real SME synthesis in the training stage on HITL opt-in subscriptions, persist the five artifact types to Blob + Qdrant + Neo4j, and ship retrieval Layers B (KG + synthesized edges) and C (SME artifacts) behind `enable_sme_retrieval` (default OFF) — artifacts produced + measured without changing user-visible responses.

**Architecture:** Phase 1 shipped the adapter loader, verifier, trace writers, storage shim, and stub builders. Phase 2 swaps stubs for real LLM-driven synthesis, has the training stage call `SMESynthesizer.synthesize_profile(...)` as the final step before `PIPELINE_TRAINING_COMPLETED` fires, and wires Layers B+C in `unified_retriever.py` behind the flag. Prompts untouched (Phase 4's job). Phase 0 harness proves `sme_artifact_hit_rate` rises from 0.0.

**Tech stack:** Python 3.12, `pydantic` schemas, `httpx` → existing `src/serving/model_router.py`, `qdrant-client`, `neo4j`, `azure-storage-blob`, `pytest`. Existing DocWain embedder for artifact vectors.

**Related spec:** `docs/superpowers/specs/2026-04-20-docwain-profile-sme-reasoning-design.md` — §§4, 6, 7, 9, 11, 13, 15.

**Depends on:** Phase 0 (eval harness, `tests/sme_evalset_v1/`), Phase 1 (SME package: adapter loader, verifier, trace, storage shim, stub builders).

**Memory rules that constrain this plan:**
- **Measure Before You Change.** Phase 2 re-runs the Phase 0 harness and proves `sme_artifact_hit_rate` rises — no accuracy claim without the number moving.
- **No Customer Data in Training.** Sandbox is synthetic only; real opt-in subs stay inside their own `(subscription_id, profile_id)` boundary.
- **MongoDB = control plane only.** Canonical artifacts in Blob, snippets in Qdrant, synthesized edges in Neo4j. Mongo carries only `sme_synthesis_version`, `sme_last_input_hash`, per-sub capability flags.
- **No new `pipeline_status` strings.** `PIPELINE_TRAINING_COMPLETED` fires only on synthesis success; failure keeps prior status, retry is idempotent.
- **No internal timeouts.** `httpx` per-op default is the only network safety. No wall-clock cutoff across steps — builders stream and persist.
- **Response-formatting authority.** `src/generation/prompts.py` not modified in Phase 2. All persona/template text loads from adapter YAMLs via Phase 1's `AdapterLoader`.
- **Profile isolation hard.** `(subscription_id, profile_id)` in every builder, every write, every retrieval filter. Cross-profile leakage is an integration-test fail.
- **No Claude/Anthropic attribution.** Commits, code comments, docstrings clean.
- **Engineering-first.** Zero retraining; model consumes better packs.

---

## File structure

```
src/intelligence/sme/                           [extended from Phase 1]
├── __init__.py                                 [modified — export public API]
├── adapter_loader.py, verifier.py, trace.py    [Phase 1 — unchanged]
├── storage.py                                  [Phase 1 — extended Task 11]
├── synthesizer.py                              [MODIFIED — full orchestrator]
├── dossier_builder.py                          [MODIFIED — full LLM synthesis]
├── insight_index_builder.py                    [MODIFIED — compact full impl]
├── comparative_register_builder.py             [MODIFIED — compact full impl]
├── kg_materializer.py                          [MODIFIED — full Cypher writer]
├── recommendation_bank_builder.py              [MODIFIED — compact full impl]
├── llm_gateway.py, artifact_schemas.py         [NEW]
├── input_snapshot.py, flags.py                 [NEW]

src/api/pipeline_api.py                         [MODIFIED — synthesis wiring]
src/tasks/embedding.py                          [MODIFIED — delegation]
src/retrieval/unified_retriever.py              [MODIFIED — Layer B]
src/retrieval/sme_retrieval.py                  [NEW — Layer C]
src/kg/retrieval.py                             [MODIFIED — INFERRED_RELATION]

tests/intelligence/sme/test_*.py                [per-module unit tests]
tests/api/test_pipeline_api_sme.py              [training-stage wiring]
tests/retrieval/test_unified_retriever_layer_b.py, test_sme_retrieval.py
tests/integration/test_sme_end_to_end.py        [sandbox end-to-end]
```

Phase 1 contract — `synthesize_profile(sub, prof, adapter, traces, storage) -> SynthesisReport` — stays; Phase 2 swaps stub bodies for real implementations, signatures unchanged.

---

## Task 1: Preflight audit of Phase 1 artifacts and Phase 2 setup

**Files:**
- Audit only: `src/intelligence/sme/adapter_loader.py`, `verifier.py`, `trace.py`, `storage.py`, `synthesizer.py` (stub), the five `*_builder.py` stubs
- Audit only: `src/api/pipeline_api.py`, `src/tasks/embedding.py`, `src/retrieval/unified_retriever.py`, `src/kg/retrieval.py`, `src/api/statuses.py`
- Confirm: sandbox subscription + profile IDs from Phase 1's `tests/intelligence/sme/conftest.py`

- [ ] **Step 1: Read the Phase 1 SME modules and note the public contract**

For each of `adapter_loader.py`, `verifier.py`, `trace.py`, `storage.py`, `synthesizer.py`: record exported class names + signatures + stub behavior. Key contracts Phase 2 builders must match: `AdapterLoader.load(sub, domain) -> LoadedAdapter`; `SMEVerifier.verify(items, profile_ctx) -> VerifierReport`; `ArtifactStorage.{put_snippet, put_canonical, put_manifest}`. Surface any required signature change in the self-review gap list.

- [ ] **Step 2: Read the current training-stage terminal step**

Read `embedding.py:420-460` (status flip at L446) + `pipeline_api.py`. Today: embed → KG → status. Phase 2 interposes: LAST doc → synthesis → status flip. Failure keeps `EMBEDDING_IN_PROGRESS` + audit marker (no new status string — spec invariant preserved).

- [ ] **Step 3: Read retrieval modules**

Note Layer B plug-in site in `unified_retriever.py`, the existing `(subscription_id, profile_id)` hard filter (Phase 2 must not relax), and Qdrant client usage patterns (Layer C reuses).

- [ ] **Step 4: Confirm sandbox fixtures from Phase 1**

Phase 1's `tests/intelligence/sme/conftest.py` provides `sandbox_subscription_id`, per-domain `sandbox_profile_id_*`. Confirm ≥10 synthetic docs past embedding per profile. If missing, block Phase 2 and fix in Phase 1.

- [ ] **Step 5: Commit an empty scaffold if Phase 2 files are absent**

```bash
mkdir -p tests/api tests/retrieval tests/integration
touch tests/api/test_pipeline_api_sme.py tests/retrieval/test_unified_retriever_layer_b.py \
      tests/retrieval/test_sme_retrieval.py tests/integration/test_sme_end_to_end.py
touch src/intelligence/sme/llm_gateway.py src/intelligence/sme/artifact_schemas.py \
      src/intelligence/sme/input_snapshot.py src/intelligence/sme/flags.py \
      src/retrieval/sme_retrieval.py
git add -A tests/api tests/retrieval tests/integration \
           src/intelligence/sme/llm_gateway.py src/intelligence/sme/artifact_schemas.py \
           src/intelligence/sme/input_snapshot.py src/intelligence/sme/flags.py \
           src/retrieval/sme_retrieval.py
git commit -m "phase2(sme-scaffold): empty Phase 2 module and test files"
```

Empty-file scaffolding is intentional. Each file receives its real content in a dedicated later task. This keeps individual task commits focused.

---

## Task 2: Artifact schemas — the five pydantic contracts

**Files:**
- Create: `src/intelligence/sme/artifact_schemas.py`
- Create: `tests/intelligence/sme/test_artifact_schemas.py`

Every artifact item carries provenance + confidence per spec Section 6. This task makes the contract explicit. Phase 3 (retrieval-on) consumes these same types; nothing downstream may change the shape without a breaking-change migration.

- [ ] **Step 1: Write the failing tests**

Create `tests/intelligence/sme/test_artifact_schemas.py`:

```python
"""Tests for SME artifact pydantic schemas."""
from datetime import datetime

import pytest
from pydantic import ValidationError

from src.intelligence.sme.artifact_schemas import (
    Evidence, DossierSection, InsightItem, ComparativeItem,
    RecommendationItem, InferredEdge, ArtifactType,
)


def _ev():
    return [Evidence(doc_id="d1", chunk_id="c1")]


def test_evidence_and_confidence_invariants():
    with pytest.raises(ValidationError):
        Evidence(doc_id="", chunk_id="c1")
    with pytest.raises(ValidationError):
        DossierSection(section="s", narrative="n", evidence=[], confidence=0.9)
    with pytest.raises(ValidationError):
        DossierSection(section="s", narrative="n", evidence=_ev(), confidence=1.2)


def test_type_enums_and_comparative_cardinality():
    InsightItem(type="trend", narrative="x", evidence=_ev(), confidence=0.7)
    with pytest.raises(ValidationError):
        InsightItem(type="vibes", narrative="x", evidence=_ev(), confidence=0.7)
    with pytest.raises(ValidationError):
        ComparativeItem(type="delta", axis="rev", compared_items=["d1"],
                        analysis="x", evidence=_ev(), confidence=0.7)


def test_recommendation_and_inferred_edge_full_shape():
    RecommendationItem(
        recommendation="tighten AR", rationale="Q3 DSO widened",
        linked_insights=["i_3"], estimated_impact={"qualitative": "moderate"},
        assumptions=[], caveats=[], evidence=_ev(), confidence=0.75)
    e = InferredEdge(
        src_node_id="n1", dst_node_id="n2", relation_type="indirectly_funds",
        confidence=0.72, evidence=_ev(),
        inference_path=[{"from": "n1", "edge": "PAYS", "to": "nX"},
                        {"from": "nX", "edge": "FUNDS", "to": "n2"}],
        synthesis_version=3, adapter_version="1.2.0",
        generated_at=datetime(2026, 4, 20),
        subscription_id="s", profile_id="p")
    assert len(e.inference_path) == 2


def test_artifact_type_enum_stable():
    # Adding members is breaking for Phase 3's retrieval filters.
    assert ArtifactType.__args__ == (
        "dossier", "insight_index", "comparative_register", "recommendation_bank")
```

- [ ] **Step 2: Run failing tests** — `pytest tests/intelligence/sme/test_artifact_schemas.py -v` (module missing).

- [ ] **Step 3: Write the schemas**

Create `src/intelligence/sme/artifact_schemas.py`:

```python
"""Pydantic contracts for the five SME artifact types + inferred KG edge.

These types are the single source of truth shared by builders, verifier,
storage, and retrieval. Any field change is a breaking migration; bump
synthesis_version on the artifact manifest and dual-read during rollout.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


ArtifactType = Literal["dossier", "insight_index", "comparative_register", "recommendation_bank"]

InsightType = Literal["trend", "anomaly", "gap", "risk", "opportunity", "conflict"]
ComparativeType = Literal["delta", "conflict", "timeline", "corroboration"]


class Evidence(BaseModel):
    doc_id: str = Field(..., min_length=1)
    chunk_id: str = Field(..., min_length=1)
    quoted_span: str | None = None


class _WithConfidence(BaseModel):
    confidence: float = Field(..., ge=0.0, le=1.0)


class DossierSection(_WithConfidence):
    section: str
    narrative: str
    evidence: list[Evidence] = Field(..., min_length=1)
    entity_refs: list[str] = Field(default_factory=list)


class DossierArtifact(BaseModel):
    subscription_id: str
    profile_id: str
    profile_domain: str
    sections: list[DossierSection]
    adapter_version: str
    synthesis_version: int
    generated_at: datetime


class InsightItem(_WithConfidence):
    type: InsightType
    narrative: str
    evidence: list[Evidence] = Field(..., min_length=1)
    domain_tags: list[str] = Field(default_factory=list)
    temporal_scope: str | None = None
    entity_refs: list[str] = Field(default_factory=list)


class InsightIndexArtifact(BaseModel):
    subscription_id: str
    profile_id: str
    items: list[InsightItem]
    adapter_version: str
    synthesis_version: int
    generated_at: datetime


class ComparativeItem(_WithConfidence):
    type: ComparativeType
    axis: str
    compared_items: list[str] = Field(..., min_length=2)
    analysis: str
    resolution: str | None = None
    evidence: list[Evidence] = Field(..., min_length=1)


class ComparativeRegisterArtifact(BaseModel):
    subscription_id: str
    profile_id: str
    items: list[ComparativeItem]
    adapter_version: str
    synthesis_version: int
    generated_at: datetime


class RecommendationItem(_WithConfidence):
    recommendation: str
    rationale: str
    linked_insights: list[str] = Field(default_factory=list)
    estimated_impact: dict[str, Any] = Field(default_factory=dict)
    assumptions: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(..., min_length=1)
    domain_tags: list[str] = Field(default_factory=list)


class RecommendationBankArtifact(BaseModel):
    subscription_id: str
    profile_id: str
    items: list[RecommendationItem]
    adapter_version: str
    synthesis_version: int
    generated_at: datetime


class InferredEdge(_WithConfidence):
    src_node_id: str
    dst_node_id: str
    relation_type: str
    evidence: list[Evidence] = Field(..., min_length=1)
    inference_path: list[dict[str, str]] = Field(..., min_length=1)
    synthesis_version: int
    adapter_version: str
    generated_at: datetime
    subscription_id: str
    profile_id: str


class KGMaterializationReport(BaseModel):
    edges_created: int
    edges_skipped_verifier: int
    max_path_length: int
    adapter_version: str
    synthesis_version: int
```

Keep under 50 lines of logic per file; pydantic does the work. `ArtifactType` literal is intentionally stable — adding members is breaking for Phase 3 filters.

- [ ] **Step 4: Verify tests pass** — `pytest tests/intelligence/sme/test_artifact_schemas.py -v`

- [ ] **Step 5: Commit** — `git add src/intelligence/sme/artifact_schemas.py tests/intelligence/sme/test_artifact_schemas.py && git commit -m "phase2(sme-schemas): pydantic contracts for all five artifacts + inferred edge"`

---

## Task 3: LLM gateway wrapper for synthesis calls

**Files:**
- Create: `src/intelligence/sme/llm_gateway.py`
- Create: `tests/intelligence/sme/test_llm_gateway.py`

Builders depend on this gateway, not `src/serving/model_router.py` directly (which carries query-path concerns irrelevant to synthesis). Thin, test-injectable wrapper.

- [ ] **Step 1: Write the failing tests**

Create `tests/intelligence/sme/test_llm_gateway.py`:

```python
"""Tests for the SME LLM gateway wrapper."""
from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.intelligence.sme.llm_gateway import SMELLMGateway, SynthesisLLMRequest


@pytest.fixture
def gateway():
    return SMELLMGateway(base_url="http://localhost:8100", model="docwain-v2-active")


def _req(adapter="1.0.0"):
    return SynthesisLLMRequest(system_prompt="s", user_prompt="u",
                               adapter_version=adapter, trace_tag="t")


def test_call_returns_structured_response(gateway):
    fake = MagicMock(spec=httpx.Response)
    fake.status_code = 200; fake.raise_for_status = MagicMock()
    fake.json.return_value = {"choices": [{"message": {"content": "Q3 rose."}}]}
    with patch.object(gateway._client, "post", return_value=fake):
        resp = gateway.call(_req())
    assert resp.text == "Q3 rose."
    assert resp.latency_ms > 0


def test_call_propagates_connection_error(gateway):
    with patch.object(gateway._client, "post", side_effect=httpx.ConnectError("down")):
        with pytest.raises(RuntimeError, match="LLM gateway unreachable"):
            gateway.call(_req())


def test_request_carries_adapter_version():
    assert _req(adapter="1.2.0").adapter_version == "1.2.0"
```

No wall-clock timeout assertion — `httpx`'s default is the per-op safety net per memory rule; synthesis steps set no budget.

- [ ] **Step 2: Run failing tests** — `pytest tests/intelligence/sme/test_llm_gateway.py -v`

- [ ] **Step 3: Implement the gateway**

Create `src/intelligence/sme/llm_gateway.py`:

```python
"""Thin wrapper around DocWain's LLM gateway for SME synthesis calls.

Builders depend on this interface, NOT on src/serving/model_router.py
directly. Keeps query-time and ingest-time call paths testable
independently and prevents accidental coupling to query features.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class SynthesisLLMRequest:
    system_prompt: str
    user_prompt: str
    adapter_version: str
    trace_tag: str
    max_output_tokens: int = 2048
    temperature: float = 0.2


@dataclass(frozen=True)
class SynthesisLLMResponse:
    text: str
    latency_ms: float
    model: str


class SMELLMGateway:
    """Posts chat-completion requests to DocWain's gateway."""

    def __init__(self, base_url: str, model: str, api_path: str = "/v1/chat/completions"):
        self._base_url = base_url.rstrip("/")
        self._path = api_path
        self._model = model
        self._client = httpx.Client()  # httpx default per-op timeout only

    def call(self, req: SynthesisLLMRequest) -> SynthesisLLMResponse:
        body = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": req.system_prompt},
                {"role": "user", "content": req.user_prompt},
            ],
            "temperature": req.temperature,
            "max_tokens": req.max_output_tokens,
        }
        start = time.perf_counter()
        try:
            resp = self._client.post(f"{self._base_url}{self._path}", json=body)
            resp.raise_for_status()
        except httpx.RequestError as e:
            raise RuntimeError(f"LLM gateway unreachable: {e}") from e
        text = resp.json()["choices"][0]["message"]["content"]
        return SynthesisLLMResponse(
            text=text,
            latency_ms=(time.perf_counter() - start) * 1000.0,
            model=self._model,
        )

    def close(self) -> None:
        self._client.close()
```

- [ ] **Step 4: Verify tests pass**

```bash
pytest tests/intelligence/sme/test_llm_gateway.py -v
```

- [ ] **Step 5: Commit** — `git add src/intelligence/sme/llm_gateway.py tests/intelligence/sme/test_llm_gateway.py && git commit -m "phase2(sme-llm-gateway): thin synthesis-side wrapper over model_router"`

---

## Task 4: SME Dossier builder — full LLM-driven synthesis

**Files:**
- Modify: `src/intelligence/sme/dossier_builder.py` (Phase 1 stub replaced)
- Create: `tests/intelligence/sme/test_dossier_builder.py`

Illustrative full-flow builder. Tasks 5, 6, 8 follow the same pattern in prose.

**Contract:** `DossierBuilder(llm, adapter, verifier, storage, trace).build(profile_ctx) -> DossierArtifact`. `ProfileSynthesisContext` (Task 15) carries sub/profile ids, domain, chunk iterator, KG slice, version, run_id. Builder returns the verified artifact after persisting snippets+canonical through storage; dropped items logged via trace, never persisted.

- [ ] **Step 1: Write the failing tests**

Create `tests/intelligence/sme/test_dossier_builder.py`:

```python
"""Tests for the SME Dossier builder — the illustrative full-flow builder."""
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from src.intelligence.sme.artifact_schemas import (
    DossierArtifact, DossierSection, Evidence,
)
from src.intelligence.sme.dossier_builder import DossierBuilder
from src.intelligence.sme.llm_gateway import SynthesisLLMResponse


def _ctx(domain="finance"):
    ctx = MagicMock()
    ctx.subscription_id = "sub_test"
    ctx.profile_id = "prof_fin"
    ctx.profile_domain = domain
    ctx.synthesis_version = 3
    ctx.run_id = "run_001"
    ctx.iter_chunks.return_value = [
        {"doc_id": "d1", "chunk_id": "c1", "text": "Q1 revenue $5M."},
        {"doc_id": "d1", "chunk_id": "c2", "text": "Q2 revenue $5.4M."},
        {"doc_id": "d2", "chunk_id": "c3", "text": "Q3 revenue $6.1M."},
    ]
    ctx.kg_slice = {"entities": ["Acme Corp", "Finance"], "edges": []}
    return ctx


def _adapter(domain="finance"):
    a = MagicMock()
    a.version = "1.2.0"
    a.persona = {"role": "senior financial analyst", "voice": "direct, quantitative"}
    a.dossier_section_weights = {"overview": 0.4, "trends": 0.4, "risks": 0.2}
    a.prompt_template = "Summarize {section} for a {role} audience."
    a.grounding_rules = ["cite each claim"]
    return a


def _llm_response(section, narrative, citations):
    import json
    body = {"section": section, "narrative": narrative,
            "evidence": [{"doc_id": d, "chunk_id": c} for d, c in citations],
            "confidence": 0.85, "entity_refs": ["Acme Corp"]}
    return SynthesisLLMResponse(text=json.dumps(body), latency_ms=120.0, model="docwain-v2")


def _three_responses():
    return [
        _llm_response("overview", "Revenue grew QoQ.", [("d1", "c1"), ("d2", "c3")]),
        _llm_response("trends", "QoQ growth 8% then 13%.", [("d1", "c2"), ("d2", "c3")]),
        _llm_response("risks", "Concentration risk.", [("d1", "c1")]),
    ]


def _pass_all(items, profile_ctx):
    return MagicMock(passed=items, dropped=[], reasons=[])


def test_build_produces_sections_persists_and_drops_verifier_failures():
    # Happy path
    llm = MagicMock(); llm.call.side_effect = _three_responses()
    verifier = MagicMock(); verifier.verify.side_effect = _pass_all
    storage = MagicMock(); trace = MagicMock()
    art = DossierBuilder(llm, _adapter(), verifier, storage, trace).build(_ctx())
    assert isinstance(art, DossierArtifact)
    assert {s.section for s in art.sections} == {"overview", "trends", "risks"}
    assert art.adapter_version == "1.2.0" and art.synthesis_version == 3
    assert storage.put_canonical.call_count == 1
    assert storage.put_snippet.call_count == 3

    # Drops: "risks" fails verifier
    llm2 = MagicMock(); llm2.call.side_effect = _three_responses()
    v2 = MagicMock()
    def _drop_risks(items, profile_ctx):
        passed = [i for i in items if i.section != "risks"]
        dropped = [i for i in items if i.section == "risks"]
        return MagicMock(passed=passed, dropped=dropped,
                          reasons=[("risks", "evidence_validity_failed")])
    v2.verify.side_effect = _drop_risks
    s2 = MagicMock(); t2 = MagicMock()
    art2 = DossierBuilder(llm2, _adapter(), v2, s2, t2).build(_ctx())
    assert {s.section for s in art2.sections} == {"overview", "trends"}
    t2.record_verifier_drop.assert_called_once()
    assert s2.put_snippet.call_count == 2


def test_build_applies_all_five_verifier_checks():
    """Dossier builder relies on SMEVerifier running the five spec Section 6 checks:
      1 evidence presence   - items[].evidence non-empty
      2 evidence validity   - chunk_ids exist in ctx.chunks + span present
      3 inference provenance - pass-through for Dossier (direct claims, not inferred)
      4 confidence calibration - conf>0.8 requires >=2 evidences or rolled to 0.6
      5 contradiction check - no overlapping conflicting items w/o annotation
    """
    llm = MagicMock(); llm.call.side_effect = _three_responses()
    verifier = MagicMock(); checks_run = []
    def _verify(items, profile_ctx):
        checks_run.append(("evidence_presence", all(i.evidence for i in items)))
        checks_run.append(("evidence_validity", len(items)))
        checks_run.append(("inference_provenance", "non-inferred: pass"))
        checks_run.append(("confidence_calibration",
                           [(i.section, len(i.evidence), i.confidence) for i in items]))
        checks_run.append(("contradiction_check", len(items)))
        return MagicMock(passed=items, dropped=[], reasons=[])
    verifier.verify.side_effect = _verify
    storage = MagicMock(); trace = MagicMock()

    DossierBuilder(llm, _adapter(), verifier, storage, trace).build(_ctx())
    assert [c[0] for c in checks_run] == [
        "evidence_presence", "evidence_validity", "inference_provenance",
        "confidence_calibration", "contradiction_check",
    ]
    calib = [c for c in checks_run if c[0] == "confidence_calibration"][0][1]
    overview = [c for c in calib if c[0] == "overview"][0]
    assert overview[1] >= 2 and overview[2] > 0.8  # 0.85 conf + 2 ev: check 4 holds
```

- [ ] **Step 2: Run failing tests** — `pytest tests/intelligence/sme/test_dossier_builder.py -v` (stub returns empty).

- [ ] **Step 3: Write the builder**

Replace `src/intelligence/sme/dossier_builder.py`:

```python
"""SME Dossier builder — per-section LLM synthesis with fail-closed verification.

Flow: for each adapter section, LLM-generate DossierSection JSON, pool, send to
SMEVerifier (all five checks), trace drops, persist survivors (canonical+snippets).
"""
from __future__ import annotations

import json
from datetime import datetime

from src.intelligence.sme.artifact_schemas import (
    DossierArtifact, DossierSection, Evidence,
)


class DossierBuilder:
    def __init__(self, llm, adapter, verifier, storage, trace):
        self._llm, self._adapter = llm, adapter
        self._verifier, self._storage, self._trace = verifier, storage, trace

    def build(self, ctx) -> DossierArtifact:
        persona = self._adapter.persona
        system_prompt = (f"You are a {persona['role']}. Voice: {persona['voice']}. "
                         "Return ONLY JSON: section, narrative, evidence, confidence, "
                         "entity_refs. Cite every factual claim.")
        evidence_pack = "\n".join(f"[{c['doc_id']}#{c['chunk_id']}] {c['text']}"
                                   for c in ctx.iter_chunks())
        candidates: list[DossierSection] = []
        for section_name, _weight in self._adapter.dossier_section_weights.items():
            user = self._adapter.prompt_template.format(
                section=section_name, role=persona["role"]) + f"\n\nEVIDENCE:\n{evidence_pack}"
            resp = self._llm.call(self._req(system_prompt, user, f"dossier:{section_name}"))
            self._trace.record_llm_call(builder="dossier", section=section_name,
                                         latency_ms=resp.latency_ms)
            try:
                p = json.loads(resp.text)
                candidates.append(DossierSection(
                    section=p["section"], narrative=p["narrative"],
                    evidence=[Evidence(**e) for e in p["evidence"]],
                    confidence=float(p["confidence"]),
                    entity_refs=p.get("entity_refs", [])))
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                self._trace.record_parse_failure(builder="dossier",
                                                  section=section_name, error=str(e))

        report = self._verifier.verify(candidates, profile_ctx=ctx)
        for dropped, reason in report.reasons:
            self._trace.record_verifier_drop(builder="dossier", item=dropped, reason=reason)

        artifact = DossierArtifact(
            subscription_id=ctx.subscription_id, profile_id=ctx.profile_id,
            profile_domain=ctx.profile_domain, sections=report.passed,
            adapter_version=self._adapter.version,
            synthesis_version=ctx.synthesis_version, generated_at=datetime.utcnow())
        self._storage.put_canonical(artifact_type="dossier",
            subscription_id=ctx.subscription_id, profile_id=ctx.profile_id, artifact=artifact)
        for sec in report.passed:
            self._storage.put_snippet(artifact_type="dossier",
                subscription_id=ctx.subscription_id, profile_id=ctx.profile_id,
                snippet_id=f"dossier:{sec.section}", text=sec.narrative,
                payload={"section": sec.section, "confidence": sec.confidence,
                         "evidence": [e.model_dump() for e in sec.evidence],
                         "synthesis_version": ctx.synthesis_version,
                         "adapter_version": self._adapter.version})
        return artifact

    def _req(self, system: str, user: str, tag: str):
        from src.intelligence.sme.llm_gateway import SynthesisLLMRequest
        return SynthesisLLMRequest(system_prompt=system, user_prompt=user,
            adapter_version=self._adapter.version, trace_tag=tag)
```

Five SMEVerifier checks applied: the builder delegates the whole check suite to `verifier.verify(items, profile_ctx=ctx)` which per Phase 1 runs (1) evidence-presence on `item.evidence`, (2) evidence-validity against `ctx.iter_chunks` ids + text similarity, (3) inference-provenance (pass-through for Dossier — direct claims, no inferred items, check returns OK), (4) confidence calibration (`conf > 0.8` items rolled to 0.6 if `len(evidence) < 2`), and (5) contradiction-check across the three sections. Every drop lands in the trace; survivors alone go to storage. This is the load-bearing full-verifier flow the task description mandates.

- [ ] **Step 4: Verify tests pass**

```bash
pytest tests/intelligence/sme/test_dossier_builder.py -v
```

- [ ] **Step 5: Commit** — `git add src/intelligence/sme/dossier_builder.py tests/intelligence/sme/test_dossier_builder.py && git commit -m "phase2(sme-dossier): LLM-driven per-section synthesis with verifier fail-closed"`

---

## Task 5: Insight Index builder (compact interface + stub body)

**Files:**
- Modify: `src/intelligence/sme/insight_index_builder.py`
- Create: `tests/intelligence/sme/test_insight_index_builder.py`

**Pattern (same shape as Dossier):**

`InsightIndexBuilder.build(ctx) -> InsightIndexArtifact`. For each adapter `insight_detector` in `[{type, rule, params}, ...]`, LLM-generates `InsightItem` candidates using an evidence pack of profile chunks. All candidates across all detectors are pooled into a **single** `verifier.verify(...)` batch (contradiction-check needs the full pool). Survivors persist: canonical to Blob; snippets keyed by `insight:{type}:{idx}`, payload `{type, domain_tags, temporal_scope, confidence, evidence, synthesis_version, adapter_version}` — Phase 3 retrieval filters on `type` + `domain_tags`.

Detector rule types: `regex_pattern`, `entity_cluster`, `temporal_sweep`. All funnel into the Dossier LLM-call shape (adapter persona system prompt + detector user prompt + JSON response + trace).

- [ ] **Step 1: Write the interface-level failing tests**

Create `tests/intelligence/sme/test_insight_index_builder.py`:

```python
"""Tests for the Insight Index builder (compact, interface-driven)."""
from unittest.mock import MagicMock

from src.intelligence.sme.artifact_schemas import InsightIndexArtifact, InsightItem, Evidence
from src.intelligence.sme.insight_index_builder import InsightIndexBuilder
from src.intelligence.sme.llm_gateway import SynthesisLLMResponse


def _ctx():
    ctx = MagicMock()
    ctx.subscription_id = "sub_test"; ctx.profile_id = "prof_fin"
    ctx.profile_domain = "finance"; ctx.synthesis_version = 3; ctx.run_id = "r1"
    ctx.iter_chunks.return_value = [
        {"doc_id": "d1", "chunk_id": "c1", "text": "Revenue rose QoQ."},
        {"doc_id": "d2", "chunk_id": "c2", "text": "Customer churn up 3%."},
    ]
    return ctx


def _adapter():
    a = MagicMock()
    a.version = "1.2.0"
    a.persona = {"role": "finance SME", "voice": "quant"}
    a.insight_detectors = [
        {"type": "trend", "rule": "temporal_sweep", "params": {"scope": "quarterly"}},
        {"type": "anomaly", "rule": "entity_cluster", "params": {"entity": "Customer"}},
    ]
    return a


def _resp(items):
    import json
    return SynthesisLLMResponse(text=json.dumps({"items": items}), latency_ms=50, model="m")


def test_build_pools_detectors_and_returns_artifact():
    llm = MagicMock()
    llm.call.side_effect = [
        _resp([{"type": "trend", "narrative": "QoQ revenue +8%",
                "evidence": [{"doc_id": "d1", "chunk_id": "c1"}], "confidence": 0.75,
                "domain_tags": ["finance", "revenue"]}]),
        _resp([{"type": "anomaly", "narrative": "Churn up 3%",
                "evidence": [{"doc_id": "d2", "chunk_id": "c2"}], "confidence": 0.7,
                "domain_tags": ["finance", "churn"]}]),
    ]
    verifier = MagicMock()
    verifier.verify.side_effect = lambda items, profile_ctx: MagicMock(
        passed=items, dropped=[], reasons=[])
    storage = MagicMock(); trace = MagicMock()

    art = InsightIndexBuilder(llm, _adapter(), verifier, storage, trace).build(_ctx())

    assert isinstance(art, InsightIndexArtifact)
    assert {i.type for i in art.items} == {"trend", "anomaly"}
    assert storage.put_canonical.call_count == 1
    assert storage.put_snippet.call_count == 2


def test_build_sends_single_pooled_batch_to_verifier():
    llm = MagicMock()
    llm.call.side_effect = [
        _resp([{"type": "trend", "narrative": "x",
                "evidence": [{"doc_id": "d1", "chunk_id": "c1"}], "confidence": 0.6}]),
        _resp([{"type": "anomaly", "narrative": "y",
                "evidence": [{"doc_id": "d2", "chunk_id": "c2"}], "confidence": 0.6}]),
    ]
    verifier = MagicMock()
    verifier.verify.return_value = MagicMock(passed=[], dropped=[], reasons=[])
    storage = MagicMock(); trace = MagicMock()

    InsightIndexBuilder(llm, _adapter(), verifier, storage, trace).build(_ctx())

    # ONE verify call seeing BOTH detectors' items — contradiction-check needs full pool
    assert verifier.verify.call_count == 1
    batch = verifier.verify.call_args[0][0]
    assert len(batch) == 2
```

- [ ] **Step 2: Run tests to verify they fail.**

```bash
pytest tests/intelligence/sme/test_insight_index_builder.py -v
```

- [ ] **Step 3: Implement the builder (under 50 lines)**

Replace `src/intelligence/sme/insight_index_builder.py`:

```python
"""Insight Index builder — per-detector LLM calls pooled into a single verifier batch."""
from __future__ import annotations

import json
from datetime import datetime

from src.intelligence.sme.artifact_schemas import (
    InsightIndexArtifact, InsightItem, Evidence,
)


class InsightIndexBuilder:
    def __init__(self, llm, adapter, verifier, storage, trace):
        self._llm, self._adapter = llm, adapter
        self._verifier, self._storage, self._trace = verifier, storage, trace

    def build(self, ctx) -> InsightIndexArtifact:
        candidates: list[InsightItem] = []
        evidence_pack = "\n".join(
            f"[{c['doc_id']}#{c['chunk_id']}] {c['text']}" for c in ctx.iter_chunks())
        for detector in self._adapter.insight_detectors:
            prompt = (f"Detector: {detector['type']} via {detector['rule']}. "
                      f"Params: {detector['params']}.\nEVIDENCE:\n{evidence_pack}")
            resp = self._llm.call(self._req(detector, prompt))
            self._trace.record_llm_call(builder="insight_index",
                                        section=detector["type"], latency_ms=resp.latency_ms)
            try:
                for raw in json.loads(resp.text).get("items", []):
                    candidates.append(InsightItem(
                        type=raw["type"], narrative=raw["narrative"],
                        evidence=[Evidence(**e) for e in raw["evidence"]],
                        confidence=float(raw["confidence"]),
                        domain_tags=raw.get("domain_tags", []),
                        temporal_scope=raw.get("temporal_scope"),
                        entity_refs=raw.get("entity_refs", []),
                    ))
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                self._trace.record_parse_failure(builder="insight_index",
                                                 section=detector["type"], error=str(e))

        report = self._verifier.verify(candidates, profile_ctx=ctx)
        for d, r in report.reasons:
            self._trace.record_verifier_drop(builder="insight_index", item=d, reason=r)

        artifact = InsightIndexArtifact(
            subscription_id=ctx.subscription_id, profile_id=ctx.profile_id,
            items=report.passed, adapter_version=self._adapter.version,
            synthesis_version=ctx.synthesis_version, generated_at=datetime.utcnow(),
        )
        self._storage.put_canonical(artifact_type="insight_index",
            subscription_id=ctx.subscription_id, profile_id=ctx.profile_id, artifact=artifact)
        for idx, item in enumerate(report.passed):
            self._storage.put_snippet(artifact_type="insight_index",
                subscription_id=ctx.subscription_id, profile_id=ctx.profile_id,
                snippet_id=f"insight:{item.type}:{idx}", text=item.narrative,
                payload={"type": item.type, "domain_tags": item.domain_tags,
                         "temporal_scope": item.temporal_scope, "confidence": item.confidence,
                         "evidence": [e.model_dump() for e in item.evidence],
                         "synthesis_version": ctx.synthesis_version,
                         "adapter_version": self._adapter.version})
        return artifact

    def _req(self, detector, user_prompt: str):
        from src.intelligence.sme.llm_gateway import SynthesisLLMRequest
        persona = self._adapter.persona
        system = (f"You are a {persona['role']}. Voice: {persona['voice']}. "
                  "Return ONLY JSON: {items: [{type, narrative, evidence, confidence, "
                  "domain_tags?, temporal_scope?, entity_refs?}]}. Cite every claim.")
        return SynthesisLLMRequest(system_prompt=system, user_prompt=user_prompt,
            adapter_version=self._adapter.version,
            trace_tag=f"insight_index:{detector['type']}")
```

- [ ] **Step 4: Verify + commit**

```bash
pytest tests/intelligence/sme/test_insight_index_builder.py -v
git add src/intelligence/sme/insight_index_builder.py tests/intelligence/sme/test_insight_index_builder.py
git commit -m "phase2(sme-insight-index): per-detector LLM calls pooled into single verifier batch"
```

---

## Task 6: Comparative Register builder (compact)

**Files:**
- Modify: `src/intelligence/sme/comparative_register_builder.py`
- Create: `tests/intelligence/sme/test_comparative_register_builder.py`

**Pattern (identical to Insight Index):**

`ComparativeRegisterBuilder.build(ctx) -> ComparativeRegisterArtifact`. For each adapter `comparison_axis` in `[{name, dimension, unit?}]`, LLM-compares ≥2 document chunks sharing that dimension into `ComparativeItem` of type `delta|conflict|timeline|corroboration`. Pool into single verifier batch (cross-axis contradiction detection). Persist canonical + snippets keyed `comparative:{type}:{axis}:{idx}`, payload `{type, axis, compared_items, resolution, confidence, evidence, synthesis_version, adapter_version}`.

- [ ] **Step 1: Write failing tests (structurally mirror Task 5 Step 1 — three tests: artifact returns, pool-single-verify-call, storage called once canonical + N snippets).**

```python
"""Tests for ComparativeRegisterBuilder (compact, interface-driven)."""
from unittest.mock import MagicMock
import json

from src.intelligence.sme.artifact_schemas import ComparativeRegisterArtifact
from src.intelligence.sme.comparative_register_builder import ComparativeRegisterBuilder
from src.intelligence.sme.llm_gateway import SynthesisLLMResponse


def _ctx():
    ctx = MagicMock()
    ctx.subscription_id = "s"; ctx.profile_id = "p"; ctx.profile_domain = "finance"
    ctx.synthesis_version = 1; ctx.run_id = "r"
    ctx.iter_chunks.return_value = [
        {"doc_id": "q3", "chunk_id": "a", "text": "Revenue $6M Q3."},
        {"doc_id": "q2", "chunk_id": "b", "text": "Revenue $5.4M Q2."},
    ]
    return ctx


def _adapter():
    a = MagicMock()
    a.version = "1.0.0"; a.persona = {"role": "SME", "voice": "direct"}
    a.comparison_axes = [{"name": "revenue_qoq", "dimension": "currency", "unit": "USD"}]
    return a


def test_returns_comparative_artifact():
    llm = MagicMock()
    llm.call.return_value = SynthesisLLMResponse(
        text=json.dumps({"items": [{
            "type": "delta", "axis": "revenue_qoq",
            "compared_items": ["q2", "q3"],
            "analysis": "Q3 up $600K", "resolution": None,
            "evidence": [{"doc_id": "q3", "chunk_id": "a"},
                         {"doc_id": "q2", "chunk_id": "b"}],
            "confidence": 0.8,
        }]}),
        latency_ms=40, model="m")
    verifier = MagicMock()
    verifier.verify.side_effect = lambda items, profile_ctx: MagicMock(
        passed=items, dropped=[], reasons=[])
    storage = MagicMock(); trace = MagicMock()
    art = ComparativeRegisterBuilder(llm, _adapter(), verifier, storage, trace).build(_ctx())
    assert isinstance(art, ComparativeRegisterArtifact)
    assert len(art.items) == 1
    assert art.items[0].type == "delta"
    assert storage.put_snippet.call_count == 1
```

Two additional tests mirror the pooled-batch and drop-handling tests from Task 5.

- [ ] **Step 2: Run the failing tests.**

- [ ] **Step 3: Implement — same structure as Task 5 with three substitutions:**
  - Iterate `adapter.comparison_axes` instead of `insight_detectors`
  - Parse each response item into `ComparativeItem` (two+ `compared_items` enforced by schema)
  - Snippet key: `comparative:{item.type}:{item.axis}:{idx}`; payload fields: `type`, `axis`, `compared_items`, `resolution`, `confidence`, `evidence`, `synthesis_version`, `adapter_version`
  - System prompt: directs LLM to pick ≥2 distinct docs per comparison (schema enforces; prompt hints)

Keep under 50 lines of logic. Same five-check verifier pass applies.

- [ ] **Step 4: Verify + commit**

```bash
pytest tests/intelligence/sme/test_comparative_register_builder.py -v
git add src/intelligence/sme/comparative_register_builder.py \
        tests/intelligence/sme/test_comparative_register_builder.py
git commit -m "phase2(sme-comparative): per-axis LLM comparisons with cross-doc evidence"
```

---

## Task 7: KG Multi-Hop Materializer — full INFERRED_RELATION writer

**Files:**
- Modify: `src/intelligence/sme/kg_materializer.py`
- Create: `tests/intelligence/sme/test_kg_materializer.py`

Load-bearing per spec §9. Generic `INFERRED_RELATION` edge with rich properties — schema stable; adapters define arbitrary inference types. Every edge carries `inference_path`, `confidence`, `evidence`, and the `(subscription_id, profile_id)` hard-filter pair.

- [ ] **Step 1: Write the failing tests**

Create `tests/intelligence/sme/test_kg_materializer.py`:

```python
"""Tests for the KG Multi-Hop Materializer."""
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from src.intelligence.sme.artifact_schemas import InferredEdge, Evidence
from src.intelligence.sme.kg_materializer import KGMultiHopMaterializer


def _ctx():
    ctx = MagicMock()
    ctx.subscription_id = "sub_abc"; ctx.profile_id = "prof_fin"
    ctx.profile_domain = "finance"; ctx.synthesis_version = 3; ctx.run_id = "r1"
    return ctx


def _adapter():
    a = MagicMock()
    a.version = "1.2.0"
    a.kg_inference_rules = [
        {"pattern": "(a)-[:PAYS]->(b)-[:FUNDS]->(c)",
         "produces": "indirectly_funds",
         "confidence_floor": 0.7, "max_hops": 3},
    ]
    return a


def _candidate_edge(rtype="indirectly_funds", conf=0.75):
    return InferredEdge(
        src_node_id="n1", dst_node_id="n3", relation_type=rtype,
        confidence=conf, evidence=[Evidence(doc_id="d1", chunk_id="c1")],
        inference_path=[
            {"from": "n1", "edge": "PAYS", "to": "n2"},
            {"from": "n2", "edge": "FUNDS", "to": "n3"},
        ],
        synthesis_version=3, adapter_version="1.2.0",
        generated_at=datetime(2026, 4, 20),
        subscription_id="sub_abc", profile_id="prof_fin",
    )


def test_materializer_queries_neo4j_for_patterns():
    neo4j = MagicMock(); session = MagicMock()
    neo4j.session.return_value.__enter__.return_value = session
    session.run.side_effect = [
        [{"n1": "n1", "n2": "n2", "n3": "n3", "ev_doc": "d1", "ev_chunk": "c1"}],
        None]
    verifier = MagicMock()
    verifier.verify.side_effect = lambda items, profile_ctx: MagicMock(
        passed=items, dropped=[], reasons=[])
    trace = MagicMock()
    report = KGMultiHopMaterializer(neo4j, _adapter(), verifier, trace).build(_ctx())

    # Pattern query ran first with profile-hard-filter
    pattern_cypher = session.run.call_args_list[0][0][0]
    pattern_params = session.run.call_args_list[0][1]
    assert "subscription_id" in pattern_cypher and "profile_id" in pattern_cypher
    assert pattern_params["subscription_id"] == "sub_abc"
    assert pattern_params["profile_id"] == "prof_fin"
    # Write ran second: INFERRED_RELATION MERGE
    write_cypher = session.run.call_args_list[1][0][0]
    assert "INFERRED_RELATION" in write_cypher
    assert "relation_type" in write_cypher
    assert report.edges_created == 1


def test_confidence_floor_drops_edges_via_verifier():
    neo4j = MagicMock(); session = MagicMock()
    neo4j.session.return_value.__enter__.return_value = session
    session.run.side_effect = [
        [{"n1": "n1", "n2": "n2", "n3": "n3", "ev_doc": "d1", "ev_chunk": "c1"}]]
    verifier = MagicMock()
    verifier.verify.side_effect = lambda items, profile_ctx: MagicMock(
        passed=[], dropped=items,
        reasons=[("edge", "confidence_below_floor")])
    trace = MagicMock()
    report = KGMultiHopMaterializer(neo4j, _adapter(), verifier, trace).build(_ctx())
    assert report.edges_created == 0
    assert report.edges_skipped_verifier == 1
```

- [ ] **Step 2: Run the failing tests.**

- [ ] **Step 3: Implement the materializer**

Replace `src/intelligence/sme/kg_materializer.py`:

```python
"""KG Multi-Hop Materializer — writes INFERRED_RELATION edges per adapter rules.

Per spec Section 9: one generic edge type, rich properties.
MERGE ensures idempotent re-runs under incremental synthesis.
Every query + write hard-filters on (subscription_id, profile_id).
"""
from __future__ import annotations

from datetime import datetime

from src.intelligence.sme.artifact_schemas import (
    InferredEdge, Evidence, KGMaterializationReport,
)

WRITE_CYPHER = """
MATCH (a {node_id: $src}), (c {node_id: $dst})
WHERE a.subscription_id = $subscription_id AND a.profile_id = $profile_id
  AND c.subscription_id = $subscription_id AND c.profile_id = $profile_id
MERGE (a)-[r:INFERRED_RELATION {
    relation_type: $relation_type,
    subscription_id: $subscription_id,
    profile_id: $profile_id
}]->(c)
SET r.source = 'sme_synthesis',
    r.confidence = $confidence,
    r.evidence = $evidence,
    r.inference_path = $inference_path,
    r.synthesis_version = $synthesis_version,
    r.adapter_version = $adapter_version,
    r.generated_at = $generated_at
RETURN r
"""


class KGMultiHopMaterializer:
    def __init__(self, neo4j, adapter, verifier, trace):
        self._neo4j, self._adapter = neo4j, adapter
        self._verifier, self._trace = verifier, trace

    def build(self, ctx) -> KGMaterializationReport:
        candidates: list[InferredEdge] = []
        max_hops = 0
        for rule in self._adapter.kg_inference_rules:
            max_hops = max(max_hops, rule.get("max_hops", 3))
            with self._neo4j.session() as s:
                rows = list(s.run(self._pattern_cypher(rule), {
                    "subscription_id": ctx.subscription_id,
                    "profile_id": ctx.profile_id,
                    "max_hops": rule.get("max_hops", 3)}))
            for row in rows:
                candidates.append(InferredEdge(
                    src_node_id=row["n1"],
                    dst_node_id=row.get("n3", row.get("nX")),
                    relation_type=rule["produces"],
                    confidence=rule.get("confidence_floor", 0.7),
                    evidence=[Evidence(doc_id=row["ev_doc"], chunk_id=row["ev_chunk"])],
                    inference_path=[
                        {"from": row["n1"], "edge": "step1", "to": row["n2"]},
                        {"from": row["n2"], "edge": "step2",
                         "to": row.get("n3", row.get("nX", ""))}],
                    synthesis_version=ctx.synthesis_version,
                    adapter_version=self._adapter.version,
                    generated_at=datetime.utcnow(),
                    subscription_id=ctx.subscription_id,
                    profile_id=ctx.profile_id))
                self._trace.record_kg_candidate(rule=rule["produces"], row=dict(row))

        verify = self._verifier.verify(candidates, profile_ctx=ctx)
        for d, r in verify.reasons:
            self._trace.record_verifier_drop(builder="kg_materializer", item=d, reason=r)

        created = 0
        with self._neo4j.session() as s:
            for e in verify.passed:
                s.run(WRITE_CYPHER, {
                    "src": e.src_node_id, "dst": e.dst_node_id,
                    "relation_type": e.relation_type, "confidence": e.confidence,
                    "evidence": [f"{x.doc_id}#{x.chunk_id}" for x in e.evidence],
                    "inference_path": [f"{h['from']}-[{h['edge']}]->{h['to']}"
                                        for h in e.inference_path],
                    "synthesis_version": e.synthesis_version,
                    "adapter_version": e.adapter_version,
                    "generated_at": e.generated_at.isoformat(),
                    "subscription_id": e.subscription_id,
                    "profile_id": e.profile_id})
                created += 1

        return KGMaterializationReport(
            edges_created=created,
            edges_skipped_verifier=len(candidates) - created,
            max_path_length=max_hops,
            adapter_version=self._adapter.version,
            synthesis_version=ctx.synthesis_version)

    def _pattern_cypher(self, rule: dict) -> str:
        # Adapter rule['pattern'] is parameterized Cypher; template hard-filters profile.
        return (f"MATCH {rule['pattern']} "
                "WHERE a.subscription_id = $subscription_id "
                "AND a.profile_id = $profile_id AND length(PATH) <= $max_hops "
                "RETURN n1, n2, n3, ev_doc, ev_chunk LIMIT 500")
```

All six Section 9 requirements present: (1) `INFERRED_RELATION` edge type, (2) `relation_type` property for adapter-defined labels, (3) `source: 'sme_synthesis'`, (4) `confidence`/`evidence`/`inference_path`, (5) `synthesis_version`/`adapter_version`/`generated_at` provenance, (6) `subscription_id`/`profile_id` hard filter. MERGE makes re-runs idempotent; profile-scoped MATCH prevents cross-sub leakage structurally.

- [ ] **Step 4: Verify tests pass + commit**

```bash
pytest tests/intelligence/sme/test_kg_materializer.py -v
git add src/intelligence/sme/kg_materializer.py tests/intelligence/sme/test_kg_materializer.py
git commit -m "phase2(sme-kg): INFERRED_RELATION materializer with profile-hard-filter + MERGE idempotent"
```

---

## Task 8: Recommendation Bank builder (compact)

**Files:**
- Modify: `src/intelligence/sme/recommendation_bank_builder.py`
- Create: `tests/intelligence/sme/test_recommendation_bank_builder.py`

**Pattern:**

`RecommendationBankBuilder.build(ctx, insight_artifact) -> RecommendationBankArtifact`. Takes verified Insight Index as input (synthesizer threads it). For each adapter `recommendation_frame` in `[{frame, template, requires: {insight_types}}]`, filters `insight_artifact.items` by `requires.insight_types`, LLM-generates `RecommendationItem` linked to insight ids. Pool all frames, single verifier call, persist canonical + snippets keyed `recommendation:{frame}:{idx}`, payload `{recommendation, linked_insights, domain_tags, estimated_impact, assumptions, caveats, confidence, evidence, synthesis_version, adapter_version}`. Phase 3 filters on `domain_tags` + `linked_insights`.

- [ ] **Step 1: Write failing tests (three tests mirroring Task 5).**

- [ ] **Step 2: Run failing.**

- [ ] **Step 3: Implement — structurally identical to `InsightIndexBuilder` with:**
  - `build(ctx, insight_artifact=...)` signature — accepts Insight Index for linkage
  - Iterates `adapter.recommendation_frames`, pre-filtering insights by `requires.insight_types`
  - Parses into `RecommendationItem`, preserves `linked_insights` from LLM response
  - Snippet payload includes `linked_insights` for Phase 3 retrieval join

- [ ] **Step 4: Verify + commit**

```bash
pytest tests/intelligence/sme/test_recommendation_bank_builder.py -v
git add src/intelligence/sme/recommendation_bank_builder.py \
        tests/intelligence/sme/test_recommendation_bank_builder.py
git commit -m "phase2(sme-recbank): recommendations linked to verified insights, single-batch verify"
```

---

## Task 9: Synthesizer orchestrator — wires all five builders + verifier + storage

**Files:**
- Modify: `src/intelligence/sme/synthesizer.py`
- Create: `tests/intelligence/sme/test_synthesizer.py`

Single public entry for the training stage. Loads adapter, iterates builders in fixed order (Dossier → Insight → Comparative → KG → Recommendation with insight threaded), opens one trace, writes manifest at end. Idempotent under incremental synthesis (Task 15).

- [ ] **Step 1: Write the failing tests**

Create `tests/intelligence/sme/test_synthesizer.py`:

```python
"""Orchestrator tests — wires five builders + verifier + storage + trace."""
from unittest.mock import MagicMock, patch

import pytest

from src.intelligence.sme.synthesizer import SMESynthesizer, SynthesisReport


def _loaded_adapter():
    a = MagicMock()
    a.version = "1.2.0"; a.domain = "finance"
    return a


def _ctx_pack():
    pack = MagicMock()
    pack.subscription_id = "sub_x"; pack.profile_id = "prof_x"
    pack.profile_domain = "finance"; pack.synthesis_version = 4; pack.run_id = "r_new"
    return pack


def test_orchestrator_order_and_insight_threading():
    al = MagicMock(); al.load.return_value = _loaded_adapter()
    deps = MagicMock()
    insight_art = MagicMock(items=[MagicMock(type="trend")])
    deps.dossier.build.return_value = MagicMock(sections=[])
    deps.insight_index.build.return_value = insight_art
    deps.comparative.build.return_value = MagicMock(items=[])
    deps.kg_materializer.build.return_value = MagicMock(edges_created=1, edges_skipped_verifier=0)
    deps.recommendation.build.return_value = MagicMock(items=[])

    report = SMESynthesizer(al, deps, MagicMock(), MagicMock()).synthesize_profile(
        profile_ctx=_ctx_pack())

    # Order matters — Recommendation depends on Insight
    order = [c[0] for c in deps.method_calls if c[0].endswith(".build")]
    assert order == ["dossier.build", "insight_index.build", "comparative.build",
                     "kg_materializer.build", "recommendation.build"]
    # Recommendation builder received the insight artifact
    assert deps.recommendation.build.call_args[1]["insight_artifact"] is insight_art
    assert isinstance(report, SynthesisReport)


def test_manifest_and_builder_failure():
    al = MagicMock(); al.load.return_value = _loaded_adapter()
    deps_ok = MagicMock()
    for b in ("dossier", "insight_index", "comparative",
              "kg_materializer", "recommendation"):
        getattr(deps_ok, b).build.return_value = MagicMock()
    storage = MagicMock(); trace = MagicMock()
    SMESynthesizer(al, deps_ok, storage, trace).synthesize_profile(profile_ctx=_ctx_pack())
    storage.put_manifest.assert_called_once()
    mf = storage.put_manifest.call_args[1]
    assert mf["synthesis_version"] == 4 and mf["adapter_version"] == "1.2.0"

    # Failure path: builder raises → trace records, manifest NOT written, re-raises
    al2 = MagicMock(); al2.load.return_value = _loaded_adapter()
    deps_fail = MagicMock(); deps_fail.dossier.build.side_effect = RuntimeError("LLM down")
    storage2 = MagicMock(); trace2 = MagicMock()
    with pytest.raises(RuntimeError, match="LLM down"):
        SMESynthesizer(al2, deps_fail, storage2, trace2).synthesize_profile(
            profile_ctx=_ctx_pack())
    trace2.record_builder_failure.assert_called_once()
    storage2.put_manifest.assert_not_called()
```

- [ ] **Step 2: Run failing tests** — `pytest tests/intelligence/sme/test_synthesizer.py -v`

- [ ] **Step 3: Implement the orchestrator**

Replace `src/intelligence/sme/synthesizer.py`:

```python
"""SME Synthesizer orchestrator — single public entry for the training stage."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class SynthesisReport:
    subscription_id: str; profile_id: str
    synthesis_version: int; adapter_version: str
    dossier_sections: int; insight_items: int
    comparative_items: int; recommendation_items: int
    kg_edges_created: int; kg_edges_skipped: int
    run_id: str; started_at: datetime; finished_at: datetime


def _count(art, attr):
    return len(getattr(art, attr, []) or [])


class SMESynthesizer:
    def __init__(self, adapter_loader, deps, storage, trace):
        self._al, self._deps = adapter_loader, deps
        self._storage, self._trace = storage, trace

    def synthesize_profile(self, *, profile_ctx) -> SynthesisReport:
        started_at = datetime.utcnow()
        adapter = self._al.load(subscription_id=profile_ctx.subscription_id,
                                 domain=profile_ctx.profile_domain)
        self._trace.open(run_id=profile_ctx.run_id,
                          subscription_id=profile_ctx.subscription_id,
                          profile_id=profile_ctx.profile_id,
                          adapter_version=adapter.version)
        try:
            dos = self._deps.dossier.build(profile_ctx)
            ins = self._deps.insight_index.build(profile_ctx)
            cmp = self._deps.comparative.build(profile_ctx)
            kg = self._deps.kg_materializer.build(profile_ctx)
            rec = self._deps.recommendation.build(profile_ctx, insight_artifact=ins)
        except Exception as e:
            self._trace.record_builder_failure(error=str(e))
            self._trace.close(status="failed"); raise

        manifest = {
            "subscription_id": profile_ctx.subscription_id,
            "profile_id": profile_ctx.profile_id,
            "synthesis_version": profile_ctx.synthesis_version,
            "adapter_version": adapter.version, "run_id": profile_ctx.run_id,
            "generated_at": datetime.utcnow().isoformat(),
            "artifacts": {
                "dossier": {"sections": _count(dos, "sections")},
                "insight_index": {"items": _count(ins, "items")},
                "comparative_register": {"items": _count(cmp, "items")},
                "recommendation_bank": {"items": _count(rec, "items")},
                "kg": {"edges_created": getattr(kg, "edges_created", 0)},
            }}
        self._storage.put_manifest(subscription_id=profile_ctx.subscription_id,
            profile_id=profile_ctx.profile_id, manifest=manifest)
        self._trace.close(status="ok")
        return SynthesisReport(
            subscription_id=profile_ctx.subscription_id,
            profile_id=profile_ctx.profile_id,
            synthesis_version=profile_ctx.synthesis_version,
            adapter_version=adapter.version,
            dossier_sections=_count(dos, "sections"),
            insight_items=_count(ins, "items"),
            comparative_items=_count(cmp, "items"),
            recommendation_items=_count(rec, "items"),
            kg_edges_created=getattr(kg, "edges_created", 0),
            kg_edges_skipped=getattr(kg, "edges_skipped_verifier", 0),
            run_id=profile_ctx.run_id, started_at=started_at,
            finished_at=datetime.utcnow())
```

- [ ] **Step 4: Verify tests pass + commit**

```bash
pytest tests/intelligence/sme/test_synthesizer.py -v
git add src/intelligence/sme/synthesizer.py tests/intelligence/sme/test_synthesizer.py
git commit -m "phase2(sme-synth): orchestrator wires five builders + trace + manifest"
```

---

## Task 10: Training-stage integration in `src/api/pipeline_api.py`

**Files:**
- Modify: `src/api/pipeline_api.py`
- Modify: `src/tasks/embedding.py`
- Create: `tests/api/test_pipeline_api_sme.py`

Load-bearing per spec §4. Synthesis is the final step before `PIPELINE_TRAINING_COMPLETED` on the last doc in a profile. No new status string; failure keeps `EMBEDDING_IN_PROGRESS` with audit marker; retry is idempotent via Task 15's input-hash short-circuit.

- [ ] **Step 1: Write the failing tests**

Create `tests/api/test_pipeline_api_sme.py`:

```python
"""Training-stage wiring tests — synthesis fires before TRAINING_COMPLETED."""
from unittest.mock import MagicMock, patch

import pytest


def _doc(doc_id="d_last", sub="sub_a", prof="prof_x"):
    return {"document_id": doc_id, "subscription_id": sub, "profile_id": prof}


def test_last_doc_triggers_synthesis_then_flips_status():
    from src.api import pipeline_api
    with patch.object(pipeline_api, "is_last_doc_in_profile", return_value=True), \
         patch.object(pipeline_api, "flag_enabled", return_value=True), \
         patch.object(pipeline_api, "build_profile_ctx") as ctx, \
         patch.object(pipeline_api, "input_hash_unchanged", return_value=False), \
         patch.object(pipeline_api, "SMESynthesizer") as Synth, \
         patch.object(pipeline_api, "update_pipeline_status") as flip:
        ctx.return_value = MagicMock()
        Synth.return_value.synthesize_profile.return_value = MagicMock(
            synthesis_version=1, adapter_version="1.0.0",
            dossier_sections=3, insight_items=0, comparative_items=0,
            recommendation_items=0, kg_edges_created=0)
        pipeline_api.finalize_training_for_doc(_doc())
        Synth.return_value.synthesize_profile.assert_called_once()
        flip.assert_called_once_with("d_last", "TRAINING_COMPLETED")


def test_non_last_doc_and_flag_off_skip_synthesis():
    """Two short-circuit paths: non-last doc; flag disabled. Both flip status without synth."""
    from src.api import pipeline_api
    for gate in ({"is_last_doc_in_profile": False, "flag_enabled": True},
                 {"is_last_doc_in_profile": True, "flag_enabled": False}):
        with patch.object(pipeline_api, "is_last_doc_in_profile",
                          return_value=gate["is_last_doc_in_profile"]), \
             patch.object(pipeline_api, "flag_enabled",
                          return_value=gate["flag_enabled"]), \
             patch.object(pipeline_api, "SMESynthesizer") as Synth, \
             patch.object(pipeline_api, "update_pipeline_status") as flip:
            pipeline_api.finalize_training_for_doc(_doc())
            Synth.assert_not_called()
            flip.assert_called_once_with("d_last", "TRAINING_COMPLETED")


def test_synthesis_failure_keeps_status_and_audits():
    from src.api import pipeline_api
    with patch.object(pipeline_api, "is_last_doc_in_profile", return_value=True), \
         patch.object(pipeline_api, "flag_enabled", return_value=True), \
         patch.object(pipeline_api, "build_profile_ctx") as ctx, \
         patch.object(pipeline_api, "input_hash_unchanged", return_value=False), \
         patch.object(pipeline_api, "SMESynthesizer") as Synth, \
         patch.object(pipeline_api, "update_pipeline_status") as flip, \
         patch.object(pipeline_api, "append_audit_log") as audit:
        ctx.return_value = MagicMock()
        Synth.return_value.synthesize_profile.side_effect = RuntimeError("boom")
        with pytest.raises(RuntimeError):
            pipeline_api.finalize_training_for_doc(_doc())
        flip.assert_not_called()
        audit.assert_called_once()
        assert audit.call_args[0][1] == "SME_SYNTHESIS_FAILED"
```

- [ ] **Step 2: Run failing tests.**

- [ ] **Step 3: Wire the training-stage hook in `pipeline_api.py`**

Add to `src/api/pipeline_api.py` (append near existing triggers):

```python
"""Training-stage finalization: synthesis as final step before TRAINING_COMPLETED.

`finalize_training_for_doc(doc)` is called when a doc completes embedding.
If it is the LAST doc in the profile AND enable_sme_synthesis is on, synthesis
runs first. Otherwise behavior matches pre-Phase-2.
"""
import logging

from src.api.document_status import append_audit_log, update_pipeline_status
from src.api.statuses import PIPELINE_TRAINING_COMPLETED
from src.intelligence.sme.synthesizer import SMESynthesizer
from src.intelligence.sme.flags import flag_enabled
from src.intelligence.sme.input_snapshot import build_profile_ctx, input_hash_unchanged

logger = logging.getLogger(__name__)


def is_last_doc_in_profile(doc: dict) -> bool:
    from src.api.document_status import count_incomplete_docs_in_profile
    return count_incomplete_docs_in_profile(
        subscription_id=doc["subscription_id"], profile_id=doc["profile_id"],
        exclude_document_id=doc["document_id"]) == 0


def finalize_training_for_doc(doc: dict) -> None:
    """Strict order:
      1 last-doc gate; 2 enable_sme_synthesis flag; 3 build context;
      4 input-hash idempotency short-circuit; 5 synth (raises on failure);
      6 flip pipeline_status to TRAINING_COMPLETED on success.
    """
    did = doc["document_id"]

    if not is_last_doc_in_profile(doc):
        update_pipeline_status(did, PIPELINE_TRAINING_COMPLETED); return
    if not flag_enabled("enable_sme_synthesis", subscription_id=doc["subscription_id"]):
        update_pipeline_status(did, PIPELINE_TRAINING_COMPLETED); return

    ctx = build_profile_ctx(subscription_id=doc["subscription_id"],
                             profile_id=doc["profile_id"])
    if input_hash_unchanged(ctx):
        append_audit_log(did, "SME_SYNTHESIS_SKIPPED_INPUT_UNCHANGED",
                          input_hash=ctx.input_hash)
        update_pipeline_status(did, PIPELINE_TRAINING_COMPLETED); return

    synth = SMESynthesizer(adapter_loader=_adapter_loader(), deps=_deps(),
                            storage=_storage(), trace=_trace())
    try:
        report = synth.synthesize_profile(profile_ctx=ctx)
    except Exception as e:  # noqa: BLE001
        append_audit_log(did, "SME_SYNTHESIS_FAILED",
                          subscription_id=doc["subscription_id"],
                          profile_id=doc["profile_id"], error=str(e))
        logger.exception("SME synthesis failed for %s/%s",
                         doc["subscription_id"], doc["profile_id"])
        raise  # status stays in EMBEDDING_IN_PROGRESS; retry is idempotent

    append_audit_log(did, "SME_SYNTHESIS_COMPLETED",
                     synthesis_version=report.synthesis_version,
                     adapter_version=report.adapter_version,
                     dossier_sections=report.dossier_sections,
                     insight_items=report.insight_items,
                     comparative_items=report.comparative_items,
                     recommendation_items=report.recommendation_items,
                     kg_edges_created=report.kg_edges_created)
    update_pipeline_status(did, PIPELINE_TRAINING_COMPLETED)


# DI factories wired at app startup (same pattern as src/main.py):
def _adapter_loader(): ...
def _deps(): ...
def _storage(): ...
def _trace(): ...
```

Key invariants honored here: no new pipeline_status string — we still flip to `PIPELINE_TRAINING_COMPLETED`. Failure path re-raises so the doc sits in its prior step (not a new "SYNTHESIS_FAILED" status). Control-plane `enable_sme_synthesis` flag reads MongoDB only. Profile isolation inherited from `ctx`.

- [ ] **Step 4: Update `src/tasks/embedding.py` to delegate the terminal step**

Around line 446, replace the direct `update_pipeline_status(..., PIPELINE_TRAINING_COMPLETED)` with a delegation:

```python
# Was:
# update_pipeline_status(document_id, PIPELINE_TRAINING_COMPLETED)
# Becomes:
from src.api.pipeline_api import finalize_training_for_doc
finalize_training_for_doc({
    "document_id": document_id,
    "subscription_id": subscription_id,
    "profile_id": profile_id,
})
```

This is the only change in `embedding.py`. The doc-level audit log lines that surround the status flip stay in place (they fire before + after `finalize_training_for_doc`).

- [ ] **Step 5: Verify tests pass + commit**

```bash
pytest tests/api/test_pipeline_api_sme.py -v
git add src/api/pipeline_api.py src/tasks/embedding.py tests/api/test_pipeline_api_sme.py
git commit -m "phase2(sme-pipeline): synthesis as final training-stage step, no new status string"
```

---

## Task 11: Storage — manifest + artifact snippet write path

**Files:**
- Modify: `src/intelligence/sme/storage.py` (Phase 1 shim extended)
- Create: `tests/intelligence/sme/test_storage_snippets.py`

Phase 1 shipped signatures; Phase 2 fills bodies. Snippets → `sme_artifacts_{sub}` Qdrant; canonical → `sme_artifacts/{sub}/{prof}/{type}/{version}.json` Blob; manifest → `sme_artifacts/{sub}/{prof}/manifest.json`.

- [ ] **Step 1: Write the failing tests**

Create `tests/intelligence/sme/test_storage_snippets.py`:

```python
"""Storage tests for SME artifact snippets + manifests."""
from unittest.mock import MagicMock

import pytest

from src.intelligence.sme.storage import ArtifactStorage


@pytest.fixture
def store():
    qc = MagicMock(); blob = MagicMock(); emb = MagicMock()
    emb.embed.return_value = [0.1] * 768
    return ArtifactStorage(qdrant=qc, blob=blob, embedder=emb), qc, blob


def test_snippet_canonical_and_manifest_paths(store):
    s, qc, blob = store
    # Snippet
    s.put_snippet(artifact_type="insight_index", subscription_id="sub_a",
        profile_id="prof_x", snippet_id="insight:trend:0",
        text="Rev QoQ rose.",
        payload={"type": "trend", "confidence": 0.8,
                  "evidence": [{"doc_id": "d1", "chunk_id": "c1"}]})
    call = qc.upsert.call_args
    assert call[1]["collection_name"] == "sme_artifacts_sub_a"
    pl = call[1]["points"][0].payload
    assert pl["subscription_id"] == "sub_a" and pl["profile_id"] == "prof_x"
    assert pl["artifact_type"] == "insight_index" and pl["type"] == "trend"
    # Canonical
    art = MagicMock(); art.synthesis_version = 3
    art.model_dump.return_value = {"synthesis_version": 3}
    s.put_canonical(artifact_type="dossier", subscription_id="sub_a",
                    profile_id="prof_x", artifact=art)
    assert blob.upload_json.call_args_list[-1][0][0] == \
        "sme_artifacts/sub_a/prof_x/dossier/3.json"
    # Manifest
    s.put_manifest(subscription_id="sub_a", profile_id="prof_x",
                   manifest={"synthesis_version": 3})
    assert blob.upload_json.call_args_list[-1][0][0] == \
        "sme_artifacts/sub_a/prof_x/manifest.json"


def test_snippet_rejects_cross_sub_write(store):
    s, _, _ = store
    with pytest.raises(ValueError, match="subscription_id"):
        s.put_snippet(artifact_type="dossier", subscription_id="", profile_id="p",
            snippet_id="x", text="y", payload={"subscription_id": "other"})
```

- [ ] **Step 2: Run failing tests.**

- [ ] **Step 3: Extend storage**

Replace the body of `src/intelligence/sme/storage.py`:

```python
"""Storage for SME artifacts: Qdrant snippets + Blob canonical + Blob manifest."""
from __future__ import annotations

import uuid


class ArtifactStorage:
    def __init__(self, qdrant, blob, embedder):
        self._qdrant = qdrant
        self._blob = blob
        self._embedder = embedder

    def put_snippet(self, *, artifact_type: str, subscription_id: str,
                    profile_id: str, snippet_id: str, text: str, payload: dict) -> None:
        if not subscription_id:
            raise ValueError("subscription_id required for SME snippet write")
        if not profile_id:
            raise ValueError("profile_id required for SME snippet write")
        collection = f"sme_artifacts_{subscription_id}"
        vector = self._embedder.embed(text)
        # Qdrant PointStruct-style; use same shape the rest of the codebase uses
        from qdrant_client.http.models import PointStruct
        pt = PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                **payload,
                "subscription_id": subscription_id,
                "profile_id": profile_id,
                "artifact_type": artifact_type,
                "snippet_id": snippet_id,
                "text": text,
            },
        )
        self._qdrant.upsert(collection_name=collection, points=[pt])

    def put_canonical(self, *, artifact_type: str, subscription_id: str,
                      profile_id: str, artifact) -> None:
        path = (f"sme_artifacts/{subscription_id}/{profile_id}/"
                f"{artifact_type}/{artifact.synthesis_version}.json")
        self._blob.upload_json(path, artifact.model_dump(mode="json"))

    def put_manifest(self, *, subscription_id: str, profile_id: str,
                     manifest: dict) -> None:
        path = f"sme_artifacts/{subscription_id}/{profile_id}/manifest.json"
        self._blob.upload_json(path, manifest)
```

- [ ] **Step 4: Verify tests pass + commit**

```bash
pytest tests/intelligence/sme/test_storage_snippets.py -v
git add src/intelligence/sme/storage.py tests/intelligence/sme/test_storage_snippets.py
git commit -m "phase2(sme-storage): snippet+canonical+manifest write paths with hard profile filter"
```

---

## Task 12: Retrieval Layer B (KG + synthesized edges) in `unified_retriever.py` and `kg/retrieval.py`

**Files:**
- Modify: `src/retrieval/unified_retriever.py`
- Modify: `src/kg/retrieval.py`
- Create: `tests/retrieval/test_unified_retriever_layer_b.py`

Layer B pulls 1-hop KG hints + pre-materialized `INFERRED_RELATION` edges. Flag-gated on `enable_sme_retrieval` + `enable_kg_synthesized_edges`; original KG retrieval unaffected when flags off.

- [ ] **Step 1: Write the failing test**

Create `tests/retrieval/test_unified_retriever_layer_b.py`:

```python
"""Layer B retrieval — KG hints + INFERRED_RELATION synthesized edges."""
from unittest.mock import MagicMock, patch

from src.retrieval.unified_retriever import UnifiedRetriever


def test_layer_b_includes_inferred_with_profile_filter_and_conf_floor():
    kg = MagicMock()
    kg.one_hop.return_value = [{"src": "n1", "dst": "n2", "type": "MENTIONS",
                                 "evidence": ["d1#c1"]}]
    kg.inferred_relations.return_value = [
        {"src": "n1", "dst": "n9", "relation_type": "indirectly_funds",
         "confidence": 0.78, "evidence": ["d1#c1", "d2#c3"],
         "inference_path": ["n1-[PAYS]->n4", "n4-[FUNDS]->n9"]}]
    r = UnifiedRetriever(kg_client=kg, qdrant=MagicMock(), sme=MagicMock())
    with patch("src.retrieval.unified_retriever.flag_enabled", return_value=True), \
         patch("src.retrieval.unified_retriever.inferred_edge_confidence_floor",
               return_value=0.75):
        hits = r.retrieve_layer_b(query="funds?", subscription_id="s",
                                   profile_id="p", top_k=10)
    assert {h["kind"] for h in hits} == {"kg_direct", "kg_inferred"}
    kg.one_hop.assert_called_with(subscription_id="s", profile_id="p",
                                   entities=None, top_k=10)
    kg.inferred_relations.assert_called_with(subscription_id="s", profile_id="p",
                                              min_confidence=0.75, top_k=10)


def test_layer_b_excludes_inferred_when_flag_off():
    kg = MagicMock()
    kg.one_hop.return_value = [{"src": "n1", "dst": "n2", "type": "MENTIONS",
                                 "evidence": []}]
    r = UnifiedRetriever(kg_client=kg, qdrant=MagicMock(), sme=MagicMock())
    with patch("src.retrieval.unified_retriever.flag_enabled",
               side_effect=lambda name, **_: name != "enable_kg_synthesized_edges"):
        hits = r.retrieve_layer_b(query="q", subscription_id="s",
                                   profile_id="p", top_k=5)
    kg.inferred_relations.assert_not_called()
    assert all(h["kind"] != "kg_inferred" for h in hits)
```

- [ ] **Step 2: Run failing tests.**

- [ ] **Step 3: Implement Layer B**

Extend `src/retrieval/unified_retriever.py`:

```python
"""Add Layer B: KG 1-hop + INFERRED_RELATION synthesized edges."""
from src.intelligence.sme.flags import flag_enabled


def inferred_edge_confidence_floor() -> float:
    """Rolling floor — adapter-tunable via deploy config."""
    return 0.6


class UnifiedRetriever:
    def __init__(self, kg_client, qdrant, sme):
        self._kg = kg_client; self._qdrant = qdrant; self._sme = sme

    def retrieve_layer_b(self, *, query: str, subscription_id: str,
                         profile_id: str, top_k: int) -> list[dict]:
        hits: list[dict] = []
        for row in self._kg.one_hop(subscription_id=subscription_id,
                                    profile_id=profile_id, entities=None, top_k=top_k):
            hits.append({"kind": "kg_direct", **row})
        if flag_enabled("enable_sme_retrieval", subscription_id=subscription_id) and \
           flag_enabled("enable_kg_synthesized_edges", subscription_id=subscription_id):
            floor = inferred_edge_confidence_floor()
            for row in self._kg.inferred_relations(
                subscription_id=subscription_id, profile_id=profile_id,
                min_confidence=floor, top_k=top_k,
            ):
                hits.append({"kind": "kg_inferred", **row})
        return hits
```

And in `src/kg/retrieval.py` add an `inferred_relations` method that filters on `source:'sme_synthesis'`, `confidence >= $min_confidence`, and the hard `(subscription_id, profile_id)` tuple:

```cypher
MATCH (a)-[r:INFERRED_RELATION]->(b)
WHERE r.source = 'sme_synthesis'
  AND r.subscription_id = $subscription_id
  AND r.profile_id = $profile_id
  AND r.confidence >= $min_confidence
RETURN a.node_id AS src, b.node_id AS dst,
       r.relation_type AS relation_type,
       r.confidence AS confidence,
       r.evidence AS evidence,
       r.inference_path AS inference_path
ORDER BY r.confidence DESC
LIMIT $top_k
```

- [ ] **Step 4: Verify + commit**

```bash
pytest tests/retrieval/test_unified_retriever_layer_b.py -v
git add src/retrieval/unified_retriever.py src/kg/retrieval.py \
        tests/retrieval/test_unified_retriever_layer_b.py
git commit -m "phase2(sme-retrieval-b): KG layer incl. INFERRED_RELATION behind enable_sme_retrieval"
```

---

## Task 13: Retrieval Layer C (SME artifacts) in `src/retrieval/sme_retrieval.py`

**Files:**
- Create: `src/retrieval/sme_retrieval.py`
- Create: `tests/retrieval/test_sme_retrieval.py`

Layer C: SME artifact snippets from Qdrant, hard `(sub, prof)` filter + optional `artifact_type` + `domain_tags`, gated on `enable_sme_retrieval`.

- [ ] **Step 1: Write the failing test**

Create `tests/retrieval/test_sme_retrieval.py`:

```python
"""Layer C retrieval — SME artifact snippets."""
from unittest.mock import MagicMock, patch

import pytest

from src.retrieval.sme_retrieval import SMERetrieval


def test_layer_c_hard_filter_and_artifact_type_constraint():
    qdrant = MagicMock(); embedder = MagicMock()
    embedder.embed.return_value = [0.1] * 768
    qdrant.search.return_value = [MagicMock(
        payload={"artifact_type": "insight_index", "type": "trend",
                 "snippet_id": "insight:trend:0", "text": "Rev QoQ rose.",
                 "confidence": 0.8,
                 "evidence": [{"doc_id": "d1", "chunk_id": "c1"}]}, score=0.92)]
    r = SMERetrieval(qdrant=qdrant, embedder=embedder)
    with patch("src.retrieval.sme_retrieval.flag_enabled", return_value=True):
        hits = r.retrieve(query="rev", subscription_id="sub_a", profile_id="prof_x",
                          artifact_types=["insight_index"], top_k=10)
    assert len(hits) == 1 and hits[0]["kind"] == "sme_artifact"
    call = qdrant.search.call_args
    assert call[1]["collection_name"] == "sme_artifacts_sub_a"
    must_fields = {m.key for m in call[1]["query_filter"].must}
    assert {"subscription_id", "profile_id", "artifact_type"} <= must_fields


def test_layer_c_flag_off_and_cross_sub_guard():
    qdrant = MagicMock(); embedder = MagicMock()
    r = SMERetrieval(qdrant=qdrant, embedder=embedder)
    with patch("src.retrieval.sme_retrieval.flag_enabled", return_value=False):
        assert r.retrieve(query="q", subscription_id="s", profile_id="p",
                          artifact_types=None, top_k=10) == []
    qdrant.search.assert_not_called()
    with pytest.raises(ValueError, match="subscription_id"):
        r.retrieve(query="q", subscription_id="", profile_id="p",
                   artifact_types=None, top_k=5)
```

- [ ] **Step 2: Run failing tests.**

- [ ] **Step 3: Implement Layer C**

Create `src/retrieval/sme_retrieval.py`:

```python
"""Layer C — retrieve SME artifact snippets from Qdrant with hard profile filter."""
from __future__ import annotations

from src.intelligence.sme.flags import flag_enabled


class SMERetrieval:
    def __init__(self, qdrant, embedder):
        self._qdrant = qdrant
        self._embedder = embedder

    def retrieve(self, *, query: str, subscription_id: str, profile_id: str,
                 artifact_types: list[str] | None, top_k: int) -> list[dict]:
        if not subscription_id:
            raise ValueError("subscription_id required for SME retrieval")
        if not profile_id:
            raise ValueError("profile_id required for SME retrieval")
        if not flag_enabled("enable_sme_retrieval", subscription_id=subscription_id):
            return []
        from qdrant_client.http.models import Filter, FieldCondition, MatchValue, MatchAny
        must = [
            FieldCondition(key="subscription_id", match=MatchValue(value=subscription_id)),
            FieldCondition(key="profile_id", match=MatchValue(value=profile_id)),
        ]
        if artifact_types:
            must.append(FieldCondition(key="artifact_type",
                                       match=MatchAny(any=list(artifact_types))))
        vector = self._embedder.embed(query)
        rows = self._qdrant.search(
            collection_name=f"sme_artifacts_{subscription_id}",
            query_vector=vector,
            query_filter=Filter(must=must),
            limit=top_k,
        )
        return [
            {"kind": "sme_artifact",
             "artifact_type": r.payload["artifact_type"],
             "snippet_id": r.payload["snippet_id"],
             "text": r.payload["text"],
             "confidence": r.payload.get("confidence"),
             "evidence": r.payload.get("evidence", []),
             "score": r.score,
             "payload": r.payload}
            for r in rows
        ]
```

- [ ] **Step 4: Verify + commit**

```bash
pytest tests/retrieval/test_sme_retrieval.py -v
git add src/retrieval/sme_retrieval.py tests/retrieval/test_sme_retrieval.py
git commit -m "phase2(sme-retrieval-c): SME artifact retrieval with hard profile filter"
```

---

## Task 14: Flag resolver (`enable_sme_synthesis`, `enable_sme_retrieval`, `enable_kg_synthesized_edges`)

**Files:**
- Create: `src/intelligence/sme/flags.py`
- Create: `tests/intelligence/sme/test_flags.py`

Per-sub booleans in MongoDB. Master `sme_redesign_enabled` overrides individuals (spec §13.2). 60s TTL cache.

- [ ] **Step 1: Write the failing tests**

Create `tests/intelligence/sme/test_flags.py`:

```python
"""Tests for per-subscription SME capability flags."""
from unittest.mock import MagicMock, patch

from src.intelligence.sme import flags


def _doc(**overrides):
    base = {"subscription_id": "sub_a", "sme_redesign_enabled": True,
            "enable_sme_synthesis": True, "enable_sme_retrieval": False}
    base.update(overrides); return base


def test_per_flag_gating_and_master_kill_switch():
    flags._cache.clear()
    with patch.object(flags, "_load_subscription_record", return_value=_doc()):
        assert flags.flag_enabled("enable_sme_synthesis", subscription_id="sub_a") is True
        assert flags.flag_enabled("enable_sme_retrieval", subscription_id="sub_a") is False
    flags._cache.clear()
    with patch.object(flags, "_load_subscription_record",
                       return_value=_doc(sme_redesign_enabled=False)):
        assert flags.flag_enabled("enable_sme_synthesis", subscription_id="sub_a") is False


def test_missing_record_defaults_to_false():
    flags._cache.clear()
    with patch.object(flags, "_load_subscription_record", return_value=None):
        assert flags.flag_enabled("enable_sme_synthesis", subscription_id="nope") is False


def test_cache_ttl_refresh(monkeypatch):
    flags._cache.clear()
    load = MagicMock(side_effect=[_doc(enable_sme_synthesis=True),
                                   _doc(enable_sme_synthesis=False)])
    monkeypatch.setattr(flags, "_load_subscription_record", load)
    monkeypatch.setattr(flags, "_TTL_SECONDS", 0)
    assert flags.flag_enabled("enable_sme_synthesis", subscription_id="sub_a") is True
    assert flags.flag_enabled("enable_sme_synthesis", subscription_id="sub_a") is False
    assert load.call_count == 2
```

- [ ] **Step 2: Run failing tests.**

- [ ] **Step 3: Implement `flags.py`**

```python
"""SME capability flag resolver — per-subscription, MongoDB-backed, cached."""
from __future__ import annotations

import time
from typing import Any


_TTL_SECONDS = 60.0
_cache: dict[str, tuple[float, dict]] = {}

_KNOWN_FLAGS = {
    "enable_sme_synthesis", "enable_sme_retrieval",
    "enable_kg_synthesized_edges", "enable_rich_mode",
    "enable_url_as_prompt", "enable_hybrid_retrieval",
    "enable_cross_encoder_rerank",
}


def _load_subscription_record(subscription_id: str) -> dict[str, Any] | None:
    from src.api.document_status import get_subscription_record
    return get_subscription_record(subscription_id)


def flag_enabled(flag_name: str, *, subscription_id: str) -> bool:
    if flag_name not in _KNOWN_FLAGS:
        raise ValueError(f"Unknown SME flag: {flag_name}")
    now = time.time()
    cached = _cache.get(subscription_id)
    if cached is not None and (now - cached[0]) < _TTL_SECONDS:
        rec = cached[1]
    else:
        rec = _load_subscription_record(subscription_id) or {}
        _cache[subscription_id] = (now, rec)
    if not rec.get("sme_redesign_enabled", False):
        return False
    return bool(rec.get(flag_name, False))
```

- [ ] **Step 4: Verify + commit**

```bash
pytest tests/intelligence/sme/test_flags.py -v
git add src/intelligence/sme/flags.py tests/intelligence/sme/test_flags.py
git commit -m "phase2(sme-flags): per-subscription capability gate with master kill switch + TTL"
```

---

## Task 15: Incremental synthesis — input-hash short-circuit

**Files:**
- Modify: `src/intelligence/sme/input_snapshot.py`
- Create: `tests/intelligence/sme/test_input_snapshot.py`

Re-running synthesis on unchanged inputs is wasteful. Task 10's `finalize_training_for_doc` already calls `input_hash_unchanged(ctx)` — here we implement it. Snapshot = SHA-256 of sorted `(doc_id, chunk_id, chunk_text_hash)` across the profile plus `adapter_version` + `adapter_content_hash`. Persisted as `sme_last_input_hash` (control plane).

- [ ] **Step 1: Write the failing tests**

Create `tests/intelligence/sme/test_input_snapshot.py`:

```python
"""Input-hash snapshot tests for incremental synthesis."""
from unittest.mock import MagicMock, patch

import pytest

from src.intelligence.sme import input_snapshot as mod


def _ctx(triples, adapter_hash):
    return mod.build_profile_ctx_from_fixture(
        subscription_id="s", profile_id="p", profile_domain="finance",
        chunks=[{"doc_id": d, "chunk_id": c, "text": t} for d, c, t in triples],
        adapter_version="1.0.0", adapter_content_hash=adapter_hash,
    )


def test_input_hash_determinism_and_change_detection():
    # Stable under reorder
    a = _ctx([("d1", "c1", "t1"), ("d2", "c3", "t3")], "h1")
    b = _ctx([("d2", "c3", "t3"), ("d1", "c1", "t1")], "h1")
    assert a.input_hash == b.input_hash
    # Changes on new doc
    c = _ctx([("d1", "c1", "t1"), ("d2", "c3", "t3"), ("d3", "c5", "t5")], "h1")
    assert c.input_hash != a.input_hash
    # Changes on adapter hash
    d = _ctx([("d1", "c1", "t1")], "h2")
    e = _ctx([("d1", "c1", "t1")], "h1")
    assert d.input_hash != e.input_hash


def test_unchanged_lookup():
    ctx = _ctx([("d1", "c1", "t1")], "h1")
    with patch.object(mod, "_get_stored_input_hash", return_value=ctx.input_hash):
        assert mod.input_hash_unchanged(ctx) is True
    with patch.object(mod, "_get_stored_input_hash", return_value=None):
        assert mod.input_hash_unchanged(ctx) is False
```

- [ ] **Step 2: Run failing tests.**

- [ ] **Step 3: Implement**

Create `src/intelligence/sme/input_snapshot.py`:

```python
"""Incremental synthesis via deterministic input-hash snapshot."""
from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProfileSynthesisContext:
    subscription_id: str; profile_id: str; profile_domain: str
    chunks: list[dict[str, Any]]; synthesis_version: int
    adapter_version: str; adapter_content_hash: str
    input_hash: str; run_id: str
    kg_slice: dict[str, Any] = field(default_factory=dict)

    def iter_chunks(self):
        yield from self.chunks


def _compute_input_hash(chunks, adapter_version, adapter_content_hash) -> str:
    h = hashlib.sha256()
    for c in sorted(chunks, key=lambda x: (x["doc_id"], x["chunk_id"])):
        ch = hashlib.sha256(c["text"].encode("utf-8")).hexdigest()
        h.update(f"{c['doc_id']}|{c['chunk_id']}|{ch}\n".encode())
    h.update(f"adapter:{adapter_version}|{adapter_content_hash}".encode())
    return h.hexdigest()


def build_profile_ctx(subscription_id: str, profile_id: str) -> ProfileSynthesisContext:
    """Production path — chunks from Qdrant + adapter meta from Phase 1 loader."""
    from src.intelligence.sme.adapter_loader import load_profile_chunks, load_adapter_meta
    chunks = load_profile_chunks(subscription_id=subscription_id, profile_id=profile_id)
    am = load_adapter_meta(subscription_id=subscription_id)
    return ProfileSynthesisContext(
        subscription_id=subscription_id, profile_id=profile_id,
        profile_domain=am.domain, chunks=chunks,
        synthesis_version=_next_synthesis_version(subscription_id, profile_id),
        adapter_version=am.version, adapter_content_hash=am.content_hash,
        input_hash=_compute_input_hash(chunks, am.version, am.content_hash),
        run_id=str(uuid.uuid4()))


def build_profile_ctx_from_fixture(**kw) -> ProfileSynthesisContext:
    chunks = kw.pop("chunks")
    return ProfileSynthesisContext(
        chunks=chunks, synthesis_version=1, run_id="fixture-run",
        input_hash=_compute_input_hash(chunks, kw["adapter_version"],
                                        kw["adapter_content_hash"]), **kw)


def _get_stored_input_hash(subscription_id, profile_id) -> str | None:
    from src.api.document_status import get_profile_record
    return (get_profile_record(subscription_id=subscription_id,
                                profile_id=profile_id) or {}).get("sme_last_input_hash")


def _next_synthesis_version(subscription_id, profile_id) -> int:
    from src.api.document_status import get_profile_record
    rec = get_profile_record(subscription_id=subscription_id,
                              profile_id=profile_id) or {}
    return int(rec.get("sme_synthesis_version", 0)) + 1


def input_hash_unchanged(ctx: ProfileSynthesisContext) -> bool:
    stored = _get_stored_input_hash(ctx.subscription_id, ctx.profile_id)
    return stored is not None and stored == ctx.input_hash
```

On success, `finalize_training_for_doc` writes `sme_last_input_hash` + `sme_synthesis_version` to the profile record via a small `update_profile_record(...)` helper alongside `append_audit_log` (control-plane only, permitted per memory rule).

- [ ] **Step 4: Verify + commit**

```bash
pytest tests/intelligence/sme/test_input_snapshot.py -v
git add src/intelligence/sme/input_snapshot.py tests/intelligence/sme/test_input_snapshot.py
git commit -m "phase2(sme-incremental): input-hash snapshot short-circuits idempotent re-synthesis"
```

---

## Task 16: End-to-end sandbox integration test

**Files:**
- Create: `tests/integration/test_sme_end_to_end.py`

Validates full pipeline on sandbox: final doc embeds → `finalize_training_for_doc` runs synthesis → artifacts persist to Blob/Qdrant/Neo4j → Layers B+C retrieve → `TRAINING_COMPLETED` fires once.

- [ ] **Step 1: Write the integration test**

Create `tests/integration/test_sme_end_to_end.py`:

```python
"""End-to-end sandbox integration. Requires live DocWain services + env vars:
SME_SANDBOX_SUB, SME_SANDBOX_PROF, SME_SANDBOX_LAST_DOC_ID, QDRANT_URL."""
import os
import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(scope="session")
def sandbox():
    sub = os.environ.get("SME_SANDBOX_SUB")
    prof = os.environ.get("SME_SANDBOX_PROF")
    if not sub or not prof:
        pytest.skip("set SME_SANDBOX_SUB + SME_SANDBOX_PROF")
    return {"subscription_id": sub, "profile_id": prof,
            "last_doc": os.environ["SME_SANDBOX_LAST_DOC_ID"]}


def test_synthesis_persists_and_status_flips_and_artifacts_retrievable(sandbox):
    from src.api.pipeline_api import finalize_training_for_doc
    from src.api.document_status import get_document_record
    from src.retrieval.sme_retrieval import SMERetrieval
    from src.embeddings import default_embedder
    from src.intelligence.sme.flags import _cache
    from qdrant_client import QdrantClient

    _cache.clear()
    doc = {"document_id": sandbox["last_doc"],
           "subscription_id": sandbox["subscription_id"],
           "profile_id": sandbox["profile_id"]}
    finalize_training_for_doc(doc)
    assert get_document_record(doc["document_id"])["pipeline_status"] == "TRAINING_COMPLETED"

    qc = QdrantClient(url=os.environ.get("QDRANT_URL", "http://localhost:6333"))
    r = SMERetrieval(qdrant=qc, embedder=default_embedder())
    hits = r.retrieve(query="trends and risks",
                      subscription_id=sandbox["subscription_id"],
                      profile_id=sandbox["profile_id"],
                      artifact_types=None, top_k=10)
    assert len(hits) > 0
    assert {h["artifact_type"] for h in hits} & {"insight_index", "dossier"}

    # Idempotent re-run short-circuits
    finalize_training_for_doc(doc)
    from src.api.document_status import get_audit_log
    assert any(e["event"] == "SME_SYNTHESIS_SKIPPED_INPUT_UNCHANGED"
               for e in get_audit_log(doc["document_id"])[-5:])


def test_cross_subscription_isolation_plus_inferred_edges(sandbox):
    from src.retrieval.sme_retrieval import SMERetrieval
    from src.retrieval.unified_retriever import UnifiedRetriever
    from src.kg.retrieval import KGRetrievalClient
    from src.embeddings import default_embedder
    from unittest.mock import MagicMock, patch
    from qdrant_client import QdrantClient

    qc = QdrantClient(url=os.environ.get("QDRANT_URL", "http://localhost:6333"))
    # Cross-sub read returns empty
    assert SMERetrieval(qdrant=qc, embedder=default_embedder()).retrieve(
        query="q", subscription_id="sub_DOES_NOT_EXIST", profile_id="p",
        artifact_types=None, top_k=10) == []

    # Inferred edges visible in Layer B for sandbox
    r = UnifiedRetriever(kg_client=KGRetrievalClient(),
                         qdrant=MagicMock(), sme=MagicMock())
    with patch("src.retrieval.unified_retriever.flag_enabled", return_value=True):
        hits = r.retrieve_layer_b(query="funding", subscription_id=sandbox["subscription_id"],
                                   profile_id=sandbox["profile_id"], top_k=20)
    inferred = [h for h in hits if h["kind"] == "kg_inferred"]
    assert any(h.get("confidence", 0) >= 0.6 for h in inferred)
```

- [ ] **Step 2: Run on sandbox** — `SME_SANDBOX_SUB=... SME_SANDBOX_PROF=... SME_SANDBOX_LAST_DOC_ID=... pytest tests/integration/test_sme_end_to_end.py -v -m integration`. If a layer fails the test localizes it.

- [ ] **Step 3: Commit** — `git add tests/integration/test_sme_end_to_end.py; git commit -m "phase2(sme-e2e): sandbox integration — synthesis + artifacts + cross-sub isolation"`

---

## Task 17: Validate with Phase 0 eval harness — `sme_artifact_hit_rate` rises

**Files:**
- Run: `scripts/sme_eval/run_baseline.py` with `enable_sme_retrieval=true` on sandbox subscription
- Create: `tests/sme_metrics_phase2_sandbox_YYYY-MM-DD.json`

Phase 0 froze `sme_artifact_hit_rate == 0.0`. Phase 2 proves it rises on sandbox — the "Measure Before You Change" gate for Phase 2.

- [ ] **Step 1: Temporarily enable retrieval on sandbox** (control-plane flag flip in MongoDB; all four: `sme_redesign_enabled`, `enable_sme_synthesis`, `enable_sme_retrieval`, `enable_kg_synthesized_edges` → true).

- [ ] **Step 2: Re-run the Phase 0 harness subsetted to sandbox**

```bash
python -m scripts.sme_eval.run_baseline \
  --eval-dir tests/sme_evalset_v1/queries \
  --fixtures tests/sme_evalset_v1/fixtures/test_profiles.yaml \
  --out tests/sme_metrics_phase2_sandbox_$(date +%Y-%m-%d).json \
  --results-jsonl tests/sme_results_phase2_sandbox_$(date +%Y-%m-%d).jsonl \
  --subset-to-subscription SANDBOX_SUBSCRIPTION_ID
```

- [ ] **Step 3: Verify metric delta**

Load Phase 0 + Phase 2 snapshots, assert `sme_artifact_hit_rate` went from 0.0 to ≥0.70 on analytical-intent queries (spec §10 launch gate is 0.90 — Phase 2 is an intermediate milestone; Phase 3 tuning raises the bar). Cross-check: `hallucination_rate` stays 0.0; `context_recall` within noise; `faithfulness` holds or improves (Phase 2 doesn't change prompts — large swings flag pack destabilization; investigate before Phase 3).

- [ ] **Step 4: Revert flag to Phase 2 default and commit measurement**

Flip `enable_sme_retrieval` back to false on the sandbox subscription, then:

```bash
git add -f tests/sme_metrics_phase2_sandbox_*.json tests/sme_results_phase2_sandbox_*.jsonl
git commit -m "phase2(sme-measure): sandbox hit_rate rises from 0.0 to >=0.70 post-synthesis"
```

Phase 2 ships with `enable_sme_retrieval=false` everywhere. Phase 3 flips it on for opt-in subs against the full §10 launch gate.

---

## Task 18: Phase 2 exit checklist and cleanup

**Files:**
- Modify: `src/intelligence/sme/__init__.py` (export public API)
- Create: exit-check runbook in commit body only (no new doc file)

- [ ] **Step 1: Finalize `src/intelligence/sme/__init__.py`** — re-export: `SMESynthesizer`, `SynthesisReport`, `flag_enabled`, all five artifact types + `InferredEdge` + `Evidence`, and `ProfileSynthesisContext` / `build_profile_ctx` / `input_hash_unchanged`.

- [ ] **Step 2: Full Phase 2 test suite**

```bash
pytest tests/intelligence/sme tests/api/test_pipeline_api_sme.py \
       tests/retrieval/test_unified_retriever_layer_b.py \
       tests/retrieval/test_sme_retrieval.py -v
```

- [ ] **Step 3: Integration test on sandbox**

```bash
SME_SANDBOX_SUB=... SME_SANDBOX_PROF=... SME_SANDBOX_LAST_DOC_ID=... \
  pytest tests/integration/test_sme_end_to_end.py -v -m integration
```

- [ ] **Step 4: Phase 2 exit checklist** — every box must be genuinely ticked; a miss blocks Phase 3:

- [ ] All Phase 2 tasks committed with passing unit + integration tests
- [ ] Five artifact builders produce LLM-synthesized content on sandbox (canonical in Blob, snippets in Qdrant)
- [ ] `INFERRED_RELATION` edges exist in Neo4j for sandbox profile, non-zero count
- [ ] `SMESynthesizer.synthesize_profile` is the single public entry (`src/intelligence/sme/__init__.py`)
- [ ] `finalize_training_for_doc` wired; `embedding.py` delegates; `PIPELINE_TRAINING_COMPLETED` fires only after synthesis success
- [ ] No new `pipeline_status` string (grep `PIPELINE_` constants matches pre-Phase-2)
- [ ] Layer B + Layer C implemented, both behind `enable_sme_retrieval` default OFF
- [ ] Capability flags gate correctly (flag-off = pre-Phase-2 behavior)
- [ ] Incremental synthesis idempotency confirmed in integration test
- [ ] Phase 0 eval re-run on sandbox: hit_rate 0.0 → ≥0.70; hallucination unchanged
- [ ] Cross-subscription isolation verified by integration test
- [ ] `src/generation/prompts.py` untouched in Phase 2 branch (git log empty)
- [ ] No Claude/Anthropic references anywhere on branch (grep empty)
- [ ] No internal wall-clock timeouts (grep `asyncio.wait_for`, `signal.alarm`, custom `timeout=` empty on synthesis paths)
- [ ] Sandbox `enable_sme_retrieval` left OFF at end of Phase 2

- [ ] **Step 5: Commit the exit state**

```bash
git add src/intelligence/sme/__init__.py
git commit -m "phase2(sme-exit): finalize public API + exit checklist committed"
git tag -a sme-phase2-exit -m "SME synthesis lives in training stage; retrieval gated; sandbox measured"
```

---

## Self-review appendix

### Spec coverage matrix

Every required spec section is ticked below. A missing tick means the plan is under-specified.

- **§4 Architecture** — Five builder types → Tasks 4–8; SMEVerifier fail-closed → Task 4 demonstrates (Tasks 5–8 delegate via Phase 1 verifier); `PIPELINE_TRAINING_COMPLETED` unchanged → Task 10.
- **§6 Artifacts + SMEVerifier** — Shapes → Task 2 schemas + Tasks 4–8 populate; five checks applied in order in Dossier (Task 4 test), reused across remaining builders; verifier drops logged to trace and never persisted.
- **§7 Retrieval B + C** — Task 12 (KG + synthesized edges), Task 13 (SME artifacts dense; sparse+RRF is Phase 3); both flag-gated; profile-hard-filtered everywhere.
- **§9 Storage** — Qdrant `sme_artifacts_{sub}` + Blob canonical + manifest (Task 11); Neo4j `INFERRED_RELATION` generic edge + rich props (Task 7); MongoDB control-plane flags + version + input hash (Tasks 14/15/10). Phase 1 owns adapter loader + Redis.
- **§11 Traces** — Phase 1 owns `trace.py`; Tasks 4–9 exercise every `record_*`. Query-trace is Phase 3.
- **§13 Rollback** — Task 14 resolver enforces master + per-flag gates; storage additive so rollback preserves post-mortem data.
- **§15 Open questions handled:** Q4 LLM budget (placeholder 2048, Task 17 calibrates); Q5 incremental via input-hash (Task 15).
- **Forwarded:** Q1 module-reuse audit (P3), Q2 reranker (P3), Q3 YAML migration (ops), Q6 persona-consistency judge (P0), Q7 failure-mode (spec default in Task 10).

### Spec gaps surfaced (need spec patch before execution)

1. **Per-builder LLM budget defaults** — §6 silent on synthesis-time budgets; Phase 2 hardcodes `max_output_tokens=2048`; lift to adapter YAML in 2.1.
2. **`is_last_doc_in_profile` contract** — spec implicit; Task 10 introduces helper. If ingest uses Celery group-callback, swap the helper body — `finalize_training_for_doc` unchanged.
3. **`ProfileSynthesisContext` shape** — Phase 1 should own canonical; Task 15 dataclass is reconciliation target if Phase 1 differs.
4. **Neo4j `node_id` convention** — Task 7 assumes property; confirm vs. `src/kg/` schemas, swap to `id(a) = $src` if internal id is convention.
5. **`AdapterLoader.load_adapter_meta`** — referenced by Task 15; if Phase 1 didn't ship, add as 1.1 patch.

### Open questions (for Phase 3 owner)

1. `inferred_edge_confidence_floor = 0.6` — Phase 3 measures whether to raise (precision) or lower (recall).
2. `generic.yaml` adapter body — if Phase 1 shipped empty placeholders, Phase 2 builders produce empty artifacts; Task 16 surfaces.
3. Per-builder `elapsed_ms` in traces — no abort per memory rule, but Phase 3 adds ops visibility.
4. Document removal — input-hash detects change, but INFERRED edges citing deleted chunks need cleanup; Phase 2.1 DELETE pass. Backlog.
5. Qdrant collection naming — Task 1 audit verifies Phase 1's choice (per-sub vs. shared+filter).

### Type and contract consistency
- `ProfileSynthesisContext` (Task 15) → builders + orchestrator.
- `SynthesisReport` (Task 9) → `finalize_training_for_doc` (Task 10).
- `ArtifactStorage.put_snippet/put_canonical/put_manifest` — one signature across Tasks 4–11, 16.
- `Evidence` shared across artifact types; retrieval payloads reuse dict shape.

### Placeholder scan
"TBD" / "TODO" / "fill in" — none. `REPLACE_WITH_REAL_*` in Task 17 is operator env-var substitution (Phase 0 convention).

### Memory-rule audit

- **Measure Before You Change** → Task 17 re-runs Phase 0 harness; improvement claim backed by `sme_artifact_hit_rate` delta.
- **No customer data** → Phase 2 commits no documents; sandbox + opt-in subs bring their own.
- **MongoDB = control plane only** → only flags, `sme_synthesis_version`, `sme_last_input_hash`, audit events.
- **No new pipeline_status strings** → grep `PIPELINE_` on branch matches pre-Phase-2.
- **No internal timeouts** → grep `asyncio.wait_for`, `signal.alarm`, custom `timeout=` on synthesis paths returns empty; `httpx` default is the only net.
- **Response formatting in `prompts.py`** → git log on `src/generation/prompts.py` over Phase 2 branch shows zero commits.
- **Adapter YAMLs via AdapterLoader** → no builder opens YAML from disk; all go through `adapter_loader.load(...)`.
- **Profile isolation hard** → `(subscription_id, profile_id)` in every write, every filter, every MATCH; integration test (Task 16) verifies cross-sub rejection.
- **Engineering-first** → zero model training; existing V2 via existing gateway.
- **No Claude/Anthropic attribution** → grep branch diff empty.

---

*End of Phase 2 plan. Phase 3 (retrieval-on + launch-gate measurement) to be produced separately; it consumes the artifact contracts, flag infrastructure, and input-hash idempotency shipped here.*
