# DocWain SME Phase 1 — Infrastructure, Dark-Launched

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the full infrastructure for SME synthesis and hybrid retrieval without changing query-time behavior for any real subscription. Every piece is wired, validated, trace-instrumented, and per-subscription flag-gated off by default. Artifact builders ship as skeletons (real classes, stub bodies); verifier, trace writers, storage adapter, and adapter loader are live. Hybrid retrieval + cross-encoder reranker are in place but gated. A sandbox subscription runs an end-to-end ingest with empty artifacts to prove the wiring holds.

**Architecture:** New package `src/intelligence/sme/` owns adapter loading (Blob + TTL cache), verifier, trace writers, artifact storage adapter, and the synthesizer orchestrator skeleton with five builder skeletons. `src/retrieval/` grows a hybrid dense+sparse fusion helper and a cross-encoder reranker. Admin endpoints under `src/api/sme_admin_api.py` handle adapter CRUD + invalidation. Feature flags live in `src/config/feature_flags.py` with per-subscription overrides in MongoDB. YAML defaults + prompt placeholders ship under `deploy/sme_adapters/defaults/`; last-resort embedded fallback at `deploy/sme_adapters/last_resort/generic.yaml`. A one-time `scripts/reindex_qdrant_sparse.py` adds sparse vectors to existing chunk collections. Nothing under `src/agent/`, `src/generation/prompts.py`, `src/intelligence/generator.py`, or `src/execution/router.py` is touched.

**Tech Stack:** Python 3.12, `pydantic` v2, `pyyaml`, `azure-storage-blob` (existing), `qdrant-client` (existing), `neo4j` (existing), `redis` (existing), `sentence-transformers` cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`, resolving spec §15 Q2), `fastapi`, `pytest` + `pytest-asyncio`.

**Related spec:** `docs/superpowers/specs/2026-04-20-docwain-profile-sme-reasoning-design.md` — §2 (scope), §3 (invariants), §5 (adapters), §6 (artifacts + verifier), §9 (storage), §13 (rollback/flags), §15 (open questions).

**Memory rules that constrain this plan:**
- Storage separation: MongoDB = control plane only. YAMLs, prompt templates, artifact canonical text, traces → Blob. Artifact snippets → Qdrant. Synthesized edges → Neo4j.
- No Claude/Anthropic references anywhere.
- No internal timeouts; only Blob fetch has a per-operation safety limit.
- MongoDB status values immutable — no new `pipeline_status`. SME synthesis is internal to the training stage.
- Response formatting stays in `src/generation/prompts.py`; Phase 1 does NOT touch it. `src/intelligence/generator.py` untouched.
- Intelligence precomputed at ingestion — Phase 1 wires the ingestion-time persistence path, gated off until Phase 2.
- V5 failure lessons — skeleton builders with live verifier; Phase 2 lands on tested infrastructure.

---

## File structure

```
src/intelligence/sme/                           [CREATE]
├── __init__.py, adapter_schema.py, adapter_loader.py, verifier.py
├── trace.py, storage.py, synthesizer.py, artifact_models.py
└── builders/__init__.py, _base.py, dossier.py, insight_index.py,
              comparative_register.py, kg_materializer.py, recommendation_bank.py

src/retrieval/hybrid_search.py, reranker.py               [ADD FILES]
src/config/__init__.py, feature_flags.py                  [CREATE]
src/api/sme_admin_api.py                                  [CREATE]
src/main.py                                               [MODIFY — one include_router]
scripts/reindex_qdrant_sparse.py                          [CREATE]

deploy/sme_adapters/
├── defaults/{generic,finance,legal,hr,medical,it_support}.yaml
├── defaults/prompts/*.md                                 (placeholder templates)
└── last_resort/generic.yaml                              (embedded emergency fallback)

tests/intelligence/sme/                                   [CREATE]
├── conftest.py
├── test_adapter_schema.py, test_adapter_loader.py, test_verifier.py
├── test_trace.py, test_storage.py, test_synthesizer.py
├── test_artifact_models.py, test_sandbox_integration.py
└── builders/test_{dossier,insight_index,comparative_register,
                    kg_materializer,recommendation_bank}.py

tests/retrieval/test_{hybrid_search,reranker}.py          [CREATE]
tests/config/test_feature_flags.py                        [CREATE]
tests/api/test_sme_admin_api.py                           [CREATE]
tests/scripts/test_reindex_qdrant_sparse.py               [CREATE]
```

Each module has one responsibility. `adapter_loader.py` only loads. `verifier.py` only verifies. `storage.py` only persists. `synthesizer.py` only orchestrates. Phase 2 replaces builder bodies with no other file touched; full rollback per spec §13.3 is a flag flip, no code-path deletion.

---

## Task 1: Scaffold directories + init files

**Files:** create `src/intelligence/sme/__init__.py`, `src/intelligence/sme/builders/__init__.py`, `src/config/__init__.py`, matching test package inits, `deploy/sme_adapters/{defaults/prompts,last_resort}/`.

- [ ] **Step 1: Verify layout**

`ls src/intelligence/ src/retrieval/ src/api/ && test -f src/main.py && echo "main exists"` — expect directories + `main exists`.

- [ ] **Step 2: Create directories and init files**

```bash
mkdir -p src/intelligence/sme/builders src/config \
    tests/intelligence/sme/builders tests/config \
    deploy/sme_adapters/defaults/prompts deploy/sme_adapters/last_resort
touch src/intelligence/sme/__init__.py src/intelligence/sme/builders/__init__.py \
      src/config/__init__.py tests/intelligence/sme/__init__.py \
      tests/intelligence/sme/builders/__init__.py tests/config/__init__.py
```

- [ ] **Step 3: Write `src/intelligence/sme/__init__.py`**

```python
"""DocWain SME (Subject Matter Expert) synthesis package.

Ingestion-time synthesis of domain-aware artifacts, per spec
docs/superpowers/specs/2026-04-20-docwain-profile-sme-reasoning-design.md.
"""
```

- [ ] **Step 4: Commit**

```bash
git add -f src/intelligence/sme/__init__.py src/intelligence/sme/builders/__init__.py \
           src/config/__init__.py tests/intelligence/sme/__init__.py \
           tests/intelligence/sme/builders/__init__.py tests/config/__init__.py \
           deploy/sme_adapters/defaults/prompts deploy/sme_adapters/last_resort
git commit -m "phase1(sme-scaffold): package directories and init files"
```

---

## Task 2: Adapter YAML Pydantic schema

**Files:** create `src/intelligence/sme/adapter_schema.py`, `tests/intelligence/sme/test_adapter_schema.py`.

The adapter YAML is load-bearing (it drives every builder, persona, retrieval cap), so the full schema is shown. Extra fields are rejected so a typo can't silently corrupt synthesis.

- [ ] **Step 1: Write failing tests**

Create `tests/intelligence/sme/test_adapter_schema.py`:

```python
"""Tests for the adapter YAML Pydantic schema."""
import pytest
from pydantic import ValidationError
from src.intelligence.sme.adapter_schema import Adapter


def _valid():
    return {
        "domain": "finance", "version": "1.0.0",
        "persona": {"role": "analyst", "voice": "direct", "grounding_rules": []},
        "dossier": {"section_weights": {"a": 0.4, "b": 0.3, "c": 0.3},
                    "prompt_template": "prompts/finance_dossier.md"},
        "insight_detectors": [{"type": "trend", "rule": "qoq_gt", "params": {"t": 0.05}}],
        "comparison_axes": [{"name": "period", "dimension": "temporal"}],
        "kg_inference_rules": [{"pattern": "(a)-[:X]->(b)", "produces": "r",
                                 "confidence_floor": 0.6, "max_hops": 3}],
        "recommendation_frames": [{"frame": "f", "template": "t",
                                    "requires": {"insight_types": ["trend"]}}],
        "response_persona_prompts": {"diagnose": "p/d.md", "analyze": "p/a.md",
                                      "recommend": "p/r.md"},
        "retrieval_caps": {"max_pack_tokens": {"analyze": 6000, "diagnose": 5000,
                                                "recommend": 4500, "investigate": 8000}},
        "output_caps": {"analyze": 1200, "diagnose": 1500, "recommend": 1000,
                        "investigate": 2000}}


def test_valid_adapter():
    a = Adapter(**_valid())
    assert a.domain == "finance" and a.version == "1.0.0"

def test_rejects_unknown_field():
    d = _valid(); d["bogus"] = 1
    with pytest.raises(ValidationError): Adapter(**d)

def test_section_weights_must_sum_to_one():
    d = _valid(); d["dossier"]["section_weights"] = {"a": 0.9, "b": 0.5}
    with pytest.raises(ValidationError, match="sum to 1.0"): Adapter(**d)

def test_max_hops_bounded():
    d = _valid(); d["kg_inference_rules"][0]["max_hops"] = 10
    with pytest.raises(ValidationError): Adapter(**d)

def test_version_semver():
    d = _valid(); d["version"] = "not-a-semver"
    with pytest.raises(ValidationError, match="semver"): Adapter(**d)

def test_generic_minimal_adapter():
    d = _valid(); d["domain"] = "generic"
    for k in ("insight_detectors", "comparison_axes",
              "kg_inference_rules", "recommendation_frames"): d[k] = []
    assert Adapter(**d).domain == "generic"
```

- [ ] **Step 2: Confirm fails**

`pytest tests/intelligence/sme/test_adapter_schema.py -v` → ModuleNotFoundError.

- [ ] **Step 3: Write the schema (full — load-bearing)**

Create `src/intelligence/sme/adapter_schema.py`:

```python
"""Pydantic schema for SME adapter YAMLs (spec §5).

Strict: extra fields rejected; semver enforced; dossier weights must sum to 1.0.
"""
from __future__ import annotations
import re
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Persona(_Strict):
    role: str
    voice: str
    grounding_rules: list[str] = Field(default_factory=list)


class DossierConfig(_Strict):
    section_weights: dict[str, float]
    prompt_template: str

    @model_validator(mode="after")
    def _weights_sum_one(self) -> "DossierConfig":
        total = sum(self.section_weights.values())
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"dossier.section_weights must sum to 1.0 (got {total:.3f})")
        return self


class InsightDetector(_Strict):
    type: Literal["trend", "anomaly", "gap", "risk", "opportunity", "conflict"]
    rule: str
    params: dict[str, Any] = Field(default_factory=dict)


class ComparisonAxis(_Strict):
    name: str
    dimension: str
    unit: str | None = None


class KGInferenceRule(_Strict):
    pattern: str
    produces: str
    confidence_floor: float = Field(ge=0.0, le=1.0)
    max_hops: int = Field(ge=2, le=5)


class RecommendationFrame(_Strict):
    frame: str
    template: str
    requires: dict[str, list[str]] = Field(default_factory=dict)


class ResponsePersonaPrompts(_Strict):
    diagnose: str
    analyze: str
    recommend: str


class RetrievalCaps(_Strict):
    max_pack_tokens: dict[str, int]


class OutputCaps(_Strict):
    analyze: int | None = None
    diagnose: int | None = None
    recommend: int | None = None
    investigate: int | None = None
    summarize: int | None = None
    compare: int | None = None


class Adapter(_Strict):
    domain: str
    version: str
    persona: Persona
    dossier: DossierConfig
    insight_detectors: list[InsightDetector] = Field(default_factory=list)
    comparison_axes: list[ComparisonAxis] = Field(default_factory=list)
    kg_inference_rules: list[KGInferenceRule] = Field(default_factory=list)
    recommendation_frames: list[RecommendationFrame] = Field(default_factory=list)
    response_persona_prompts: ResponsePersonaPrompts
    retrieval_caps: RetrievalCaps
    output_caps: OutputCaps
    # Runtime-injected by AdapterLoader after parsing; not part of the YAML.
    # Exposed so callers (Phase 2+ synthesizer, Phase 4 recommendation grounding)
    # can read them directly off the Adapter instance. See ERRATA §1.
    content_hash: str | None = None
    source_path: str | None = None

    @field_validator("version")
    @classmethod
    def _semver(cls, v: str) -> str:
        if not _SEMVER_RE.match(v):
            raise ValueError(f"version must be semver (got {v!r})")
        return v

    @field_validator("domain")
    @classmethod
    def _slug(cls, v: str) -> str:
        if not re.match(r"^[a-z][a-z0-9_]{1,31}$", v):
            raise ValueError(f"domain must be lowercase slug (got {v!r})")
        return v
```

- [ ] **Step 4: Run green**

`pytest tests/intelligence/sme/test_adapter_schema.py -v` → 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/intelligence/sme/adapter_schema.py tests/intelligence/sme/test_adapter_schema.py
git commit -m "phase1(sme-adapter): YAML schema with strict pydantic validation"
```

---

## Task 3: AdapterLoader — Blob fetch + TTL cache + embedded fallback

**Files:** create `src/intelligence/sme/adapter_loader.py`, `tests/intelligence/sme/test_adapter_loader.py`.

Loader is load-bearing: every synthesis + query path flows through it. Full fetch logic shown.

- [ ] **Step 1: Write failing tests**

Create `tests/intelligence/sme/test_adapter_loader.py`:

```python
"""Tests for AdapterLoader: Blob fetch, TTL cache, last-resort fallback."""
import time
from pathlib import Path
from unittest.mock import MagicMock
import pytest
from src.intelligence.sme.adapter_loader import (
    AdapterLoader, AdapterLoadError, BlobReader)
from src.intelligence.sme.adapter_schema import Adapter

_GENERIC = """
domain: generic
version: 1.0.0
persona: {role: smex, voice: neutral, grounding_rules: [cite sources]}
dossier: {section_weights: {overview: 0.5, findings: 0.5}, prompt_template: p/g.md}
insight_detectors: []
comparison_axes: []
kg_inference_rules: []
recommendation_frames: []
response_persona_prompts: {diagnose: p/d.md, analyze: p/a.md, recommend: p/r.md}
retrieval_caps: {max_pack_tokens: {analyze: 6000, diagnose: 5000, recommend: 4500, investigate: 8000}}
output_caps: {analyze: 1200, diagnose: 1500, recommend: 1000, investigate: 2000}
"""


@pytest.fixture
def blob():
    b = MagicMock(spec=BlobReader); b.read_text.return_value = _GENERIC; return b


@pytest.fixture
def lr(tmp_path: Path) -> Path:
    p = tmp_path / "generic.yaml"; p.write_text(_GENERIC); return p


def _L(blob, lr, ttl=60):
    return AdapterLoader(blob=blob, last_resort_path=lr, ttl_seconds=ttl)


def test_resolves_subscription_override_first(blob, lr):
    _L(blob, lr).load("sub_a", "finance")
    assert blob.read_text.call_args_list[0][0][0] == \
        "sme_adapters/subscription/sub_a/finance.yaml"


def test_falls_through_to_global(blob, lr):
    def side(p):
        if p.startswith("sme_adapters/subscription/"): raise FileNotFoundError(p)
        return _GENERIC
    blob.read_text.side_effect = side
    assert isinstance(_L(blob, lr).load("sub_a", "finance"), Adapter)


def test_falls_through_to_generic(blob, lr):
    def side(p):
        if "mystery" in p: raise FileNotFoundError(p)
        return _GENERIC
    blob.read_text.side_effect = side
    assert _L(blob, lr).load("sub_a", "mystery").domain == "generic"


def test_ttl_cache_hits(blob, lr):
    l = _L(blob, lr, ttl=60); l.load("sub_a", "finance"); l.load("sub_a", "finance")
    assert blob.read_text.call_count == 1


def test_ttl_cache_expires(blob, lr):
    l = _L(blob, lr, ttl=0.01); l.load("sub_a", "finance")
    time.sleep(0.05); l.load("sub_a", "finance")
    assert blob.read_text.call_count == 2


def test_invalidate_forces_refetch(blob, lr):
    l = _L(blob, lr, ttl=3600); l.load("sub_a", "finance")
    l.invalidate("sub_a", "finance"); l.load("sub_a", "finance")
    assert blob.read_text.call_count == 2


def test_blob_unreachable_uses_last_resort(lr):
    bad = MagicMock(spec=BlobReader)
    bad.read_text.side_effect = ConnectionError("blob down")
    l = AdapterLoader(blob=bad, last_resort_path=lr, ttl_seconds=60)
    assert l.load("sub_a", "finance").domain == "generic"
    assert l.health_status() == "degraded"


def test_records_version_and_hash(blob, lr):
    l = _L(blob, lr); l.load("sub_a", "finance")
    meta = l.last_load_metadata("sub_a", "finance")
    assert meta["version"] == "1.0.0" and meta["content_hash"]


def test_missing_last_resort_raises(blob, tmp_path):
    with pytest.raises(AdapterLoadError):
        AdapterLoader(blob=blob, last_resort_path=tmp_path / "nope.yaml", ttl_seconds=60)
```

- [ ] **Step 2: Confirm fails** — ImportError.

- [ ] **Step 3: Write loader (full — load-bearing)**

Create `src/intelligence/sme/adapter_loader.py`:

```python
"""AdapterLoader: Blob fetch with TTL cache + layered fallback (spec §5).
Resolution: subscription → global → generic. Blob unreachable → last-cached;
no cache → embedded last-resort; mark health degraded.
"""
from __future__ import annotations
import hashlib, threading, time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
import yaml
from src.intelligence.sme.adapter_schema import Adapter

_GLOBAL = "sme_adapters/global"
_SUB = "sme_adapters/subscription"


class AdapterLoadError(RuntimeError): ...


class BlobReader(Protocol):
    def read_text(self, path: str) -> str: ...


@dataclass
class _Cached:
    adapter: Adapter
    loaded_at: float
    version: str
    content_hash: str
    source_path: str


def _hash(s: str) -> str: return hashlib.sha256(s.encode()).hexdigest()


class AdapterLoader:
    """ttl_seconds is cache freshness, NOT a fetch timeout (spec §3 inv. 8)."""
    def __init__(self, *, blob: BlobReader, last_resort_path: Path,
                 ttl_seconds: float = 300.0) -> None:
        if not last_resort_path.exists():
            raise AdapterLoadError(f"Embedded last-resort missing: {last_resort_path}")
        self._blob = blob
        self._lr = last_resort_path
        self._ttl = ttl_seconds
        self._cache: dict[tuple[str, str], _Cached] = {}
        self._lock = threading.Lock()
        self._status = "healthy"

    def load(self, sub_id: str, domain: str) -> Adapter:
        """Canonical: resolve adapter for (sub_id, domain). Returns Adapter
        with content_hash + version populated on the instance itself."""
        key = (sub_id, domain)
        now = time.monotonic()
        with self._lock:
            e = self._cache.get(key)
            if e and (now - e.loaded_at) < self._ttl: return e.adapter
        try:
            e = self._fetch(sub_id, domain)
            with self._lock:
                self._cache[key] = e; self._status = "healthy"
            return e.adapter
        except (ConnectionError, OSError):
            with self._lock:
                stale = self._cache.get(key)
                if stale is not None:
                    self._status = "degraded"; return stale.adapter
            e = self._load_last_resort()
            with self._lock:
                self._cache[key] = e; self._status = "degraded"
            return e.adapter

    # Back-compat alias — deprecated, will be removed after Phase 2 lands
    get = load

    def invalidate(self, sub_id: str, domain: str) -> None:
        with self._lock: self._cache.pop((sub_id, domain), None)

    def invalidate_all(self) -> None:
        with self._lock: self._cache.clear()

    def last_load_metadata(self, sub_id: str, domain: str) -> dict[str, str]:
        with self._lock:
            e = self._cache.get((sub_id, domain))
            return {} if e is None else {
                "version": e.version, "content_hash": e.content_hash,
                "source_path": e.source_path}

    def health_status(self) -> str:
        with self._lock: return self._status

    def _fetch(self, sub_id: str, domain: str) -> _Cached:
        paths = [f"{_SUB}/{sub_id}/{domain}.yaml",
                 f"{_GLOBAL}/{domain}.yaml",
                 f"{_GLOBAL}/generic.yaml"]
        last: Exception | None = None
        for path in paths:
            try: raw = self._blob.read_text(path)
            except FileNotFoundError as e: last = e; continue
            a = self._parse(raw, source_path=path)
            return _Cached(a, time.monotonic(), a.version, _hash(raw), path)
        raise AdapterLoadError(f"No adapter resolvable (last: {last})")

    def _load_last_resort(self) -> _Cached:
        raw = self._lr.read_text()
        a = self._parse(raw, source_path=f"embedded:{self._lr.name}")
        return _Cached(a, time.monotonic(), a.version, _hash(raw),
                       f"embedded:{self._lr.name}")

    @staticmethod
    def _parse(raw: str, *, source_path: str) -> Adapter:
        data = yaml.safe_load(raw)
        if not isinstance(data, dict):
            raise AdapterLoadError(f"Adapter at {source_path} is not a YAML mapping")
        a = Adapter(**data)
        # Populate runtime-injected fields per ERRATA §1
        return a.model_copy(update={
            "content_hash": _hash(raw),
            "source_path": source_path,
        })


# Module-level singleton factory (ERRATA §1)
_adapter_loader_singleton: AdapterLoader | None = None


def get_adapter_loader() -> AdapterLoader:
    """Return the process-wide AdapterLoader. Non-FastAPI callers use this;
    FastAPI lifespan wires the same instance into app.state."""
    global _adapter_loader_singleton
    if _adapter_loader_singleton is None:
        raise RuntimeError(
            "AdapterLoader not initialized — call init_adapter_loader() at startup"
        )
    return _adapter_loader_singleton


def init_adapter_loader(*, blob: BlobReader, last_resort_path: Path,
                        ttl_seconds: float = 300.0) -> AdapterLoader:
    """Initialize the singleton. Called once from app startup / CLI entrypoint."""
    global _adapter_loader_singleton
    _adapter_loader_singleton = AdapterLoader(
        blob=blob, last_resort_path=last_resort_path, ttl_seconds=ttl_seconds,
    )
    return _adapter_loader_singleton
```

- [ ] **Step 4: Run green** → 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/intelligence/sme/adapter_loader.py tests/intelligence/sme/test_adapter_loader.py
git commit -m "phase1(sme-adapter): Blob loader with TTL cache and last-resort fallback"
```

---

## Task 4: Adapter admin API endpoints

**Files:** create `src/api/sme_admin_api.py`, `tests/api/test_sme_admin_api.py`; modify `src/main.py` (single `include_router` line).

Endpoints (admin-gated by existing auth middleware):
- `PUT /admin/sme-adapters/{scope}/{domain}` — upload YAML body; `scope ∈ {global, sub/{sub_id}}`
- `DELETE /admin/sme-adapters/{scope}/{domain}`
- `GET /admin/sme-adapters/{scope}/{domain}` — returns parsed JSON + version + hash
- `POST /admin/sme-adapters/invalidate` — body `{subscription_id?, domain?}`; empty = invalidate_all

- [ ] **Step 1: Write representative tests**

Create `tests/api/test_sme_admin_api.py`:

```python
"""Tests for SME adapter admin endpoints."""
from unittest.mock import MagicMock
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from src.api.sme_admin_api import AdapterAdminDeps, build_router

_YAML = """domain: finance
version: 1.2.0
persona: {role: x, voice: y, grounding_rules: []}
dossier: {section_weights: {a: 1.0}, prompt_template: p/x.md}
insight_detectors: []
comparison_axes: []
kg_inference_rules: []
recommendation_frames: []
response_persona_prompts: {diagnose: p/d.md, analyze: p/a.md, recommend: p/r.md}
retrieval_caps: {max_pack_tokens: {analyze: 6000, diagnose: 5000, recommend: 4500, investigate: 8000}}
output_caps: {analyze: 1200, diagnose: 1500, recommend: 1000, investigate: 2000}
"""
_H = {"Content-Type": "application/x-yaml"}


@pytest.fixture
def deps(): return AdapterAdminDeps(loader=MagicMock(), blob_writer=MagicMock())


@pytest.fixture
def client(deps):
    app = FastAPI(); app.include_router(build_router(deps))
    return TestClient(app)


def test_put_global_adapter(client, deps):
    r = client.put("/admin/sme-adapters/global/finance", content=_YAML, headers=_H)
    assert r.status_code == 200
    assert deps.blob_writer.write_text.call_args[0][0] == \
        "sme_adapters/global/finance.yaml"
    deps.loader.invalidate_all.assert_called()


def test_put_rejects_invalid_yaml(client):
    r = client.put("/admin/sme-adapters/global/finance",
                   content="not: valid: yaml", headers=_H)
    assert r.status_code in (400, 422)


def test_put_rejects_mismatched_domain(client):
    r = client.put("/admin/sme-adapters/global/legal", content=_YAML, headers=_H)
    assert r.status_code == 400


def test_invalidate_scoped_and_all(client, deps):
    r = client.post("/admin/sme-adapters/invalidate",
                    json={"subscription_id": "sub_a", "domain": "finance"})
    assert r.status_code == 200
    deps.loader.invalidate.assert_called_with("sub_a", "finance")
    r2 = client.post("/admin/sme-adapters/invalidate", json={})
    assert r2.status_code == 200
    deps.loader.invalidate_all.assert_called_once()
```

- [ ] **Step 2: Implementation**

Create `src/api/sme_admin_api.py`:

```python
"""Admin endpoints for SME adapter YAMLs. Admin auth attaches at mount-level."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
import yaml
from fastapi import APIRouter, Body, HTTPException, Request
from src.intelligence.sme.adapter_loader import AdapterLoader
from src.intelligence.sme.adapter_schema import Adapter


class BlobWriter(Protocol):
    def write_text(self, path: str, content: str) -> None: ...
    def delete(self, path: str) -> None: ...
    def read_text(self, path: str) -> str: ...


@dataclass
class AdapterAdminDeps:
    loader: AdapterLoader; blob_writer: BlobWriter


def build_router(deps: AdapterAdminDeps) -> APIRouter:
    r = APIRouter(prefix="/admin/sme-adapters", tags=["sme-admin"])

    def _path(scope: str, domain: str) -> str:
        if scope == "global": return f"sme_adapters/global/{domain}.yaml"
        if scope.startswith("sub/"):
            return f"sme_adapters/subscription/{scope[4:]}/{domain}.yaml"
        raise HTTPException(400, f"Invalid scope {scope!r}")

    def _invalidate(scope: str, domain: str) -> None:
        if scope.startswith("sub/"): deps.loader.invalidate(scope[4:], domain)
        else: deps.loader.invalidate_all()

    @r.put("/{scope}/{domain}")
    async def put_adapter(scope: str, domain: str, request: Request) -> dict:
        raw = (await request.body()).decode("utf-8")
        try:
            data = yaml.safe_load(raw)
        except yaml.YAMLError as e:
            raise HTTPException(400, f"YAML parse error: {e}")
        if not isinstance(data, dict):
            raise HTTPException(400, "YAML body must be a mapping")
        try:
            adapter = Adapter(**data)
        except Exception as e:
            raise HTTPException(422, f"Adapter validation failed: {e}")
        if adapter.domain != domain:
            raise HTTPException(400,
                f"URL domain {domain!r} != YAML domain {adapter.domain!r}")
        path = _path(scope, domain)
        deps.blob_writer.write_text(path, raw)
        _invalidate(scope, domain)
        return {"status": "ok", "path": path, "version": adapter.version}

    @r.delete("/{scope}/{domain}")
    async def delete_adapter(scope: str, domain: str) -> dict:
        path = _path(scope, domain)
        deps.blob_writer.delete(path)
        _invalidate(scope, domain)
        return {"status": "ok", "path": path}

    @r.get("/{scope}/{domain}")
    async def get_adapter(scope: str, domain: str) -> dict:
        path = _path(scope, domain)
        raw = deps.blob_writer.read_text(path)
        return {"path": path,
                "adapter": Adapter(**yaml.safe_load(raw)).model_dump(mode="json")}

    @r.post("/invalidate")
    async def invalidate(body: dict = Body(default={})) -> dict:
        sub = body.get("subscription_id"); domain = body.get("domain")
        if sub and domain:
            deps.loader.invalidate(sub, domain)
            return {"status": "ok", "scope": f"{sub}/{domain}"}
        deps.loader.invalidate_all()
        return {"status": "ok", "scope": "all"}

    return r
```

- [ ] **Step 3: Wire in `src/main.py`** — add one import and one `include_router` call next to other admin routers. Wire the `AdapterLoader` + blob writer singletons into the app lifespan section:

```python
from src.api.sme_admin_api import build_router as _build_sme_admin_router, AdapterAdminDeps
# inside lifespan, after blob + loader singletons constructed:
app.include_router(_build_sme_admin_router(
    AdapterAdminDeps(loader=sme_adapter_loader, blob_writer=sme_blob_client)))
```

No production path consumes the loader in Phase 1; the router is registered but unused. Phase 2 wires synthesis consumers.

- [ ] **Step 4: Run green**

```bash
pytest tests/api/test_sme_admin_api.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/api/sme_admin_api.py src/main.py tests/api/test_sme_admin_api.py
git commit -m "phase1(sme-admin): adapter PUT/DELETE/GET/invalidate endpoints"
```

---

## Task 5: Default adapter YAMLs + prompt template placeholders

**Files:** create `deploy/sme_adapters/defaults/{generic,finance,legal,hr,medical,it_support}.yaml`, `deploy/sme_adapters/last_resort/generic.yaml`, and prompt placeholders under `deploy/sme_adapters/defaults/prompts/`.

No Python; only content. Every YAML must validate against `Adapter` from Task 2. Two full YAMLs shown (`generic.yaml`, `finance.yaml`); the remaining four follow the same shape with domain-appropriate values described below.

- [ ] **Step 1: Write `deploy/sme_adapters/defaults/generic.yaml`**

```yaml
# Generic SME adapter — always-works fallback per spec §5.
domain: generic
version: 1.0.0
persona:
  role: subject matter expert
  voice: neutral, precise, evidence-cautious
  grounding_rules:
    - cite a source for every factual claim
    - prefer quoting over paraphrasing when evidence is direct
    - flag low-confidence claims explicitly
dossier:
  section_weights: {overview: 0.35, key_findings: 0.35, open_questions: 0.30}
  prompt_template: prompts/generic_dossier.md
insight_detectors: []
comparison_axes: []
kg_inference_rules: []
recommendation_frames: []
response_persona_prompts:
  diagnose: prompts/generic_diagnose.md
  analyze:  prompts/generic_analyze.md
  recommend: prompts/generic_recommend.md
retrieval_caps:
  max_pack_tokens: {analyze: 6000, diagnose: 5000, recommend: 4500, investigate: 8000}
output_caps:
  analyze: 1200
  diagnose: 1500
  recommend: 1000
  investigate: 2000
```

- [ ] **Step 2: Write `deploy/sme_adapters/defaults/finance.yaml`**

```yaml
domain: finance
version: 1.0.0
persona:
  role: senior financial analyst advising the C-suite
  voice: direct, quantitative, appropriately hedged
  grounding_rules:
    - cite a source for every quantitative claim
    - label estimates as estimates; never as facts
    - state the reporting period for every financial figure
dossier:
  section_weights: {financial_health: 0.30, trends: 0.25, risks: 0.25, opportunities: 0.20}
  prompt_template: prompts/finance_dossier.md
insight_detectors:
  - {type: trend,       rule: qoq_change_gt,    params: {threshold: 0.05}}
  - {type: anomaly,     rule: ratio_outlier,    params: {z_threshold: 2.5}}
  - {type: risk,        rule: covenant_breach,  params: {}}
  - {type: opportunity, rule: margin_expansion, params: {min_delta: 0.03}}
comparison_axes:
  - {name: period,  dimension: temporal}
  - {name: revenue, dimension: monetary, unit: USD}
  - {name: margin,  dimension: ratio}
kg_inference_rules:
  - {pattern: "(a:Account)-[:FUNDS]->()-[:FUNDS]->(b:Account)",
     produces: indirectly_funds, confidence_floor: 0.6, max_hops: 3}
recommendation_frames:
  - {frame: trend_based, template: "Given the {trend} in {axis}, consider {action}.",
     requires: {insight_types: [trend]}}
response_persona_prompts:
  diagnose:  prompts/finance_diagnose.md
  analyze:   prompts/finance_analyze.md
  recommend: prompts/finance_recommend.md
retrieval_caps:
  max_pack_tokens: {analyze: 6000, diagnose: 5000, recommend: 4500, investigate: 8000}
output_caps: {analyze: 1200, diagnose: 1500, recommend: 1000, investigate: 2000}
```

- [ ] **Step 3: Write remaining four domain YAMLs**

Each follows the `finance.yaml` shape. Domain-specific values:

- `legal.yaml` — persona: senior contracts counsel; axes: party, effective_date, jurisdiction, obligation_type; detectors: conflict/gap/risk; frames: compliance, remediation; dossier: parties (0.30) + obligations (0.30) + risks (0.20) + open_items (0.20).
- `hr.yaml` — persona: HR business partner; axes: employee_band, tenure, performance_tier; detectors: trend (attrition), gap (policy), risk (compliance); frames: policy_update, intervention; dossier sums to 1.0 across headcount/performance/policy_coverage/risks.
- `medical.yaml` — persona: clinical documentation specialist (synthesizer, not diagnostician); axes: episode_date, acuity, medication_class; detectors: anomaly/gap/risk; frames: follow_up, escalation; dossier: episodes + medications + labs + risks.
- `it_support.yaml` — persona: senior support engineer; axes: symptom, component, severity; detectors: anomaly, risk (recurrence), opportunity (process_fix); frames: triage_ranked, permanent_fix; dossier: symptom_map + root_causes + workarounds + permanent_fixes.

- [ ] **Step 4: Copy `generic.yaml` to the last-resort path**

```bash
cp deploy/sme_adapters/defaults/generic.yaml deploy/sme_adapters/last_resort/generic.yaml
```

- [ ] **Step 5: Write prompt template placeholders**

Each file is a short Markdown template consumed verbatim by future builders. Phase 1 ships placeholders; Phase 2 replaces bodies without renaming files.

`deploy/sme_adapters/defaults/prompts/generic_dossier.md` is a Markdown template with `{persona.role}`, `{persona.voice}`, `{section_list}`, `{grounding_rules}`, and `{profile_content_block}` placeholders; mandates inline `(doc_id, chunk_id)` citations.

Create `generic_analyze.md`, `generic_diagnose.md`, `generic_recommend.md` with the same skeleton and intent-specific instructions (analyze → exec summary + observations + patterns; diagnose → symptom + causes + fixes; recommend → top-N + rationale + evidence + impact + assumptions).

One `{domain}_dossier.md` per domain with the persona prefilled — 5 more files. Domain-specific intent files land in Phase 4. Total shipped files: 9 (4 generic + 5 domain dossiers).

- [ ] **Step 6: Sanity-validate every YAML parses through the schema**

Run a one-liner that iterates every file under `deploy/sme_adapters/` and constructs `Adapter(**yaml.safe_load(p.read_text()))`. Expected: no exceptions for any of the 7 YAMLs.

- [ ] **Step 7: Commit**

```bash
git add -f deploy/sme_adapters
git commit -m "phase1(sme-adapter): default YAMLs for 5 domains + generic + last-resort"
```

---

## Task 6: SMEVerifier — five-check grounding gate

**Files:** create `src/intelligence/sme/artifact_models.py`, `src/intelligence/sme/verifier.py`, `tests/intelligence/sme/test_artifact_models.py`, `tests/intelligence/sme/test_verifier.py`.

Verifier is load-bearing per spec §6. All five checks shown in full.

- [ ] **Step 1: Artifact item models (compact)**

Create `src/intelligence/sme/artifact_models.py`:

```python
"""Pydantic models for individual artifact items.

Every item carries provenance (`evidence`) + `confidence`. Verifier depends on these.
"""
from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field


class EvidenceRef(BaseModel):
    model_config = ConfigDict(frozen=True)
    doc_id: str
    chunk_id: str
    quote: str | None = None


class ArtifactItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    item_id: str
    artifact_type: Literal["dossier", "insight", "comparison", "kg_edge", "recommendation"]
    subscription_id: str
    profile_id: str
    text: str
    evidence: list[EvidenceRef] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    inference_path: list[dict[str, Any]] = Field(default_factory=list)
    domain_tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
```

Tests in `tests/intelligence/sme/test_artifact_models.py` cover happy path, unknown-field rejection, and confidence bounds (three short tests, ~15 lines total).

- [ ] **Step 2: Representative verifier test — one case per check**

Create `tests/intelligence/sme/test_verifier.py`:

```python
"""Tests for SMEVerifier — one representative case per check."""
from unittest.mock import MagicMock
import pytest
from src.intelligence.sme.artifact_models import ArtifactItem, EvidenceRef
from src.intelligence.sme.verifier import SMEVerifier, VerifierChunkStore


def _item(**over):
    b = dict(item_id="i1", artifact_type="insight",
             subscription_id="sub_a", profile_id="prof_x",
             text="Revenue rose 12% in Q3.",
             evidence=[EvidenceRef(doc_id="d1", chunk_id="c1",
                                    quote="Q3 revenue up 12%")],
             confidence=0.9); b.update(over); return ArtifactItem(**b)


@pytest.fixture
def cs():
    s = MagicMock(spec=VerifierChunkStore)
    s.chunk_exists.return_value = True
    s.chunk_text.return_value = "Q3 revenue up 12% year over year."
    return s


@pytest.fixture
def v(cs): return SMEVerifier(chunk_store=cs, max_inference_hops=3)


def test_check1_evidence_presence(v):
    r = v.verify_one(_item(evidence=[]))
    assert not r.passed and r.failing_check == "evidence_presence" and r.drop_reason

def test_check2_chunk_missing(cs, v):
    cs.chunk_exists.return_value = False
    r = v.verify_one(_item())
    assert not r.passed and r.failing_check == "evidence_validity"

def test_check2_text_not_substantively_present(cs, v):
    cs.chunk_text.return_value = "Unrelated weather text."
    r = v.verify_one(_item(text="Revenue rose 12% in Q3."))
    assert not r.passed and r.failing_check == "evidence_validity"

def test_check3_inference_provenance_exceeds_max_hops(v):
    r = v.verify_one(_item(inference_path=[{"a": 1}] * 4))
    assert not r.passed and r.failing_check == "inference_provenance"

def test_check4_rollback_single_source(v):
    r = v.verify_one(_item(confidence=0.9))
    assert r.passed and r.adjusted_item.confidence <= 0.6

def test_check4_passes_with_two_sources(v):
    r = v.verify_one(_item(confidence=0.9, evidence=[
        EvidenceRef(doc_id="d1", chunk_id="c1", quote="Q3 revenue up 12%"),
        EvidenceRef(doc_id="d2", chunk_id="c7", quote="Q3 sales rose")]))
    assert r.passed and r.adjusted_item.confidence == pytest.approx(0.9)

def test_check5_contradiction_drops_lower_confidence(v):
    batch = [_item(item_id="i1", text="Q3 revenue rose.", confidence=0.95,
                    evidence=[EvidenceRef(doc_id="d1", chunk_id="c1"),
                              EvidenceRef(doc_id="d2", chunk_id="c2")]),
             _item(item_id="i2", text="Q3 revenue fell.", confidence=0.5)]
    vs = v.verify_batch(batch)
    assert [x.adjusted_item.item_id for x in vs if x.passed] == ["i1"]
    assert next(x for x in vs if not x.passed).failing_check == "contradiction"
```

- [ ] **Step 3: Write `verifier.py` (full — load-bearing)**

Create `src/intelligence/sme/verifier.py`:

```python
"""SMEVerifier: five-check fail-closed grounding gate (spec §6).
1. Evidence presence    — ≥1 evidence ref
2. Evidence validity    — chunks exist + item text substantively present
3. Inference provenance — inference_path ≤ adapter max_hops
4. Confidence calibration — >0.8 requires ≥2 evidence sources or rolls to 0.6
5. Contradiction        — contradicts higher-confidence w/o `conflict` tag → drop
"""
from __future__ import annotations
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Protocol
from src.intelligence.sme.artifact_models import ArtifactItem


class VerifierChunkStore(Protocol):
    def chunk_exists(self, doc_id: str, chunk_id: str) -> bool: ...
    def chunk_text(self, doc_id: str, chunk_id: str) -> str: ...


@dataclass
class Verdict:
    item_id: str
    passed: bool
    adjusted_item: ArtifactItem | None
    failing_check: str | None = None
    drop_reason: str | None = None


_ROLLBACK = 0.6
_HIGH = 0.8
_TEXT_SIM = 0.25
_CONTR_SIM = 0.6
_OPP = [("rose", "fell"), ("increase", "decrease"), ("up", "down"),
        ("grew", "declined"), ("gain", "loss"), ("profit", "loss")]


def _overlap(a, b): return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _opposite(a, b):
    al, bl = a.lower(), b.lower()
    return any((x in al and y in bl) or (y in al and x in bl) for x, y in _OPP)


class SMEVerifier:
    def __init__(self, *, chunk_store: VerifierChunkStore,
                 max_inference_hops: int = 3) -> None:
        self._cs = chunk_store; self._max = max_inference_hops

    def verify_batch(self, items: list[ArtifactItem]) -> list[Verdict]:
        vs: list[Verdict] = []
        accepted: list[ArtifactItem] = []
        for item in sorted(items, key=lambda i: -i.confidence):
            base = self.verify_one(item)
            if not base.passed: vs.append(base); continue
            c5 = self._contradiction(base.adjusted_item, accepted)
            if not c5.passed: vs.append(c5); continue
            accepted.append(base.adjusted_item); vs.append(base)
        return [next(v for v in vs if v.item_id == i.item_id) for i in items]

    def verify_one(self, it: ArtifactItem) -> Verdict:
        if not it.evidence:
            return Verdict(it.item_id, False, None,
                           "evidence_presence", "no evidence refs")
        any_present = False
        for ref in it.evidence:
            if not self._cs.chunk_exists(ref.doc_id, ref.chunk_id):
                return Verdict(it.item_id, False, None, "evidence_validity",
                               f"chunk {ref.doc_id}#{ref.chunk_id} missing")
            ctx = self._cs.chunk_text(ref.doc_id, ref.chunk_id)
            if _overlap(it.text, ctx) >= _TEXT_SIM: any_present = True
            if ref.quote and _overlap(ref.quote, ctx) < 0.5:
                return Verdict(it.item_id, False, None, "evidence_validity",
                               "cited quote not present in chunk")
        if not any_present:
            return Verdict(it.item_id, False, None, "evidence_validity",
                           "item text not substantively in any cited chunk")
        if it.inference_path and len(it.inference_path) > self._max:
            return Verdict(it.item_id, False, None, "inference_provenance",
                           f"path length {len(it.inference_path)} > {self._max}")
        adjusted = it
        if it.confidence > _HIGH and len(it.evidence) < 2:
            adjusted = it.model_copy(update={"confidence": _ROLLBACK})
        return Verdict(it.item_id, True, adjusted, None, None)

    def _contradiction(self, it: ArtifactItem,
                       accepted: list[ArtifactItem]) -> Verdict:
        tagged = "conflict" in it.domain_tags or \
                 it.metadata.get("conflict_annotation") is True
        if tagged:
            return Verdict(it.item_id, True, it, None, None)
        for prior in accepted:
            if prior.confidence <= it.confidence: continue
            if _overlap(prior.text, it.text) >= _CONTR_SIM and \
               _opposite(prior.text, it.text):
                return Verdict(it.item_id, False, None, "contradiction",
                               f"contradicts higher-confidence {prior.item_id}")
        return Verdict(it.item_id, True, it, None, None)
```

- [ ] **Step 4: Run green** → all pass.

- [ ] **Step 5: Commit**

```bash
git add src/intelligence/sme/artifact_models.py src/intelligence/sme/verifier.py \
        tests/intelligence/sme/test_artifact_models.py tests/intelligence/sme/test_verifier.py
git commit -m "phase1(sme-verifier): five-check fail-closed grounding gate"
```

---

## Task 7: Synthesis + query trace writers (Blob JSONL)

**Files:** create `src/intelligence/sme/trace.py`, `tests/intelligence/sme/test_trace.py`.

Paths per spec §9: synthesis → `sme_traces/synthesis/{sub}/{prof}/{synthesis_id}.jsonl`; query → `sme_traces/queries/{sub}/{prof}/{YYYY-MM-DD}/{query_id}.jsonl`.

- [ ] **Step 1: Representative tests**

Create `tests/intelligence/sme/test_trace.py`:

```python
"""Tests for SynthesisTraceWriter + QueryTraceWriter."""
from datetime import datetime
from unittest.mock import MagicMock
import pytest
from src.intelligence.sme.trace import (
    SynthesisTraceWriter, QueryTraceWriter, TraceBlobAppender)


@pytest.fixture
def appender(): return MagicMock(spec=TraceBlobAppender)


def test_synthesis_path(appender):
    w = SynthesisTraceWriter(appender)
    w.open(subscription_id="s", profile_id="p", synthesis_id="syn1")
    w.append({"stage": "start"}); w.close()
    assert appender.append.call_args_list[0][0][0] == \
        "sme_traces/synthesis/s/p/syn1.jsonl"


def test_appends_jsonl(appender):
    w = SynthesisTraceWriter(appender)
    w.open(subscription_id="s", profile_id="p", synthesis_id="syn1")
    w.append({"stage": "builder_start", "builder": "dossier"})
    lines = [c[0][1] for c in appender.append.call_args_list]
    assert lines[0].endswith("\n") and '"builder": "dossier"' in lines[0]


def test_query_path_uses_date(appender):
    fixed = datetime(2026, 4, 20, 14, 30)
    w = QueryTraceWriter(appender, now=lambda: fixed)
    w.open(subscription_id="s", profile_id="p", query_id="q42")
    w.append({"stage": "retrieval"})
    assert appender.append.call_args_list[0][0][0] == \
        "sme_traces/queries/s/p/2026-04-20/q42.jsonl"


def test_refuses_record_before_open(appender):
    with pytest.raises(RuntimeError, match="open"):
        SynthesisTraceWriter(appender).append({"x": 1})
```

- [ ] **Step 2: Implementation**

Create `src/intelligence/sme/trace.py`:

```python
"""Blob-backed JSONL trace writers (spec §11)."""
from __future__ import annotations
import json
from datetime import datetime
from typing import Any, Callable, Protocol


class TraceBlobAppender(Protocol):
    def append(self, path: str, line: str) -> None: ...


class _Base:
    def __init__(self, appender: TraceBlobAppender) -> None:
        self._a = appender; self._path: str | None = None

    def append(self, event: dict[str, Any]) -> None:
        """Canonical per ERRATA §5 — matches underlying TraceBlobAppender.append()."""
        if self._path is None:
            raise RuntimeError("Trace writer not open; call open() first")
        self._a.append(self._path,
                       json.dumps(event, default=str, ensure_ascii=False) + "\n")

    # Back-compat alias — deprecated
    record = append

    def close(self) -> None: self._path = None


class SynthesisTraceWriter(_Base):
    def open(self, *, subscription_id: str, profile_id: str, synthesis_id: str) -> None:
        self._path = f"sme_traces/synthesis/{subscription_id}/{profile_id}/{synthesis_id}.jsonl"


class QueryTraceWriter(_Base):
    def __init__(self, appender: TraceBlobAppender, *,
                 now: Callable[[], datetime] = datetime.utcnow) -> None:
        super().__init__(appender); self._now = now

    def open(self, *, subscription_id: str, profile_id: str, query_id: str) -> None:
        day = self._now().strftime("%Y-%m-%d")
        self._path = f"sme_traces/queries/{subscription_id}/{profile_id}/{day}/{query_id}.jsonl"
```

- [ ] **Step 3: Run green** → 4 passed.

- [ ] **Step 4: Commit**

```bash
git add src/intelligence/sme/trace.py tests/intelligence/sme/test_trace.py
git commit -m "phase1(sme-trace): Blob JSONL synthesis + query trace writers"
```

---

## Task 8: Artifact storage adapter (Blob + Qdrant + Neo4j)

**Files:** create `src/intelligence/sme/storage.py`, `tests/intelligence/sme/test_storage.py`.

Spec §9: canonical JSON → Blob `sme_artifacts/{sub}/{prof}/{type}/{version}.json`; snippets → Qdrant `sme_artifacts_{sub}`; edges → Neo4j `INFERRED_RELATION` (only for `kg_edge`).

- [ ] **Step 1: Representative tests**

Create `tests/intelligence/sme/test_storage.py`:

```python
"""Tests for SMEArtifactStorage."""
from unittest.mock import MagicMock
import pytest
from src.intelligence.sme.artifact_models import ArtifactItem, EvidenceRef
from src.intelligence.sme.storage import SMEArtifactStorage, StorageDeps


@pytest.fixture
def st():
    return SMEArtifactStorage(StorageDeps(
        blob=MagicMock(), qdrant=MagicMock(), neo4j=MagicMock()))


def _insight(**o):
    b = dict(item_id="i1", artifact_type="insight",
             subscription_id="sub_a", profile_id="prof_x",
             text="Revenue rose 12% QoQ.",
             evidence=[EvidenceRef(doc_id="d1", chunk_id="c1")],
             confidence=0.75, domain_tags=["trend"]); b.update(o)
    return ArtifactItem(**b)


def test_writes_blob_and_indexes_qdrant_with_isolation(st):
    st.persist_items("sub_a", "prof_x", "insight", [_insight()], version=1)
    assert st.deps.blob.write_text.call_args[0][0] == \
        "sme_artifacts/sub_a/prof_x/insight/1.json"
    kw = st.deps.qdrant.upsert_points.call_args[1]
    assert kw["collection"] == "sme_artifacts_sub_a"
    p = kw["points"][0]["payload"]
    assert p["subscription_id"] == "sub_a" and p["profile_id"] == "prof_x"


def test_neo4j_only_for_kg_edges(st):
    st.persist_items("s", "p", "insight", [_insight()], version=1)
    st.deps.neo4j.write_inferred_edges.assert_not_called()
    edge = _insight(artifact_type="kg_edge",
                    metadata={"from_node": "a", "to_node": "b",
                              "relation_type": "indirectly_funds"})
    st.persist_items("s", "p", "kg_edge", [edge], version=2)
    st.deps.neo4j.write_inferred_edges.assert_called_once()


def test_delete_version_clears_stores(st):
    st.delete_version("sub_a", "prof_x", "insight", version=1)
    st.deps.blob.delete.assert_called_once()
    st.deps.qdrant.delete_by_filter.assert_called_once()
```

- [ ] **Step 2: Implementation**

Create `src/intelligence/sme/storage.py`:

```python
"""Persistence for SME artifacts (spec §9). Profile isolation enforced."""
from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Any, Protocol
from src.intelligence.sme.artifact_models import ArtifactItem


class BlobStore(Protocol):
    def write_text(self, path: str, content: str) -> None: ...
    def delete(self, path: str) -> None: ...


class QdrantBridge(Protocol):
    def upsert_points(self, *, collection: str, points: list[dict[str, Any]]) -> None: ...
    def delete_by_filter(self, *, collection: str, filter: dict[str, Any]) -> None: ...


class Neo4jBridge(Protocol):
    def write_inferred_edges(self, edges: list[dict[str, Any]]) -> None: ...


@dataclass
class StorageDeps:
    """Phase 1 skeleton deps. Phase 2 extends with `embedder` for
    real put_snippet vector computation (see Phase 2 Task 11). `embedder`
    here is optional so Phase 1 tests that don't exercise vector writes
    can omit it; Phase 2 makes it required at call sites."""
    blob: BlobStore
    qdrant: QdrantBridge
    neo4j: Neo4jBridge
    embedder: object | None = None  # Phase 2 populates; Phase 1 skeleton may leave None


class SMEArtifactStorage:
    def __init__(self, deps: StorageDeps) -> None: self.deps = deps

    def persist_items(self, sub: str, prof: str, atype: str,
                      items: list[ArtifactItem], *, version: int) -> None:
        path = f"sme_artifacts/{sub}/{prof}/{atype}/{version}.json"
        body = {"subscription_id": sub, "profile_id": prof,
                "artifact_type": atype, "version": version,
                "items": [it.model_dump(mode="json") for it in items]}
        self.deps.blob.write_text(path, json.dumps(body, ensure_ascii=False))
        # Vector insertion lands in Phase 2; payload-only in Phase 1.
        self.deps.qdrant.upsert_points(
            collection=f"sme_artifacts_{sub}",
            points=[{"id": it.item_id, "payload": {
                "subscription_id": sub, "profile_id": prof, "artifact_type": atype,
                "text": it.text, "confidence": it.confidence,
                "domain_tags": it.domain_tags,
                "evidence": [e.model_dump() for e in it.evidence]}}
                for it in items])
        if atype == "kg_edge":
            self.deps.neo4j.write_inferred_edges([{
                "subscription_id": sub, "profile_id": prof,
                "from_node": it.metadata["from_node"],
                "to_node": it.metadata["to_node"],
                "relation_type": it.metadata["relation_type"],
                "confidence": it.confidence,
                "evidence": [f"{e.doc_id}#{e.chunk_id}" for e in it.evidence],
                "inference_path": it.inference_path} for it in items])

    def delete_version(self, sub: str, prof: str, atype: str, *, version: int) -> None:
        self.deps.blob.delete(f"sme_artifacts/{sub}/{prof}/{atype}/{version}.json")
        self.deps.qdrant.delete_by_filter(
            collection=f"sme_artifacts_{sub}",
            filter={"must": [{"key": "subscription_id", "value": sub},
                             {"key": "profile_id", "value": prof},
                             {"key": "artifact_type", "value": atype},
                             {"key": "version", "value": version}]})

    # ------ Facade methods per ERRATA §2 (consumers expect these names) ------

    def put_snippet(self, sub: str, prof: str, item: ArtifactItem,
                    *, synthesis_version: int) -> None:
        """Write one retrievable snippet to Qdrant sme_artifacts_{sub}.
        Canonical single-item write; callers use this during streaming persist."""
        self.deps.qdrant.upsert_points(
            collection=f"sme_artifacts_{sub}",
            points=[{"id": item.item_id, "payload": {
                "subscription_id": sub, "profile_id": prof,
                "artifact_type": item.artifact_type,
                "text": item.text, "confidence": item.confidence,
                "domain_tags": item.domain_tags,
                "evidence": [e.model_dump() for e in item.evidence],
                "synthesis_version": synthesis_version}}])

    def put_canonical(self, sub: str, prof: str, atype: str,
                      items: list[ArtifactItem], *, synthesis_version: int) -> None:
        """Write canonical JSON to Blob at
        sme_artifacts/{sub}/{prof}/{atype}/{synthesis_version}.json."""
        path = f"sme_artifacts/{sub}/{prof}/{atype}/{synthesis_version}.json"
        body = {"subscription_id": sub, "profile_id": prof,
                "artifact_type": atype, "version": synthesis_version,
                "items": [it.model_dump(mode="json") for it in items]}
        self.deps.blob.write_text(path, json.dumps(body, ensure_ascii=False))

    def put_manifest(self, sub: str, prof: str, manifest: dict) -> None:
        """Write synthesis run manifest (pointers to the above)."""
        path = f"sme_artifacts/{sub}/{prof}/manifest/{manifest['synthesis_id']}.json"
        self.deps.blob.write_text(path, json.dumps(manifest, ensure_ascii=False))
```

- [ ] **Step 3: Run green** → 5 passed.

- [ ] **Step 4: Commit**

```bash
git add src/intelligence/sme/storage.py tests/intelligence/sme/test_storage.py
git commit -m "phase1(sme-storage): artifact persistence across Blob/Qdrant/Neo4j"
```

---

## Task 9: Synthesizer orchestrator skeleton

**Files:** create `src/intelligence/sme/synthesizer.py`, `tests/intelligence/sme/test_synthesizer.py`.

Real control flow; builders return empty lists in Phase 1 so the end-to-end path stays exercisable. Phase 2 fills builders without touching this file.

- [ ] **Step 1: Representative tests**

Create `tests/intelligence/sme/test_synthesizer.py`:

```python
"""Tests for SMESynthesizer skeleton — control flow only."""
from unittest.mock import MagicMock
import pytest
from src.intelligence.sme.synthesizer import SMESynthesizer, SynthesizerDeps
from src.intelligence.sme.adapter_schema import Adapter


@pytest.fixture
def deps():
    d = SynthesizerDeps(
        adapter_loader=MagicMock(), storage=MagicMock(), verifier=MagicMock(),
        trace_writer=MagicMock(),
        builders={k: MagicMock() for k in
                  ("dossier", "insight", "comparison", "kg_edge", "recommendation")})
    a = MagicMock(spec=Adapter)
    a.version = "1.0.0"; a.domain = "generic"
    a.content_hash = "h"; a.source_path = "x"
    d.adapter_loader.load.return_value = a
    # last_load_metadata kept for introspection tests but no longer load-bearing
    d.adapter_loader.last_load_metadata.return_value = {
        "version": "1.0.0", "content_hash": "h", "source_path": "x"}
    for b in d.builders.values(): b.build.return_value = []
    d.verifier.verify_batch.return_value = []
    return d


def _run(deps):
    SMESynthesizer(deps).run(subscription_id="s", profile_id="p",
                             profile_domain="generic", synthesis_version=1)


def test_opens_and_closes_trace(deps):
    _run(deps)
    deps.trace_writer.open.assert_called_once_with(
        subscription_id="s", profile_id="p", synthesis_id="s:p:1")
    deps.trace_writer.close.assert_called_once()


def test_calls_all_five_builders(deps):
    _run(deps)
    for b in deps.builders.values(): b.build.assert_called_once()


def test_verifies_and_persists(deps):
    _run(deps)
    assert deps.verifier.verify_batch.call_count == 5
    assert deps.storage.persist_items.call_count == 5


def test_records_adapter_version(deps):
    _run(deps)
    events = [c[0][0] for c in deps.trace_writer.append.call_args_list]
    assert any(e.get("stage") == "start" for e in events)
    assert any(e.get("adapter_version") == "1.0.0" for e in events)
```

- [ ] **Step 2: Implementation (skeleton control flow)**

Create `src/intelligence/sme/synthesizer.py`:

```python
"""SMESynthesizer orchestrator skeleton. Phase 1 wires control flow;
builders return []. Phase 2 fills builder bodies, no change here."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Protocol
from src.intelligence.sme.adapter_loader import AdapterLoader
from src.intelligence.sme.artifact_models import ArtifactItem
from src.intelligence.sme.storage import SMEArtifactStorage
from src.intelligence.sme.trace import SynthesisTraceWriter
from src.intelligence.sme.verifier import SMEVerifier


class ArtifactBuilder(Protocol):
    artifact_type: str
    def build(self, *, subscription_id: str, profile_id: str,
              adapter: Any, version: int) -> list[ArtifactItem]: ...


@dataclass
class SynthesizerDeps:
    adapter_loader: AdapterLoader
    storage: SMEArtifactStorage
    verifier: SMEVerifier
    trace_writer: SynthesisTraceWriter
    builders: dict[str, ArtifactBuilder]


class SMESynthesizer:
    def __init__(self, deps: SynthesizerDeps) -> None: self._d = deps

    def run(self, *, subscription_id: str, profile_id: str,
            profile_domain: str, synthesis_version: int) -> dict[str, int]:
        sid = f"{subscription_id}:{profile_id}:{synthesis_version}"
        tw, d = self._d.trace_writer, self._d
        tw.open(subscription_id=subscription_id, profile_id=profile_id,
                synthesis_id=sid)
        try:
            adapter = d.adapter_loader.load(subscription_id, profile_domain)
            # content_hash + version are attributes on Adapter per ERRATA §1
            tw.append({"stage": "start", "synthesis_id": sid,
                       "adapter_version": adapter.version,
                       "adapter_hash": adapter.content_hash})
            counts: dict[str, int] = {}
            for atype, builder in d.builders.items():
                items = builder.build(subscription_id=subscription_id,
                                      profile_id=profile_id, adapter=adapter,
                                      version=synthesis_version)
                verdicts = d.verifier.verify_batch(items)
                accepted = [v.adjusted_item for v in verdicts if v.passed]
                for v in (v for v in verdicts if not v.passed):
                    tw.append({"stage": "verifier_drop", "builder": atype,
                               "item_id": v.item_id,
                               "failing_check": v.failing_check,
                               "drop_reason": v.drop_reason})
                d.storage.persist_items(subscription_id, profile_id, atype,
                                        accepted, version=synthesis_version)
                counts[atype] = len(accepted)
                tw.append({"stage": "builder_complete", "builder": atype,
                           "accepted": len(accepted),
                           "dropped": len(verdicts) - len(accepted)})
            tw.append({"stage": "complete", "counts": counts})
            return counts
        finally:
            tw.close()
```

- [ ] **Step 3: Run green** → 4 passed.

- [ ] **Step 4: Commit**

```bash
git add src/intelligence/sme/synthesizer.py tests/intelligence/sme/test_synthesizer.py
git commit -m "phase1(sme-synth): orchestrator skeleton — control flow, empty builders"
```

---

## Task 10: Five artifact builder skeletons

**Files:** create `src/intelligence/sme/builders/{_base,dossier,insight_index,comparative_register,kg_materializer,recommendation_bank}.py`; create `tests/intelligence/sme/builders/test_dossier.py` (plus four sibling copies per builder).

Phase 1 builders are real classes that return `[]`. Phase 2 replaces the `_synthesize()` body only; class names, constructor signature, and `artifact_type` slugs are frozen now.

- [ ] **Step 1: Base ABC**

Create `src/intelligence/sme/builders/_base.py`:

```python
"""Base class for SME artifact builders."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Protocol
from src.intelligence.sme.adapter_schema import Adapter
from src.intelligence.sme.artifact_models import ArtifactItem


class BuilderContext(Protocol):
    def iter_profile_chunks(self, sub: str, prof: str) -> list[dict[str, Any]]: ...
    def iter_profile_kg(self, sub: str, prof: str) -> list[dict[str, Any]]: ...


class ArtifactBuilder(ABC):
    artifact_type: str

    def __init__(self, *, ctx: BuilderContext) -> None:
        self._ctx = ctx

    def build(self, *, subscription_id: str, profile_id: str,
              adapter: Adapter, version: int) -> list[ArtifactItem]:
        return self._synthesize(subscription_id=subscription_id,
                                profile_id=profile_id, adapter=adapter,
                                version=version)

    @abstractmethod
    def _synthesize(self, **kwargs) -> list[ArtifactItem]: ...
```

- [ ] **Step 2: Five skeleton builders**

Create `src/intelligence/sme/builders/dossier.py`:

```python
"""SMEDossierBuilder — domain-aware dossier artifact (spec §6).
Phase 1 skeleton returns []. Phase 2 _synthesize(): one ArtifactItem per dossier
section with section text, evidence refs, entity mentions.
"""
from __future__ import annotations
from src.intelligence.sme.artifact_models import ArtifactItem
from src.intelligence.sme.builders._base import ArtifactBuilder


class SMEDossierBuilder(ArtifactBuilder):
    artifact_type = "dossier"
    def _synthesize(self, **kwargs) -> list[ArtifactItem]: return []
```

Same skeleton shape for the other four, class name + `artifact_type` + docstring of Phase 2 contract:
- `insight_index.py` → `InsightIndexBuilder`, `"insight"`. Phase 2: typed (trend/anomaly/gap/risk/opportunity/conflict) items with narrative + evidence + domain_tags + optional temporal_scope.
- `comparative_register.py` → `ComparativeRegisterBuilder`, `"comparison"`. Phase 2: delta/conflict/timeline/corroboration items citing ≥2 documents via `compared_items[]`.
- `kg_materializer.py` → `KGMultiHopMaterializer`, `"kg_edge"`. Phase 2: items whose metadata carries `from_node`, `to_node`, `relation_type`; storage converts to Neo4j `INFERRED_RELATION` edges.
- `recommendation_bank.py` → `RecommendationBankBuilder`, `"recommendation"`. Phase 2: recommendation + rationale + linked_insights + estimated_impact + assumptions + caveats.

- [ ] **Step 3: Representative stub test (one builder, sibling copies for four more)**

Create `tests/intelligence/sme/builders/test_dossier.py`:

```python
"""Skeleton test for SMEDossierBuilder (Phase 1)."""
from unittest.mock import MagicMock
from src.intelligence.sme.builders.dossier import SMEDossierBuilder


def test_skeleton_returns_empty_list():
    b = SMEDossierBuilder(ctx=MagicMock())
    assert b.build(subscription_id="s", profile_id="p",
                   adapter=MagicMock(), version=1) == []
    assert b.artifact_type == "dossier"
```

Create four sibling tests (`test_insight_index.py`, `test_comparative_register.py`, `test_kg_materializer.py`, `test_recommendation_bank.py`) — each ~10 lines, identical shape with class + artifact_type swapped. Phase 2 replaces each with real assertions.

- [ ] **Step 4: Run green**

```bash
pytest tests/intelligence/sme/builders -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/intelligence/sme/builders tests/intelligence/sme/builders
git commit -m "phase1(sme-builders): 5 skeleton builders + ABC — Phase 2 fills bodies"
```

---

## Task 11: Hybrid dense+sparse retrieval helper

**Files:** create `src/retrieval/hybrid_search.py`, `tests/retrieval/test_hybrid_search.py`.

Spec §7 Stage 1. Helper is flag-agnostic — the `enable_hybrid_retrieval` decision lives at the call site. Default flag OFF in Phase 1.

- [ ] **Step 1: Tests**

Create `tests/retrieval/test_hybrid_search.py`:

```python
"""Tests for hybrid dense+sparse RRF fusion."""
from unittest.mock import MagicMock
from src.retrieval.hybrid_search import HybridSearcher, HybridConfig


def _S(dense, sparse):
    q = MagicMock()
    q.search_dense.return_value = dense
    q.search_sparse.return_value = sparse
    return HybridSearcher(qdrant=q, config=HybridConfig(rrf_k=60)), q


_F = {"must": []}


def test_rrf_fuses_dense_and_sparse():
    s, _ = _S([{"id": "a"}, {"id": "b"}, {"id": "c"}],
              [{"id": "b"}, {"id": "d"}, {"id": "a"}])
    out = s.search(collection="c", query_text="x", query_vector=[0.0],
                   top_k=5, query_filter=_F)
    ids = [r.item_id for r in out]
    assert ids[0] == "b" and set(ids) == {"a", "b", "c", "d"}


def test_falls_back_to_dense_when_sparse_unavailable():
    q = MagicMock()
    q.search_sparse.side_effect = NotImplementedError
    q.search_dense.return_value = [{"id": "a"}]
    out = HybridSearcher(qdrant=q, config=HybridConfig()).search(
        collection="c", query_text="x", query_vector=[0.0], top_k=5, query_filter=_F)
    assert [r.item_id for r in out] == ["a"]


def test_respects_top_k_cap():
    s, _ = _S([{"id": f"d{i}"} for i in range(50)],
              [{"id": f"s{i}"} for i in range(50)])
    out = s.search(collection="c", query_text="x", query_vector=[0.0],
                   top_k=10, query_filter=_F)
    assert len(out) == 10


def test_filter_forwarded_to_both_backends():
    s, q = _S([], [])
    f = {"must": [{"key": "subscription_id", "value": "sub_a"},
                  {"key": "profile_id", "value": "prof_x"}]}
    s.search(collection="c", query_text="x", query_vector=[0.0],
             top_k=5, query_filter=f)
    assert q.search_dense.call_args[1]["query_filter"] == f
    assert q.search_sparse.call_args[1]["query_filter"] == f
```

- [ ] **Step 2: Implementation**

Create `src/retrieval/hybrid_search.py`:

```python
"""Hybrid dense+sparse RRF helper. Profile filter forwarded verbatim (spec §3)."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Protocol


class QdrantBridge(Protocol):
    def search_dense(self, *, collection: str, query_vector: list[float],
                     top_k: int, query_filter: dict[str, Any]) -> list[dict]: ...
    def search_sparse(self, *, collection: str, query_text: str,
                      top_k: int, query_filter: dict[str, Any]) -> list[dict]: ...


@dataclass
class HybridConfig:
    rrf_k: int = 60
    candidate_multiplier: int = 4


@dataclass
class HybridResult:
    item_id: str
    rrf_score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None
    payload: dict[str, Any] = field(default_factory=dict)


class HybridSearcher:
    def __init__(self, *, qdrant: QdrantBridge, config: HybridConfig) -> None:
        self._q = qdrant; self._c = config

    def search(self, *, collection: str, query_text: str,
               query_vector: list[float], top_k: int,
               query_filter: dict[str, Any]) -> list[HybridResult]:
        fetch = top_k * self._c.candidate_multiplier
        dense = self._q.search_dense(collection=collection, query_vector=query_vector,
                                     top_k=fetch, query_filter=query_filter)
        try:
            sparse = self._q.search_sparse(collection=collection, query_text=query_text,
                                           top_k=fetch, query_filter=query_filter)
        except NotImplementedError:
            sparse = []
        k = self._c.rrf_k
        by_id: dict[str, HybridResult] = {}
        for rank, d in enumerate(dense):
            r = by_id.setdefault(d["id"], HybridResult(item_id=d["id"], rrf_score=0.0,
                                                      payload=d.get("payload", {})))
            r.dense_rank = rank; r.rrf_score += 1.0 / (k + rank + 1)
        for rank, s in enumerate(sparse):
            r = by_id.setdefault(s["id"], HybridResult(item_id=s["id"], rrf_score=0.0,
                                                      payload=s.get("payload", {})))
            r.sparse_rank = rank; r.rrf_score += 1.0 / (k + rank + 1)
        return sorted(by_id.values(), key=lambda r: -r.rrf_score)[:top_k]
```

- [ ] **Step 3: Run green** → 4 passed.

- [ ] **Step 4: Commit**

```bash
git add src/retrieval/hybrid_search.py tests/retrieval/test_hybrid_search.py
git commit -m "phase1(retrieval): hybrid dense+sparse RRF helper — flag-gated off"
```

---

## Task 12: Cross-encoder reranker wiring

**Files:** create `src/retrieval/reranker.py`, `tests/retrieval/test_reranker.py`.

Default model (resolving spec §15 Q2): `cross-encoder/ms-marco-MiniLM-L-6-v2` — small (22M), CPU-fast, public, widely validated. Swap to a trained DocWain reranker in a future phase by changing one constructor arg.

Gated by `enable_cross_encoder_rerank` (default OFF in Phase 1).

- [ ] **Step 1: Tests**

Create `tests/retrieval/test_reranker.py`:

```python
"""Tests for CrossEncoderReranker."""
from unittest.mock import MagicMock, patch
from src.retrieval.reranker import CrossEncoderReranker, RerankCandidate


def _C(id, text): return RerankCandidate(id=id, text=text)


def test_loads_model_lazily():
    with patch("src.retrieval.reranker._load_model") as loader:
        loader.return_value = MagicMock()
        r = CrossEncoderReranker(model_name="m")
        assert loader.call_count == 0
        r.rerank("q", [_C("a", "x")])
        assert loader.call_count == 1


def test_preserves_top_n_ordering():
    m = MagicMock(); m.predict.return_value = [0.1, 0.9, 0.5]
    with patch("src.retrieval.reranker._load_model", return_value=m):
        out = CrossEncoderReranker(model_name="m").rerank(
            "q", [_C("a", "1"), _C("b", "2"), _C("c", "3")], top_n=2)
        assert [c.id for c in out] == ["b", "c"]


def test_empty_candidates_short_circuits():
    with patch("src.retrieval.reranker._load_model") as loader:
        assert CrossEncoderReranker(model_name="m").rerank("q", []) == []
        assert loader.call_count == 0


def test_caps_at_candidate_count():
    m = MagicMock(); m.predict.return_value = [0.5, 0.6]
    with patch("src.retrieval.reranker._load_model", return_value=m):
        out = CrossEncoderReranker(model_name="m").rerank(
            "q", [_C("a", "1"), _C("b", "2")], top_n=10)
        assert len(out) == 2
```

- [ ] **Step 2: Implementation**

Create `src/retrieval/reranker.py`:

```python
"""Cross-encoder reranker (spec §7 stage 2). Lazy-loaded; gated off in Phase 1."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class RerankCandidate:
    id: str
    text: str


def _load_model(name: str):  # isolated for tests to patch
    from sentence_transformers import CrossEncoder
    return CrossEncoder(name)


class CrossEncoderReranker:
    def __init__(self, *, model_name: str) -> None:
        self._name = model_name
        self._model = None

    def _ensure(self) -> None:
        if self._model is None:
            self._model = _load_model(self._name)

    def rerank(self, query: str, candidates: list[RerankCandidate],
               *, top_n: int = 10) -> list[RerankCandidate]:
        if not candidates: return []
        self._ensure()
        scores = self._model.predict([(query, c.text) for c in candidates])
        scored = sorted(zip(candidates, scores), key=lambda x: -float(x[1]))
        return [c for c, _ in scored[:top_n]]
```

- [ ] **Step 3: Run green** → 4 passed.

- [ ] **Step 4: Commit**

```bash
git add src/retrieval/reranker.py tests/retrieval/test_reranker.py
git commit -m "phase1(retrieval): cross-encoder reranker wiring — flag-gated off"
```

---

## Task 13: Feature flag facility

**Files:** create `src/config/feature_flags.py`, `tests/config/test_feature_flags.py`.

Flag names match spec §13.2 exactly. Resolution: per-subscription MongoDB override → global default (all OFF in Phase 1). Master flag `sme_redesign_enabled=false` forces all dependent flags off. Infrastructure-independent flags (`enable_hybrid_retrieval`, `enable_cross_encoder_rerank`) bypass master per §13.5.

- [ ] **Step 1: Tests**

Create `tests/config/test_feature_flags.py`:

```python
"""Tests for SMEFeatureFlags resolution."""
from unittest.mock import MagicMock
import pytest
from src.config.feature_flags import SMEFeatureFlags, FlagStore


@pytest.fixture
def store():
    s = MagicMock(spec=FlagStore); s.get_subscription_overrides.return_value = {}
    return s


def test_all_default_off(store):
    f = SMEFeatureFlags(store=store)
    for n in ("sme_redesign_enabled", "enable_sme_synthesis", "enable_sme_retrieval",
              "enable_kg_synthesized_edges", "enable_rich_mode",
              "enable_url_as_prompt", "enable_hybrid_retrieval",
              "enable_cross_encoder_rerank"):
        assert f.is_enabled("sub_a", n) is False


def test_master_gating(store):
    # master off → dependent off even if overridden on
    store.get_subscription_overrides.return_value = {"enable_sme_synthesis": True}
    assert SMEFeatureFlags(store=store).is_enabled("sub_a", "enable_sme_synthesis") is False
    # master on → overrides come through
    store.get_subscription_overrides.return_value = {
        "sme_redesign_enabled": True, "enable_sme_synthesis": True}
    f = SMEFeatureFlags(store=store)
    assert f.is_enabled("sub_a", "sme_redesign_enabled") is True
    assert f.is_enabled("sub_a", "enable_sme_synthesis") is True


def test_infrastructure_flags_bypass_master(store):
    """Spec §13.5: hybrid + reranker survive master rollback."""
    store.get_subscription_overrides.return_value = {
        "enable_hybrid_retrieval": True, "enable_cross_encoder_rerank": True}
    f = SMEFeatureFlags(store=store)
    assert f.is_enabled("sub_a", "enable_hybrid_retrieval") is True
    assert f.is_enabled("sub_a", "enable_cross_encoder_rerank") is True
    with pytest.raises(KeyError, match="unknown"):
        f.is_enabled("sub_a", "totally_made_up")
```

- [ ] **Step 2: Implementation**

Create `src/config/feature_flags.py`:

```python
"""SME feature flag resolution (spec §13). MongoDB feature_flags collection
= control-plane data (allowed by storage-separation rule)."""
from __future__ import annotations
from typing import Final, Protocol

# ---- Exported flag-name constants (canonical per ERRATA §4) ----
SME_REDESIGN_ENABLED: Final[str] = "sme_redesign_enabled"
ENABLE_SME_SYNTHESIS: Final[str] = "enable_sme_synthesis"
ENABLE_SME_RETRIEVAL: Final[str] = "enable_sme_retrieval"
ENABLE_KG_SYNTHESIZED_EDGES: Final[str] = "enable_kg_synthesized_edges"
ENABLE_RICH_MODE: Final[str] = "enable_rich_mode"
ENABLE_URL_AS_PROMPT: Final[str] = "enable_url_as_prompt"
ENABLE_HYBRID_RETRIEVAL: Final[str] = "enable_hybrid_retrieval"
ENABLE_CROSS_ENCODER_RERANK: Final[str] = "enable_cross_encoder_rerank"

_MASTER = SME_REDESIGN_ENABLED
_DEPENDENT = {ENABLE_SME_SYNTHESIS, ENABLE_SME_RETRIEVAL,
              ENABLE_KG_SYNTHESIZED_EDGES, ENABLE_RICH_MODE,
              ENABLE_URL_AS_PROMPT}
_INDEPENDENT = {ENABLE_HYBRID_RETRIEVAL, ENABLE_CROSS_ENCODER_RERANK}
_ALL = {_MASTER, *_DEPENDENT, *_INDEPENDENT}
_DEFAULTS: dict[str, bool] = {n: False for n in _ALL}


class FlagStore(Protocol):
    def get_subscription_overrides(self, subscription_id: str) -> dict[str, bool]: ...


class SMEFeatureFlags:
    def __init__(self, *, store: FlagStore) -> None:
        self._s = store

    def is_enabled(self, subscription_id: str, flag: str) -> bool:
        if flag not in _ALL:
            raise KeyError(f"unknown feature flag {flag!r}")
        ov = self._s.get_subscription_overrides(subscription_id)
        master_on = ov.get(_MASTER, _DEFAULTS[_MASTER])
        if flag in _DEPENDENT and not master_on:
            return False
        return bool(ov.get(flag, _DEFAULTS[flag]))


# ---- Module-level singleton factory per ERRATA §4 ----
_flag_resolver_singleton: SMEFeatureFlags | None = None


def get_flag_resolver() -> SMEFeatureFlags:
    """Return the process-wide SMEFeatureFlags instance.
    Non-FastAPI callers use this; FastAPI lifespan wires the same instance
    into app.state. Call init_flag_resolver() once at startup before use."""
    global _flag_resolver_singleton
    if _flag_resolver_singleton is None:
        raise RuntimeError(
            "SMEFeatureFlags not initialized — call init_flag_resolver() at startup"
        )
    return _flag_resolver_singleton


def init_flag_resolver(*, store: FlagStore) -> SMEFeatureFlags:
    """Initialize the singleton. Called once from app startup / CLI entrypoint."""
    global _flag_resolver_singleton
    _flag_resolver_singleton = SMEFeatureFlags(store=store)
    return _flag_resolver_singleton
```

- [ ] **Step 3: Run green** → 5 passed.

- [ ] **Step 4: Commit**

```bash
git add src/config/__init__.py src/config/feature_flags.py \
        tests/config/__init__.py tests/config/test_feature_flags.py
git commit -m "phase1(flags): SME feature flag resolution — master + dependent + independent"
```

---

## Task 14: Qdrant sparse-vector re-index migration script

**Files:** create `scripts/reindex_qdrant_sparse.py`, `tests/scripts/test_reindex_qdrant_sparse.py`.

Spec §12 Phase 1: "Qdrant re-index to add sparse vectors (per-sub rolling, dense-only fallback during migration)". Per §13.5, sparse re-index survives full rollback.

- [ ] **Step 1: Tests**

Create `tests/scripts/test_reindex_qdrant_sparse.py`:

```python
"""Tests for scripts/reindex_qdrant_sparse.py."""
from unittest.mock import MagicMock
import pytest
from scripts.reindex_qdrant_sparse import SparseReindexer, ReindexDeps


@pytest.fixture
def deps(): return ReindexDeps(qdrant=MagicMock(), sparse_encoder=MagicMock())


def test_reindex_iterates_one_subscription(deps):
    deps.qdrant.scroll_points.return_value = iter([
        [{"id": "p1", "payload": {"text": "hi"}, "vector": [0.0]},
         {"id": "p2", "payload": {"text": "foo"}, "vector": [0.1]}]])
    deps.sparse_encoder.encode_batch.return_value = [
        {"indices": [1], "values": [0.3]}, {"indices": [2], "values": [0.9]}]
    assert SparseReindexer(deps).reindex_subscription("sub_a") == 2
    deps.qdrant.update_points_sparse.assert_called_once()


def test_idempotent_and_skips_has_sparse(deps):
    deps.qdrant.scroll_points.return_value = iter([])
    assert SparseReindexer(deps).reindex_subscription("sub_a") == 0
    deps.qdrant.scroll_points.return_value = iter([
        [{"id": "p1", "payload": {"text": "t", "has_sparse": True},
          "vector": [0.0]}]])
    assert SparseReindexer(deps).reindex_subscription("sub_a") == 0
    deps.qdrant.update_points_sparse.assert_not_called()
```

- [ ] **Step 2: Implementation**

Create `scripts/reindex_qdrant_sparse.py`:

```python
"""One-time rolling migration: add sparse vectors to existing chunk points.
Idempotent: skips points whose payload has `has_sparse=True`.
Usage: python -m scripts.reindex_qdrant_sparse --subscription sub_abc | --all
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from typing import Any, Iterable, Protocol


class QdrantBridge(Protocol):
    def scroll_points(self, collection: str, batch_size: int = 256
                      ) -> Iterable[list[dict[str, Any]]]: ...
    def update_points_sparse(self, *, collection: str,
                             point_updates: list[dict[str, Any]]) -> None: ...


class SparseEncoder(Protocol):
    def encode_batch(self, texts: list[str]) -> list[dict[str, Any]]: ...


@dataclass
class ReindexDeps:
    qdrant: QdrantBridge; sparse_encoder: SparseEncoder


class SparseReindexer:
    def __init__(self, deps: ReindexDeps) -> None: self._d = deps

    def reindex_subscription(self, subscription_id: str) -> int:
        total = 0
        for batch in self._d.qdrant.scroll_points(subscription_id):
            needy = [p for p in batch if not p["payload"].get("has_sparse")]
            if not needy: continue
            texts = [p["payload"].get("text", "") for p in needy]
            sparse = self._d.sparse_encoder.encode_batch(texts)
            updates = [{"id": p["id"], "sparse_vector": s,
                        "payload_patch": {"has_sparse": True}}
                       for p, s in zip(needy, sparse)]
            self._d.qdrant.update_points_sparse(
                collection=subscription_id, point_updates=updates)
            total += len(updates)
        return total


def main() -> None:  # pragma: no cover — operator path wired in Phase 2
    p = argparse.ArgumentParser()
    p.add_argument("--subscription", action="append")
    p.add_argument("--all", action="store_true")
    p.parse_args()
    raise SystemExit("Wire via app lifespan in Phase 2.")


if __name__ == "__main__":  # pragma: no cover
    main()
```

- [ ] **Step 3: Run green** → 3 passed.

- [ ] **Step 4: Commit**

```bash
git add scripts/reindex_qdrant_sparse.py tests/scripts/test_reindex_qdrant_sparse.py
git commit -m "phase1(retrieval): one-time rolling sparse re-index script (idempotent)"
```

---

## Task 15: Sandbox end-to-end integration test + Phase 1 exit checklist

**Files:** create `tests/intelligence/sme/test_sandbox_integration.py`, `tests/intelligence/sme/conftest.py`.

Proves every seam from Tasks 2–10 holds under one end-to-end run on a sandbox subscription. Skeleton builders return `[]`; verifier, storage, and trace writers still run; Blob, Qdrant, Neo4j fakes observe all writes.

- [ ] **Step 1: In-memory fakes**

Create `tests/intelligence/sme/conftest.py`:

```python
"""Shared in-memory fakes for SME integration testing."""
from __future__ import annotations
from collections import defaultdict
from typing import Any


class InMemoryBlob:
    def __init__(self): self.files: dict[str, str] = {}
    def read_text(self, path):
        if path not in self.files: raise FileNotFoundError(path)
        return self.files[path]
    def write_text(self, path, content): self.files[path] = content
    def delete(self, path): self.files.pop(path, None)
    def append(self, path, line): self.files[path] = self.files.get(path, "") + line


class InMemoryQdrant:
    def __init__(self): self.points: dict[str, list[dict]] = defaultdict(list)
    def upsert_points(self, *, collection, points): self.points[collection].extend(points)
    def delete_by_filter(self, *, collection, filter): self.points[collection] = []
    def search_dense(self, **_): return []
    def search_sparse(self, **_): return []


class InMemoryNeo4j:
    def __init__(self): self.edges: list[dict] = []
    def write_inferred_edges(self, edges): self.edges.extend(edges)
```

- [ ] **Step 2: End-to-end integration test**

Create `tests/intelligence/sme/test_sandbox_integration.py`:

```python
"""Sandbox end-to-end plumbing — all skeletons, verifier + storage + trace live."""
import json
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from src.intelligence.sme.adapter_loader import AdapterLoader
from src.intelligence.sme.builders.comparative_register import ComparativeRegisterBuilder
from src.intelligence.sme.builders.dossier import SMEDossierBuilder
from src.intelligence.sme.builders.insight_index import InsightIndexBuilder
from src.intelligence.sme.builders.kg_materializer import KGMultiHopMaterializer
from src.intelligence.sme.builders.recommendation_bank import RecommendationBankBuilder
from src.intelligence.sme.storage import SMEArtifactStorage, StorageDeps
from src.intelligence.sme.synthesizer import SMESynthesizer, SynthesizerDeps
from src.intelligence.sme.trace import SynthesisTraceWriter
from src.intelligence.sme.verifier import SMEVerifier
from tests.intelligence.sme.conftest import InMemoryBlob, InMemoryNeo4j, InMemoryQdrant


LR = Path("deploy/sme_adapters/last_resort/generic.yaml")


@pytest.fixture
def wired():
    blob = InMemoryBlob()
    blob.files["sme_adapters/global/generic.yaml"] = LR.read_text()
    qdrant, neo4j = InMemoryQdrant(), InMemoryNeo4j()
    loader = AdapterLoader(blob=blob, last_resort_path=LR, ttl_seconds=60)
    storage = SMEArtifactStorage(StorageDeps(blob=blob, qdrant=qdrant, neo4j=neo4j))
    cs = MagicMock(); cs.chunk_exists.return_value = True
    cs.chunk_text.return_value = ""
    ctx = MagicMock(); ctx.iter_profile_chunks.return_value = []
    ctx.iter_profile_kg.return_value = []
    builders = {
        "dossier": SMEDossierBuilder(ctx=ctx),
        "insight": InsightIndexBuilder(ctx=ctx),
        "comparison": ComparativeRegisterBuilder(ctx=ctx),
        "kg_edge": KGMultiHopMaterializer(ctx=ctx),
        "recommendation": RecommendationBankBuilder(ctx=ctx),
    }
    syn = SMESynthesizer(SynthesizerDeps(
        adapter_loader=loader, storage=storage,
        verifier=SMEVerifier(chunk_store=cs, max_inference_hops=3),
        trace_writer=SynthesisTraceWriter(blob), builders=builders))
    return syn, blob, qdrant, neo4j


def test_sandbox_end_to_end_plumbing(wired):
    syn, blob, qdrant, neo4j = wired
    counts = syn.run(subscription_id="sandbox", profile_id="prof_1",
                     profile_domain="generic", synthesis_version=1)
    assert counts == {"dossier": 0, "insight": 0, "comparison": 0,
                      "kg_edge": 0, "recommendation": 0}
    for atype in ("dossier", "insight", "comparison", "kg_edge", "recommendation"):
        body = json.loads(blob.files[f"sme_artifacts/sandbox/prof_1/{atype}/1.json"])
        assert body["items"] == []
        assert body["subscription_id"] == "sandbox"
        assert body["profile_id"] == "prof_1"
    assert qdrant.points["sme_artifacts_sandbox"] == []
    assert neo4j.edges == []
    trace = blob.files["sme_traces/synthesis/sandbox/prof_1/sandbox:prof_1:1.jsonl"]
    stages = [json.loads(l)["stage"] for l in trace.splitlines()]
    assert stages[0] == "start" and stages[-1] == "complete"
    assert stages.count("builder_complete") == 5


def test_cross_subscription_isolation(wired):
    syn, _b, qdrant, _n = wired
    syn.run(subscription_id="sandbox", profile_id="p1",
            profile_domain="generic", synthesis_version=1)
    syn.run(subscription_id="other_sub", profile_id="p1",
            profile_domain="generic", synthesis_version=1)
    for coll in ("sme_artifacts_sandbox", "sme_artifacts_other_sub"):
        for p in qdrant.points[coll]:
            assert p["payload"]["subscription_id"] == \
                coll.removeprefix("sme_artifacts_")
```

- [ ] **Step 3: Run green** → 2 passed.

- [ ] **Step 4: Phase 1 exit checklist — run before declaring done**

Each box genuinely ticked, not wishfully checked. Matches spec §12 Phase 1 gate.

- [ ] Scaffolding committed; `pytest tests/intelligence/sme -v` discovers and passes every file.
- [ ] Adapter schema rejects unknown fields, enforces semver, requires section weights to sum to 1.0.
- [ ] AdapterLoader resolves `sub → global → generic`, TTL-caches, degrades to last-resort on Blob outage.
- [ ] Admin adapter CRUD + invalidate endpoints mount cleanly; PUT rejects URL-vs-body domain mismatch.
- [ ] Default YAMLs for `generic, finance, legal, hr, medical, it_support` ship under `deploy/sme_adapters/defaults/`; each parses through `Adapter`.
- [ ] `deploy/sme_adapters/last_resort/generic.yaml` exists and loads.
- [ ] SMEVerifier fires on all 5 checks; >0.8 with one source rolls to ≤0.6; contradiction drops only against higher-confidence un-tagged items.
- [ ] Trace writers append JSONL at spec Blob paths and refuse `record()` before `open()`.
- [ ] Storage writes canonical Blob JSON, indexes per-subscription Qdrant, writes Neo4j edges only for `kg_edge`.
- [ ] Synthesizer runs all 5 builders, verifies, persists, records trace with `start` + 5 × `builder_complete` + `complete`.
- [ ] 5 builder skeletons subclass `ArtifactBuilder`, return `[]`; Phase 2 can fill `_synthesize()` without other changes.
- [ ] `HybridSearcher` fuses RRF, falls back to dense-only on sparse unavailable, forwards filters verbatim.
- [ ] `CrossEncoderReranker` lazy-loads; default `cross-encoder/ms-marco-MiniLM-L-6-v2`; preserves top-n ordering.
- [ ] `SMEFeatureFlags` all False at Phase 1 defaults; master gating works; hybrid + reranker flags survive master off.
- [ ] `scripts/reindex_qdrant_sparse.py` idempotent; skips points with `payload.has_sparse=True`.
- [ ] Sandbox integration test proves end-to-end plumbing with empty artifacts; cross-subscription isolation holds.
- [ ] `src/agent/`, `src/generation/prompts.py`, `src/intelligence/generator.py`, `src/execution/router.py` untouched.
- [ ] No new `pipeline_status` string (`git grep -n 'PIPELINE_' src/` zero net additions).
- [ ] No hardcoded domain YAMLs under `src/` (`git grep -l 'domain: finance' src/` empty).
- [ ] Full suite green: `pytest tests/intelligence/sme tests/retrieval tests/config tests/api/test_sme_admin_api.py tests/scripts/test_reindex_qdrant_sparse.py -v`.
- [ ] `sme_redesign_enabled=false` globally; zero production behavior change.

- [ ] **Step 5: Commit**

```bash
git add tests/intelligence/sme/conftest.py tests/intelligence/sme/test_sandbox_integration.py
git commit -m "phase1(sme-integration): sandbox end-to-end plumbing + exit checklist"
```

---

## Self-review appendix

**Spec coverage:**
- §2 scope — adapter loader + admin API + defaults (3/4/5), verifier (6), trace (7), storage (8), synthesizer (9), five builder skeletons (10), Qdrant sparse re-index (14), reranker (12).
- §3 invariants — (1) query path untouched; (2) ingestion-time synthesis wired; (3) no new `pipeline_status`; (4) profile isolation at 8 + 15; (5) MongoDB control-plane only; YAMLs/artifacts/traces in Blob; snippets in Qdrant; edges in Neo4j; (6) no domain strings in `src/`; (7) fail-closed verifier (6); (8) no internal timeouts — only `adapter_loader` Blob fetch has a safety timeout; (9) `prompts.py` untouched (exit-gate).
- §5/6/9/13 — Tasks 2–5, 6, 8, 13.
- §15 Q2 — cross-encoder resolved: `cross-encoder/ms-marco-MiniLM-L-6-v2` (Task 12).

**Type consistency:**
- Adapter schema models defined once in `adapter_schema.py`; `ArtifactItem`/`EvidenceRef` once in `artifact_models.py`; `Verdict` once in `verifier.py`.
- Blob protocols split narrow per consumer (`BlobReader`/`BlobStore`/`BlobWriter`); one concrete Azure Blob client satisfies all three structurally. `QdrantBridge` defined in Tasks 8 and 11 with disjoint methods; same client satisfies both.
- Each `Neo4jBridge`, `TraceBlobAppender`, `FlagStore`, `SparseEncoder` defined once and used only by its owner.

**No redefinition:** every file created in exactly one task. Only `src/main.py` is touched outside of creation (a single `include_router` line in Task 4). `scripts/reindex_qdrant_sparse.py::main()` is an intentional stub — runtime wiring lands in Phase 2.

**Placeholder scan:** no "TBD" / "TODO" / "x.y.z" tokens in any task body.

**Memory-rule pass:** no Claude / Anthropic references; no new `pipeline_status`; MongoDB only for control-plane flags; YAMLs/prompts/artifacts/traces in Blob; `prompts.py` + `generator.py` untouched; Teams app isolated.

---

*End of Phase 1 plan. Phase 2 fills builder bodies, wires the synthesizer into the training stage as the final step before `PIPELINE_TRAINING_COMPLETED`, and lands Qdrant vector insertion that Phase 1 stubs on payload-only.*
