# DocWain SME Phase 6 — Pattern Mining & Monthly Review Loop

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the engineering-first → pattern-capture → future-training loop. Build a monthly batch job that clusters SME synthesis + query traces (from Azure Blob) into interpretable success patterns, failure patterns, artifact-utility rankings, and persona-effect reports, writes a monthly Markdown findings file, and produces a training-candidate list that feeds sub-project F only after human review.

**Architecture:** A standalone batch job in `scripts/mine_sme_patterns.py` reads the trace-store schemas created in Phase 1 (`sme_traces/synthesis/...` and `sme_traces/queries/...` Azure Blob prefixes), merges Redis feedback from `src/intelligence/feedback_tracker.py`, and runs four transparent clustering passes (rule-based + tf-idf + k-means so every cluster can be explained in one sentence). A systemd timer runs it monthly. A separate evaluator script `scripts/evaluate_training_trigger.py` compares the current month's failure clusters against prior months and emits a training-candidate list when clusters stabilize. No fine-tuning is triggered automatically — sub-project F remains separate, human-gated.

**Tech Stack:** Python 3.12, `scikit-learn` for TfidfVectorizer + KMeans, `pydantic` for schemas, `pyyaml` for config, `pytest` with `unittest.mock` for trace-store injection, `jinja2` for the analytics template render, existing `src/storage/azure_blob_client.py` for Blob reads, existing `src/intelligence/feedback_tracker.py` for Redis feedback merge. No LLM calls — this is evidence production, not generation.

**Related spec:** `docs/superpowers/specs/2026-04-20-docwain-profile-sme-reasoning-design.md` Sections 11 (pattern-capture instrumentation), 12 Phase 6 (monthly review loop), 13.3/13.6 (rollback post-mortem integration).

**Prior plan prerequisites (derive contracts from spec if missing):**
- Phase 0 (`2026-04-20-docwain-sme-phase0-baseline.md`) — exists; defines `tests/sme_evalset_v1/`, baseline schema patterns, CLI conventions this plan follows.
- Phases 1–5 — NOT WRITTEN AT TIME OF THIS PLAN. Synthesis + query trace record schemas derived directly from spec Section 11 below (see "Trace record contract"). If a later Phase 1 plan refines these fields, `scripts/sme_patterns/trace_loader.py` is the single seam to update; cluster code uses only the frozen projection produced by the loader. Phase 6 runs 30 days after Phase 4 wide rollout per spec Section 12.

**Memory rules that constrain this plan (hard):**
- **No Claude Attribution** — No Anthropic / Claude / Co-Authored-By references anywhere.
- **No Timeouts on internal paths** — The batch job is not latency-sensitive. The only timeouts are Azure Blob SDK's per-request safety timeouts (already handled upstream in `src/storage/azure_blob_client.py`); the batch loop has no wall-clock abort.
- **Traces live in Azure Blob, not Mongo** — Reader reads JSONL Blobs. No Mongo writes.
- **Response formatting in `src/generation/prompts.py` only** — Phase 6 touches zero prompt code. Analytics Markdown is rendered from a Jinja2 template under `analytics/templates/`, never from prompt modules.
- **Engineering-first** — The trigger-condition script produces evidence, never triggers retraining. Sub-project F remains a separate human-gated project.
- **Profile isolation** — Pattern mining operates per `(subscription_id, profile_id)`. Cross-subscription aggregation only appears in the "global rollup" section of the report, and even there every cluster carries its source subscription list.
- **No customer data in training** — Patterns are evidence for future training; any rendered query text in the analytics file is truncated to 120 chars and hashed for unique reference. The raw trace blobs themselves are the source of record.

---

## Trace record contract (Phase 1 / Phase 6 seam, derived from spec Section 11)

Phase 1 materializes these; Phase 6's loader consumes the projections. `scripts/sme_patterns/trace_loader.py` is the only module that reads raw JSONL — everything else operates on the projections.

### Synthesis trace (one `.jsonl` per synthesis run)

Blob path: `sme_traces/synthesis/{subscription_id}/{profile_id}/{synthesis_id}.jsonl`. Each line is one event. The loader projects into:

```
SynthesisRun {
  subscription_id: str
  profile_id: str
  synthesis_id: str
  started_at: datetime
  completed_at: datetime | None
  adapter_version: str
  adapter_content_hash: str
  profile_domain: str
  per_builder: dict[str, BuilderTrace]   # dossier / insight_index / comparative_register / kg_materializer / recommendation_bank
  verifier_drops: list[VerifierDrop]     # per-item drops with reason codes
}

BuilderTrace {
  builder_name: str
  items_produced: int
  items_persisted: int        # items_produced - verifier_drops for this builder
  duration_ms: float | None   # informational, not a gate
  errors: list[str]
}

VerifierDrop {
  item_id: str
  builder: str
  reason_code: str            # e.g. "evidence_presence", "confidence_calibration", "contradiction"
  detail: str
}
```

### Query trace

Blob path: `sme_traces/queries/{subscription_id}/{profile_id}/{YYYY-MM-DD}/{query_id}.jsonl`. Projection:

```
QueryRun {
  subscription_id: str
  profile_id: str
  profile_domain: str
  query_id: str
  query_text: str
  query_fingerprint: str         # sha1 over tokenized query (stable across cases/whitespace)
  intent: str                    # one of the intent labels (spec Section 8)
  format_hint: str | None        # "compact" | "rich" | None
  adapter_version: str
  adapter_persona_role: str      # redundant-but-useful copy of persona.role
  retrieval_layers: {
    chunks: int, kg: int, sme_artifacts: int, url: int
  }
  pack_tokens: int
  reasoner_prompt_hash: str
  response_len_tokens: int
  citation_verifier_drops: int
  honest_compact_fallback: bool
  url_present: bool
  url_fetch_ok: bool | None
  timing_ms: {understand, retrieval, reasoner, compose, total}
  feedback: QueryFeedback | None
  captured_at: datetime
}

QueryFeedback {
  rating: int | None             # -1 / 0 / +1 (thumbs-down / none / thumbs-up)
  edited: bool
  follow_up_count: int
  source: str                    # "feedback_tracker" | "implicit"
}
```

Feedback is filled from the explicit feedback field Phase 1 writes into the query trace, or (when missing) from the Redis `feedback_tracker` aggregates. Phase 6 tests construct synthetic traces against this contract; any Phase 1 deviation is caught by the integration test in Task 14.

---

## File structure

```
scripts/sme_patterns/                              [NEW]
├── __init__.py
├── trace_loader.py                                # Reads Blob JSONL → SynthesisRun / QueryRun projections
├── feedback_merger.py                             # Merges Redis feedback_tracker signals into QueryRun
├── fingerprint.py                                 # Query shape fingerprinting (token-normalized sha1)
├── clustering/
│   ├── __init__.py
│   ├── _shared.py                                 # TfidfVectorizer + KMeans helpers, explanation helpers
│   ├── success_patterns.py                        # Pass 1
│   ├── failure_patterns.py                        # Pass 2
│   ├── artifact_utility.py                        # Pass 3
│   └── persona_effect.py                          # Pass 4
├── report/
│   ├── __init__.py
│   ├── model.py                                   # PatternReport pydantic schema
│   └── renderer.py                                # Jinja2 renderer against analytics/templates/*.md
└── run.py                                         # CLI orchestrator for scripts/mine_sme_patterns.py

scripts/mine_sme_patterns.py                       # [NEW] Thin entry-point re-exports run.main

scripts/evaluate_training_trigger.py               # [NEW] Cross-month stabilization + candidate list

analytics/                                         [NEW]
├── README.md                                       # Monthly review runbook
├── templates/
│   └── sme_patterns_template.md                    # Jinja2 markdown template
└── .gitkeep                                        # Keeps the dir checked-in pre-first-run

systemd/docwain-sme-pattern-mining.service         # [NEW]
systemd/docwain-sme-pattern-mining.timer           # [NEW]
deploy/sme-pattern-mining.sh                       # [NEW] Wrapper invoked by systemd

tests/scripts/sme_patterns/                        [NEW]
├── __init__.py
├── conftest.py
├── fixtures/
│   ├── __init__.py
│   ├── synth_trace_factory.py
│   └── query_trace_factory.py
├── test_trace_loader.py
├── test_feedback_merger.py
├── test_fingerprint.py
├── clustering/
│   ├── __init__.py
│   ├── test_success_patterns.py
│   ├── test_failure_patterns.py
│   ├── test_artifact_utility.py
│   └── test_persona_effect.py
├── report/
│   ├── __init__.py
│   ├── test_model.py
│   └── test_renderer.py
├── test_run.py
├── test_evaluate_training_trigger.py
└── test_monthly_end_to_end.py
```

Each clustering pass is independent — the monthly report keeps rendering even when one pass finds zero clusters.

---

## Task 1: Preflight audit and directory scaffolding

**Files:**
- Create: `scripts/sme_patterns/__init__.py` (empty)
- Create: `scripts/sme_patterns/clustering/__init__.py` (empty)
- Create: `scripts/sme_patterns/report/__init__.py` (empty)
- Create: `tests/scripts/sme_patterns/__init__.py` (empty)
- Create: `tests/scripts/sme_patterns/clustering/__init__.py` (empty)
- Create: `tests/scripts/sme_patterns/report/__init__.py` (empty)
- Create: `tests/scripts/sme_patterns/fixtures/__init__.py` (empty)
- Create: `analytics/.gitkeep` (empty)
- Create: `analytics/templates/` (directory sentinel)
- Audit only: `src/storage/azure_blob_client.py`, `src/intelligence/feedback_tracker.py`, `scripts/sme_eval/*` (Phase 0 conventions), `docs/superpowers/specs/2026-04-20-docwain-profile-sme-reasoning-design.md` Sections 11–13

- [ ] **Step 1: Read the support surface**

Confirm:
- `src/storage/azure_blob_client.py` — `get_document_container_client()` + `download_blob().readall()` for JSONL reads.
- `src/intelligence/feedback_tracker.py` — `get_profile_metrics(profile_id)` is the merge source (aggregate metrics, incl. `low_confidence_count`).
- `scripts/sme_eval/schema.py` (Phase 0) — pydantic conventions this plan follows.

- [ ] **Step 2: Create empty package files and analytics dir**

```bash
mkdir -p scripts/sme_patterns/clustering scripts/sme_patterns/report \
    tests/scripts/sme_patterns/clustering tests/scripts/sme_patterns/report \
    tests/scripts/sme_patterns/fixtures \
    analytics/templates
touch scripts/sme_patterns/__init__.py \
      scripts/sme_patterns/clustering/__init__.py \
      scripts/sme_patterns/report/__init__.py \
      tests/scripts/sme_patterns/__init__.py \
      tests/scripts/sme_patterns/clustering/__init__.py \
      tests/scripts/sme_patterns/report/__init__.py \
      tests/scripts/sme_patterns/fixtures/__init__.py \
      analytics/.gitkeep
```

- [ ] **Step 3: Commit the scaffolding**

```bash
git add -f scripts/sme_patterns/__init__.py scripts/sme_patterns/clustering/__init__.py \
    scripts/sme_patterns/report/__init__.py \
    tests/scripts/sme_patterns/__init__.py tests/scripts/sme_patterns/clustering/__init__.py \
    tests/scripts/sme_patterns/report/__init__.py tests/scripts/sme_patterns/fixtures/__init__.py \
    analytics/.gitkeep analytics/templates
git commit -m "phase6(sme-patterns): scaffold pattern-mining directories"
```

---

## Task 2: Pattern mining schema

Materializes the trace-record contract above as pydantic models (`SynthesisRun`, `QueryRun`, `Cluster`, `PatternReport`, `TrainingCandidate`).

**Files:**
- Create: `scripts/sme_patterns/schema.py`
- Create: `tests/scripts/sme_patterns/test_schema.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/scripts/sme_patterns/test_schema.py`:

```python
"""Tests for pattern-mining schema."""
from datetime import datetime

import pytest
from pydantic import ValidationError

from scripts.sme_patterns.schema import (
    BuilderTrace,
    Cluster,
    ClusterType,
    PatternReport,
    QueryFeedback,
    QueryRun,
    SynthesisRun,
    TrainingCandidate,
    VerifierDrop,
)


def _synth_run_dict():
    return {
        "subscription_id": "sub_finance_1",
        "profile_id": "prof_fin_q1",
        "synthesis_id": "syn_001",
        "started_at": datetime(2026, 4, 1, 2, 0, 0),
        "completed_at": datetime(2026, 4, 1, 2, 14, 30),
        "adapter_version": "1.2.0",
        "adapter_content_hash": "abc123",
        "profile_domain": "finance",
        "per_builder": {
            "dossier": {
                "builder_name": "dossier",
                "items_produced": 20,
                "items_persisted": 18,
                "duration_ms": 4200.0,
                "errors": [],
            }
        },
        "verifier_drops": [
            {
                "item_id": "dossier_5",
                "builder": "dossier",
                "reason_code": "evidence_presence",
                "detail": "no cited chunk",
            }
        ],
    }


def test_synthesis_run_valid():
    r = SynthesisRun(**_synth_run_dict())
    assert r.subscription_id == "sub_finance_1"
    assert r.per_builder["dossier"].items_persisted == 18
    assert r.verifier_drops[0].reason_code == "evidence_presence"


def test_synthesis_run_rejects_missing_field():
    d = _synth_run_dict()
    del d["adapter_version"]
    with pytest.raises(ValidationError):
        SynthesisRun(**d)


def test_query_run_valid():
    q = QueryRun(
        subscription_id="s",
        profile_id="p",
        profile_domain="finance",
        query_id="q_001",
        query_text="analyze Q3 trend",
        query_fingerprint="abc1",
        intent="analyze",
        format_hint=None,
        adapter_version="1.2.0",
        adapter_persona_role="senior financial analyst",
        retrieval_layers={"chunks": 12, "kg": 5, "sme_artifacts": 5, "url": 0},
        pack_tokens=4200,
        reasoner_prompt_hash="hashy",
        response_len_tokens=780,
        citation_verifier_drops=0,
        honest_compact_fallback=False,
        url_present=False,
        url_fetch_ok=None,
        timing_ms={"understand": 40, "retrieval": 210, "reasoner": 8400, "compose": 60, "total": 8710},
        feedback=QueryFeedback(rating=1, edited=False, follow_up_count=0, source="feedback_tracker"),
        captured_at=datetime(2026, 4, 5, 10, 0, 0),
    )
    assert q.feedback.rating == 1
    assert q.retrieval_layers["sme_artifacts"] == 5


def test_query_feedback_rating_range():
    ok = QueryFeedback(rating=-1, edited=False, follow_up_count=0, source="implicit")
    assert ok.rating == -1
    with pytest.raises(ValidationError):
        QueryFeedback(rating=5, edited=False, follow_up_count=0, source="implicit")


def test_cluster_type_covers_all_passes():
    names = {c.value for c in ClusterType}
    assert names == {"success", "failure", "artifact_utility", "persona_effect"}


def test_cluster_roundtrips():
    c = Cluster(
        cluster_id="fail_001",
        cluster_type=ClusterType.FAILURE,
        size=42,
        subscription_ids=["sub_a", "sub_b"],
        primary_intent="recommend",
        profile_domain="finance",
        fingerprint_samples=["abc1", "def2"],
        short_description="Recommendation queries on finance profile hit verifier drops ≥2",
        signal_score=0.72,
        evidence={"avg_verifier_drops": 2.3, "avg_rating": -0.4},
        notes="2026-04 run",
    )
    d = c.model_dump()
    c2 = Cluster(**d)
    assert c2.cluster_id == "fail_001"
    assert c2.size == 42


def test_pattern_report_serializes():
    rep = PatternReport(
        run_id="patterns_2026-04",
        period_start=datetime(2026, 4, 1),
        period_end=datetime(2026, 4, 30),
        num_synth_runs=18,
        num_query_runs=12400,
        successes=[],
        failures=[],
        artifact_utility=[],
        persona_effect=[],
        training_candidates=[],
        rollback_links=["analytics/sme_rollback_2026-04-12.md"],
    )
    assert rep.run_id == "patterns_2026-04"
    assert rep.rollback_links == ["analytics/sme_rollback_2026-04-12.md"]


def test_training_candidate_requires_stabilization_evidence():
    tc = TrainingCandidate(
        candidate_id="tc_001",
        cluster_ids=["fail_001", "fail_001_prev"],
        months_present=2,
        total_volume=84,
        stabilization_score=0.78,
        dominant_intent="recommend",
        dominant_domain="finance",
        short_description="recurring ungrounded recommendation queries on finance",
    )
    assert tc.months_present >= 2  # schema requirement below
```

- [ ] **Step 2: Run the tests and confirm failure**

Run: `pytest tests/scripts/sme_patterns/test_schema.py -v`
Expected: FAIL with `ModuleNotFoundError: scripts.sme_patterns.schema`.

- [ ] **Step 3: Write the schema**

Create `scripts/sme_patterns/schema.py`:

```python
"""Schema for Phase 6 pattern mining.

These models define the Phase 1 → Phase 6 seam. trace_loader.py is the only
module that reads raw JSONL; everything else operates on SynthesisRun / QueryRun.
Changing a field in Phase 1 requires updating exactly this file + trace_loader.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class VerifierDrop(BaseModel):
    item_id: str
    builder: str
    reason_code: str
    detail: str = ""


class BuilderTrace(BaseModel):
    builder_name: str
    items_produced: int = 0
    items_persisted: int = 0
    duration_ms: float | None = None
    errors: list[str] = Field(default_factory=list)


class SynthesisRun(BaseModel):
    subscription_id: str
    profile_id: str
    synthesis_id: str
    started_at: datetime
    completed_at: datetime | None = None
    adapter_version: str
    adapter_content_hash: str
    profile_domain: str
    per_builder: dict[str, BuilderTrace] = Field(default_factory=dict)
    verifier_drops: list[VerifierDrop] = Field(default_factory=list)


class QueryFeedback(BaseModel):
    rating: Literal[-1, 0, 1] | None = None
    edited: bool = False
    follow_up_count: int = 0
    source: str = "implicit"


class QueryRun(BaseModel):
    subscription_id: str
    profile_id: str
    profile_domain: str
    query_id: str
    query_text: str
    query_fingerprint: str
    intent: str
    format_hint: str | None = None
    adapter_version: str
    adapter_persona_role: str = ""
    retrieval_layers: dict[str, int] = Field(default_factory=dict)
    pack_tokens: int = 0
    reasoner_prompt_hash: str = ""
    response_len_tokens: int = 0
    citation_verifier_drops: int = 0
    honest_compact_fallback: bool = False
    url_present: bool = False
    url_fetch_ok: bool | None = None
    timing_ms: dict[str, float] = Field(default_factory=dict)
    feedback: QueryFeedback | None = None
    captured_at: datetime


class ClusterType(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    ARTIFACT_UTILITY = "artifact_utility"
    PERSONA_EFFECT = "persona_effect"


class Cluster(BaseModel):
    cluster_id: str
    cluster_type: ClusterType
    size: int
    subscription_ids: list[str] = Field(default_factory=list)
    primary_intent: str | None = None
    profile_domain: str | None = None
    fingerprint_samples: list[str] = Field(default_factory=list)
    short_description: str
    signal_score: float  # pass-specific interpretation; always in [0.0, 1.0] or a rate
    evidence: dict[str, Any] = Field(default_factory=dict)
    notes: str = ""


class TrainingCandidate(BaseModel):
    candidate_id: str
    cluster_ids: list[str]
    months_present: int
    total_volume: int
    stabilization_score: float
    dominant_intent: str
    dominant_domain: str
    short_description: str

    @model_validator(mode="after")
    def _require_stabilization(self) -> "TrainingCandidate":
        if self.months_present < 2:
            raise ValueError("TrainingCandidate requires months_present >= 2")
        return self


class PatternReport(BaseModel):
    run_id: str
    period_start: datetime
    period_end: datetime
    num_synth_runs: int
    num_query_runs: int
    successes: list[Cluster] = Field(default_factory=list)
    failures: list[Cluster] = Field(default_factory=list)
    artifact_utility: list[Cluster] = Field(default_factory=list)
    persona_effect: list[Cluster] = Field(default_factory=list)
    training_candidates: list[TrainingCandidate] = Field(default_factory=list)
    rollback_links: list[str] = Field(default_factory=list)
```

- [ ] **Step 4: Run tests and confirm pass**

Run: `pytest tests/scripts/sme_patterns/test_schema.py -v`
Expected: PASS for all 7 tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/sme_patterns/schema.py tests/scripts/sme_patterns/test_schema.py
git commit -m "phase6(sme-patterns): pattern-mining pydantic schema"
```

---

## Task 3: Query-shape fingerprint

Used by every clustering pass to group similar query shapes while preserving privacy — tokenize, lowercase, drop stop words, sha1 the remainder (16-hex prefix).

**Files:**
- Create: `scripts/sme_patterns/fingerprint.py`
- Create: `tests/scripts/sme_patterns/test_fingerprint.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/scripts/sme_patterns/test_fingerprint.py`:

```python
"""Tests for query fingerprinting."""
import pytest

from scripts.sme_patterns.fingerprint import (
    fingerprint_query,
    normalize_tokens,
)


def test_normalize_lowercase_and_strip_punct():
    toks = normalize_tokens("What ARE the Q3 Revenue Trends?")
    assert toks == ["q3", "revenue", "trends"]


def test_normalize_drops_stop_words():
    toks = normalize_tokens("the invoice for our vendor is paid")
    assert "the" not in toks
    assert "is" not in toks
    assert "invoice" in toks
    assert "vendor" in toks


def test_normalize_preserves_identifier_patterns():
    toks = normalize_tokens("invoice INV-2026-Q3-0048 status")
    # Preserve doc-ID-like tokens as-is (dashes kept; content lowercased)
    assert "inv-2026-q3-0048" in toks


def test_fingerprint_is_stable_across_whitespace_and_case():
    a = fingerprint_query("Analyze Q3 revenue trend.")
    b = fingerprint_query("  analyze   Q3  REVENUE   trend  ")
    assert a == b


def test_fingerprint_distinguishes_meaningfully_different_queries():
    a = fingerprint_query("Analyze Q3 revenue trend.")
    b = fingerprint_query("Diagnose why gross margin dropped.")
    assert a != b


def test_fingerprint_is_deterministic_sha1_prefix():
    fp = fingerprint_query("test query")
    assert len(fp) == 16  # 16-hex-char prefix of sha1
    assert all(c in "0123456789abcdef" for c in fp)


def test_fingerprint_never_empty_even_for_stopwords_only():
    fp = fingerprint_query("the and of is")
    assert fp != ""
    assert len(fp) == 16
```

- [ ] **Step 2: Run tests and confirm failure**

Run: `pytest tests/scripts/sme_patterns/test_fingerprint.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Write the fingerprint module**

Create `scripts/sme_patterns/fingerprint.py`:

```python
"""Query-shape fingerprinting.

Deliberately simple: normalize tokens, drop stop words, sha1 the remaining
joined form. 16-hex-char prefix is enough collision resistance for cluster
grouping without carrying raw query text into analytics files.
"""
from __future__ import annotations

import hashlib
import re

_WORD = re.compile(r"[A-Za-z0-9_\-]+")

# Intentionally minimal stop list — we want query shapes, not topics.
_STOP: frozenset[str] = frozenset(
    {
        "a", "an", "the", "of", "for", "to", "in", "on", "at", "by",
        "is", "are", "was", "were", "be", "been", "being",
        "do", "does", "did", "done",
        "has", "have", "had",
        "i", "we", "you", "they", "he", "she", "it",
        "me", "us", "them", "him", "her",
        "my", "our", "your", "their", "his",
        "and", "or", "but", "if", "so", "than",
        "that", "this", "these", "those",
        "what", "which", "who", "when", "where", "how", "why",
    }
)


def normalize_tokens(text: str) -> list[str]:
    """Tokenize + lowercase + drop stop words. Preserves alphanumerics and
    dashes/underscores (so identifiers like INV-2026-Q3-0048 survive).
    """
    raw = _WORD.findall(text or "")
    out: list[str] = []
    for tok in raw:
        low = tok.lower()
        if low in _STOP:
            continue
        out.append(low)
    return out


def fingerprint_query(text: str) -> str:
    """Return a 16-hex-char sha1 prefix over normalized tokens.

    Stable across whitespace/case differences. Returns a fingerprint even
    when the input is stop-words-only (using the normalized-empty sentinel)
    so downstream code never sees an empty fingerprint.
    """
    toks = normalize_tokens(text)
    canonical = " ".join(toks) if toks else "__empty_query__"
    h = hashlib.sha1(canonical.encode("utf-8"), usedforsecurity=False).hexdigest()
    return h[:16]
```

- [ ] **Step 4: Run tests and confirm pass**

Run: `pytest tests/scripts/sme_patterns/test_fingerprint.py -v`
Expected: PASS for all 7 tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/sme_patterns/fingerprint.py tests/scripts/sme_patterns/test_fingerprint.py
git commit -m "phase6(sme-patterns): query-shape fingerprint helper"
```

---

## Task 4: Trace loader (Azure Blob JSONL → projections)

**Files:**
- Create: `scripts/sme_patterns/trace_loader.py`
- Create: `tests/scripts/sme_patterns/fixtures/synth_trace_factory.py`
- Create: `tests/scripts/sme_patterns/fixtures/query_trace_factory.py`
- Create: `tests/scripts/sme_patterns/test_trace_loader.py`

- [ ] **Step 1: Write the trace fixtures**

Create `tests/scripts/sme_patterns/fixtures/synth_trace_factory.py`:

```python
"""Build synthetic synthesis-trace JSONL for tests."""
from __future__ import annotations

import json
from datetime import datetime, timedelta


def make_synth_jsonl(
    *,
    synthesis_id: str = "syn_001",
    subscription_id: str = "sub_a",
    profile_id: str = "prof_a",
    profile_domain: str = "finance",
    adapter_version: str = "1.2.0",
    adapter_content_hash: str = "abc123",
    started_at: datetime | None = None,
    duration_s: float = 120.0,
    builders_ok: tuple[str, ...] = ("dossier", "insight_index", "comparative_register",
                                    "kg_materializer", "recommendation_bank"),
    drop_count: int = 0,
) -> str:
    """Produce a multi-line JSONL string conforming to the spec Section 11 schema."""
    started_at = started_at or datetime(2026, 4, 1, 2, 0, 0)
    ended_at = started_at + timedelta(seconds=duration_s)
    lines: list[str] = []

    lines.append(json.dumps({
        "event": "synthesis_started",
        "synthesis_id": synthesis_id,
        "subscription_id": subscription_id,
        "profile_id": profile_id,
        "profile_domain": profile_domain,
        "adapter_version": adapter_version,
        "adapter_content_hash": adapter_content_hash,
        "started_at": started_at.isoformat(),
    }))

    for b in builders_ok:
        lines.append(json.dumps({
            "event": "builder_complete",
            "builder": b,
            "items_produced": 10,
            "items_persisted": 10 - drop_count,
            "duration_ms": 1500.0,
            "errors": [],
        }))

    for i in range(drop_count):
        lines.append(json.dumps({
            "event": "verifier_drop",
            "item_id": f"{builders_ok[0]}_{i}",
            "builder": builders_ok[0],
            "reason_code": "evidence_presence",
            "detail": f"dropped item {i}",
        }))

    lines.append(json.dumps({
        "event": "synthesis_completed",
        "synthesis_id": synthesis_id,
        "completed_at": ended_at.isoformat(),
    }))

    return "\n".join(lines) + "\n"
```

Create `tests/scripts/sme_patterns/fixtures/query_trace_factory.py`:

```python
"""Build synthetic query-trace JSONL for tests."""
from __future__ import annotations

import json
from datetime import datetime


def make_query_jsonl(
    *,
    query_id: str = "q_001",
    subscription_id: str = "sub_a",
    profile_id: str = "prof_a",
    profile_domain: str = "finance",
    query_text: str = "analyze Q3 trend",
    query_fingerprint: str = "abc",
    intent: str = "analyze",
    format_hint: str | None = None,
    adapter_version: str = "1.2.0",
    adapter_persona_role: str = "senior financial analyst",
    sme_artifacts: int = 5,
    citation_verifier_drops: int = 0,
    honest_compact_fallback: bool = False,
    rating: int | None = 1,
    captured_at: datetime | None = None,
) -> str:
    captured_at = captured_at or datetime(2026, 4, 5, 10, 0, 0)
    payload = {
        "event": "query_complete",
        "query_id": query_id,
        "subscription_id": subscription_id,
        "profile_id": profile_id,
        "profile_domain": profile_domain,
        "query_text": query_text,
        "query_fingerprint": query_fingerprint,
        "intent": intent,
        "format_hint": format_hint,
        "adapter_version": adapter_version,
        "adapter_persona_role": adapter_persona_role,
        "retrieval_layers": {"chunks": 12, "kg": 5, "sme_artifacts": sme_artifacts, "url": 0},
        "pack_tokens": 4200,
        "reasoner_prompt_hash": "hashy",
        "response_len_tokens": 780,
        "citation_verifier_drops": citation_verifier_drops,
        "honest_compact_fallback": honest_compact_fallback,
        "url_present": False,
        "url_fetch_ok": None,
        "timing_ms": {"understand": 40, "retrieval": 210, "reasoner": 8400, "compose": 60,
                      "total": 8710},
        "feedback": {"rating": rating, "edited": False, "follow_up_count": 0,
                     "source": "feedback_tracker"} if rating is not None else None,
        "captured_at": captured_at.isoformat(),
    }
    return json.dumps(payload) + "\n"
```

- [ ] **Step 2: Write the failing loader tests**

Create `tests/scripts/sme_patterns/test_trace_loader.py`:

```python
"""Tests for the trace loader.

The loader depends on an injected 'blob reader' (Callable[[str], Iterable[str]])
so tests need not mock azure-storage-blob directly.
"""
from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from scripts.sme_patterns.trace_loader import (
    TraceLoader,
    TraceWindow,
    parse_synth_jsonl,
    parse_query_jsonl,
)
from scripts.sme_patterns.schema import QueryRun, SynthesisRun
from tests.scripts.sme_patterns.fixtures.synth_trace_factory import make_synth_jsonl
from tests.scripts.sme_patterns.fixtures.query_trace_factory import make_query_jsonl


def test_parse_synth_jsonl_happy_path():
    text = make_synth_jsonl(synthesis_id="syn_42", drop_count=2)
    run = parse_synth_jsonl(text)
    assert isinstance(run, SynthesisRun)
    assert run.synthesis_id == "syn_42"
    assert len(run.verifier_drops) == 2
    assert run.per_builder["dossier"].items_persisted == 8


def test_parse_synth_jsonl_handles_no_completed():
    text = make_synth_jsonl(synthesis_id="syn_pending")
    # Simulate a crashed run — remove the completed event
    lines = [ln for ln in text.splitlines() if "synthesis_completed" not in ln]
    run = parse_synth_jsonl("\n".join(lines) + "\n")
    assert run.completed_at is None


def test_parse_query_jsonl_happy_path():
    text = make_query_jsonl(query_id="q_1", rating=-1, citation_verifier_drops=3)
    qs = list(parse_query_jsonl(text))
    assert len(qs) == 1
    assert isinstance(qs[0], QueryRun)
    assert qs[0].feedback.rating == -1
    assert qs[0].citation_verifier_drops == 3


def test_parse_query_jsonl_skips_malformed_lines():
    good = make_query_jsonl(query_id="q_1")
    bad = '{"event": "not_query_complete"}\n'
    malformed = 'this is not json\n'
    qs = list(parse_query_jsonl(good + bad + malformed))
    assert len(qs) == 1
    assert qs[0].query_id == "q_1"


def test_loader_iterates_synth_blobs_in_window():
    blobs = {
        "sme_traces/synthesis/sub_a/prof_a/syn_in.jsonl": make_synth_jsonl(
            synthesis_id="syn_in",
            started_at=datetime(2026, 4, 10, 0, 0, 0),
        ),
        "sme_traces/synthesis/sub_a/prof_a/syn_out.jsonl": make_synth_jsonl(
            synthesis_id="syn_out",
            started_at=datetime(2026, 3, 1, 0, 0, 0),  # outside window
        ),
    }
    list_blobs = MagicMock(return_value=list(blobs.keys()))
    read_blob = MagicMock(side_effect=lambda name: blobs[name])

    loader = TraceLoader(list_blobs=list_blobs, read_blob=read_blob)
    window = TraceWindow(
        start=datetime(2026, 4, 1),
        end=datetime(2026, 4, 30, 23, 59, 59),
    )
    runs = list(loader.iter_synthesis_runs(window))
    assert len(runs) == 1
    assert runs[0].synthesis_id == "syn_in"


def test_loader_filters_query_blobs_by_date_prefix():
    blobs = {
        "sme_traces/queries/sub_a/prof_a/2026-04-05/q1.jsonl": make_query_jsonl(
            query_id="q1",
            captured_at=datetime(2026, 4, 5, 10, 0, 0),
        ),
        "sme_traces/queries/sub_a/prof_a/2026-03-05/q_old.jsonl": make_query_jsonl(
            query_id="q_old",
            captured_at=datetime(2026, 3, 5, 10, 0, 0),
        ),
    }
    list_blobs = MagicMock(return_value=list(blobs.keys()))
    read_blob = MagicMock(side_effect=lambda name: blobs[name])

    loader = TraceLoader(list_blobs=list_blobs, read_blob=read_blob)
    window = TraceWindow(
        start=datetime(2026, 4, 1),
        end=datetime(2026, 4, 30, 23, 59, 59),
    )
    qs = list(loader.iter_query_runs(window))
    assert [q.query_id for q in qs] == ["q1"]


def test_loader_never_raises_on_single_bad_blob():
    blobs = {
        "sme_traces/queries/sub_a/prof_a/2026-04-05/ok.jsonl": make_query_jsonl(query_id="ok"),
        "sme_traces/queries/sub_a/prof_a/2026-04-05/bad.jsonl": "not jsonl",
    }
    list_blobs = MagicMock(return_value=list(blobs.keys()))
    read_blob = MagicMock(side_effect=lambda name: blobs[name])

    loader = TraceLoader(list_blobs=list_blobs, read_blob=read_blob)
    window = TraceWindow(
        start=datetime(2026, 4, 1),
        end=datetime(2026, 4, 30, 23, 59, 59),
    )
    qs = list(loader.iter_query_runs(window))
    # 'ok' survived; 'bad' was skipped silently (loader logs but does not raise)
    assert [q.query_id for q in qs] == ["ok"]
```

- [ ] **Step 3: Run tests and confirm failure**

Run: `pytest tests/scripts/sme_patterns/test_trace_loader.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 4: Write the trace loader**

Create `scripts/sme_patterns/trace_loader.py`:

```python
"""Trace loader for SME synthesis + query JSONL blobs.

Reads the Azure-Blob paths defined in spec Section 9:
  sme_traces/synthesis/{sub}/{prof}/{synthesis_id}.jsonl
  sme_traces/queries/{sub}/{prof}/{YYYY-MM-DD}/{query_id}.jsonl

Depends on two callables injected at construction:
  list_blobs(prefix: str) -> Iterable[str]
  read_blob(name: str) -> str

This indirection keeps the loader testable without Azure SDK mocks. The
production CLI wires these to src/storage/azure_blob_client.py at startup.
"""
from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime

from scripts.sme_patterns.schema import (
    BuilderTrace,
    QueryFeedback,
    QueryRun,
    SynthesisRun,
    VerifierDrop,
)

logger = logging.getLogger(__name__)

_DATE_DIR = re.compile(r"/(\d{4}-\d{2}-\d{2})/")
_SYNTH_PREFIX = "sme_traces/synthesis/"
_QUERY_PREFIX = "sme_traces/queries/"


@dataclass(frozen=True)
class TraceWindow:
    start: datetime
    end: datetime

    def contains(self, ts: datetime) -> bool:
        return self.start <= ts <= self.end


def _parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00").split("+")[0])


def parse_synth_jsonl(text: str) -> SynthesisRun:
    """Assemble a SynthesisRun from a synthesis trace blob's JSONL content."""
    started = None
    completed = None
    subscription_id = profile_id = profile_domain = ""
    synthesis_id = ""
    adapter_version = ""
    adapter_content_hash = ""
    per_builder: dict[str, BuilderTrace] = {}
    drops: list[VerifierDrop] = []

    for raw in text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            ev = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("skipping malformed synth trace line")
            continue
        kind = ev.get("event")
        if kind == "synthesis_started":
            synthesis_id = ev.get("synthesis_id", synthesis_id)
            subscription_id = ev.get("subscription_id", subscription_id)
            profile_id = ev.get("profile_id", profile_id)
            profile_domain = ev.get("profile_domain", profile_domain)
            adapter_version = ev.get("adapter_version", adapter_version)
            adapter_content_hash = ev.get("adapter_content_hash", adapter_content_hash)
            ts = ev.get("started_at")
            if ts:
                started = _parse_iso(ts)
        elif kind == "synthesis_completed":
            ts = ev.get("completed_at")
            if ts:
                completed = _parse_iso(ts)
        elif kind == "builder_complete":
            bn = ev.get("builder", "unknown")
            per_builder[bn] = BuilderTrace(
                builder_name=bn,
                items_produced=int(ev.get("items_produced", 0)),
                items_persisted=int(ev.get("items_persisted", 0)),
                duration_ms=(float(ev["duration_ms"]) if ev.get("duration_ms") is not None else None),
                errors=list(ev.get("errors", []) or []),
            )
        elif kind == "verifier_drop":
            drops.append(
                VerifierDrop(
                    item_id=str(ev.get("item_id", "")),
                    builder=str(ev.get("builder", "")),
                    reason_code=str(ev.get("reason_code", "unknown")),
                    detail=str(ev.get("detail", "")),
                )
            )
        else:
            # Unknown event kinds tolerated — Phase 1 may add new ones.
            continue

    if started is None:
        # Synthesize a floor so downstream window filter still works.
        started = datetime.min
    return SynthesisRun(
        subscription_id=subscription_id,
        profile_id=profile_id,
        synthesis_id=synthesis_id,
        started_at=started,
        completed_at=completed,
        adapter_version=adapter_version,
        adapter_content_hash=adapter_content_hash,
        profile_domain=profile_domain,
        per_builder=per_builder,
        verifier_drops=drops,
    )


def parse_query_jsonl(text: str) -> Iterator[QueryRun]:
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            ev = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("skipping malformed query trace line")
            continue
        if ev.get("event") != "query_complete":
            continue
        try:
            fb_raw = ev.get("feedback")
            feedback = QueryFeedback(**fb_raw) if fb_raw is not None else None
            yield QueryRun(
                subscription_id=str(ev.get("subscription_id", "")),
                profile_id=str(ev.get("profile_id", "")),
                profile_domain=str(ev.get("profile_domain", "")),
                query_id=str(ev.get("query_id", "")),
                query_text=str(ev.get("query_text", "")),
                query_fingerprint=str(ev.get("query_fingerprint", "")),
                intent=str(ev.get("intent", "unknown")),
                format_hint=ev.get("format_hint"),
                adapter_version=str(ev.get("adapter_version", "")),
                adapter_persona_role=str(ev.get("adapter_persona_role", "")),
                retrieval_layers={k: int(v) for k, v in (ev.get("retrieval_layers", {}) or {}).items()},
                pack_tokens=int(ev.get("pack_tokens", 0)),
                reasoner_prompt_hash=str(ev.get("reasoner_prompt_hash", "")),
                response_len_tokens=int(ev.get("response_len_tokens", 0)),
                citation_verifier_drops=int(ev.get("citation_verifier_drops", 0)),
                honest_compact_fallback=bool(ev.get("honest_compact_fallback", False)),
                url_present=bool(ev.get("url_present", False)),
                url_fetch_ok=ev.get("url_fetch_ok"),
                timing_ms={k: float(v) for k, v in (ev.get("timing_ms", {}) or {}).items()},
                feedback=feedback,
                captured_at=_parse_iso(ev.get("captured_at", "1970-01-01T00:00:00")),
            )
        except Exception:
            logger.exception("skipping query event that failed to validate")
            continue


class TraceLoader:
    """Pulls SynthesisRun and QueryRun records from Azure Blob."""

    def __init__(
        self,
        *,
        list_blobs: Callable[[str], Iterable[str]],
        read_blob: Callable[[str], str],
    ) -> None:
        self._list = list_blobs
        self._read = read_blob

    def iter_synthesis_runs(self, window: TraceWindow) -> Iterator[SynthesisRun]:
        for name in self._list(_SYNTH_PREFIX):
            try:
                text = self._read(name)
            except Exception:
                logger.exception("failed to read synth blob %s", name)
                continue
            try:
                run = parse_synth_jsonl(text)
            except Exception:
                logger.exception("failed to parse synth blob %s", name)
                continue
            if run.started_at == datetime.min:
                # No start timestamp discovered — skip.
                continue
            if not window.contains(run.started_at):
                continue
            yield run

    def iter_query_runs(self, window: TraceWindow) -> Iterator[QueryRun]:
        for name in self._list(_QUERY_PREFIX):
            # Fast-reject by date prefix in the path to avoid downloading
            # blobs that are obviously outside the window.
            m = _DATE_DIR.search(name)
            if m:
                try:
                    day = datetime.strptime(m.group(1), "%Y-%m-%d")
                    if day.date() < window.start.date() or day.date() > window.end.date():
                        continue
                except ValueError:
                    pass
            try:
                text = self._read(name)
            except Exception:
                logger.exception("failed to read query blob %s", name)
                continue
            for q in parse_query_jsonl(text):
                if window.contains(q.captured_at):
                    yield q
```

- [ ] **Step 5: Run tests and confirm pass**

Run: `pytest tests/scripts/sme_patterns/test_trace_loader.py -v`
Expected: PASS for all 7 tests.

- [ ] **Step 6: Commit**

```bash
git add scripts/sme_patterns/trace_loader.py \
    tests/scripts/sme_patterns/fixtures/synth_trace_factory.py \
    tests/scripts/sme_patterns/fixtures/query_trace_factory.py \
    tests/scripts/sme_patterns/test_trace_loader.py
git commit -m "phase6(sme-patterns): Blob JSONL trace loader + fixtures"
```

---

## Task 5: Feedback merger (Redis → QueryRun)

When a QueryRun has no feedback, consult the existing Redis `FeedbackTracker` aggregates and attach an implicit signal if the profile is running high on low-confidence events.

**Files:**
- Create: `scripts/sme_patterns/feedback_merger.py`
- Create: `tests/scripts/sme_patterns/test_feedback_merger.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/scripts/sme_patterns/test_feedback_merger.py`:

```python
"""Tests for feedback merger."""
from datetime import datetime
from unittest.mock import MagicMock

from scripts.sme_patterns.feedback_merger import merge_feedback
from scripts.sme_patterns.schema import QueryFeedback, QueryRun


def _run(qid, *, feedback=None):
    return QueryRun(
        subscription_id="s",
        profile_id="p",
        profile_domain="finance",
        query_id=qid,
        query_text="analyze Q3",
        query_fingerprint="abc",
        intent="analyze",
        adapter_version="1.0.0",
        captured_at=datetime(2026, 4, 5, 10, 0, 0),
        feedback=feedback,
    )


def test_merge_preserves_explicit_feedback():
    runs = [_run("q1", feedback=QueryFeedback(rating=1, source="feedback_tracker"))]
    tracker = MagicMock()
    out = merge_feedback(runs, tracker)
    assert out[0].feedback.rating == 1
    tracker.get_profile_metrics.assert_not_called()


def test_merge_fills_missing_feedback_from_redis_low_confidence_list():
    runs = [
        _run("q1"),                           # missing feedback
        _run("q2"),                           # missing feedback
    ]
    tracker = MagicMock()
    tracker.get_profile_metrics.return_value = {
        "total_queries": 2,
        "avg_confidence": 0.4,
        "grounded_ratio": 0.5,
        "low_confidence_count": 1,
    }
    out = merge_feedback(runs, tracker, implicit_rating_when_low=-1)
    # Both runs get an implicit feedback because the profile's low_confidence_ratio > 0
    for q in out:
        assert q.feedback is not None
        assert q.feedback.source == "implicit"
        assert q.feedback.rating == -1


def test_merge_no_metrics_leaves_feedback_none():
    runs = [_run("q1")]
    tracker = MagicMock()
    tracker.get_profile_metrics.return_value = {
        "total_queries": 0,
        "avg_confidence": 0.0,
        "grounded_ratio": 0.0,
        "low_confidence_count": 0,
    }
    out = merge_feedback(runs, tracker)
    assert out[0].feedback is None


def test_merge_never_raises_when_tracker_errors():
    runs = [_run("q1")]
    tracker = MagicMock()
    tracker.get_profile_metrics.side_effect = RuntimeError("redis down")
    out = merge_feedback(runs, tracker)
    # Fall back to None feedback — do not block the whole mining run
    assert out[0].feedback is None
```

- [ ] **Step 2: Run tests and confirm failure**

Run: `pytest tests/scripts/sme_patterns/test_feedback_merger.py -v`
Expected: FAIL.

- [ ] **Step 3: Write the merger**

Create `scripts/sme_patterns/feedback_merger.py`:

```python
"""Merge feedback signals into QueryRuns.

Preference:
  1. Explicit feedback already on the QueryRun (set by Phase 1 query-trace writer).
  2. Redis FeedbackTracker aggregates — only used when (a) QueryRun has no
     feedback and (b) the profile's current low-confidence ratio indicates
     the Reasoner had trouble. This is an honest fallback signal; the
     monthly report labels all merged feedback 'implicit' so downstream
     analysis can weight it accordingly.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Iterable

from scripts.sme_patterns.schema import QueryFeedback, QueryRun

logger = logging.getLogger(__name__)


def merge_feedback(
    runs: Iterable[QueryRun],
    feedback_tracker,
    *,
    implicit_rating_when_low: int = -1,
    low_confidence_threshold: float = 0.3,
) -> list[QueryRun]:
    """Return a new list of QueryRuns with feedback filled where missing."""
    runs = list(runs)

    # Group runs lacking feedback by profile for a single tracker call each.
    missing_by_profile: dict[str, list[int]] = defaultdict(list)
    for idx, r in enumerate(runs):
        if r.feedback is None:
            missing_by_profile[r.profile_id].append(idx)

    if not missing_by_profile:
        return runs

    out = list(runs)
    for profile_id, idxs in missing_by_profile.items():
        try:
            metrics = feedback_tracker.get_profile_metrics(profile_id) or {}
        except Exception:
            logger.exception("feedback_tracker raised for profile %s; leaving implicit None",
                             profile_id)
            continue
        total = int(metrics.get("total_queries", 0) or 0)
        low_count = int(metrics.get("low_confidence_count", 0) or 0)
        ratio = (low_count / total) if total > 0 else 0.0
        if ratio <= low_confidence_threshold:
            continue
        implicit = QueryFeedback(
            rating=implicit_rating_when_low,
            edited=False,
            follow_up_count=0,
            source="implicit",
        )
        for i in idxs:
            out[i] = out[i].model_copy(update={"feedback": implicit})
    return out
```

- [ ] **Step 4: Run tests and confirm pass**

Run: `pytest tests/scripts/sme_patterns/test_feedback_merger.py -v`
Expected: PASS for all 4 tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/sme_patterns/feedback_merger.py tests/scripts/sme_patterns/test_feedback_merger.py
git commit -m "phase6(sme-patterns): Redis feedback merger for QueryRuns"
```

---

## Task 6: Clustering shared helpers (tf-idf + k-means wrapper)

Thin explainable wrapper around `TfidfVectorizer` + `KMeans`. Clusters carry top-k feature terms so the monthly reviewer can read each cluster in one glance.

**Files:**
- Create: `scripts/sme_patterns/clustering/_shared.py`
- Create: `tests/scripts/sme_patterns/clustering/test_shared.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/scripts/sme_patterns/clustering/test_shared.py`:

```python
"""Tests for clustering shared helpers."""
import pytest

from scripts.sme_patterns.clustering._shared import (
    TextCluster,
    choose_k,
    cluster_texts,
    summarize_cluster_terms,
)


def test_choose_k_small_sample():
    # 3 docs — choose_k should stay small (<= ~ sqrt(n))
    assert choose_k(3) == 2
    assert choose_k(1) == 1
    assert choose_k(0) == 1


def test_choose_k_scales_with_sample_size():
    assert choose_k(50) >= 3
    assert choose_k(500) <= 20  # bounded


def test_cluster_texts_groups_obviously_similar_strings():
    texts = [
        "analyze Q3 revenue trend",
        "analyze Q3 revenue pattern",
        "diagnose login error symptom",
        "diagnose login error cause",
        "recommend cost reduction plan",
    ]
    clusters = cluster_texts(texts, k=3)
    assert len(clusters) == 3
    # Every text belongs to exactly one cluster
    all_idxs = sorted([i for c in clusters for i in c.member_indexes])
    assert all_idxs == list(range(len(texts)))
    # The two "diagnose login error..." strings end up together
    diag_idxs = {2, 3}
    assert any(diag_idxs.issubset(set(c.member_indexes)) for c in clusters)


def test_cluster_texts_empty_returns_empty():
    assert cluster_texts([], k=3) == []


def test_cluster_texts_fewer_than_k_returns_one_per():
    texts = ["single input"]
    clusters = cluster_texts(texts, k=5)
    assert len(clusters) == 1
    assert clusters[0].member_indexes == [0]


def test_summarize_cluster_terms_returns_top_features():
    texts = [
        "analyze Q3 revenue trend",
        "analyze Q3 revenue pattern",
    ]
    clusters = cluster_texts(texts, k=1)
    summary = summarize_cluster_terms(clusters[0], top_n=3)
    assert isinstance(summary, list)
    assert all(isinstance(s, str) for s in summary)
    # At least one of the canonical tokens is present.
    assert any(t in {"analyze", "q3", "revenue"} for t in summary)
```

- [ ] **Step 2: Run tests and confirm failure**

Run: `pytest tests/scripts/sme_patterns/clustering/test_shared.py -v`
Expected: FAIL.

- [ ] **Step 3: Write the helpers**

Create `scripts/sme_patterns/clustering/_shared.py`:

```python
"""Shared clustering primitives.

Simple, explainable, rule-based + tf-idf + k-means. No black-box models.
Each cluster carries its member indexes and top tf-idf terms so the monthly
report can explain the cluster in one sentence.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class TextCluster:
    centroid_index: int
    member_indexes: list[int] = field(default_factory=list)
    terms: list[str] = field(default_factory=list)
    vectorizer: TfidfVectorizer | None = None
    cluster_vector: np.ndarray | None = None


def choose_k(n_samples: int) -> int:
    """Heuristic: k = min(20, max(1, round(sqrt(n / 3))))."""
    if n_samples <= 1:
        return 1
    raw = round(math.sqrt(n_samples / 3.0))
    return max(2 if n_samples > 1 else 1, min(20, raw))


def cluster_texts(texts: list[str], *, k: int | None = None) -> list[TextCluster]:
    """Cluster text inputs into k groups; preserves original index ordering."""
    if not texts:
        return []

    chosen_k = k or choose_k(len(texts))
    chosen_k = max(1, min(chosen_k, len(texts)))

    vectorizer = TfidfVectorizer(
        stop_words="english",
        token_pattern=r"(?u)\b[A-Za-z0-9_\-]{2,}\b",
        max_features=5000,
    )
    X = vectorizer.fit_transform(texts)

    if chosen_k == 1:
        cluster = TextCluster(
            centroid_index=0,
            member_indexes=list(range(len(texts))),
            vectorizer=vectorizer,
            cluster_vector=np.asarray(X.mean(axis=0)).ravel(),
        )
        cluster.terms = _top_terms_for_vector(cluster.cluster_vector, vectorizer, top_n=5)
        return [cluster]

    km = KMeans(n_clusters=chosen_k, random_state=42, n_init="auto")
    labels = km.fit_predict(X)

    out: list[TextCluster] = []
    for cid in range(chosen_k):
        members = [i for i, lab in enumerate(labels) if lab == cid]
        if not members:
            continue
        centroid = km.cluster_centers_[cid]
        cluster = TextCluster(
            centroid_index=cid,
            member_indexes=members,
            vectorizer=vectorizer,
            cluster_vector=centroid,
        )
        cluster.terms = _top_terms_for_vector(centroid, vectorizer, top_n=5)
        out.append(cluster)
    return out


def _top_terms_for_vector(vec: np.ndarray, vectorizer: TfidfVectorizer, *, top_n: int) -> list[str]:
    vec = np.asarray(vec).ravel()
    if vec.size == 0:
        return []
    top_idx = np.argsort(-vec)[: top_n * 2]
    try:
        feature_names = vectorizer.get_feature_names_out()
    except AttributeError:
        feature_names = np.array(vectorizer.get_feature_names())
    terms: list[str] = []
    for i in top_idx:
        if i < len(feature_names):
            terms.append(str(feature_names[i]))
        if len(terms) >= top_n:
            break
    return terms


def summarize_cluster_terms(cluster: TextCluster, *, top_n: int = 5) -> list[str]:
    if not cluster.terms:
        if cluster.vectorizer is not None and cluster.cluster_vector is not None:
            return _top_terms_for_vector(cluster.cluster_vector, cluster.vectorizer, top_n=top_n)
        return []
    return cluster.terms[:top_n]
```

- [ ] **Step 4: Run tests and confirm pass**

Run: `pytest tests/scripts/sme_patterns/clustering/test_shared.py -v`
Expected: PASS for all 6 tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/sme_patterns/clustering/_shared.py \
    tests/scripts/sme_patterns/clustering/test_shared.py
git commit -m "phase6(sme-patterns): explainable tf-idf + k-means helpers"
```

---

## Task 7: Pass 1 — Success patterns clustering

**Success criterion:** explicit `rating == +1` OR (no rating AND `citation_verifier_drops == 0` AND not `honest_compact_fallback`); AND `retrieval_layers.sme_artifacts >= 1` (proves reasoning-layer value); AND intent in `{analyze, diagnose, recommend, investigate, compare, summarize}`. Group passing queries by `(profile_domain, intent)`; cluster texts; emit top-N clusters sorted by size.

**Files:**
- Create: `scripts/sme_patterns/clustering/success_patterns.py`
- Create: `tests/scripts/sme_patterns/clustering/test_success_patterns.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/scripts/sme_patterns/clustering/test_success_patterns.py`:

```python
"""Tests for success-pattern clustering."""
from datetime import datetime

from scripts.sme_patterns.clustering.success_patterns import (
    SuccessPatternsConfig,
    cluster_success_patterns,
    is_success_query,
)
from scripts.sme_patterns.schema import ClusterType, QueryFeedback, QueryRun


def _run(qid, *, domain="finance", intent="analyze", rating=1, sme=5, drops=0,
         honest_fallback=False, text="analyze Q3 revenue trend"):
    fb = QueryFeedback(rating=rating, source="feedback_tracker") if rating is not None else None
    return QueryRun(
        subscription_id="s",
        profile_id="p",
        profile_domain=domain,
        query_id=qid,
        query_text=text,
        query_fingerprint=qid,
        intent=intent,
        adapter_version="1.0.0",
        adapter_persona_role="role",
        retrieval_layers={"chunks": 12, "kg": 5, "sme_artifacts": sme, "url": 0},
        pack_tokens=4200,
        citation_verifier_drops=drops,
        honest_compact_fallback=honest_fallback,
        captured_at=datetime(2026, 4, 5, 10, 0, 0),
        feedback=fb,
    )


def test_is_success_requires_analytical_intent():
    assert is_success_query(_run("q", intent="analyze", rating=1))
    assert not is_success_query(_run("q", intent="lookup", rating=1))


def test_is_success_requires_sme_artifacts_contribution():
    assert not is_success_query(_run("q", intent="analyze", rating=1, sme=0))
    assert is_success_query(_run("q", intent="analyze", rating=1, sme=1))


def test_is_success_requires_clean_citations_when_no_rating():
    # No feedback rating, but clean: counts as success
    assert is_success_query(_run("q", intent="analyze", rating=None, drops=0,
                                  honest_fallback=False, sme=3))
    # No rating, verifier drops > 0: not success
    assert not is_success_query(_run("q", intent="analyze", rating=None, drops=2, sme=3))


def test_cluster_success_patterns_returns_top_clusters_sorted_by_size():
    runs = [
        _run(f"q_rev_{i}", text="analyze Q3 revenue trend growth") for i in range(6)
    ] + [
        _run(f"q_cost_{i}", text="analyze cost structure breakdown") for i in range(3)
    ] + [
        _run("q_rec", intent="recommend", text="recommend SaaS consolidation plan"),
    ]

    config = SuccessPatternsConfig(top_n=5)
    clusters = cluster_success_patterns(runs, config)

    assert clusters, "expected at least one cluster"
    assert all(c.cluster_type == ClusterType.SUCCESS for c in clusters)
    assert len(clusters) <= 5
    # Sorted size descending
    sizes = [c.size for c in clusters]
    assert sizes == sorted(sizes, reverse=True)
    # Largest cluster should contain multiple revenue-trend members
    assert clusters[0].size >= 3


def test_cluster_success_patterns_excludes_non_success():
    runs = [
        _run("bad", intent="analyze", rating=-1, drops=3),
        _run("ok", intent="analyze", rating=1, sme=3),
    ]
    config = SuccessPatternsConfig(top_n=5)
    clusters = cluster_success_patterns(runs, config)
    total_members = sum(c.size for c in clusters)
    assert total_members == 1


def test_cluster_success_patterns_empty_input():
    clusters = cluster_success_patterns([], SuccessPatternsConfig(top_n=5))
    assert clusters == []
```

- [ ] **Step 2: Run tests and confirm failure**

Run: `pytest tests/scripts/sme_patterns/clustering/test_success_patterns.py -v`
Expected: FAIL.

- [ ] **Step 3: Write the success-patterns pass**

Create `scripts/sme_patterns/clustering/success_patterns.py`:

```python
"""Pass 1 — Success-pattern clustering.

A 'success' query is a high-signal query where the SME reasoning layer
demonstrably helped: clean citations, SME artifacts contributed, either
explicit thumbs-up or no negative implicit signal. We cluster these to
answer "what kinds of queries are we winning on?".
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from scripts.sme_patterns.clustering._shared import cluster_texts
from scripts.sme_patterns.schema import Cluster, ClusterType, QueryRun

_ANALYTICAL_INTENTS: frozenset[str] = frozenset(
    {"analyze", "diagnose", "recommend", "investigate", "compare", "summarize"}
)


@dataclass(frozen=True)
class SuccessPatternsConfig:
    top_n: int = 10
    k_per_group: int | None = None  # None → choose_k()


def is_success_query(q: QueryRun) -> bool:
    if q.intent not in _ANALYTICAL_INTENTS:
        return False
    if q.retrieval_layers.get("sme_artifacts", 0) < 1:
        return False
    if q.feedback and q.feedback.rating == 1:
        # Explicit thumbs-up beats all other checks.
        return True
    if q.feedback and q.feedback.rating == -1:
        return False
    # No explicit rating — require clean citations + no fallback
    if q.citation_verifier_drops > 0:
        return False
    if q.honest_compact_fallback:
        return False
    return True


def cluster_success_patterns(
    runs: Iterable[QueryRun],
    config: SuccessPatternsConfig,
) -> list[Cluster]:
    eligible = [q for q in runs if is_success_query(q)]
    if not eligible:
        return []

    grouped: dict[tuple[str, str], list[QueryRun]] = defaultdict(list)
    for q in eligible:
        grouped[(q.profile_domain, q.intent)].append(q)

    produced: list[Cluster] = []
    for (domain, intent), group in grouped.items():
        texts = [q.query_text for q in group]
        text_clusters = cluster_texts(texts, k=config.k_per_group)
        for tc in text_clusters:
            member_runs = [group[i] for i in tc.member_indexes]
            subs = sorted({r.subscription_id for r in member_runs})
            fps = sorted({r.query_fingerprint for r in member_runs})[:5]
            avg_artifacts = _avg(r.retrieval_layers.get("sme_artifacts", 0) for r in member_runs)
            pos_rate = _avg(1.0 if (r.feedback and r.feedback.rating == 1) else 0.0 for r in member_runs)
            short = (
                f"Successful {intent} queries on {domain} — top terms: "
                f"{', '.join(tc.terms[:3]) or 'n/a'}"
            )
            produced.append(
                Cluster(
                    cluster_id=f"succ_{domain}_{intent}_{tc.centroid_index}",
                    cluster_type=ClusterType.SUCCESS,
                    size=len(member_runs),
                    subscription_ids=subs,
                    primary_intent=intent,
                    profile_domain=domain,
                    fingerprint_samples=fps,
                    short_description=short,
                    signal_score=pos_rate,
                    evidence={
                        "avg_sme_artifacts": round(avg_artifacts, 2),
                        "explicit_thumbs_up_rate": round(pos_rate, 2),
                        "top_terms": tc.terms,
                    },
                )
            )

    produced.sort(key=lambda c: c.size, reverse=True)
    return produced[: config.top_n]


def _avg(values: Iterable[float]) -> float:
    vs = list(values)
    return (sum(vs) / len(vs)) if vs else 0.0
```

- [ ] **Step 4: Run tests and confirm pass**

Run: `pytest tests/scripts/sme_patterns/clustering/test_success_patterns.py -v`
Expected: PASS for all 6 tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/sme_patterns/clustering/success_patterns.py \
    tests/scripts/sme_patterns/clustering/test_success_patterns.py
git commit -m "phase6(sme-patterns): Pass 1 — success pattern clustering"
```

---

## Task 8: Pass 2 — Failure patterns clustering

**Failure criterion** (any holds): `rating == -1`; `citation_verifier_drops > 0`; `honest_compact_fallback == True`; or recurring fingerprint (same `query_fingerprint` ≥3× with net-negative aggregate rating). Group by `(profile_domain, intent)`; cluster texts; sort by size × severity.

**Files:**
- Create: `scripts/sme_patterns/clustering/failure_patterns.py`
- Create: `tests/scripts/sme_patterns/clustering/test_failure_patterns.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/scripts/sme_patterns/clustering/test_failure_patterns.py`:

```python
"""Tests for failure-pattern clustering."""
from datetime import datetime

from scripts.sme_patterns.clustering.failure_patterns import (
    FailurePatternsConfig,
    cluster_failure_patterns,
    is_failure_query,
)
from scripts.sme_patterns.schema import ClusterType, QueryFeedback, QueryRun


def _run(qid, *, domain="finance", intent="recommend", rating=None, sme=1, drops=0,
         honest_fallback=False, text="recommend cost cutting actions",
         fingerprint=None):
    fb = QueryFeedback(rating=rating, source="feedback_tracker") if rating is not None else None
    return QueryRun(
        subscription_id="s",
        profile_id="p",
        profile_domain=domain,
        query_id=qid,
        query_text=text,
        query_fingerprint=fingerprint or qid,
        intent=intent,
        adapter_version="1.0.0",
        adapter_persona_role="role",
        retrieval_layers={"chunks": 12, "kg": 5, "sme_artifacts": sme, "url": 0},
        pack_tokens=4200,
        citation_verifier_drops=drops,
        honest_compact_fallback=honest_fallback,
        captured_at=datetime(2026, 4, 5, 10, 0, 0),
        feedback=fb,
    )


def test_is_failure_catches_explicit_thumbs_down():
    assert is_failure_query(_run("q", rating=-1))


def test_is_failure_catches_verifier_drops():
    assert is_failure_query(_run("q", rating=None, drops=2))


def test_is_failure_catches_honest_fallback():
    assert is_failure_query(_run("q", rating=None, honest_fallback=True))


def test_is_failure_passes_clean_positive():
    assert not is_failure_query(_run("q", rating=1, drops=0, honest_fallback=False))


def test_cluster_failure_returns_sorted_by_size_and_severity():
    runs = [
        _run(f"bad_rec_{i}", rating=-1,
             text="recommend SaaS consolidation plan") for i in range(5)
    ] + [
        _run(f"drop_{i}", rating=None, drops=3,
             text="diagnose login authentication failure") for i in range(2)
    ] + [
        _run("good", rating=1),
    ]
    clusters = cluster_failure_patterns(runs, FailurePatternsConfig(top_n=10))
    assert clusters, "expected at least one failure cluster"
    assert all(c.cluster_type == ClusterType.FAILURE for c in clusters)
    # Every member was a failure
    total = sum(c.size for c in clusters)
    assert total == 7  # 5 bad_rec + 2 drop_
    # Sorted size descending
    sizes = [c.size for c in clusters]
    assert sizes == sorted(sizes, reverse=True)


def test_recurring_fingerprint_with_net_negative_counts_as_failure():
    # 3 queries with same fingerprint; net-negative aggregate
    runs = [
        _run("r1", rating=-1, fingerprint="fp_recurr",
             text="why is authentication broken"),
        _run("r2", rating=0, fingerprint="fp_recurr",
             text="why is authentication broken"),
        _run("r3", rating=-1, fingerprint="fp_recurr",
             text="why is authentication broken"),
    ]
    clusters = cluster_failure_patterns(runs, FailurePatternsConfig(top_n=5))
    assert sum(c.size for c in clusters) == 3


def test_empty_input():
    assert cluster_failure_patterns([], FailurePatternsConfig()) == []
```

- [ ] **Step 2: Run tests and confirm failure**

Run: `pytest tests/scripts/sme_patterns/clustering/test_failure_patterns.py -v`
Expected: FAIL.

- [ ] **Step 3: Write the failure-patterns pass**

Create `scripts/sme_patterns/clustering/failure_patterns.py`:

```python
"""Pass 2 — Failure-pattern clustering.

A 'failure' query is any where the SME reasoning stack visibly stumbled:
explicit thumbs-down, citation-verifier drops, honest-compact fallback, or
a recurring fingerprint with net-negative aggregate rating.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from scripts.sme_patterns.clustering._shared import cluster_texts
from scripts.sme_patterns.schema import Cluster, ClusterType, QueryRun


@dataclass(frozen=True)
class FailurePatternsConfig:
    top_n: int = 10
    k_per_group: int | None = None
    recurring_min: int = 3
    recurring_net_neg_threshold: float = 0.0  # avg rating <= this is "net-negative"


def is_failure_query(q: QueryRun) -> bool:
    if q.feedback and q.feedback.rating == -1:
        return True
    if q.citation_verifier_drops > 0:
        return True
    if q.honest_compact_fallback:
        return True
    return False


def _recurring_bad_fingerprints(runs: list[QueryRun], cfg: FailurePatternsConfig) -> set[str]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for r in runs:
        if r.feedback and r.feedback.rating is not None:
            buckets[r.query_fingerprint].append(float(r.feedback.rating))
    bad: set[str] = set()
    for fp, ratings in buckets.items():
        if len(ratings) < cfg.recurring_min:
            continue
        if (sum(ratings) / len(ratings)) <= cfg.recurring_net_neg_threshold:
            bad.add(fp)
    return bad


def cluster_failure_patterns(
    runs: Iterable[QueryRun],
    config: FailurePatternsConfig,
) -> list[Cluster]:
    runs = list(runs)
    if not runs:
        return []

    recurring = _recurring_bad_fingerprints(runs, config)
    eligible = [r for r in runs if is_failure_query(r) or r.query_fingerprint in recurring]
    if not eligible:
        return []

    grouped: dict[tuple[str, str], list[QueryRun]] = defaultdict(list)
    for q in eligible:
        grouped[(q.profile_domain, q.intent)].append(q)

    produced: list[Cluster] = []
    for (domain, intent), group in grouped.items():
        texts = [q.query_text for q in group]
        text_clusters = cluster_texts(texts, k=config.k_per_group)
        for tc in text_clusters:
            member_runs = [group[i] for i in tc.member_indexes]
            subs = sorted({r.subscription_id for r in member_runs})
            fp_counter = Counter(r.query_fingerprint for r in member_runs)
            fps = [fp for fp, _ in fp_counter.most_common(5)]
            avg_drops = _avg(r.citation_verifier_drops for r in member_runs)
            fallback_rate = _avg(1.0 if r.honest_compact_fallback else 0.0 for r in member_runs)
            neg_rate = _avg(
                1.0 if (r.feedback and r.feedback.rating == -1) else 0.0 for r in member_runs
            )
            severity = 0.4 * neg_rate + 0.3 * min(1.0, avg_drops / 3.0) + 0.3 * fallback_rate

            short = (
                f"Failing {intent} queries on {domain} — drops≈{avg_drops:.1f}, "
                f"neg-rate={neg_rate:.0%}; top terms: {', '.join(tc.terms[:3]) or 'n/a'}"
            )
            produced.append(
                Cluster(
                    cluster_id=f"fail_{domain}_{intent}_{tc.centroid_index}",
                    cluster_type=ClusterType.FAILURE,
                    size=len(member_runs),
                    subscription_ids=subs,
                    primary_intent=intent,
                    profile_domain=domain,
                    fingerprint_samples=fps,
                    short_description=short,
                    signal_score=round(min(1.0, severity), 3),
                    evidence={
                        "avg_verifier_drops": round(avg_drops, 2),
                        "honest_compact_fallback_rate": round(fallback_rate, 2),
                        "thumbs_down_rate": round(neg_rate, 2),
                        "recurring_fingerprints": list(recurring & set(fps)),
                        "top_terms": tc.terms,
                    },
                )
            )

    produced.sort(key=lambda c: (c.size, c.signal_score), reverse=True)
    return produced[: config.top_n]


def _avg(values: Iterable[float]) -> float:
    vs = [float(v) for v in values]
    return (sum(vs) / len(vs)) if vs else 0.0
```

- [ ] **Step 4: Run tests and confirm pass**

Run: `pytest tests/scripts/sme_patterns/clustering/test_failure_patterns.py -v`
Expected: PASS for all 7 tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/sme_patterns/clustering/failure_patterns.py \
    tests/scripts/sme_patterns/clustering/test_failure_patterns.py
git commit -m "phase6(sme-patterns): Pass 2 — failure pattern clustering"
```

---

## Task 9: Pass 3 — Artifact utility ranking

Per retrieval layer (chunks / kg / sme_artifacts / url), compute `retrieval_rate` (fraction of queries pulling ≥1 item), `citation_rate` (in those queries, fraction with `citation_verifier_drops == 0` AND non-negative feedback), and `dead_weight_flag` (`retrieval_rate >= 0.5 AND citation_rate < 0.25`). One `Cluster` per layer with `cluster_type == ARTIFACT_UTILITY`.

**Files:**
- Create: `scripts/sme_patterns/clustering/artifact_utility.py`
- Create: `tests/scripts/sme_patterns/clustering/test_artifact_utility.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/scripts/sme_patterns/clustering/test_artifact_utility.py`:

```python
"""Tests for artifact-utility analysis."""
from datetime import datetime

from scripts.sme_patterns.clustering.artifact_utility import (
    ArtifactUtilityConfig,
    analyze_artifact_utility,
)
from scripts.sme_patterns.schema import ClusterType, QueryFeedback, QueryRun


def _run(qid, layers, *, rating=1, drops=0):
    fb = QueryFeedback(rating=rating, source="feedback_tracker") if rating is not None else None
    return QueryRun(
        subscription_id="s",
        profile_id="p",
        profile_domain="finance",
        query_id=qid,
        query_text="q",
        query_fingerprint=qid,
        intent="analyze",
        adapter_version="1.0.0",
        retrieval_layers=layers,
        citation_verifier_drops=drops,
        captured_at=datetime(2026, 4, 5, 10, 0, 0),
        feedback=fb,
    )


def test_every_layer_is_emitted():
    runs = [_run("q1", {"chunks": 5, "kg": 3, "sme_artifacts": 2, "url": 0})]
    clusters = analyze_artifact_utility(runs, ArtifactUtilityConfig())
    assert {c.cluster_id for c in clusters} == {
        "artifact_chunks", "artifact_kg", "artifact_sme_artifacts", "artifact_url",
    }
    assert all(c.cluster_type == ClusterType.ARTIFACT_UTILITY for c in clusters)


def test_retrieval_rate_computed_correctly():
    runs = [
        _run("q1", {"chunks": 0, "kg": 1, "sme_artifacts": 2, "url": 0}),
        _run("q2", {"chunks": 5, "kg": 0, "sme_artifacts": 2, "url": 0}),
        _run("q3", {"chunks": 5, "kg": 0, "sme_artifacts": 0, "url": 0}),
        _run("q4", {"chunks": 5, "kg": 0, "sme_artifacts": 0, "url": 0}),
    ]
    clusters = analyze_artifact_utility(runs, ArtifactUtilityConfig())
    by_id = {c.cluster_id: c for c in clusters}
    # chunks used in 3/4
    assert by_id["artifact_chunks"].evidence["retrieval_rate"] == 0.75
    assert by_id["artifact_kg"].evidence["retrieval_rate"] == 0.25
    # sme_artifacts used in 2/4
    assert by_id["artifact_sme_artifacts"].evidence["retrieval_rate"] == 0.5
    # url never used
    assert by_id["artifact_url"].evidence["retrieval_rate"] == 0.0


def test_dead_weight_flag_triggers_when_high_retrieval_low_citation():
    runs = [
        _run("q1", {"chunks": 5, "kg": 0, "sme_artifacts": 3, "url": 0}, rating=-1, drops=2),
        _run("q2", {"chunks": 5, "kg": 0, "sme_artifacts": 3, "url": 0}, rating=-1, drops=1),
        _run("q3", {"chunks": 5, "kg": 0, "sme_artifacts": 3, "url": 0}, rating=-1, drops=2),
    ]
    clusters = analyze_artifact_utility(runs, ArtifactUtilityConfig())
    by_id = {c.cluster_id: c for c in clusters}
    assert by_id["artifact_sme_artifacts"].evidence["dead_weight_flag"] is True


def test_empty_input_returns_zero_rows_for_each_layer():
    clusters = analyze_artifact_utility([], ArtifactUtilityConfig())
    # Emits every layer even with zero data, rate=0, size=0
    assert len(clusters) == 4
    assert all(c.size == 0 for c in clusters)
    assert all(c.evidence["retrieval_rate"] == 0.0 for c in clusters)
```

- [ ] **Step 2: Run tests and confirm failure**

Run: `pytest tests/scripts/sme_patterns/clustering/test_artifact_utility.py -v`
Expected: FAIL.

- [ ] **Step 3: Write the artifact-utility pass**

Create `scripts/sme_patterns/clustering/artifact_utility.py`:

```python
"""Pass 3 — Artifact-utility ranking.

For each retrieval layer, compute retrieval_rate (how often it fires) and
citation_rate (proxy — share of positive-outcome queries among those that
pulled from the layer). Dead-weight layers (high retrieval + low citation)
are flagged for review.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from scripts.sme_patterns.schema import Cluster, ClusterType, QueryRun

_LAYERS: tuple[str, ...] = ("chunks", "kg", "sme_artifacts", "url")


@dataclass(frozen=True)
class ArtifactUtilityConfig:
    dead_weight_retrieval_min: float = 0.5
    dead_weight_citation_max: float = 0.25


def _positive_outcome(q: QueryRun) -> bool:
    if q.citation_verifier_drops > 0:
        return False
    if q.feedback and q.feedback.rating == -1:
        return False
    return True


def analyze_artifact_utility(
    runs: Iterable[QueryRun],
    config: ArtifactUtilityConfig,
) -> list[Cluster]:
    runs = list(runs)
    total = len(runs)

    out: list[Cluster] = []
    for layer in _LAYERS:
        used = [q for q in runs if q.retrieval_layers.get(layer, 0) >= 1]
        retrieval_rate = (len(used) / total) if total else 0.0
        if used:
            positive = sum(1 for q in used if _positive_outcome(q))
            citation_rate = positive / len(used)
        else:
            citation_rate = 0.0
        dead_weight = (
            retrieval_rate >= config.dead_weight_retrieval_min
            and citation_rate < config.dead_weight_citation_max
        )
        subs = sorted({q.subscription_id for q in used})
        short = (
            f"Layer '{layer}': used in {retrieval_rate:.0%} of queries, "
            f"positive-outcome rate {citation_rate:.0%}"
            + (" — DEAD WEIGHT" if dead_weight else "")
        )
        out.append(
            Cluster(
                cluster_id=f"artifact_{layer}",
                cluster_type=ClusterType.ARTIFACT_UTILITY,
                size=len(used),
                subscription_ids=subs,
                primary_intent=None,
                profile_domain=None,
                fingerprint_samples=[],
                short_description=short,
                signal_score=round(citation_rate, 3),
                evidence={
                    "layer": layer,
                    "retrieval_rate": round(retrieval_rate, 3),
                    "citation_rate": round(citation_rate, 3),
                    "total_queries": total,
                    "dead_weight_flag": dead_weight,
                },
            )
        )
    return out
```

- [ ] **Step 4: Run tests and confirm pass**

Run: `pytest tests/scripts/sme_patterns/clustering/test_artifact_utility.py -v`
Expected: PASS for all 4 tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/sme_patterns/clustering/artifact_utility.py \
    tests/scripts/sme_patterns/clustering/test_artifact_utility.py
git commit -m "phase6(sme-patterns): Pass 3 — artifact utility ranking"
```

---

## Task 10: Pass 4 — Persona effect

Group queries by `(profile_domain, adapter_persona_role)`. Compute per-group SME-score proxy:

```
sme_score_proxy = 0.5 * positive_outcome_rate
                + 0.3 * (1 - clip(avg_drops/3, 0, 1))
                + 0.2 * (1 - honest_fallback_rate)
```

A persona under-performing its domain baseline by more than `config.regression_delta` is flagged. One cluster per `(domain, persona)` with `cluster_type == PERSONA_EFFECT`, sorted worst-first.

**Files:**
- Create: `scripts/sme_patterns/clustering/persona_effect.py`
- Create: `tests/scripts/sme_patterns/clustering/test_persona_effect.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/scripts/sme_patterns/clustering/test_persona_effect.py`:

```python
"""Tests for persona-effect analysis."""
from datetime import datetime

from scripts.sme_patterns.clustering.persona_effect import (
    PersonaEffectConfig,
    analyze_persona_effect,
    sme_score_proxy,
)
from scripts.sme_patterns.schema import ClusterType, QueryFeedback, QueryRun


def _run(qid, *, domain="finance", persona="senior financial analyst",
         rating=1, drops=0, honest_fallback=False):
    fb = QueryFeedback(rating=rating, source="feedback_tracker") if rating is not None else None
    return QueryRun(
        subscription_id="s",
        profile_id="p",
        profile_domain=domain,
        query_id=qid,
        query_text="q",
        query_fingerprint=qid,
        intent="analyze",
        adapter_version="1.0.0",
        adapter_persona_role=persona,
        retrieval_layers={"chunks": 1, "kg": 0, "sme_artifacts": 1, "url": 0},
        citation_verifier_drops=drops,
        honest_compact_fallback=honest_fallback,
        captured_at=datetime(2026, 4, 5, 10, 0, 0),
        feedback=fb,
    )


def test_sme_score_proxy_in_range():
    runs = [_run("q", rating=1, drops=0, honest_fallback=False)]
    score = sme_score_proxy(runs)
    assert 0.0 <= score <= 1.0


def test_sme_score_proxy_penalizes_drops_and_fallback():
    good = [_run(f"g{i}", rating=1, drops=0) for i in range(5)]
    bad = [_run(f"b{i}", rating=-1, drops=2, honest_fallback=True) for i in range(5)]
    assert sme_score_proxy(good) > sme_score_proxy(bad)


def test_analyze_persona_effect_emits_one_cluster_per_pair():
    runs = [
        _run(f"fa_{i}", domain="finance", persona="senior financial analyst", rating=1)
        for i in range(5)
    ] + [
        _run(f"fb_{i}", domain="finance", persona="experimental draft persona",
             rating=-1, drops=2, honest_fallback=True)
        for i in range(5)
    ] + [
        _run(f"le_{i}", domain="legal", persona="senior legal counsel", rating=1)
        for i in range(3)
    ]
    clusters = analyze_persona_effect(runs, PersonaEffectConfig())
    assert len(clusters) == 3
    assert all(c.cluster_type == ClusterType.PERSONA_EFFECT for c in clusters)
    # Worst persona sorted first
    assert "experimental draft persona" in clusters[0].short_description


def test_regression_flag_triggers_when_under_baseline():
    runs = [
        _run(f"ok_{i}", domain="finance", persona="senior financial analyst",
             rating=1) for i in range(10)
    ] + [
        _run(f"bad_{i}", domain="finance", persona="rogue persona",
             rating=-1, drops=2, honest_fallback=True) for i in range(10)
    ]
    clusters = analyze_persona_effect(runs, PersonaEffectConfig(regression_delta=0.1))
    by_desc = {c.short_description: c for c in clusters}
    bad = next(c for c in clusters if "rogue persona" in c.short_description)
    assert bad.evidence["regression_flag"] is True


def test_empty_input():
    clusters = analyze_persona_effect([], PersonaEffectConfig())
    assert clusters == []
```

- [ ] **Step 2: Run tests and confirm failure**

Run: `pytest tests/scripts/sme_patterns/clustering/test_persona_effect.py -v`
Expected: FAIL.

- [ ] **Step 3: Write the persona-effect pass**

Create `scripts/sme_patterns/clustering/persona_effect.py`:

```python
"""Pass 4 — Persona effect.

Compute a per-persona SME-score proxy per domain; flag personas whose proxy
regresses under the domain baseline by more than a configurable delta.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from scripts.sme_patterns.schema import Cluster, ClusterType, QueryRun


@dataclass(frozen=True)
class PersonaEffectConfig:
    regression_delta: float = 0.15
    min_queries_per_persona: int = 5


def _positive_outcome(q: QueryRun) -> bool:
    if q.citation_verifier_drops > 0:
        return False
    if q.feedback and q.feedback.rating == -1:
        return False
    return True


def sme_score_proxy(runs: list[QueryRun]) -> float:
    if not runs:
        return 0.0
    n = len(runs)
    pos = sum(1 for q in runs if _positive_outcome(q)) / n
    drops = sum(q.citation_verifier_drops for q in runs) / n
    drops_norm = max(0.0, 1.0 - min(1.0, drops / 3.0))
    fallback_rate = sum(1 for q in runs if q.honest_compact_fallback) / n
    score = 0.5 * pos + 0.3 * drops_norm + 0.2 * (1.0 - fallback_rate)
    return round(max(0.0, min(1.0, score)), 3)


def analyze_persona_effect(
    runs: Iterable[QueryRun],
    config: PersonaEffectConfig,
) -> list[Cluster]:
    runs = list(runs)
    if not runs:
        return []

    by_domain: dict[str, list[QueryRun]] = defaultdict(list)
    by_pair: dict[tuple[str, str], list[QueryRun]] = defaultdict(list)
    for q in runs:
        by_domain[q.profile_domain].append(q)
        by_pair[(q.profile_domain, q.adapter_persona_role)].append(q)

    # Domain-level baseline
    baseline_score = {dom: sme_score_proxy(rs) for dom, rs in by_domain.items()}

    produced: list[Cluster] = []
    for (domain, persona), group in by_pair.items():
        if len(group) < config.min_queries_per_persona:
            continue
        score = sme_score_proxy(group)
        base = baseline_score.get(domain, 0.0)
        regression_flag = (base - score) >= config.regression_delta
        subs = sorted({q.subscription_id for q in group})
        short = (
            f"Persona '{persona}' on {domain}: proxy={score:.2f} "
            f"(domain baseline {base:.2f})"
            + ("  REGRESSION" if regression_flag else "")
        )
        produced.append(
            Cluster(
                cluster_id=f"persona_{domain}_{hash(persona) & 0xFFFF:04x}",
                cluster_type=ClusterType.PERSONA_EFFECT,
                size=len(group),
                subscription_ids=subs,
                primary_intent=None,
                profile_domain=domain,
                fingerprint_samples=[],
                short_description=short,
                signal_score=score,
                evidence={
                    "persona_role": persona,
                    "sme_score_proxy": score,
                    "domain_baseline": base,
                    "regression_flag": regression_flag,
                    "queries": len(group),
                },
            )
        )

    produced.sort(key=lambda c: c.signal_score)  # ascending → worst first
    return produced
```

- [ ] **Step 4: Run tests and confirm pass**

Run: `pytest tests/scripts/sme_patterns/clustering/test_persona_effect.py -v`
Expected: PASS for all 5 tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/sme_patterns/clustering/persona_effect.py \
    tests/scripts/sme_patterns/clustering/test_persona_effect.py
git commit -m "phase6(sme-patterns): Pass 4 — persona effect analysis"
```

---

## Task 11: Report model + Jinja2 Markdown renderer

Monthly Markdown is rendered from a template so the format is tunable without touching Python. Writes `analytics/sme_patterns_{YYYY-MM}.md` and appends links to rollback post-mortems under `analytics/sme_rollback_*.md` whose date falls in the month (spec Section 13.3).

**Files:**
- Create: `scripts/sme_patterns/report/model.py`
- Create: `scripts/sme_patterns/report/renderer.py`
- Create: `analytics/templates/sme_patterns_template.md`
- Create: `tests/scripts/sme_patterns/report/test_model.py`
- Create: `tests/scripts/sme_patterns/report/test_renderer.py`

- [ ] **Step 1: Write the failing model tests**

Create `tests/scripts/sme_patterns/report/test_model.py`:

```python
"""Tests for report composer."""
from datetime import datetime

from scripts.sme_patterns.report.model import compose_pattern_report
from scripts.sme_patterns.schema import Cluster, ClusterType, QueryRun, SynthesisRun
from tests.scripts.sme_patterns.fixtures.query_trace_factory import make_query_jsonl


def _query(qid):
    from scripts.sme_patterns.trace_loader import parse_query_jsonl
    txt = make_query_jsonl(query_id=qid)
    return next(iter(parse_query_jsonl(txt)))


def test_compose_report_counts_inputs():
    runs = [_query(f"q{i}") for i in range(10)]
    rep = compose_pattern_report(
        query_runs=runs,
        synth_runs=[],
        successes=[], failures=[], artifact_utility=[], persona_effect=[],
        training_candidates=[],
        period_start=datetime(2026, 4, 1),
        period_end=datetime(2026, 4, 30),
        rollback_links=[],
    )
    assert rep.num_query_runs == 10
    assert rep.num_synth_runs == 0
    assert rep.run_id.startswith("patterns_2026-04")


def test_compose_report_carries_cluster_lists():
    suc = Cluster(
        cluster_id="s1", cluster_type=ClusterType.SUCCESS, size=1,
        short_description="desc", signal_score=0.9,
    )
    fail = Cluster(
        cluster_id="f1", cluster_type=ClusterType.FAILURE, size=1,
        short_description="desc", signal_score=0.2,
    )
    rep = compose_pattern_report(
        query_runs=[], synth_runs=[],
        successes=[suc], failures=[fail],
        artifact_utility=[], persona_effect=[],
        training_candidates=[],
        period_start=datetime(2026, 4, 1),
        period_end=datetime(2026, 4, 30),
        rollback_links=["analytics/sme_rollback_2026-04-12.md"],
    )
    assert rep.successes == [suc]
    assert rep.failures == [fail]
    assert rep.rollback_links == ["analytics/sme_rollback_2026-04-12.md"]
```

- [ ] **Step 2: Write the failing renderer tests**

Create `tests/scripts/sme_patterns/report/test_renderer.py`:

```python
"""Tests for the Jinja2 markdown renderer."""
from datetime import datetime
from pathlib import Path

from scripts.sme_patterns.report.model import compose_pattern_report
from scripts.sme_patterns.report.renderer import render_pattern_report
from scripts.sme_patterns.schema import Cluster, ClusterType, TrainingCandidate


def _sample_report(rollback_links=None):
    s = Cluster(
        cluster_id="succ_finance_analyze_0",
        cluster_type=ClusterType.SUCCESS,
        size=18,
        subscription_ids=["sub_a"],
        profile_domain="finance",
        primary_intent="analyze",
        short_description="Successful analyze queries on finance — top terms: revenue, q3, trend",
        signal_score=0.88,
        evidence={"top_terms": ["revenue", "q3", "trend"], "avg_sme_artifacts": 3.2},
    )
    f = Cluster(
        cluster_id="fail_finance_recommend_0",
        cluster_type=ClusterType.FAILURE,
        size=12,
        subscription_ids=["sub_a", "sub_b"],
        profile_domain="finance",
        primary_intent="recommend",
        short_description="Failing recommend queries on finance — drops ≈ 2.1, neg-rate 60%",
        signal_score=0.54,
        evidence={"avg_verifier_drops": 2.1, "thumbs_down_rate": 0.6},
    )
    a = Cluster(
        cluster_id="artifact_sme_artifacts",
        cluster_type=ClusterType.ARTIFACT_UTILITY,
        size=900,
        short_description="Layer 'sme_artifacts': used in 80% of queries",
        signal_score=0.72,
        evidence={"retrieval_rate": 0.8, "citation_rate": 0.72,
                  "dead_weight_flag": False, "layer": "sme_artifacts"},
    )
    p = Cluster(
        cluster_id="persona_finance_abcd",
        cluster_type=ClusterType.PERSONA_EFFECT,
        size=50,
        profile_domain="finance",
        short_description="Persona 'cfo advisor' on finance: proxy=0.88 (baseline 0.85)",
        signal_score=0.88,
        evidence={"persona_role": "cfo advisor", "regression_flag": False,
                  "sme_score_proxy": 0.88, "domain_baseline": 0.85},
    )
    tc = TrainingCandidate(
        candidate_id="tc_001",
        cluster_ids=["fail_finance_recommend_0", "fail_finance_recommend_0_prev"],
        months_present=2,
        total_volume=48,
        stabilization_score=0.7,
        dominant_intent="recommend",
        dominant_domain="finance",
        short_description="recurring ungrounded recommendations on finance",
    )
    return compose_pattern_report(
        query_runs=[], synth_runs=[],
        successes=[s], failures=[f],
        artifact_utility=[a], persona_effect=[p],
        training_candidates=[tc],
        period_start=datetime(2026, 4, 1),
        period_end=datetime(2026, 4, 30),
        rollback_links=rollback_links or [],
    )


def test_renderer_produces_markdown_sections(tmp_path: Path):
    rep = _sample_report()
    out = tmp_path / "sme_patterns_2026-04.md"
    path = render_pattern_report(rep, out)
    text = Path(path).read_text()
    assert "# DocWain SME Patterns — 2026-04" in text
    assert "## Executive summary" in text
    assert "## Success patterns" in text
    assert "## Failure patterns" in text
    assert "## Artifact utility" in text
    assert "## Persona performance" in text
    assert "## Training candidates" in text
    # Content propagates
    assert "cfo advisor" in text
    assert "tc_001" in text
    # No raw query texts — only fingerprint-style samples
    assert "sub_a" in text


def test_renderer_embeds_rollback_links(tmp_path: Path):
    rep = _sample_report(rollback_links=["analytics/sme_rollback_2026-04-12.md"])
    out = tmp_path / "sme_patterns_2026-04.md"
    path = render_pattern_report(rep, out)
    text = Path(path).read_text()
    assert "## Rollback post-mortems" in text
    assert "analytics/sme_rollback_2026-04-12.md" in text


def test_renderer_with_no_rollbacks_omits_section(tmp_path: Path):
    rep = _sample_report(rollback_links=[])
    out = tmp_path / "sme_patterns_2026-04.md"
    path = render_pattern_report(rep, out)
    text = Path(path).read_text()
    # Section header should NOT appear when no rollbacks
    assert "## Rollback post-mortems" not in text
```

- [ ] **Step 3: Run tests and confirm failure**

Run: `pytest tests/scripts/sme_patterns/report -v`
Expected: FAIL.

- [ ] **Step 4: Write the report model**

Create `scripts/sme_patterns/report/model.py`:

```python
"""Compose PatternReport from clusters + input counts.

Thin constructor — no I/O here. rendering lives in renderer.py.
"""
from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime

from scripts.sme_patterns.schema import (
    Cluster,
    PatternReport,
    QueryRun,
    SynthesisRun,
    TrainingCandidate,
)


def compose_pattern_report(
    *,
    query_runs: Iterable[QueryRun],
    synth_runs: Iterable[SynthesisRun],
    successes: list[Cluster],
    failures: list[Cluster],
    artifact_utility: list[Cluster],
    persona_effect: list[Cluster],
    training_candidates: list[TrainingCandidate],
    period_start: datetime,
    period_end: datetime,
    rollback_links: list[str],
) -> PatternReport:
    query_runs = list(query_runs)
    synth_runs = list(synth_runs)
    return PatternReport(
        run_id=f"patterns_{period_start.strftime('%Y-%m')}",
        period_start=period_start,
        period_end=period_end,
        num_synth_runs=len(synth_runs),
        num_query_runs=len(query_runs),
        successes=successes,
        failures=failures,
        artifact_utility=artifact_utility,
        persona_effect=persona_effect,
        training_candidates=training_candidates,
        rollback_links=rollback_links,
    )
```

- [ ] **Step 5: Write the Jinja2 template**

Create `analytics/templates/sme_patterns_template.md`:

```markdown
# DocWain SME Patterns — {{ report.period_start.strftime('%Y-%m') }}

Generated {{ generated_at }} from {{ report.num_query_runs }} query runs and {{ report.num_synth_runs }} synthesis runs in the window {{ report.period_start.strftime('%Y-%m-%d') }} → {{ report.period_end.strftime('%Y-%m-%d') }}.

## Executive summary

- Success clusters surfaced: **{{ report.successes | length }}**
- Failure clusters surfaced: **{{ report.failures | length }}**
- Artifact utility rows: **{{ report.artifact_utility | length }}**
- Persona performance rows: **{{ report.persona_effect | length }}**
- Training candidates for sub-project F: **{{ report.training_candidates | length }}**
{% if report.rollback_links %}
- Rollback post-mortems this month: **{{ report.rollback_links | length }}**
{%- endif %}

Phase 6 produces evidence for future retraining. No retraining is triggered automatically; sub-project F remains a separate, human-gated project.

## Success patterns

{% if report.successes %}
{% for c in report.successes %}
### {{ c.cluster_id }}

- **Size:** {{ c.size }}
- **Domain / intent:** {{ c.profile_domain or 'any' }} / {{ c.primary_intent or 'any' }}
- **Subscriptions:** {{ c.subscription_ids | join(', ') or 'n/a' }}
- **Signal score:** {{ '%.2f' | format(c.signal_score) }}
- **Summary:** {{ c.short_description }}
- **Evidence:**
{% for k, v in c.evidence.items() %}  - `{{ k }}`: {{ v }}
{% endfor %}

{% endfor %}
{% else %}
_No success clusters surfaced this month._
{% endif %}

## Failure patterns

{% if report.failures %}
{% for c in report.failures %}
### {{ c.cluster_id }}

- **Size:** {{ c.size }}
- **Domain / intent:** {{ c.profile_domain or 'any' }} / {{ c.primary_intent or 'any' }}
- **Subscriptions:** {{ c.subscription_ids | join(', ') or 'n/a' }}
- **Severity score:** {{ '%.2f' | format(c.signal_score) }}
- **Summary:** {{ c.short_description }}
- **Fingerprint samples:** {{ c.fingerprint_samples | join(', ') or 'n/a' }}
- **Evidence:**
{% for k, v in c.evidence.items() %}  - `{{ k }}`: {{ v }}
{% endfor %}

{% endfor %}
{% else %}
_No failure clusters surfaced this month._
{% endif %}

## Artifact utility

{% if report.artifact_utility %}
| Layer | Retrieval rate | Citation rate | Dead-weight? |
|---|---|---|---|
{% for c in report.artifact_utility %}| {{ c.evidence.layer }} | {{ '%.0f%%' | format(c.evidence.retrieval_rate * 100) }} | {{ '%.0f%%' | format(c.evidence.citation_rate * 100) }} | {{ 'yes' if c.evidence.dead_weight_flag else 'no' }} |
{% endfor %}
{% else %}
_No artifact utility rows._
{% endif %}

## Persona performance

{% if report.persona_effect %}
| Persona | Domain | Proxy score | Baseline | Regression? | Queries |
|---|---|---|---|---|---|
{% for c in report.persona_effect %}| {{ c.evidence.persona_role }} | {{ c.profile_domain }} | {{ '%.2f' | format(c.evidence.sme_score_proxy) }} | {{ '%.2f' | format(c.evidence.domain_baseline) }} | {{ 'yes' if c.evidence.regression_flag else 'no' }} | {{ c.evidence.queries }} |
{% endfor %}
{% else %}
_No persona rows._
{% endif %}

## Training candidates

Failure patterns that have stabilized across ≥2 months become candidates for sub-project F. Candidates listed here are evidence; sub-project F is triggered by human decision after review.

{% if report.training_candidates %}
{% for tc in report.training_candidates %}
### {{ tc.candidate_id }}

- **Months present:** {{ tc.months_present }}
- **Total volume:** {{ tc.total_volume }}
- **Stabilization score:** {{ '%.2f' | format(tc.stabilization_score) }}
- **Dominant intent / domain:** {{ tc.dominant_intent }} / {{ tc.dominant_domain }}
- **Cluster ids:** {{ tc.cluster_ids | join(', ') }}
- **Summary:** {{ tc.short_description }}

{% endfor %}
{% else %}
_No stabilized candidates this month._
{% endif %}

{% if report.rollback_links %}
## Rollback post-mortems

{% for link in report.rollback_links %}- [{{ link }}]({{ link }})
{% endfor %}
{% endif %}
```

- [ ] **Step 6: Write the renderer**

Create `scripts/sme_patterns/report/renderer.py`:

```python
"""Render PatternReport to Markdown via Jinja2.

Template lives in analytics/templates/ so the format can be tuned without
touching Python. The renderer never mutates the report; it writes to a
target path and returns it.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from scripts.sme_patterns.schema import PatternReport

_TEMPLATE_NAME = "sme_patterns_template.md"


def render_pattern_report(
    report: PatternReport,
    out_path: Path | str,
    *,
    templates_dir: Path | str | None = None,
    generated_at: datetime | None = None,
) -> str:
    """Render and return the written path as a string."""
    templates_dir = Path(templates_dir or Path(__file__).resolve().parents[3] / "analytics" / "templates")
    env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=select_autoescape(enabled_extensions=(), default_for_string=False),
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    template = env.get_template(_TEMPLATE_NAME)
    text = template.render(
        report=report,
        generated_at=(generated_at or datetime.utcnow()).isoformat(timespec="seconds"),
    )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return str(out)
```

- [ ] **Step 7: Run tests and confirm pass**

Run: `pytest tests/scripts/sme_patterns/report -v`
Expected: PASS for all 5 tests.

- [ ] **Step 8: Commit**

```bash
git add scripts/sme_patterns/report/model.py scripts/sme_patterns/report/renderer.py \
    analytics/templates/sme_patterns_template.md \
    tests/scripts/sme_patterns/report/test_model.py \
    tests/scripts/sme_patterns/report/test_renderer.py
git commit -m "phase6(sme-patterns): report model + Jinja2 Markdown renderer"
```

---

## Task 12: Training-trigger evaluator

`scripts/evaluate_training_trigger.py` compares the current month's failure clusters against the prior 1+ months' reports and produces a `TrainingCandidate` list when clusters stabilize:

**Stabilization score formula (per cluster):**
```
stabilization = 0.5 * months_present_ratio  +  0.3 * volume_ratio  +  0.2 * severity_avg

where:
  months_present_ratio = months_present_for_cluster / total_months_window
  volume_ratio         = min(1.0, cluster.total_volume / volume_ref)   (volume_ref default 50)
  severity_avg         = mean of cluster.signal_score across months
```

A cluster is a **training candidate** when `months_present >= min_months` (default 2), `total_volume >= min_volume` (default 20), and `stabilization >= stabilization_threshold` (default 0.55). Sub-project F remains separate; this script produces evidence only.

**Files:**
- Create: `scripts/evaluate_training_trigger.py`
- Create: `tests/scripts/sme_patterns/test_evaluate_training_trigger.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/scripts/sme_patterns/test_evaluate_training_trigger.py`:

```python
"""Tests for the training-trigger evaluator."""
from datetime import datetime
from pathlib import Path

from scripts.evaluate_training_trigger import (
    TrainingTriggerConfig,
    evaluate_candidates,
    load_reports_from_dir,
    match_clusters_across_months,
)
from scripts.sme_patterns.schema import (
    Cluster,
    ClusterType,
    PatternReport,
    TrainingCandidate,
)


def _fail_cluster(cid, size, severity, domain="finance", intent="recommend"):
    return Cluster(
        cluster_id=cid, cluster_type=ClusterType.FAILURE, size=size,
        profile_domain=domain, primary_intent=intent,
        short_description="x", signal_score=severity,
        fingerprint_samples=["fp_x", "fp_y"],
    )


def _report(month, clusters):
    return PatternReport(
        run_id=f"patterns_2026-{month:02d}",
        period_start=datetime(2026, month, 1),
        period_end=datetime(2026, month, 28),
        num_synth_runs=0, num_query_runs=0,
        successes=[], failures=clusters, artifact_utility=[], persona_effect=[],
        training_candidates=[], rollback_links=[],
    )


def test_match_clusters_across_months_matches_by_domain_intent_and_terms():
    # Same cluster characteristic appears in 2 months
    m1 = _fail_cluster("fail_finance_recommend_0", 20, 0.6)
    m2 = _fail_cluster("fail_finance_recommend_0", 30, 0.5)
    groups = match_clusters_across_months([
        _report(3, [m1]),
        _report(4, [m2]),
    ])
    assert len(groups) == 1
    assert groups[0].months_present == 2
    assert groups[0].total_volume == 50


def test_match_distinguishes_different_intents():
    m1 = _fail_cluster("fail_finance_recommend_0", 20, 0.6, intent="recommend")
    m2 = _fail_cluster("fail_finance_diagnose_0", 30, 0.5, intent="diagnose")
    groups = match_clusters_across_months([
        _report(3, [m1, m2]),
    ])
    assert len(groups) == 2


def test_evaluate_candidates_applies_thresholds():
    m1 = _fail_cluster("fail_finance_recommend_0", 20, 0.6)
    m2 = _fail_cluster("fail_finance_recommend_0", 30, 0.55)
    cfg = TrainingTriggerConfig(
        min_months=2, min_volume=20, stabilization_threshold=0.5, total_months_window=2,
        volume_ref=50,
    )
    cands = evaluate_candidates([_report(3, [m1]), _report(4, [m2])], cfg)
    assert len(cands) == 1
    assert isinstance(cands[0], TrainingCandidate)
    assert cands[0].months_present == 2
    assert cands[0].total_volume == 50
    assert cands[0].stabilization_score >= 0.5


def test_evaluate_candidates_filters_single_month_below_min_months():
    cfg = TrainingTriggerConfig(min_months=2, min_volume=1,
                                stabilization_threshold=0.0, total_months_window=2)
    m = _fail_cluster("fail_finance_recommend_0", 100, 1.0)
    cands = evaluate_candidates([_report(4, [m])], cfg)
    assert cands == []


def test_evaluate_candidates_filters_low_volume():
    m1 = _fail_cluster("fail_finance_recommend_0", 2, 0.9)
    m2 = _fail_cluster("fail_finance_recommend_0", 3, 0.9)
    cfg = TrainingTriggerConfig(min_months=2, min_volume=20,
                                stabilization_threshold=0.0, total_months_window=2)
    cands = evaluate_candidates([_report(3, [m1]), _report(4, [m2])], cfg)
    assert cands == []


def test_load_reports_from_dir(tmp_path: Path):
    r3 = _report(3, [_fail_cluster("fail_x", 1, 0.5)])
    r4 = _report(4, [_fail_cluster("fail_x", 1, 0.5)])
    (tmp_path / "sme_patterns_2026-03.json").write_text(r3.model_dump_json())
    (tmp_path / "sme_patterns_2026-04.json").write_text(r4.model_dump_json())
    (tmp_path / "notes.txt").write_text("ignored")

    reps = load_reports_from_dir(tmp_path)
    assert [r.run_id for r in reps] == ["patterns_2026-03", "patterns_2026-04"]
```

- [ ] **Step 2: Run tests and confirm failure**

Run: `pytest tests/scripts/sme_patterns/test_evaluate_training_trigger.py -v`
Expected: FAIL.

- [ ] **Step 3: Write the evaluator**

Create `scripts/evaluate_training_trigger.py`:

```python
"""Training-trigger evaluator.

Reads the last N monthly pattern reports and outputs TrainingCandidate
records for failure clusters that have stabilized. Sub-project F is a
separate human-gated project; this script writes evidence, nothing more.

CLI:
    python scripts/evaluate_training_trigger.py \\
        --reports-dir analytics \\
        --out analytics/training_candidates_$(date +%Y-%m).json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from scripts.sme_patterns.schema import (
    Cluster,
    ClusterType,
    PatternReport,
    TrainingCandidate,
)

logger = logging.getLogger(__name__)

_REPORT_GLOB = "sme_patterns_*.json"


@dataclass(frozen=True)
class TrainingTriggerConfig:
    min_months: int = 2
    min_volume: int = 20
    stabilization_threshold: float = 0.55
    total_months_window: int = 2
    volume_ref: float = 50.0


@dataclass
class _Group:
    key: tuple[str, str, tuple[str, ...]]  # (domain, intent, terms)
    clusters: list[Cluster]
    months_present: int = 0
    total_volume: int = 0


def _cluster_terms_key(c: Cluster) -> tuple[str, ...]:
    """Produce a stable signature for cross-month matching.

    Prefer `evidence.top_terms` (set by each clustering pass); fall back
    to the cluster_id shape suffix (the centroid hash). This keeps matches
    interpretable even when top-terms drift slightly month to month.
    """
    terms = c.evidence.get("top_terms") or []
    if isinstance(terms, list) and terms:
        # sort + truncate to top 3 for stability
        return tuple(sorted([str(t) for t in terms])[:3])
    # Fallback: use the cluster_id (without the centroid index)
    parts = c.cluster_id.rsplit("_", 1)
    return (parts[0],)


def match_clusters_across_months(reports: list[PatternReport]) -> list[_Group]:
    """Group failure clusters across monthly reports by (domain, intent, top-terms)."""
    buckets: dict[tuple[str, str, tuple[str, ...]], _Group] = {}
    months_seen: dict[tuple[str, str, tuple[str, ...]], set[str]] = defaultdict(set)

    for rep in reports:
        month_key = rep.period_start.strftime("%Y-%m")
        for c in rep.failures:
            k = (
                c.profile_domain or "",
                c.primary_intent or "",
                _cluster_terms_key(c),
            )
            if k not in buckets:
                buckets[k] = _Group(key=k, clusters=[])
            buckets[k].clusters.append(c)
            months_seen[k].add(month_key)

    out = list(buckets.values())
    for g in out:
        g.months_present = len(months_seen[g.key])
        g.total_volume = sum(c.size for c in g.clusters)
    return out


def evaluate_candidates(
    reports: list[PatternReport],
    config: TrainingTriggerConfig,
) -> list[TrainingCandidate]:
    """Return the list of TrainingCandidates for stabilized failure clusters."""
    groups = match_clusters_across_months(reports)
    candidates: list[TrainingCandidate] = []
    for g in groups:
        if g.months_present < config.min_months:
            continue
        if g.total_volume < config.min_volume:
            continue

        months_ratio = g.months_present / max(1, config.total_months_window)
        volume_ratio = min(1.0, g.total_volume / max(1.0, config.volume_ref))
        severity_avg = (
            sum(c.signal_score for c in g.clusters) / len(g.clusters)
            if g.clusters else 0.0
        )
        stabilization = round(
            0.5 * months_ratio + 0.3 * volume_ratio + 0.2 * severity_avg,
            3,
        )
        if stabilization < config.stabilization_threshold:
            continue

        domain, intent, terms = g.key
        candidate_id = "tc_" + ("_".join(filter(None, [domain, intent, *terms])) or "root")
        candidate_id = candidate_id[:120]
        short = (
            f"recurring {intent or 'failure'} clusters on {domain or 'any domain'} "
            f"— {g.months_present} months present, volume {g.total_volume}, "
            f"severity≈{severity_avg:.2f}"
        )
        candidates.append(
            TrainingCandidate(
                candidate_id=candidate_id,
                cluster_ids=[c.cluster_id for c in g.clusters],
                months_present=g.months_present,
                total_volume=g.total_volume,
                stabilization_score=stabilization,
                dominant_intent=intent or "unknown",
                dominant_domain=domain or "unknown",
                short_description=short,
            )
        )

    candidates.sort(key=lambda c: c.stabilization_score, reverse=True)
    return candidates


def load_reports_from_dir(dirpath: Path | str) -> list[PatternReport]:
    p = Path(dirpath)
    reports: list[PatternReport] = []
    for file in sorted(p.glob(_REPORT_GLOB)):
        if not file.name.endswith(".json"):
            continue
        try:
            reports.append(PatternReport.model_validate_json(file.read_text(encoding="utf-8")))
        except Exception:
            logger.exception("failed to load report %s; skipping", file)
            continue
    return reports


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate training triggers from monthly reports")
    parser.add_argument("--reports-dir", type=Path, default=Path("analytics"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-months", type=int, default=2)
    parser.add_argument("--min-volume", type=int, default=20)
    parser.add_argument("--stabilization-threshold", type=float, default=0.55)
    parser.add_argument("--total-months-window", type=int, default=2)
    args = parser.parse_args(argv)

    reports = load_reports_from_dir(args.reports_dir)
    if not reports:
        print(f"[evaluate_training_trigger] no reports under {args.reports_dir}", file=sys.stderr)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("[]", encoding="utf-8")
        return 0

    config = TrainingTriggerConfig(
        min_months=args.min_months,
        min_volume=args.min_volume,
        stabilization_threshold=args.stabilization_threshold,
        total_months_window=args.total_months_window,
    )
    candidates = evaluate_candidates(reports, config)
    payload = [c.model_dump() for c in candidates]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"[evaluate_training_trigger] wrote {len(candidates)} candidates to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests and confirm pass**

Run: `pytest tests/scripts/sme_patterns/test_evaluate_training_trigger.py -v`
Expected: PASS for all 6 tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/evaluate_training_trigger.py \
    tests/scripts/sme_patterns/test_evaluate_training_trigger.py
git commit -m "phase6(sme-patterns): training-trigger evaluator with stabilization score"
```

---

## Task 13: Batch orchestrator + CLI entry-point

Wires the loader, merger, four clustering passes, and renderer into one CLI. `scripts/mine_sme_patterns.py` is a thin top-level entry delegating to `scripts/sme_patterns/run.py`.

**Files:**
- Create: `scripts/sme_patterns/run.py`
- Create: `scripts/mine_sme_patterns.py`
- Create: `tests/scripts/sme_patterns/test_run.py`

- [ ] **Step 1: Write the failing orchestrator tests**

Create `tests/scripts/sme_patterns/test_run.py`:

```python
"""Tests for the pattern-mining orchestrator."""
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

from scripts.sme_patterns.run import (
    RunConfig,
    default_window,
    run_monthly_mining,
)
from scripts.sme_patterns.schema import PatternReport, QueryFeedback
from tests.scripts.sme_patterns.fixtures.query_trace_factory import make_query_jsonl
from tests.scripts.sme_patterns.fixtures.synth_trace_factory import make_synth_jsonl


def _installed_blobs():
    """Build a small Azure-Blob snapshot the loader will read."""
    blobs = {}
    blobs["sme_traces/synthesis/sub_a/prof_a/syn_1.jsonl"] = make_synth_jsonl(
        synthesis_id="syn_1", started_at=datetime(2026, 4, 1, 2, 0, 0),
        drop_count=1,
    )
    for i in range(5):
        blobs[f"sme_traces/queries/sub_a/prof_a/2026-04-05/succ_{i}.jsonl"] = make_query_jsonl(
            query_id=f"succ_{i}", intent="analyze", rating=1,
            sme_artifacts=3, citation_verifier_drops=0,
            captured_at=datetime(2026, 4, 5, 10, 0, 0),
        )
    for i in range(3):
        blobs[f"sme_traces/queries/sub_a/prof_a/2026-04-10/fail_{i}.jsonl"] = make_query_jsonl(
            query_id=f"fail_{i}", intent="recommend",
            rating=-1, sme_artifacts=1, citation_verifier_drops=2,
            captured_at=datetime(2026, 4, 10, 10, 0, 0),
        )
    return blobs


def test_default_window_is_last_full_month():
    w = default_window(now=datetime(2026, 5, 3, 2, 0, 0))
    assert w.start == datetime(2026, 4, 1, 0, 0, 0)
    assert w.end.day == 30
    assert w.end.month == 4


def test_run_monthly_mining_produces_report_file(tmp_path: Path):
    blobs = _installed_blobs()
    list_blobs = MagicMock(side_effect=lambda prefix: [k for k in blobs if k.startswith(prefix)])
    read_blob = MagicMock(side_effect=lambda name: blobs[name])
    tracker = MagicMock()
    tracker.get_profile_metrics.return_value = {
        "total_queries": 10, "avg_confidence": 0.7,
        "grounded_ratio": 0.7, "low_confidence_count": 0,
    }

    cfg = RunConfig(
        window_start=datetime(2026, 4, 1),
        window_end=datetime(2026, 4, 30, 23, 59, 59),
        analytics_dir=tmp_path,
        rollback_glob="sme_rollback_*.md",
    )
    out_path = run_monthly_mining(
        cfg,
        list_blobs=list_blobs,
        read_blob=read_blob,
        feedback_tracker=tracker,
    )
    assert Path(out_path).exists()
    md = Path(out_path).read_text()
    assert "# DocWain SME Patterns — 2026-04" in md
    # Both success and failure sections populated
    assert "Failing recommend queries" in md or "failure" in md.lower()
    # JSON snapshot also written alongside Markdown
    json_snap = Path(out_path).with_suffix(".json")
    assert json_snap.exists()
    rep = PatternReport.model_validate_json(json_snap.read_text())
    assert rep.num_query_runs == 8
    assert rep.num_synth_runs == 1


def test_run_monthly_mining_includes_rollback_links(tmp_path: Path):
    blobs = _installed_blobs()
    list_blobs = MagicMock(side_effect=lambda prefix: [k for k in blobs if k.startswith(prefix)])
    read_blob = MagicMock(side_effect=lambda name: blobs[name])
    tracker = MagicMock()
    tracker.get_profile_metrics.return_value = {"total_queries": 0, "avg_confidence": 0.0,
                                                 "grounded_ratio": 0.0, "low_confidence_count": 0}
    (tmp_path / "sme_rollback_2026-04-12.md").write_text("post-mortem body")

    cfg = RunConfig(
        window_start=datetime(2026, 4, 1),
        window_end=datetime(2026, 4, 30, 23, 59, 59),
        analytics_dir=tmp_path,
        rollback_glob="sme_rollback_*.md",
    )
    out_path = run_monthly_mining(
        cfg,
        list_blobs=list_blobs,
        read_blob=read_blob,
        feedback_tracker=tracker,
    )
    assert "## Rollback post-mortems" in Path(out_path).read_text()
```

- [ ] **Step 2: Run tests and confirm failure**

Run: `pytest tests/scripts/sme_patterns/test_run.py -v`
Expected: FAIL.

- [ ] **Step 3: Write the orchestrator**

Create `scripts/sme_patterns/run.py`:

```python
"""Monthly pattern-mining orchestrator.

Wires loader + feedback merger + four clustering passes + renderer into a
single monthly run. No LLM calls, no destructive writes — output is an
idempotent Markdown + JSON pair under analytics/.
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from scripts.sme_patterns.clustering.artifact_utility import (
    ArtifactUtilityConfig,
    analyze_artifact_utility,
)
from scripts.sme_patterns.clustering.failure_patterns import (
    FailurePatternsConfig,
    cluster_failure_patterns,
)
from scripts.sme_patterns.clustering.persona_effect import (
    PersonaEffectConfig,
    analyze_persona_effect,
)
from scripts.sme_patterns.clustering.success_patterns import (
    SuccessPatternsConfig,
    cluster_success_patterns,
)
from scripts.sme_patterns.feedback_merger import merge_feedback
from scripts.sme_patterns.report.model import compose_pattern_report
from scripts.sme_patterns.report.renderer import render_pattern_report
from scripts.sme_patterns.schema import PatternReport
from scripts.sme_patterns.trace_loader import TraceLoader, TraceWindow

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunConfig:
    window_start: datetime
    window_end: datetime
    analytics_dir: Path
    rollback_glob: str = "sme_rollback_*.md"


def default_window(*, now: datetime | None = None) -> TraceWindow:
    """Return the prior calendar month."""
    now = now or datetime.utcnow()
    first_of_this = datetime(now.year, now.month, 1, 0, 0, 0)
    # last day of prior month = one second before first of this month
    end = first_of_this.replace(microsecond=0)
    # step back to previous month
    if first_of_this.month == 1:
        prior_first = datetime(first_of_this.year - 1, 12, 1, 0, 0, 0)
    else:
        prior_first = datetime(first_of_this.year, first_of_this.month - 1, 1, 0, 0, 0)
    # last moment of the prior month
    # (naive approach: subtract one second from first_of_this)
    import datetime as _dt
    prior_end = first_of_this - _dt.timedelta(seconds=1)
    return TraceWindow(start=prior_first, end=prior_end)


def _collect_rollback_links(analytics_dir: Path, glob: str, window: TraceWindow) -> list[str]:
    links: list[str] = []
    for f in sorted(analytics_dir.glob(glob)):
        # Filename pattern: sme_rollback_YYYY-MM-DD.md
        name = f.stem  # 'sme_rollback_2026-04-12'
        parts = name.rsplit("_", 1)
        if len(parts) != 2:
            continue
        try:
            ts = datetime.strptime(parts[1], "%Y-%m-%d")
        except ValueError:
            continue
        if window.start.date() <= ts.date() <= window.end.date():
            # Store as repo-relative path
            links.append(f"analytics/{f.name}")
    return links


def run_monthly_mining(
    config: RunConfig,
    *,
    list_blobs: Callable[[str], Iterable[str]],
    read_blob: Callable[[str], str],
    feedback_tracker,
    success_cfg: SuccessPatternsConfig | None = None,
    failure_cfg: FailurePatternsConfig | None = None,
    artifact_cfg: ArtifactUtilityConfig | None = None,
    persona_cfg: PersonaEffectConfig | None = None,
) -> str:
    """Execute one month of mining. Returns the written Markdown path."""
    success_cfg = success_cfg or SuccessPatternsConfig()
    failure_cfg = failure_cfg or FailurePatternsConfig()
    artifact_cfg = artifact_cfg or ArtifactUtilityConfig()
    persona_cfg = persona_cfg or PersonaEffectConfig()

    window = TraceWindow(start=config.window_start, end=config.window_end)
    loader = TraceLoader(list_blobs=list_blobs, read_blob=read_blob)

    synth_runs = list(loader.iter_synthesis_runs(window))
    query_runs = list(loader.iter_query_runs(window))
    logger.info("loaded %d synth runs and %d query runs for window %s → %s",
                len(synth_runs), len(query_runs), window.start, window.end)

    query_runs = merge_feedback(query_runs, feedback_tracker)

    successes = cluster_success_patterns(query_runs, success_cfg)
    failures = cluster_failure_patterns(query_runs, failure_cfg)
    artifacts = analyze_artifact_utility(query_runs, artifact_cfg)
    personas = analyze_persona_effect(query_runs, persona_cfg)

    rollback_links = _collect_rollback_links(config.analytics_dir, config.rollback_glob, window)

    report: PatternReport = compose_pattern_report(
        query_runs=query_runs,
        synth_runs=synth_runs,
        successes=successes,
        failures=failures,
        artifact_utility=artifacts,
        persona_effect=personas,
        training_candidates=[],  # filled later by evaluate_training_trigger
        period_start=config.window_start,
        period_end=config.window_end,
        rollback_links=rollback_links,
    )

    month_slug = config.window_start.strftime("%Y-%m")
    config.analytics_dir.mkdir(parents=True, exist_ok=True)
    md_path = config.analytics_dir / f"sme_patterns_{month_slug}.md"
    json_path = config.analytics_dir / f"sme_patterns_{month_slug}.json"

    render_pattern_report(report, md_path)
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    logger.info("wrote %s and %s", md_path, json_path)
    return str(md_path)


def _real_list_blobs(prefix: str) -> Iterable[str]:
    from src.storage.azure_blob_client import get_document_container_client
    container = get_document_container_client()
    for blob in container.list_blobs(name_starts_with=prefix):
        yield blob.name


def _real_read_blob(name: str) -> str:
    from src.storage.azure_blob_client import get_document_container_client
    container = get_document_container_client()
    client = container.get_blob_client(name)
    return client.download_blob().readall().decode("utf-8")


def _real_feedback_tracker():
    from src.intelligence.feedback_tracker import FeedbackTracker
    from src.utils.redis_client import get_redis_client  # assumed helper; fallback below

    try:
        r = get_redis_client()
    except Exception:
        logger.exception("no Redis client available; using offline stub")

        class _Stub:
            def get_profile_metrics(self, _profile_id: str):
                return {"total_queries": 0, "avg_confidence": 0.0,
                        "grounded_ratio": 0.0, "low_confidence_count": 0}

        return _Stub()
    return FeedbackTracker(r)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Monthly SME pattern-mining batch job")
    parser.add_argument("--analytics-dir", type=Path, default=Path("analytics"))
    parser.add_argument("--window-start", type=str, default=None,
                        help="ISO date; default = first day of prior month")
    parser.add_argument("--window-end", type=str, default=None,
                        help="ISO date; default = last day of prior month")
    args = parser.parse_args(argv)

    if args.window_start and args.window_end:
        start = datetime.fromisoformat(args.window_start)
        end = datetime.fromisoformat(args.window_end)
    else:
        w = default_window()
        start, end = w.start, w.end

    config = RunConfig(
        window_start=start,
        window_end=end,
        analytics_dir=args.analytics_dir,
    )

    try:
        out_path = run_monthly_mining(
            config,
            list_blobs=_real_list_blobs,
            read_blob=_real_read_blob,
            feedback_tracker=_real_feedback_tracker(),
        )
    except Exception:
        logger.exception("pattern mining failed")
        return 2

    print(f"[mine_sme_patterns] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Write the thin entry-point**

Create `scripts/mine_sme_patterns.py`:

```python
"""Thin top-level entry-point — delegates to scripts.sme_patterns.run.

Kept at top-level so operators and systemd can invoke it by name. All real
logic lives in scripts/sme_patterns/run.py.
"""
from __future__ import annotations

import sys

from scripts.sme_patterns.run import main

if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run tests and confirm pass**

Run: `pytest tests/scripts/sme_patterns/test_run.py -v`
Expected: PASS for all 3 tests.

- [ ] **Step 6: Commit**

```bash
git add scripts/sme_patterns/run.py scripts/mine_sme_patterns.py \
    tests/scripts/sme_patterns/test_run.py
git commit -m "phase6(sme-patterns): monthly-mining orchestrator + CLI"
```

---

## Task 14: End-to-end integration test

Spins up an in-memory synthetic month of traces, runs the full pipeline, and asserts the Markdown and JSON outputs are consistent. Guards against regressions where one pass's schema changes and another silently breaks.

**Files:**
- Create: `tests/scripts/sme_patterns/test_monthly_end_to_end.py`

- [ ] **Step 1: Write the test**

Create `tests/scripts/sme_patterns/test_monthly_end_to_end.py`:

```python
"""End-to-end integration for the monthly pattern-mining pipeline.

Uses in-memory Azure Blob stubs — no network I/O, no Redis. Validates the
whole chain from blob bytes to rendered Markdown + JSON.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

from scripts.sme_patterns.run import RunConfig, run_monthly_mining
from scripts.sme_patterns.schema import PatternReport
from tests.scripts.sme_patterns.fixtures.query_trace_factory import make_query_jsonl
from tests.scripts.sme_patterns.fixtures.synth_trace_factory import make_synth_jsonl


def _build_blobs() -> dict[str, str]:
    blobs: dict[str, str] = {}

    # Two synth runs (one clean, one with drops)
    blobs["sme_traces/synthesis/sub_a/prof_a/syn_clean.jsonl"] = make_synth_jsonl(
        synthesis_id="syn_clean",
        started_at=datetime(2026, 4, 2, 2, 0, 0),
        drop_count=0,
    )
    blobs["sme_traces/synthesis/sub_a/prof_a/syn_drops.jsonl"] = make_synth_jsonl(
        synthesis_id="syn_drops",
        started_at=datetime(2026, 4, 9, 2, 0, 0),
        drop_count=4,
    )

    # Successful queries
    for i in range(10):
        blobs[f"sme_traces/queries/sub_a/prof_a/2026-04-05/succ_{i}.jsonl"] = make_query_jsonl(
            query_id=f"succ_{i}",
            intent="analyze",
            rating=1,
            sme_artifacts=3,
            citation_verifier_drops=0,
            adapter_persona_role="senior financial analyst",
            captured_at=datetime(2026, 4, 5, 10, 0, 0),
        )

    # Failing recommend queries (will appear as a cluster)
    for i in range(6):
        blobs[f"sme_traces/queries/sub_a/prof_a/2026-04-12/fail_rec_{i}.jsonl"] = make_query_jsonl(
            query_id=f"fail_rec_{i}",
            intent="recommend",
            rating=-1,
            sme_artifacts=1,
            citation_verifier_drops=2,
            adapter_persona_role="experimental cfo persona",
            query_text="recommend cost reduction across SaaS stack",
            captured_at=datetime(2026, 4, 12, 10, 0, 0),
        )

    # Out-of-window query (must not be counted)
    blobs["sme_traces/queries/sub_a/prof_a/2026-03-25/skip.jsonl"] = make_query_jsonl(
        query_id="skip",
        captured_at=datetime(2026, 3, 25, 10, 0, 0),
    )

    return blobs


def test_monthly_end_to_end(tmp_path: Path):
    blobs = _build_blobs()
    list_blobs = MagicMock(side_effect=lambda prefix: [k for k in blobs if k.startswith(prefix)])
    read_blob = MagicMock(side_effect=lambda name: blobs[name])

    # Feedback tracker returns low-confidence-free metrics so merger leaves
    # explicit feedback alone.
    tracker = MagicMock()
    tracker.get_profile_metrics.return_value = {
        "total_queries": 16, "avg_confidence": 0.7,
        "grounded_ratio": 0.7, "low_confidence_count": 0,
    }

    cfg = RunConfig(
        window_start=datetime(2026, 4, 1),
        window_end=datetime(2026, 4, 30, 23, 59, 59),
        analytics_dir=tmp_path,
    )
    md_path = run_monthly_mining(
        cfg,
        list_blobs=list_blobs,
        read_blob=read_blob,
        feedback_tracker=tracker,
    )

    md_text = Path(md_path).read_text()
    # Every required section header present
    for header in (
        "## Executive summary",
        "## Success patterns",
        "## Failure patterns",
        "## Artifact utility",
        "## Persona performance",
        "## Training candidates",
    ):
        assert header in md_text, f"missing section: {header}"

    # JSON snapshot matches
    json_path = Path(md_path).with_suffix(".json")
    rep = PatternReport.model_validate_json(json_path.read_text())
    assert rep.num_query_runs == 16  # 10 success + 6 failure
    assert rep.num_synth_runs == 2
    assert len(rep.failures) >= 1
    assert len(rep.successes) >= 1

    # Persona pass fires; experimental persona is the worst
    assert rep.persona_effect, "expected at least one persona row"

    # Artifact utility rows cover all four layers
    artifact_layers = {c.evidence["layer"] for c in rep.artifact_utility}
    assert artifact_layers == {"chunks", "kg", "sme_artifacts", "url"}

    # No training candidates — single month only (stabilization requires 2+)
    assert rep.training_candidates == []
```

- [ ] **Step 2: Run the integration test**

Run: `pytest tests/scripts/sme_patterns/test_monthly_end_to_end.py -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/scripts/sme_patterns/test_monthly_end_to_end.py
git commit -m "phase6(sme-patterns): end-to-end monthly pipeline integration test"
```

---

## Task 15: systemd timer + service + shell wrapper

Monthly at 02:00 UTC on the first of each month. Matches repo convention (`systemd/docwain-vllm-fast.service`, `systemd/docwain-gpu-scheduler.service`, `deploy/docwain-app.service`).

**Files:**
- Create: `systemd/docwain-sme-pattern-mining.service`
- Create: `systemd/docwain-sme-pattern-mining.timer`
- Create: `deploy/sme-pattern-mining.sh`

- [ ] **Step 1: Write the shell wrapper**

Create `deploy/sme-pattern-mining.sh`:

```bash
#!/usr/bin/env bash
#
# Wrapper invoked by systemd.timer for the monthly SME pattern-mining job.
# Runs both the pattern-miner and the training-trigger evaluator, in order.
# No retraining is triggered automatically.

set -euo pipefail

REPO_ROOT="${DOCWAIN_REPO_ROOT:-/home/ubuntu/PycharmProjects/DocWain}"
PY="${REPO_ROOT}/.venv/bin/python"
ANALYTICS_DIR="${REPO_ROOT}/analytics"
MONTH="$(date -u +%Y-%m)"

cd "${REPO_ROOT}"

echo "[sme-pattern-mining] start ${MONTH}"

"${PY}" -m scripts.mine_sme_patterns --analytics-dir "${ANALYTICS_DIR}"

"${PY}" "${REPO_ROOT}/scripts/evaluate_training_trigger.py" \
    --reports-dir "${ANALYTICS_DIR}" \
    --out "${ANALYTICS_DIR}/training_candidates_${MONTH}.json"

echo "[sme-pattern-mining] done"
```

Make it executable:

```bash
chmod +x deploy/sme-pattern-mining.sh
```

- [ ] **Step 2: Write the systemd service**

Create `systemd/docwain-sme-pattern-mining.service`:

```
[Unit]
Description=DocWain SME Pattern Mining (monthly batch)
After=network.target

[Service]
Type=oneshot
User=ubuntu
WorkingDirectory=/home/ubuntu/PycharmProjects/DocWain
EnvironmentFile=/home/ubuntu/PycharmProjects/DocWain/.env
Environment="PATH=/home/ubuntu/PycharmProjects/DocWain/.venv/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=/home/ubuntu/PycharmProjects/DocWain/deploy/sme-pattern-mining.sh

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=docwain-sme-pattern-mining

# Hardening
NoNewPrivileges=true
ProtectSystem=strict
ReadWritePaths=/home/ubuntu/PycharmProjects/DocWain /tmp
PrivateTmp=true
```

- [ ] **Step 3: Write the systemd timer**

Create `systemd/docwain-sme-pattern-mining.timer`:

```
[Unit]
Description=Run DocWain SME Pattern Mining on the 1st of each month at 02:00 UTC

[Timer]
OnCalendar=*-*-01 02:00:00 UTC
Persistent=true
Unit=docwain-sme-pattern-mining.service

[Install]
WantedBy=timers.target
```

- [ ] **Step 4: Verify the files exist and are syntactically well-formed**

```bash
systemd-analyze verify systemd/docwain-sme-pattern-mining.service \
    systemd/docwain-sme-pattern-mining.timer 2>&1 || true
head -n 2 deploy/sme-pattern-mining.sh
```

The `systemd-analyze verify` command may warn about [Install] sections on oneshot services — that's expected; the timer is what schedules it. The key check is that neither file errors out on parse.

- [ ] **Step 5: Commit**

```bash
git add -f systemd/docwain-sme-pattern-mining.service \
    systemd/docwain-sme-pattern-mining.timer \
    deploy/sme-pattern-mining.sh
git commit -m "phase6(sme-patterns): monthly systemd timer + service + wrapper"
```

---

## Task 16: Monthly review runbook

The runbook explains how to interpret the monthly Markdown file, when to pursue a training candidate, and how rollback post-mortems integrate.

**Files:**
- Create: `analytics/README.md`

- [ ] **Step 1: Write the runbook**

Create `analytics/README.md`:

```markdown
# DocWain SME Analytics & Monthly Review Runbook

This directory holds the outputs of the Phase 6 pattern-mining loop:

- `sme_patterns_{YYYY-MM}.md` — the human-reviewable monthly findings
- `sme_patterns_{YYYY-MM}.json` — the machine-readable snapshot (same data)
- `training_candidates_{YYYY-MM}.json` — stabilized failure clusters flagged
  as candidates for sub-project F (separate, human-gated training project)
- `sme_rollback_{YYYY-MM-DD}.md` — post-mortems for any SME full rollback
  (spec Section 13.3)
- `templates/` — Jinja2 source for the monthly Markdown

## Pipeline and schedule

A systemd timer (`systemd/docwain-sme-pattern-mining.timer`) invokes
`deploy/sme-pattern-mining.sh` at 02:00 UTC on the 1st of every month.
The wrapper runs:

1. `python -m scripts.mine_sme_patterns --analytics-dir analytics` — writes
   the monthly Markdown + JSON snapshot.
2. `python scripts/evaluate_training_trigger.py --reports-dir analytics
   --out analytics/training_candidates_YYYY-MM.json` — cross-month
   stabilization eval; writes a candidate list for sub-project F.

Neither step triggers retraining. The list is evidence.

## Reviewing the monthly patterns file

Open `sme_patterns_YYYY-MM.md`. Walk the six sections in order:

### 1. Executive summary
Quick gauges. If failure clusters spike 3×+ month-over-month, pause and
diagnose before continuing the review.

### 2. Success patterns
What the SME layer is winning on. These inform:
- Where to keep investment (adapter tuning, persona refinement).
- The "gold-standard" template answers for Phase 0 regression eval.

### 3. Failure patterns
What is failing. Severity score weights:
- Explicit thumbs-down rate (40%)
- Citation-verifier drops (30%)
- Honest-compact fallback rate (30%)

For each cluster, read the `Evidence` block. Ask: does this look like a
tuning problem (persona, adapter thresholds) or a design problem (grounding
semantics, intent router, SMEVerifier logic)? Open an issue tagged
`sme-failure-cluster:<cluster_id>` in either case.

### 4. Artifact utility
The four retrieval layers' retrieval rate and positive-outcome rate. A
"dead weight" flag means a layer is pulled often but correlates with bad
outcomes. Dead-weight layers are candidates for:
- Adapter threshold bumps (confidence floor, max-hops for KG edges).
- Layer gating change (skip this layer for these intents).
- Turning the layer off for a subscription via its feature flag.

### 5. Persona performance
Per-persona SME-score proxy vs the domain baseline. Regression-flagged
personas should be compared against the adapter YAML changes in the last
30 days. Rollback the adapter via the standard per-phase rollback path if
regression holds.

### 6. Training candidates
Stabilized failure clusters with ≥2 months' presence, ≥20 total volume,
and ≥0.55 stabilization score. **This is the bridge to sub-project F.**

Decision framework per candidate:
- If the cluster reads as an **engineering** problem (wrong persona, missing
  intent handling, grounding too loose): fix in engineering first; sub-F
  stays closed.
- If the cluster persists across 2+ months after engineering fixes and the
  severity + volume remain high: convene a sub-F kickoff decision. The
  candidate record is evidence only — a human owner decides.

## Rollback post-mortems

When a full SME rollback happens (spec Section 13.3), write the post-mortem
file `sme_rollback_YYYY-MM-DD.md` into this directory. The next monthly
pattern-mining run auto-discovers and links it from
`sme_patterns_YYYY-MM.md` under the "Rollback post-mortems" section.

Minimum post-mortem contents:
1. Trigger condition (which Section 13.4 item fired).
2. Scope (which flag: per-subscription or `sme_redesign_enabled=false`).
3. What tuning was attempted before rollback.
4. What traces show (cite cluster_ids from the prior monthly report).
5. Next steps — keep engineering-first; only escalate to sub-F if traces
   say an engineering ceiling was hit.

## Memory rules applied

- **Engineering-first** — this loop produces evidence; it never triggers
  retraining.
- **Profile isolation** — clusters carry `subscription_ids`; cross-sub
  rollup only at the persona/artifact-utility level.
- **No customer data in training** — the monthly Markdown contains only
  fingerprints and cluster-level aggregates; raw query text stays in the
  Blob trace store.
- **Traces in Azure Blob, not Mongo** — loader reads JSONL blobs via the
  existing `src/storage/azure_blob_client.py`.

## Operating notes

- If the timer fires while sub-F is already in flight for a given cluster,
  the monthly report re-lists the same candidate — that's intentional.
  Candidates are idempotent.
- If a month has zero traces (e.g. master flag off), the monthly file still
  renders with "no … clusters this month" placeholders; that's a signal,
  not a failure.
- Re-run the month manually: `python -m scripts.mine_sme_patterns
  --window-start 2026-04-01 --window-end 2026-04-30T23:59:59
  --analytics-dir analytics`.
- Threshold tuning: edit the flags to `scripts/evaluate_training_trigger.py`
  (`--min-months`, `--min-volume`, `--stabilization-threshold`) in the
  wrapper once a reviewer gains enough historical data to recalibrate.
```

- [ ] **Step 2: Commit**

```bash
git add -f analytics/README.md
git commit -m "phase6(sme-patterns): monthly review runbook"
```

---

## Task 17: Self-review + exit checklist

Hard self-check before declaring Phase 6 done.

- [ ] **Step 1: Run the whole test suite**

```bash
pytest tests/scripts/sme_patterns -v
```

Every test green. If anything is red, fix it; do NOT skip with xfail/skip markers.

- [ ] **Step 2: Lint critical invariants**

```bash
# No Claude/Anthropic references crept in
grep -RIn "Anthropic\|Co-Authored-By\|Claude " scripts/sme_patterns scripts/mine_sme_patterns.py scripts/evaluate_training_trigger.py analytics/ systemd/docwain-sme-pattern-mining.* deploy/sme-pattern-mining.sh 2>/dev/null || echo "clean"
# No mongo writes in Phase 6 code
grep -RIn "mongo\|Mongo" scripts/sme_patterns scripts/mine_sme_patterns.py scripts/evaluate_training_trigger.py 2>/dev/null || echo "clean"
# No prompt code touched
git log --oneline --name-only phase6 2>/dev/null | grep "src/generation/prompts.py" && echo "VIOLATION: prompts.py touched" || echo "clean"
```

- [ ] **Step 3: Dry-run the CLI (no traces expected locally)**

```bash
mkdir -p analytics
python -m scripts.mine_sme_patterns --analytics-dir analytics \
    --window-start 2026-04-01 --window-end 2026-04-30T23:59:59 || true
```

If Azure Blob creds are unconfigured the loader emits a traceback; that's
fine — the unit tests cover the happy path. The CLI should exit without
corrupting the analytics directory.

- [ ] **Step 4: Verify the systemd units parse**

```bash
systemd-analyze verify systemd/docwain-sme-pattern-mining.service \
    systemd/docwain-sme-pattern-mining.timer 2>&1 | head
```

- [ ] **Step 5: Final commit**

```bash
git add -u
git commit --allow-empty -m "phase6(sme-patterns): self-review pass — all green"
```

---

## Phase 6 exit checklist

Every box must be genuinely ticked, not wishfully checked.

- [ ] All 17 tasks committed with passing tests.
- [ ] `pytest tests/scripts/sme_patterns -v` shows all green, zero skips/xfails.
- [ ] `scripts/mine_sme_patterns.py`, `scripts/evaluate_training_trigger.py`, and the four clustering passes all have a dedicated unit-test file and an integration test covering the full pipeline.
- [ ] Monthly Markdown is rendered from `analytics/templates/sme_patterns_template.md` via Jinja2. No Markdown formatting lives in Python code.
- [ ] Clustering is rule-based + tf-idf + k-means. No black-box ML; every cluster carries its top terms and evidence block in the rendered output.
- [ ] Trigger-condition script emits `TrainingCandidate` records. Nothing in the batch pipeline invokes retraining directly.
- [ ] systemd timer installed at `systemd/docwain-sme-pattern-mining.timer`; wrapper at `deploy/sme-pattern-mining.sh`; runs 1st of month at 02:00 UTC.
- [ ] Monthly review runbook committed at `analytics/README.md`.
- [ ] Post-mortem integration verified: the renderer emits the "Rollback post-mortems" section iff `rollback_links` is non-empty (tested in `test_renderer.py::test_renderer_embeds_rollback_links`).
- [ ] Profile isolation preserved: every `Cluster` carries `subscription_ids`; cross-sub aggregation only appears in the artifact/persona rollups.
- [ ] No Claude/Anthropic/Co-Authored-By references anywhere under `scripts/sme_patterns`, `scripts/mine_sme_patterns.py`, `scripts/evaluate_training_trigger.py`, `analytics/`, or the systemd/deploy files.
- [ ] No files under `src/generation/prompts.py` touched. No files under `src/` (production code) touched at all by Phase 6.
- [ ] First monthly run completes 30 days after Phase 4 wide rollout (spec Section 12 Phase 6 timing). This is an operator milestone, not a code milestone; record the first-run artifacts (`analytics/sme_patterns_2026-05.md` or next month in scope) in the project log when they land.
- [ ] Engineering-first invariant holds: the monthly pipeline produces evidence for sub-project F; no retraining auto-triggered.

---

## Self-review appendix

**Spec coverage check.** Every item in spec Section 11 and Section 12 Phase 6 has a task above:

- **Trace inputs** (Section 11) — Tasks 4, 5; derived contract documented up front.
- **Clustering pass 1 Success** — Task 7.
- **Clustering pass 2 Failure** — Task 8.
- **Clustering pass 3 Artifact utility** — Task 9.
- **Clustering pass 4 Persona effect** — Task 10.
- **Output `analytics/sme_patterns_{YYYY-MM}.md`** — Tasks 11, 13.
- **Analytics template in `analytics/templates/`** — Task 11.
- **Trigger conditions / stabilization** — Task 12.
- **Scheduling (systemd timer)** — Task 15.
- **Monthly review runbook** — Task 16.
- **Post-mortem integration** — Tasks 11 (renderer section), 13 (`_collect_rollback_links`), 16 (runbook).
- **Phase 6 exit** — exit checklist above.

**Type consistency.** `SynthesisRun`, `QueryRun`, `Cluster`, `PatternReport`, `TrainingCandidate` live once in `scripts/sme_patterns/schema.py` and are imported by every other module. `cluster_success_patterns` / `cluster_failure_patterns` / `analyze_artifact_utility` / `analyze_persona_effect` all return `list[Cluster]` with the correct `ClusterType`. `run_monthly_mining` composes `PatternReport` exactly once.

**Placeholder scan.** No `TODO`, `TBD`, `FIXME` in task code. Trace-record contract notes precisely what Phase 1 must deliver, and the integration test asserts the contract.

**No redefinition.** Each file is created exactly once. No task claims to modify a file another task creates in a later step.

**Engineering-first invariant.** No Python code in this plan calls any retraining API, kicks off any fine-tune, or writes to `finetune_artifacts/` — verify by grep in Task 17 Step 2.

**Memory-rule conformance.**
- No Claude/Anthropic references (Task 17 grep).
- No timeouts on internal paths — the mining loop has none; the Azure Blob SDK's default is a per-request safety, not a quality cutoff.
- Traces live in Blob — loader's only sources are `sme_traces/synthesis/` and `sme_traces/queries/` prefixes.
- `src/generation/prompts.py` untouched — integration test runs without importing it.
- Engineering-first — the evaluator is purely evidentiary; sub-F remains a separate human-gated project.
- Profile isolation — every Cluster includes `subscription_ids`.
- No customer data — the rendered Markdown carries only fingerprints and cluster-level aggregates; raw query text lives only in the Blob traces (already synthetic-only upstream).
