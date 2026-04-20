# DocWain SME Phase 0 — Measurement Baseline Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Capture a trustworthy before-picture of DocWain's current answer quality, latency, and SME behavior so that every subsequent phase of sub-project A is gated on real numbers.

**Architecture:** Zero production code changes. New eval tooling under `scripts/sme_eval/` hits production DocWain's `/api/ask` endpoint against a versioned 600-query eval set (`tests/sme_evalset_v1/`, 100 queries × 6 domains). Runs existing RAGAS metrics plus six new reasoning metrics as pure measurement. Outputs a dated baseline snapshot and a nightly-regression JSON file. Extends the existing `scripts/ragas_evaluator.py` and `tests/ragas_metrics.json` rather than replacing them.

**Tech Stack:** Python 3.12, `httpx` for API calls (already used in `scripts/intensive_test.py`), `pydantic` for schemas, `pytest` for tests, `pyyaml` for eval set config, `pandas`/`numpy` for percentile aggregation, existing DocWain LLM gateway (`src/serving/model_router.py` or its client) for the LLM-judge metric.

**Related spec:** `docs/superpowers/specs/2026-04-20-docwain-profile-sme-reasoning-design.md` Sections 10 (measurement framework), 12 (Phase 0 exit gate).

**Memory rules that constrain this plan:**
- "Measure Before You Change" — this plan IS the measurement infrastructure
- "No Customer Data in Training" — eval queries must be synthetic, never real customer queries
- "No Claude Attribution" — no Anthropic/Claude references in commits, code, or docs
- "No Timeouts; Use Efficiency" — applies to runtime, not this measurement tool; the eval runner uses httpx timeouts (external I/O safety) only

---

## File structure

```
scripts/sme_eval/                              [NEW]
├── __init__.py
├── schema.py                                  # EvalQuery, EvalResult, MetricResult pydantic models
├── query_runner.py                            # Hits /api/ask, captures response+latency
├── result_store.py                            # Persists individual results (JSONL)
├── metrics/
│   ├── __init__.py
│   ├── _base.py                               # Metric base class
│   ├── ragas_wrapper.py                       # Calls existing scripts/ragas_evaluator.py
│   ├── recommendation_groundedness.py
│   ├── cross_doc_integration_rate.py
│   ├── insight_novelty.py
│   ├── sme_persona_consistency.py             # LLM-judge based
│   ├── verified_removal_rate.py
│   └── sme_artifact_hit_rate.py
├── aggregate.py                               # p50/p95/p99 + per-metric rollup
├── human_rating.py                            # CSV export/import for SME expert rating
└── run_baseline.py                            # Main CLI; orchestrates everything

tests/sme_evalset_v1/                          [NEW]
├── README.md
├── fixtures/
│   └── test_profiles.yaml                     # Test subscription/profile IDs per domain
└── queries/
    ├── finance.yaml                           # 100 queries
    ├── legal.yaml                             # 100 queries
    ├── hr.yaml                                # 100 queries
    ├── medical.yaml                           # 100 queries
    ├── it_support.yaml                        # 100 queries
    └── generic.yaml                           # 100 queries

tests/scripts/sme_eval/                        [NEW]
├── __init__.py
├── conftest.py
├── test_schema.py
├── test_query_runner.py
├── test_result_store.py
├── test_aggregate.py
├── test_human_rating.py
└── metrics/
    ├── __init__.py
    ├── test_ragas_wrapper.py
    ├── test_recommendation_groundedness.py
    ├── test_cross_doc_integration_rate.py
    ├── test_insight_novelty.py
    ├── test_sme_persona_consistency.py
    ├── test_verified_removal_rate.py
    └── test_sme_artifact_hit_rate.py

tests/sme_metrics_baseline_{YYYY-MM-DD}.json   # Frozen baseline snapshot (committed)
tests/sme_metrics_daily.json                   # Overwritten nightly (gitignored except on CI)
```

Each file does one thing. `schema.py` owns types; `query_runner.py` owns API calls; each metric file owns exactly one metric. This isolation matters because the metrics will evolve as domain adapters land in later phases — independent metric files let us modify one without touching others.

---

## Task 1: Preflight audit and directory scaffolding

**Files:**
- Create: `scripts/sme_eval/__init__.py` (empty)
- Create: `scripts/sme_eval/metrics/__init__.py` (empty)
- Create: `tests/sme_evalset_v1/README.md`
- Create: `tests/scripts/sme_eval/__init__.py` (empty)
- Create: `tests/scripts/sme_eval/metrics/__init__.py` (empty)
- Audit only: `scripts/ragas_evaluator.py`, `scripts/intensive_test.py`, `tests/ragas_metrics.json`, `tests/eval/test_eval_runner.py`

- [ ] **Step 1: Read the existing eval tooling**

Read these files and note what is reusable vs. what needs wrapping:
- `scripts/ragas_evaluator.py` — 249 lines; computes 4 RAGAS metrics against `/tmp/intensive_test_results.json`. Has banned-phrase heuristics and regex-based hallucination markers. **Reusable as a library if we call its functions directly.**
- `scripts/intensive_test.py` — 548 lines; hits `/api/ask` at `localhost:8000` with hardcoded subscription/profile IDs for recruit + contract data. Uses httpx with 600s timeout. **Reusable as a pattern for the query runner; do NOT import directly.**
- `tests/ragas_metrics.json` — current baseline output. Fields: `answer_faithfulness`, `hallucination_rate`, `context_recall`, `grounding_bypass_rate`.
- `tests/eval/test_eval_runner.py` — existing pytest for an `eval_runner`. Check whether it's reusable.

- [ ] **Step 2: Create empty package files**

```bash
mkdir -p scripts/sme_eval/metrics tests/sme_evalset_v1/{queries,fixtures} tests/scripts/sme_eval/metrics
touch scripts/sme_eval/__init__.py
touch scripts/sme_eval/metrics/__init__.py
touch tests/scripts/sme_eval/__init__.py
touch tests/scripts/sme_eval/metrics/__init__.py
```

- [ ] **Step 3: Write the evalset README**

Create `tests/sme_evalset_v1/README.md`:

```markdown
# DocWain SME Evaluation Set v1

Versioned eval set for DocWain's Profile-SME reasoning sub-project (A).
100 queries per major domain × 6 domains = 600 queries total.

## Domains
- finance — financial SME queries (cost, revenue, trends, recommendations)
- legal — contract/obligation/party queries
- hr — employee/policy/benefits queries
- medical — diagnosis/treatment/record queries
- it_support — symptom/troubleshooting/fix queries
- generic — domain-agnostic document queries

## Query file schema
See `scripts/sme_eval/schema.py` for the authoritative schema.

## Fixture profiles
See `fixtures/test_profiles.yaml` for test subscription/profile IDs.
All test profiles MUST contain synthetic data only — no customer documents.

## Running the baseline
See `scripts/sme_eval/run_baseline.py --help`.

## When to regenerate
The eval set is frozen at v1 for Phase 0 baseline. Subsequent phases may
add queries as `tests/sme_evalset_v2/` — never modify v1 in place.
```

- [ ] **Step 4: Commit the scaffolding**

```bash
git add -f scripts/sme_eval/__init__.py scripts/sme_eval/metrics/__init__.py \
    tests/sme_evalset_v1/README.md \
    tests/scripts/sme_eval/__init__.py tests/scripts/sme_eval/metrics/__init__.py
git commit -m "phase0(sme-eval): scaffold eval tooling directories"
```

---

## Task 2: Eval query schema

**Files:**
- Create: `scripts/sme_eval/schema.py`
- Create: `tests/scripts/sme_eval/test_schema.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/scripts/sme_eval/test_schema.py`:

```python
"""Tests for eval query schema."""
from datetime import datetime

import pytest
from pydantic import ValidationError

from scripts.sme_eval.schema import (
    EvalQuery,
    EvalResult,
    LatencyBreakdown,
    MetricResult,
)


def _valid_query_dict():
    return {
        "query_id": "finance_001",
        "query_text": "Summarize our Q3 revenue trends.",
        "intent": "analyze",
        "profile_domain": "finance",
        "subscription_id": "test_sub_finance",
        "profile_id": "test_prof_finance_1",
        "expected_behavior": "Should identify QoQ trend and cite Q1/Q2/Q3 reports",
        "tags": ["trend", "revenue"],
    }


def test_eval_query_valid():
    q = EvalQuery(**_valid_query_dict())
    assert q.query_id == "finance_001"
    assert q.intent == "analyze"
    assert q.profile_domain == "finance"


def test_eval_query_rejects_missing_required_field():
    d = _valid_query_dict()
    del d["query_text"]
    with pytest.raises(ValidationError):
        EvalQuery(**d)


def test_eval_query_rejects_invalid_domain():
    d = _valid_query_dict()
    d["profile_domain"] = "rocket_science"
    with pytest.raises(ValidationError):
        EvalQuery(**d)


def test_eval_query_tags_default_empty():
    d = _valid_query_dict()
    del d["tags"]
    q = EvalQuery(**d)
    assert q.tags == []


def test_latency_breakdown_required_fields():
    lb = LatencyBreakdown(ttft_ms=820.5, total_ms=4200.0)
    assert lb.ttft_ms == 820.5
    assert lb.total_ms == 4200.0


def test_eval_result_serializes_to_dict():
    q = EvalQuery(**_valid_query_dict())
    r = EvalResult(
        query=q,
        response_text="Q3 revenue rose 12%...",
        sources=[{"doc_id": "d1", "chunk_id": "c1"}],
        metadata={"grounded": True},
        latency=LatencyBreakdown(ttft_ms=800.0, total_ms=4000.0),
        run_id="run_20260420",
        captured_at=datetime(2026, 4, 20, 10, 0, 0),
        api_status=200,
    )
    d = r.model_dump(mode="json")
    assert d["query"]["query_id"] == "finance_001"
    assert d["latency"]["ttft_ms"] == 800.0
    assert d["api_status"] == 200


def test_metric_result_has_value_and_details():
    m = MetricResult(
        metric_name="recommendation_groundedness",
        value=0.92,
        details={"passed": 46, "failed": 4},
    )
    assert m.metric_name == "recommendation_groundedness"
    assert m.value == 0.92
    assert m.details["passed"] == 46
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/scripts/sme_eval/test_schema.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.sme_eval.schema'`

- [ ] **Step 3: Write the schema**

Create `scripts/sme_eval/schema.py`:

```python
"""Schema for DocWain SME evaluation set.

Pydantic models are the single source of truth for eval queries, results,
and metric outputs. The YAML query files in tests/sme_evalset_v1/queries/
must deserialize cleanly into EvalQuery instances.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

Domain = Literal["finance", "legal", "hr", "medical", "it_support", "generic"]

Intent = Literal[
    # Existing intents from src/generation/prompts.py TASK_FORMATS
    "greeting", "identity", "lookup", "list", "count",
    "summarize", "compare", "overview", "investigate",
    "extract", "aggregate",
    # New intents introduced by sub-project A
    "diagnose", "analyze", "recommend",
]


class EvalQuery(BaseModel):
    """One query in the evaluation set."""

    query_id: str
    query_text: str
    intent: Intent
    profile_domain: Domain
    subscription_id: str
    profile_id: str
    expected_behavior: str = ""
    tags: list[str] = Field(default_factory=list)

    # Optional — populated for queries that test URL-as-prompt
    urls_in_query: list[str] = Field(default_factory=list)
    # Optional — human-graded correctness signal from prior runs
    gold_answer_snippets: list[str] = Field(default_factory=list)


class LatencyBreakdown(BaseModel):
    """Timing captured per query."""

    ttft_ms: float | None = None  # Time-to-first-token; None if non-streaming
    total_ms: float
    retrieval_ms: float | None = None
    generation_ms: float | None = None


class EvalResult(BaseModel):
    """One query run against DocWain, fully captured."""

    query: EvalQuery
    response_text: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    latency: LatencyBreakdown
    run_id: str
    captured_at: datetime
    api_status: int
    api_error: str | None = None


class MetricResult(BaseModel):
    """One metric's computed value over an EvalResult or a batch."""

    metric_name: str
    value: float
    details: dict[str, Any] = Field(default_factory=dict)


class BaselineSnapshot(BaseModel):
    """Full baseline captured on one run; frozen and committed."""

    run_id: str
    captured_at: datetime
    git_sha: str
    api_base_url: str
    num_queries: int
    per_domain_counts: dict[Domain, int]

    # Metric rollups
    ragas: dict[str, float]  # faithfulness, hallucination_rate, etc.
    sme_metrics: dict[str, MetricResult]  # keyed by metric_name

    # Latency percentiles, per-intent
    latency_p50_per_intent: dict[Intent, float]
    latency_p95_per_intent: dict[Intent, float]
    latency_p99_per_intent: dict[Intent, float]

    # Human rating rollup (populated after expert rating pass)
    human_rated_sme_score_avg: float | None = None
    human_rated_count: int = 0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/scripts/sme_eval/test_schema.py -v`
Expected: PASS for all 7 tests

- [ ] **Step 5: Commit**

```bash
git add scripts/sme_eval/schema.py tests/scripts/sme_eval/test_schema.py
git commit -m "phase0(sme-eval): eval query, result, metric, snapshot schemas"
```

---

## Task 3: Test profile fixtures

**Files:**
- Create: `tests/sme_evalset_v1/fixtures/test_profiles.yaml`

This file defines the test subscriptions and profiles the eval runner hits. These profiles must pre-exist in production DocWain with synthetic documents ingested. Profile creation / doc ingest is out of scope for this plan — it's a one-time operator setup. The plan captures the contract.

- [ ] **Step 1: Write the fixtures file**

Create `tests/sme_evalset_v1/fixtures/test_profiles.yaml`:

```yaml
# DocWain SME eval — test subscription/profile fixtures.
#
# These profiles MUST exist in production DocWain before running the baseline.
# Each profile contains ≥10 synthetic documents of the corresponding domain.
# NO customer documents in test profiles. See memory rule: No Customer Data in Training.
#
# Operator setup (one-time, out of scope for this plan):
#   1. Create test subscription via existing profiles API
#   2. Create 6 test profiles under the subscription (one per domain)
#   3. Upload 10+ synthetic documents per profile
#   4. Run full ingest: screening + embedding + KG build
#   5. Verify PIPELINE_TRAINING_COMPLETED on every document
#   6. Record the subscription_id and profile_id values below

version: 1

# Single test subscription hosting all six test profiles
test_subscription:
  subscription_id: "REPLACE_WITH_REAL_TEST_SUBSCRIPTION_ID"
  description: "DocWain SME eval v1 — synthetic documents only"

profiles:
  finance:
    profile_id: "REPLACE_WITH_REAL_FINANCE_PROFILE_ID"
    description: "10+ synthetic quarterly reports, invoices, P&L statements"
    min_document_count: 10

  legal:
    profile_id: "REPLACE_WITH_REAL_LEGAL_PROFILE_ID"
    description: "10+ synthetic contracts, NDAs, employment agreements"
    min_document_count: 10

  hr:
    profile_id: "REPLACE_WITH_REAL_HR_PROFILE_ID"
    description: "10+ synthetic resumes, employee handbooks, benefits docs"
    min_document_count: 10

  medical:
    profile_id: "REPLACE_WITH_REAL_MEDICAL_PROFILE_ID"
    description: "10+ synthetic patient records, discharge notes, prescriptions"
    min_document_count: 10

  it_support:
    profile_id: "REPLACE_WITH_REAL_ITSUPPORT_PROFILE_ID"
    description: "10+ synthetic troubleshooting guides, incident reports, KB articles"
    min_document_count: 10

  generic:
    profile_id: "REPLACE_WITH_REAL_GENERIC_PROFILE_ID"
    description: "10+ synthetic mixed-domain documents"
    min_document_count: 10

# API endpoint under test
api:
  base_url: "http://localhost:8000"  # override via DOCWAIN_API_URL env
  path: "/api/ask"
  user_id: "sme_eval_baseline@docwain.internal"
```

- [ ] **Step 2: Commit**

```bash
git add -f tests/sme_evalset_v1/fixtures/test_profiles.yaml
git commit -m "phase0(sme-eval): test profile fixtures contract"
```

Note: actual subscription_id/profile_id values are filled in by the operator at baseline run time and NOT committed (see Task 15 which documents this).

---

## Task 4: Query runner

**Files:**
- Create: `scripts/sme_eval/query_runner.py`
- Create: `tests/scripts/sme_eval/test_query_runner.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/scripts/sme_eval/test_query_runner.py`:

```python
"""Tests for the query runner."""
from datetime import datetime
from unittest.mock import patch, MagicMock

import httpx
import pytest

from scripts.sme_eval.query_runner import QueryRunner, RunnerConfig
from scripts.sme_eval.schema import EvalQuery, LatencyBreakdown


@pytest.fixture
def sample_query():
    return EvalQuery(
        query_id="finance_001",
        query_text="Summarize Q3 revenue trends.",
        intent="analyze",
        profile_domain="finance",
        subscription_id="test_sub",
        profile_id="test_prof",
    )


@pytest.fixture
def config():
    return RunnerConfig(
        base_url="http://localhost:8000",
        path="/api/ask",
        user_id="test@docwain.internal",
        fetch_timeout_s=120.0,
    )


def _mock_response(status=200, payload=None):
    r = MagicMock(spec=httpx.Response)
    r.status_code = status
    r.json.return_value = payload or {
        "payload": {
            "response": "Q3 revenue rose 12%.",
            "sources": [{"doc_id": "d1", "chunk_id": "c1"}],
            "grounded": True,
            "context_found": True,
            "metadata": {},
        }
    }
    r.raise_for_status = MagicMock()
    return r


def test_runner_executes_one_query(config, sample_query):
    runner = QueryRunner(config)

    with patch.object(runner._client, "post", return_value=_mock_response()) as mock_post:
        result = runner.run_one(sample_query, run_id="run_test")

    assert result.query.query_id == "finance_001"
    assert result.response_text == "Q3 revenue rose 12%."
    assert result.api_status == 200
    assert result.latency.total_ms > 0
    assert result.run_id == "run_test"
    assert isinstance(result.captured_at, datetime)

    # Correct payload submitted
    call_args = mock_post.call_args
    assert call_args[1]["json"]["query"] == "Summarize Q3 revenue trends."
    assert call_args[1]["json"]["subscription_id"] == "test_sub"
    assert call_args[1]["json"]["profile_id"] == "test_prof"


def test_runner_captures_http_error(config, sample_query):
    runner = QueryRunner(config)
    err_response = _mock_response(status=500, payload={"detail": "server broke"})
    err_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "500", request=MagicMock(), response=err_response
    )

    with patch.object(runner._client, "post", return_value=err_response):
        result = runner.run_one(sample_query, run_id="run_test")

    assert result.api_status == 500
    assert result.api_error is not None
    assert result.response_text == ""


def test_runner_captures_network_timeout(config, sample_query):
    runner = QueryRunner(config)
    with patch.object(runner._client, "post", side_effect=httpx.TimeoutException("timed out")):
        result = runner.run_one(sample_query, run_id="run_test")

    assert result.api_status == 0
    assert "timed out" in result.api_error.lower()


def test_runner_batch_runs_all(config, sample_query):
    runner = QueryRunner(config)
    queries = [sample_query, sample_query.model_copy(update={"query_id": "finance_002"})]

    with patch.object(runner._client, "post", return_value=_mock_response()):
        results = list(runner.run_batch(queries, run_id="run_test"))

    assert len(results) == 2
    assert {r.query.query_id for r in results} == {"finance_001", "finance_002"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/scripts/sme_eval/test_query_runner.py -v`
Expected: FAIL — module does not exist yet

- [ ] **Step 3: Write the runner**

Create `scripts/sme_eval/query_runner.py`:

```python
"""Query runner: hits DocWain /api/ask and captures results + latency.

Extracts the pattern from scripts/intensive_test.py into a reusable, testable
component. The runner uses httpx with a per-request safety timeout — this is
the ONLY timeout in the eval path, and it exists to prevent a dead server
from blocking the whole baseline. It is NOT a response-quality cutoff.
"""
from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime

import httpx

from scripts.sme_eval.schema import EvalQuery, EvalResult, LatencyBreakdown


@dataclass(frozen=True)
class RunnerConfig:
    base_url: str
    path: str = "/api/ask"
    user_id: str = "sme_eval@docwain.internal"
    fetch_timeout_s: float = 120.0  # per-operation safety; not a quality cutoff


class QueryRunner:
    """Synchronous query runner. Hits DocWain /api/ask once per query.

    Sequential by design — parallelism would confound latency measurement.
    """

    def __init__(self, config: RunnerConfig):
        self._config = config
        self._client = httpx.Client(timeout=config.fetch_timeout_s)

    def run_one(self, query: EvalQuery, *, run_id: str) -> EvalResult:
        """Run one query. Captures response or error; never raises."""
        payload = {
            "query": query.query_text,
            "subscription_id": query.subscription_id,
            "profile_id": query.profile_id,
            "user_id": self._config.user_id,
        }

        url = f"{self._config.base_url}{self._config.path}"
        start = time.perf_counter()
        captured_at = datetime.utcnow()

        try:
            response = self._client.post(url, json=payload)
            total_ms = (time.perf_counter() - start) * 1000.0
            response.raise_for_status()
            body = response.json()
            resp_payload = body.get("payload", body)
            return EvalResult(
                query=query,
                response_text=resp_payload.get("response", "") or "",
                sources=resp_payload.get("sources", []) or [],
                metadata=resp_payload.get("metadata", {}) or {},
                latency=LatencyBreakdown(total_ms=total_ms),
                run_id=run_id,
                captured_at=captured_at,
                api_status=response.status_code,
                api_error=None,
            )
        except httpx.HTTPStatusError as e:
            total_ms = (time.perf_counter() - start) * 1000.0
            return EvalResult(
                query=query,
                response_text="",
                sources=[],
                metadata={},
                latency=LatencyBreakdown(total_ms=total_ms),
                run_id=run_id,
                captured_at=captured_at,
                api_status=e.response.status_code,
                api_error=f"HTTPStatusError: {e}",
            )
        except (httpx.TimeoutException, httpx.NetworkError, httpx.ConnectError) as e:
            total_ms = (time.perf_counter() - start) * 1000.0
            return EvalResult(
                query=query,
                response_text="",
                sources=[],
                metadata={},
                latency=LatencyBreakdown(total_ms=total_ms),
                run_id=run_id,
                captured_at=captured_at,
                api_status=0,
                api_error=f"{type(e).__name__}: {e}",
            )

    def run_batch(self, queries: Iterable[EvalQuery], *, run_id: str) -> Iterator[EvalResult]:
        """Run queries sequentially. Yields results as they complete."""
        for q in queries:
            yield self.run_one(q, run_id=run_id)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "QueryRunner":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/scripts/sme_eval/test_query_runner.py -v`
Expected: PASS for all 4 tests

- [ ] **Step 5: Commit**

```bash
git add scripts/sme_eval/query_runner.py tests/scripts/sme_eval/test_query_runner.py
git commit -m "phase0(sme-eval): query runner with error capture"
```

---

## Task 5: Result persistence (JSONL store)

**Files:**
- Create: `scripts/sme_eval/result_store.py`
- Create: `tests/scripts/sme_eval/test_result_store.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/scripts/sme_eval/test_result_store.py`:

```python
"""Tests for the result store."""
from datetime import datetime
from pathlib import Path

from scripts.sme_eval.result_store import JsonlResultStore
from scripts.sme_eval.schema import EvalQuery, EvalResult, LatencyBreakdown


def _result(qid="finance_001") -> EvalResult:
    return EvalResult(
        query=EvalQuery(
            query_id=qid,
            query_text="q",
            intent="lookup",
            profile_domain="finance",
            subscription_id="s",
            profile_id="p",
        ),
        response_text="r",
        sources=[],
        metadata={},
        latency=LatencyBreakdown(total_ms=100.0),
        run_id="run_test",
        captured_at=datetime(2026, 4, 20, 10, 0, 0),
        api_status=200,
    )


def test_store_appends_and_reads(tmp_path: Path):
    store = JsonlResultStore(tmp_path / "results.jsonl")
    store.append(_result("a"))
    store.append(_result("b"))

    loaded = list(store.iter_all())
    assert len(loaded) == 2
    assert {r.query.query_id for r in loaded} == {"a", "b"}


def test_store_creates_parent_dirs(tmp_path: Path):
    path = tmp_path / "nested" / "deeply" / "results.jsonl"
    store = JsonlResultStore(path)
    store.append(_result())
    assert path.exists()


def test_store_is_append_only(tmp_path: Path):
    store = JsonlResultStore(tmp_path / "r.jsonl")
    store.append(_result("a"))
    store.append(_result("b"))

    # Open a new store instance against same path — should read existing
    store2 = JsonlResultStore(tmp_path / "r.jsonl")
    loaded = list(store2.iter_all())
    assert len(loaded) == 2


def test_store_filter_by_run_id(tmp_path: Path):
    store = JsonlResultStore(tmp_path / "r.jsonl")
    r1 = _result("a")
    r2 = _result("b")
    r2 = r2.model_copy(update={"run_id": "run_other"})
    store.append(r1)
    store.append(r2)

    loaded = list(store.iter_run("run_test"))
    assert len(loaded) == 1
    assert loaded[0].query.query_id == "a"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/scripts/sme_eval/test_result_store.py -v`
Expected: FAIL — module does not exist

- [ ] **Step 3: Write the result store**

Create `scripts/sme_eval/result_store.py`:

```python
"""JSONL-backed result store for eval runs.

One line per EvalResult. Append-only. Small enough to ship in-repo for
the 600-query baseline (~6 MB); large enough volumes later could move to
Blob following the storage-separation rule.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from scripts.sme_eval.schema import EvalResult


class JsonlResultStore:
    """Append-only JSONL store for EvalResult records."""

    def __init__(self, path: Path | str):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def append(self, result: EvalResult) -> None:
        line = result.model_dump_json()
        with self._path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def iter_all(self) -> Iterator[EvalResult]:
        if not self._path.exists():
            return
        with self._path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield EvalResult.model_validate_json(line)

    def iter_run(self, run_id: str) -> Iterator[EvalResult]:
        for r in self.iter_all():
            if r.run_id == run_id:
                yield r
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/scripts/sme_eval/test_result_store.py -v`
Expected: PASS for all 4 tests

- [ ] **Step 5: Commit**

```bash
git add scripts/sme_eval/result_store.py tests/scripts/sme_eval/test_result_store.py
git commit -m "phase0(sme-eval): JSONL result store"
```

---

## Task 6: Metric base class + RAGAS wrapper

**Files:**
- Create: `scripts/sme_eval/metrics/_base.py`
- Create: `scripts/sme_eval/metrics/ragas_wrapper.py`
- Create: `tests/scripts/sme_eval/metrics/test_ragas_wrapper.py`

- [ ] **Step 1: Write the base class**

Create `scripts/sme_eval/metrics/_base.py`:

```python
"""Base class for SME eval metrics.

Each metric implements compute(results) -> MetricResult. Metrics operate
over a full batch of EvalResult records and return a single aggregated
score with per-query details.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from scripts.sme_eval.schema import EvalResult, MetricResult


class Metric(ABC):
    """A metric computed over a batch of eval results."""

    name: str  # must be set on subclass

    @abstractmethod
    def compute(self, results: Iterable[EvalResult]) -> MetricResult:
        """Compute the metric value over a batch.

        Returns a MetricResult whose `value` is the aggregated score in
        [0.0, 1.0] (or natural unit if non-fractional) and `details`
        carries per-query breakdowns needed for debugging.
        """
```

- [ ] **Step 2: Write the failing RAGAS wrapper tests**

Create `tests/scripts/sme_eval/metrics/test_ragas_wrapper.py`:

```python
"""Tests for the RAGAS wrapper."""
from datetime import datetime
from unittest.mock import patch

import pytest

from scripts.sme_eval.metrics.ragas_wrapper import RagasMetrics
from scripts.sme_eval.schema import EvalQuery, EvalResult, LatencyBreakdown


def _result(qid, response, sources=None, grounded=True):
    return EvalResult(
        query=EvalQuery(
            query_id=qid,
            query_text="q",
            intent="lookup",
            profile_domain="finance",
            subscription_id="s",
            profile_id="p",
        ),
        response_text=response,
        sources=sources or [{"doc_id": "d1", "chunk_id": "c1"}],
        metadata={"grounded": grounded},
        latency=LatencyBreakdown(total_ms=100.0),
        run_id="run_test",
        captured_at=datetime(2026, 4, 20, 10, 0, 0),
        api_status=200,
    )


def test_ragas_wrapper_computes_four_metrics():
    metric = RagasMetrics()
    results = [
        _result("a", "The answer is 42."),
        _result("b", "The answer is 17."),
    ]
    batch = metric.compute(results)

    assert batch.metric_name == "ragas"
    assert "answer_faithfulness" in batch.details
    assert "hallucination_rate" in batch.details
    assert "context_recall" in batch.details
    assert "grounding_bypass_rate" in batch.details
    # Value is the faithfulness score (primary gate metric)
    assert 0.0 <= batch.value <= 1.0


def test_ragas_wrapper_flags_hallucination_markers():
    metric = RagasMetrics()
    hallucinating = _result("a", "As an AI language model, I cannot access...")
    clean = _result("b", "The answer is 42.")
    batch = metric.compute([hallucinating, clean])

    # At least one result flagged as hallucination
    assert batch.details["hallucination_rate"] > 0.0


def test_ragas_wrapper_empty_batch_returns_zero():
    metric = RagasMetrics()
    batch = metric.compute([])
    assert batch.value == 0.0
    assert batch.details["num_results"] == 0
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/scripts/sme_eval/metrics/test_ragas_wrapper.py -v`
Expected: FAIL — module does not exist

- [ ] **Step 4: Write the RAGAS wrapper**

Create `scripts/sme_eval/metrics/ragas_wrapper.py`:

```python
"""Wraps the existing scripts/ragas_evaluator.py heuristics.

We intentionally do NOT shell out to the old script — we import and reuse its
helper functions where available. This keeps the 4 existing RAGAS metrics
consistent with the historical tests/ragas_metrics.json baseline.
"""
from __future__ import annotations

import re
from collections.abc import Iterable

from scripts.sme_eval.metrics._base import Metric
from scripts.sme_eval.schema import EvalResult, MetricResult

# Banned phrases aligned with scripts/ragas_evaluator.py BANNED list (kept in sync)
_HALLUCINATION_MARKERS: tuple[str, ...] = (
    "as an ai",
    "i don't have access",
    "i cannot",
    "i'm unable",
    "unfortunately, i",
    "i apologize",
    "as a language model",
    "missing_reason",
    "section_id",
    "chunk_type",
    "page_start",
    "embedding_text",
    "canonical_text",
)

# Tokens that indicate the response bypassed grounding (e.g., template leakage)
_GROUNDING_BYPASS_MARKERS: tuple[str, ...] = (
    "tool:resumes",
    "tool:medical",
    "tool:insights",
    "tool:lawhere",
    "tool:email",
    "tool:action",
)


def _has_any_marker(text: str, markers: tuple[str, ...]) -> bool:
    low = text.lower()
    return any(m in low for m in markers)


def _word_overlap(text: str, expected_snippets: list[str]) -> float:
    """Lightweight context-recall proxy: fraction of expected snippets present."""
    if not expected_snippets:
        return 1.0
    low = text.lower()
    found = sum(1 for s in expected_snippets if s.lower() in low)
    return found / len(expected_snippets)


def _response_cites_evidence(result: EvalResult) -> bool:
    """Faithfulness proxy: either the metadata flags grounded, or the response
    contains a doc/chunk reference that resolves to the result's sources."""
    meta_grounded = bool(result.metadata.get("grounded", False))
    if meta_grounded and result.sources:
        return True
    # Check for inline citation pattern [doc§section] or [source N]
    cite_pattern = re.compile(r"\[([^\]]+?)\s*§\s*[^\]]+?\]|\[source\s+\d+\]", re.I)
    if cite_pattern.search(result.response_text):
        return True
    return False


class RagasMetrics(Metric):
    """Computes the four legacy RAGAS metrics used in tests/ragas_metrics.json."""

    name = "ragas"

    def compute(self, results: Iterable[EvalResult]) -> MetricResult:
        results = list(results)
        if not results:
            return MetricResult(
                metric_name=self.name,
                value=0.0,
                details={
                    "num_results": 0,
                    "answer_faithfulness": 0.0,
                    "hallucination_rate": 0.0,
                    "context_recall": 0.0,
                    "grounding_bypass_rate": 0.0,
                },
            )

        n = len(results)
        faithful = sum(1 for r in results if _response_cites_evidence(r))
        hallucinating = sum(
            1
            for r in results
            if _has_any_marker(r.response_text, _HALLUCINATION_MARKERS)
        )
        bypass = sum(
            1
            for r in results
            if _has_any_marker(r.response_text, _GROUNDING_BYPASS_MARKERS)
        )
        recall = (
            sum(_word_overlap(r.response_text, r.query.gold_answer_snippets) for r in results) / n
        )

        faithfulness = faithful / n

        return MetricResult(
            metric_name=self.name,
            value=faithfulness,
            details={
                "num_results": n,
                "answer_faithfulness": faithfulness,
                "hallucination_rate": hallucinating / n,
                "context_recall": recall,
                "grounding_bypass_rate": bypass / n,
            },
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/scripts/sme_eval/metrics/test_ragas_wrapper.py -v`
Expected: PASS for all 3 tests

- [ ] **Step 6: Commit**

```bash
git add scripts/sme_eval/metrics/_base.py scripts/sme_eval/metrics/ragas_wrapper.py \
    tests/scripts/sme_eval/metrics/test_ragas_wrapper.py
git commit -m "phase0(sme-eval): metric base class + RAGAS wrapper"
```

---

## Task 7: recommendation_groundedness metric

Measures the fraction of `recommend`-intent responses where every recommendation traces to a Recommendation Bank item *or* exposes its ad-hoc reasoning explicitly. At Phase 0 the Recommendation Bank does not exist yet, so every recommendation must either quote evidence or expose reasoning — else it counts as ungrounded. This gives the before-picture: how often does current DocWain make recommendations without grounding them? Likely low, since current DocWain rarely makes recommendations at all.

**Files:**
- Create: `scripts/sme_eval/metrics/recommendation_groundedness.py`
- Create: `tests/scripts/sme_eval/metrics/test_recommendation_groundedness.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/scripts/sme_eval/metrics/test_recommendation_groundedness.py`:

```python
from datetime import datetime

from scripts.sme_eval.metrics.recommendation_groundedness import (
    RecommendationGroundedness,
    extract_recommendation_sentences,
)
from scripts.sme_eval.schema import EvalQuery, EvalResult, LatencyBreakdown


def _result(qid, intent, response, sources=None):
    return EvalResult(
        query=EvalQuery(
            query_id=qid,
            query_text="q",
            intent=intent,
            profile_domain="finance",
            subscription_id="s",
            profile_id="p",
        ),
        response_text=response,
        sources=sources or [{"doc_id": "d1", "chunk_id": "c1"}],
        metadata={},
        latency=LatencyBreakdown(total_ms=100.0),
        run_id="run_test",
        captured_at=datetime(2026, 4, 20, 10, 0, 0),
        api_status=200,
    )


def test_extract_recommendation_sentences_picks_imperatives():
    text = (
        "Revenue is up 12%. Consider consolidating your SaaS vendors. "
        "We should reduce redundant expenses."
    )
    sents = extract_recommendation_sentences(text)
    assert any("consolidat" in s.lower() for s in sents)
    assert any("reduce" in s.lower() for s in sents)


def test_extract_recommendation_sentences_ignores_descriptive():
    text = "Revenue is up 12%. Expenses rose 5%. The trend is stable."
    sents = extract_recommendation_sentences(text)
    # No imperatives — should return empty
    assert sents == []


def test_metric_skips_non_recommend_intent():
    """Only recommend-intent results count toward this metric."""
    metric = RecommendationGroundedness()
    results = [_result("a", "lookup", "Some lookup answer.")]
    batch = metric.compute(results)
    # With no recommend-intent queries, value defaults to 1.0 (nothing to fail)
    assert batch.value == 1.0
    assert batch.details["num_recommend_queries"] == 0


def test_metric_grounded_when_recommendations_cite_sources():
    metric = RecommendationGroundedness()
    results = [
        _result(
            "rec1",
            "recommend",
            "Consolidate vendors (see invoice_2026_03) to reduce cost by 12%.",
            sources=[{"doc_id": "invoice_2026_03", "chunk_id": "c1"}],
        )
    ]
    batch = metric.compute(results)
    assert batch.value == 1.0
    assert batch.details["num_grounded"] == 1


def test_metric_ungrounded_when_recommendation_has_no_evidence():
    metric = RecommendationGroundedness()
    results = [
        _result(
            "rec1",
            "recommend",
            "You should consolidate your vendors.",  # no citation, no sources
            sources=[],
        )
    ]
    batch = metric.compute(results)
    assert batch.value == 0.0
    assert batch.details["num_ungrounded"] == 1


def test_metric_mixed_batch():
    metric = RecommendationGroundedness()
    results = [
        _result("rec_good", "recommend",
                "Reduce cloud spend by 10% (from aws_bill_q3).",
                sources=[{"doc_id": "aws_bill_q3", "chunk_id": "c1"}]),
        _result("rec_bad", "recommend",
                "You could try switching providers.",
                sources=[]),
        _result("ana", "analyze",
                "Revenue is stable."),
    ]
    batch = metric.compute(results)
    assert batch.details["num_recommend_queries"] == 2
    assert batch.value == 0.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/scripts/sme_eval/metrics/test_recommendation_groundedness.py -v`
Expected: FAIL — module does not exist

- [ ] **Step 3: Write the metric**

Create `scripts/sme_eval/metrics/recommendation_groundedness.py`:

```python
"""recommendation_groundedness metric.

Measures: fraction of `recommend`-intent responses where every extracted
recommendation sentence is grounded in evidence (either cited inline,
supported by sources in the response payload, or exposed as explicit ad-hoc
reasoning).

At Phase 0 (no Recommendation Bank yet) this is a baseline measurement — it
will be re-run post-Phase 4 to measure the uplift.
"""
from __future__ import annotations

import re
from collections.abc import Iterable

from scripts.sme_eval.metrics._base import Metric
from scripts.sme_eval.schema import EvalResult, MetricResult

# Imperative verbs that commonly open a recommendation sentence
_RECOMMENDATION_VERBS: tuple[str, ...] = (
    "consolidate", "reduce", "increase", "eliminate", "switch",
    "adopt", "implement", "replace", "review", "renegotiate",
    "investigate", "consider", "explore", "audit", "prioritize",
    "streamline", "automate", "outsource", "hire", "defer",
)

_SHOULD_PATTERN = re.compile(
    r"\b(should|recommend(ed)?|suggest(ed)?|propose(d)?|advise[d]?|could|might)\b",
    re.IGNORECASE,
)

_INLINE_CITATION = re.compile(r"\[[^\]]+?\]|\(see [^)]+\)|\bfrom\s+[A-Za-z0-9_\-§.]+", re.IGNORECASE)

_ADHOC_REASONING_MARKERS: tuple[str, ...] = (
    "because", "since", "given that", "the data shows", "this implies",
)


def extract_recommendation_sentences(text: str) -> list[str]:
    """Split text into sentences and return only those that look like recommendations."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    found: list[str] = []
    for s in sentences:
        low = s.lower().strip()
        if not low:
            continue
        opens_imperative = any(low.startswith(v) for v in _RECOMMENDATION_VERBS)
        has_recommend_verb = (
            _SHOULD_PATTERN.search(low) is not None
            or any(v in low for v in _RECOMMENDATION_VERBS)
        )
        if opens_imperative or has_recommend_verb:
            found.append(s.strip())
    return found


def _is_grounded(sentence: str, result: EvalResult) -> bool:
    """A sentence counts as grounded if any of:
    - It contains an inline citation pattern, OR
    - The response has at least one source AND the sentence references a doc_id,
    - The sentence exposes ad-hoc reasoning (because/since/given that/...)
    """
    if _INLINE_CITATION.search(sentence):
        return True
    if result.sources:
        for src in result.sources:
            doc_id = src.get("doc_id", "")
            if doc_id and doc_id.lower() in sentence.lower():
                return True
    low = sentence.lower()
    if any(m in low for m in _ADHOC_REASONING_MARKERS):
        return True
    return False


class RecommendationGroundedness(Metric):
    name = "recommendation_groundedness"

    def compute(self, results: Iterable[EvalResult]) -> MetricResult:
        rec_results = [r for r in results if r.query.intent == "recommend"]

        if not rec_results:
            return MetricResult(
                metric_name=self.name,
                value=1.0,
                details={
                    "num_recommend_queries": 0,
                    "num_grounded": 0,
                    "num_ungrounded": 0,
                },
            )

        grounded = 0
        ungrounded = 0
        per_query: list[dict] = []
        for r in rec_results:
            recs = extract_recommendation_sentences(r.response_text)
            if not recs:
                # Response is a recommend-intent query with no recommendations.
                # Counts as ungrounded — the model produced no actionable output.
                ungrounded += 1
                per_query.append(
                    {"query_id": r.query.query_id, "recommendations_found": 0}
                )
                continue
            all_grounded = all(_is_grounded(s, r) for s in recs)
            if all_grounded:
                grounded += 1
            else:
                ungrounded += 1
            per_query.append(
                {
                    "query_id": r.query.query_id,
                    "recommendations_found": len(recs),
                    "all_grounded": all_grounded,
                }
            )

        total = len(rec_results)
        return MetricResult(
            metric_name=self.name,
            value=grounded / total,
            details={
                "num_recommend_queries": total,
                "num_grounded": grounded,
                "num_ungrounded": ungrounded,
                "per_query": per_query,
            },
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/scripts/sme_eval/metrics/test_recommendation_groundedness.py -v`
Expected: PASS for all 6 tests

- [ ] **Step 5: Commit**

```bash
git add scripts/sme_eval/metrics/recommendation_groundedness.py \
    tests/scripts/sme_eval/metrics/test_recommendation_groundedness.py
git commit -m "phase0(sme-eval): recommendation_groundedness metric"
```

---

## Task 8: cross_doc_integration_rate metric

Measures fraction of analytical-intent responses that cite evidence from ≥2 distinct documents in the profile.

**Files:**
- Create: `scripts/sme_eval/metrics/cross_doc_integration_rate.py`
- Create: `tests/scripts/sme_eval/metrics/test_cross_doc_integration_rate.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/scripts/sme_eval/metrics/test_cross_doc_integration_rate.py`:

```python
from datetime import datetime

from scripts.sme_eval.metrics.cross_doc_integration_rate import CrossDocIntegrationRate
from scripts.sme_eval.schema import EvalQuery, EvalResult, LatencyBreakdown

_ANALYTICAL_INTENTS = ("analyze", "diagnose", "recommend", "investigate", "compare")


def _result(qid, intent, sources):
    return EvalResult(
        query=EvalQuery(
            query_id=qid,
            query_text="q",
            intent=intent,
            profile_domain="finance",
            subscription_id="s",
            profile_id="p",
        ),
        response_text="r",
        sources=sources,
        metadata={},
        latency=LatencyBreakdown(total_ms=100.0),
        run_id="run_test",
        captured_at=datetime(2026, 4, 20, 10, 0, 0),
        api_status=200,
    )


def test_no_analytical_queries_returns_one():
    metric = CrossDocIntegrationRate()
    results = [_result("a", "lookup", [{"doc_id": "d1"}])]
    batch = metric.compute(results)
    assert batch.value == 1.0


def test_integrates_across_docs():
    metric = CrossDocIntegrationRate()
    results = [
        _result("a", "analyze", [{"doc_id": "d1"}, {"doc_id": "d2"}]),
    ]
    batch = metric.compute(results)
    assert batch.value == 1.0
    assert batch.details["num_integrated"] == 1


def test_single_doc_not_integrated():
    metric = CrossDocIntegrationRate()
    results = [_result("a", "analyze", [{"doc_id": "d1"}, {"doc_id": "d1"}])]
    batch = metric.compute(results)
    assert batch.value == 0.0
    assert batch.details["num_integrated"] == 0


def test_mixed_batch():
    metric = CrossDocIntegrationRate()
    results = [
        _result("a", "analyze", [{"doc_id": "d1"}, {"doc_id": "d2"}]),
        _result("b", "diagnose", [{"doc_id": "d3"}]),
        _result("c", "recommend", [{"doc_id": "d4"}, {"doc_id": "d5"}, {"doc_id": "d6"}]),
        _result("d", "lookup", [{"doc_id": "d7"}]),
    ]
    batch = metric.compute(results)
    assert batch.details["num_analytical"] == 3
    assert batch.details["num_integrated"] == 2
    assert abs(batch.value - 2 / 3) < 1e-9
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/scripts/sme_eval/metrics/test_cross_doc_integration_rate.py -v`
Expected: FAIL — module does not exist

- [ ] **Step 3: Write the metric**

Create `scripts/sme_eval/metrics/cross_doc_integration_rate.py`:

```python
"""cross_doc_integration_rate metric.

Measures: fraction of analytical-intent responses that cite evidence from
≥2 distinct documents in the profile. Analytical intents are compare,
analyze, diagnose, recommend, investigate.
"""
from __future__ import annotations

from collections.abc import Iterable

from scripts.sme_eval.metrics._base import Metric
from scripts.sme_eval.schema import EvalResult, MetricResult

ANALYTICAL_INTENTS: frozenset[str] = frozenset(
    {"analyze", "diagnose", "recommend", "investigate", "compare"}
)


def _distinct_docs(result: EvalResult) -> int:
    doc_ids = {s.get("doc_id") for s in result.sources if s.get("doc_id")}
    return len(doc_ids)


class CrossDocIntegrationRate(Metric):
    name = "cross_doc_integration_rate"

    def compute(self, results: Iterable[EvalResult]) -> MetricResult:
        analytical = [r for r in results if r.query.intent in ANALYTICAL_INTENTS]

        if not analytical:
            return MetricResult(
                metric_name=self.name,
                value=1.0,
                details={"num_analytical": 0, "num_integrated": 0},
            )

        integrated = sum(1 for r in analytical if _distinct_docs(r) >= 2)
        total = len(analytical)
        return MetricResult(
            metric_name=self.name,
            value=integrated / total,
            details={
                "num_analytical": total,
                "num_integrated": integrated,
                "per_query": [
                    {
                        "query_id": r.query.query_id,
                        "distinct_docs": _distinct_docs(r),
                    }
                    for r in analytical
                ],
            },
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/scripts/sme_eval/metrics/test_cross_doc_integration_rate.py -v`
Expected: PASS for all 4 tests

- [ ] **Step 5: Commit**

```bash
git add scripts/sme_eval/metrics/cross_doc_integration_rate.py \
    tests/scripts/sme_eval/metrics/test_cross_doc_integration_rate.py
git commit -m "phase0(sme-eval): cross_doc_integration_rate metric"
```

---

## Task 9: insight_novelty metric

Measures fraction of analytical-intent responses whose claims go beyond single-document facts. At Phase 0, the pre-computed per-doc summaries referenced in the spec don't exist yet, so the Phase 0 implementation approximates via a lexical proxy: response n-grams that don't appear verbatim in any single source's excerpt counted as "novel." The metric is re-calibrated in Phase 2 when real per-doc summaries are available.

**Files:**
- Create: `scripts/sme_eval/metrics/insight_novelty.py`
- Create: `tests/scripts/sme_eval/metrics/test_insight_novelty.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/scripts/sme_eval/metrics/test_insight_novelty.py`:

```python
from datetime import datetime

from scripts.sme_eval.metrics.insight_novelty import InsightNovelty
from scripts.sme_eval.schema import EvalQuery, EvalResult, LatencyBreakdown


def _result(qid, intent, response, source_excerpts):
    return EvalResult(
        query=EvalQuery(
            query_id=qid,
            query_text="q",
            intent=intent,
            profile_domain="finance",
            subscription_id="s",
            profile_id="p",
        ),
        response_text=response,
        sources=[
            {"doc_id": f"d{i}", "chunk_id": f"c{i}", "excerpt": ex}
            for i, ex in enumerate(source_excerpts)
        ],
        metadata={},
        latency=LatencyBreakdown(total_ms=100.0),
        run_id="run_test",
        captured_at=datetime(2026, 4, 20, 10, 0, 0),
        api_status=200,
    )


def test_skips_non_analytical():
    metric = InsightNovelty()
    results = [_result("a", "lookup", "Answer is 42.", ["Source says 42."])]
    batch = metric.compute(results)
    # Non-analytical queries don't contribute; value is 0.0 (no novelty to measure)
    assert batch.details["num_analytical"] == 0


def test_response_fully_lifted_has_zero_novelty():
    metric = InsightNovelty()
    results = [
        _result(
            "a",
            "analyze",
            "Revenue rose 12 percent in Q3.",
            ["Revenue rose 12 percent in Q3."],
        )
    ]
    batch = metric.compute(results)
    assert batch.value < 0.1  # almost entirely lifted


def test_response_with_new_ngrams_has_some_novelty():
    metric = InsightNovelty()
    results = [
        _result(
            "a",
            "analyze",
            "Revenue rose 12 percent. This indicates accelerating growth from prior quarters.",
            ["Revenue rose 12 percent."],
        )
    ]
    batch = metric.compute(results)
    assert batch.value > 0.2


def test_multiple_results_aggregated():
    metric = InsightNovelty()
    results = [
        _result("a", "analyze", "Revenue rose 12%. Growth is accelerating.", ["Revenue rose 12%."]),
        _result("b", "diagnose", "The error is caused by auth token expiry.", ["Log shows token expired."]),
    ]
    batch = metric.compute(results)
    assert batch.details["num_analytical"] == 2
    assert 0.0 <= batch.value <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/scripts/sme_eval/metrics/test_insight_novelty.py -v`
Expected: FAIL — module does not exist

- [ ] **Step 3: Write the metric**

Create `scripts/sme_eval/metrics/insight_novelty.py`:

```python
"""insight_novelty metric (Phase 0 lexical proxy).

Measures: fraction of analytical-intent response n-grams (trigrams) that do
not appear verbatim in any single source excerpt attached to the response.

Phase 0 implementation uses source `excerpt` fields when present; falls back
to empty excerpt (treating the response as maximally novel) when excerpts
aren't available. Phase 2 will re-target this against pre-computed per-doc
summaries for a more semantic measure.
"""
from __future__ import annotations

import re
from collections.abc import Iterable

from scripts.sme_eval.metrics._base import Metric
from scripts.sme_eval.metrics.cross_doc_integration_rate import ANALYTICAL_INTENTS
from scripts.sme_eval.schema import EvalResult, MetricResult

_WORD = re.compile(r"\w+")


def _trigrams(text: str) -> set[tuple[str, str, str]]:
    toks = [t.lower() for t in _WORD.findall(text)]
    if len(toks) < 3:
        return set()
    return {(toks[i], toks[i + 1], toks[i + 2]) for i in range(len(toks) - 2)}


def _novelty_ratio(response: str, source_text: str) -> float:
    resp = _trigrams(response)
    if not resp:
        return 0.0
    src = _trigrams(source_text)
    novel = resp - src
    return len(novel) / len(resp)


class InsightNovelty(Metric):
    name = "insight_novelty"

    def compute(self, results: Iterable[EvalResult]) -> MetricResult:
        analytical = [r for r in results if r.query.intent in ANALYTICAL_INTENTS]

        if not analytical:
            return MetricResult(
                metric_name=self.name,
                value=0.0,
                details={"num_analytical": 0},
            )

        per_query: list[dict] = []
        total_ratio = 0.0
        for r in analytical:
            src_text = " ".join(s.get("excerpt", "") for s in r.sources)
            ratio = _novelty_ratio(r.response_text, src_text)
            per_query.append(
                {"query_id": r.query.query_id, "novelty_ratio": round(ratio, 3)}
            )
            total_ratio += ratio

        return MetricResult(
            metric_name=self.name,
            value=total_ratio / len(analytical),
            details={"num_analytical": len(analytical), "per_query": per_query},
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/scripts/sme_eval/metrics/test_insight_novelty.py -v`
Expected: PASS for all 4 tests

- [ ] **Step 5: Commit**

```bash
git add scripts/sme_eval/metrics/insight_novelty.py \
    tests/scripts/sme_eval/metrics/test_insight_novelty.py
git commit -m "phase0(sme-eval): insight_novelty metric (lexical proxy)"
```

---

## Task 10: sme_persona_consistency metric (LLM-judge)

Measures (0–5 scale) whether the response voice matches the expected adapter persona for its `profile_domain`. At Phase 0 no adapters exist, so we measure against a hand-written reference persona per domain. The metric uses an LLM judge via the existing DocWain LLM gateway. Because this adds cost/latency, it's opt-in behind a flag.

**Files:**
- Create: `scripts/sme_eval/metrics/sme_persona_consistency.py`
- Create: `tests/scripts/sme_eval/metrics/test_sme_persona_consistency.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/scripts/sme_eval/metrics/test_sme_persona_consistency.py`:

```python
from datetime import datetime
from unittest.mock import MagicMock

from scripts.sme_eval.metrics.sme_persona_consistency import (
    SmePersonaConsistency,
    _REFERENCE_PERSONAS,
)
from scripts.sme_eval.schema import EvalQuery, EvalResult, LatencyBreakdown


def _result(qid, domain, response):
    return EvalResult(
        query=EvalQuery(
            query_id=qid,
            query_text="q",
            intent="analyze",
            profile_domain=domain,
            subscription_id="s",
            profile_id="p",
        ),
        response_text=response,
        sources=[],
        metadata={},
        latency=LatencyBreakdown(total_ms=100.0),
        run_id="run_test",
        captured_at=datetime(2026, 4, 20, 10, 0, 0),
        api_status=200,
    )


def test_reference_personas_cover_all_domains():
    for dom in ("finance", "legal", "hr", "medical", "it_support", "generic"):
        assert dom in _REFERENCE_PERSONAS


def test_metric_aggregates_judge_scores():
    judge = MagicMock(return_value=4.2)
    metric = SmePersonaConsistency(judge_fn=judge)
    results = [
        _result("a", "finance", "Q3 revenue rose 12%, driven by..."),
        _result("b", "legal", "The contract obliges..."),
    ]
    batch = metric.compute(results)
    assert batch.value == 4.2
    assert batch.details["num_judged"] == 2
    assert judge.call_count == 2


def test_metric_handles_judge_failure():
    calls = [5.0, Exception("gateway down")]

    def judge(*_args, **_kwargs):
        val = calls.pop(0)
        if isinstance(val, Exception):
            raise val
        return val

    metric = SmePersonaConsistency(judge_fn=judge)
    results = [
        _result("a", "finance", "Answer A"),
        _result("b", "finance", "Answer B"),
    ]
    batch = metric.compute(results)
    # One failed judgment — excluded from numerator and denominator
    assert batch.details["num_judged"] == 1
    assert batch.details["num_failed"] == 1
    assert batch.value == 5.0


def test_metric_empty_batch():
    judge = MagicMock()
    metric = SmePersonaConsistency(judge_fn=judge)
    batch = metric.compute([])
    assert batch.value == 0.0
    assert batch.details["num_judged"] == 0
    judge.assert_not_called()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/scripts/sme_eval/metrics/test_sme_persona_consistency.py -v`
Expected: FAIL — module does not exist

- [ ] **Step 3: Write the metric**

Create `scripts/sme_eval/metrics/sme_persona_consistency.py`:

```python
"""sme_persona_consistency metric.

Uses an LLM judge to rate (0–5) how well the response's voice matches a
reference persona for the profile's domain. Phase 0 uses hand-written
reference personas; Phase 4 will swap them for the actual adapter personas.

The judge_fn dependency is injected so tests don't call a real LLM.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable

from scripts.sme_eval.metrics._base import Metric
from scripts.sme_eval.schema import EvalResult, MetricResult

# Phase 0 reference personas — will be replaced by adapter personas in Phase 4.
_REFERENCE_PERSONAS: dict[str, str] = {
    "finance": (
        "A senior financial analyst: direct, quantitative, hedged with uncertainty "
        "bounds. Cites absolute values behind every percentage. Distinguishes "
        "explicit facts from inferences."
    ),
    "legal": (
        "A senior legal counsel: precise, explicit about obligations and dates, "
        "careful to distinguish authoritative text from commentary. Flags ambiguity "
        "rather than smoothing over it."
    ),
    "hr": (
        "A seasoned HR business partner: professional, policy-anchored, respectful "
        "of privacy. Balances individual facts with organizational context."
    ),
    "medical": (
        "A careful clinical informationist: strictly evidence-grounded, explicit "
        "about differential possibilities, never prescribes. Caveats confidently."
    ),
    "it_support": (
        "A senior support engineer: structured symptom→cause→fix flow, precise "
        "about systems and conditions, explicit about assumptions."
    ),
    "generic": (
        "A domain-agnostic subject-matter expert: clear, evidence-grounded, "
        "explicitly hedged; distinguishes explicit facts from inferences."
    ),
}

_JUDGE_PROMPT_TEMPLATE = """You are evaluating whether a DocWain response matches a target persona.

Target persona for domain '{domain}':
{persona}

Response under evaluation:
{response}

Rate the persona match on a 0-5 scale:
- 0: Voice is entirely wrong for the persona (e.g., casual chatter where formal expertise is expected)
- 1: Major mismatches in tone, hedging, or specificity
- 2: Partial match with clear gaps
- 3: Acceptable match with minor issues
- 4: Good match; minor polish needed
- 5: Excellent match — reads as though written by the persona

Return ONLY a single number (0, 1, 2, 3, 4, or 5). No explanation.
"""


def default_judge_fn(prompt: str) -> float:
    """Default implementation — must be wired to DocWain LLM gateway.

    Left as a stub so the metric module imports cleanly even without a
    gateway available. The baseline CLI injects the real judge_fn when
    running; tests inject a mock.
    """
    raise NotImplementedError(
        "default_judge_fn is a stub; inject a real judge_fn from the baseline CLI"
    )


class SmePersonaConsistency(Metric):
    name = "sme_persona_consistency"

    def __init__(self, judge_fn: Callable[[str], float] = default_judge_fn):
        self._judge_fn = judge_fn

    def compute(self, results: Iterable[EvalResult]) -> MetricResult:
        results = list(results)
        if not results:
            return MetricResult(
                metric_name=self.name,
                value=0.0,
                details={"num_judged": 0, "num_failed": 0},
            )

        scores: list[float] = []
        failures: list[dict] = []
        for r in results:
            persona = _REFERENCE_PERSONAS.get(
                r.query.profile_domain, _REFERENCE_PERSONAS["generic"]
            )
            prompt = _JUDGE_PROMPT_TEMPLATE.format(
                domain=r.query.profile_domain,
                persona=persona,
                response=r.response_text[:2000],  # cap for judge input
            )
            try:
                score = float(self._judge_fn(prompt))
                scores.append(max(0.0, min(5.0, score)))
            except Exception as e:
                failures.append({"query_id": r.query.query_id, "error": str(e)})

        avg = sum(scores) / len(scores) if scores else 0.0
        return MetricResult(
            metric_name=self.name,
            value=avg,
            details={
                "num_judged": len(scores),
                "num_failed": len(failures),
                "failures": failures,
                "scale_max": 5.0,
            },
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/scripts/sme_eval/metrics/test_sme_persona_consistency.py -v`
Expected: PASS for all 4 tests

- [ ] **Step 5: Commit**

```bash
git add scripts/sme_eval/metrics/sme_persona_consistency.py \
    tests/scripts/sme_eval/metrics/test_sme_persona_consistency.py
git commit -m "phase0(sme-eval): sme_persona_consistency metric (LLM-judge)"
```

---

## Task 11: verified_removal_rate metric

Measures fraction of responses where no content was dropped by citation verification (a 0.85 floor: safety drops should happen *sometimes*, not never and not often). Since Phase 0 doesn't have the post-A citation verifier enhancements, this reads `metadata.citation_verifier_dropped` from the API response if present; defaults to "no drops happened" otherwise.

**Files:**
- Create: `scripts/sme_eval/metrics/verified_removal_rate.py`
- Create: `tests/scripts/sme_eval/metrics/test_verified_removal_rate.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/scripts/sme_eval/metrics/test_verified_removal_rate.py`:

```python
from datetime import datetime

from scripts.sme_eval.metrics.verified_removal_rate import VerifiedRemovalRate
from scripts.sme_eval.schema import EvalQuery, EvalResult, LatencyBreakdown


def _result(qid, metadata):
    return EvalResult(
        query=EvalQuery(
            query_id=qid,
            query_text="q",
            intent="analyze",
            profile_domain="finance",
            subscription_id="s",
            profile_id="p",
        ),
        response_text="r",
        sources=[],
        metadata=metadata,
        latency=LatencyBreakdown(total_ms=100.0),
        run_id="run_test",
        captured_at=datetime(2026, 4, 20, 10, 0, 0),
        api_status=200,
    )


def test_no_drops_flagged_returns_one():
    metric = VerifiedRemovalRate()
    results = [_result("a", {}), _result("b", {"citation_verifier_dropped": 0})]
    batch = metric.compute(results)
    assert batch.value == 1.0


def test_some_drops_reduce_value():
    metric = VerifiedRemovalRate()
    results = [
        _result("a", {"citation_verifier_dropped": 0}),
        _result("b", {"citation_verifier_dropped": 2}),
    ]
    batch = metric.compute(results)
    assert batch.value == 0.5
    assert batch.details["num_with_drops"] == 1


def test_empty_batch():
    metric = VerifiedRemovalRate()
    batch = metric.compute([])
    assert batch.value == 1.0
    assert batch.details["num_results"] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/scripts/sme_eval/metrics/test_verified_removal_rate.py -v`
Expected: FAIL — module does not exist

- [ ] **Step 3: Write the metric**

Create `scripts/sme_eval/metrics/verified_removal_rate.py`:

```python
"""verified_removal_rate metric.

Reads `metadata.citation_verifier_dropped` from each response's metadata.
Value is fraction of responses with zero drops. The gate threshold is ≥0.85 —
drops should happen sometimes (proving the verifier works) but not often.

At Phase 0, the production API may not emit this field. Treat missing-or-zero
as "no drops"; that's honest for the baseline.
"""
from __future__ import annotations

from collections.abc import Iterable

from scripts.sme_eval.metrics._base import Metric
from scripts.sme_eval.schema import EvalResult, MetricResult


def _had_drops(result: EvalResult) -> bool:
    return int(result.metadata.get("citation_verifier_dropped", 0)) > 0


class VerifiedRemovalRate(Metric):
    name = "verified_removal_rate"

    def compute(self, results: Iterable[EvalResult]) -> MetricResult:
        results = list(results)
        if not results:
            return MetricResult(
                metric_name=self.name,
                value=1.0,
                details={"num_results": 0, "num_with_drops": 0},
            )

        with_drops = sum(1 for r in results if _had_drops(r))
        total = len(results)
        return MetricResult(
            metric_name=self.name,
            value=(total - with_drops) / total,
            details={"num_results": total, "num_with_drops": with_drops},
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/scripts/sme_eval/metrics/test_verified_removal_rate.py -v`
Expected: PASS for all 3 tests

- [ ] **Step 5: Commit**

```bash
git add scripts/sme_eval/metrics/verified_removal_rate.py \
    tests/scripts/sme_eval/metrics/test_verified_removal_rate.py
git commit -m "phase0(sme-eval): verified_removal_rate metric"
```

---

## Task 12: sme_artifact_hit_rate metric

Measures fraction of analytical queries whose response included at least one SME artifact in its evidence. Phase 0 baseline will be **0.0** because no SME artifacts exist yet — that IS the before-picture and is informative. Reads `metadata.retrieval_layers.sme_artifacts` (or similar) when available; defaults to "no SME evidence" otherwise.

**Files:**
- Create: `scripts/sme_eval/metrics/sme_artifact_hit_rate.py`
- Create: `tests/scripts/sme_eval/metrics/test_sme_artifact_hit_rate.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/scripts/sme_eval/metrics/test_sme_artifact_hit_rate.py`:

```python
from datetime import datetime

from scripts.sme_eval.metrics.sme_artifact_hit_rate import SmeArtifactHitRate
from scripts.sme_eval.schema import EvalQuery, EvalResult, LatencyBreakdown


def _result(qid, intent, metadata=None):
    return EvalResult(
        query=EvalQuery(
            query_id=qid,
            query_text="q",
            intent=intent,
            profile_domain="finance",
            subscription_id="s",
            profile_id="p",
        ),
        response_text="r",
        sources=[],
        metadata=metadata or {},
        latency=LatencyBreakdown(total_ms=100.0),
        run_id="run_test",
        captured_at=datetime(2026, 4, 20, 10, 0, 0),
        api_status=200,
    )


def test_phase0_baseline_zero():
    """At Phase 0 no SME artifacts exist — value should be 0.0."""
    metric = SmeArtifactHitRate()
    results = [_result("a", "analyze"), _result("b", "diagnose")]
    batch = metric.compute(results)
    assert batch.value == 0.0


def test_with_sme_artifacts_counted():
    metric = SmeArtifactHitRate()
    results = [
        _result("a", "analyze", metadata={"retrieval_layers": {"sme_artifacts_count": 3}}),
        _result("b", "analyze", metadata={"retrieval_layers": {"sme_artifacts_count": 0}}),
    ]
    batch = metric.compute(results)
    assert batch.value == 0.5


def test_skips_non_analytical():
    metric = SmeArtifactHitRate()
    results = [_result("a", "lookup", metadata={"retrieval_layers": {"sme_artifacts_count": 3}})]
    batch = metric.compute(results)
    # No analytical queries — value is 0.0 with num_analytical=0 in details
    assert batch.details["num_analytical"] == 0


def test_empty_batch():
    metric = SmeArtifactHitRate()
    batch = metric.compute([])
    assert batch.value == 0.0
    assert batch.details["num_analytical"] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/scripts/sme_eval/metrics/test_sme_artifact_hit_rate.py -v`
Expected: FAIL — module does not exist

- [ ] **Step 3: Write the metric**

Create `scripts/sme_eval/metrics/sme_artifact_hit_rate.py`:

```python
"""sme_artifact_hit_rate metric.

Fraction of analytical-intent queries whose response included at least one
SME artifact in its evidence. Phase 0 baseline is 0.0 (no SME artifacts
exist yet); the metric is provisioned so Phase 3+ can measure uplift against
an already-calibrated before-picture.
"""
from __future__ import annotations

from collections.abc import Iterable

from scripts.sme_eval.metrics._base import Metric
from scripts.sme_eval.metrics.cross_doc_integration_rate import ANALYTICAL_INTENTS
from scripts.sme_eval.schema import EvalResult, MetricResult


def _has_sme_artifact(result: EvalResult) -> bool:
    layers = result.metadata.get("retrieval_layers", {}) or {}
    return int(layers.get("sme_artifacts_count", 0)) > 0


class SmeArtifactHitRate(Metric):
    name = "sme_artifact_hit_rate"

    def compute(self, results: Iterable[EvalResult]) -> MetricResult:
        analytical = [r for r in results if r.query.intent in ANALYTICAL_INTENTS]
        if not analytical:
            return MetricResult(
                metric_name=self.name,
                value=0.0,
                details={"num_analytical": 0, "num_hit": 0},
            )
        hits = sum(1 for r in analytical if _has_sme_artifact(r))
        total = len(analytical)
        return MetricResult(
            metric_name=self.name,
            value=hits / total,
            details={"num_analytical": total, "num_hit": hits},
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/scripts/sme_eval/metrics/test_sme_artifact_hit_rate.py -v`
Expected: PASS for all 4 tests

- [ ] **Step 5: Commit**

```bash
git add scripts/sme_eval/metrics/sme_artifact_hit_rate.py \
    tests/scripts/sme_eval/metrics/test_sme_artifact_hit_rate.py
git commit -m "phase0(sme-eval): sme_artifact_hit_rate metric"
```

---

## Task 13: Latency aggregation (p50/p95/p99 per intent)

**Files:**
- Create: `scripts/sme_eval/aggregate.py`
- Create: `tests/scripts/sme_eval/test_aggregate.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/scripts/sme_eval/test_aggregate.py`:

```python
from datetime import datetime

from scripts.sme_eval.aggregate import aggregate_latency_per_intent
from scripts.sme_eval.schema import EvalQuery, EvalResult, LatencyBreakdown


def _result(qid, intent, total_ms):
    return EvalResult(
        query=EvalQuery(
            query_id=qid,
            query_text="q",
            intent=intent,
            profile_domain="finance",
            subscription_id="s",
            profile_id="p",
        ),
        response_text="r",
        sources=[],
        metadata={},
        latency=LatencyBreakdown(total_ms=total_ms),
        run_id="run_test",
        captured_at=datetime(2026, 4, 20, 10, 0, 0),
        api_status=200,
    )


def test_aggregates_per_intent():
    results = [
        _result("l1", "lookup", 1000),
        _result("l2", "lookup", 2000),
        _result("l3", "lookup", 3000),
        _result("a1", "analyze", 5000),
        _result("a2", "analyze", 7000),
    ]
    agg = aggregate_latency_per_intent(results)
    assert "lookup" in agg
    assert "analyze" in agg
    assert agg["lookup"]["p50"] == 2000
    assert agg["analyze"]["p50"] == 6000


def test_empty_returns_empty_dict():
    assert aggregate_latency_per_intent([]) == {}


def test_skips_failed_calls():
    """API status != 200 excluded from latency stats."""
    r1 = _result("a", "lookup", 1000)
    r2 = _result("b", "lookup", 99999)
    r2 = r2.model_copy(update={"api_status": 500})
    agg = aggregate_latency_per_intent([r1, r2])
    assert agg["lookup"]["count"] == 1
    assert agg["lookup"]["p50"] == 1000
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/scripts/sme_eval/test_aggregate.py -v`
Expected: FAIL — module does not exist

- [ ] **Step 3: Write the aggregator**

Create `scripts/sme_eval/aggregate.py`:

```python
"""Latency aggregation per intent.

Computes p50/p95/p99 of total_ms per intent, excluding failed calls.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import numpy as np

from scripts.sme_eval.schema import EvalResult


def aggregate_latency_per_intent(results: Iterable[EvalResult]) -> dict[str, dict[str, float]]:
    """Return { intent: {count, p50, p95, p99, mean, max} }."""
    grouped: dict[str, list[float]] = defaultdict(list)
    for r in results:
        if r.api_status != 200:
            continue
        grouped[r.query.intent].append(r.latency.total_ms)

    out: dict[str, dict[str, float]] = {}
    for intent, samples in grouped.items():
        arr = np.asarray(samples, dtype=float)
        out[intent] = {
            "count": int(arr.size),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "mean": float(arr.mean()),
            "max": float(arr.max()),
        }
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/scripts/sme_eval/test_aggregate.py -v`
Expected: PASS for all 3 tests

- [ ] **Step 5: Commit**

```bash
git add scripts/sme_eval/aggregate.py tests/scripts/sme_eval/test_aggregate.py
git commit -m "phase0(sme-eval): per-intent latency aggregation"
```

---

## Task 14: Human-rating CSV export/import

Produces a CSV for domain experts to rate responses on a 1–5 SME scale and imports their rated CSV back into the baseline snapshot.

**Files:**
- Create: `scripts/sme_eval/human_rating.py`
- Create: `tests/scripts/sme_eval/test_human_rating.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/scripts/sme_eval/test_human_rating.py`:

```python
from datetime import datetime
from pathlib import Path

from scripts.sme_eval.human_rating import export_for_rating, import_ratings
from scripts.sme_eval.schema import EvalQuery, EvalResult, LatencyBreakdown


def _result(qid, domain="finance"):
    return EvalResult(
        query=EvalQuery(
            query_id=qid,
            query_text=f"query_{qid}",
            intent="analyze",
            profile_domain=domain,
            subscription_id="s",
            profile_id="p",
        ),
        response_text=f"response_{qid}",
        sources=[],
        metadata={},
        latency=LatencyBreakdown(total_ms=100.0),
        run_id="run_test",
        captured_at=datetime(2026, 4, 20, 10, 0, 0),
        api_status=200,
    )


def test_export_creates_csv(tmp_path: Path):
    out = tmp_path / "rate_me.csv"
    export_for_rating([_result("a"), _result("b")], out)
    assert out.exists()
    content = out.read_text()
    assert "query_id" in content
    assert "sme_score_1_to_5" in content
    assert "query_a" in content


def test_import_ratings_parses_csv(tmp_path: Path):
    csv_path = tmp_path / "rated.csv"
    csv_path.write_text(
        "query_id,profile_domain,query_text,response_text,sme_score_1_to_5,rater_notes\n"
        "query_a,finance,q,r,4,good voice\n"
        "query_b,finance,q,r,3,missing citations\n"
    )
    ratings = import_ratings(csv_path)
    assert ratings == {"query_a": 4, "query_b": 3}


def test_import_ratings_ignores_blank_scores(tmp_path: Path):
    csv_path = tmp_path / "rated.csv"
    csv_path.write_text(
        "query_id,profile_domain,query_text,response_text,sme_score_1_to_5,rater_notes\n"
        "query_a,finance,q,r,,no rating\n"
        "query_b,finance,q,r,5,excellent\n"
    )
    ratings = import_ratings(csv_path)
    assert ratings == {"query_b": 5}


def test_import_ratings_validates_score_range(tmp_path: Path):
    csv_path = tmp_path / "rated.csv"
    csv_path.write_text(
        "query_id,profile_domain,query_text,response_text,sme_score_1_to_5,rater_notes\n"
        "query_a,finance,q,r,7,out of range\n"
    )
    ratings = import_ratings(csv_path)
    # Out-of-range ratings dropped, not crashed
    assert ratings == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/scripts/sme_eval/test_human_rating.py -v`
Expected: FAIL — module does not exist

- [ ] **Step 3: Write the tool**

Create `scripts/sme_eval/human_rating.py`:

```python
"""Human rating CSV export/import.

Domain experts rate responses on a 1–5 SME scale in CSV. The import step
validates ratings and returns a {query_id: int} mapping that the baseline
snapshot merges into its human_rated_sme_score_avg.
"""
from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path

from scripts.sme_eval.schema import EvalResult

_COLUMNS = (
    "query_id",
    "profile_domain",
    "query_text",
    "response_text",
    "sme_score_1_to_5",
    "rater_notes",
)


def export_for_rating(results: Iterable[EvalResult], out_path: Path | str) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_COLUMNS)
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "query_id": r.query.query_id,
                    "profile_domain": r.query.profile_domain,
                    "query_text": r.query.query_text,
                    "response_text": r.response_text,
                    "sme_score_1_to_5": "",
                    "rater_notes": "",
                }
            )


def import_ratings(csv_path: Path | str) -> dict[str, int]:
    """Return {query_id: rating}. Silently drops blanks and out-of-range scores."""
    path = Path(csv_path)
    out: dict[str, int] = {}
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = (row.get("sme_score_1_to_5") or "").strip()
            if not raw:
                continue
            try:
                score = int(raw)
            except ValueError:
                continue
            if score < 1 or score > 5:
                continue
            out[row["query_id"]] = score
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/scripts/sme_eval/test_human_rating.py -v`
Expected: PASS for all 4 tests

- [ ] **Step 5: Commit**

```bash
git add scripts/sme_eval/human_rating.py tests/scripts/sme_eval/test_human_rating.py
git commit -m "phase0(sme-eval): human rating CSV export/import"
```

---

## Task 15: Baseline snapshot CLI

Orchestrates everything: loads queries from `tests/sme_evalset_v1/queries/*.yaml`, runs each against production DocWain via `QueryRunner`, persists raw results to JSONL, computes all metrics, aggregates latency, writes `tests/sme_metrics_baseline_{YYYY-MM-DD}.json`.

**Files:**
- Create: `scripts/sme_eval/run_baseline.py`
- Create: `tests/scripts/sme_eval/test_run_baseline.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/scripts/sme_eval/test_run_baseline.py`:

```python
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml

from scripts.sme_eval.run_baseline import (
    load_queries_from_yaml,
    compose_snapshot,
    DEFAULT_METRICS,
)
from scripts.sme_eval.schema import EvalQuery, EvalResult, LatencyBreakdown


def test_load_queries_parses_yaml(tmp_path: Path):
    y = tmp_path / "finance.yaml"
    y.write_text(
        yaml.safe_dump({
            "queries": [
                {
                    "query_id": "finance_001",
                    "query_text": "q1",
                    "intent": "analyze",
                    "profile_domain": "finance",
                    "subscription_id": "s",
                    "profile_id": "p",
                },
                {
                    "query_id": "finance_002",
                    "query_text": "q2",
                    "intent": "lookup",
                    "profile_domain": "finance",
                    "subscription_id": "s",
                    "profile_id": "p",
                },
            ]
        })
    )
    queries = load_queries_from_yaml(y)
    assert len(queries) == 2
    assert queries[0].query_id == "finance_001"


def test_load_queries_validates_each(tmp_path: Path):
    y = tmp_path / "bad.yaml"
    y.write_text(
        yaml.safe_dump({
            "queries": [
                {"query_id": "x", "query_text": "q", "intent": "analyze",
                 "profile_domain": "unknown_domain", "subscription_id": "s", "profile_id": "p"},
            ]
        })
    )
    try:
        load_queries_from_yaml(y)
        raised = False
    except Exception:
        raised = True
    assert raised


def test_compose_snapshot_rolls_up_metrics(tmp_path: Path):
    def _result(qid, intent):
        return EvalResult(
            query=EvalQuery(
                query_id=qid,
                query_text="q",
                intent=intent,
                profile_domain="finance",
                subscription_id="s",
                profile_id="p",
            ),
            response_text="r",
            sources=[{"doc_id": "d1"}],
            metadata={"grounded": True},
            latency=LatencyBreakdown(total_ms=1000.0),
            run_id="run1",
            captured_at=datetime(2026, 4, 20, 10, 0, 0),
            api_status=200,
        )

    results = [_result("a", "analyze"), _result("b", "lookup")]
    judge_fn = MagicMock(return_value=4.0)
    snap = compose_snapshot(
        results,
        run_id="run1",
        git_sha="abcd123",
        api_base_url="http://localhost:8000",
        judge_fn=judge_fn,
    )
    assert snap.num_queries == 2
    assert snap.per_domain_counts["finance"] == 2
    assert "analyze" in snap.latency_p50_per_intent
    assert "sme_persona_consistency" in snap.sme_metrics
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/scripts/sme_eval/test_run_baseline.py -v`
Expected: FAIL — module does not exist

- [ ] **Step 3: Write the CLI**

Create `scripts/sme_eval/run_baseline.py`:

```python
"""Baseline CLI — orchestrates end-to-end evaluation and writes snapshot.

Usage:
    python -m scripts.sme_eval.run_baseline \\
        --eval-dir tests/sme_evalset_v1/queries \\
        --fixtures tests/sme_evalset_v1/fixtures/test_profiles.yaml \\
        --out tests/sme_metrics_baseline_$(date +%Y-%m-%d).json \\
        --api-base-url http://localhost:8000 \\
        --skip-llm-judge           # optional, skips sme_persona_consistency
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Iterable
from datetime import datetime
from pathlib import Path

import yaml

from scripts.sme_eval.aggregate import aggregate_latency_per_intent
from scripts.sme_eval.metrics._base import Metric
from scripts.sme_eval.metrics.cross_doc_integration_rate import CrossDocIntegrationRate
from scripts.sme_eval.metrics.insight_novelty import InsightNovelty
from scripts.sme_eval.metrics.ragas_wrapper import RagasMetrics
from scripts.sme_eval.metrics.recommendation_groundedness import RecommendationGroundedness
from scripts.sme_eval.metrics.sme_artifact_hit_rate import SmeArtifactHitRate
from scripts.sme_eval.metrics.sme_persona_consistency import SmePersonaConsistency
from scripts.sme_eval.metrics.verified_removal_rate import VerifiedRemovalRate
from scripts.sme_eval.query_runner import QueryRunner, RunnerConfig
from scripts.sme_eval.result_store import JsonlResultStore
from scripts.sme_eval.schema import BaselineSnapshot, EvalQuery, EvalResult, MetricResult


DEFAULT_METRICS_NON_LLM: tuple[type[Metric], ...] = (
    RagasMetrics,
    RecommendationGroundedness,
    CrossDocIntegrationRate,
    InsightNovelty,
    VerifiedRemovalRate,
    SmeArtifactHitRate,
)

DEFAULT_METRICS = DEFAULT_METRICS_NON_LLM + (SmePersonaConsistency,)


def load_queries_from_yaml(path: Path) -> list[EvalQuery]:
    """Load and validate queries from one domain YAML file."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    queries_raw = raw.get("queries", [])
    return [EvalQuery(**q) for q in queries_raw]


def load_all_queries(eval_dir: Path) -> list[EvalQuery]:
    all_q: list[EvalQuery] = []
    for yaml_path in sorted(Path(eval_dir).glob("*.yaml")):
        all_q.extend(load_queries_from_yaml(yaml_path))
    return all_q


def _current_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def _docwain_llm_judge(prompt: str) -> float:
    """Judge via DocWain's own LLM gateway, keeping eval self-contained.

    Hits a chat completion endpoint compatible with OpenAI's /v1/chat/completions
    (per src/serving/vllm_manager.py). Extracts the first integer 0..5.
    """
    import re

    import httpx

    url = os.environ.get("DOCWAIN_LLM_URL", "http://localhost:8100/v1/chat/completions")
    model = os.environ.get("DOCWAIN_LLM_MODEL", "docwain-fast")
    resp = httpx.post(
        url,
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 4,
        },
        timeout=30.0,
    )
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"].strip()
    m = re.search(r"[0-5]", text)
    if not m:
        raise ValueError(f"Could not parse judge output: {text!r}")
    return float(m.group(0))


def compose_snapshot(
    results: Iterable[EvalResult],
    *,
    run_id: str,
    git_sha: str,
    api_base_url: str,
    judge_fn: Callable[[str], float] | None = None,
) -> BaselineSnapshot:
    results = list(results)
    counter = Counter(r.query.profile_domain for r in results)

    # Non-LLM metrics — always run
    ragas = RagasMetrics().compute(results)
    rec_ground = RecommendationGroundedness().compute(results)
    xdoc = CrossDocIntegrationRate().compute(results)
    novelty = InsightNovelty().compute(results)
    verif_rem = VerifiedRemovalRate().compute(results)
    art_hit = SmeArtifactHitRate().compute(results)

    sme_metrics: dict[str, MetricResult] = {
        rec_ground.metric_name: rec_ground,
        xdoc.metric_name: xdoc,
        novelty.metric_name: novelty,
        verif_rem.metric_name: verif_rem,
        art_hit.metric_name: art_hit,
    }

    # LLM-judge metric — opt-in
    if judge_fn is not None:
        persona = SmePersonaConsistency(judge_fn=judge_fn).compute(results)
        sme_metrics[persona.metric_name] = persona
    else:
        sme_metrics["sme_persona_consistency"] = MetricResult(
            metric_name="sme_persona_consistency",
            value=0.0,
            details={"skipped": True, "reason": "judge_fn not provided"},
        )

    # Latency per intent
    per_intent = aggregate_latency_per_intent(results)
    p50 = {intent: s["p50"] for intent, s in per_intent.items()}
    p95 = {intent: s["p95"] for intent, s in per_intent.items()}
    p99 = {intent: s["p99"] for intent, s in per_intent.items()}

    return BaselineSnapshot(
        run_id=run_id,
        captured_at=datetime.utcnow(),
        git_sha=git_sha,
        api_base_url=api_base_url,
        num_queries=len(results),
        per_domain_counts=dict(counter),
        ragas=ragas.details,
        sme_metrics=sme_metrics,
        latency_p50_per_intent=p50,
        latency_p95_per_intent=p95,
        latency_p99_per_intent=p99,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DocWain SME eval baseline runner")
    parser.add_argument("--eval-dir", type=Path, default=Path("tests/sme_evalset_v1/queries"))
    parser.add_argument("--fixtures", type=Path,
                        default=Path("tests/sme_evalset_v1/fixtures/test_profiles.yaml"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--results-jsonl", type=Path,
                        default=Path("tests/sme_results.jsonl"))
    parser.add_argument("--api-base-url", default=os.environ.get("DOCWAIN_API_URL",
                                                                 "http://localhost:8000"))
    parser.add_argument("--skip-llm-judge", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Limit # of queries (debug)")
    args = parser.parse_args(argv)

    # Load queries
    queries = load_all_queries(args.eval_dir)
    if args.limit:
        queries = queries[: args.limit]
    if not queries:
        print(f"ERROR: no queries found under {args.eval_dir}", file=sys.stderr)
        return 2

    # Override subscription/profile IDs from fixtures (queries ship with placeholder IDs;
    # fixtures file supplies the real ones at run time)
    fixtures = yaml.safe_load(args.fixtures.read_text(encoding="utf-8"))
    sub_id = fixtures["test_subscription"]["subscription_id"]
    domain_to_profile = {
        dom: cfg["profile_id"] for dom, cfg in fixtures["profiles"].items()
    }
    for q in queries:
        q_sub = sub_id
        q_prof = domain_to_profile.get(q.profile_domain, q.profile_id)
        q.subscription_id = q_sub
        q.profile_id = q_prof

    # Run
    run_id = f"baseline_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    store = JsonlResultStore(args.results_jsonl)
    config = RunnerConfig(base_url=args.api_base_url)

    print(f"[run_baseline] running {len(queries)} queries; run_id={run_id}")
    with QueryRunner(config) as runner:
        for i, q in enumerate(queries, 1):
            result = runner.run_one(q, run_id=run_id)
            store.append(result)
            if i % 25 == 0 or i == len(queries):
                print(f"  … {i}/{len(queries)} done")

    # Compose snapshot
    judge_fn = None if args.skip_llm_judge else _docwain_llm_judge
    results = list(store.iter_run(run_id))
    snapshot = compose_snapshot(
        results,
        run_id=run_id,
        git_sha=_current_git_sha(),
        api_base_url=args.api_base_url,
        judge_fn=judge_fn,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(snapshot.model_dump_json(indent=2))
    print(f"[run_baseline] snapshot written: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/scripts/sme_eval/test_run_baseline.py -v`
Expected: PASS for all 3 tests

- [ ] **Step 5: Commit**

```bash
git add scripts/sme_eval/run_baseline.py tests/scripts/sme_eval/test_run_baseline.py
git commit -m "phase0(sme-eval): baseline CLI orchestrator"
```

---

## Task 16: Seed query curation — 600 queries across 6 domains

This is a **content task**, not a code task. An engineer cannot generate 600 realistic synthetic SME queries from a plan without domain knowledge. Split into 6 curation sub-tasks (one per domain). The engineer coordinates with a domain expert for each or uses the provided templates + an LLM-assisted generation pass reviewed by an expert.

**Files:**
- Create: `tests/sme_evalset_v1/queries/finance.yaml` — 100 queries
- Create: `tests/sme_evalset_v1/queries/legal.yaml` — 100 queries
- Create: `tests/sme_evalset_v1/queries/hr.yaml` — 100 queries
- Create: `tests/sme_evalset_v1/queries/medical.yaml` — 100 queries
- Create: `tests/sme_evalset_v1/queries/it_support.yaml` — 100 queries
- Create: `tests/sme_evalset_v1/queries/generic.yaml` — 100 queries

### Distribution per domain (100 queries)

Each domain YAML must contain an even intent mix to exercise every code path:

| Intent | Count per domain | Purpose |
|---|---|---|
| `lookup` | 20 | Trivial-intent baseline (today's fast path) |
| `extract` | 15 | Structured extraction baseline |
| `list` / `aggregate` / `count` | 10 combined | Summary baselines |
| `summarize` / `overview` | 10 combined | Borderline-intent baseline |
| `compare` | 10 | Cross-doc baseline (Phase 3 improvement target) |
| `investigate` | 5 | Rich-intent baseline |
| `analyze` | 10 | NEW intent baseline (will show ~0.0 analytical coverage today) |
| `diagnose` | 10 | NEW intent baseline (troubleshooting where applicable, esp. it_support) |
| `recommend` | 10 | NEW intent baseline (will show recommendation_groundedness ~0.0 today) |

Each query includes `expected_behavior` in prose — what a good SME answer would look like — to help the human rater AND to prompt-engineer later phases without re-creating the eval set.

### Seed query templates per domain

Each domain uses domain-appropriate query templates. Copy-paste starters:

- [ ] **Step 1: Create `tests/sme_evalset_v1/queries/finance.yaml` with 100 queries**

Template starter — expand to 100:

```yaml
version: 1
domain: finance
queries:
  # lookup (20 total)
  - query_id: finance_lookup_001
    query_text: "What was the total Q3 revenue?"
    intent: lookup
    profile_domain: finance
    subscription_id: PLACEHOLDER_SUBSCRIPTION
    profile_id: PLACEHOLDER_FINANCE_PROFILE
    expected_behavior: "Direct answer with the number from Q3 report, bolded."
    tags: [trivial, quarterly]
  - query_id: finance_lookup_002
    query_text: "When was invoice INV-2026-Q3-0048 paid?"
    intent: lookup
    profile_domain: finance
    subscription_id: PLACEHOLDER_SUBSCRIPTION
    profile_id: PLACEHOLDER_FINANCE_PROFILE
    expected_behavior: "Date answer, bolded, with source citation."
    tags: [trivial, invoice]
  # ... continue 18 more lookup queries
  # extract (15 total)
  - query_id: finance_extract_001
    query_text: "List all line items over $10,000 from Q2 invoices."
    intent: extract
    profile_domain: finance
    subscription_id: PLACEHOLDER_SUBSCRIPTION
    profile_id: PLACEHOLDER_FINANCE_PROFILE
    expected_behavior: "Markdown table: item / vendor / amount / date. Bold amounts."
    tags: [structured]
  # ... continue 14 more extract queries
  # analyze (10 total) — these exercise the new intent
  - query_id: finance_analyze_001
    query_text: "Analyze revenue trends across Q1, Q2, and Q3."
    intent: analyze
    profile_domain: finance
    subscription_id: PLACEHOLDER_SUBSCRIPTION
    profile_id: PLACEHOLDER_FINANCE_PROFILE
    expected_behavior: "Executive summary -> observations -> trend interpretation -> caveats. Must cite ≥2 quarters' reports."
    tags: [rich, trend]
  # ... continue 9 more analyze queries
  # recommend (10 total)
  - query_id: finance_recommend_001
    query_text: "Where can we cut costs next quarter to improve margins?"
    intent: recommend
    profile_domain: finance
    subscription_id: PLACEHOLDER_SUBSCRIPTION
    profile_id: PLACEHOLDER_FINANCE_PROFILE
    expected_behavior: "3-5 recommendations, each with rationale, evidence from profile docs, estimated impact, caveats."
    tags: [rich, cost_opt]
  # ... continue 9 more recommend queries
  # diagnose (10 total) — less natural for finance but still possible: "why did margin drop?"
  - query_id: finance_diagnose_001
    query_text: "Why did our Q3 gross margin drop?"
    intent: diagnose
    profile_domain: finance
    subscription_id: PLACEHOLDER_SUBSCRIPTION
    profile_id: PLACEHOLDER_FINANCE_PROFILE
    expected_behavior: "Symptom -> ranked candidate causes with evidence -> cross-quarter comparison -> caveats."
    tags: [rich, causal]
  # ... continue per the distribution table above
```

Engineer process for each domain YAML:
1. Review `tests/sme_evalset_v1/fixtures/test_profiles.yaml` for the profile's synthetic documents (inventory provided by operator).
2. Write queries grounded in those documents' actual content — not invented facts.
3. For each query, write the `expected_behavior` prose from an SME perspective: what would a good answer look like?
4. Include 3-5 realistic `tags` per query for filterability.
5. Keep `gold_answer_snippets` empty for now — filled in during human-rating pass.
6. No customer data, ever. Synthetic only.

- [ ] **Step 2: Create the remaining five domain YAMLs the same way**

Apply the same distribution and template structure to `legal.yaml`, `hr.yaml`, `medical.yaml`, `it_support.yaml`, `generic.yaml`. Each 100 queries. Adjust query phrasing to the domain's vocabulary:

- **legal:** parties, obligations, effective dates, termination clauses, conflicts between versions, governing law
- **hr:** employees, reporting lines, benefits, policies, tenure, certifications, hiring patterns
- **medical:** patients, diagnoses, prescriptions, treatment timelines, drug interactions (be extra careful with grounding — these queries must NOT elicit advice)
- **it_support:** symptoms, error codes, systems, log patterns, incident history, known issues, fix procedures. Heavier on `diagnose` intent.
- **generic:** domain-neutral; test the `generic.yaml` adapter fallback path — mixed content, general Q&A.

- [ ] **Step 3: Validate YAMLs load cleanly**

Run: `python -c "from scripts.sme_eval.run_baseline import load_all_queries; qs = load_all_queries('tests/sme_evalset_v1/queries'); print(f'loaded {len(qs)} queries'); print({q.profile_domain for q in qs})"`
Expected: `loaded 600 queries` and the set `{'finance', 'legal', 'hr', 'medical', 'it_support', 'generic'}`

- [ ] **Step 4: Commit**

```bash
git add -f tests/sme_evalset_v1/queries/
git commit -m "phase0(sme-eval): 600-query eval set across 6 domains"
```

---

## Task 17: Documentation and runbook

**Files:**
- Modify: `tests/sme_evalset_v1/README.md` (extend with runbook)
- Create: `scripts/sme_eval/README.md`

- [ ] **Step 1: Expand the evalset README with operator runbook**

Modify `tests/sme_evalset_v1/README.md` — append:

```markdown
## Running the baseline — operator runbook

### Prerequisites
1. DocWain API is running (defaults to `http://localhost:8000`; override via `DOCWAIN_API_URL` env).
2. DocWain LLM gateway is reachable (defaults `http://localhost:8100/v1/chat/completions`; override via `DOCWAIN_LLM_URL`).
3. Test subscription + 6 test profiles exist in production DocWain with synthetic documents ingested; their IDs are filled into `tests/sme_evalset_v1/fixtures/test_profiles.yaml`.
4. All test-profile documents have `PIPELINE_TRAINING_COMPLETED`.

### Run the baseline
```bash
python -m scripts.sme_eval.run_baseline \
    --eval-dir tests/sme_evalset_v1/queries \
    --fixtures tests/sme_evalset_v1/fixtures/test_profiles.yaml \
    --out tests/sme_metrics_baseline_$(date +%Y-%m-%d).json \
    --results-jsonl tests/sme_results_$(date +%Y-%m-%d).jsonl
```

- Runtime: ~10-40 minutes depending on DocWain latency. Sequential by design.
- Output: `tests/sme_metrics_baseline_YYYY-MM-DD.json` (committed, frozen)
- Raw results: `tests/sme_results_YYYY-MM-DD.jsonl` (committed; optional — can be gitignored if size grows)

### Human rating pass (after automated run completes)
```bash
python -c "
from scripts.sme_eval.result_store import JsonlResultStore
from scripts.sme_eval.human_rating import export_for_rating
results = list(JsonlResultStore('tests/sme_results_YYYY-MM-DD.jsonl').iter_all())
export_for_rating(results, 'tests/sme_human_rating_YYYY-MM-DD.csv')
"
# Distribute the CSV to domain experts. Each rates sme_score_1_to_5.
# Collect their rated CSVs, merge, import:
python -c "
from scripts.sme_eval.human_rating import import_ratings
ratings = import_ratings('tests/sme_human_rating_YYYY-MM-DD.csv')
import json
snap = json.load(open('tests/sme_metrics_baseline_YYYY-MM-DD.json'))
vals = list(ratings.values())
snap['human_rated_sme_score_avg'] = sum(vals) / len(vals) if vals else None
snap['human_rated_count'] = len(vals)
json.dump(snap, open('tests/sme_metrics_baseline_YYYY-MM-DD.json', 'w'), indent=2)
"
```

### Interpreting the baseline
The baseline snapshot's `ragas` block should roughly match the pre-existing
`tests/ragas_metrics.json` within noise. If it diverges materially, that's
a signal that either the eval set has drifted or the RAGAS wrapper's heuristics
aren't aligned with `scripts/ragas_evaluator.py`. Investigate before trusting
any later phase's gate.

### Subsequent-phase regression run
Phase 2+ re-runs this baseline against the phase's build and compares to
`tests/sme_metrics_baseline_YYYY-MM-DD.json`. Launch-gate conditions (Section 10
of the design spec) must hold.
```

- [ ] **Step 2: Write the scripts/sme_eval README**

Create `scripts/sme_eval/README.md`:

```markdown
# DocWain SME Evaluation Tooling

Measurement harness for sub-project A (Profile-SME reasoning layer).

## Modules
- `schema.py` — Pydantic models (EvalQuery, EvalResult, MetricResult, BaselineSnapshot)
- `query_runner.py` — HTTP client against DocWain /api/ask
- `result_store.py` — Append-only JSONL store for results
- `metrics/` — One metric per file (RAGAS wrapper + 6 new reasoning metrics)
- `aggregate.py` — p50/p95/p99 per intent
- `human_rating.py` — CSV export/import for expert rating
- `run_baseline.py` — CLI orchestrator

## Running locally
See `tests/sme_evalset_v1/README.md` for the operator runbook.

## Adding a new metric
1. Create `scripts/sme_eval/metrics/<name>.py` subclassing `Metric` from `_base.py`.
2. Create `tests/scripts/sme_eval/metrics/test_<name>.py` with at least 3 test cases.
3. Add to `DEFAULT_METRICS` in `run_baseline.py`.
4. Document the metric's value range, pass threshold, and interpretation in the spec.

## Memory rules applied
- No customer data — eval queries are synthetic, see `tests/sme_evalset_v1/README.md`.
- Measurement tool only — no production code changes under `src/` touched by Phase 0.
- The per-request httpx timeout is a per-operation safety limit, NOT a response-latency cutoff (consistent with "No Timeouts; Use Efficiency" rule).
```

- [ ] **Step 3: Commit**

```bash
git add -f tests/sme_evalset_v1/README.md scripts/sme_eval/README.md
git commit -m "phase0(sme-eval): operator runbook + module docs"
```

---

## Task 18: Run the baseline and freeze the before-picture

**Files:**
- Create: `tests/sme_metrics_baseline_YYYY-MM-DD.json` (replace YYYY-MM-DD with actual run date)
- Create: `tests/sme_results_YYYY-MM-DD.jsonl`

**This is an operator task that produces committed output**, not a code task. The engineer executes the runbook from Task 17 and commits the resulting baseline.

- [ ] **Step 1: Confirm preconditions**

- Fixtures filled in with real subscription + profile IDs (see Task 3 placeholder)
- All 600 queries validated (Task 16 Step 3 passed)
- DocWain API reachable and healthy
- DocWain LLM gateway reachable

- [ ] **Step 2: Run the baseline**

```bash
python -m scripts.sme_eval.run_baseline \
    --eval-dir tests/sme_evalset_v1/queries \
    --fixtures tests/sme_evalset_v1/fixtures/test_profiles.yaml \
    --out tests/sme_metrics_baseline_$(date +%Y-%m-%d).json \
    --results-jsonl tests/sme_results_$(date +%Y-%m-%d).jsonl
```

- [ ] **Step 3: Sanity-check the output**

```bash
python -c "
import json
snap = json.load(open('tests/sme_metrics_baseline_$(date +%Y-%m-%d).json'))
print('num_queries:', snap['num_queries'])
print('per_domain:', snap['per_domain_counts'])
print('ragas.faithfulness:', snap['ragas']['answer_faithfulness'])
print('ragas.hallucination:', snap['ragas']['hallucination_rate'])
for name, m in snap['sme_metrics'].items():
    print(f'{name}: {m[\"value\"]:.3f}')
for intent, p50 in snap['latency_p50_per_intent'].items():
    print(f'latency p50 {intent}: {p50:.0f}ms')
"
```

Expected:
- `num_queries == 600`
- `per_domain` shows 100 per domain across all 6
- `ragas.answer_faithfulness` within ~10% of the 0.514 in `tests/ragas_metrics.json`
- `ragas.hallucination_rate` near 0.0
- `recommendation_groundedness` expected low (likely < 0.3) — model rarely produces grounded recommendations today
- `cross_doc_integration_rate` expected low-to-mid — measures how often current responses cite multiple docs
- `insight_novelty` expected mid (~0.3-0.6 — today's responses are partly extractive, partly synthesizing)
- `verified_removal_rate` near 1.0 (today's API doesn't emit the verifier-dropped field)
- `sme_artifact_hit_rate == 0.0` (artifacts don't exist yet — this is the measured zero)

- [ ] **Step 4: Run the human-rating export**

```bash
python -c "
from scripts.sme_eval.result_store import JsonlResultStore
from scripts.sme_eval.human_rating import export_for_rating
import datetime
today = datetime.date.today().isoformat()
store = JsonlResultStore(f'tests/sme_results_{today}.jsonl')
export_for_rating(list(store.iter_all()), f'tests/sme_human_rating_{today}.csv')
"
```

Distribute the CSV to domain experts (one per domain is sufficient). Collect rated CSVs. Merge into a single CSV (same columns) and run the import step per the runbook to update the snapshot.

- [ ] **Step 5: Commit the frozen baseline**

```bash
git add -f tests/sme_metrics_baseline_*.json tests/sme_results_*.jsonl
git commit -m "phase0(sme-eval): frozen baseline — run $(date +%Y-%m-%d)"
```

- [ ] **Step 6: Tag the baseline**

```bash
git tag -a sme-baseline-v1 -m "Frozen SME eval baseline (Phase 0 exit)"
```

---

## Phase 0 exit checklist

Run this before declaring Phase 0 done. Each box must be genuinely ticked, not wishfully checked.

- [ ] All 18 tasks committed with passing tests
- [ ] `tests/sme_metrics_baseline_YYYY-MM-DD.json` committed and tagged
- [ ] RAGAS portion of baseline within ~10% of historical `tests/ragas_metrics.json`
- [ ] Human-rated SME score collected for at least 50 queries per domain (300 total)
- [ ] Per-intent latency distributions recorded
- [ ] Operator runbook exercised end-to-end by a second engineer
- [ ] Spec Section 10 launch gates are now expressible as concrete deltas from this baseline
- [ ] No changes to any file under `src/` in the entire Phase 0 branch
- [ ] `pytest tests/scripts/sme_eval -v` shows all green

---

## Self-review appendix

**Spec coverage check:** every item in Section 10 of the design spec (Measurement framework) has a task above. Specifically:
- Existing RAGAS baselines → Task 6 (wrapper) + Task 18 (run)
- 6 new reasoning metrics → Tasks 7-12 (one each)
- Per-intent latency distributions → Task 4 (capture) + Task 13 (aggregate) + Task 18 (run)
- Human-rated SME score → Task 14 (tool) + Task 18 Step 4 (run)
- Eval set at `tests/sme_evalset_v1/` → Task 3 (fixtures) + Task 16 (queries)
- Measurement-only, no production changes → implicit throughout; enforced by exit checklist

**Placeholder scan:** re-read every task for "TBD", "TODO", "fill in" — none found.

**Type consistency:** `EvalQuery`, `EvalResult`, `MetricResult`, `BaselineSnapshot` defined once in `schema.py` and referenced consistently across all tasks. `Metric.compute(results)` signature identical in base class and all 7 subclass implementations.

**No redefinition:** each file created exactly once; no task claims to "modify" a file another task creates in a later step.
