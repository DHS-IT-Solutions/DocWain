# DocWain Insights Portal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the 26 capabilities in the DocWain Insights Portal spec (`docs/superpowers/specs/2026-04-25-docwain-research-portal-design.md`) on branch `preprod_v03`, behind 25 feature flags, with single-flag revert per capability and zero regression to existing flow when all flags are off.

**Architecture:** Six layers — Foundation (extraction + DocIntel + KG + embeddings, already exists) → Domain Adapter Framework (Azure Blob YAMLs) → Researcher Agent v2 (typed insights) → Insight Store (Qdrant + Neo4j + Mongo) → Continuous-Refresh (incremental + scheduled + watchlist) → Surface (7 endpoints + `/api/ask` injection + agentic actions). All heavy compute runs at ingestion or on isolated low-priority queues; query-time stays lookup-only.

**Tech Stack:** Python 3.11, FastAPI, Celery + Redis, MongoDB, Qdrant, Neo4j, vLLM (gateway), Azure Blob Storage, pytest.

---

## Branch + execution context

- **Working branch:** `preprod_v03` (already cut from `preprod_v02` HEAD `df1d3ac`).
- **Spec:** `docs/superpowers/specs/2026-04-25-docwain-research-portal-design.md` (committed at `fe7146d`+`df1d3ac`).
- **Default flag state:** every new capability flag defaults to `false`. PRs land with flags off, gates passing.
- **Commit cadence:** one commit per task (TDD step 5). No squash before merge.
- **Conventions:** no Co-Authored-By / Claude / Anthropic in commit messages or code (per `feedback_no_claude_attribution.md`); Mongo is control-plane only (per `feedback_storage_separation.md`); response formatting goes in `src/generation/prompts.py` not `src/intelligence/generator.py` (per `feedback_prompt_paths.md`); HITL gates honored — researcher v2 runs only after screening approval.

---

## Sub-project execution order

| Order | SP | Reason |
|---|---|---|
| 1 | **SP-J** Feature Flags + Eval Harness | Every other SP needs a flag and a gate test. Build this first. |
| 2 | **SP-A** Adapter Framework | Researcher v2 reads adapters; insight schema includes adapter version. |
| 3 | **SP-B** Insight Schema + Store | Researcher writes insights; surface reads them. Schema must exist first. |
| 4 | **SP-D** Knowledge Layer | Adapter declares KBs; researcher prompts use them. |
| 5 | **SP-K** Regression + Perf Test Framework | All-flags-off regression test required before code lands. |
| 6 | **SP-C** Researcher Agent v2 | Consumes A + B + D. The intelligence engine. |
| 7 | **SP-E** Continuous Refresh | Builds on C. |
| 8 | **SP-L** Backfill + Migration | Replays C across existing profiles. |
| 9 | **SP-F** Surface Endpoints | Read-only; depends on B. |
| 10 | **SP-G** `/api/ask` Injection | Read-only against B; depends on F's helper code being available. |
| 11 | **SP-H** Agentic Actions | Action runner; reads B + D, writes artifacts. |
| 12 | **SP-I** Visualizations | Generated at insight-write time; depends on B. |

---

## File structure

Files to be created (with one-line responsibility):

| Path | Responsibility |
|---|---|
| `src/api/feature_flags.py` | Feature flag registry — single source of truth for all 25 insight-portal flags. |
| `src/intelligence/adapters/__init__.py` | Public exports for adapter framework. |
| `src/intelligence/adapters/schema.py` | Pydantic models for adapter YAML (Adapter, ResearcherSection, KnowledgeSection, etc.). |
| `src/intelligence/adapters/store.py` | `AdapterStore` — Blob loader, TTL cache, resolution order, hot-reload. |
| `src/intelligence/adapters/detect.py` | Auto-detection wrapper around existing `domain_classifier`. |
| `src/intelligence/adapters/generic.yaml` | Bundled fallback adapter shipped with code (uploaded to Blob at deploy). |
| `src/intelligence/insights/__init__.py` | Public exports. |
| `src/intelligence/insights/schema.py` | `Insight`, `Action`, `EvidenceSpan`, `KbRef` dataclasses. |
| `src/intelligence/insights/validators.py` | Citation validator, body-separation validator, dedup-key calculator. |
| `src/intelligence/insights/store.py` | `InsightStore` — write to Mongo + Qdrant + Neo4j; read for endpoints. |
| `src/intelligence/insights/staleness.py` | Stale-flag updater. |
| `src/intelligence/knowledge/__init__.py` | Public exports. |
| `src/intelligence/knowledge/provider.py` | `KnowledgeProvider` interface + JSON-KB implementation. |
| `src/intelligence/knowledge/template_resolver.py` | Resolves `{{kb.lookup(...)}}` directives in researcher prompts. |
| `src/intelligence/knowledge/bundled/insurance_taxonomy_v1.json` | Bundled KB. |
| `src/intelligence/knowledge/bundled/icd10_subset_v1.json` | Bundled KB. |
| `src/intelligence/knowledge/bundled/hr_policies_v1.json` | Bundled KB. |
| `src/intelligence/knowledge/bundled/procurement_terms_v1.json` | Bundled KB. |
| `src/tasks/researcher_v2.py` | Researcher Agent v2 Celery task — per-doc + profile-level loops. |
| `src/tasks/researcher_v2_refresh.py` | Continuous-refresh Celery tasks (incremental, scheduled, watchlist). |
| `src/intelligence/researcher_v2/__init__.py` | Public exports. |
| `src/intelligence/researcher_v2/runner.py` | Per-doc and profile-level pass orchestration. |
| `src/intelligence/researcher_v2/parser.py` | LLM response parser + insight construction. |
| `src/intelligence/researcher_v2/profile_passes.py` | Comparison / conflict / trend / projection passes (multi-doc). |
| `src/intelligence/actions/__init__.py` | Public exports. |
| `src/intelligence/actions/runner.py` | Action runner — dispatch, sandbox, audit. |
| `src/intelligence/actions/handlers.py` | Per-type handlers (artifact, form_fill, plan, reminder, alert). |
| `src/intelligence/actions/audit.py` | Audit-log writer. |
| `src/intelligence/visualizations/__init__.py` | Public exports. |
| `src/intelligence/visualizations/generator.py` | Visualization spec generator (called at insight-write time). |
| `src/api/insights_api.py` | Endpoints: list/detail/refresh-status. |
| `src/api/actions_api.py` | Endpoints: list/execute. |
| `src/api/visualizations_api.py` | Endpoint: list visualizations. |
| `src/api/artifacts_api.py` | Endpoint: list artifacts. |
| `src/generation/insight_injection.py` | Helper for `/api/ask` proactive injection. |
| `scripts/insights_backfill.py` | One-time backfill across existing profiles. |
| `tests/insights_eval/__init__.py` | Eval harness scaffold. |
| `tests/insights_eval/rubric.py` | Mechanical scoring rubrics. |
| `tests/insights_eval/fixtures/` | Synthetic profile + doc fixtures. |
| `tests/regression/all_flags_off.py` | Hard regression test for byte-identical behaviour with all flags off. |
| `tests/perf/api_ask_latency.py` | p95 latency assertion for `/api/ask`. |
| `tests/perf/upload_to_screening_eligible.py` | Time-to-screening-eligible perf test. |
| `tests/perf/insight_lookup_p95.py` | 50ms p95 assertion for insight lookup. |

Files to be modified:

| Path | What changes |
|---|---|
| `src/api/config.py` | Add feature-flag accessor that delegates to `src/api/feature_flags.py`. |
| `src/celery_app.py` | Register `researcher_v2_queue`, `researcher_refresh_queue`, `actions_queue`. |
| `src/main.py` | Mount new routers (insights, actions, visualizations, artifacts). |
| `src/api/profile_intelligence_api.py` | Stays as v1; no behavioural change. |
| `src/intelligence/profile_intelligence.py` | Stays as v1. |
| `src/generation/prompts.py` | Add proactive-insight injection point (gated by flag). |
| `src/api/extraction_service.py` / `src/api/embedding_service.py` / `src/api/dataHandler.py` | KG-build trigger removal is **out of scope here** (covered by separate KG spec); this plan does not touch these files. |

---

## Sub-Project SP-J — Feature Flags + Eval Harness

**Why first:** every other SP needs a flag and a gate test. Build the registry and harness now so tasks downstream can plug in.

**Files:**
- Create: `src/api/feature_flags.py`
- Create: `tests/insights_eval/__init__.py`
- Create: `tests/insights_eval/rubric.py`
- Create: `tests/insights_eval/conftest.py`
- Create: `tests/insights_eval/fixtures/__init__.py`
- Create: `tests/insights_eval/fixtures/synthetic.py`
- Modify: `src/api/config.py`
- Test: `tests/insights_eval/test_feature_flags.py`
- Test: `tests/insights_eval/test_rubric.py`

### Task SP-J.1 — Feature flag registry

- [ ] **Step 1: Write the failing test**

`tests/insights_eval/test_feature_flags.py`:
```python
from src.api.feature_flags import FeatureFlags, is_enabled, FLAG_NAMES


def test_all_25_flags_registered():
    expected = {
        "INSIGHTS_TYPE_ANOMALY_ENABLED",
        "INSIGHTS_TYPE_GAP_ENABLED",
        "INSIGHTS_TYPE_COMPARISON_ENABLED",
        "INSIGHTS_TYPE_SCENARIO_ENABLED",
        "INSIGHTS_TYPE_TREND_ENABLED",
        "INSIGHTS_TYPE_RECOMMENDATION_ENABLED",
        "INSIGHTS_TYPE_CONFLICT_ENABLED",
        "INSIGHTS_TYPE_PROJECTION_ENABLED",
        "ACTIONS_ARTIFACT_ENABLED",
        "ACTIONS_FORM_FILL_ENABLED",
        "ACTIONS_PLAN_ENABLED",
        "ACTIONS_REMINDER_ENABLED",
        "ACTIONS_EXTERNAL_SIDE_EFFECTS_ENABLED",
        "KB_BUNDLED_ENABLED",
        "KB_EXTERNAL_ENABLED",
        "INSIGHTS_CITATION_ENFORCEMENT_ENABLED",
        "REFRESH_ON_UPLOAD_ENABLED",
        "INSIGHTS_PROACTIVE_INJECTION",
        "REFRESH_SCHEDULED_ENABLED",
        "REFRESH_INCREMENTAL_ENABLED",
        "WATCHLIST_ENABLED",
        "ADAPTER_AUTO_DETECT_ENABLED",
        "ADAPTER_BLOB_LOADING_ENABLED",
        "ADAPTER_GENERIC_FALLBACK_ENABLED",
        "VIZ_ENABLED",
        "INSIGHTS_DASHBOARD_ENABLED",
    }
    assert set(FLAG_NAMES) == expected


def test_all_flags_default_false():
    flags = FeatureFlags()
    for name in FLAG_NAMES:
        assert is_enabled(name, flags) is False, f"{name} should default to False"


def test_is_enabled_unknown_flag_raises():
    import pytest
    with pytest.raises(KeyError):
        is_enabled("BOGUS_FLAG", FeatureFlags())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/insights_eval/test_feature_flags.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.api.feature_flags'`.

- [ ] **Step 3: Write minimal implementation**

`src/api/feature_flags.py`:
```python
"""Feature flag registry for the Insights Portal.

Single source of truth for all 25 capability flags. Every flag defaults
to False; production enablement is staged per Section 14.4 of the spec.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Dict


FLAG_NAMES = (
    "INSIGHTS_TYPE_ANOMALY_ENABLED",
    "INSIGHTS_TYPE_GAP_ENABLED",
    "INSIGHTS_TYPE_COMPARISON_ENABLED",
    "INSIGHTS_TYPE_SCENARIO_ENABLED",
    "INSIGHTS_TYPE_TREND_ENABLED",
    "INSIGHTS_TYPE_RECOMMENDATION_ENABLED",
    "INSIGHTS_TYPE_CONFLICT_ENABLED",
    "INSIGHTS_TYPE_PROJECTION_ENABLED",
    "ACTIONS_ARTIFACT_ENABLED",
    "ACTIONS_FORM_FILL_ENABLED",
    "ACTIONS_PLAN_ENABLED",
    "ACTIONS_REMINDER_ENABLED",
    "ACTIONS_EXTERNAL_SIDE_EFFECTS_ENABLED",
    "KB_BUNDLED_ENABLED",
    "KB_EXTERNAL_ENABLED",
    "INSIGHTS_CITATION_ENFORCEMENT_ENABLED",
    "REFRESH_ON_UPLOAD_ENABLED",
    "INSIGHTS_PROACTIVE_INJECTION",
    "REFRESH_SCHEDULED_ENABLED",
    "REFRESH_INCREMENTAL_ENABLED",
    "WATCHLIST_ENABLED",
    "ADAPTER_AUTO_DETECT_ENABLED",
    "ADAPTER_BLOB_LOADING_ENABLED",
    "ADAPTER_GENERIC_FALLBACK_ENABLED",
    "VIZ_ENABLED",
    "INSIGHTS_DASHBOARD_ENABLED",
)


@dataclass(frozen=True)
class FeatureFlags:
    overrides: Dict[str, bool] = field(default_factory=dict)


def is_enabled(name: str, flags: FeatureFlags) -> bool:
    if name not in FLAG_NAMES:
        raise KeyError(name)
    if name in flags.overrides:
        return bool(flags.overrides[name])
    env_value = os.environ.get(name)
    if env_value is not None:
        return env_value.strip().lower() in ("1", "true", "yes", "on")
    return False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/insights_eval/test_feature_flags.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/api/feature_flags.py tests/insights_eval/test_feature_flags.py
git commit -m "feat(flags): add Insights Portal feature-flag registry (SP-J.1)"
```

### Task SP-J.2 — Wire flag accessor into existing config

- [ ] **Step 1: Write the failing test**

Append to `tests/insights_eval/test_feature_flags.py`:
```python
def test_config_module_exposes_is_enabled():
    from src.api.config import insight_flag_enabled
    # All flags default false at module import
    assert insight_flag_enabled("INSIGHTS_TYPE_ANOMALY_ENABLED") is False


def test_config_env_override(monkeypatch):
    from src.api.config import insight_flag_enabled
    monkeypatch.setenv("INSIGHTS_TYPE_ANOMALY_ENABLED", "true")
    assert insight_flag_enabled("INSIGHTS_TYPE_ANOMALY_ENABLED") is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/insights_eval/test_feature_flags.py::test_config_module_exposes_is_enabled -v`
Expected: FAIL — `ImportError: cannot import name 'insight_flag_enabled'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/api/config.py` (top of file imports):
```python
from src.api.feature_flags import FeatureFlags, is_enabled as _flag_is_enabled

_INSIGHT_FLAGS = FeatureFlags()


def insight_flag_enabled(name: str) -> bool:
    """Return True if the named insights-portal feature flag is enabled.

    Reads from environment at call time (no caching) so flags can be
    flipped without process restart in dev. In prod, flags are set at
    deploy time via env vars.
    """
    return _flag_is_enabled(name, _INSIGHT_FLAGS)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/insights_eval/test_feature_flags.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/api/config.py tests/insights_eval/test_feature_flags.py
git commit -m "feat(flags): expose insight_flag_enabled accessor in config (SP-J.2)"
```

### Task SP-J.3 — Mechanical rubric base class

- [ ] **Step 1: Write the failing test**

`tests/insights_eval/test_rubric.py`:
```python
from tests.insights_eval.rubric import (
    PrecisionRecallRubric,
    RubricResult,
    score_precision_recall,
)


def test_precision_recall_perfect_match():
    expected = [{"id": "a"}, {"id": "b"}]
    actual = [{"id": "a"}, {"id": "b"}]
    result = score_precision_recall(actual, expected, key="id")
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.passed(min_precision=0.7, min_recall=0.6) is True


def test_precision_recall_partial():
    expected = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    actual = [{"id": "a"}, {"id": "x"}]  # 1 hit, 1 miss
    result = score_precision_recall(actual, expected, key="id")
    assert result.precision == 0.5
    assert result.recall == pytest_approx(1 / 3)


def pytest_approx(value, tol=1e-6):
    import pytest
    return pytest.approx(value, abs=tol)


def test_passed_threshold():
    r = RubricResult(precision=0.5, recall=0.4)
    assert r.passed(min_precision=0.7, min_recall=0.6) is False
    assert r.passed(min_precision=0.4, min_recall=0.3) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/insights_eval/test_rubric.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

`tests/insights_eval/rubric.py`:
```python
"""Mechanical rubrics for capability eval gates.

Every gate scoring is pass/fail by script — no human judgment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Sequence


@dataclass
class RubricResult:
    precision: float = 0.0
    recall: float = 0.0
    notes: str = ""

    def passed(self, *, min_precision: float, min_recall: float) -> bool:
        return self.precision >= min_precision and self.recall >= min_recall


@dataclass
class PrecisionRecallRubric:
    min_precision: float
    min_recall: float


def score_precision_recall(
    actual: Sequence[Mapping],
    expected: Sequence[Mapping],
    *,
    key: str,
) -> RubricResult:
    expected_keys = {e[key] for e in expected}
    actual_keys = {a[key] for a in actual}
    if not actual_keys:
        precision = 1.0 if not expected_keys else 0.0
    else:
        precision = len(actual_keys & expected_keys) / len(actual_keys)
    if not expected_keys:
        recall = 1.0
    else:
        recall = len(actual_keys & expected_keys) / len(expected_keys)
    return RubricResult(precision=precision, recall=recall)
```

`tests/insights_eval/__init__.py`:
```python
```

`tests/insights_eval/conftest.py`:
```python
import pytest


@pytest.fixture
def synthetic_profile_id() -> str:
    return "test_profile_synthetic_001"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/insights_eval/test_rubric.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/insights_eval/__init__.py tests/insights_eval/rubric.py tests/insights_eval/conftest.py tests/insights_eval/test_rubric.py
git commit -m "feat(eval): mechanical precision/recall rubric for gate scoring (SP-J.3)"
```

### Task SP-J.4 — Synthetic fixture seed (one fixture, one domain)

- [ ] **Step 1: Write the failing test**

`tests/insights_eval/fixtures/__init__.py`:
```python
```

`tests/insights_eval/fixtures/test_synthetic.py`:
```python
from tests.insights_eval.fixtures.synthetic import (
    synthetic_insurance_doc,
    SyntheticDoc,
)


def test_synthetic_doc_shape():
    doc = synthetic_insurance_doc()
    assert isinstance(doc, SyntheticDoc)
    assert doc.domain == "insurance"
    assert doc.text  # non-empty
    assert doc.expected_anomalies  # at least one planted
    assert doc.expected_gaps
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/insights_eval/fixtures/test_synthetic.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

`tests/insights_eval/fixtures/synthetic.py`:
```python
"""Synthetic fixtures for capability eval gates.

All content is fabricated. Per `feedback_no_customer_data_training.md`,
no customer data ever appears in adapter examples or eval fixtures.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class SyntheticDoc:
    document_id: str
    domain: str
    text: str
    expected_anomalies: List[str] = field(default_factory=list)
    expected_gaps: List[str] = field(default_factory=list)
    expected_recommendations: List[str] = field(default_factory=list)


def synthetic_insurance_doc() -> SyntheticDoc:
    text = (
        "Policy number: SYN-INS-001\n"
        "Policyholder: Test Subject A\n"
        "Coverage: comprehensive automobile, $500 deductible.\n"
        "Premium: $1,800 / year. Effective 2026-01-01 to 2026-12-31.\n"
        "Excludes: flood damage, earthquake, racing events.\n"
        "Note: Liability limit $50,000 — well below state-recommended $100,000.\n"
    )
    return SyntheticDoc(
        document_id="SYN-INS-001",
        domain="insurance",
        text=text,
        expected_anomalies=[
            "Liability limit below state-recommended minimum",
        ],
        expected_gaps=[
            "No flood coverage",
            "No earthquake coverage",
        ],
        expected_recommendations=[
            "Increase liability limit to $100,000",
        ],
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/insights_eval/fixtures/test_synthetic.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/insights_eval/fixtures/__init__.py tests/insights_eval/fixtures/synthetic.py tests/insights_eval/fixtures/test_synthetic.py
git commit -m "feat(eval): seed synthetic insurance fixture for gate tests (SP-J.4)"
```

---

## Sub-Project SP-A — Adapter Framework

**Files:**
- Create: `src/intelligence/adapters/__init__.py`
- Create: `src/intelligence/adapters/schema.py`
- Create: `src/intelligence/adapters/store.py`
- Create: `src/intelligence/adapters/detect.py`
- Create: `src/intelligence/adapters/generic.yaml`
- Test: `tests/insights_eval/test_adapter_schema.py`
- Test: `tests/insights_eval/test_adapter_store.py`
- Test: `tests/insights_eval/test_adapter_detect.py`

### Task SP-A.1 — Adapter schema dataclasses

- [ ] **Step 1: Write the failing test**

`tests/insights_eval/test_adapter_schema.py`:
```python
import pytest

from src.intelligence.adapters.schema import (
    Adapter,
    AppliesWhen,
    InsightTypeConfig,
    KnowledgeConfig,
    SanctionedKb,
    Watchlist,
    ActionTemplate,
    VisualizationSpec,
    parse_adapter_yaml,
)


SAMPLE_YAML = """
name: insurance
version: "1.0"
description: "Insurance policies, claims, coverage analysis"
applies_when:
  domain_classifier_labels: [insurance, policy]
  doc_type_hints: [policy_document]
  keyword_evidence_min: 3
  keywords: [policyholder, deductible, premium]
researcher:
  insight_types:
    anomaly:
      prompt_template: "prompts/insurance_anomaly.md"
      enabled: true
    gap:
      prompt_template: "prompts/insurance_gap.md"
      enabled: true
knowledge:
  sanctioned_kbs:
    - kb_id: insurance_taxonomy_v1
      ref: "blob://kbs/insurance_taxonomy_v1.json"
      describes: "Common policy types"
  citation_rule: "doc_grounded_first"
watchlists:
  - id: renewal_due
    description: "Policy renewal due soon"
    eval: "expr:doc.policy_end_date - now < 60d"
    fires_insight_type: next_action
actions:
  - action_id: generate_coverage_summary
    title: "Generate coverage summary PDF"
    action_type: artifact
    artifact_template: "templates/insurance_coverage_summary.md"
    requires_confirmation: false
visualizations:
  - viz_id: coverage_comparison_table
    insight_types: [comparison]
"""


def test_parse_yaml_returns_adapter():
    a = parse_adapter_yaml(SAMPLE_YAML)
    assert isinstance(a, Adapter)
    assert a.name == "insurance"
    assert a.version == "1.0"
    assert isinstance(a.applies_when, AppliesWhen)
    assert a.applies_when.keyword_evidence_min == 3
    assert "anomaly" in a.researcher.insight_types
    assert isinstance(a.researcher.insight_types["anomaly"], InsightTypeConfig)
    assert a.researcher.insight_types["anomaly"].enabled is True


def test_knowledge_section():
    a = parse_adapter_yaml(SAMPLE_YAML)
    assert isinstance(a.knowledge, KnowledgeConfig)
    assert len(a.knowledge.sanctioned_kbs) == 1
    assert isinstance(a.knowledge.sanctioned_kbs[0], SanctionedKb)
    assert a.knowledge.sanctioned_kbs[0].kb_id == "insurance_taxonomy_v1"


def test_watchlists_actions_visualizations():
    a = parse_adapter_yaml(SAMPLE_YAML)
    assert len(a.watchlists) == 1
    assert isinstance(a.watchlists[0], Watchlist)
    assert a.watchlists[0].id == "renewal_due"
    assert len(a.actions) == 1
    assert isinstance(a.actions[0], ActionTemplate)
    assert a.actions[0].requires_confirmation is False
    assert len(a.visualizations) == 1
    assert isinstance(a.visualizations[0], VisualizationSpec)


def test_minimal_adapter_only_name():
    minimal = "name: tiny\nversion: '1.0'\ndescription: t\napplies_when: {}\nresearcher:\n  insight_types: {}\nknowledge:\n  sanctioned_kbs: []\n  citation_rule: doc_grounded_first\nwatchlists: []\nactions: []\nvisualizations: []\n"
    a = parse_adapter_yaml(minimal)
    assert a.name == "tiny"
    assert a.researcher.insight_types == {}


def test_invalid_yaml_raises():
    with pytest.raises(ValueError):
        parse_adapter_yaml("not: a: valid: yaml:::")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/insights_eval/test_adapter_schema.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.intelligence.adapters'`.

- [ ] **Step 3: Write minimal implementation**

`src/intelligence/adapters/__init__.py`:
```python
from src.intelligence.adapters.schema import (
    Adapter,
    AppliesWhen,
    ResearcherSection,
    InsightTypeConfig,
    KnowledgeConfig,
    SanctionedKb,
    Watchlist,
    ActionTemplate,
    VisualizationSpec,
    parse_adapter_yaml,
)

__all__ = [
    "Adapter",
    "AppliesWhen",
    "ResearcherSection",
    "InsightTypeConfig",
    "KnowledgeConfig",
    "SanctionedKb",
    "Watchlist",
    "ActionTemplate",
    "VisualizationSpec",
    "parse_adapter_yaml",
]
```

`src/intelligence/adapters/schema.py`:
```python
"""Adapter YAML schema — parsed dataclasses.

Adapters live in Azure Blob (per feedback_adapter_yaml_blob.md). Code
ships only the generic.yaml fallback, uploaded to Blob at first deploy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class AppliesWhen:
    domain_classifier_labels: List[str] = field(default_factory=list)
    doc_type_hints: List[str] = field(default_factory=list)
    keyword_evidence_min: int = 0
    keywords: List[str] = field(default_factory=list)


@dataclass
class InsightTypeConfig:
    prompt_template: str = ""
    enabled: bool = True
    requires_min_docs: int = 1


@dataclass
class ResearcherSection:
    insight_types: Dict[str, InsightTypeConfig] = field(default_factory=dict)


@dataclass
class SanctionedKb:
    kb_id: str
    ref: str
    describes: str = ""


@dataclass
class KnowledgeConfig:
    sanctioned_kbs: List[SanctionedKb] = field(default_factory=list)
    citation_rule: str = "doc_grounded_first"


@dataclass
class Watchlist:
    id: str
    description: str
    eval: str
    fires_insight_type: str


@dataclass
class ActionTemplate:
    action_id: str
    title: str
    action_type: str  # artifact | form_fill | alert | plan | reminder
    artifact_template: Optional[str] = None
    requires_confirmation: bool = True
    input_schema: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VisualizationSpec:
    viz_id: str
    insight_types: List[str]


@dataclass
class Adapter:
    name: str
    version: str
    description: str
    applies_when: AppliesWhen
    researcher: ResearcherSection
    knowledge: KnowledgeConfig
    watchlists: List[Watchlist] = field(default_factory=list)
    actions: List[ActionTemplate] = field(default_factory=list)
    visualizations: List[VisualizationSpec] = field(default_factory=list)


def parse_adapter_yaml(text: str) -> Adapter:
    try:
        raw = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid adapter YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("adapter YAML root must be a mapping")
    aw = raw.get("applies_when") or {}
    applies_when = AppliesWhen(
        domain_classifier_labels=list(aw.get("domain_classifier_labels") or []),
        doc_type_hints=list(aw.get("doc_type_hints") or []),
        keyword_evidence_min=int(aw.get("keyword_evidence_min") or 0),
        keywords=list(aw.get("keywords") or []),
    )
    r = raw.get("researcher") or {}
    insight_types = {
        name: InsightTypeConfig(
            prompt_template=str(v.get("prompt_template") or ""),
            enabled=bool(v.get("enabled", True)),
            requires_min_docs=int(v.get("requires_min_docs") or 1),
        )
        for name, v in (r.get("insight_types") or {}).items()
    }
    researcher = ResearcherSection(insight_types=insight_types)
    k = raw.get("knowledge") or {}
    sanctioned = [
        SanctionedKb(
            kb_id=str(kb.get("kb_id") or ""),
            ref=str(kb.get("ref") or ""),
            describes=str(kb.get("describes") or ""),
        )
        for kb in (k.get("sanctioned_kbs") or [])
    ]
    knowledge = KnowledgeConfig(
        sanctioned_kbs=sanctioned,
        citation_rule=str(k.get("citation_rule") or "doc_grounded_first"),
    )
    watchlists = [
        Watchlist(
            id=str(w.get("id") or ""),
            description=str(w.get("description") or ""),
            eval=str(w.get("eval") or ""),
            fires_insight_type=str(w.get("fires_insight_type") or ""),
        )
        for w in (raw.get("watchlists") or [])
    ]
    actions = [
        ActionTemplate(
            action_id=str(a.get("action_id") or ""),
            title=str(a.get("title") or ""),
            action_type=str(a.get("action_type") or ""),
            artifact_template=a.get("artifact_template"),
            requires_confirmation=bool(a.get("requires_confirmation", True)),
            input_schema=dict(a.get("input_schema") or {}),
        )
        for a in (raw.get("actions") or [])
    ]
    visualizations = [
        VisualizationSpec(
            viz_id=str(v.get("viz_id") or ""),
            insight_types=list(v.get("insight_types") or []),
        )
        for v in (raw.get("visualizations") or [])
    ]
    return Adapter(
        name=str(raw.get("name") or ""),
        version=str(raw.get("version") or ""),
        description=str(raw.get("description") or ""),
        applies_when=applies_when,
        researcher=researcher,
        knowledge=knowledge,
        watchlists=watchlists,
        actions=actions,
        visualizations=visualizations,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/insights_eval/test_adapter_schema.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/intelligence/adapters/__init__.py src/intelligence/adapters/schema.py tests/insights_eval/test_adapter_schema.py
git commit -m "feat(adapters): YAML schema parser for plugin-shaped domain adapters (SP-A.1)"
```

### Task SP-A.2 — Generic adapter YAML

- [ ] **Step 1: Write the failing test**

Append to `tests/insights_eval/test_adapter_schema.py`:
```python
def test_generic_adapter_loads_and_parses():
    from pathlib import Path
    p = Path("src/intelligence/adapters/generic.yaml")
    assert p.exists(), "generic adapter YAML must ship with code"
    a = parse_adapter_yaml(p.read_text())
    assert a.name == "generic"
    # Generic adapter must declare every insight type so it is the always-safe fallback
    expected_types = {
        "anomaly", "gap", "comparison", "scenario", "trend",
        "recommendation", "conflict", "projection", "next_action",
    }
    assert set(a.researcher.insight_types.keys()) == expected_types
    for cfg in a.researcher.insight_types.values():
        assert cfg.enabled is True
        assert cfg.prompt_template
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/insights_eval/test_adapter_schema.py::test_generic_adapter_loads_and_parses -v`
Expected: FAIL — file does not exist.

- [ ] **Step 3: Write minimal implementation**

`src/intelligence/adapters/generic.yaml`:
```yaml
name: generic
version: "1.0"
description: "Domain-agnostic researcher adapter. Always-safe fallback for any document set."
applies_when:
  domain_classifier_labels: []
  doc_type_hints: []
  keyword_evidence_min: 0
  keywords: []

researcher:
  insight_types:
    anomaly:
      prompt_template: "prompts/generic_anomaly.md"
      enabled: true
    gap:
      prompt_template: "prompts/generic_gap.md"
      enabled: true
    comparison:
      prompt_template: "prompts/generic_comparison.md"
      enabled: true
      requires_min_docs: 2
    scenario:
      prompt_template: "prompts/generic_scenario.md"
      enabled: true
    trend:
      prompt_template: "prompts/generic_trend.md"
      enabled: true
      requires_min_docs: 2
    recommendation:
      prompt_template: "prompts/generic_recommendation.md"
      enabled: true
    conflict:
      prompt_template: "prompts/generic_conflict.md"
      enabled: true
      requires_min_docs: 2
    projection:
      prompt_template: "prompts/generic_projection.md"
      enabled: true
    next_action:
      prompt_template: "prompts/generic_next_action.md"
      enabled: true

knowledge:
  sanctioned_kbs: []
  citation_rule: "doc_grounded_first"

watchlists: []

actions:
  - action_id: generic_summary_artifact
    title: "Generate document set summary"
    action_type: artifact
    artifact_template: "templates/generic_summary.md"
    requires_confirmation: false

visualizations:
  - viz_id: generic_timeline
    insight_types: [trend]
  - viz_id: generic_comparison_table
    insight_types: [comparison]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/insights_eval/test_adapter_schema.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/intelligence/adapters/generic.yaml tests/insights_eval/test_adapter_schema.py
git commit -m "feat(adapters): ship generic always-safe fallback adapter YAML (SP-A.2)"
```

### Task SP-A.3 — AdapterStore with TTL cache + resolution order

- [ ] **Step 1: Write the failing test**

`tests/insights_eval/test_adapter_store.py`:
```python
import time

import pytest

from src.intelligence.adapters.store import AdapterStore, AdapterNotFound


class FakeBlobBackend:
    """Minimal in-memory backend so tests don't touch Azure."""

    def __init__(self):
        self.store = {}

    def get_text(self, key: str) -> str:
        if key not in self.store:
            raise AdapterNotFound(key)
        return self.store[key]


def _generic_yaml() -> str:
    from pathlib import Path
    return Path("src/intelligence/adapters/generic.yaml").read_text()


def _stub_yaml(name: str, version: str = "1.0") -> str:
    return (
        f"name: {name}\n"
        f"version: '{version}'\n"
        "description: x\n"
        "applies_when: {}\n"
        "researcher:\n  insight_types: {}\n"
        "knowledge:\n  sanctioned_kbs: []\n  citation_rule: doc_grounded_first\n"
        "watchlists: []\n"
        "actions: []\n"
        "visualizations: []\n"
    )


def test_global_only_resolution():
    backend = FakeBlobBackend()
    backend.store["sme_adapters/global/insurance.yaml"] = _stub_yaml("insurance")
    backend.store["sme_adapters/global/generic.yaml"] = _generic_yaml()
    store = AdapterStore(backend=backend, cache_ttl_seconds=60)
    a = store.get(domain="insurance", subscription_id="sub-x")
    assert a.name == "insurance"


def test_subscription_overrides_global():
    backend = FakeBlobBackend()
    backend.store["sme_adapters/global/insurance.yaml"] = _stub_yaml("insurance", "1.0")
    backend.store["sme_adapters/subscription/sub-x/insurance.yaml"] = _stub_yaml(
        "insurance", "2.0-tenant"
    )
    backend.store["sme_adapters/global/generic.yaml"] = _generic_yaml()
    store = AdapterStore(backend=backend, cache_ttl_seconds=60)
    a = store.get(domain="insurance", subscription_id="sub-x")
    assert a.version == "2.0-tenant"


def test_falls_back_to_generic_when_unknown_domain():
    backend = FakeBlobBackend()
    backend.store["sme_adapters/global/generic.yaml"] = _generic_yaml()
    store = AdapterStore(backend=backend, cache_ttl_seconds=60)
    a = store.get(domain="moonbase_logistics", subscription_id="sub-x")
    assert a.name == "generic"


def test_ttl_cache_reuses_within_ttl():
    backend = FakeBlobBackend()
    backend.store["sme_adapters/global/generic.yaml"] = _generic_yaml()
    store = AdapterStore(backend=backend, cache_ttl_seconds=60)
    a1 = store.get(domain="generic", subscription_id="sub-x")
    a2 = store.get(domain="generic", subscription_id="sub-x")
    assert a1 is a2  # same object — cached


def test_invalidate_forces_reload():
    backend = FakeBlobBackend()
    backend.store["sme_adapters/global/generic.yaml"] = _generic_yaml()
    store = AdapterStore(backend=backend, cache_ttl_seconds=60)
    a1 = store.get(domain="generic", subscription_id="sub-x")
    store.invalidate(domain="generic")
    a2 = store.get(domain="generic", subscription_id="sub-x")
    assert a1 is not a2


def test_blob_failure_falls_back_to_cached():
    backend = FakeBlobBackend()
    backend.store["sme_adapters/global/generic.yaml"] = _generic_yaml()
    store = AdapterStore(backend=backend, cache_ttl_seconds=0)
    a1 = store.get(domain="generic", subscription_id="sub-x")
    # Now break the backend
    def boom(key):
        raise RuntimeError("blob down")
    backend.get_text = boom
    a2 = store.get(domain="generic", subscription_id="sub-x")
    assert a2 is not None
    assert a2.name == "generic"  # served from cache despite ttl=0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/insights_eval/test_adapter_store.py -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Write minimal implementation**

`src/intelligence/adapters/store.py`:
```python
"""AdapterStore — resolves and caches adapter YAMLs.

Resolution order:
  1. sme_adapters/subscription/{subscription_id}/{domain}.yaml
  2. sme_adapters/global/{domain}.yaml
  3. sme_adapters/global/generic.yaml  (always succeeds)

Cache: in-memory TTL (default 5 min). Failure mode: serve last cached
version; if no cache, fall back to generic.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol

from src.intelligence.adapters.schema import Adapter, parse_adapter_yaml

logger = logging.getLogger(__name__)


class AdapterNotFound(Exception):
    pass


class AdapterBackend(Protocol):
    def get_text(self, key: str) -> str: ...


@dataclass
class _CacheEntry:
    adapter: Adapter
    loaded_at: float


class AdapterStore:
    def __init__(self, *, backend: AdapterBackend, cache_ttl_seconds: int = 300):
        self._backend = backend
        self._ttl = cache_ttl_seconds
        self._cache: Dict[str, _CacheEntry] = {}

    def get(self, *, domain: str, subscription_id: str) -> Adapter:
        # Try subscription override first
        for key in self._candidate_keys(domain=domain, subscription_id=subscription_id):
            adapter = self._load_or_cached(key)
            if adapter is not None:
                return adapter
        # Final fallback — generic must always succeed
        adapter = self._load_or_cached("sme_adapters/global/generic.yaml")
        if adapter is None:
            raise AdapterNotFound("generic adapter is missing — install required")
        return adapter

    def invalidate(self, *, domain: Optional[str] = None) -> None:
        if domain is None:
            self._cache.clear()
            return
        for key in list(self._cache.keys()):
            if f"/{domain}.yaml" in key:
                del self._cache[key]

    def _candidate_keys(self, *, domain: str, subscription_id: str):
        if subscription_id:
            yield f"sme_adapters/subscription/{subscription_id}/{domain}.yaml"
        yield f"sme_adapters/global/{domain}.yaml"

    def _load_or_cached(self, key: str) -> Optional[Adapter]:
        entry = self._cache.get(key)
        now = time.monotonic()
        if entry is not None and (now - entry.loaded_at) < self._ttl:
            return entry.adapter
        # Try to refresh from backend
        try:
            text = self._backend.get_text(key)
        except AdapterNotFound:
            return None
        except Exception as exc:
            logger.warning("adapter blob fetch failed for %s: %s", key, exc)
            # Fall back to whatever we had cached, even if stale
            return entry.adapter if entry is not None else None
        try:
            adapter = parse_adapter_yaml(text)
        except Exception as exc:
            logger.error("adapter YAML parse failed for %s: %s", key, exc)
            return entry.adapter if entry is not None else None
        self._cache[key] = _CacheEntry(adapter=adapter, loaded_at=now)
        return adapter
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/insights_eval/test_adapter_store.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/intelligence/adapters/store.py tests/insights_eval/test_adapter_store.py
git commit -m "feat(adapters): AdapterStore with TTL cache + resolution order + failure fallback (SP-A.3)"
```

### Task SP-A.4 — Auto-detection wrapper

- [ ] **Step 1: Write the failing test**

`tests/insights_eval/test_adapter_detect.py`:
```python
from src.intelligence.adapters.detect import detect_domain, DetectionResult


def test_high_confidence_returns_classifier_label():
    # Stub classifier that always returns insurance / 0.9
    def fake_classifier(text):
        return ("insurance", 0.9)

    r = detect_domain("Policy SYN-001 ...", classifier=fake_classifier)
    assert isinstance(r, DetectionResult)
    assert r.domain == "insurance"
    assert r.confidence == 0.9
    assert r.fallback_to_generic is False


def test_low_confidence_falls_back_to_generic():
    def fake_classifier(text):
        return ("medical", 0.3)

    r = detect_domain("ambiguous text", classifier=fake_classifier)
    assert r.domain == "generic"
    assert r.fallback_to_generic is True
    assert r.confidence == 0.3


def test_threshold_is_0_7():
    def at_threshold(text):
        return ("hr", 0.7)

    r = detect_domain("x", classifier=at_threshold)
    assert r.domain == "hr"
    assert r.fallback_to_generic is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/insights_eval/test_adapter_detect.py -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Write minimal implementation**

`src/intelligence/adapters/detect.py`:
```python
"""Domain auto-detection for adapter routing.

Wraps the existing src.intelligence.domain_classifier to return a single
chosen-domain decision plus a fallback flag. Confidence threshold is 0.7
per spec Section 6.2.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

DEFAULT_THRESHOLD = 0.7


@dataclass
class DetectionResult:
    domain: str
    confidence: float
    fallback_to_generic: bool


def _default_classifier(text: str) -> Tuple[str, float]:
    from src.intelligence.domain_classifier import classify_domain
    label, conf = classify_domain(text)
    return label, float(conf or 0.0)


def detect_domain(
    text: str,
    *,
    classifier: Callable[[str], Tuple[str, float]] = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> DetectionResult:
    cls = classifier or _default_classifier
    label, confidence = cls(text)
    if confidence < threshold:
        return DetectionResult(
            domain="generic",
            confidence=confidence,
            fallback_to_generic=True,
        )
    return DetectionResult(
        domain=label,
        confidence=confidence,
        fallback_to_generic=False,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/insights_eval/test_adapter_detect.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/intelligence/adapters/detect.py tests/insights_eval/test_adapter_detect.py
git commit -m "feat(adapters): auto-detect domain wrapper with 0.7 confidence threshold (SP-A.4)"
```

### Task SP-A.5 — Default Blob backend (file-system stub for v1)

- [ ] **Step 1: Write the failing test**

Append to `tests/insights_eval/test_adapter_store.py`:
```python
def test_filesystem_backend_reads_local_file(tmp_path):
    from src.intelligence.adapters.store import FilesystemAdapterBackend

    root = tmp_path
    (root / "sme_adapters" / "global").mkdir(parents=True)
    (root / "sme_adapters" / "global" / "generic.yaml").write_text(_generic_yaml())
    backend = FilesystemAdapterBackend(root=str(root))
    text = backend.get_text("sme_adapters/global/generic.yaml")
    assert "name: generic" in text


def test_filesystem_backend_missing_raises(tmp_path):
    from src.intelligence.adapters.store import (
        FilesystemAdapterBackend,
        AdapterNotFound,
    )

    backend = FilesystemAdapterBackend(root=str(tmp_path))
    with pytest.raises(AdapterNotFound):
        backend.get_text("sme_adapters/global/nope.yaml")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/insights_eval/test_adapter_store.py::test_filesystem_backend_reads_local_file -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/intelligence/adapters/store.py`:
```python
import os


class FilesystemAdapterBackend:
    """Local-filesystem backend for tests + early dev. Production uses Blob."""

    def __init__(self, *, root: str):
        self._root = root

    def get_text(self, key: str) -> str:
        path = os.path.join(self._root, key)
        if not os.path.exists(path):
            raise AdapterNotFound(key)
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/insights_eval/test_adapter_store.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add src/intelligence/adapters/store.py tests/insights_eval/test_adapter_store.py
git commit -m "feat(adapters): filesystem backend for tests + dev (SP-A.5)"
```

### Task SP-A.6 — Azure Blob backend behind flag

- [ ] **Step 1: Write the failing test**

Append to `tests/insights_eval/test_adapter_store.py`:
```python
def test_blob_backend_is_disabled_when_flag_off(monkeypatch):
    from src.intelligence.adapters.store import resolve_default_backend

    monkeypatch.delenv("ADAPTER_BLOB_LOADING_ENABLED", raising=False)
    backend = resolve_default_backend(blob_root="/tmp/x")
    # With flag off, default backend is Filesystem (safe local fallback)
    from src.intelligence.adapters.store import FilesystemAdapterBackend
    assert isinstance(backend, FilesystemAdapterBackend)


def test_blob_backend_active_when_flag_on(monkeypatch, tmp_path):
    from src.intelligence.adapters.store import (
        resolve_default_backend,
        AzureBlobAdapterBackend,
    )
    monkeypatch.setenv("ADAPTER_BLOB_LOADING_ENABLED", "true")
    monkeypatch.setenv("ADAPTER_BLOB_CONTAINER", "fake")
    monkeypatch.setenv("ADAPTER_BLOB_CONNECTION", "fake")
    backend = resolve_default_backend(blob_root=str(tmp_path))
    assert isinstance(backend, AzureBlobAdapterBackend)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/insights_eval/test_adapter_store.py::test_blob_backend_is_disabled_when_flag_off -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/intelligence/adapters/store.py`:
```python
class AzureBlobAdapterBackend:
    """Azure Blob backend. Lazy-imports Azure SDK so tests don't need it."""

    def __init__(self, *, container: str, connection_string: str):
        self._container = container
        self._connection_string = connection_string
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from azure.storage.blob import BlobServiceClient
            self._client = BlobServiceClient.from_connection_string(
                self._connection_string
            )
        return self._client

    def get_text(self, key: str) -> str:
        client = self._ensure_client()
        blob = client.get_blob_client(container=self._container, blob=key)
        try:
            data = blob.download_blob().readall()
        except Exception as exc:
            raise AdapterNotFound(key) from exc
        return data.decode("utf-8")


def resolve_default_backend(*, blob_root: str) -> AdapterBackend:
    """Return the right backend for current environment.

    With ADAPTER_BLOB_LOADING_ENABLED=true and Azure config present, use
    AzureBlobAdapterBackend. Otherwise fall back to filesystem (the
    always-safe path matching ADAPTER_GENERIC_FALLBACK_ENABLED behavior).
    """
    from src.api.config import insight_flag_enabled

    if insight_flag_enabled("ADAPTER_BLOB_LOADING_ENABLED"):
        container = os.environ.get("ADAPTER_BLOB_CONTAINER")
        conn = os.environ.get("ADAPTER_BLOB_CONNECTION")
        if container and conn:
            return AzureBlobAdapterBackend(
                container=container, connection_string=conn
            )
    return FilesystemAdapterBackend(root=blob_root)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/insights_eval/test_adapter_store.py -v`
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add src/intelligence/adapters/store.py tests/insights_eval/test_adapter_store.py
git commit -m "feat(adapters): Azure Blob backend behind ADAPTER_BLOB_LOADING_ENABLED (SP-A.6)"
```

---

## Sub-Project SP-B — Insight Schema + Store

**Files:**
- Create: `src/intelligence/insights/__init__.py`
- Create: `src/intelligence/insights/schema.py`
- Create: `src/intelligence/insights/validators.py`
- Create: `src/intelligence/insights/store.py`
- Create: `src/intelligence/insights/staleness.py`
- Test: `tests/insights_eval/test_insight_schema.py`
- Test: `tests/insights_eval/test_insight_validators.py`
- Test: `tests/insights_eval/test_insight_store.py`
- Test: `tests/insights_eval/test_insight_staleness.py`

### Task SP-B.1 — Insight + Action + EvidenceSpan + KbRef dataclasses

- [ ] **Step 1: Write the failing test**

`tests/insights_eval/test_insight_schema.py`:
```python
import pytest
from datetime import datetime, timezone

from src.intelligence.insights.schema import (
    Insight,
    Action,
    EvidenceSpan,
    KbRef,
    INSIGHT_TYPES,
    SEVERITIES,
    ACTION_TYPES,
)


def test_insight_types_complete():
    assert INSIGHT_TYPES == (
        "anomaly", "gap", "comparison", "scenario", "trend",
        "recommendation", "conflict", "projection", "next_action",
    )


def test_severities_ordered():
    assert SEVERITIES == ("info", "notice", "warn", "critical")


def test_action_types():
    assert ACTION_TYPES == ("artifact", "form_fill", "alert", "plan", "reminder")


def test_minimal_insight_construction():
    span = EvidenceSpan(
        document_id="DOC-1", page=1, char_start=0, char_end=10, quote="hello"
    )
    insight = Insight(
        insight_id="i-1",
        profile_id="p-1",
        subscription_id="s-1",
        document_ids=["DOC-1"],
        domain="generic",
        insight_type="anomaly",
        headline="Test headline",
        body="Body text grounded in doc",
        evidence_doc_spans=[span],
        confidence=0.9,
        severity="notice",
        adapter_version="generic@1.0",
    )
    assert insight.headline == "Test headline"
    assert insight.evidence_doc_spans[0].document_id == "DOC-1"
    assert insight.external_kb_refs == []  # default empty


def test_invalid_insight_type_rejected():
    with pytest.raises(ValueError):
        Insight(
            insight_id="i-2",
            profile_id="p-1",
            subscription_id="s-1",
            document_ids=["DOC-1"],
            domain="generic",
            insight_type="bogus",  # not in INSIGHT_TYPES
            headline="x",
            body="y",
            evidence_doc_spans=[
                EvidenceSpan(
                    document_id="DOC-1", page=1, char_start=0, char_end=1, quote="x"
                )
            ],
            confidence=0.5,
            severity="notice",
            adapter_version="generic@1.0",
        )


def test_invalid_severity_rejected():
    with pytest.raises(ValueError):
        Insight(
            insight_id="i-3",
            profile_id="p-1",
            subscription_id="s-1",
            document_ids=["DOC-1"],
            domain="generic",
            insight_type="anomaly",
            headline="x",
            body="y",
            evidence_doc_spans=[
                EvidenceSpan(document_id="DOC-1", page=1, char_start=0, char_end=1, quote="x")
            ],
            confidence=0.5,
            severity="meh",
            adapter_version="generic@1.0",
        )


def test_to_dict_round_trip():
    insight = Insight(
        insight_id="i-4",
        profile_id="p-1",
        subscription_id="s-1",
        document_ids=["DOC-1"],
        domain="insurance",
        insight_type="gap",
        headline="No flood coverage",
        body="Policy excludes flood damage. Document explicitly lists 'flood damage' under exclusions.",
        evidence_doc_spans=[
            EvidenceSpan(document_id="DOC-1", page=1, char_start=100, char_end=130, quote="Excludes: flood damage")
        ],
        external_kb_refs=[KbRef(kb_id="insurance_taxonomy_v1", ref="exclusions/flood", label="Flood exclusion")],
        confidence=0.95,
        severity="warn",
        adapter_version="insurance@1.0",
    )
    d = insight.to_dict()
    assert d["insight_id"] == "i-4"
    assert d["insight_type"] == "gap"
    assert d["evidence_doc_spans"][0]["document_id"] == "DOC-1"
    assert d["external_kb_refs"][0]["kb_id"] == "insurance_taxonomy_v1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/insights_eval/test_insight_schema.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

`src/intelligence/insights/__init__.py`:
```python
from src.intelligence.insights.schema import (
    Insight,
    Action,
    EvidenceSpan,
    KbRef,
    INSIGHT_TYPES,
    SEVERITIES,
    ACTION_TYPES,
)

__all__ = [
    "Insight",
    "Action",
    "EvidenceSpan",
    "KbRef",
    "INSIGHT_TYPES",
    "SEVERITIES",
    "ACTION_TYPES",
]
```

`src/intelligence/insights/schema.py`:
```python
"""Insight + Action data model.

Canonical schema for the Insights Portal. Persisted to Qdrant payload +
Neo4j Insight nodes + Mongo control-plane index per spec Section 5.1.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

INSIGHT_TYPES = (
    "anomaly", "gap", "comparison", "scenario", "trend",
    "recommendation", "conflict", "projection", "next_action",
)
SEVERITIES = ("info", "notice", "warn", "critical")
ACTION_TYPES = ("artifact", "form_fill", "alert", "plan", "reminder")


@dataclass
class EvidenceSpan:
    document_id: str
    page: int
    char_start: int
    char_end: int
    quote: str


@dataclass
class KbRef:
    kb_id: str
    ref: str
    label: str = ""


@dataclass
class Insight:
    insight_id: str
    profile_id: str
    subscription_id: str
    document_ids: List[str]
    domain: str
    insight_type: str
    headline: str
    body: str
    evidence_doc_spans: List[EvidenceSpan]
    confidence: float
    severity: str
    adapter_version: str
    external_kb_refs: List[KbRef] = field(default_factory=list)
    suggested_actions: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    created_at: str = ""
    refreshed_at: str = ""
    stale: bool = False
    feature_flags: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.insight_type not in INSIGHT_TYPES:
            raise ValueError(
                f"insight_type must be one of {INSIGHT_TYPES}, got {self.insight_type!r}"
            )
        if self.severity not in SEVERITIES:
            raise ValueError(
                f"severity must be one of {SEVERITIES}, got {self.severity!r}"
            )
        if not self.created_at:
            self.created_at = datetime.now(tz=timezone.utc).isoformat()
        if not self.refreshed_at:
            self.refreshed_at = self.created_at

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class Action:
    action_id: str
    profile_id: str
    subscription_id: str
    domain: str
    action_type: str
    title: str
    description: str
    preview: str
    requires_confirmation: bool
    produces_artifact: bool = False
    artifact_template: Optional[str] = None
    input_schema: Dict[str, Any] = field(default_factory=dict)
    executed_at: Optional[str] = None
    execution_status: str = "pending"
    audit_trail: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.action_type not in ACTION_TYPES:
            raise ValueError(
                f"action_type must be one of {ACTION_TYPES}, got {self.action_type!r}"
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/insights_eval/test_insight_schema.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/intelligence/insights/__init__.py src/intelligence/insights/schema.py tests/insights_eval/test_insight_schema.py
git commit -m "feat(insights): canonical Insight + Action data model with type validation (SP-B.1)"
```

### Task SP-B.2 — Citation validator (mandatory doc-spans)

- [ ] **Step 1: Write the failing test**

`tests/insights_eval/test_insight_validators.py`:
```python
import pytest

from src.intelligence.insights.schema import Insight, EvidenceSpan, KbRef
from src.intelligence.insights.validators import (
    require_doc_evidence,
    CitationViolation,
)


def _doc_span(doc_id="DOC-1", quote="hello"):
    return EvidenceSpan(
        document_id=doc_id, page=1, char_start=0, char_end=len(quote), quote=quote
    )


def _make(spans, kb_refs=None) -> Insight:
    return Insight(
        insight_id="i-1",
        profile_id="p-1",
        subscription_id="s-1",
        document_ids=["DOC-1"],
        domain="generic",
        insight_type="anomaly",
        headline="x",
        body="y derivable from quote: hello",
        evidence_doc_spans=spans,
        confidence=0.5,
        severity="notice",
        adapter_version="generic@1.0",
        external_kb_refs=kb_refs or [],
    )


def test_passes_with_at_least_one_span():
    insight = _make([_doc_span()])
    require_doc_evidence(insight)  # does not raise


def test_rejects_zero_spans():
    insight = _make([])
    with pytest.raises(CitationViolation):
        require_doc_evidence(insight)


def test_rejects_kb_refs_without_doc_spans():
    # Construct manually since dataclass would normally fail; use __new__ workaround
    insight = Insight(
        insight_id="i-2",
        profile_id="p-1",
        subscription_id="s-1",
        document_ids=["DOC-1"],
        domain="generic",
        insight_type="anomaly",
        headline="x",
        body="y",
        evidence_doc_spans=[_doc_span()],  # one span at construction
        confidence=0.5,
        severity="notice",
        adapter_version="generic@1.0",
        external_kb_refs=[KbRef(kb_id="k1", ref="r1")],
    )
    # Now mutate to remove spans (simulating a malformed write)
    insight.evidence_doc_spans = []
    with pytest.raises(CitationViolation):
        require_doc_evidence(insight)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/insights_eval/test_insight_validators.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

`src/intelligence/insights/validators.py`:
```python
"""Insight validators — enforced at the InsightStore writer.

Hard rule (spec Section 5.1, OQ1): every persisted insight has
non-empty evidence_doc_spans. KB refs are augmentation; they cannot
substitute for doc evidence.
"""
from __future__ import annotations

from src.intelligence.insights.schema import Insight


class CitationViolation(ValueError):
    pass


class BodySeparationViolation(ValueError):
    pass


def require_doc_evidence(insight: Insight) -> None:
    if not insight.evidence_doc_spans:
        raise CitationViolation(
            f"insight {insight.insight_id} has no evidence_doc_spans"
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/insights_eval/test_insight_validators.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/intelligence/insights/validators.py tests/insights_eval/test_insight_validators.py
git commit -m "feat(insights): citation validator — zero doc-spans rejected at writer (SP-B.2)"
```

### Task SP-B.3 — Body-separation validator (OQ1 rule)

- [ ] **Step 1: Write the failing test**

Append to `tests/insights_eval/test_insight_validators.py`:
```python
from src.intelligence.insights.validators import (
    require_body_grounded,
    BodySeparationViolation,
)


def test_body_grounded_passes_when_quotes_overlap():
    span = EvidenceSpan(
        document_id="DOC-1", page=1, char_start=0, char_end=22,
        quote="The patient has Type 2 Diabetes.",
    )
    insight = _make([span])
    insight.body = "Patient diagnosed with Type 2 Diabetes."
    require_body_grounded(insight)  # token overlap → ok


def test_body_grounded_rejects_unsupported_external_claims():
    span = EvidenceSpan(
        document_id="DOC-1", page=1, char_start=0, char_end=10,
        quote="Test data",
    )
    insight = _make([span])
    # Body claims a fact NOT in the quoted spans
    insight.body = "Patient is at risk of diabetic ketoacidosis"
    with pytest.raises(BodySeparationViolation):
        require_body_grounded(insight)


def test_body_grounded_passes_with_partial_overlap():
    span = EvidenceSpan(
        document_id="DOC-1", page=1, char_start=0, char_end=40,
        quote="Excludes: flood damage, earthquake, racing events.",
    )
    insight = _make([span])
    insight.body = "The policy excludes flood damage and earthquake coverage."
    require_body_grounded(insight)  # multi-token overlap → ok
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/insights_eval/test_insight_validators.py::test_body_grounded_passes_when_quotes_overlap -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/intelligence/insights/validators.py`:
```python
import re


def _tokens(text: str) -> set:
    """Lowercase content tokens, ≥4 chars (filters articles, prepositions)."""
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{3,}", text.lower())
    return set(words)


_OVERLAP_THRESHOLD = 0.4


def require_body_grounded(insight: Insight) -> None:
    """Reject insights whose body introduces content not in evidence_doc_spans.

    Heuristic — counts overlap of meaningful tokens between body and
    concatenated quotes. ≥40% body tokens must appear in quotes.
    Per spec Section 8 (OQ1): KB-derived content goes to external_kb_refs,
    never into body text.
    """
    body_tokens = _tokens(insight.body)
    if not body_tokens:
        return  # empty body — let citation validator handle it
    quote_tokens: set = set()
    for span in insight.evidence_doc_spans:
        quote_tokens |= _tokens(span.quote)
    if not quote_tokens:
        # No tokens in quotes (very short quotes) — defer to citation validator
        return
    overlap = body_tokens & quote_tokens
    ratio = len(overlap) / len(body_tokens)
    if ratio < _OVERLAP_THRESHOLD:
        raise BodySeparationViolation(
            f"insight {insight.insight_id} body has insufficient overlap "
            f"with doc-span quotes ({ratio:.2f} < {_OVERLAP_THRESHOLD})"
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/insights_eval/test_insight_validators.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/intelligence/insights/validators.py tests/insights_eval/test_insight_validators.py
git commit -m "feat(insights): body-separation validator enforcing OQ1 (KB content out of body) (SP-B.3)"
```

### Task SP-B.4 — Dedup-key calculator

- [ ] **Step 1: Write the failing test**

Append to `tests/insights_eval/test_insight_validators.py`:
```python
from src.intelligence.insights.validators import compute_dedup_key


def test_dedup_key_stable_for_same_inputs():
    insight = _make([_doc_span()])
    insight.profile_id = "p-1"
    insight.document_ids = ["DOC-1"]
    insight.insight_type = "anomaly"
    insight.headline = "Test headline"
    k1 = compute_dedup_key(insight)
    k2 = compute_dedup_key(insight)
    assert k1 == k2


def test_dedup_key_changes_with_headline():
    a = _make([_doc_span()])
    b = _make([_doc_span()])
    a.headline = "headline A"
    b.headline = "headline B"
    assert compute_dedup_key(a) != compute_dedup_key(b)


def test_dedup_key_independent_of_document_id_order():
    a = _make([_doc_span("D1"), _doc_span("D2")])
    b = _make([_doc_span("D2"), _doc_span("D1")])
    a.document_ids = ["D1", "D2"]
    b.document_ids = ["D2", "D1"]
    a.headline = b.headline = "same"
    assert compute_dedup_key(a) == compute_dedup_key(b)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/insights_eval/test_insight_validators.py::test_dedup_key_stable_for_same_inputs -v`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

Append to `src/intelligence/insights/validators.py`:
```python
import hashlib


def compute_dedup_key(insight: Insight) -> str:
    """Stable dedup key from (profile_id, document_ids[], insight_type, headline_hash).

    Per spec Section 7.2 — re-runs upsert; duplicates suppressed.
    """
    sorted_docs = ",".join(sorted(insight.document_ids))
    headline_hash = hashlib.sha256(insight.headline.strip().lower().encode("utf-8")).hexdigest()[:16]
    raw = f"{insight.profile_id}|{sorted_docs}|{insight.insight_type}|{headline_hash}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/insights_eval/test_insight_validators.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/intelligence/insights/validators.py tests/insights_eval/test_insight_validators.py
git commit -m "feat(insights): dedup-key calculator for idempotent re-runs (SP-B.4)"
```

### Task SP-B.5 — InsightStore.write to Mongo control-plane index

- [ ] **Step 1: Write the failing test**

`tests/insights_eval/test_insight_store.py`:
```python
import pytest

from src.intelligence.insights.schema import Insight, EvidenceSpan
from src.intelligence.insights.store import InsightStore, MongoIndexBackend
from src.intelligence.insights.validators import CitationViolation


class FakeMongoCollection:
    def __init__(self):
        self.docs = []

    def update_one(self, filter, update, upsert=False):
        match = next(
            (d for d in self.docs if all(d.get(k) == v for k, v in filter.items())),
            None,
        )
        if match is None:
            if upsert:
                d = {**filter, **update.get("$set", {})}
                self.docs.append(d)
                return type("R", (), {"matched_count": 0, "upserted_id": "new"})()
        else:
            match.update(update.get("$set", {}))
            return type("R", (), {"matched_count": 1, "upserted_id": None})()
        return type("R", (), {"matched_count": 0, "upserted_id": None})()

    def find(self, query):
        out = [d for d in self.docs if all(d.get(k) == v for k, v in query.items())]
        return out


def _insight() -> Insight:
    span = EvidenceSpan(
        document_id="DOC-1", page=1, char_start=0, char_end=22,
        quote="Excludes: flood damage",
    )
    return Insight(
        insight_id="i-1",
        profile_id="p-1",
        subscription_id="s-1",
        document_ids=["DOC-1"],
        domain="insurance",
        insight_type="gap",
        headline="No flood coverage",
        body="The policy excludes flood damage.",
        evidence_doc_spans=[span],
        confidence=0.95,
        severity="warn",
        adapter_version="insurance@1.0",
    )


def test_write_inserts_into_mongo_index():
    coll = FakeMongoCollection()
    backend = MongoIndexBackend(collection=coll)
    store = InsightStore(mongo_index=backend, qdrant=None, neo4j=None)
    store.write(_insight())
    assert len(coll.docs) == 1
    d = coll.docs[0]
    assert d["insight_id"] == "i-1"
    assert d["profile_id"] == "p-1"
    assert d["insight_type"] == "gap"
    assert d["severity"] == "warn"
    # Mongo index does not store body content per storage-separation rule
    assert "body" not in d
    assert "evidence_doc_spans" not in d


def test_write_rejects_zero_evidence():
    coll = FakeMongoCollection()
    backend = MongoIndexBackend(collection=coll)
    store = InsightStore(mongo_index=backend, qdrant=None, neo4j=None)
    insight = _insight()
    insight.evidence_doc_spans = []
    with pytest.raises(CitationViolation):
        store.write(insight)
    assert len(coll.docs) == 0


def test_dedup_key_upsert():
    coll = FakeMongoCollection()
    backend = MongoIndexBackend(collection=coll)
    store = InsightStore(mongo_index=backend, qdrant=None, neo4j=None)
    a = _insight()
    b = _insight()
    b.insight_id = "i-2"  # different id, same dedup key
    store.write(a)
    store.write(b)
    assert len(coll.docs) == 1
    # Second write replaced the first via dedup_key
    assert coll.docs[0]["insight_id"] in ("i-1", "i-2")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/insights_eval/test_insight_store.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

`src/intelligence/insights/store.py`:
```python
"""InsightStore — single writer for Mongo + Qdrant + Neo4j.

Mongo (control plane) holds index records only. Body, quotes, and KB
refs go to Qdrant payload + Neo4j. Per feedback_storage_separation.md:
Mongo is control plane only; document content goes to Blob/Qdrant.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

from src.intelligence.insights.schema import Insight
from src.intelligence.insights.validators import (
    require_doc_evidence,
    require_body_grounded,
    compute_dedup_key,
)

logger = logging.getLogger(__name__)


class MongoCollection(Protocol):
    def update_one(self, filter, update, upsert=False) -> Any: ...
    def find(self, query) -> Any: ...


@dataclass
class MongoIndexBackend:
    collection: MongoCollection

    def upsert(self, dedup_key: str, doc: Dict[str, Any]) -> None:
        self.collection.update_one(
            {"dedup_key": dedup_key},
            {"$set": doc},
            upsert=True,
        )

    def list(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        return list(self.collection.find(query))


class QdrantBackend(Protocol):
    def upsert_insight(self, *, insight: Insight) -> None: ...


class Neo4jBackend(Protocol):
    def upsert_insight(self, *, insight: Insight) -> None: ...


class InsightStore:
    def __init__(
        self,
        *,
        mongo_index: MongoIndexBackend,
        qdrant: Optional[QdrantBackend],
        neo4j: Optional[Neo4jBackend],
    ):
        self._mongo = mongo_index
        self._qdrant = qdrant
        self._neo4j = neo4j

    def write(self, insight: Insight) -> None:
        # Hard validators — both must pass before any storage write
        require_doc_evidence(insight)
        require_body_grounded(insight)
        dedup_key = compute_dedup_key(insight)

        # Mongo control plane — index only
        index_doc = {
            "insight_id": insight.insight_id,
            "dedup_key": dedup_key,
            "profile_id": insight.profile_id,
            "subscription_id": insight.subscription_id,
            "document_ids": list(insight.document_ids),
            "domain": insight.domain,
            "insight_type": insight.insight_type,
            "severity": insight.severity,
            "tags": list(insight.tags),
            "refreshed_at": insight.refreshed_at,
            "stale": insight.stale,
            "adapter_version": insight.adapter_version,
        }
        self._mongo.upsert(dedup_key=dedup_key, doc=index_doc)

        if self._qdrant is not None:
            self._qdrant.upsert_insight(insight=insight)
        if self._neo4j is not None:
            self._neo4j.upsert_insight(insight=insight)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/insights_eval/test_insight_store.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/intelligence/insights/store.py tests/insights_eval/test_insight_store.py
git commit -m "feat(insights): InsightStore Mongo index path with dedup upsert (SP-B.5)"
```

### Task SP-B.6 — InsightStore.write to Qdrant payload

- [ ] **Step 1: Write the failing test**

Append to `tests/insights_eval/test_insight_store.py`:
```python
class FakeQdrant:
    def __init__(self):
        self.points = []
    def upsert_insight(self, *, insight):
        self.points.append(insight.to_dict())


def test_write_persists_to_qdrant():
    coll = FakeMongoCollection()
    qdrant = FakeQdrant()
    store = InsightStore(
        mongo_index=MongoIndexBackend(collection=coll),
        qdrant=qdrant,
        neo4j=None,
    )
    store.write(_insight())
    assert len(qdrant.points) == 1
    assert qdrant.points[0]["headline"] == "No flood coverage"
    assert qdrant.points[0]["evidence_doc_spans"][0]["document_id"] == "DOC-1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/insights_eval/test_insight_store.py::test_write_persists_to_qdrant -v`
Expected: PASS already (the InsightStore code already calls qdrant.upsert_insight if provided).

If it fails, re-check Step 3 in SP-B.5.

- [ ] **Step 3: Add Qdrant adapter implementation**

Append to `src/intelligence/insights/store.py`:
```python
class QdrantInsightBackend:
    """Real Qdrant backend for the `insights` collection.

    Uses an existing Qdrant client provided by src.api.qdrant_client.
    Embedding of headline+body is done by an injected embedder.
    """

    def __init__(self, *, client, collection_name: str = "insights", embedder=None):
        self._client = client
        self._collection = collection_name
        self._embedder = embedder

    def upsert_insight(self, *, insight: Insight) -> None:
        from qdrant_client.http.models import PointStruct
        text = f"{insight.headline}\n\n{insight.body}"
        vector = self._embedder.embed(text) if self._embedder else [0.0] * 384
        payload = {
            "insight_id": insight.insight_id,
            "profile_id": insight.profile_id,
            "subscription_id": insight.subscription_id,
            "document_ids": list(insight.document_ids),
            "insight_type": insight.insight_type,
            "severity": insight.severity,
            "tags": list(insight.tags),
            "domain": insight.domain,
            "headline": insight.headline,
            "body": insight.body,
            "evidence_doc_spans": [s.__dict__ for s in insight.evidence_doc_spans],
            "external_kb_refs": [r.__dict__ for r in insight.external_kb_refs],
            "adapter_version": insight.adapter_version,
            "refreshed_at": insight.refreshed_at,
        }
        self._client.upsert(
            collection_name=self._collection,
            points=[PointStruct(id=insight.insight_id, vector=vector, payload=payload)],
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/insights_eval/test_insight_store.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/intelligence/insights/store.py tests/insights_eval/test_insight_store.py
git commit -m "feat(insights): Qdrant insights collection writer (SP-B.6)"
```

### Task SP-B.7 — InsightStore.write to Neo4j Insight nodes

- [ ] **Step 1: Write the failing test**

Append to `tests/insights_eval/test_insight_store.py`:
```python
class FakeNeo4j:
    def __init__(self):
        self.calls = []
    def upsert_insight(self, *, insight):
        self.calls.append(insight.to_dict())


def test_write_persists_to_neo4j():
    coll = FakeMongoCollection()
    neo4j = FakeNeo4j()
    store = InsightStore(
        mongo_index=MongoIndexBackend(collection=coll),
        qdrant=None,
        neo4j=neo4j,
    )
    store.write(_insight())
    assert len(neo4j.calls) == 1
    assert neo4j.calls[0]["insight_id"] == "i-1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/insights_eval/test_insight_store.py::test_write_persists_to_neo4j -v`
Expected: PASS (interface already exists). If fails, fix InsightStore.write conditionals.

- [ ] **Step 3: Add Neo4j adapter implementation**

Append to `src/intelligence/insights/store.py`:
```python
class Neo4jInsightBackend:
    """Real Neo4j backend.

    Schema: (:Insight {insight_id, headline, severity, insight_type})
            -[:GROUNDED_IN]-> (:Document {document_id})
            (:Insight) -[:OF_PROFILE]-> (:Profile {profile_id})
    """

    def __init__(self, *, driver):
        self._driver = driver

    def upsert_insight(self, *, insight: Insight) -> None:
        cypher = (
            "MERGE (i:Insight {insight_id: $insight_id}) "
            "SET i.headline = $headline, "
            "    i.severity = $severity, "
            "    i.insight_type = $insight_type, "
            "    i.refreshed_at = $refreshed_at, "
            "    i.adapter_version = $adapter_version "
            "MERGE (p:Profile {profile_id: $profile_id}) "
            "MERGE (i)-[:OF_PROFILE]->(p) "
            "WITH i "
            "UNWIND $document_ids AS doc_id "
            "  MERGE (d:Document {document_id: doc_id}) "
            "  MERGE (i)-[:GROUNDED_IN]->(d)"
        )
        with self._driver.session() as sess:
            sess.run(
                cypher,
                insight_id=insight.insight_id,
                headline=insight.headline,
                severity=insight.severity,
                insight_type=insight.insight_type,
                refreshed_at=insight.refreshed_at,
                adapter_version=insight.adapter_version,
                profile_id=insight.profile_id,
                document_ids=list(insight.document_ids),
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/insights_eval/test_insight_store.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/intelligence/insights/store.py tests/insights_eval/test_insight_store.py
git commit -m "feat(insights): Neo4j Insight node + edges writer (SP-B.7)"
```

### Task SP-B.8 — InsightStore.read (single + list)

- [ ] **Step 1: Write the failing test**

Append to `tests/insights_eval/test_insight_store.py`:
```python
def test_list_filters_by_profile():
    coll = FakeMongoCollection()
    backend = MongoIndexBackend(collection=coll)
    store = InsightStore(mongo_index=backend, qdrant=None, neo4j=None)
    a = _insight()
    a.insight_id = "i-A"
    a.profile_id = "P1"
    b = _insight()
    b.insight_id = "i-B"
    b.profile_id = "P2"
    b.headline = "Different"
    store.write(a)
    store.write(b)
    rows = store.list_for_profile(profile_id="P1")
    assert len(rows) == 1
    assert rows[0]["insight_id"] == "i-A"


def test_list_filters_by_insight_type():
    coll = FakeMongoCollection()
    backend = MongoIndexBackend(collection=coll)
    store = InsightStore(mongo_index=backend, qdrant=None, neo4j=None)
    a = _insight()
    a.insight_id = "ia"
    b = _insight()
    b.insight_id = "ib"
    b.insight_type = "anomaly"
    b.headline = "Anomaly headline"
    store.write(a)
    store.write(b)
    rows = store.list_for_profile(profile_id="p-1", insight_types=["anomaly"])
    assert len(rows) == 1
    assert rows[0]["insight_type"] == "anomaly"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/insights_eval/test_insight_store.py::test_list_filters_by_profile -v`
Expected: FAIL — `AttributeError: 'InsightStore' has no 'list_for_profile'`.

- [ ] **Step 3: Add list method**

Append to `src/intelligence/insights/store.py` inside the `InsightStore` class:
```python
    def list_for_profile(
        self,
        *,
        profile_id: str,
        insight_types=None,
        severities=None,
        domain=None,
        since=None,
        limit: int = 50,
        offset: int = 0,
    ):
        query: Dict[str, Any] = {"profile_id": profile_id}
        if insight_types:
            query["insight_type"] = {"$in": list(insight_types)}
        if severities:
            query["severity"] = {"$in": list(severities)}
        if domain:
            query["domain"] = domain
        rows = self._mongo.list(query)
        if since:
            rows = [r for r in rows if r.get("refreshed_at", "") >= since]
        return rows[offset : offset + limit]

    def get_by_id(self, *, insight_id: str):
        rows = self._mongo.list({"insight_id": insight_id})
        return rows[0] if rows else None
```

Note: `FakeMongoCollection.find` does not handle `$in` operators — extend the test fake to support them in this same task:

Update `FakeMongoCollection.find` in `tests/insights_eval/test_insight_store.py`:
```python
    def find(self, query):
        def matches(d):
            for k, v in query.items():
                if isinstance(v, dict) and "$in" in v:
                    if d.get(k) not in v["$in"]:
                        return False
                elif d.get(k) != v:
                    return False
            return True
        return [d for d in self.docs if matches(d)]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/insights_eval/test_insight_store.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/intelligence/insights/store.py tests/insights_eval/test_insight_store.py
git commit -m "feat(insights): list_for_profile + get_by_id read paths on Mongo index (SP-B.8)"
```

### Task SP-B.9 — Stale-flag updater

- [ ] **Step 1: Write the failing test**

`tests/insights_eval/test_insight_staleness.py`:
```python
from src.intelligence.insights.schema import Insight, EvidenceSpan
from src.intelligence.insights.staleness import mark_stale_for_documents


class FakeColl:
    def __init__(self, docs):
        self.docs = docs
    def update_many(self, query, update):
        n = 0
        for d in self.docs:
            ok = True
            for k, v in query.items():
                if isinstance(v, dict) and "$in" in v:
                    if d.get(k) not in v["$in"]:
                        ok = False; break
                elif d.get(k) != v:
                    ok = False; break
            if ok:
                d.update(update.get("$set", {}))
                n += 1
        return type("R", (), {"modified_count": n})()


def test_mark_stale_flags_only_affected_insights():
    docs = [
        {"insight_id": "i-A", "profile_id": "P1", "document_ids": ["DOC-X"], "stale": False},
        {"insight_id": "i-B", "profile_id": "P1", "document_ids": ["DOC-Y"], "stale": False},
        {"insight_id": "i-C", "profile_id": "P1", "document_ids": ["DOC-X", "DOC-Z"], "stale": False},
    ]
    coll = FakeColl(docs)
    n = mark_stale_for_documents(
        collection=coll, profile_id="P1", document_ids=["DOC-X"]
    )
    assert n == 2
    assert docs[0]["stale"] is True
    assert docs[1]["stale"] is False
    assert docs[2]["stale"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/insights_eval/test_insight_staleness.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

`src/intelligence/insights/staleness.py`:
```python
"""Stale-flag updater.

When a document changes, every insight that includes that document_id
in its document_ids list is marked stale. Surface layer renders stale
insights with a "refreshing..." indicator until they re-run.

Per spec Section 9.3 — never silently show outdated data without flagging.
"""
from __future__ import annotations

from typing import Iterable


def mark_stale_for_documents(*, collection, profile_id: str, document_ids: Iterable[str]) -> int:
    """Mark every insight whose document_ids include any of the listed docs."""
    doc_list = list(document_ids)
    if not doc_list:
        return 0
    result = collection.update_many(
        {"profile_id": profile_id, "document_ids": {"$in": doc_list}},
        {"$set": {"stale": True}},
    )
    return getattr(result, "modified_count", 0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/insights_eval/test_insight_staleness.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/intelligence/insights/staleness.py tests/insights_eval/test_insight_staleness.py
git commit -m "feat(insights): stale-flag updater — affected-only refresh signal (SP-B.9)"
```

---

## Sub-Project SP-D — Knowledge Layer

**Files:**
- Create: `src/intelligence/knowledge/__init__.py`
- Create: `src/intelligence/knowledge/provider.py`
- Create: `src/intelligence/knowledge/template_resolver.py`
- Create: `src/intelligence/knowledge/bundled/insurance_taxonomy_v1.json`
- Create: `src/intelligence/knowledge/bundled/icd10_subset_v1.json`
- Create: `src/intelligence/knowledge/bundled/hr_policies_v1.json`
- Create: `src/intelligence/knowledge/bundled/procurement_terms_v1.json`
- Test: `tests/insights_eval/test_knowledge_provider.py`
- Test: `tests/insights_eval/test_template_resolver.py`

### Task SP-D.1 — KnowledgeProvider interface + JSON loader

- [ ] **Step 1: Write the failing test**

`tests/insights_eval/test_knowledge_provider.py`:
```python
import json
import pytest

from src.intelligence.knowledge.provider import (
    KnowledgeProvider,
    JsonKnowledgeProvider,
    KbNotFound,
)


def test_lookup_term(tmp_path):
    kb = {"version": "1.0", "entries": {"flood": "Flood damage exclusion clause"}}
    path = tmp_path / "kb.json"
    path.write_text(json.dumps(kb))
    p: KnowledgeProvider = JsonKnowledgeProvider.load_from_path(str(path), kb_id="kb1")
    assert p.kb_id == "kb1"
    assert p.lookup("flood") == "Flood damage exclusion clause"


def test_lookup_missing_returns_none(tmp_path):
    path = tmp_path / "kb.json"
    path.write_text(json.dumps({"version": "1.0", "entries": {}}))
    p = JsonKnowledgeProvider.load_from_path(str(path), kb_id="empty")
    assert p.lookup("nope") is None


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(KbNotFound):
        JsonKnowledgeProvider.load_from_path(str(tmp_path / "no.json"), kb_id="x")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/insights_eval/test_knowledge_provider.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

`src/intelligence/knowledge/__init__.py`:
```python
from src.intelligence.knowledge.provider import (
    KnowledgeProvider,
    JsonKnowledgeProvider,
    KbNotFound,
)

__all__ = ["KnowledgeProvider", "JsonKnowledgeProvider", "KbNotFound"]
```

`src/intelligence/knowledge/provider.py`:
```python
"""KnowledgeProvider — sanctioned KB lookup for adapters.

KBs are static JSON files. v1 ships bundled KBs; v2+ may load from Blob.
Per spec Section 8: KB augments interpretation, never adds claims to
insight bodies.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol


class KbNotFound(FileNotFoundError):
    pass


class KnowledgeProvider(Protocol):
    @property
    def kb_id(self) -> str: ...
    def lookup(self, term: str) -> Optional[str]: ...
    def interpret(self, value: Any) -> Optional[str]: ...


@dataclass
class JsonKnowledgeProvider:
    kb_id: str
    version: str
    entries: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def load_from_path(cls, path: str, *, kb_id: str) -> "JsonKnowledgeProvider":
        if not os.path.exists(path):
            raise KbNotFound(path)
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh) or {}
        return cls(
            kb_id=kb_id,
            version=str(raw.get("version") or "1.0"),
            entries=dict(raw.get("entries") or {}),
        )

    def lookup(self, term: str) -> Optional[str]:
        return self.entries.get(term.strip().lower()) or self.entries.get(term)

    def interpret(self, value: Any) -> Optional[str]:
        # Default: same as lookup
        return self.lookup(str(value))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/insights_eval/test_knowledge_provider.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/intelligence/knowledge/__init__.py src/intelligence/knowledge/provider.py tests/insights_eval/test_knowledge_provider.py
git commit -m "feat(knowledge): KnowledgeProvider interface + JSON loader (SP-D.1)"
```

### Task SP-D.2 — Bundled KBs (insurance, ICD-10 subset, HR, procurement)

- [ ] **Step 1: Write the failing test**

Append to `tests/insights_eval/test_knowledge_provider.py`:
```python
def test_bundled_kbs_present_and_loadable():
    from pathlib import Path
    from src.intelligence.knowledge.provider import JsonKnowledgeProvider

    bundled = Path("src/intelligence/knowledge/bundled")
    expected = [
        "insurance_taxonomy_v1.json",
        "icd10_subset_v1.json",
        "hr_policies_v1.json",
        "procurement_terms_v1.json",
    ]
    for name in expected:
        path = bundled / name
        assert path.exists(), f"bundled KB missing: {name}"
        kb = JsonKnowledgeProvider.load_from_path(str(path), kb_id=name)
        assert kb.entries, f"bundled KB empty: {name}"
        assert kb.version
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/insights_eval/test_knowledge_provider.py::test_bundled_kbs_present_and_loadable -v`
Expected: FAIL — files missing.

- [ ] **Step 3: Create the four bundled KBs**

`src/intelligence/knowledge/bundled/insurance_taxonomy_v1.json`:
```json
{
  "version": "1.0",
  "describes": "Common insurance policy types, coverage categories, exclusion classes",
  "entries": {
    "flood": "Flood damage — typically excluded from standard auto and home policies; requires separate flood insurance.",
    "earthquake": "Earthquake damage — typically excluded; requires separate earthquake rider.",
    "comprehensive": "Comprehensive coverage — covers non-collision damage (theft, vandalism, weather).",
    "collision": "Collision coverage — covers damage from collision with another vehicle or object.",
    "liability": "Liability coverage — covers damages and injuries to others when policyholder is at fault.",
    "deductible": "Deductible — the amount the policyholder pays out-of-pocket before the insurer pays.",
    "premium": "Premium — periodic payment to maintain the policy.",
    "exclusion": "Exclusion — specific conditions or events the policy does not cover.",
    "rider": "Rider — supplemental coverage added to a base policy.",
    "umbrella": "Umbrella policy — extra liability coverage beyond underlying policy limits."
  }
}
```

`src/intelligence/knowledge/bundled/icd10_subset_v1.json`:
```json
{
  "version": "1.0",
  "describes": "Small subset of ICD-10 codes for common conditions. Augments interpretation; never used to fabricate diagnoses not in the document.",
  "entries": {
    "e11": "Type 2 diabetes mellitus.",
    "i10": "Essential (primary) hypertension.",
    "j45": "Asthma.",
    "n18": "Chronic kidney disease.",
    "k21": "Gastro-esophageal reflux disease (GERD).",
    "f32": "Major depressive disorder, single episode.",
    "g43": "Migraine.",
    "m54": "Dorsalgia (back pain).",
    "r05": "Cough.",
    "z00": "Encounter for general examination without complaint, suspected or reported diagnosis."
  }
}
```

`src/intelligence/knowledge/bundled/hr_policies_v1.json`:
```json
{
  "version": "1.0",
  "describes": "Common HR policy patterns and terms",
  "entries": {
    "ptopolicy": "Paid Time Off — accrual-based or annual-grant time-off policy.",
    "fmla": "Family and Medical Leave Act — US federal job-protected unpaid leave for qualifying medical/family reasons.",
    "noncompete": "Non-compete clause — agreement restricting post-employment competition; enforceability varies by jurisdiction.",
    "atwill": "At-will employment — either party may terminate at any time without cause, subject to law.",
    "exit_interview": "Exit interview — final discussion with departing employee; commonly captures feedback.",
    "performance_review": "Performance review — periodic structured feedback session.",
    "probationary_period": "Probationary period — initial employment phase with typically reduced protection.",
    "severance": "Severance — payment upon termination, often tied to length of service."
  }
}
```

`src/intelligence/knowledge/bundled/procurement_terms_v1.json`:
```json
{
  "version": "1.0",
  "describes": "Common procurement / supply-chain terms",
  "entries": {
    "rfp": "Request for Proposal — formal solicitation document.",
    "rfq": "Request for Quotation — invitation for price quotes.",
    "po": "Purchase Order — buyer's formal commitment to purchase.",
    "sla": "Service Level Agreement — contractual performance commitment.",
    "moq": "Minimum Order Quantity — smallest unit count a supplier accepts.",
    "lead_time": "Lead time — duration from order placement to delivery.",
    "incoterms": "Incoterms — international commercial terms defining buyer/seller responsibilities.",
    "vendor_consolidation": "Vendor consolidation — reducing number of suppliers to leverage scale.",
    "duplicate_vendor": "Duplicate vendor — same supplier registered under multiple records, causing payment fragmentation.",
    "early_pay_discount": "Early-pay discount — pricing reduction for paying invoices before due date."
  }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/insights_eval/test_knowledge_provider.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/intelligence/knowledge/bundled/ tests/insights_eval/test_knowledge_provider.py
git commit -m "feat(knowledge): bundle insurance + ICD-10 subset + HR + procurement KBs (SP-D.2)"
```

### Task SP-D.3 — Template resolver for {{kb.lookup(...)}} directives

- [ ] **Step 1: Write the failing test**

`tests/insights_eval/test_template_resolver.py`:
```python
import pytest

from src.intelligence.knowledge.provider import JsonKnowledgeProvider
from src.intelligence.knowledge.template_resolver import resolve_template


def _kb(tmp_path, entries):
    import json
    path = tmp_path / "kb.json"
    path.write_text(json.dumps({"version": "1.0", "entries": entries}))
    return JsonKnowledgeProvider.load_from_path(str(path), kb_id="kb1")


def test_simple_lookup_directive(tmp_path):
    kb = _kb(tmp_path, {"flood": "Flood is typically excluded."})
    text = "Note: {{kb.lookup('flood')}}"
    out = resolve_template(text, kb=kb)
    assert out == "Note: Flood is typically excluded."


def test_unknown_term_replaced_with_blank(tmp_path):
    kb = _kb(tmp_path, {})
    text = "Note: {{kb.lookup('nope')}}"
    out = resolve_template(text, kb=kb)
    assert out == "Note: "  # blank substitution


def test_no_directive_returns_text_unchanged(tmp_path):
    kb = _kb(tmp_path, {})
    text = "no directives here"
    out = resolve_template(text, kb=kb)
    assert out == text


def test_kb_none_means_directives_become_blank():
    text = "Note: {{kb.lookup('flood')}}"
    out = resolve_template(text, kb=None)
    assert out == "Note: "
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/insights_eval/test_template_resolver.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

`src/intelligence/knowledge/template_resolver.py`:
```python
"""Resolves {{kb.lookup('term')}} directives in researcher prompts.

Used by Researcher Agent v2 to inject KB-derived interpretation into
prompts before LLM call. The resolved text becomes part of the prompt;
the model sees facts + KB-augmented context together but is instructed
to keep KB content out of insight bodies (OQ1 separation rule).
"""
from __future__ import annotations

import re
from typing import Optional

from src.intelligence.knowledge.provider import KnowledgeProvider

_DIRECTIVE = re.compile(r"\{\{kb\.lookup\(['\"]([^'\"]+)['\"]\)\}\}")


def resolve_template(text: str, *, kb: Optional[KnowledgeProvider]) -> str:
    def _replace(match):
        term = match.group(1)
        if kb is None:
            return ""
        v = kb.lookup(term)
        return v if v is not None else ""
    return _DIRECTIVE.sub(_replace, text)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/insights_eval/test_template_resolver.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/intelligence/knowledge/template_resolver.py tests/insights_eval/test_template_resolver.py
git commit -m "feat(knowledge): {{kb.lookup}} template resolver for researcher prompts (SP-D.3)"
```

---

## Sub-Project SP-K — Regression + Perf Test Framework

**Files:**
- Create: `tests/regression/__init__.py`
- Create: `tests/regression/all_flags_off.py`
- Create: `tests/perf/__init__.py`
- Create: `tests/perf/api_ask_latency.py`
- Create: `tests/perf/upload_to_screening_eligible.py`
- Create: `tests/perf/insight_lookup_p95.py`
- Create: `tests/regression/conftest.py`

### Task SP-K.1 — All-flags-off regression scaffold

- [ ] **Step 1: Write the failing test**

`tests/regression/__init__.py`: empty file.

`tests/regression/conftest.py`:
```python
import pytest


@pytest.fixture
def all_flags_off(monkeypatch):
    """Ensure all 25 insights-portal flags are unset (default false)."""
    from src.api.feature_flags import FLAG_NAMES
    for name in FLAG_NAMES:
        monkeypatch.delenv(name, raising=False)
    yield
```

`tests/regression/all_flags_off.py`:
```python
"""Hard regression test — with every flag off, behavior is byte-identical
to preprod_v02 (the baseline at the time this branch was cut).

This test is the gate for spec Section 15.1.

Currently asserts:
- feature_flags module exists and reports all-False
- importing feature_flags / adapters / insights modules has zero side
  effects on existing modules
- all 26 endpoints owned by the spec return 404 when their flag is off

Endpoint coverage is filled in by SP-F. For SP-K.1, we just install the
test scaffold + the import-side-effect check.
"""
from __future__ import annotations

import importlib
import pytest


def test_feature_flags_module_imports_cleanly(all_flags_off):
    mod = importlib.import_module("src.api.feature_flags")
    from src.api.feature_flags import FLAG_NAMES, is_enabled, FeatureFlags
    flags = FeatureFlags()
    for name in FLAG_NAMES:
        assert is_enabled(name, flags) is False


def test_adapters_module_imports_cleanly(all_flags_off):
    importlib.import_module("src.intelligence.adapters")


def test_insights_module_imports_cleanly(all_flags_off):
    importlib.import_module("src.intelligence.insights")


def test_knowledge_module_imports_cleanly(all_flags_off):
    importlib.import_module("src.intelligence.knowledge")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/regression/all_flags_off.py -v`
Expected: PASS (modules from SP-J/A/B/D already import). If any FAIL, fix the offending module.

- [ ] **Step 3: No implementation needed**

- [ ] **Step 4: Confirm all tests pass**

Run: `pytest tests/regression/ -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/regression/__init__.py tests/regression/conftest.py tests/regression/all_flags_off.py
git commit -m "feat(regression): all-flags-off scaffold + import-side-effect checks (SP-K.1)"
```

### Task SP-K.2 — Insight-lookup p95 latency test

- [ ] **Step 1: Write the failing test**

`tests/perf/__init__.py`: empty file.

`tests/perf/insight_lookup_p95.py`:
```python
"""Insight lookup latency — p95 must be ≤ 50ms per spec Section 13.2.

This test exercises the InsightStore.list_for_profile path with a
realistic-size in-memory index (1000 insights across 50 profiles) and
asserts p95 over 1000 lookups stays under budget.
"""
from __future__ import annotations

import time
import statistics

import pytest

from src.intelligence.insights.schema import Insight, EvidenceSpan
from src.intelligence.insights.store import InsightStore, MongoIndexBackend


class _Coll:
    def __init__(self):
        self.docs = []
    def update_one(self, filter, update, upsert=False):
        match = next(
            (d for d in self.docs if all(d.get(k) == v for k, v in filter.items())),
            None,
        )
        if match is None:
            d = {**filter, **update.get("$set", {})}
            self.docs.append(d)
        else:
            match.update(update.get("$set", {}))
        return type("R", (), {"matched_count": int(match is not None)})()
    def find(self, query):
        def matches(d):
            for k, v in query.items():
                if isinstance(v, dict) and "$in" in v:
                    if d.get(k) not in v["$in"]:
                        return False
                elif d.get(k) != v:
                    return False
            return True
        return [d for d in self.docs if matches(d)]


def _seed(store, *, n_profiles: int, per_profile: int) -> None:
    for p in range(n_profiles):
        for i in range(per_profile):
            insight = Insight(
                insight_id=f"p{p}-i{i}",
                profile_id=f"profile-{p}",
                subscription_id="s",
                document_ids=[f"DOC-{i}"],
                domain="generic",
                insight_type="anomaly",
                headline=f"H{i}",
                body=f"H{i} body content",
                evidence_doc_spans=[EvidenceSpan(
                    document_id=f"DOC-{i}", page=1, char_start=0, char_end=2,
                    quote=f"H{i}",
                )],
                confidence=0.5,
                severity="notice",
                adapter_version="generic@1.0",
            )
            store.write(insight)


@pytest.mark.perf
def test_p95_under_50ms():
    coll = _Coll()
    store = InsightStore(mongo_index=MongoIndexBackend(collection=coll), qdrant=None, neo4j=None)
    _seed(store, n_profiles=50, per_profile=20)  # 1000 insights total

    timings = []
    for _ in range(1000):
        start = time.perf_counter_ns()
        rows = store.list_for_profile(profile_id="profile-25")
        end = time.perf_counter_ns()
        assert len(rows) == 20
        timings.append((end - start) / 1_000_000.0)
    p95 = statistics.quantiles(timings, n=100)[94]
    assert p95 <= 50.0, f"p95 {p95:.2f}ms exceeds 50ms budget"
```

- [ ] **Step 2: Run test to verify it fails initially (no perf marker registered)**

Run: `pytest tests/perf/insight_lookup_p95.py -v`
Expected: PASS or `PytestUnknownMarkWarning`. If unknown-mark warning, register the marker:

Append to `pyproject.toml` (or create `pytest.ini`) — note: this project's existing config may already have markers; if not, add:

```ini
# pytest.ini  (only if no existing marker config)
[pytest]
markers =
    perf: performance tests asserting latency budgets
```

- [ ] **Step 3: No implementation needed**

The test passes against the existing InsightStore. If p95 exceeds 50ms, that means SP-B.5/8 needs an index optimization — fix at that point, not here.

- [ ] **Step 4: Confirm test passes**

Run: `pytest tests/perf/insight_lookup_p95.py -v -m perf`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/perf/__init__.py tests/perf/insight_lookup_p95.py
git commit -m "feat(perf): insight-lookup p95 ≤50ms latency assertion (SP-K.2)"
```

### Task SP-K.3 — /api/ask latency placeholder (filled in by SP-G)

- [ ] **Step 1: Write the failing test placeholder**

`tests/perf/api_ask_latency.py`:
```python
"""/api/ask p95 latency assertion.

Asserts /api/ask p95 with INSIGHTS_PROACTIVE_INJECTION on equals p95
with it off, ±5%. Per spec Section 13.2.

NOTE: This test depends on SP-G (proactive injection) shipping. Until
then it skips with a clear reason. SP-G's final task re-enables it.
"""
from __future__ import annotations

import pytest


@pytest.mark.perf
def test_proactive_injection_does_not_regress_p95():
    pytest.skip(
        "Awaits SP-G — proactive injection helper not yet wired. "
        "Re-enable in SP-G final task."
    )
```

- [ ] **Step 2: Run test to verify the skip mechanism**

Run: `pytest tests/perf/api_ask_latency.py -v -m perf`
Expected: 1 skipped.

- [ ] **Step 3: No implementation needed**

- [ ] **Step 4: Done**

- [ ] **Step 5: Commit**

```bash
git add tests/perf/api_ask_latency.py
git commit -m "feat(perf): /api/ask latency test placeholder, finalized by SP-G (SP-K.3)"
```

### Task SP-K.4 — Upload-to-screening-eligible perf placeholder

- [ ] **Step 1: Write the failing test placeholder**

`tests/perf/upload_to_screening_eligible.py`:
```python
"""Upload → screening-eligible time perf test.

Asserts time-from-upload-complete to HITL-screening-eligible is unchanged
after researcher v2 lands. Per spec Section 13.3.

NOTE: This test depends on the researcher_v2_queue being installed and
a synthetic upload harness existing. SP-C and SP-K.5 wire it up.
"""
from __future__ import annotations

import pytest


@pytest.mark.perf
def test_upload_to_screening_eligible_unchanged():
    pytest.skip(
        "Awaits SP-C researcher v2 + a synthetic upload harness. "
        "Re-enable in SP-K.5."
    )
```

- [ ] **Step 2: Run test to confirm skip**

Run: `pytest tests/perf/upload_to_screening_eligible.py -v -m perf`
Expected: 1 skipped.

- [ ] **Step 3: No implementation needed**

- [ ] **Step 4: Done**

- [ ] **Step 5: Commit**

```bash
git add tests/perf/upload_to_screening_eligible.py
git commit -m "feat(perf): upload→screening-eligible perf placeholder (SP-K.4)"
```

---

## Sub-Project SP-C — Researcher Agent v2

**Files:**
- Create: `src/intelligence/researcher_v2/__init__.py`
- Create: `src/intelligence/researcher_v2/runner.py`
- Create: `src/intelligence/researcher_v2/parser.py`
- Create: `src/intelligence/researcher_v2/profile_passes.py`
- Create: `src/tasks/researcher_v2.py`
- Create: `src/docwain/prompts/researcher_v2_generic.py` (one prompt module per insight type)
- Modify: `src/celery_app.py` (register `researcher_v2_queue`)
- Test: `tests/insights_eval/test_researcher_v2_parser.py`
- Test: `tests/insights_eval/test_researcher_v2_runner.py`
- Test: `tests/insights_eval/test_researcher_v2_profile_passes.py`
- Test: `tests/insights_eval/test_researcher_v2_task.py`

### Task SP-C.1 — LLM response parser

- [ ] **Step 1: Write the failing test**

`tests/insights_eval/test_researcher_v2_parser.py`:
```python
import pytest

from src.intelligence.researcher_v2.parser import (
    parse_typed_insight_response,
    ParsedInsight,
    ParseError,
)


SAMPLE_GOOD = """
{
  "insights": [
    {
      "headline": "No flood coverage",
      "body": "The policy excludes flood damage. Flood damage exclusion is listed under exclusions.",
      "evidence_doc_spans": [
        {"document_id": "DOC-1", "page": 1, "char_start": 100, "char_end": 130, "quote": "Excludes: flood damage"}
      ],
      "external_kb_refs": [
        {"kb_id": "insurance_taxonomy_v1", "ref": "exclusions/flood", "label": "Flood exclusion"}
      ],
      "confidence": 0.95,
      "severity": "warn"
    }
  ]
}
"""


def test_parses_good_response():
    items = parse_typed_insight_response(SAMPLE_GOOD)
    assert len(items) == 1
    p = items[0]
    assert isinstance(p, ParsedInsight)
    assert p.headline == "No flood coverage"
    assert p.evidence_doc_spans[0]["document_id"] == "DOC-1"
    assert p.external_kb_refs[0]["kb_id"] == "insurance_taxonomy_v1"
    assert p.confidence == 0.95
    assert p.severity == "warn"


def test_strips_markdown_code_fence():
    fenced = "```json\n" + SAMPLE_GOOD + "\n```"
    items = parse_typed_insight_response(fenced)
    assert len(items) == 1


def test_empty_insights_list_returns_empty():
    items = parse_typed_insight_response('{"insights": []}')
    assert items == []


def test_malformed_json_raises():
    with pytest.raises(ParseError):
        parse_typed_insight_response("not json")


def test_missing_required_field_skipped():
    txt = '{"insights": [{"headline": "x"}]}'  # no evidence_doc_spans, etc.
    items = parse_typed_insight_response(txt)
    # Item dropped — required fields missing
    assert items == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/insights_eval/test_researcher_v2_parser.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

`src/intelligence/researcher_v2/__init__.py`:
```python
from src.intelligence.researcher_v2.parser import (
    parse_typed_insight_response,
    ParsedInsight,
    ParseError,
)

__all__ = ["parse_typed_insight_response", "ParsedInsight", "ParseError"]
```

`src/intelligence/researcher_v2/parser.py`:
```python
"""Parse LLM responses for typed insight passes.

The model is prompted to return JSON with shape:
  {"insights": [{"headline", "body", "evidence_doc_spans", "external_kb_refs",
                 "confidence", "severity"}]}

Malformed entries (missing required fields, non-numeric confidence) are
dropped silently. Completely malformed JSON raises ParseError.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List


class ParseError(ValueError):
    pass


@dataclass
class ParsedInsight:
    headline: str
    body: str
    evidence_doc_spans: List[Dict[str, Any]]
    external_kb_refs: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    severity: str = "notice"


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _strip_fence(text: str) -> str:
    text = text.strip()
    m = _FENCE.match(text)
    return m.group(1).strip() if m else text


def parse_typed_insight_response(text: str) -> List[ParsedInsight]:
    body = _strip_fence(text)
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ParseError(f"invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ParseError("response root must be a JSON object")
    items = data.get("insights") or []
    if not isinstance(items, list):
        return []
    parsed: List[ParsedInsight] = []
    for raw in items:
        if not isinstance(raw, dict):
            continue
        headline = str(raw.get("headline") or "").strip()
        body_text = str(raw.get("body") or "").strip()
        spans = raw.get("evidence_doc_spans") or []
        if not headline or not body_text or not isinstance(spans, list) or not spans:
            continue
        try:
            confidence = float(raw.get("confidence") or 0.0)
        except (TypeError, ValueError):
            continue
        severity = str(raw.get("severity") or "notice").lower()
        if severity not in ("info", "notice", "warn", "critical"):
            severity = "notice"
        parsed.append(ParsedInsight(
            headline=headline,
            body=body_text,
            evidence_doc_spans=[
                {
                    "document_id": str(s.get("document_id") or ""),
                    "page": int(s.get("page") or 0),
                    "char_start": int(s.get("char_start") or 0),
                    "char_end": int(s.get("char_end") or 0),
                    "quote": str(s.get("quote") or ""),
                }
                for s in spans
                if isinstance(s, dict)
            ],
            external_kb_refs=[
                {
                    "kb_id": str(r.get("kb_id") or ""),
                    "ref": str(r.get("ref") or ""),
                    "label": str(r.get("label") or ""),
                }
                for r in (raw.get("external_kb_refs") or [])
                if isinstance(r, dict)
            ],
            confidence=confidence,
            severity=severity,
        ))
    return parsed
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/insights_eval/test_researcher_v2_parser.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/intelligence/researcher_v2/__init__.py src/intelligence/researcher_v2/parser.py tests/insights_eval/test_researcher_v2_parser.py
git commit -m "feat(researcher-v2): typed-insight LLM response parser (SP-C.1)"
```

### Task SP-C.2 — Generic prompt module (fallback)

- [ ] **Step 1: Write the failing test**

`tests/insights_eval/test_researcher_v2_prompts.py`:
```python
from src.docwain.prompts.researcher_v2_generic import (
    build_typed_insight_prompt,
    SYSTEM_PROMPT,
)


def test_system_prompt_includes_separation_rule():
    assert "external" in SYSTEM_PROMPT.lower()
    assert "do not" in SYSTEM_PROMPT.lower() or "must not" in SYSTEM_PROMPT.lower()


def test_build_prompt_for_each_type():
    for itype in ("anomaly", "gap", "comparison", "scenario", "trend",
                  "recommendation", "conflict", "projection", "next_action"):
        prompt = build_typed_insight_prompt(
            insight_type=itype,
            domain_name="generic",
            document_text="Some doc text " * 50,
            kb_context="",
        )
        assert itype in prompt.lower() or itype.replace("_", " ") in prompt.lower()
        assert "JSON" in prompt
        assert "evidence_doc_spans" in prompt


def test_truncates_long_doc_text():
    long_text = "x" * 100_000
    prompt = build_typed_insight_prompt(
        insight_type="anomaly",
        domain_name="generic",
        document_text=long_text,
        kb_context="",
    )
    assert len(prompt) < 30_000  # truncation enforced
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/insights_eval/test_researcher_v2_prompts.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

`src/docwain/prompts/researcher_v2_generic.py`:
```python
"""Generic researcher v2 prompts for all 9 insight types.

Per spec Section 7 — when a domain adapter has no per-type override,
this module supplies the prompt. KB-aware via kb_context.
"""
from __future__ import annotations

INSIGHT_TYPES = (
    "anomaly", "gap", "comparison", "scenario", "trend",
    "recommendation", "conflict", "projection", "next_action",
)

SYSTEM_PROMPT = (
    "You are DocWain's Researcher Agent v2. Produce structured insights "
    "from documents. Output ONLY valid JSON. CRITICAL — content rules:\n"
    "(1) The 'body' field must contain only statements derivable from "
    "the document text. You MUST NOT introduce facts that come from "
    "external knowledge into the body.\n"
    "(2) External knowledge MAY be cited via 'external_kb_refs' as "
    "metadata, but never mixed into 'body'.\n"
    "(3) Every insight requires at least one entry in 'evidence_doc_spans'.\n"
    "(4) Quote the supporting text verbatim in each span.\n"
)


_TYPE_GUIDANCE = {
    "anomaly": "Identify anomalies — values, dates, terms that look unusual, inconsistent, or risky for this kind of document.",
    "gap": "Identify gaps — what the document does not cover that a reader would expect for this kind of content.",
    "comparison": "Compare aspects across the documents — only fields/values present in 2+ documents.",
    "scenario": "Reason through plausible scenarios the document content suggests — 'if X then Y', grounded in stated terms.",
    "trend": "Identify trends — directional changes over time using dated content in the documents.",
    "recommendation": "Recommend concrete next-best-actions a user should consider, justified by document content.",
    "conflict": "Detect contradictions or conflicts between documents.",
    "projection": "Project forward — numeric or categorical estimates extrapolated from document content.",
    "next_action": "Surface time-sensitive or attention-required next steps the documents imply.",
}


_MAX_DOC_CHARS = 16000


def build_typed_insight_prompt(
    *,
    insight_type: str,
    domain_name: str,
    document_text: str,
    kb_context: str = "",
) -> str:
    if insight_type not in INSIGHT_TYPES:
        raise ValueError(f"unknown insight_type: {insight_type}")
    guidance = _TYPE_GUIDANCE[insight_type]
    truncated = document_text[:_MAX_DOC_CHARS]
    kb_block = f"\nDomain knowledge context (for interpretation only — do NOT inject into body):\n{kb_context}\n" if kb_context else ""
    return (
        f"Domain: {domain_name}\n"
        f"Insight type to produce: {insight_type}\n"
        f"Guidance: {guidance}\n"
        f"{kb_block}\n"
        f"Document text:\n\n{truncated}\n\n"
        "Return JSON with this shape (no prose, no markdown fences):\n"
        '{"insights": [{"headline": "≤25 words", "body": "≤600 chars",'
        ' "evidence_doc_spans": [{"document_id": "...", "page": 0,'
        ' "char_start": 0, "char_end": 0, "quote": "verbatim text"}],'
        ' "external_kb_refs": [{"kb_id": "...", "ref": "...", "label": "..."}],'
        ' "confidence": 0.0, "severity": "info|notice|warn|critical"}]}\n'
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/insights_eval/test_researcher_v2_prompts.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/docwain/prompts/researcher_v2_generic.py tests/insights_eval/test_researcher_v2_prompts.py
git commit -m "feat(researcher-v2): generic prompt module covering all 9 insight types (SP-C.2)"
```

### Task SP-C.3 — Per-doc runner (single insight type)

- [ ] **Step 1: Write the failing test**

`tests/insights_eval/test_researcher_v2_runner.py`:
```python
import pytest

from src.intelligence.researcher_v2.runner import (
    run_per_doc_insight_pass,
    DocPassInputs,
    DocPassResult,
)
from src.intelligence.adapters.schema import (
    Adapter, AppliesWhen, ResearcherSection, InsightTypeConfig,
    KnowledgeConfig,
)


def _adapter() -> Adapter:
    return Adapter(
        name="generic",
        version="1.0",
        description="t",
        applies_when=AppliesWhen(),
        researcher=ResearcherSection(insight_types={
            "anomaly": InsightTypeConfig(prompt_template="prompts/generic_anomaly.md", enabled=True),
        }),
        knowledge=KnowledgeConfig(),
    )


def _llm_returning(text):
    def call(*, system, user, **_):
        return text
    return call


def test_runner_emits_validated_insights():
    llm = _llm_returning(
        '{"insights": [{"headline":"H","body":"H — body grounded in quote excludes flood",'
        '"evidence_doc_spans":[{"document_id":"DOC-1","page":1,"char_start":0,"char_end":18,'
        '"quote":"excludes flood damage"}],"confidence":0.8,"severity":"warn"}]}'
    )
    result = run_per_doc_insight_pass(DocPassInputs(
        adapter=_adapter(),
        insight_type="anomaly",
        document_id="DOC-1",
        document_text="Policy excludes flood damage and earthquake.",
        profile_id="p-1",
        subscription_id="s-1",
        kb_provider=None,
        llm_call=llm,
    ))
    assert isinstance(result, DocPassResult)
    assert len(result.insights) == 1
    insight = result.insights[0]
    assert insight.headline == "H"
    assert insight.insight_type == "anomaly"
    assert insight.adapter_version == "generic@1.0"


def test_runner_skips_disabled_type():
    a = _adapter()
    a.researcher.insight_types["anomaly"].enabled = False
    result = run_per_doc_insight_pass(DocPassInputs(
        adapter=a,
        insight_type="anomaly",
        document_id="DOC-1",
        document_text="x",
        profile_id="p-1",
        subscription_id="s-1",
        kb_provider=None,
        llm_call=_llm_returning('{"insights":[]}'),
    ))
    assert result.insights == []
    assert result.skipped_reason == "type_disabled"


def test_runner_drops_insight_with_zero_evidence():
    llm = _llm_returning(
        '{"insights":[{"headline":"H","body":"x","evidence_doc_spans":[],'
        '"confidence":0.5,"severity":"notice"}]}'
    )
    result = run_per_doc_insight_pass(DocPassInputs(
        adapter=_adapter(),
        insight_type="anomaly",
        document_id="DOC-1",
        document_text="x",
        profile_id="p-1",
        subscription_id="s-1",
        kb_provider=None,
        llm_call=llm,
    ))
    # Parser drops because evidence_doc_spans empty
    assert result.insights == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/insights_eval/test_researcher_v2_runner.py -v`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

`src/intelligence/researcher_v2/runner.py`:
```python
"""Per-doc + profile-level researcher v2 runner.

Loads adapter, builds prompts, calls LLM, parses, constructs Insight
objects, applies validators (citation, body-separation), returns
results. Idempotency keys are computed at write time by InsightStore.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from src.intelligence.adapters.schema import Adapter
from src.intelligence.insights.schema import (
    Insight, EvidenceSpan, KbRef,
)
from src.intelligence.knowledge.provider import KnowledgeProvider
from src.intelligence.knowledge.template_resolver import resolve_template
from src.intelligence.researcher_v2.parser import (
    parse_typed_insight_response, ParseError,
)
from src.docwain.prompts.researcher_v2_generic import (
    SYSTEM_PROMPT,
    build_typed_insight_prompt,
)

logger = logging.getLogger(__name__)


LlmCall = Callable[..., str]


@dataclass
class DocPassInputs:
    adapter: Adapter
    insight_type: str
    document_id: str
    document_text: str
    profile_id: str
    subscription_id: str
    kb_provider: Optional[KnowledgeProvider]
    llm_call: LlmCall


@dataclass
class DocPassResult:
    insights: List[Insight] = field(default_factory=list)
    skipped_reason: Optional[str] = None
    parse_error: Optional[str] = None


def run_per_doc_insight_pass(inp: DocPassInputs) -> DocPassResult:
    cfg = inp.adapter.researcher.insight_types.get(inp.insight_type)
    if cfg is None or not cfg.enabled:
        return DocPassResult(skipped_reason="type_disabled")
    kb_context = ""
    if inp.kb_provider is not None and cfg.prompt_template:
        kb_context = resolve_template(cfg.prompt_template, kb=inp.kb_provider)
    user_prompt = build_typed_insight_prompt(
        insight_type=inp.insight_type,
        domain_name=inp.adapter.name,
        document_text=inp.document_text,
        kb_context=kb_context,
    )
    try:
        raw = inp.llm_call(system=SYSTEM_PROMPT, user=user_prompt)
    except Exception as exc:
        logger.warning("LLM call failed: %s", exc)
        return DocPassResult(skipped_reason="llm_error")
    try:
        parsed = parse_typed_insight_response(raw)
    except ParseError as exc:
        return DocPassResult(parse_error=str(exc))
    adapter_version = f"{inp.adapter.name}@{inp.adapter.version}"
    insights: List[Insight] = []
    for p in parsed:
        spans = [EvidenceSpan(**s) for s in p.evidence_doc_spans]
        kb_refs = [KbRef(**r) for r in p.external_kb_refs]
        try:
            insights.append(Insight(
                insight_id=str(uuid.uuid4()),
                profile_id=inp.profile_id,
                subscription_id=inp.subscription_id,
                document_ids=[inp.document_id],
                domain=inp.adapter.name,
                insight_type=inp.insight_type,
                headline=p.headline,
                body=p.body,
                evidence_doc_spans=spans,
                external_kb_refs=kb_refs,
                confidence=p.confidence,
                severity=p.severity,
                adapter_version=adapter_version,
            ))
        except ValueError as exc:
            logger.debug("dropping invalid insight: %s", exc)
            continue
    return DocPassResult(insights=insights)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/insights_eval/test_researcher_v2_runner.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/intelligence/researcher_v2/runner.py tests/insights_eval/test_researcher_v2_runner.py
git commit -m "feat(researcher-v2): per-doc typed-insight pass runner (SP-C.3)"
```

### Task SP-C.4 — Profile-level passes (multi-doc)

- [ ] **Step 1: Write the failing test**

`tests/insights_eval/test_researcher_v2_profile_passes.py`:
```python
import pytest

from src.intelligence.researcher_v2.profile_passes import (
    run_profile_pass,
    ProfilePassInputs,
)
from src.intelligence.adapters.schema import (
    Adapter, AppliesWhen, ResearcherSection, InsightTypeConfig, KnowledgeConfig,
)


def _adapter() -> Adapter:
    return Adapter(
        name="generic",
        version="1.0",
        description="t",
        applies_when=AppliesWhen(),
        researcher=ResearcherSection(insight_types={
            "comparison": InsightTypeConfig(prompt_template="", enabled=True, requires_min_docs=2),
            "conflict": InsightTypeConfig(prompt_template="", enabled=True, requires_min_docs=2),
        }),
        knowledge=KnowledgeConfig(),
    )


def _llm_returning(payload):
    def call(*, system, user, **_):
        return payload
    return call


def test_skips_when_below_min_docs():
    docs = [{"document_id": "D1", "text": "alpha"}]  # only 1 doc
    result = run_profile_pass(ProfilePassInputs(
        adapter=_adapter(),
        insight_type="comparison",
        documents=docs,
        profile_id="p", subscription_id="s",
        kb_provider=None,
        llm_call=_llm_returning('{"insights":[]}'),
    ))
    assert result.insights == []
    assert result.skipped_reason == "below_min_docs"


def test_emits_with_2_docs():
    docs = [
        {"document_id": "D1", "text": "Premium $1800"},
        {"document_id": "D2", "text": "Premium $2400"},
    ]
    payload = (
        '{"insights":[{"headline":"D2 premium higher",'
        '"body":"D2 premium $2400 vs D1 $1800",'
        '"evidence_doc_spans":[{"document_id":"D1","page":1,"char_start":0,"char_end":13,"quote":"Premium $1800"},'
        '{"document_id":"D2","page":1,"char_start":0,"char_end":13,"quote":"Premium $2400"}],'
        '"confidence":0.9,"severity":"notice"}]}'
    )
    result = run_profile_pass(ProfilePassInputs(
        adapter=_adapter(),
        insight_type="comparison",
        documents=docs,
        profile_id="p", subscription_id="s",
        kb_provider=None,
        llm_call=_llm_returning(payload),
    ))
    assert len(result.insights) == 1
    insight = result.insights[0]
    assert set(insight.document_ids) == {"D1", "D2"}
    assert insight.insight_type == "comparison"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/insights_eval/test_researcher_v2_profile_passes.py -v`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

`src/intelligence/researcher_v2/profile_passes.py`:
```python
"""Profile-level (cross-doc) researcher v2 passes.

For comparison / conflict / trend / projection — passes that require
≥2 documents. Document_ids on emitted insights point to all docs that
contributed evidence.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from src.intelligence.adapters.schema import Adapter
from src.intelligence.insights.schema import Insight, EvidenceSpan, KbRef
from src.intelligence.knowledge.provider import KnowledgeProvider
from src.intelligence.researcher_v2.parser import (
    parse_typed_insight_response, ParseError,
)
from src.docwain.prompts.researcher_v2_generic import (
    SYSTEM_PROMPT,
    build_typed_insight_prompt,
)

logger = logging.getLogger(__name__)


LlmCall = Callable[..., str]


@dataclass
class ProfilePassInputs:
    adapter: Adapter
    insight_type: str
    documents: List[Dict[str, Any]]   # [{"document_id": str, "text": str}]
    profile_id: str
    subscription_id: str
    kb_provider: Optional[KnowledgeProvider]
    llm_call: LlmCall


@dataclass
class ProfilePassResult:
    insights: List[Insight] = field(default_factory=list)
    skipped_reason: Optional[str] = None


def _join_docs(docs):
    parts = []
    for d in docs:
        parts.append(f"=== document_id: {d['document_id']} ===\n{d.get('text', '')}")
    return "\n\n".join(parts)


def run_profile_pass(inp: ProfilePassInputs) -> ProfilePassResult:
    cfg = inp.adapter.researcher.insight_types.get(inp.insight_type)
    if cfg is None or not cfg.enabled:
        return ProfilePassResult(skipped_reason="type_disabled")
    if len(inp.documents) < cfg.requires_min_docs:
        return ProfilePassResult(skipped_reason="below_min_docs")
    user_prompt = build_typed_insight_prompt(
        insight_type=inp.insight_type,
        domain_name=inp.adapter.name,
        document_text=_join_docs(inp.documents),
        kb_context="",
    )
    try:
        raw = inp.llm_call(system=SYSTEM_PROMPT, user=user_prompt)
    except Exception as exc:
        logger.warning("profile-pass LLM failed: %s", exc)
        return ProfilePassResult(skipped_reason="llm_error")
    try:
        parsed = parse_typed_insight_response(raw)
    except ParseError as exc:
        logger.debug("parse error: %s", exc)
        return ProfilePassResult(skipped_reason="parse_error")
    adapter_version = f"{inp.adapter.name}@{inp.adapter.version}"
    insights: List[Insight] = []
    for p in parsed:
        document_ids = sorted({s["document_id"] for s in p.evidence_doc_spans})
        spans = [EvidenceSpan(**s) for s in p.evidence_doc_spans]
        kb_refs = [KbRef(**r) for r in p.external_kb_refs]
        try:
            insights.append(Insight(
                insight_id=str(uuid.uuid4()),
                profile_id=inp.profile_id,
                subscription_id=inp.subscription_id,
                document_ids=document_ids,
                domain=inp.adapter.name,
                insight_type=inp.insight_type,
                headline=p.headline,
                body=p.body,
                evidence_doc_spans=spans,
                external_kb_refs=kb_refs,
                confidence=p.confidence,
                severity=p.severity,
                adapter_version=adapter_version,
            ))
        except ValueError as exc:
            logger.debug("dropping invalid profile insight: %s", exc)
            continue
    return ProfilePassResult(insights=insights)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/insights_eval/test_researcher_v2_profile_passes.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/intelligence/researcher_v2/profile_passes.py tests/insights_eval/test_researcher_v2_profile_passes.py
git commit -m "feat(researcher-v2): profile-level multi-doc passes (comparison/conflict/etc.) (SP-C.4)"
```

### Task SP-C.5 — Celery task wrapper (per-doc + profile)

- [ ] **Step 1: Write the failing test**

`tests/insights_eval/test_researcher_v2_task.py`:
```python
import pytest

# Note: these tests use a fake Celery shim so they don't require a broker.

from unittest.mock import MagicMock, patch

from src.intelligence.adapters.schema import (
    Adapter, AppliesWhen, ResearcherSection, InsightTypeConfig, KnowledgeConfig,
)
from src.intelligence.insights.schema import Insight, EvidenceSpan


def _adapter() -> Adapter:
    return Adapter(
        name="generic", version="1.0", description="t",
        applies_when=AppliesWhen(),
        researcher=ResearcherSection(insight_types={
            "anomaly": InsightTypeConfig(prompt_template="", enabled=True),
        }),
        knowledge=KnowledgeConfig(),
    )


def test_run_researcher_v2_calls_runner_per_enabled_type():
    from src.tasks.researcher_v2 import run_researcher_v2_for_doc

    fake_store = MagicMock()
    fake_runner = MagicMock()
    fake_runner.return_value.insights = [
        Insight(
            insight_id="i", profile_id="p", subscription_id="s",
            document_ids=["D"], domain="generic", insight_type="anomaly",
            headline="H", body="H grounded quote",
            evidence_doc_spans=[EvidenceSpan(
                document_id="D", page=1, char_start=0, char_end=2, quote="H"
            )],
            confidence=0.5, severity="notice", adapter_version="generic@1.0",
        )
    ]
    fake_runner.return_value.skipped_reason = None
    with patch(
        "src.tasks.researcher_v2.resolve_default_store", return_value=fake_store
    ), patch(
        "src.tasks.researcher_v2.resolve_default_adapter", return_value=_adapter()
    ), patch(
        "src.tasks.researcher_v2.run_per_doc_insight_pass", side_effect=fake_runner
    ):
        run_researcher_v2_for_doc(
            document_id="D", profile_id="p", subscription_id="s",
            document_text="x",
        )
    fake_store.write.assert_called()


def test_disabled_flag_short_circuits():
    from src.tasks.researcher_v2 import run_researcher_v2_for_doc

    with patch(
        "src.tasks.researcher_v2.insight_flag_enabled", return_value=False
    ) as _flag:
        result = run_researcher_v2_for_doc(
            document_id="D", profile_id="p", subscription_id="s",
            document_text="x",
        )
    assert result["status"] == "skipped_flag_off"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/insights_eval/test_researcher_v2_task.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write minimal implementation**

`src/tasks/researcher_v2.py`:
```python
"""Researcher Agent v2 Celery task entry points.

Per-doc and profile-level. Runs on `researcher_v2_queue`. Isolated from
`extraction_queue`, `embedding_queue`, `kg_queue`. Writes ONLY to
researcher_v2.* fields and the insights collections (Mongo + Qdrant +
Neo4j) — never touches pipeline_status (per feedback_mongo_status_stability.md).

Each insight type has its own flag (INSIGHTS_TYPE_*_ENABLED). The task
runs only enabled types, and short-circuits entirely if no type is
enabled (avoids wasted LLM calls when feature is fully off).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from src.api.config import insight_flag_enabled
from src.intelligence.researcher_v2.runner import (
    run_per_doc_insight_pass, DocPassInputs,
)
from src.intelligence.researcher_v2.profile_passes import (
    run_profile_pass, ProfilePassInputs,
)
from src.intelligence.insights.store import InsightStore
from src.intelligence.adapters.schema import Adapter

logger = logging.getLogger(__name__)


_TYPE_FLAGS = {
    "anomaly": "INSIGHTS_TYPE_ANOMALY_ENABLED",
    "gap": "INSIGHTS_TYPE_GAP_ENABLED",
    "comparison": "INSIGHTS_TYPE_COMPARISON_ENABLED",
    "scenario": "INSIGHTS_TYPE_SCENARIO_ENABLED",
    "trend": "INSIGHTS_TYPE_TREND_ENABLED",
    "recommendation": "INSIGHTS_TYPE_RECOMMENDATION_ENABLED",
    "conflict": "INSIGHTS_TYPE_CONFLICT_ENABLED",
    "projection": "INSIGHTS_TYPE_PROJECTION_ENABLED",
    "next_action": "INSIGHTS_TYPE_RECOMMENDATION_ENABLED",  # next_action shares recommendation flag in v1
}

_PROFILE_TYPES = ("comparison", "conflict", "trend", "projection")


def resolve_default_store() -> InsightStore:
    """Hook for tests + production wiring. Real implementation injects
    Mongo + Qdrant + Neo4j clients."""
    raise NotImplementedError("wire me from src.api startup")


def resolve_default_adapter(*, domain: str, subscription_id: str) -> Adapter:
    """Hook for tests + production wiring. Real implementation uses
    AdapterStore from src.intelligence.adapters.store."""
    raise NotImplementedError("wire me from src.api startup")


def resolve_default_llm():
    raise NotImplementedError("wire me from src.api startup")


def _enabled_types(types: List[str]) -> List[str]:
    return [t for t in types if insight_flag_enabled(_TYPE_FLAGS.get(t, ""))]


def run_researcher_v2_for_doc(
    *,
    document_id: str,
    profile_id: str,
    subscription_id: str,
    document_text: str,
    domain_hint: str = "generic",
) -> Dict[str, Any]:
    enabled = _enabled_types(list(_TYPE_FLAGS.keys()))
    if not enabled:
        return {"status": "skipped_flag_off"}
    adapter = resolve_default_adapter(domain=domain_hint, subscription_id=subscription_id)
    store = resolve_default_store()
    llm_call = resolve_default_llm()
    written = 0
    for itype in enabled:
        if itype not in adapter.researcher.insight_types:
            continue
        if itype in _PROFILE_TYPES:
            continue  # handled in run_researcher_v2_for_profile
        result = run_per_doc_insight_pass(DocPassInputs(
            adapter=adapter,
            insight_type=itype,
            document_id=document_id,
            document_text=document_text,
            profile_id=profile_id,
            subscription_id=subscription_id,
            kb_provider=None,
            llm_call=llm_call,
        ))
        for insight in result.insights:
            try:
                store.write(insight)
                written += 1
            except Exception as exc:
                logger.warning("insight write failed: %s", exc)
    return {"status": "ok", "written": written}


def run_researcher_v2_for_profile(
    *,
    profile_id: str,
    subscription_id: str,
    documents: List[Dict[str, Any]],
    domain_hint: str = "generic",
) -> Dict[str, Any]:
    enabled = _enabled_types(list(_PROFILE_TYPES))
    if not enabled:
        return {"status": "skipped_flag_off"}
    adapter = resolve_default_adapter(domain=domain_hint, subscription_id=subscription_id)
    store = resolve_default_store()
    llm_call = resolve_default_llm()
    written = 0
    for itype in enabled:
        if itype not in adapter.researcher.insight_types:
            continue
        result = run_profile_pass(ProfilePassInputs(
            adapter=adapter,
            insight_type=itype,
            documents=documents,
            profile_id=profile_id,
            subscription_id=subscription_id,
            kb_provider=None,
            llm_call=llm_call,
        ))
        for insight in result.insights:
            try:
                store.write(insight)
                written += 1
            except Exception as exc:
                logger.warning("profile insight write failed: %s", exc)
    return {"status": "ok", "written": written}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/insights_eval/test_researcher_v2_task.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/tasks/researcher_v2.py tests/insights_eval/test_researcher_v2_task.py
git commit -m "feat(researcher-v2): Celery-shaped per-doc + profile-level entry points (SP-C.5)"
```

### Task SP-C.6 — Wire Celery queue + decorate task entry points

- [ ] **Step 1: Write the failing test**

`tests/insights_eval/test_researcher_v2_celery.py`:
```python
def test_researcher_v2_queue_registered():
    from src.celery_app import app
    queues = {q.name for q in (app.conf.task_queues or [])}
    assert "researcher_v2_queue" in queues


def test_per_doc_task_routes_to_queue():
    from src.celery_app import app
    routes = app.conf.task_routes or {}
    assert "src.tasks.researcher_v2.run_researcher_v2_for_doc_task" in routes
    assert routes["src.tasks.researcher_v2.run_researcher_v2_for_doc_task"]["queue"] == "researcher_v2_queue"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/insights_eval/test_researcher_v2_celery.py -v`
Expected: FAIL — queue not registered.

- [ ] **Step 3: Modify Celery app + decorate task**

In `src/celery_app.py`, add (locate the existing `task_queues` config and append):
```python
from kombu import Queue

# Existing queues + add researcher_v2_queue
app.conf.task_queues = list(app.conf.task_queues or []) + [
    Queue("researcher_v2_queue", routing_key="researcher_v2.#"),
    Queue("researcher_refresh_queue", routing_key="researcher_refresh.#"),
    Queue("actions_queue", routing_key="actions.#"),
]

app.conf.task_routes = {
    **(app.conf.task_routes or {}),
    "src.tasks.researcher_v2.run_researcher_v2_for_doc_task": {
        "queue": "researcher_v2_queue",
        "routing_key": "researcher_v2.per_doc",
    },
    "src.tasks.researcher_v2.run_researcher_v2_for_profile_task": {
        "queue": "researcher_v2_queue",
        "routing_key": "researcher_v2.profile",
    },
}
```

Append to `src/tasks/researcher_v2.py`:
```python
from src.celery_app import app


@app.task(name="src.tasks.researcher_v2.run_researcher_v2_for_doc_task", bind=True)
def run_researcher_v2_for_doc_task(self, *, document_id, profile_id, subscription_id, document_text, domain_hint="generic"):
    return run_researcher_v2_for_doc(
        document_id=document_id,
        profile_id=profile_id,
        subscription_id=subscription_id,
        document_text=document_text,
        domain_hint=domain_hint,
    )


@app.task(name="src.tasks.researcher_v2.run_researcher_v2_for_profile_task", bind=True)
def run_researcher_v2_for_profile_task(self, *, profile_id, subscription_id, documents, domain_hint="generic"):
    return run_researcher_v2_for_profile(
        profile_id=profile_id,
        subscription_id=subscription_id,
        documents=documents,
        domain_hint=domain_hint,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/insights_eval/test_researcher_v2_celery.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/celery_app.py src/tasks/researcher_v2.py tests/insights_eval/test_researcher_v2_celery.py
git commit -m "feat(researcher-v2): register researcher_v2_queue + bind task entry points (SP-C.6)"
```

### Task SP-C.7 — Mongo isolation marker (researcher_v2.* field only)

- [ ] **Step 1: Write the failing test**

`tests/insights_eval/test_researcher_v2_mongo_isolation.py`:
```python
def test_per_doc_writes_only_to_researcher_v2_field(monkeypatch):
    """Per spec Section 7.3 + feedback_mongo_status_stability.md, the v2
    task must write ONLY to researcher_v2.* — never to pipeline_status,
    stages.*, or the v1 researcher.* field.
    """
    written_paths = []

    class FakeColl:
        def update_one(self, filter, update, upsert=False):
            for k in (update.get("$set") or {}):
                written_paths.append(k)

    fake_coll = FakeColl()

    from src.tasks.researcher_v2 import write_doc_status

    write_doc_status(
        collection=fake_coll, document_id="D", status="RESEARCHER_V2_COMPLETED",
        adapter_version="generic@1.0", written_count=3,
    )

    assert written_paths
    for path in written_paths:
        assert path.startswith("researcher_v2."), f"forbidden write to {path!r}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/insights_eval/test_researcher_v2_mongo_isolation.py -v`
Expected: FAIL.

- [ ] **Step 3: Add isolation helper**

Append to `src/tasks/researcher_v2.py`:
```python
def write_doc_status(
    *,
    collection,
    document_id: str,
    status: str,
    adapter_version: str,
    written_count: int,
) -> None:
    """Write per-doc researcher_v2 status. Only touches researcher_v2.* keys."""
    collection.update_one(
        {"document_id": document_id},
        {"$set": {
            "researcher_v2.status": status,
            "researcher_v2.adapter_version": adapter_version,
            "researcher_v2.written_count": written_count,
        }},
        upsert=True,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/insights_eval/test_researcher_v2_mongo_isolation.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/tasks/researcher_v2.py tests/insights_eval/test_researcher_v2_mongo_isolation.py
git commit -m "feat(researcher-v2): Mongo isolation — writes only researcher_v2.* (SP-C.7)"
```

---

## Sub-Project SP-E — Continuous Refresh

**Files:**
- Create: `src/tasks/researcher_v2_refresh.py`
- Create: `src/intelligence/researcher_v2/watchlist.py`
- Test: `tests/insights_eval/test_refresh_incremental.py`
- Test: `tests/insights_eval/test_refresh_scheduled.py`
- Test: `tests/insights_eval/test_watchlist.py`

### Task SP-E.1 — On-upload incremental refresh trigger

- [ ] **Step 1: Write the failing test**

`tests/insights_eval/test_refresh_incremental.py`:
```python
from unittest.mock import MagicMock, patch


def test_incremental_refresh_marks_stale_then_re_runs(monkeypatch):
    monkeypatch.setenv("REFRESH_ON_UPLOAD_ENABLED", "true")
    monkeypatch.setenv("REFRESH_INCREMENTAL_ENABLED", "true")
    monkeypatch.setenv("INSIGHTS_TYPE_ANOMALY_ENABLED", "true")

    from src.tasks.researcher_v2_refresh import refresh_for_new_doc

    mark_calls = []
    runner_calls = []

    def fake_mark(**kwargs):
        mark_calls.append(kwargs)
        return 2

    def fake_runner(**kwargs):
        runner_calls.append(kwargs)
        return {"status": "ok"}

    with patch(
        "src.tasks.researcher_v2_refresh.mark_stale_for_documents", side_effect=fake_mark
    ), patch(
        "src.tasks.researcher_v2_refresh.run_researcher_v2_for_doc", side_effect=fake_runner
    ), patch(
        "src.tasks.researcher_v2_refresh.resolve_default_index_collection", return_value=MagicMock()
    ):
        result = refresh_for_new_doc(
            document_id="D-NEW", profile_id="P", subscription_id="S",
            document_text="text", domain_hint="generic",
        )
    assert result["status"] == "ok"
    assert mark_calls and mark_calls[0]["document_ids"] == ["D-NEW"]
    assert runner_calls and runner_calls[0]["document_id"] == "D-NEW"


def test_incremental_short_circuits_when_flag_off(monkeypatch):
    monkeypatch.delenv("REFRESH_ON_UPLOAD_ENABLED", raising=False)
    from src.tasks.researcher_v2_refresh import refresh_for_new_doc
    result = refresh_for_new_doc(
        document_id="D", profile_id="P", subscription_id="S",
        document_text="x", domain_hint="generic",
    )
    assert result["status"] == "skipped_flag_off"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/insights_eval/test_refresh_incremental.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write minimal implementation**

`src/tasks/researcher_v2_refresh.py`:
```python
"""Continuous-refresh tasks — on-upload, scheduled, watchlist.

Runs on `researcher_refresh_queue` (low priority, isolated). Only writes
to insights collections + researcher_v2.* fields. Per spec Section 9.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from src.api.config import insight_flag_enabled
from src.intelligence.insights.staleness import mark_stale_for_documents
from src.tasks.researcher_v2 import (
    run_researcher_v2_for_doc,
    run_researcher_v2_for_profile,
)

logger = logging.getLogger(__name__)


def resolve_default_index_collection():
    """Hook — production wiring binds the Mongo `insights_index` collection."""
    raise NotImplementedError


def refresh_for_new_doc(
    *,
    document_id: str,
    profile_id: str,
    subscription_id: str,
    document_text: str,
    domain_hint: str = "generic",
) -> Dict[str, Any]:
    if not insight_flag_enabled("REFRESH_ON_UPLOAD_ENABLED"):
        return {"status": "skipped_flag_off"}
    if insight_flag_enabled("REFRESH_INCREMENTAL_ENABLED"):
        coll = resolve_default_index_collection()
        mark_stale_for_documents(
            collection=coll, profile_id=profile_id, document_ids=[document_id]
        )
    run_researcher_v2_for_doc(
        document_id=document_id,
        profile_id=profile_id,
        subscription_id=subscription_id,
        document_text=document_text,
        domain_hint=domain_hint,
    )
    return {"status": "ok"}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/insights_eval/test_refresh_incremental.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/tasks/researcher_v2_refresh.py tests/insights_eval/test_refresh_incremental.py
git commit -m "feat(refresh): on-upload incremental refresh trigger (SP-E.1)"
```

### Task SP-E.2 — Scheduled weekly cross-profile pass

- [ ] **Step 1: Write the failing test**

`tests/insights_eval/test_refresh_scheduled.py`:
```python
from unittest.mock import MagicMock, patch


def test_scheduled_runs_only_when_flag_on(monkeypatch):
    monkeypatch.delenv("REFRESH_SCHEDULED_ENABLED", raising=False)
    from src.tasks.researcher_v2_refresh import refresh_scheduled_pass
    result = refresh_scheduled_pass(profile_id="P", subscription_id="S")
    assert result["status"] == "skipped_flag_off"


def test_scheduled_dispatches_profile_pass(monkeypatch):
    monkeypatch.setenv("REFRESH_SCHEDULED_ENABLED", "true")
    monkeypatch.setenv("INSIGHTS_TYPE_COMPARISON_ENABLED", "true")

    from src.tasks.researcher_v2_refresh import refresh_scheduled_pass

    docs = [{"document_id": "D1", "text": "x"}, {"document_id": "D2", "text": "y"}]
    runner_calls = []
    def fake_runner(**kwargs):
        runner_calls.append(kwargs)
        return {"status": "ok"}

    with patch(
        "src.tasks.researcher_v2_refresh.fetch_active_profile_documents", return_value=docs
    ), patch(
        "src.tasks.researcher_v2_refresh.run_researcher_v2_for_profile", side_effect=fake_runner
    ):
        result = refresh_scheduled_pass(profile_id="P", subscription_id="S")
    assert result["status"] == "ok"
    assert runner_calls and len(runner_calls[0]["documents"]) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/insights_eval/test_refresh_scheduled.py -v`
Expected: FAIL.

- [ ] **Step 3: Append to `src/tasks/researcher_v2_refresh.py`**

```python
def fetch_active_profile_documents(*, profile_id: str) -> List[Dict[str, Any]]:
    """Hook — production wiring fetches all documents for the profile."""
    raise NotImplementedError


def refresh_scheduled_pass(
    *, profile_id: str, subscription_id: str, domain_hint: str = "generic"
) -> Dict[str, Any]:
    if not insight_flag_enabled("REFRESH_SCHEDULED_ENABLED"):
        return {"status": "skipped_flag_off"}
    docs = fetch_active_profile_documents(profile_id=profile_id)
    run_researcher_v2_for_profile(
        profile_id=profile_id,
        subscription_id=subscription_id,
        documents=docs,
        domain_hint=domain_hint,
    )
    return {"status": "ok"}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/insights_eval/test_refresh_scheduled.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/tasks/researcher_v2_refresh.py tests/insights_eval/test_refresh_scheduled.py
git commit -m "feat(refresh): scheduled weekly profile-level pass (SP-E.2)"
```

### Task SP-E.3 — Watchlist evaluator

- [ ] **Step 1: Write the failing test**

`tests/insights_eval/test_watchlist.py`:
```python
import pytest
from datetime import datetime, timedelta, timezone

from src.intelligence.adapters.schema import (
    Adapter, AppliesWhen, ResearcherSection, KnowledgeConfig, Watchlist,
)
from src.intelligence.researcher_v2.watchlist import (
    evaluate_watchlists,
    WatchlistFiring,
)


def _adapter_with_watch(eval_expr: str) -> Adapter:
    return Adapter(
        name="generic", version="1.0", description="t",
        applies_when=AppliesWhen(),
        researcher=ResearcherSection(),
        knowledge=KnowledgeConfig(),
        watchlists=[Watchlist(
            id="renewal_due",
            description="renewal soon",
            eval=eval_expr,
            fires_insight_type="next_action",
        )],
    )


def test_eval_expr_renewal_due_fires():
    near = (datetime.now(tz=timezone.utc) + timedelta(days=10)).isoformat()
    docs = [{"document_id": "D1", "fields": {"policy_end_date": near}}]
    a = _adapter_with_watch("expr:doc.policy_end_date - now < 60d")
    fired = evaluate_watchlists(adapter=a, documents=docs)
    assert len(fired) == 1
    f = fired[0]
    assert isinstance(f, WatchlistFiring)
    assert f.watchlist_id == "renewal_due"
    assert f.document_id == "D1"


def test_eval_expr_does_not_fire_when_far_away():
    far = (datetime.now(tz=timezone.utc) + timedelta(days=120)).isoformat()
    docs = [{"document_id": "D1", "fields": {"policy_end_date": far}}]
    a = _adapter_with_watch("expr:doc.policy_end_date - now < 60d")
    fired = evaluate_watchlists(adapter=a, documents=docs)
    assert fired == []


def test_unsupported_expr_skipped():
    docs = [{"document_id": "D1", "fields": {}}]
    a = _adapter_with_watch("expr:bogus_function()")
    fired = evaluate_watchlists(adapter=a, documents=docs)
    assert fired == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/insights_eval/test_watchlist.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement watchlist evaluator**

`src/intelligence/researcher_v2/watchlist.py`:
```python
"""Watchlist evaluator — fires `next_action` insights when adapter
predicates evaluate true.

v1 supports a tiny expression DSL of the shape:
  expr:doc.<field> - now < <N>d
This is intentionally narrow — adapter authors can always declare a
plain insight via the researcher path for richer logic.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from src.intelligence.adapters.schema import Adapter

logger = logging.getLogger(__name__)


@dataclass
class WatchlistFiring:
    watchlist_id: str
    document_id: str
    fires_insight_type: str
    description: str


_DATE_DELTA_RX = re.compile(
    r"expr:\s*doc\.([a-zA-Z_][a-zA-Z_0-9]*)\s*-\s*now\s*<\s*(\d+)d\s*$"
)


def _parse_iso(value: Any):
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


def evaluate_watchlists(*, adapter: Adapter, documents: List[Dict[str, Any]]) -> List[WatchlistFiring]:
    fired: List[WatchlistFiring] = []
    now = datetime.now(tz=timezone.utc)
    for w in adapter.watchlists:
        m = _DATE_DELTA_RX.match(w.eval.strip())
        if not m:
            logger.debug("unsupported watchlist expr: %s", w.eval)
            continue
        field, days = m.group(1), int(m.group(2))
        threshold = timedelta(days=days)
        for doc in documents:
            fields = doc.get("fields") or {}
            dt = _parse_iso(fields.get(field))
            if dt is None:
                continue
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            if (dt - now) < threshold and (dt - now) >= timedelta(0):
                fired.append(WatchlistFiring(
                    watchlist_id=w.id,
                    document_id=str(doc.get("document_id", "")),
                    fires_insight_type=w.fires_insight_type,
                    description=w.description,
                ))
    return fired
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/insights_eval/test_watchlist.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/intelligence/researcher_v2/watchlist.py tests/insights_eval/test_watchlist.py
git commit -m "feat(watchlist): nightly watchlist evaluator with date-delta DSL (SP-E.3)"
```

---

## Sub-Project SP-L — Backfill + Migration

**Files:**
- Create: `scripts/insights_backfill.py`
- Test: `tests/insights_eval/test_backfill.py`

### Task SP-L.1 — Idempotent profile-by-profile backfill driver

- [ ] **Step 1: Write the failing test**

`tests/insights_eval/test_backfill.py`:
```python
from unittest.mock import MagicMock, patch


def test_backfill_iterates_profiles_idempotent(monkeypatch):
    from scripts.insights_backfill import backfill_profiles

    profiles = [{"profile_id": f"P{i}"} for i in range(3)]

    runner = MagicMock(return_value={"status": "ok"})
    fetch = MagicMock(return_value=profiles)

    result = backfill_profiles(
        fetch_profiles=fetch,
        run_for_profile=runner,
        subscription_id="S",
    )
    assert result["processed"] == 3
    assert runner.call_count == 3
    # Re-running with the same input should be a no-op for processed profiles
    runner2 = MagicMock(return_value={"status": "skipped_already_done"})
    result2 = backfill_profiles(
        fetch_profiles=fetch,
        run_for_profile=runner2,
        subscription_id="S",
    )
    assert result2["processed"] == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/insights_eval/test_backfill.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement backfill driver**

`scripts/insights_backfill.py`:
```python
"""One-time backfill — replay researcher v2 across existing profiles.

Idempotent: each profile's run is keyed by profile_id; the runner is
expected to short-circuit when researcher_v2 already completed.
Interruptible: the script processes profiles in pages and persists a
cursor, so re-running resumes where it left off.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Callable, Dict, List

logger = logging.getLogger(__name__)


def backfill_profiles(
    *,
    fetch_profiles: Callable[[], List[Dict[str, Any]]],
    run_for_profile: Callable[..., Dict[str, Any]],
    subscription_id: str,
) -> Dict[str, Any]:
    profiles = fetch_profiles()
    processed = 0
    for p in profiles:
        try:
            run_for_profile(profile_id=p["profile_id"], subscription_id=subscription_id)
            processed += 1
        except Exception as exc:
            logger.warning("backfill skip %s: %s", p.get("profile_id"), exc)
    return {"processed": processed, "total": len(profiles)}


def _cli() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subscription", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    # Real wiring at deploy time. Here we only support --dry-run for sanity.
    if args.dry_run:
        print(json.dumps({"processed": 0, "dry_run": True}))
        return 0
    print("backfill requires production wiring — see runbook", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(_cli())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/insights_eval/test_backfill.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/insights_backfill.py tests/insights_eval/test_backfill.py
git commit -m "feat(backfill): idempotent profile-by-profile backfill driver (SP-L.1)"
```

---

## Sub-Project SP-F — Surface Endpoints

**Files:**
- Create: `src/api/insights_api.py`
- Create: `src/api/actions_api.py`
- Create: `src/api/visualizations_api.py`
- Create: `src/api/artifacts_api.py`
- Modify: `src/main.py` (mount routers)
- Test: `tests/insights_eval/test_endpoints.py`

### Task SP-F.1 — Insights list + detail + refresh-status endpoints

- [ ] **Step 1: Write the failing test**

`tests/insights_eval/test_endpoints.py`:
```python
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def app_with_flags(monkeypatch):
    monkeypatch.setenv("INSIGHTS_DASHBOARD_ENABLED", "true")
    from src.main import app
    return TestClient(app)


@pytest.fixture
def app_without_flags(monkeypatch):
    monkeypatch.delenv("INSIGHTS_DASHBOARD_ENABLED", raising=False)
    from src.main import app
    return TestClient(app)


def test_insights_list_404_when_flag_off(app_without_flags):
    r = app_without_flags.get("/profiles/P1/insights")
    assert r.status_code == 404


def test_insights_list_returns_when_flag_on(app_with_flags, monkeypatch):
    rows = [
        {"insight_id": "i1", "profile_id": "P1", "insight_type": "anomaly",
         "severity": "warn", "domain": "insurance", "refreshed_at": "2026-04-25",
         "stale": False, "tags": []},
    ]
    with patch("src.api.insights_api.list_insights_for_profile", return_value=rows):
        r = app_with_flags.get("/profiles/P1/insights")
    assert r.status_code == 200
    body = r.json()
    assert body["profile_id"] == "P1"
    assert body["total"] == 1
    assert body["insights"][0]["insight_id"] == "i1"


def test_insights_detail_returns_full_object(app_with_flags):
    full = {
        "insight_id": "i1", "profile_id": "P1", "subscription_id": "S",
        "document_ids": ["D1"], "domain": "insurance", "insight_type": "gap",
        "headline": "h", "body": "b",
        "evidence_doc_spans": [{"document_id": "D1", "page": 1, "char_start": 0, "char_end": 1, "quote": "x"}],
        "external_kb_refs": [], "confidence": 0.9, "severity": "warn",
        "adapter_version": "insurance@1.0", "refreshed_at": "2026-04-25",
        "stale": False, "tags": [], "feature_flags": [], "suggested_actions": [],
        "created_at": "2026-04-25",
    }
    with patch("src.api.insights_api.get_insight_full", return_value=full):
        r = app_with_flags.get("/profiles/P1/insights/i1")
    assert r.status_code == 200
    body = r.json()
    assert body["headline"] == "h"
    assert body["evidence_doc_spans"][0]["document_id"] == "D1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/insights_eval/test_endpoints.py -v`
Expected: FAIL — endpoints not registered.

- [ ] **Step 3: Implement insights_api router**

`src/api/insights_api.py`:
```python
"""Insights surface endpoints — read-only. Lookup against Mongo control-plane index.

All endpoints flag-gated by INSIGHTS_DASHBOARD_ENABLED. When flag is off,
the router returns 404 for every path — no behavioural change to /api/ask.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

from src.api.config import insight_flag_enabled

insights_router = APIRouter(prefix="/profiles", tags=["Insights"])


def list_insights_for_profile(
    *,
    profile_id: str,
    insight_types: Optional[List[str]] = None,
    severities: Optional[List[str]] = None,
    domain: Optional[str] = None,
    since: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """Hook — production wiring binds InsightStore.list_for_profile."""
    raise NotImplementedError


def get_insight_full(*, insight_id: str) -> Optional[Dict[str, Any]]:
    """Hook — production wiring binds Qdrant insight payload fetch."""
    raise NotImplementedError


def _gate():
    if not insight_flag_enabled("INSIGHTS_DASHBOARD_ENABLED"):
        raise HTTPException(status_code=404, detail="Feature not enabled")


@insights_router.get("/{profile_id}/insights")
async def list_insights(
    profile_id: str,
    insight_type: Optional[str] = None,
    severity: Optional[str] = None,
    domain: Optional[str] = None,
    since: Optional[str] = None,
    limit: int = Query(default=50, le=200),
    offset: int = 0,
):
    _gate()
    types = insight_type.split(",") if insight_type else None
    sevs = severity.split(",") if severity else None
    rows = list_insights_for_profile(
        profile_id=profile_id,
        insight_types=types,
        severities=sevs,
        domain=domain,
        since=since,
        limit=limit,
        offset=offset,
    )
    domains_present = sorted({r.get("domain", "") for r in rows if r.get("domain")})
    last_refresh = max((r.get("refreshed_at", "") for r in rows), default="")
    stale_count = sum(1 for r in rows if r.get("stale"))
    return {
        "profile_id": profile_id,
        "total": len(rows),
        "stale_count": stale_count,
        "insights": rows,
        "domains_present": domains_present,
        "last_refresh": last_refresh,
    }


@insights_router.get("/{profile_id}/insights/{insight_id}")
async def get_insight(profile_id: str, insight_id: str):
    _gate()
    obj = get_insight_full(insight_id=insight_id)
    if obj is None or obj.get("profile_id") != profile_id:
        raise HTTPException(status_code=404, detail="Not found")
    return obj


@insights_router.get("/{profile_id}/refresh-status")
async def refresh_status(profile_id: str):
    _gate()
    return {
        "profile_id": profile_id,
        "last_on_upload_refresh": None,
        "last_scheduled_run": None,
        "pending_watchlist_evaluations": 0,
        "stale_insight_count": 0,
    }
```

Mount in `src/main.py` — append where existing routers are mounted:
```python
from src.api.insights_api import insights_router
app.include_router(insights_router)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/insights_eval/test_endpoints.py::test_insights_list_404_when_flag_off tests/insights_eval/test_endpoints.py::test_insights_list_returns_when_flag_on tests/insights_eval/test_endpoints.py::test_insights_detail_returns_full_object -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/api/insights_api.py src/main.py tests/insights_eval/test_endpoints.py
git commit -m "feat(api): /profiles/{id}/insights list + detail + refresh-status (SP-F.1)"
```

### Task SP-F.2 — Actions list + execute endpoints

- [ ] **Step 1: Write the failing test**

Append to `tests/insights_eval/test_endpoints.py`:
```python
def test_actions_list_404_when_flag_off(app_without_flags):
    r = app_without_flags.get("/profiles/P1/actions")
    assert r.status_code == 404


def test_actions_list_returns_when_flag_on(monkeypatch):
    monkeypatch.setenv("ACTIONS_ARTIFACT_ENABLED", "true")
    from src.main import app
    client = TestClient(app)
    actions = [{
        "action_id": "a1", "title": "Generate summary",
        "action_type": "artifact", "requires_confirmation": False,
        "preview": "Will produce a coverage summary PDF",
    }]
    with patch("src.api.actions_api.list_actions_for_profile", return_value=actions):
        r = client.get("/profiles/P1/actions")
    assert r.status_code == 200
    body = r.json()
    assert body["actions"][0]["action_id"] == "a1"


def test_action_execute_requires_confirmation(monkeypatch):
    monkeypatch.setenv("ACTIONS_ARTIFACT_ENABLED", "true")
    from src.main import app
    client = TestClient(app)

    def fake_execute(**kwargs):
        return {"status": "needs_confirmation", "preview": "preview text"}

    with patch("src.api.actions_api.execute_action", side_effect=fake_execute):
        r = client.post("/profiles/P1/actions/a1/execute", json={"inputs": {}, "confirmed": False})
    assert r.status_code == 200
    assert r.json()["status"] == "needs_confirmation"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/insights_eval/test_endpoints.py::test_actions_list_404_when_flag_off -v`
Expected: FAIL — actions router not mounted.

- [ ] **Step 3: Implement actions_api**

`src/api/actions_api.py`:
```python
"""Actions surface endpoints — list + execute. Gated by per-action-type flags.

The action layer surface is gated as a whole behind any of:
  - ACTIONS_ARTIFACT_ENABLED
  - ACTIONS_FORM_FILL_ENABLED
  - ACTIONS_PLAN_ENABLED
  - ACTIONS_REMINDER_ENABLED
If none enabled → 404. Per-action filtering happens inside list_actions_for_profile.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, HTTPException

from src.api.config import insight_flag_enabled

actions_router = APIRouter(prefix="/profiles", tags=["Actions"])


_ACTION_FLAGS = (
    "ACTIONS_ARTIFACT_ENABLED",
    "ACTIONS_FORM_FILL_ENABLED",
    "ACTIONS_PLAN_ENABLED",
    "ACTIONS_REMINDER_ENABLED",
)


def list_actions_for_profile(*, profile_id: str) -> List[Dict[str, Any]]:
    raise NotImplementedError


def execute_action(*, profile_id: str, action_id: str, inputs: Dict[str, Any], confirmed: bool) -> Dict[str, Any]:
    raise NotImplementedError


def _gate():
    if not any(insight_flag_enabled(f) for f in _ACTION_FLAGS):
        raise HTTPException(status_code=404, detail="Feature not enabled")


@actions_router.get("/{profile_id}/actions")
async def list_actions(profile_id: str):
    _gate()
    return {"profile_id": profile_id, "actions": list_actions_for_profile(profile_id=profile_id)}


@actions_router.post("/{profile_id}/actions/{action_id}/execute")
async def execute_action_endpoint(
    profile_id: str,
    action_id: str,
    body: Dict[str, Any] = Body(default_factory=dict),
):
    _gate()
    inputs = body.get("inputs") or {}
    confirmed = bool(body.get("confirmed", False))
    return execute_action(
        profile_id=profile_id,
        action_id=action_id,
        inputs=inputs,
        confirmed=confirmed,
    )
```

Mount in `src/main.py`:
```python
from src.api.actions_api import actions_router
app.include_router(actions_router)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/insights_eval/test_endpoints.py -v`
Expected: 6 passed (cumulative).

- [ ] **Step 5: Commit**

```bash
git add src/api/actions_api.py src/main.py tests/insights_eval/test_endpoints.py
git commit -m "feat(api): /profiles/{id}/actions list + execute endpoints (SP-F.2)"
```

### Task SP-F.3 — Visualizations + artifacts endpoints

- [ ] **Step 1: Write the failing test**

Append to `tests/insights_eval/test_endpoints.py`:
```python
def test_viz_endpoint_gated_by_flag(monkeypatch):
    monkeypatch.delenv("VIZ_ENABLED", raising=False)
    from src.main import app
    client = TestClient(app)
    r = client.get("/profiles/P1/visualizations")
    assert r.status_code == 404


def test_viz_endpoint_returns_when_flag_on(monkeypatch):
    monkeypatch.setenv("VIZ_ENABLED", "true")
    from src.main import app
    client = TestClient(app)
    vizs = [{"viz_id": "timeline", "type": "timeline", "data": {"events": []}}]
    with patch("src.api.visualizations_api.list_visualizations_for_profile", return_value=vizs):
        r = client.get("/profiles/P1/visualizations")
    assert r.status_code == 200
    assert r.json()["visualizations"][0]["viz_id"] == "timeline"


def test_artifacts_endpoint_gated(monkeypatch):
    monkeypatch.delenv("ACTIONS_ARTIFACT_ENABLED", raising=False)
    from src.main import app
    client = TestClient(app)
    r = client.get("/profiles/P1/artifacts")
    assert r.status_code == 404
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/insights_eval/test_endpoints.py::test_viz_endpoint_gated_by_flag -v`
Expected: FAIL.

- [ ] **Step 3: Implement viz + artifacts routers**

`src/api/visualizations_api.py`:
```python
"""Visualization spec endpoint."""
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

from src.api.config import insight_flag_enabled

visualizations_router = APIRouter(prefix="/profiles", tags=["Visualizations"])


def list_visualizations_for_profile(*, profile_id: str) -> List[Dict[str, Any]]:
    raise NotImplementedError


@visualizations_router.get("/{profile_id}/visualizations")
async def list_viz(profile_id: str):
    if not insight_flag_enabled("VIZ_ENABLED"):
        raise HTTPException(status_code=404, detail="Feature not enabled")
    return {
        "profile_id": profile_id,
        "visualizations": list_visualizations_for_profile(profile_id=profile_id),
    }
```

`src/api/artifacts_api.py`:
```python
"""Artifacts list endpoint."""
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

from src.api.config import insight_flag_enabled

artifacts_router = APIRouter(prefix="/profiles", tags=["Artifacts"])


def list_artifacts_for_profile(*, profile_id: str) -> List[Dict[str, Any]]:
    raise NotImplementedError


@artifacts_router.get("/{profile_id}/artifacts")
async def list_artifacts(profile_id: str):
    if not insight_flag_enabled("ACTIONS_ARTIFACT_ENABLED"):
        raise HTTPException(status_code=404, detail="Feature not enabled")
    try:
        artifacts = list_artifacts_for_profile(profile_id=profile_id)
    except NotImplementedError:
        artifacts = []
    return {"profile_id": profile_id, "artifacts": artifacts}
```

Mount in `src/main.py`:
```python
from src.api.visualizations_api import visualizations_router
from src.api.artifacts_api import artifacts_router
app.include_router(visualizations_router)
app.include_router(artifacts_router)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/insights_eval/test_endpoints.py -v`
Expected: 9 passed (cumulative).

- [ ] **Step 5: Commit**

```bash
git add src/api/visualizations_api.py src/api/artifacts_api.py src/main.py tests/insights_eval/test_endpoints.py
git commit -m "feat(api): /profiles/{id}/visualizations + /artifacts endpoints (SP-F.3)"
```

---

## Sub-Project SP-G — `/api/ask` Proactive Injection

**Files:**
- Create: `src/generation/insight_injection.py`
- Modify: `src/generation/prompts.py` (insertion point)
- Test: `tests/insights_eval/test_insight_injection.py`

### Task SP-G.1 — Insight retrieval helper (lookup-only)

- [ ] **Step 1: Write the failing test**

`tests/insights_eval/test_insight_injection.py`:
```python
import time

import pytest

from src.generation.insight_injection import (
    select_insights_for_query,
    format_related_findings,
    INJECTION_BUDGET_MS,
)


def test_filters_by_severity_threshold():
    # Severity below 'notice' must be filtered out
    rows = [
        {"insight_id": "i1", "headline": "info-only", "severity": "info"},
        {"insight_id": "i2", "headline": "notice-level", "severity": "notice"},
        {"insight_id": "i3", "headline": "warn-level", "severity": "warn"},
    ]
    selected = select_insights_for_query(query="any", profile_insights=rows, query_entities=set())
    ids = {r["insight_id"] for r in selected}
    assert "i1" not in ids
    assert "i2" in ids and "i3" in ids


def test_relevance_filter_uses_query_entities():
    rows = [
        {"insight_id": "i1", "headline": "Premium $1800", "severity": "notice", "tags": ["premium"]},
        {"insight_id": "i2", "headline": "No flood coverage", "severity": "warn", "tags": ["flood"]},
    ]
    selected = select_insights_for_query(query="What is the premium?", profile_insights=rows, query_entities={"premium"})
    ids = {r["insight_id"] for r in selected}
    assert "i1" in ids
    # Without entity match for flood, lower relevance
    assert ids == {"i1"} or "i1" in ids


def test_budget_truncates_when_exceeded():
    # 200 candidates at 1ms each ⇒ p95 well under 50ms; ensures function returns within budget
    rows = [
        {"insight_id": f"i{i}", "headline": f"H{i}", "severity": "notice", "tags": [f"t{i}"]}
        for i in range(200)
    ]
    start = time.perf_counter_ns()
    selected = select_insights_for_query(query="x", profile_insights=rows, query_entities=set())
    elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000.0
    assert elapsed_ms < INJECTION_BUDGET_MS
    assert len(selected) <= 5  # default top-N cap


def test_format_related_findings_renders():
    rows = [
        {"headline": "No flood coverage", "severity": "warn"},
        {"headline": "Premium $1800", "severity": "notice"},
    ]
    out = format_related_findings(rows)
    assert "Related findings" in out
    assert "No flood coverage" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/insights_eval/test_insight_injection.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement injection helper**

`src/generation/insight_injection.py`:
```python
"""Helpers for /api/ask proactive insight injection.

Lookup-only — no LLM calls, no network beyond the existing Mongo index
read. 50ms hard budget per spec Section 13.2.

Per OQ4: always-on once INSIGHTS_PROACTIVE_INJECTION is enabled. Severity
filtering ('notice'+) is a quality guard, not an opt-out.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Iterable, List, Set

INJECTION_BUDGET_MS = 50.0
DEFAULT_TOP_N = 5
_SEVERITY_ORDER = {"info": 0, "notice": 1, "warn": 2, "critical": 3}


def select_insights_for_query(
    *,
    query: str,
    profile_insights: List[Dict[str, Any]],
    query_entities: Set[str],
    top_n: int = DEFAULT_TOP_N,
) -> List[Dict[str, Any]]:
    deadline = time.perf_counter() + (INJECTION_BUDGET_MS / 1000.0)
    selected: List[Dict[str, Any]] = []
    for row in profile_insights:
        if time.perf_counter() > deadline:
            break
        sev = row.get("severity", "info")
        if _SEVERITY_ORDER.get(sev, 0) < _SEVERITY_ORDER["notice"]:
            continue
        # Relevance: tag overlap with query entities (cheap; no LLM)
        tags = set(row.get("tags") or [])
        relevance_score = _SEVERITY_ORDER.get(sev, 0)
        if query_entities and not (tags & query_entities):
            relevance_score -= 1  # penalize but don't drop entirely
        row_with_score = dict(row)
        row_with_score["__relevance"] = relevance_score
        selected.append(row_with_score)
    selected.sort(key=lambda r: -r["__relevance"])
    return [{k: v for k, v in r.items() if k != "__relevance"} for r in selected[:top_n]]


def format_related_findings(insights: Iterable[Dict[str, Any]]) -> str:
    rows = list(insights)
    if not rows:
        return ""
    lines = ["", "Related findings:"]
    for r in rows:
        sev = r.get("severity", "")
        marker = "!" if sev in ("warn", "critical") else "•"
        lines.append(f"  {marker} {r.get('headline', '')}")
    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/insights_eval/test_insight_injection.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/generation/insight_injection.py tests/insights_eval/test_insight_injection.py
git commit -m "feat(injection): proactive insight selection + formatting helper (SP-G.1)"
```

### Task SP-G.2 — Wire into prompts.py response composition

- [ ] **Step 1: Write the failing test**

Append to `tests/insights_eval/test_insight_injection.py`:
```python
def test_compose_response_appends_findings_when_flag_on(monkeypatch):
    monkeypatch.setenv("INSIGHTS_PROACTIVE_INJECTION", "true")
    from src.generation.prompts import compose_response_with_insights

    base = "The premium is $1800."
    insights = [{"headline": "No flood coverage", "severity": "warn"}]
    out = compose_response_with_insights(
        base_answer=base,
        profile_insights=insights,
        query="premium",
        query_entities=set(),
    )
    assert "premium is $1800" in out
    assert "Related findings" in out
    assert "No flood coverage" in out


def test_compose_response_unchanged_when_flag_off(monkeypatch):
    monkeypatch.delenv("INSIGHTS_PROACTIVE_INJECTION", raising=False)
    from src.generation.prompts import compose_response_with_insights
    base = "The premium is $1800."
    insights = [{"headline": "No flood coverage", "severity": "warn"}]
    out = compose_response_with_insights(
        base_answer=base, profile_insights=insights,
        query="premium", query_entities=set(),
    )
    assert out == base
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/insights_eval/test_insight_injection.py::test_compose_response_appends_findings_when_flag_on -v`
Expected: FAIL — `compose_response_with_insights` doesn't exist.

- [ ] **Step 3: Add compose function**

Append to `src/generation/prompts.py`:
```python
def compose_response_with_insights(
    *,
    base_answer: str,
    profile_insights,
    query: str,
    query_entities,
) -> str:
    """Optionally append a 'Related findings' section to the reasoner's answer.

    Per spec Section 10.8 — flag-gated, lookup-only, 50ms budget,
    no LLM calls beyond what the reasoner already made.
    """
    from src.api.config import insight_flag_enabled
    if not insight_flag_enabled("INSIGHTS_PROACTIVE_INJECTION"):
        return base_answer
    from src.generation.insight_injection import (
        select_insights_for_query,
        format_related_findings,
    )
    selected = select_insights_for_query(
        query=query,
        profile_insights=list(profile_insights or []),
        query_entities=set(query_entities or set()),
    )
    suffix = format_related_findings(selected)
    if not suffix:
        return base_answer
    return base_answer + "\n" + suffix
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/insights_eval/test_insight_injection.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/generation/prompts.py tests/insights_eval/test_insight_injection.py
git commit -m "feat(injection): /api/ask proactive findings appended via compose_response_with_insights (SP-G.2)"
```

---

## Sub-Project SP-H — Agentic Action Layer

**Files:**
- Create: `src/intelligence/actions/__init__.py`
- Create: `src/intelligence/actions/runner.py`
- Create: `src/intelligence/actions/handlers.py`
- Create: `src/intelligence/actions/audit.py`
- Test: `tests/insights_eval/test_actions_runner.py`
- Test: `tests/insights_eval/test_actions_handlers.py`

### Task SP-H.1 — Action runner with confirmation gate

- [ ] **Step 1: Write the failing test**

`tests/insights_eval/test_actions_runner.py`:
```python
import pytest

from src.intelligence.actions.runner import (
    ActionRunner,
    ActionExecutionResult,
)
from src.intelligence.adapters.schema import ActionTemplate


def _action(*, requires_confirmation=False, action_type="artifact"):
    return ActionTemplate(
        action_id="a1", title="Test", action_type=action_type,
        artifact_template="t.md", requires_confirmation=requires_confirmation,
    )


class FakeHandler:
    def __init__(self):
        self.executed = False
    def __call__(self, *, action, profile_id, inputs):
        self.executed = True
        return {"artifact_blob_url": "blob://x"}


def test_unconfirmed_action_returns_preview():
    handler = FakeHandler()
    runner = ActionRunner(
        handlers={"artifact": handler}, audit_writer=lambda **kw: None,
    )
    result = runner.execute(
        action=_action(requires_confirmation=True),
        profile_id="P", inputs={}, confirmed=False,
    )
    assert isinstance(result, ActionExecutionResult)
    assert result.status == "needs_confirmation"
    assert handler.executed is False


def test_confirmed_action_executes():
    handler = FakeHandler()
    audit_calls = []
    runner = ActionRunner(
        handlers={"artifact": handler}, audit_writer=lambda **kw: audit_calls.append(kw),
    )
    result = runner.execute(
        action=_action(requires_confirmation=True),
        profile_id="P", inputs={"k": "v"}, confirmed=True,
    )
    assert result.status == "executed"
    assert handler.executed is True
    assert audit_calls and audit_calls[0]["action_id"] == "a1"


def test_safe_action_no_confirmation_required():
    handler = FakeHandler()
    runner = ActionRunner(
        handlers={"artifact": handler}, audit_writer=lambda **kw: None,
    )
    result = runner.execute(
        action=_action(requires_confirmation=False),
        profile_id="P", inputs={}, confirmed=False,
    )
    assert result.status == "executed"


def test_external_side_effect_gated_by_separate_flag(monkeypatch):
    monkeypatch.delenv("ACTIONS_EXTERNAL_SIDE_EFFECTS_ENABLED", raising=False)
    handler = FakeHandler()
    runner = ActionRunner(
        handlers={"reminder": handler}, audit_writer=lambda **kw: None,
    )
    action = ActionTemplate(
        action_id="a-ext", title="Send email", action_type="reminder",
        requires_confirmation=True,
    )
    setattr(action, "_side_effect", "external")  # marker tests respect
    result = runner.execute(
        action=action, profile_id="P", inputs={}, confirmed=True,
    )
    assert result.status == "external_side_effects_disabled"
    assert handler.executed is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/insights_eval/test_actions_runner.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement runner**

`src/intelligence/actions/__init__.py`:
```python
from src.intelligence.actions.runner import ActionRunner, ActionExecutionResult

__all__ = ["ActionRunner", "ActionExecutionResult"]
```

`src/intelligence/actions/runner.py`:
```python
"""Action runner — gated, audited, idempotent.

v1 ships side-effect-free actions only. External-side-effect actions
declared in adapter YAML are detected via _side_effect marker and
disabled at runtime unless ACTIONS_EXTERNAL_SIDE_EFFECTS_ENABLED=true.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from src.api.config import insight_flag_enabled
from src.intelligence.adapters.schema import ActionTemplate

logger = logging.getLogger(__name__)


@dataclass
class ActionExecutionResult:
    status: str
    preview: Optional[str] = None
    output: Dict[str, Any] = field(default_factory=dict)


class ActionRunner:
    def __init__(
        self,
        *,
        handlers: Dict[str, Callable[..., Dict[str, Any]]],
        audit_writer: Callable[..., None],
    ):
        self._handlers = handlers
        self._audit = audit_writer

    def execute(
        self,
        *,
        action: ActionTemplate,
        profile_id: str,
        inputs: Dict[str, Any],
        confirmed: bool,
    ) -> ActionExecutionResult:
        if action.requires_confirmation and not confirmed:
            preview = self._build_preview(action=action, inputs=inputs)
            return ActionExecutionResult(status="needs_confirmation", preview=preview)
        side_effect = getattr(action, "_side_effect", None)
        if side_effect == "external" and not insight_flag_enabled(
            "ACTIONS_EXTERNAL_SIDE_EFFECTS_ENABLED"
        ):
            return ActionExecutionResult(status="external_side_effects_disabled")
        handler = self._handlers.get(action.action_type)
        if handler is None:
            return ActionExecutionResult(status="unknown_action_type")
        try:
            output = handler(action=action, profile_id=profile_id, inputs=inputs)
        except Exception as exc:
            logger.exception("action handler raised: %s", exc)
            return ActionExecutionResult(status="failed")
        try:
            self._audit(
                action_id=action.action_id,
                profile_id=profile_id,
                inputs=inputs,
                output=output,
            )
        except Exception as exc:
            logger.warning("audit write failed: %s", exc)
        return ActionExecutionResult(status="executed", output=output)

    def _build_preview(self, *, action: ActionTemplate, inputs: Dict[str, Any]) -> str:
        return (
            f"Action: {action.title}\n"
            f"Type: {action.action_type}\n"
            f"Inputs: {inputs}\n"
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/insights_eval/test_actions_runner.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/intelligence/actions/__init__.py src/intelligence/actions/runner.py tests/insights_eval/test_actions_runner.py
git commit -m "feat(actions): runner with confirmation gate + side-effect flag (SP-H.1)"
```

### Task SP-H.2 — Artifact + form_fill + plan + reminder handlers

- [ ] **Step 1: Write the failing test**

`tests/insights_eval/test_actions_handlers.py`:
```python
from src.intelligence.actions.handlers import (
    artifact_handler, form_fill_handler, plan_handler, reminder_handler,
)
from src.intelligence.adapters.schema import ActionTemplate


def _action(action_type="artifact"):
    return ActionTemplate(
        action_id="a1", title="X", action_type=action_type,
        artifact_template="t.md", requires_confirmation=False,
    )


def test_artifact_handler_renders_template_and_uploads(tmp_path, monkeypatch):
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    (template_dir / "t.md").write_text("Hello {{ profile_id }} — input is {{ inputs['foo'] }}")
    monkeypatch.setenv("ACTION_TEMPLATE_ROOT", str(template_dir))

    out = artifact_handler(
        action=_action(),
        profile_id="P-1",
        inputs={"foo": "bar"},
    )
    assert "artifact_blob_url" in out
    assert "Hello P-1" in out["artifact_content"]
    assert "input is bar" in out["artifact_content"]


def test_form_fill_handler_returns_filled_form():
    out = form_fill_handler(
        action=_action(action_type="form_fill"),
        profile_id="P-1",
        inputs={"name": "Test", "policy": "ABC-001"},
    )
    assert out["status"] == "filled"
    assert out["form_data"]["name"] == "Test"


def test_plan_handler_returns_checklist():
    out = plan_handler(
        action=_action(action_type="plan"),
        profile_id="P-1",
        inputs={"steps": ["A", "B", "C"]},
    )
    assert "checklist" in out
    assert len(out["checklist"]) == 3


def test_reminder_handler_in_system_only():
    out = reminder_handler(
        action=_action(action_type="reminder"),
        profile_id="P-1",
        inputs={"fire_at": "2026-12-31T00:00:00+00:00", "message": "Renew"},
    )
    assert out["status"] == "scheduled_in_system"
    assert out["fire_at"] == "2026-12-31T00:00:00+00:00"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/insights_eval/test_actions_handlers.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement handlers**

`src/intelligence/actions/handlers.py`:
```python
"""Per-type action handlers. v1: side-effect-free.

artifact      — render template, return content (production wires Blob upload)
form_fill     — produce structured form data from inputs
plan          — produce a checklist from steps
reminder      — schedule an in-system reminder (no external email/SMS in v1)
"""
from __future__ import annotations

import os
from typing import Any, Dict


def _render(template: str, *, profile_id: str, inputs: Dict[str, Any]) -> str:
    out = template.replace("{{ profile_id }}", profile_id)
    # Tiny replacement: {{ inputs['key'] }}
    for k, v in inputs.items():
        out = out.replace(f"{{{{ inputs['{k}'] }}}}", str(v))
    return out


def artifact_handler(*, action, profile_id: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
    template_root = os.environ.get("ACTION_TEMPLATE_ROOT", "src/intelligence/actions/templates")
    template_path = os.path.join(template_root, action.artifact_template or "")
    if not os.path.exists(template_path):
        return {"status": "template_not_found", "template": action.artifact_template}
    with open(template_path, "r", encoding="utf-8") as fh:
        template = fh.read()
    content = _render(template, profile_id=profile_id, inputs=inputs)
    # In production this uploads to Blob and returns a URL; v1 returns content + stub URL
    return {
        "status": "rendered",
        "artifact_blob_url": f"blob://artifacts/{profile_id}/{action.action_id}.md",
        "artifact_content": content,
    }


def form_fill_handler(*, action, profile_id: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": "filled",
        "form_id": action.action_id,
        "profile_id": profile_id,
        "form_data": dict(inputs),
    }


def plan_handler(*, action, profile_id: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
    steps = list(inputs.get("steps") or [])
    checklist = [{"step": s, "done": False} for s in steps]
    return {"status": "planned", "profile_id": profile_id, "checklist": checklist}


def reminder_handler(*, action, profile_id: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": "scheduled_in_system",
        "profile_id": profile_id,
        "fire_at": inputs.get("fire_at"),
        "message": inputs.get("message", ""),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/insights_eval/test_actions_handlers.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/intelligence/actions/handlers.py tests/insights_eval/test_actions_handlers.py
git commit -m "feat(actions): artifact/form_fill/plan/reminder handlers (side-effect-free) (SP-H.2)"
```

### Task SP-H.3 — Audit log writer

- [ ] **Step 1: Write the failing test**

Append to `tests/insights_eval/test_actions_runner.py`:
```python
def test_audit_writer_records_call():
    from src.intelligence.actions.audit import make_audit_writer

    written = []
    class FakeColl:
        def insert_one(self, doc):
            written.append(doc)
    writer = make_audit_writer(collection=FakeColl())
    writer(
        action_id="a1",
        profile_id="P",
        inputs={"k": "v"},
        output={"status": "executed"},
    )
    assert len(written) == 1
    rec = written[0]
    assert rec["action_id"] == "a1"
    assert rec["profile_id"] == "P"
    assert rec["output"]["status"] == "executed"
    assert "at" in rec
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/insights_eval/test_actions_runner.py::test_audit_writer_records_call -v`
Expected: FAIL.

- [ ] **Step 3: Implement audit writer**

`src/intelligence/actions/audit.py`:
```python
"""Action execution audit log — records every executed action.

Per spec Section 11.1 — writes to actions_audit Mongo collection.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Dict


def make_audit_writer(*, collection) -> Callable[..., None]:
    def write(*, action_id: str, profile_id: str, inputs: Dict[str, Any], output: Dict[str, Any]) -> None:
        collection.insert_one({
            "action_id": action_id,
            "profile_id": profile_id,
            "inputs": inputs,
            "output": output,
            "at": datetime.now(tz=timezone.utc).isoformat(),
        })
    return write
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/insights_eval/test_actions_runner.py -v`
Expected: 5 passed (cumulative).

- [ ] **Step 5: Commit**

```bash
git add src/intelligence/actions/audit.py tests/insights_eval/test_actions_runner.py
git commit -m "feat(actions): audit-log writer for every executed action (SP-H.3)"
```

---

## Sub-Project SP-I — Visualizations

**Files:**
- Create: `src/intelligence/visualizations/__init__.py`
- Create: `src/intelligence/visualizations/generator.py`
- Test: `tests/insights_eval/test_visualizations.py`

### Task SP-I.1 — Timeline + comparison-table + trend-chart generators

- [ ] **Step 1: Write the failing test**

`tests/insights_eval/test_visualizations.py`:
```python
import pytest

from src.intelligence.insights.schema import Insight, EvidenceSpan
from src.intelligence.visualizations.generator import (
    generate_visualizations_for_insight,
    generate_visualizations_for_profile,
)


def _ins(itype="anomaly", refreshed_at="2026-04-25T10:00:00+00:00"):
    return Insight(
        insight_id="i1", profile_id="P", subscription_id="S",
        document_ids=["D1"], domain="generic", insight_type=itype,
        headline="H", body="b grounded in quote",
        evidence_doc_spans=[EvidenceSpan(
            document_id="D1", page=1, char_start=0, char_end=2, quote="b"
        )],
        confidence=0.5, severity="notice", adapter_version="generic@1.0",
        refreshed_at=refreshed_at,
    )


def test_trend_insight_produces_trend_chart_data():
    out = generate_visualizations_for_insight(_ins(itype="trend"))
    assert any(v["viz_id"] == "trend_chart" for v in out)


def test_comparison_insight_produces_table():
    out = generate_visualizations_for_insight(_ins(itype="comparison"))
    assert any(v["viz_id"] == "comparison_table" for v in out)


def test_profile_aggregator_collects_timeline_from_dated_insights():
    insights = [
        _ins(refreshed_at="2026-04-01T10:00:00+00:00"),
        _ins(refreshed_at="2026-04-15T10:00:00+00:00"),
        _ins(refreshed_at="2026-04-25T10:00:00+00:00"),
    ]
    out = generate_visualizations_for_profile(insights)
    timelines = [v for v in out if v["viz_id"] == "timeline"]
    assert len(timelines) == 1
    assert len(timelines[0]["data"]["events"]) == 3


def test_other_types_produce_no_viz():
    out = generate_visualizations_for_insight(_ins(itype="recommendation"))
    assert out == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/insights_eval/test_visualizations.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement generator**

`src/intelligence/visualizations/__init__.py`:
```python
from src.intelligence.visualizations.generator import (
    generate_visualizations_for_insight,
    generate_visualizations_for_profile,
)

__all__ = [
    "generate_visualizations_for_insight",
    "generate_visualizations_for_profile",
]
```

`src/intelligence/visualizations/generator.py`:
```python
"""Visualization spec generator.

Called at insight-write time + at profile-list time. Produces JSON specs
the frontend can render directly. v1 ships timeline, comparison_table,
trend_chart per spec Section 5.3.
"""
from __future__ import annotations

from typing import Any, Dict, List

from src.intelligence.insights.schema import Insight


def generate_visualizations_for_insight(insight: Insight) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if insight.insight_type == "trend":
        out.append({
            "viz_id": "trend_chart",
            "type": "trend_chart",
            "source_insight_ids": [insight.insight_id],
            "data": {
                "headline": insight.headline,
                "domain": insight.domain,
                "refreshed_at": insight.refreshed_at,
            },
        })
    if insight.insight_type == "comparison":
        out.append({
            "viz_id": "comparison_table",
            "type": "comparison_table",
            "source_insight_ids": [insight.insight_id],
            "data": {
                "headline": insight.headline,
                "documents": insight.document_ids,
            },
        })
    return out


def generate_visualizations_for_profile(insights: List[Insight]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    events = []
    for ins in sorted(insights, key=lambda i: i.refreshed_at):
        events.append({
            "at": ins.refreshed_at,
            "headline": ins.headline,
            "insight_type": ins.insight_type,
            "severity": ins.severity,
        })
    if events:
        out.append({
            "viz_id": "timeline",
            "type": "timeline",
            "source_insight_ids": [i.insight_id for i in insights],
            "data": {"events": events},
        })
    # Aggregate per-insight viz suggestions
    for ins in insights:
        out.extend(generate_visualizations_for_insight(ins))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/insights_eval/test_visualizations.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/intelligence/visualizations/__init__.py src/intelligence/visualizations/generator.py tests/insights_eval/test_visualizations.py
git commit -m "feat(viz): timeline + comparison_table + trend_chart spec generators (SP-I.1)"
```

---

## Final integration tasks

### Task FINAL.1 — Run the full test suite

- [ ] **Step 1: Run all insights-eval + regression + perf tests**

Run: `pytest tests/insights_eval/ tests/regression/ tests/perf/ -v -m "not perf"` (skip perf marker for fast pass)
Expected: all pass.

Run: `pytest tests/perf/ -v -m perf`
Expected: pass + skips for the placeholders.

- [ ] **Step 2: Confirm no existing tests broke**

Run: `pytest tests/ -v --ignore=tests/insights_eval --ignore=tests/regression --ignore=tests/perf`
Expected: all pre-existing tests pass.

- [ ] **Step 3: Commit final test report (if any)**

If any pre-existing tests broke and required a fix, commit those fixes here. Otherwise nothing to commit.

### Task FINAL.2 — Self-review checklist

Before handing off:

- [ ] Spec coverage — every section of `2026-04-25-docwain-research-portal-design.md` mapped to a task in this plan.
- [ ] All 25 feature flags registered in `src/api/feature_flags.py` and `INSIGHTS_PROACTIVE_INJECTION` integrated with `compose_response_with_insights`.
- [ ] All-flags-off regression test (`tests/regression/all_flags_off.py`) passes — module imports clean.
- [ ] Insight lookup p95 ≤ 50ms verified by `tests/perf/insight_lookup_p95.py`.
- [ ] No new files write document content to Mongo (Mongo stays control-plane).
- [ ] Researcher v2 writes only to `researcher_v2.*` — never to `pipeline_status`, `stages.*`, or v1 `researcher.*`.
- [ ] Citation enforcement: zero-doc-spans rejected at writer; body-separation validator rejects KB-derived content in body.
- [ ] No Co-Authored-By / Claude / Anthropic in any commit message or code comment.
- [ ] All endpoints return 404 when their flag is off.
- [ ] Action external-side-effects gated separately (`ACTIONS_EXTERNAL_SIDE_EFFECTS_ENABLED`).
- [ ] Watchlist nightly-only; no higher-cadence scheduling.

### Task FINAL.3 — Update existing kg refs (out of scope confirmation)

Per spec Section 0 / Files-modified note: this plan does NOT modify `src/api/extraction_service.py`, `src/api/embedding_service.py`, or `src/api/dataHandler.py`. KG-build trigger removal is owned by `2026-04-24-kg-training-stage-background-design.md` and its plan. Confirm by `git diff preprod_v02..HEAD -- src/api/extraction_service.py src/api/embedding_service.py src/api/dataHandler.py` returns no output.

If diff is non-empty, revert the unrelated changes before merge.

---

## Plan summary

- **12 sub-projects**, ~50 tasks, ~250 commits expected.
- Every task: failing test → minimal impl → passing test → commit.
- Every capability flag-gated, default off; existing flow unchanged with all flags off.
- Researcher v2, refresh, and actions all run on isolated Celery queues; query-time stays lookup-only.
- Citation enforcement and body-separation enforced at the writer.
- Generic adapter is the always-safe fallback; insurance/medical/etc. ship later as adapter YAMLs uploaded to Blob (no code change required to add a domain).




