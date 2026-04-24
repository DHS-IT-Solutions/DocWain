# Unified DocWain — Wave 2b Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Close the remaining Wave 2 deferrals: (1) central prompt registry as a discoverability index (no code migration); (2) weekend full-set Researcher refresh via Celery beat.

**Architecture:**
- Prompt registry is a thin index — a dict mapping canonical prompt name → (module, attribute). `get_prompt(name)` and `list_prompts()` expose it. Existing prompts stay where they are; no import-path changes for callers.
- Weekend refresh is a new Celery task `researcher_weekly_refresh` scheduled via `celery beat` cron. It enumerates profiles with embedded docs and re-dispatches `run_researcher_agent.delay(...)` per doc. Idempotent (Researcher overwrites prior insights). Flag-gated; default OFF so operators opt in after verifying load.

**Tech stack:** existing Celery + Redis + MongoDB. No new deps. `celery[beat]` already present (Celery includes beat).

**Non-goals:** no migration of existing prompt code from `src/generation/prompts.py` or `src/extraction/vision/*.py` into `src/docwain/prompts/` (deferred indefinitely — the spec accepted co-location with callers). No per-domain adapter authoring (ongoing SME work, out of code scope).

Spec cross-ref: `docs/superpowers/specs/2026-04-24-unified-docwain-engineering-layer-design.md` §5.2 + §5.6.

---

## File structure

**New files:**
- `src/docwain/prompts/registry.py` — index + `get_prompt` / `list_prompts`
- `src/tasks/researcher_refresh.py` — `researcher_weekly_refresh` Celery task
- `tests/unit/docwain/test_prompt_registry.py`
- `tests/unit/api/test_researcher_refresh.py`

**Modified files:**
- `src/celery_app.py` — add beat schedule entry (flag-gated)
- `src/api/config.py` — add `Config.Researcher.WEEKEND_REFRESH_ENABLED` (default `false`) and `Config.Researcher.WEEKEND_REFRESH_CRON` (default `"0 3 * * 0"` — Sunday 03:00 UTC)

**Git:** continue on `preprod_v02`.

---

### Task 1: Prompt registry index (TDD)

**Files:**
- Create: `src/docwain/prompts/registry.py`
- Create: `tests/unit/docwain/test_prompt_registry.py`

#### Step 1 — Failing tests

Create `tests/unit/docwain/test_prompt_registry.py`:

```python
"""Prompt registry exposes known DocWain prompts by name."""
import pytest


def test_registry_lists_known_prompts():
    from src.docwain.prompts.registry import list_prompts
    names = list_prompts()
    # Wave 1 + Wave 2 prompts land in the registry:
    expected = {
        "entity_extraction",
        "researcher",
        "chart_generation",
        "docintel_classifier",
        "docintel_coverage_verifier",
        "vision_extractor",
    }
    missing = expected - set(names)
    assert not missing, f"registry missing prompts: {missing}"


def test_get_prompt_returns_non_empty_string():
    from src.docwain.prompts.registry import get_prompt
    text = get_prompt("entity_extraction")
    assert isinstance(text, str)
    assert len(text) > 50


def test_get_prompt_raises_on_unknown_name():
    from src.docwain.prompts.registry import get_prompt
    with pytest.raises(KeyError):
        get_prompt("nonexistent_prompt")


def test_all_registered_prompts_resolve():
    """Every name in the registry must resolve to a non-empty prompt string."""
    from src.docwain.prompts.registry import get_prompt, list_prompts
    for name in list_prompts():
        text = get_prompt(name)
        assert isinstance(text, str) and text, f"prompt {name!r} resolved to empty/non-string"
```

#### Step 2 — Implement

Create `src/docwain/prompts/registry.py`:

```python
"""Prompt registry — canonical index of DocWain's task-specific prompts.

Prompts live in whichever module owns the capability (co-location with the
caller). This registry provides discoverability: `list_prompts()` enumerates
everything known; `get_prompt(name)` returns the system-prompt string. No
migration — this module is introspection only.

Spec: 2026-04-24-unified-docwain-engineering-layer-design.md §5.2
"""
from __future__ import annotations

from importlib import import_module
from typing import Dict, List, Tuple


# name -> (module dotted path, attribute name)
_PROMPT_INDEX: Dict[str, Tuple[str, str]] = {
    "entity_extraction": ("src.docwain.prompts.entity_extraction", "ENTITY_EXTRACTION_SYSTEM_PROMPT"),
    "researcher": ("src.docwain.prompts.researcher", "RESEARCHER_SYSTEM_PROMPT"),
    "chart_generation": ("src.docwain.prompts.chart_generation", "CHART_GENERATION_SYSTEM_PROMPT"),
    "docintel_classifier": ("src.extraction.vision.docintel", "CLASSIFIER_SYSTEM_PROMPT"),
    "docintel_coverage_verifier": ("src.extraction.vision.docintel", "COVERAGE_SYSTEM_PROMPT"),
    "vision_extractor": ("src.extraction.vision.extractor", "EXTRACTOR_SYSTEM_PROMPT"),
}


def list_prompts() -> List[str]:
    """Return the sorted list of registered prompt names."""
    return sorted(_PROMPT_INDEX.keys())


def get_prompt(name: str) -> str:
    """Return the system-prompt string for the given registered name.

    Raises KeyError if the name is not registered.
    Raises ImportError / AttributeError if the module or attribute is missing
    (indicates the registry is out of sync with the codebase).
    """
    if name not in _PROMPT_INDEX:
        raise KeyError(f"unknown prompt name: {name!r}. Known: {list_prompts()}")
    module_path, attr = _PROMPT_INDEX[name]
    module = import_module(module_path)
    value = getattr(module, attr)
    if not isinstance(value, str):
        raise TypeError(f"registered prompt {name!r} is not a string (got {type(value).__name__})")
    return value


def register_prompt(name: str, module_path: str, attribute: str) -> None:
    """Register a new prompt under the given name. Used when callers add new
    task-specific prompts.
    """
    _PROMPT_INDEX[name] = (module_path, attribute)
```

#### Step 3 — Verify + commit

```bash
cd /home/ubuntu/.config/superpowers/worktrees/DocWain/preprod_v02
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/pytest tests/unit/docwain/test_prompt_registry.py -x -q
git add src/docwain/prompts/registry.py tests/unit/docwain/test_prompt_registry.py
git commit -m "unified: prompt registry — discoverability index for DocWain task prompts"
```

Expected: 4 passed. No Claude/Anthropic/Co-Authored-By.

---

### Task 2: Weekend Researcher refresh (Celery beat) — TDD

**Files:**
- Modify: `src/api/config.py` — add refresh flag + cron config
- Create: `src/tasks/researcher_refresh.py` — `researcher_weekly_refresh` task
- Modify: `src/celery_app.py` — wire beat schedule (flag-gated)
- Create: `tests/unit/api/test_researcher_refresh.py`

#### Step 1 — Add config flags

In `src/api/config.py`, extend the existing `Researcher` nested class (added in Wave 1 T1):

```python
class Researcher:
    ENABLED = os.getenv("DOCWAIN_RESEARCHER_ENABLED", "true").lower() == "true"
    MAX_TOKENS = int(os.getenv("DOCWAIN_RESEARCHER_MAX_TOKENS", "4096"))
    # Weekend refresh — default OFF so operators opt in after verifying load
    WEEKEND_REFRESH_ENABLED = os.getenv("DOCWAIN_RESEARCHER_WEEKEND_REFRESH_ENABLED", "false").lower() == "true"
    # Crontab string for the beat schedule. Default: Sunday 03:00 UTC.
    WEEKEND_REFRESH_CRON = os.getenv("DOCWAIN_RESEARCHER_WEEKEND_REFRESH_CRON", "0 3 * * 0")
```

Do NOT modify the existing `ENABLED` / `MAX_TOKENS` lines. Only APPEND.

#### Step 2 — Failing test

Create `tests/unit/api/test_researcher_refresh.py`:

```python
"""Weekend Researcher refresh — enumerates profiles + dispatches run_researcher_agent."""
from unittest.mock import MagicMock


def test_weekly_refresh_dispatches_researcher_per_embedded_doc(monkeypatch):
    """For each document whose pipeline_status is TRAINING_COMPLETED, dispatch run_researcher_agent.delay(...)."""
    from src.tasks import researcher_refresh as rr

    # Fake MongoDB returning 3 docs across 2 profiles, all TRAINING_COMPLETED.
    docs = [
        {"document_id": "d1", "subscription_id": "sub-a", "profile_id": "prof-a",
         "pipeline_status": "TRAINING_COMPLETED"},
        {"document_id": "d2", "subscription_id": "sub-a", "profile_id": "prof-a",
         "pipeline_status": "TRAINING_COMPLETED"},
        {"document_id": "d3", "subscription_id": "sub-b", "profile_id": "prof-b",
         "pipeline_status": "TRAINING_COMPLETED"},
        # Non-completed doc — should NOT be refreshed:
        {"document_id": "d4", "subscription_id": "sub-a", "profile_id": "prof-a",
         "pipeline_status": "EMBEDDING_IN_PROGRESS"},
    ]

    class FakeCol:
        def find(self, filter, projection=None):
            # Honor filter if provided; our task filters by pipeline_status.
            if filter and "pipeline_status" in filter:
                status = filter["pipeline_status"]
                if isinstance(status, dict) and "$in" in status:
                    return iter([d for d in docs if d.get("pipeline_status") in status["$in"]])
                return iter([d for d in docs if d.get("pipeline_status") == status])
            return iter(docs)

    monkeypatch.setattr(rr, "_get_documents_collection", lambda: FakeCol(), raising=False)

    dispatched = []
    def fake_delay(document_id, subscription_id, profile_id):
        dispatched.append((document_id, subscription_id, profile_id))
        return MagicMock()

    monkeypatch.setattr("src.tasks.researcher.run_researcher_agent.delay", fake_delay)

    # Run the task synchronously.
    result = rr.researcher_weekly_refresh.apply().get()

    assert result["dispatched_count"] == 3
    assert sorted(dispatched) == sorted([
        ("d1", "sub-a", "prof-a"),
        ("d2", "sub-a", "prof-a"),
        ("d3", "sub-b", "prof-b"),
    ])


def test_weekly_refresh_returns_zero_when_disabled(monkeypatch):
    from src.tasks import researcher_refresh as rr
    from src.api import config as cfg_mod
    monkeypatch.setattr(cfg_mod.Config.Researcher, "ENABLED", False, raising=False)

    dispatched = []
    monkeypatch.setattr("src.tasks.researcher.run_researcher_agent.delay",
                        lambda *a, **kw: dispatched.append(a) or MagicMock())

    result = rr.researcher_weekly_refresh.apply().get()
    assert result.get("skipped") is True
    assert result.get("dispatched_count", 0) == 0
    assert not dispatched


def test_weekly_refresh_swallows_individual_dispatch_failures(monkeypatch):
    """If dispatching fails for one doc, other docs still get dispatched."""
    from src.tasks import researcher_refresh as rr

    docs = [
        {"document_id": f"d{i}", "subscription_id": "sub", "profile_id": "prof",
         "pipeline_status": "TRAINING_COMPLETED"}
        for i in range(3)
    ]
    class FakeCol:
        def find(self, filter, projection=None):
            return iter(docs)
    monkeypatch.setattr(rr, "_get_documents_collection", lambda: FakeCol(), raising=False)

    dispatched = []
    def flaky_delay(document_id, subscription_id, profile_id):
        if document_id == "d1":
            raise RuntimeError("broker full")
        dispatched.append(document_id)
        return MagicMock()
    monkeypatch.setattr("src.tasks.researcher.run_researcher_agent.delay", flaky_delay)

    result = rr.researcher_weekly_refresh.apply().get()
    # Three attempts; one failed; two succeeded.
    assert result["dispatched_count"] == 2
    assert result["failed_count"] == 1
    assert sorted(dispatched) == ["d0", "d2"]
```

#### Step 3 — Implement the task

Create `src/tasks/researcher_refresh.py`:

```python
"""Weekend full-set Researcher refresh.

Enumerates documents in TRAINING_COMPLETED (i.e., past-HITL-training) state
and re-dispatches `run_researcher_agent` for each. Cadence: Celery beat
weekly, typically Sunday 03:00 UTC (configurable).

Isolation: runs on `researcher_queue` same as the per-doc Researcher task.
Never touches `pipeline_status`. Flag-gated via
`Config.Researcher.WEEKEND_REFRESH_ENABLED`.

Spec: project_researcher_agent_vision.md + 2026-04-24-unified-docwain-wave2b-plan.md
"""
from __future__ import annotations

import logging
from typing import Any

from src.celery_app import app

logger = logging.getLogger(__name__)


def _get_documents_collection():
    """Return the MongoDB documents collection. Wrapped for test monkeypatching."""
    try:
        from src.api.dw_newron import get_mongo_collection
        return get_mongo_collection("documents")
    except Exception:
        # Fallback chain — some modules expose different helper names.
        try:
            from src.api.document_status import get_documents_collection
            return get_documents_collection()
        except Exception as exc:
            logger.error("Cannot obtain documents collection: %s", exc)
            raise


@app.task(bind=True, name="src.tasks.researcher_refresh.researcher_weekly_refresh",
           max_retries=0, soft_time_limit=3600)
def researcher_weekly_refresh(self):
    """Weekly full-set Researcher refresh across all TRAINING_COMPLETED docs."""
    try:
        from src.api.config import Config
        researcher_cfg = getattr(Config, "Researcher", None)
        enabled = getattr(researcher_cfg, "ENABLED", False) if researcher_cfg else False
    except Exception:
        enabled = False

    if not enabled:
        logger.info("Researcher disabled; skipping weekly refresh.")
        return {"skipped": True, "dispatched_count": 0, "failed_count": 0}

    try:
        from src.tasks.researcher import run_researcher_agent
    except Exception as exc:
        logger.error("Cannot import run_researcher_agent: %s", exc)
        return {"skipped": True, "dispatched_count": 0, "failed_count": 0,
                "error": f"import failed: {exc}"}

    col = _get_documents_collection()

    projection = {"document_id": 1, "subscription_id": 1, "profile_id": 1,
                  "pipeline_status": 1}
    # Include both "TRAINING_COMPLETED" and "TRAINING_PARTIALLY_COMPLETED" per
    # canonical status values. Exclude failed/blocked.
    filter_ = {"pipeline_status": {"$in": [
        "TRAINING_COMPLETED",
        "TRAINING_PARTIALLY_COMPLETED",
    ]}}

    dispatched_count = 0
    failed_count = 0

    try:
        cursor = col.find(filter_, projection)
    except TypeError:
        # Some fake collections ignore projection; retry without it.
        cursor = col.find(filter_)

    for doc in cursor:
        doc_id = doc.get("document_id") or doc.get("_id")
        sub_id = doc.get("subscription_id")
        profile_id = doc.get("profile_id")
        if not (doc_id and sub_id and profile_id):
            continue
        try:
            run_researcher_agent.delay(doc_id, sub_id, profile_id)
            dispatched_count += 1
        except Exception as exc:
            failed_count += 1
            logger.warning("Researcher refresh dispatch failed for %s: %s", doc_id, exc)

    logger.info("Weekly Researcher refresh: dispatched=%d failed=%d",
                dispatched_count, failed_count)
    return {"dispatched_count": dispatched_count, "failed_count": failed_count,
            "skipped": False}
```

#### Step 4 — Wire beat schedule (flag-gated)

In `src/celery_app.py`, add near the app configuration (after `app = Celery(...)` and before the `task_routes`):

```python
# Weekend Researcher refresh beat schedule (flag-gated).
def _build_researcher_beat_schedule():
    try:
        from src.api.config import Config
        from celery.schedules import crontab
        researcher_cfg = getattr(Config, "Researcher", None)
        if not researcher_cfg or not getattr(researcher_cfg, "WEEKEND_REFRESH_ENABLED", False):
            return {}
        cron_expr = getattr(researcher_cfg, "WEEKEND_REFRESH_CRON", "0 3 * * 0")
        parts = cron_expr.split()
        if len(parts) != 5:
            return {}
        minute, hour, day_of_month, month_of_year, day_of_week = parts
        return {
            "researcher-weekly-refresh": {
                "task": "src.tasks.researcher_refresh.researcher_weekly_refresh",
                "schedule": crontab(
                    minute=minute, hour=hour,
                    day_of_week=day_of_week,
                    day_of_month=day_of_month,
                    month_of_year=month_of_year,
                ),
                "options": {"queue": "researcher_queue"},
            }
        }
    except Exception:
        return {}


# Apply the beat schedule if the flag is on; empty dict otherwise (beat runs
# with no scheduled tasks).
try:
    app.conf.beat_schedule = _build_researcher_beat_schedule()
except Exception:
    pass
```

Also add `src.tasks.researcher_refresh` to the autodiscover tasks list if one exists (check existing `app.autodiscover_tasks(...)` invocation and add).

#### Step 5 — Run tests

```bash
cd /home/ubuntu/.config/superpowers/worktrees/DocWain/preprod_v02
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/pytest tests/unit/api/test_researcher_refresh.py -x -q
```

Expected: 3 passed.

#### Step 6 — Sanity broader

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/pytest tests/unit -q --timeout=30 2>&1 | tail -6
```

Expected: no new failures.

#### Step 7 — Commit

```bash
git add src/api/config.py src/tasks/researcher_refresh.py src/celery_app.py tests/unit/api/test_researcher_refresh.py
git commit -m "unified: weekend Researcher refresh via Celery beat (flag-gated, default off)"
```

No Claude/Anthropic/Co-Authored-By.

---

### Task 3: Validation

- [ ] Run full suites + bench + imports smoke.

```bash
cd /home/ubuntu/.config/superpowers/worktrees/DocWain/preprod_v02
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/pytest tests/unit/extraction tests/unit/kg tests/unit/api tests/unit/llm tests/unit/docwain tests/integration -q --timeout=30 2>&1 | tail -6

/home/ubuntu/PycharmProjects/DocWain/.venv/bin/python -m tests.extraction_bench.runner 2>&1 | tail -12

/home/ubuntu/PycharmProjects/DocWain/.venv/bin/python -c "
from src.docwain.prompts.registry import list_prompts, get_prompt
from src.tasks.researcher_refresh import researcher_weekly_refresh
from src.api.config import Config
assert hasattr(Config.Researcher, 'WEEKEND_REFRESH_ENABLED')
print('WAVE 2B OK')
print(f'registered prompts: {list_prompts()}')
print(f'weekend refresh enabled: {Config.Researcher.WEEKEND_REFRESH_ENABLED}')
"
```

No commit.

---

## Self-review

- **§5.2 prompt registry:** Task 1 ✓ (index only; no migration, per spec scope)
- **§5.6 weekend refresh:** Task 2 ✓
- **Per-domain YAML authoring:** out of scope (SME content work, not code)

Placeholder scan: all code shown verbatim. Type consistency: `get_prompt`, `list_prompts`, `register_prompt` defined Task 1, referenced nowhere else in this plan; `researcher_weekly_refresh` task + `_get_documents_collection` helper defined Task 2, no cross-refs needed.

Plan complete.
