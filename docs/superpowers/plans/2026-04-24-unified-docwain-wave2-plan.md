# Unified DocWain — Wave 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the remaining user-visible pieces of the unified DocWain engineering layer (domain adapter framework, chart generation) and close two log-driven defects found in the 2026-04-24 log analysis (Ollama `qwen3:14b` stale reference, INFO-level spam).

**Architecture:** Log hygiene is a small per-module adjustment (downgrade routine INFO→DEBUG, add a request-path filter for health-check polls on the correlation middleware, mute Azure SDK verbose). The qwen3:14b fix is surgical: find the stale call site and route it through the gateway (which now has vLLM primary). Domain adapter framework adds an Azure Blob YAML loader with TTL cache; seed adapter `generic.yaml`. Chart generation module adds a prompt + parser so DocWain emits DOCWAIN_VIZ JSON payloads deterministically when queries warrant charts/comparisons.

**Tech Stack:** Python 3.12, PyYAML (stdlib addition), azure-storage-blob (already present), existing Celery + vLLM + LLMGateway.

**Non-goals:** no central prompt registry refactor (deferred to Wave 2b), no weekend-refresh Celery-beat scheduler (deferred to Wave 2b), no per-domain adapter authoring beyond a seed `generic.yaml` (each domain-specific YAML is a separate piece of content, authored by domain SMEs later).

Spec: `docs/superpowers/specs/2026-04-24-unified-docwain-engineering-layer-design.md` §5.4 (chart gen) and §5.5 (domain adapters).

---

## File structure

**New files:**
- `src/docwain/logging_config.py` — centralized log-level adjustments (Azure SDK mute, HTTP middleware filter, EvidenceVerifier downgrade)
- `src/docwain/prompts/chart_generation.py` — chart-generation prompt + DOCWAIN_VIZ parser
- `src/docwain/adapters/__init__.py`
- `src/docwain/adapters/loader.py` — Azure Blob YAML loader with TTL cache
- `src/docwain/adapters/schema.py` — `DomainAdapter` dataclass
- `src/docwain/adapters/seed/generic.yaml` — the fallback adapter used when no domain match
- `tests/unit/docwain/test_chart_generation.py`
- `tests/unit/docwain/test_adapter_loader.py`
- `tests/unit/docwain/test_logging_config.py`

**Modified files:**
- `src/main.py` — wire `logging_config.apply_log_hygiene()` on app startup (idempotent; safe to call multiple times)
- `src/celery_app.py` — wire `logging_config.apply_log_hygiene()` on worker startup
- `src/middleware/correlation.py` — add a filter that demotes `/api/extract/progress` + `/api/train/progress` logs to DEBUG (or suppresses them)
- `src/intelligence/evidence_verifier.py` — downgrade the `[EvidenceVerifier] Verified result:` INFO log to DEBUG
- `src/llm/clients.py` (if needed) — if the stale qwen3:14b reference is here, fix
- `src/api/config.py` — add `Config.DomainAdapters.BLOB_PREFIX`, `Config.DomainAdapters.CACHE_TTL_SECONDS`

**Git:** continue on `preprod_v02`. One commit per task. Flags default safe.

---

### Task 1: Log hygiene module + wire it in (TDD)

**Files:**
- Create: `src/docwain/logging_config.py`
- Create: `tests/unit/docwain/test_logging_config.py`
- Modify: `src/main.py` (wire `apply_log_hygiene()` on startup)
- Modify: `src/celery_app.py` (same, on worker startup)
- Modify: `src/middleware/correlation.py` (filter health-check polls)
- Modify: `src/intelligence/evidence_verifier.py` (downgrade INFO → DEBUG)

**What the module does:**
1. Silences `azsdk-*` loggers to WARN (suppresses the verbose request/response DEBUG logs that pollute celery_worker.log).
2. Installs a filter on the HTTP middleware logger that demotes log records whose request path is `/api/extract/progress` or `/api/train/progress` to DEBUG (or drops them entirely, configurable via env).
3. Downgrades the one-line `[EvidenceVerifier] Verified result: ...` INFO log to DEBUG at the emission site.

#### Step 1 — Write failing test

Create `tests/unit/docwain/test_logging_config.py`:

```python
"""Verify log-hygiene settings apply as expected."""
import logging

from src.docwain.logging_config import (
    apply_log_hygiene,
    HealthCheckPathFilter,
)


def test_apply_log_hygiene_silences_azure_sdk_to_warn():
    # Reset first
    azure_logger = logging.getLogger("azure.core.pipeline.policies.http_logging_policy")
    azure_logger.setLevel(logging.DEBUG)
    apply_log_hygiene()
    assert azure_logger.level == logging.WARNING


def test_apply_log_hygiene_silences_azure_storage_blob():
    blob_logger = logging.getLogger("azure.storage.blob")
    blob_logger.setLevel(logging.DEBUG)
    apply_log_hygiene()
    assert blob_logger.level >= logging.WARNING


def test_healthcheck_filter_demotes_progress_polls():
    filter_ = HealthCheckPathFilter()
    rec = logging.LogRecord(
        name="src.middleware.correlation", level=logging.INFO,
        pathname="x", lineno=1,
        msg="Request completed: GET /api/extract/progress -> 200 (1300.0ms)",
        args=(), exc_info=None,
    )
    # The filter returns False to drop INFO records for progress endpoints,
    # True otherwise.
    assert filter_.filter(rec) is False


def test_healthcheck_filter_allows_normal_paths():
    filter_ = HealthCheckPathFilter()
    rec = logging.LogRecord(
        name="src.middleware.correlation", level=logging.INFO,
        pathname="x", lineno=1,
        msg="Request completed: POST /api/ask -> 200 (520.0ms)",
        args=(), exc_info=None,
    )
    assert filter_.filter(rec) is True


def test_healthcheck_filter_allows_errors_on_progress_paths():
    """A 500 on /api/extract/progress SHOULD still surface — only INFO is muted."""
    filter_ = HealthCheckPathFilter()
    rec = logging.LogRecord(
        name="src.middleware.correlation", level=logging.ERROR,
        pathname="x", lineno=1,
        msg="Request completed: GET /api/extract/progress -> 500",
        args=(), exc_info=None,
    )
    assert filter_.filter(rec) is True
```

#### Step 2 — Run to confirm fail

```bash
cd /home/ubuntu/.config/superpowers/worktrees/DocWain/preprod_v02
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/pytest tests/unit/docwain/test_logging_config.py -x -q
```

Expected: ImportError on the module.

#### Step 3 — Implement the module

Create `src/docwain/logging_config.py`:

```python
"""Centralized log-hygiene configuration for DocWain services.

Called at process startup (both API app and Celery worker). Idempotent — safe
to call multiple times.

Applies:
- Azure SDK loggers demoted to WARNING (suppresses verbose DEBUG of blob ops)
- HealthCheckPathFilter installed on the correlation middleware logger to drop
  INFO-level health-check polls on /api/extract/progress and /api/train/progress

Log-analysis findings (2026-04-24):
- Azure SDK produced 44KB of DEBUG-level HTTP dialogue in a single celery log tail
- /api/extract/progress logged 754x in 2000 lines at INFO
- /api/train/progress similar

Spec: 2026-04-24-unified-docwain-wave2-plan.md Task 1
"""
from __future__ import annotations

import logging
import os
from typing import Iterable


HEALTHCHECK_PATHS = ("/api/extract/progress", "/api/train/progress")

AZURE_SDK_LOGGERS = (
    "azure",
    "azure.core",
    "azure.core.pipeline.policies.http_logging_policy",
    "azure.storage",
    "azure.storage.blob",
    "azure.identity",
    "urllib3",
)


class HealthCheckPathFilter(logging.Filter):
    """Drop INFO-level log records that describe health-check polls on progress endpoints.

    Records at WARNING or above (even for progress paths) pass through unchanged
    so operators still see errors. The only thing muted is routine successful
    progress polls logged at INFO by `src/middleware/correlation.py`.
    """

    HEALTHCHECK_PATHS: tuple = HEALTHCHECK_PATHS

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            if record.levelno >= logging.WARNING:
                return True
            msg = record.getMessage() if record.args else str(record.msg)
            for path in self.HEALTHCHECK_PATHS:
                if path in msg:
                    return False
            return True
        except Exception:
            return True  # never break logging


def _set_level(names: Iterable[str], level: int) -> None:
    for name in names:
        logging.getLogger(name).setLevel(level)


_APPLIED = False


def apply_log_hygiene() -> None:
    """Apply centralized log-hygiene settings. Idempotent."""
    global _APPLIED
    if _APPLIED:
        return

    # Azure SDK + urllib3 chatter → WARNING.
    _set_level(AZURE_SDK_LOGGERS, logging.WARNING)

    # HTTP middleware progress-path filter.
    correlation_logger = logging.getLogger("src.middleware.correlation")
    # Avoid adding the same filter twice.
    if not any(isinstance(f, HealthCheckPathFilter) for f in correlation_logger.filters):
        correlation_logger.addFilter(HealthCheckPathFilter())

    _APPLIED = True


def is_applied() -> bool:
    return _APPLIED
```

#### Step 4 — Verify tests pass

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/pytest tests/unit/docwain/test_logging_config.py -x -q
```

Expected: 5 passed.

#### Step 5 — Wire into app startup

Find where the FastAPI app is initialized in `src/main.py` (look for `app = FastAPI(...)` or similar, and the `lifespan` / startup handler). Add early (ideally before any other imports that might log):

```python
from src.docwain.logging_config import apply_log_hygiene
apply_log_hygiene()
```

Place this EARLY in `src/main.py` — immediately after the imports, before route registration or app creation. If the app uses a `lifespan` handler, apply BEFORE that too (so handlers that log during startup respect hygiene).

#### Step 6 — Wire into Celery worker startup

In `src/celery_app.py`, add near the top after imports and before `app = Celery(...)`:

```python
from src.docwain.logging_config import apply_log_hygiene
apply_log_hygiene()
```

#### Step 7 — Downgrade EvidenceVerifier log

In `src/intelligence/evidence_verifier.py`, find the `[EvidenceVerifier] Verified result:` log line (around line 224 per analysis). Change `logger.info(...)` to `logger.debug(...)`. Nothing else.

```bash
grep -n "Verified result" src/intelligence/evidence_verifier.py
```

Edit the line. If the logger is also emitting additional INFO lines in the verifier that the analysis didn't name but are clearly routine, demote those too. Be conservative — if in doubt, leave them.

#### Step 8 — Sanity import + existing tests

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/python -c "from src.main import app; print('app OK')" 2>&1 | tail -3
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/python -c "from src import celery_app; print('celery OK')"
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/pytest tests/unit/docwain -q --timeout=30 2>&1 | tail -5
```

Expected: `app OK`, `celery OK`, all docwain tests pass.

#### Step 9 — Commit

```bash
git add src/docwain/logging_config.py tests/unit/docwain/test_logging_config.py \
        src/main.py src/celery_app.py src/intelligence/evidence_verifier.py
git commit -m "unified: centralized log hygiene — suppress Azure SDK + health-check polls + routine KG verifier INFO"
```

No Claude/Anthropic/Co-Authored-By.

---

### Task 2: Fix stale qwen3:14b reference

**Files:** TBD based on grep findings.

- [ ] **Step 1: Find the stale reference**

```bash
cd /home/ubuntu/.config/superpowers/worktrees/DocWain/preprod_v02
grep -rn --include='*.py' "qwen3:14b\|qwen3-14b\|qwen3:14\|qwen3_14" src/ 2>&1 | head -30
echo "---"
grep -n "OLLAMA_LOCAL_MODEL\|OLLAMA_MODEL" /home/ubuntu/PycharmProjects/DocWain/.env 2>&1 || true
echo "---"
grep -rn --include='*.py' "OLLAMA_LOCAL_MODEL\|OLLAMA_MODEL\b" src/ 2>&1 | head -10
```

Classify findings:
- **Code reference** (hardcoded `qwen3:14b` or `qwen3_14b` in `src/`) → fix.
- **Env reference** (via `os.getenv("OLLAMA_LOCAL_MODEL")`) where the env defaults to `qwen3:14b` → fix either the default or the caller.
- **Already removed / quarantined** (e.g., only in a stubbed `src/extraction/vision_extractor.py` which Plan 1 renamed and left unused) → remove the stub entirely.

- [ ] **Step 2: Pick the right fix based on findings**

Decision tree:
- If the reference is in a code path THAT IS STILL CALLED at runtime (i.e., the 45 log hits are coming from live traffic):
  - Route that call through `LLMGateway` instead of a direct Ollama client. Gateway primary is vLLM local now (per Wave 1), so the call lands on DocWain-14B-v2.
- If the reference is in a DEAD code path (nobody calls it anymore):
  - Delete the dead module.
- If the reference is in `OLLAMA_LOCAL_MODEL` env default in code:
  - Change the code default to match what's actually available locally (per local Ollama output: `DHS/DocWain:latest` is the closest fit). Note: don't touch the prod `.env` file; just the code fallback.

For whichever fix applies, add a unit test that verifies the code path no longer issues a call to `qwen3:14b`. Example skeleton:

```python
def test_no_hardcoded_qwen3_14b_in_src():
    """qwen3:14b is not present as a hardcoded model name in production source."""
    import subprocess
    out = subprocess.check_output(
        ["grep", "-rn", "--include=*.py", "qwen3:14b", "src/"],
        cwd="/home/ubuntu/.config/superpowers/worktrees/DocWain/preprod_v02",
        stderr=subprocess.STDOUT,
    ).decode() if False else ""  # grep returns non-zero on no matches; handle via try
    # Use shell helper:
    import os
    result = os.popen("cd /home/ubuntu/.config/superpowers/worktrees/DocWain/preprod_v02 && grep -rn --include='*.py' 'qwen3:14b' src/ 2>/dev/null").read()
    assert result.strip() == "", f"stale qwen3:14b reference found:\n{result}"
```

Place this in `tests/unit/docwain/test_no_stale_qwen_ref.py`.

- [ ] **Step 3: Apply the fix + re-run the test**

After the fix:
```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/pytest tests/unit/docwain/test_no_stale_qwen_ref.py -x -q
```

Expected: passes.

- [ ] **Step 4: Sanity broader suite**

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/pytest tests/unit -q --timeout=30 2>&1 | tail -6
```

Expected: no new failures.

- [ ] **Step 5: Commit**

```bash
git add -A src/ tests/unit/docwain/test_no_stale_qwen_ref.py
git commit -m "unified: remove stale qwen3:14b reference; route legacy LLM calls through gateway"
```

Report exactly what was changed and which files. Commit message stays tight.

---

### Task 3: Domain adapter framework (TDD)

**Files:**
- Create: `src/docwain/adapters/__init__.py` (one-line comment)
- Create: `src/docwain/adapters/schema.py` — `DomainAdapter` dataclass
- Create: `src/docwain/adapters/loader.py` — Azure Blob YAML loader with TTL cache
- Create: `src/docwain/adapters/seed/generic.yaml` — minimal generic adapter
- Modify: `src/api/config.py` — add `Config.DomainAdapters.BLOB_PREFIX`, `Config.DomainAdapters.CACHE_TTL_SECONDS`
- Create: `tests/unit/docwain/test_adapter_loader.py`

The loader:
- Fetches a domain's YAML from Azure Blob at path `{BLOB_PREFIX}/{subscription_id}/{domain}.yaml` first (per-subscription override), falling back to `{BLOB_PREFIX}/global/{domain}.yaml`, falling back to the baked-in generic seed YAML (no cloud call needed).
- Caches parsed `DomainAdapter` objects in-process with TTL (default 300 seconds).
- On any Azure Blob exception, falls back silently to cached or seed value.
- Returns a `DomainAdapter` dataclass with fields: `domain`, `version`, `prompt_fragment`, `key_entities` (list), `analysis_hints` (dict), `questions_to_ask` (list).

#### Step 1 — Add config

In `src/api/config.py`:

```python
class DomainAdapters:
    BLOB_PREFIX = os.getenv("DOCWAIN_ADAPTER_BLOB_PREFIX", "sme_adapters")
    CACHE_TTL_SECONDS = int(os.getenv("DOCWAIN_ADAPTER_CACHE_TTL", "300"))
    BLOB_CONTAINER = os.getenv("DOCWAIN_ADAPTER_BLOB_CONTAINER", "docwain-configs")
```

Match the style of other Config nested classes.

#### Step 2 — Schema dataclass

Create `src/docwain/adapters/__init__.py`:
```
# DocWain domain adapter framework
```

Create `src/docwain/adapters/schema.py`:

```python
"""DomainAdapter dataclass — the shape of a domain YAML after parsing."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class DomainAdapter:
    domain: str = "generic"
    version: str = "v1"
    prompt_fragment: str = ""
    key_entities: List[str] = field(default_factory=list)
    analysis_hints: Dict[str, Any] = field(default_factory=dict)
    questions_to_ask: List[str] = field(default_factory=list)
```

#### Step 3 — Seed generic YAML

Create `src/docwain/adapters/seed/generic.yaml`:

```yaml
domain: generic
version: v1
prompt_fragment: |
  Treat the document as general content. Extract entities, key facts, and
  actionable insights without assuming a specific domain.
key_entities:
  - person
  - organization
  - date
  - location
  - amount
analysis_hints:
  summary_style: factual
  insight_depth: medium
  emphasize: [facts, dates, amounts]
questions_to_ask:
  - "What is the main purpose of this document?"
  - "Who are the key parties or entities involved?"
  - "What dates or deadlines are mentioned?"
  - "Are there any specific amounts, values, or metrics?"
```

#### Step 4 — Failing tests

Create `tests/unit/docwain/test_adapter_loader.py`:

```python
"""Domain adapter loader — Blob fetch, TTL cache, generic fallback."""
import time
from unittest.mock import MagicMock

import pytest

from src.docwain.adapters.loader import AdapterLoader
from src.docwain.adapters.schema import DomainAdapter


def test_generic_seed_loaded_when_blob_unavailable(monkeypatch):
    """If Blob fetch raises, loader returns the baked-in generic adapter."""
    loader = AdapterLoader(subscription_id="sub-x")
    # Monkeypatch _fetch_from_blob to raise (simulate Blob down or not configured).
    monkeypatch.setattr(loader, "_fetch_from_blob", lambda path: (_ for _ in ()).throw(RuntimeError("blob down")))
    adapter = loader.load("generic")
    assert isinstance(adapter, DomainAdapter)
    assert adapter.domain == "generic"
    assert adapter.prompt_fragment  # non-empty


def test_unknown_domain_falls_back_to_generic(monkeypatch):
    """Asking for 'alien_domain' when only generic exists → returns generic adapter."""
    loader = AdapterLoader(subscription_id="sub-x")
    monkeypatch.setattr(loader, "_fetch_from_blob", lambda path: (_ for _ in ()).throw(FileNotFoundError(path)))
    adapter = loader.load("alien_domain")
    assert adapter.domain == "generic"


def test_subscription_override_tried_first(monkeypatch):
    """Loader tries {BLOB_PREFIX}/{sub_id}/{domain}.yaml before /global/{domain}.yaml."""
    loader = AdapterLoader(subscription_id="sub-x")

    paths_tried = []

    def fake_fetch(path):
        paths_tried.append(path)
        raise FileNotFoundError(path)  # all paths fail → generic fallback

    monkeypatch.setattr(loader, "_fetch_from_blob", fake_fetch)
    loader.load("finance")
    assert any("sub-x" in p and "finance" in p for p in paths_tried), paths_tried
    assert any("global" in p and "finance" in p for p in paths_tried), paths_tried


def test_cache_honors_ttl(monkeypatch):
    """Second load within TTL returns the cached adapter without re-fetching."""
    loader = AdapterLoader(subscription_id="sub-x", cache_ttl_seconds=10)
    call_count = {"n": 0}

    def fake_fetch(path):
        call_count["n"] += 1
        if "generic" in path:
            # Return the YAML text for a minimal generic adapter
            return "domain: generic\nversion: v1\nprompt_fragment: hi\nkey_entities: []\nanalysis_hints: {}\nquestions_to_ask: []\n"
        raise FileNotFoundError(path)

    monkeypatch.setattr(loader, "_fetch_from_blob", fake_fetch)
    a1 = loader.load("generic")
    a2 = loader.load("generic")
    assert a1 is a2 or (a1.domain == a2.domain)
    # Only called once — second load hit the cache
    # (call_count may be >1 for sub-specific then global path attempts; we just assert no third call on second load)
    count_after_first = call_count["n"]
    loader.load("generic")
    assert call_count["n"] == count_after_first  # no additional fetches on cached load


def test_parses_yaml_into_dataclass(monkeypatch):
    loader = AdapterLoader(subscription_id="sub-x")
    yaml_text = (
        "domain: finance\nversion: v2\n"
        "prompt_fragment: Focus on financial details.\n"
        "key_entities: [person, money]\n"
        "analysis_hints: {summary_style: concise}\n"
        "questions_to_ask: [What is the total?]\n"
    )

    def fake_fetch(path):
        if "finance" in path:
            return yaml_text
        raise FileNotFoundError(path)

    monkeypatch.setattr(loader, "_fetch_from_blob", fake_fetch)
    a = loader.load("finance")
    assert a.domain == "finance"
    assert a.version == "v2"
    assert "Focus on financial" in a.prompt_fragment
    assert "person" in a.key_entities
    assert a.analysis_hints.get("summary_style") == "concise"
    assert any("total" in q.lower() for q in a.questions_to_ask)
```

#### Step 5 — Implement the loader

Create `src/docwain/adapters/loader.py`:

```python
"""Domain adapter loader.

Fetches per-domain YAML from Azure Blob at:
    {BLOB_PREFIX}/{subscription_id}/{domain}.yaml  (per-subscription override)
    {BLOB_PREFIX}/global/{domain}.yaml             (global default)

Falls back to a baked-in generic seed YAML on any Blob failure. Parsed
adapters are cached in-process with a TTL.

Spec: feedback_adapter_yaml_blob.md + unified-docwain spec §5.5
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from src.docwain.adapters.schema import DomainAdapter


logger = logging.getLogger(__name__)


_SEED_DIR = Path(__file__).parent / "seed"


def _load_seed_yaml(domain: str) -> Optional[str]:
    path = _SEED_DIR / f"{domain}.yaml"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None


class AdapterLoader:
    def __init__(
        self,
        *,
        subscription_id: str = "",
        cache_ttl_seconds: Optional[int] = None,
        blob_prefix: Optional[str] = None,
    ):
        from src.api.config import Config
        da_cfg = getattr(Config, "DomainAdapters", None)
        self.subscription_id = subscription_id
        self.cache_ttl_seconds = (
            cache_ttl_seconds
            if cache_ttl_seconds is not None
            else (getattr(da_cfg, "CACHE_TTL_SECONDS", 300) if da_cfg else 300)
        )
        self.blob_prefix = (
            blob_prefix
            or (getattr(da_cfg, "BLOB_PREFIX", "sme_adapters") if da_cfg else "sme_adapters")
        )
        self._cache: Dict[str, tuple[float, DomainAdapter]] = {}

    def _fetch_from_blob(self, path: str) -> str:
        """Fetch a YAML text from Azure Blob. Raises FileNotFoundError if not found."""
        try:
            from azure.storage.blob import BlobServiceClient
            from src.api.config import Config
            # Connection details live in env / config; adapt to the project's usual wiring.
            # This is intentionally simple — if the project has a helper, prefer it.
            import os
            conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
            if not conn_str:
                raise RuntimeError("AZURE_STORAGE_CONNECTION_STRING not set")
            svc = BlobServiceClient.from_connection_string(conn_str)
            da_cfg = getattr(Config, "DomainAdapters", None)
            container = getattr(da_cfg, "BLOB_CONTAINER", "docwain-configs") if da_cfg else "docwain-configs"
            blob = svc.get_blob_client(container=container, blob=path)
            return blob.download_blob().readall().decode("utf-8")
        except Exception as exc:
            # Re-raise as FileNotFoundError so callers can treat "not found" and
            # "blob down" uniformly.
            raise FileNotFoundError(f"blob {path}: {exc}") from exc

    def _parse_yaml(self, yaml_text: str) -> DomainAdapter:
        data = yaml.safe_load(yaml_text) or {}
        if not isinstance(data, dict):
            raise ValueError(f"adapter YAML must be a mapping, got {type(data)}")
        return DomainAdapter(
            domain=str(data.get("domain", "generic")),
            version=str(data.get("version", "v1")),
            prompt_fragment=str(data.get("prompt_fragment", "")),
            key_entities=list(data.get("key_entities") or []),
            analysis_hints=dict(data.get("analysis_hints") or {}),
            questions_to_ask=list(data.get("questions_to_ask") or []),
        )

    def _cache_key(self, domain: str) -> str:
        return f"{self.subscription_id}:{domain}"

    def _get_cached(self, domain: str) -> Optional[DomainAdapter]:
        key = self._cache_key(domain)
        entry = self._cache.get(key)
        if not entry:
            return None
        expires_at, adapter = entry
        if time.time() > expires_at:
            self._cache.pop(key, None)
            return None
        return adapter

    def _put_cached(self, domain: str, adapter: DomainAdapter) -> None:
        self._cache[self._cache_key(domain)] = (time.time() + self.cache_ttl_seconds, adapter)

    def load(self, domain: str) -> DomainAdapter:
        """Return the best-available DomainAdapter for the given domain."""
        # Cache
        cached = self._get_cached(domain)
        if cached is not None:
            return cached

        # Subscription override
        if self.subscription_id:
            sub_path = f"{self.blob_prefix}/{self.subscription_id}/{domain}.yaml"
            try:
                text = self._fetch_from_blob(sub_path)
                adapter = self._parse_yaml(text)
                self._put_cached(domain, adapter)
                return adapter
            except FileNotFoundError:
                pass

        # Global
        global_path = f"{self.blob_prefix}/global/{domain}.yaml"
        try:
            text = self._fetch_from_blob(global_path)
            adapter = self._parse_yaml(text)
            self._put_cached(domain, adapter)
            return adapter
        except FileNotFoundError:
            pass

        # Seed fallback — only `generic` is guaranteed to exist; any other
        # domain falls back to generic.
        seed_text = _load_seed_yaml(domain) or _load_seed_yaml("generic")
        if seed_text is None:
            logger.error("No seed YAML found for domain %r and no generic seed", domain)
            adapter = DomainAdapter()  # defaults
        else:
            try:
                adapter = self._parse_yaml(seed_text)
            except Exception as exc:
                logger.error("Failed to parse seed YAML for %r: %s", domain, exc)
                adapter = DomainAdapter()
        self._put_cached(domain, adapter)
        return adapter
```

#### Step 6 — Run tests

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/pytest tests/unit/docwain/test_adapter_loader.py -x -q
```

Expected: 5 passed.

#### Step 7 — Commit

```bash
git add src/docwain/adapters/__init__.py src/docwain/adapters/schema.py \
        src/docwain/adapters/loader.py src/docwain/adapters/seed/generic.yaml \
        src/api/config.py tests/unit/docwain/test_adapter_loader.py
git commit -m "unified: domain adapter framework — Blob YAML loader with TTL cache + generic seed"
```

No Claude/Anthropic/Co-Authored-By.

---

### Task 4: Chart / DOCWAIN_VIZ generation prompt (TDD)

**Files:**
- Create: `src/docwain/prompts/chart_generation.py`
- Create: `tests/unit/docwain/test_chart_generation.py`

The module provides:
- `CHART_GENERATION_SYSTEM_PROMPT` — instruction for DocWain to emit a `DOCWAIN_VIZ` JSON block when the query warrants a chart.
- `should_emit_chart(query: str) -> bool` — heuristic matcher: queries asking for charts, comparisons, or aggregations.
- `extract_viz_block(response_text: str) -> Optional[dict]` — parses `<!--DOCWAIN_VIZ ... -->` comments from model output (existing Reasoner already emits them; this module makes parsing canonical).

The Reasoner's existing chart emission is preserved; this module makes the behavior testable and reusable.

#### Step 1 — Failing tests

Create `tests/unit/docwain/test_chart_generation.py`:

```python
import json

from src.docwain.prompts.chart_generation import (
    CHART_GENERATION_SYSTEM_PROMPT,
    extract_viz_block,
    should_emit_chart,
)


def test_prompt_is_non_empty():
    assert len(CHART_GENERATION_SYSTEM_PROMPT) > 100
    assert "DOCWAIN_VIZ" in CHART_GENERATION_SYSTEM_PROMPT


def test_should_emit_chart_for_comparison_query():
    assert should_emit_chart("Compare revenue between Q1 and Q2") is True
    assert should_emit_chart("show me a chart of monthly expenses") is True
    assert should_emit_chart("plot the trend over time") is True


def test_should_not_emit_chart_for_factual_query():
    assert should_emit_chart("What is the candidate's name?") is False
    assert should_emit_chart("List the vendors") is False


def test_extract_viz_block_parses_html_comment():
    response = """
Some text before.
<!--DOCWAIN_VIZ
{"chart_type": "bar", "title": "Expenses", "labels": ["Jan", "Feb"], "values": [100, 200], "unit": "USD"}
-->
Some text after.
"""
    viz = extract_viz_block(response)
    assert viz is not None
    assert viz["chart_type"] == "bar"
    assert viz["labels"] == ["Jan", "Feb"]
    assert viz["values"] == [100, 200]


def test_extract_viz_block_returns_none_when_absent():
    assert extract_viz_block("no viz here") is None


def test_extract_viz_block_tolerates_malformed_json():
    response = "<!--DOCWAIN_VIZ\n{malformed json}\n-->"
    assert extract_viz_block(response) is None  # malformed → None, not raise
```

#### Step 2 — Implement

Create `src/docwain/prompts/chart_generation.py`:

```python
"""Chart / DOCWAIN_VIZ generation support.

DocWain already emits `<!--DOCWAIN_VIZ ... -->` blocks from the existing
Reasoner system prompt. This module adds:

- A dedicated system-prompt fragment that can be appended to the Reasoner
  prompt when a query warrants a chart (routed by `should_emit_chart`).
- A canonical parser for `DOCWAIN_VIZ` blocks (used by the frontend and by
  tests).
- A heuristic query classifier that determines when a chart is appropriate.

Spec: 2026-04-24-unified-docwain-engineering-layer-design.md §5.4
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional


CHART_GENERATION_SYSTEM_PROMPT = (
    "When the user's question asks for a comparison, trend, distribution, or "
    "aggregation across multiple values that can be visualized, emit a "
    "DOCWAIN_VIZ block after your natural-language answer. Format:\n\n"
    "<!--DOCWAIN_VIZ\n"
    "{\n"
    '  "chart_type": "bar" | "line" | "pie" | "horizontal_bar" | "table",\n'
    '  "title": string,\n'
    '  "labels": [string, ...],\n'
    '  "values": [number, ...] or [[number, ...], ...] for multi-series,\n'
    '  "unit": string (e.g., "USD", "%", "count")\n'
    "}\n"
    "-->\n\n"
    "Rules:\n"
    "- Emit DOCWAIN_VIZ only when the data is meaningfully visualizable. Do "
    "  not emit for single-value facts or yes/no answers.\n"
    "- Use only values grounded in the retrieved documents. Do not fabricate.\n"
    "- If the chart would have fewer than 2 data points, omit it."
)


_CHART_KEYWORDS = (
    r"\bcompare\b",
    r"\bcomparison\b",
    r"\bchart\b",
    r"\bgraph\b",
    r"\bplot\b",
    r"\btrend\b",
    r"\bover time\b",
    r"\bmonthly\b",
    r"\byearly\b",
    r"\bquarterly\b",
    r"\bdistribution\b",
    r"\bbreakdown\b",
    r"\bversus\b",
    r"\bvs\.?\b",
)

_CHART_RE = re.compile("|".join(_CHART_KEYWORDS), re.IGNORECASE)


def should_emit_chart(query: str) -> bool:
    """Heuristic: does this query warrant a chart / DOCWAIN_VIZ in the response?"""
    if not query:
        return False
    return bool(_CHART_RE.search(query))


_VIZ_BLOCK_RE = re.compile(r"<!--\s*DOCWAIN_VIZ\s*(.+?)\s*-->", re.DOTALL)


def extract_viz_block(response_text: str) -> Optional[Dict[str, Any]]:
    """Extract a DOCWAIN_VIZ JSON payload from an HTML comment block. None on failure."""
    if not response_text:
        return None
    match = _VIZ_BLOCK_RE.search(response_text)
    if not match:
        return None
    payload_text = match.group(1).strip()
    try:
        data = json.loads(payload_text)
        if isinstance(data, dict):
            return data
        return None
    except Exception:
        return None
```

#### Step 3 — Verify + commit

```bash
cd /home/ubuntu/.config/superpowers/worktrees/DocWain/preprod_v02
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/pytest tests/unit/docwain/test_chart_generation.py -x -q

git add src/docwain/prompts/chart_generation.py tests/unit/docwain/test_chart_generation.py
git commit -m "unified: chart generation prompt + DOCWAIN_VIZ parser + query heuristic"
```

Expected: 6 passed. No Claude/Anthropic/Co-Authored-By.

---

### Task 5: Full-suite + bench + smoke validation

Not a code task.

- [ ] Step 1 — Full suites

```bash
cd /home/ubuntu/.config/superpowers/worktrees/DocWain/preprod_v02
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/pytest tests/unit/extraction tests/unit/kg tests/unit/api tests/unit/llm tests/unit/docwain tests/integration -q --timeout=30 2>&1 | tail -8
```

Expected: count higher than Wave 1 (414); ~20 new tests added across Tasks 1/3/4.

- [ ] Step 2 — Bench

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/python -m tests.extraction_bench.runner 2>&1 | tail -12
```

Expected: 7/7 at 1.000 (unchanged).

- [ ] Step 3 — Broader sanity

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/pytest tests/unit -q --timeout=30 2>&1 | tail -8
```

Expected: the 2 pre-existing unrelated failures still expected; no new failures.

- [ ] Step 4 — Config + imports smoke

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/python -c "
from src.docwain.logging_config import apply_log_hygiene, HealthCheckPathFilter
from src.docwain.adapters.loader import AdapterLoader
from src.docwain.adapters.schema import DomainAdapter
from src.docwain.prompts.chart_generation import should_emit_chart, extract_viz_block, CHART_GENERATION_SYSTEM_PROMPT
from src.api.config import Config
assert hasattr(Config, 'DomainAdapters')
assert isinstance(Config.DomainAdapters.BLOB_PREFIX, str)
print('WAVE 2 IMPORTS + CONFIG OK')
apply_log_hygiene()
print('log hygiene applied')
loader = AdapterLoader(subscription_id='test')
a = loader.load('generic')
print(f'generic adapter domain={a.domain} version={a.version} key_entities={len(a.key_entities)}')
"
```

Expected: `WAVE 2 IMPORTS + CONFIG OK`, `log hygiene applied`, and generic adapter loaded with non-zero key_entities.

No commit — validation only.

---

## Self-review

**Spec coverage:**
- §5.4 Chart gen: Task 4 ✓
- §5.5 Domain adapter framework: Task 3 ✓
- Log hygiene (analysis-driven): Task 1 ✓
- qwen3:14b fix (analysis-driven): Task 2 ✓
- §5.2 Central prompt registry — deferred to Wave 2b ⏭
- Researcher weekend refresh — deferred to Wave 2b ⏭

**Placeholder scan:** none. Every step has concrete code or exact commands.

**Type consistency:**
- `DomainAdapter` defined in Task 3 Step 2, used in Task 3 Step 5.
- `HealthCheckPathFilter` / `apply_log_hygiene` defined in Task 1 Step 3.
- `CHART_GENERATION_SYSTEM_PROMPT` / `should_emit_chart` / `extract_viz_block` defined in Task 4.

Plan complete.
