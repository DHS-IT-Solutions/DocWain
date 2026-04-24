# Unified DocWain — Wave 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver three engineering-only features that complete the "unified DocWain" vision without any training: (1) gateway routed to vLLM local primary with identity shim; (2) entity + relation extraction in the KG pipeline; (3) Researcher Agent as a third parallel task in the training stage.

**Architecture:** Task-specific prompts under `src/docwain/prompts/`. Gateway swaps primary from Ollama Cloud → vLLM local and prepends an identity-shim system prefix. KG adapter calls a new entity extractor before building its payload. A new Celery task `run_researcher_agent` dispatches alongside `embed_document` and `build_knowledge_graph` from `trigger_embedding`, writes insights to Qdrant + Neo4j, isolated from pipeline_status. Every component is feature-flag gated with env-var rollback.

**Tech Stack:** Python 3.12, Celery (existing brokers + workers), Redis (existing observability), Neo4j (existing store), Qdrant (existing vector + payload store), vLLM (existing serving). Zero new external dependencies.

**Non-goals:** no training runs, no chart-gen improvements (Wave 2), no domain adapter framework (Wave 2), no central prompt registry refactor (Wave 2), no weekend-refresh scheduler (Wave 2).

Spec: `docs/superpowers/specs/2026-04-24-unified-docwain-engineering-layer-design.md`.

---

## File structure

**New files:**
- `src/docwain/__init__.py` (one-line comment)
- `src/docwain/prompts/__init__.py` (one-line comment)
- `src/docwain/prompts/entity_extraction.py` — system prompt + `parse_entity_response`
- `src/docwain/prompts/researcher.py` — system prompt + output schema + `parse_researcher_response`
- `src/tasks/researcher.py` — Celery task `run_researcher_agent`
- `tests/unit/docwain/__init__.py`
- `tests/unit/docwain/test_entity_extraction.py`
- `tests/unit/docwain/test_researcher_prompt.py`
- `tests/unit/api/test_pipeline_api_researcher_dispatch.py`
- `tests/unit/kg/test_entity_enrichment.py`
- `tests/integration/test_researcher_isolation.py`

**Modified files:**
- `src/llm/gateway.py` — swap primary order; add identity shim
- `src/api/config.py` — add `Config.Model.PRIMARY_BACKEND`, `Config.Model.IDENTITY_SHIM_ENABLED`, `Config.Model.IDENTITY_SHIM_TEXT`, `Config.KG.ENTITY_EXTRACTION_ENABLED`, `Config.Researcher.ENABLED`
- `src/api/statuses.py` — add `RESEARCHER_PENDING`, `RESEARCHER_IN_PROGRESS`, `RESEARCHER_COMPLETED`, `RESEARCHER_FAILED`
- `src/tasks/kg.py` — `_canonical_to_graph_payload` optionally calls entity extraction
- `src/api/pipeline_api.py` — `trigger_embedding` also dispatches `run_researcher_agent`
- `src/celery_app.py` — register `researcher_queue` + route `src.tasks.researcher.run_researcher_agent`

**Git:** continue on branch `preprod_v02`. One commit per task.

---

### Task 1: Gateway primary swap to vLLM + identity shim (TDD)

**Files:**
- Modify: `src/llm/gateway.py`
- Modify: `src/api/config.py`
- Create: `tests/unit/llm/__init__.py` (if missing — `# llm unit tests\n`)
- Create: `tests/unit/llm/test_gateway_primary_and_shim.py`

- [ ] **Step 1: Inspect current state**

```bash
cd /home/ubuntu/.config/superpowers/worktrees/DocWain/preprod_v02
grep -n "class Model\|class VLLM\|class Researcher\|class KG\b" src/api/config.py
sed -n '1,50p' src/llm/gateway.py
sed -n '80,180p' src/llm/gateway.py
```

Note: (a) the current `_init_clients()` order (Ollama Cloud primary, Azure fallback); (b) the `Config` class structure so new entries slot in correctly.

- [ ] **Step 2: Add config flags**

In `src/api/config.py`, under the appropriate `Config` inner class (or add a new `Model` nested class if one doesn't exist):

```python
class Model:
    PRIMARY_BACKEND = os.getenv("DOCWAIN_MODEL_PRIMARY_BACKEND", "vllm")
    IDENTITY_SHIM_ENABLED = os.getenv("DOCWAIN_MODEL_IDENTITY_SHIM_ENABLED", "true").lower() == "true"
    IDENTITY_SHIM_TEXT = os.getenv(
        "DOCWAIN_MODEL_IDENTITY_SHIM_TEXT",
        "You are DocWain, an enterprise document research agent. "
        "Respond grounded in the provided documents. Be accurate and concise.",
    )
```

Also add placeholders for the flags Tasks 2 and 3 need (add them all here so Tasks 2/3 don't re-modify config.py):

```python
class KG:
    # ... existing fields ...
    ENTITY_EXTRACTION_ENABLED = os.getenv("DOCWAIN_KG_ENTITY_EXTRACTION_ENABLED", "true").lower() == "true"


class Researcher:
    ENABLED = os.getenv("DOCWAIN_RESEARCHER_ENABLED", "true").lower() == "true"
    MAX_TOKENS = int(os.getenv("DOCWAIN_RESEARCHER_MAX_TOKENS", "4096"))
```

If `Config.KG` doesn't already exist, add a minimal class with just the new flag.

- [ ] **Step 3: Write failing test**

Create `tests/unit/llm/test_gateway_primary_and_shim.py`:

```python
"""Gateway primary backend swap + identity shim behavior.

Verifies:
- Config.Model.PRIMARY_BACKEND == 'vllm' → gateway._primary is the vLLM client (OpenAICompatibleClient).
- Config.Model.PRIMARY_BACKEND == 'cloud' → gateway._primary is OllamaClient.
- Config.Model.IDENTITY_SHIM_ENABLED == True → outgoing calls have the shim prepended to system prompt.
- Config.Model.IDENTITY_SHIM_ENABLED == False → system prompt unmodified.
"""
from unittest.mock import MagicMock, patch


def test_gateway_uses_vllm_primary_when_flag_is_vllm(monkeypatch):
    monkeypatch.setenv("DOCWAIN_MODEL_PRIMARY_BACKEND", "vllm")
    # Reload Config + gateway modules to pick up env var
    import importlib
    from src.api import config as cfg_mod
    importlib.reload(cfg_mod)
    from src.llm import gateway as gw_mod
    importlib.reload(gw_mod)

    gw = gw_mod.LLMGateway()
    # Primary client should be OpenAICompatibleClient (vLLM) — check class name
    primary_name = type(gw._primary).__name__ if gw._primary else None
    assert primary_name == "OpenAICompatibleClient", f"expected OpenAICompatibleClient, got {primary_name!r}"


def test_gateway_uses_cloud_primary_when_flag_is_cloud(monkeypatch):
    monkeypatch.setenv("DOCWAIN_MODEL_PRIMARY_BACKEND", "cloud")
    import importlib
    from src.api import config as cfg_mod
    importlib.reload(cfg_mod)
    from src.llm import gateway as gw_mod
    importlib.reload(gw_mod)

    gw = gw_mod.LLMGateway()
    primary_name = type(gw._primary).__name__ if gw._primary else None
    assert primary_name == "OllamaClient", f"expected OllamaClient, got {primary_name!r}"


def test_identity_shim_prepended_when_enabled(monkeypatch):
    monkeypatch.setenv("DOCWAIN_MODEL_IDENTITY_SHIM_ENABLED", "true")
    monkeypatch.setenv("DOCWAIN_MODEL_IDENTITY_SHIM_TEXT", "You are DocWain (test).")
    import importlib
    from src.api import config as cfg_mod
    importlib.reload(cfg_mod)
    from src.llm import gateway as gw_mod
    importlib.reload(gw_mod)

    # Capture the system prompt that reaches the underlying client.
    captured_system = {}

    def fake_generate(self, *args, **kwargs):
        captured_system["system"] = kwargs.get("system") or (args[1] if len(args) > 1 else None)
        return "ok"

    def fake_generate_with_metadata(self, *args, **kwargs):
        captured_system["system"] = kwargs.get("system") or (args[1] if len(args) > 1 else None)
        m = MagicMock()
        m.text = "ok"
        return m

    # Monkeypatch the primary's .generate / .generate_with_metadata — whichever the gateway calls.
    from src.llm import clients as clients_mod
    monkeypatch.setattr(clients_mod.OpenAICompatibleClient, "generate", fake_generate, raising=False)
    monkeypatch.setattr(clients_mod.OpenAICompatibleClient, "generate_with_metadata", fake_generate_with_metadata, raising=False)
    monkeypatch.setattr(clients_mod.OllamaClient, "generate", fake_generate, raising=False)
    monkeypatch.setattr(clients_mod.OllamaClient, "generate_with_metadata", fake_generate_with_metadata, raising=False)

    gw = gw_mod.LLMGateway()
    # Call with a bare system prompt; shim should prepend to it.
    try:
        gw.generate(prompt="hi", system="custom system prompt here")
    except Exception:
        # If gateway.generate signature differs, try generate_with_metadata
        gw.generate_with_metadata(prompt="hi", system="custom system prompt here")

    seen_system = captured_system.get("system") or ""
    assert "You are DocWain (test)" in seen_system, f"shim not prepended; saw {seen_system[:200]!r}"
    assert "custom system prompt here" in seen_system, "original system dropped"


def test_identity_shim_absent_when_disabled(monkeypatch):
    monkeypatch.setenv("DOCWAIN_MODEL_IDENTITY_SHIM_ENABLED", "false")
    monkeypatch.setenv("DOCWAIN_MODEL_IDENTITY_SHIM_TEXT", "You are DocWain (test).")
    import importlib
    from src.api import config as cfg_mod
    importlib.reload(cfg_mod)
    from src.llm import gateway as gw_mod
    importlib.reload(gw_mod)

    captured_system = {}

    def fake_generate(self, *args, **kwargs):
        captured_system["system"] = kwargs.get("system") or (args[1] if len(args) > 1 else None)
        return "ok"

    def fake_generate_with_metadata(self, *args, **kwargs):
        captured_system["system"] = kwargs.get("system") or (args[1] if len(args) > 1 else None)
        m = MagicMock(); m.text = "ok"
        return m

    from src.llm import clients as clients_mod
    monkeypatch.setattr(clients_mod.OpenAICompatibleClient, "generate", fake_generate, raising=False)
    monkeypatch.setattr(clients_mod.OpenAICompatibleClient, "generate_with_metadata", fake_generate_with_metadata, raising=False)
    monkeypatch.setattr(clients_mod.OllamaClient, "generate", fake_generate, raising=False)
    monkeypatch.setattr(clients_mod.OllamaClient, "generate_with_metadata", fake_generate_with_metadata, raising=False)

    gw = gw_mod.LLMGateway()
    try:
        gw.generate(prompt="hi", system="custom system prompt here")
    except Exception:
        gw.generate_with_metadata(prompt="hi", system="custom system prompt here")

    seen_system = captured_system.get("system") or ""
    assert "You are DocWain (test)" not in seen_system
    assert "custom system prompt here" in seen_system
```

- [ ] **Step 4: Run to confirm fail**

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/pytest tests/unit/llm/test_gateway_primary_and_shim.py -x -q
```

Expected: tests fail (gateway doesn't yet route by flag or prepend shim).

- [ ] **Step 5: Modify `_init_clients()`**

In `src/llm/gateway.py`, locate `_init_clients()`. Rewrite the initialization order to be driven by `Config.Model.PRIMARY_BACKEND`:

```python
def _init_clients(self) -> None:
    from src.api.config import Config
    primary_backend = getattr(getattr(Config, "Model", None), "PRIMARY_BACKEND", "vllm")

    # Build all three potential clients (best-effort; each may fail to initialize).
    vllm_client = None
    cloud_client = None
    azure_client = None

    # vLLM local
    try:
        from src.llm.clients import OpenAICompatibleClient
        vllm_endpoint = getattr(getattr(Config, "VLLM", None), "VLLM_ENDPOINT",
                                "http://localhost:8100/v1/chat/completions")
        vllm_model = getattr(getattr(Config, "VLLM", None), "VLLM_MODEL_NAME", "docwain-fast")
        vllm_client = OpenAICompatibleClient(endpoint=vllm_endpoint, model_name=vllm_model)
        logger.info("vLLM local client initialised (model=%s)", vllm_model)
    except Exception as exc:
        logger.warning("vLLM local client init failed: %s", exc)

    # Ollama Cloud
    try:
        from src.llm.clients import OllamaClient
        cloud_model = os.getenv("OLLAMA_CLOUD_MODEL") or os.getenv("OLLAMA_MODEL") or "qwen3.5:397b"
        cloud_client = OllamaClient(model_name=cloud_model)
        logger.info("Ollama Cloud client initialised (model=%s)", cloud_model)
    except Exception as exc:
        logger.warning("Ollama Cloud client init failed: %s", exc)

    # Azure GPT-4o
    try:
        from src.llm.clients import OpenAIClient
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
        if endpoint and api_key and deployment:
            azure_client = OpenAIClient(
                endpoint=endpoint, api_key=api_key,
                deployment_name=deployment,
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview"),
            )
            logger.info("Azure GPT-4o client initialised (model=%s)", deployment)
    except Exception as exc:
        logger.warning("Azure GPT-4o client init failed: %s", exc)

    # Order primary/fallback by flag
    if primary_backend == "vllm" and vllm_client is not None:
        self._primary = vllm_client
        self._fallback = cloud_client or azure_client
    elif primary_backend == "cloud" and cloud_client is not None:
        self._primary = cloud_client
        self._fallback = azure_client or vllm_client
    elif primary_backend == "azure" and azure_client is not None:
        self._primary = azure_client
        self._fallback = cloud_client or vllm_client
    else:
        # Requested backend unavailable; fall back to any available in preferred order.
        self._primary = vllm_client or cloud_client or azure_client
        self._fallback = cloud_client if self._primary is not cloud_client else (azure_client or vllm_client)

    self.model_name = getattr(self._primary, "model_name", "unknown") if self._primary else "none"
    if self._primary is None:
        raise RuntimeError("No LLM backend available — vLLM, Cloud, and Azure all failed to initialize")
```

Preserve any existing `_health_monitor` setup, stats dicts, and other side effects in `_init_clients` that exist today — this rewrite only replaces the primary-selection logic.

- [ ] **Step 6: Add shim-prepending wrapper on `generate` / `generate_with_metadata`**

In the `LLMGateway` class, find the `generate` and `generate_with_metadata` methods (or whatever the top-level call sites are named). Wrap the `system` kwarg with the shim when enabled. Minimal diff: at the top of each method, add:

```python
def _apply_identity_shim(self, system: str | None) -> str | None:
    from src.api.config import Config
    model_cfg = getattr(Config, "Model", None)
    if not model_cfg or not getattr(model_cfg, "IDENTITY_SHIM_ENABLED", False):
        return system
    shim = getattr(model_cfg, "IDENTITY_SHIM_TEXT", "") or ""
    if not shim.strip():
        return system
    if system:
        return f"{shim}\n\n{system}"
    return shim
```

Inside `generate` / `generate_with_metadata`, before calling the underlying client:

```python
    if "system" in kwargs:
        kwargs["system"] = self._apply_identity_shim(kwargs.get("system"))
    # ... rest of existing call logic unchanged
```

If the method signature has `system` as a positional arg, similarly prepend. Inspect the real signature and adapt.

- [ ] **Step 7: Run tests to verify pass**

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/pytest tests/unit/llm/test_gateway_primary_and_shim.py -x -q
```

Expected: 4 passed. If any test fails due to signature mismatch, adapt the test to match the real signature but KEEP the behavioral assertions (primary type + shim presence/absence).

- [ ] **Step 8: Sanity — broader test suite**

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/pytest tests/unit -q --timeout=30 2>&1 | tail -6
```

Expected: no new failures vs baseline (2 pre-existing unrelated finetune failures still expected).

- [ ] **Step 9: Commit**

```bash
git add src/llm/gateway.py src/api/config.py tests/unit/llm/__init__.py tests/unit/llm/test_gateway_primary_and_shim.py
git commit -m "unified: gateway primary = vLLM local, identity shim prepended to system prompt (flag-gated)"
```

No Claude/Anthropic/Co-Authored-By.

---

### Task 2: Entity extraction module + KG wiring (TDD)

**Files:**
- Create: `src/docwain/__init__.py` (one-line comment)
- Create: `src/docwain/prompts/__init__.py` (one-line comment)
- Create: `src/docwain/prompts/entity_extraction.py`
- Modify: `src/tasks/kg.py` (`_canonical_to_graph_payload` optionally calls extractor)
- Create: `tests/unit/docwain/__init__.py`, `tests/unit/docwain/test_entity_extraction.py`
- Create: `tests/unit/kg/test_entity_enrichment.py`

- [ ] **Step 1: Package markers**

```bash
cd /home/ubuntu/.config/superpowers/worktrees/DocWain/preprod_v02
mkdir -p src/docwain/prompts tests/unit/docwain
[ -f src/docwain/__init__.py ] || printf '# DocWain unified engineering layer\n' > src/docwain/__init__.py
[ -f src/docwain/prompts/__init__.py ] || printf '# task-specific DocWain prompts\n' > src/docwain/prompts/__init__.py
[ -f tests/unit/docwain/__init__.py ] || printf '# docwain unit tests\n' > tests/unit/docwain/__init__.py
```

- [ ] **Step 2: Failing test for prompt parser**

Create `tests/unit/docwain/test_entity_extraction.py`:

```python
import json

from src.docwain.prompts.entity_extraction import (
    ENTITY_EXTRACTION_SYSTEM_PROMPT,
    ExtractedEntities,
    parse_entity_response,
)


def test_parse_entity_response_well_formed():
    text = json.dumps({
        "entities": [
            {"text": "Acme Corp", "type": "ORGANIZATION", "confidence": 0.9},
            {"text": "John Doe", "type": "PERSON", "confidence": 0.85},
        ],
        "relationships": [
            {"source": "John Doe", "target": "Acme Corp", "type": "WORKS_AT"},
        ],
    })
    out = parse_entity_response(text)
    assert isinstance(out, ExtractedEntities)
    assert len(out.entities) == 2
    assert out.entities[0]["text"] == "Acme Corp"
    assert len(out.relationships) == 1
    assert out.relationships[0]["type"] == "WORKS_AT"


def test_parse_entity_response_tolerates_code_fences():
    text = "```json\n" + json.dumps({"entities": [{"text": "X", "type": "T"}], "relationships": []}) + "\n```"
    out = parse_entity_response(text)
    assert len(out.entities) == 1


def test_parse_entity_response_returns_empty_on_garbage():
    out = parse_entity_response("not json at all")
    assert out.entities == []
    assert out.relationships == []


def test_system_prompt_non_empty():
    assert isinstance(ENTITY_EXTRACTION_SYSTEM_PROMPT, str)
    assert len(ENTITY_EXTRACTION_SYSTEM_PROMPT) > 100
    assert "entities" in ENTITY_EXTRACTION_SYSTEM_PROMPT.lower()
```

- [ ] **Step 3: Run to confirm fail, then implement**

Create `src/docwain/prompts/entity_extraction.py`:

```python
"""Entity + relation extraction prompt for DocWain.

Called by `src.tasks.kg._canonical_to_graph_payload` before building the
GraphIngestPayload. Empty-on-error fallback keeps KG ingestion resilient.

Spec: 2026-04-24-unified-docwain-engineering-layer-design.md §5.3
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List


ENTITY_EXTRACTION_SYSTEM_PROMPT = (
    "You extract named entities and relationships from document text. Given the "
    "concatenated text of a document, return a JSON object describing the "
    "entities present and their relationships.\n\n"
    "Entity types to use: PERSON, ORGANIZATION, LOCATION, DATE, MONEY, PRODUCT, "
    "EVENT, DOCUMENT_REF, OTHER. Set type=OTHER for anything that doesn't fit.\n\n"
    "Output ONLY valid JSON (no prose, no markdown fences):\n"
    "{\n"
    '  "entities": [\n'
    '    { "text": string, "type": string, "confidence": number 0..1 }\n'
    "  ],\n"
    '  "relationships": [\n'
    '    { "source": string, "target": string, "type": string }\n'
    "  ]\n"
    "}\n\n"
    "Rules:\n"
    "- Extract only entities that are explicitly mentioned in the provided text.\n"
    "- Do not infer or synthesize entities not in the text.\n"
    "- For relationships, `source` and `target` must match entity `text` values "
    "exactly.\n"
    "- Confidence reflects how unambiguous the mention is (higher = clearer)."
)


@dataclass
class ExtractedEntities:
    entities: List[Dict[str, Any]] = field(default_factory=list)
    relationships: List[Dict[str, Any]] = field(default_factory=list)


def _strip_code_fence(text: str) -> str:
    t = text.strip()
    m = re.match(r"```(?:json)?\s*(.*?)```", t, flags=re.DOTALL)
    return m.group(1).strip() if m else t


def _first_json_object(text: str) -> str:
    t = _strip_code_fence(text)
    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return t
    return t[start : end + 1]


def parse_entity_response(text: str) -> ExtractedEntities:
    """Best-effort parse; empty on failure."""
    try:
        data = json.loads(_first_json_object(text))
    except Exception:
        return ExtractedEntities()
    entities = data.get("entities") or []
    relationships = data.get("relationships") or []
    if not isinstance(entities, list):
        entities = []
    if not isinstance(relationships, list):
        relationships = []
    return ExtractedEntities(entities=entities, relationships=relationships)


def build_user_prompt(*, document_text: str, max_chars: int = 16000) -> str:
    """Build the user-prompt payload for the entity extractor call."""
    truncated = document_text[:max_chars]
    return f"Document text:\n\n{truncated}\n\nReturn the entity + relationship JSON."
```

Run: `/home/ubuntu/PycharmProjects/DocWain/.venv/bin/pytest tests/unit/docwain/test_entity_extraction.py -x -q`.

Expected: 4 passed.

- [ ] **Step 4: Failing test for KG wiring**

Create `tests/unit/kg/test_entity_enrichment.py`:

```python
"""Verify `_canonical_to_graph_payload` calls entity extraction when enabled.

Flag-gated: with `Config.KG.ENTITY_EXTRACTION_ENABLED=True`, the adapter
enriches payload.entities from a DocWain call. With flag off, behaves as today.
"""
from unittest.mock import MagicMock


def _canonical_extraction_with_text() -> dict:
    return {
        "doc_id": "doc-e1",
        "format": "pdf_native",
        "path_taken": "native",
        "pages": [
            {
                "page_num": 1,
                "blocks": [
                    {"text": "Acme Corp invoice for $1,000 dated 2025-01-15.", "block_type": "paragraph"},
                ],
                "tables": [],
                "images": [],
            }
        ],
        "sheets": [],
        "slides": [],
        "metadata": {"doc_intel": {"doc_type_hint": "invoice"}, "coverage": {}, "extraction_version": "v1"},
    }


def test_entity_enrichment_calls_docwain_and_populates_entities(monkeypatch):
    """When flag ON, extractor is called and result populates payload.entities."""
    from src.api import config as cfg_mod
    monkeypatch.setattr(cfg_mod.Config.KG, "ENTITY_EXTRACTION_ENABLED", True, raising=False)

    # Stub the gateway.generate_with_metadata used by the extractor helper.
    from src.docwain.prompts import entity_extraction as ee
    import src.tasks.kg as kg_mod

    def fake_extract(document_text: str) -> ee.ExtractedEntities:
        return ee.ExtractedEntities(
            entities=[
                {"text": "Acme Corp", "type": "ORGANIZATION", "confidence": 0.9},
            ],
            relationships=[],
        )

    monkeypatch.setattr(kg_mod, "_call_entity_extractor", fake_extract, raising=False)

    payload = kg_mod._canonical_to_graph_payload(
        extraction=_canonical_extraction_with_text(),
        screening=None,
        subscription_id="sub-x",
        profile_id="prof-x",
        document_id="doc-e1",
    )
    assert len(payload.entities) == 1
    assert payload.entities[0]["text"] == "Acme Corp"


def test_entity_enrichment_skipped_when_flag_off(monkeypatch):
    """When flag OFF, entities stay empty (Plan 3 baseline behavior)."""
    from src.api import config as cfg_mod
    monkeypatch.setattr(cfg_mod.Config.KG, "ENTITY_EXTRACTION_ENABLED", False, raising=False)

    import src.tasks.kg as kg_mod

    called = {"n": 0}

    def fake_extract(document_text: str):
        called["n"] += 1
        from src.docwain.prompts.entity_extraction import ExtractedEntities
        return ExtractedEntities(entities=[{"text": "SHOULD_NOT_APPEAR", "type": "OTHER"}], relationships=[])

    monkeypatch.setattr(kg_mod, "_call_entity_extractor", fake_extract, raising=False)

    payload = kg_mod._canonical_to_graph_payload(
        extraction=_canonical_extraction_with_text(),
        screening=None,
        subscription_id="sub-x",
        profile_id="prof-x",
        document_id="doc-e1",
    )
    assert payload.entities == []
    assert called["n"] == 0


def test_entity_enrichment_swallows_extractor_failures(monkeypatch):
    """When the extractor raises, payload is still valid with empty entities."""
    from src.api import config as cfg_mod
    monkeypatch.setattr(cfg_mod.Config.KG, "ENTITY_EXTRACTION_ENABLED", True, raising=False)

    import src.tasks.kg as kg_mod

    def boom(document_text: str):
        raise RuntimeError("gateway down")

    monkeypatch.setattr(kg_mod, "_call_entity_extractor", boom, raising=False)

    payload = kg_mod._canonical_to_graph_payload(
        extraction=_canonical_extraction_with_text(),
        screening=None,
        subscription_id="sub-x",
        profile_id="prof-x",
        document_id="doc-e1",
    )
    # Empty entities, no exception raised — KG continues with Document + mentions only.
    assert payload.entities == []
```

- [ ] **Step 5: Run to confirm fail**

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/pytest tests/unit/kg/test_entity_enrichment.py -x -q
```

Expected: tests FAIL — `_call_entity_extractor` doesn't exist yet, `Config.KG.ENTITY_EXTRACTION_ENABLED` may not exist.

- [ ] **Step 6: Implement `_call_entity_extractor` and wire into `_canonical_to_graph_payload`**

In `src/tasks/kg.py`, add a module-level helper (place it above `_canonical_to_graph_payload`):

```python
def _call_entity_extractor(document_text: str) -> "ExtractedEntities":
    """Call DocWain to extract entities+relationships from concatenated doc text.

    Returns an ExtractedEntities with empty lists on any failure. Never raises.

    Spec: 2026-04-24-unified-docwain-engineering-layer-design.md §5.3
    """
    from src.docwain.prompts.entity_extraction import (
        ENTITY_EXTRACTION_SYSTEM_PROMPT,
        ExtractedEntities,
        build_user_prompt,
        parse_entity_response,
    )
    if not document_text or not document_text.strip():
        return ExtractedEntities()
    try:
        from src.llm.gateway import LLMGateway
        gw = LLMGateway()
        user = build_user_prompt(document_text=document_text)
        result = gw.generate_with_metadata(
            prompt=user,
            system=ENTITY_EXTRACTION_SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=2048,
        )
        text = getattr(result, "text", None) or ""
        return parse_entity_response(text)
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("entity extractor failed: %s", exc)
        return ExtractedEntities()
```

In `_canonical_to_graph_payload`, immediately BEFORE the `return GraphIngestPayload(...)` line, add:

```python
    # Plan 4 wave 1: optionally enrich with entities via DocWain.
    try:
        from src.api.config import Config
        kg_cfg = getattr(Config, "KG", None)
        entity_extraction_enabled = getattr(kg_cfg, "ENTITY_EXTRACTION_ENABLED", False) if kg_cfg else False
    except Exception:
        entity_extraction_enabled = False

    enriched_entities: List[Dict[str, Any]] = []
    enriched_relationships: List[Dict[str, Any]] = []
    if entity_extraction_enabled:
        try:
            # Concatenate all mention text into a single doc blob for extraction
            doc_text = "\n\n".join(
                (m.get("text") if isinstance(m, dict) else getattr(m, "text", "")) or ""
                for m in mentions
            ).strip()
            extracted = _call_entity_extractor(doc_text)
            enriched_entities = list(extracted.entities or [])
            enriched_relationships = list(extracted.relationships or [])
        except Exception:
            # Safety net — the helper already swallows, but double-guard.
            pass
```

Then update the `GraphIngestPayload(...)` constructor call to pass `entities=enriched_entities` and `typed_relationships=enriched_relationships` instead of empty lists.

Note: if `mentions` in the real code is a list of `GraphMention` dataclass instances (not dicts), use `getattr(m, "text", "")` — the sketch already handles both.

- [ ] **Step 7: Run tests**

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/pytest tests/unit/docwain/test_entity_extraction.py tests/unit/kg/test_entity_enrichment.py tests/unit/kg -x -q
```

Expected: all pass (4 from entity_extraction + 3 new from enrichment + prior kg tests).

- [ ] **Step 8: Commit**

```bash
git add src/docwain/__init__.py src/docwain/prompts/__init__.py src/docwain/prompts/entity_extraction.py \
        tests/unit/docwain/__init__.py tests/unit/docwain/test_entity_extraction.py \
        tests/unit/kg/test_entity_enrichment.py src/tasks/kg.py
git commit -m "unified: entity + relationship extraction via DocWain; KG adapter enriches canonical payload (flag-gated)"
```

---

### Task 3: Researcher Agent — prompt, task, dispatch (TDD)

**Files:**
- Create: `src/docwain/prompts/researcher.py`
- Create: `tests/unit/docwain/test_researcher_prompt.py`
- Create: `src/tasks/researcher.py`
- Modify: `src/celery_app.py` (register queue + route)
- Modify: `src/api/pipeline_api.py::trigger_embedding` (dispatch researcher in parallel)
- Modify: `src/api/statuses.py` (add `RESEARCHER_*` values)
- Create: `tests/unit/api/test_pipeline_api_researcher_dispatch.py`
- Create: `tests/integration/test_researcher_isolation.py`

- [ ] **Step 1: Prompt module + parser tests**

Create `tests/unit/docwain/test_researcher_prompt.py`:

```python
import json

from src.docwain.prompts.researcher import (
    RESEARCHER_SYSTEM_PROMPT,
    ResearcherInsights,
    parse_researcher_response,
)


def test_parse_well_formed():
    text = json.dumps({
        "summary": "Invoice from Acme for $1,000.",
        "key_facts": ["Acme Corp is the vendor", "Amount is $1,000"],
        "entities": [{"text": "Acme Corp", "type": "ORGANIZATION"}],
        "recommendations": ["Verify vendor credentials"],
        "anomalies": [],
        "questions_to_ask": ["Is this a recurring vendor?"],
        "confidence": 0.85,
    })
    out = parse_researcher_response(text)
    assert isinstance(out, ResearcherInsights)
    assert out.summary.startswith("Invoice from Acme")
    assert len(out.key_facts) == 2
    assert out.confidence == 0.85


def test_parse_garbage_returns_empty():
    out = parse_researcher_response("nonsense")
    assert out.summary == ""
    assert out.key_facts == []
    assert out.confidence == 0.0


def test_prompt_non_empty():
    assert len(RESEARCHER_SYSTEM_PROMPT) > 100
    assert "insight" in RESEARCHER_SYSTEM_PROMPT.lower() or "summary" in RESEARCHER_SYSTEM_PROMPT.lower()
```

Create `src/docwain/prompts/researcher.py`:

```python
"""Researcher Agent prompt for DocWain.

Called by `src.tasks.researcher.run_researcher_agent` during the training stage.
Produces domain-aware insights from an extracted document. Writes to Qdrant
payload + Neo4j Insight nodes.

Spec: 2026-04-24-unified-docwain-engineering-layer-design.md §5.6
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List


RESEARCHER_SYSTEM_PROMPT = (
    "You are DocWain's Researcher Agent. Given an extracted document, produce "
    "structured domain-aware insights — the kind a domain expert would surface "
    "proactively without being asked. You will be called during ingestion, not "
    "at query time; your output is persisted and served to users later.\n\n"
    "Output ONLY valid JSON (no prose, no markdown fences):\n"
    "{\n"
    '  "summary": string (2-4 sentences),\n'
    '  "key_facts": [string, ...] (5-10 factual bullets verbatim from the doc),\n'
    '  "entities": [ { "text": string, "type": string } ],\n'
    '  "recommendations": [string, ...] (actionable suggestions based on doc content),\n'
    '  "anomalies": [string, ...] (anything unusual, inconsistent, or risky),\n'
    '  "questions_to_ask": [string, ...] (questions a user might want to ask about this doc),\n'
    '  "confidence": number 0..1\n'
    "}\n\n"
    "Rules:\n"
    "- Ground every key_fact and anomaly in text explicitly present in the document.\n"
    "- Do not fabricate numbers, names, or dates.\n"
    "- recommendations and questions_to_ask may be inferential but must be "
    "  supported by what the document actually contains.\n"
    "- Keep lists concise — quality over quantity."
)


@dataclass
class ResearcherInsights:
    summary: str = ""
    key_facts: List[str] = field(default_factory=list)
    entities: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    anomalies: List[str] = field(default_factory=list)
    questions_to_ask: List[str] = field(default_factory=list)
    confidence: float = 0.0


def _strip_code_fence(text: str) -> str:
    t = text.strip()
    m = re.match(r"```(?:json)?\s*(.*?)```", t, flags=re.DOTALL)
    return m.group(1).strip() if m else t


def _first_json_object(text: str) -> str:
    t = _strip_code_fence(text)
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end <= start:
        return t
    return t[start : end + 1]


def _str_list(value) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(x) for x in value if isinstance(x, (str, int, float))]


def parse_researcher_response(text: str) -> ResearcherInsights:
    try:
        data = json.loads(_first_json_object(text))
    except Exception:
        return ResearcherInsights()
    try:
        confidence = float(data.get("confidence", 0.0))
    except Exception:
        confidence = 0.0
    entities = data.get("entities") or []
    if not isinstance(entities, list):
        entities = []
    return ResearcherInsights(
        summary=str(data.get("summary") or ""),
        key_facts=_str_list(data.get("key_facts")),
        entities=entities,
        recommendations=_str_list(data.get("recommendations")),
        anomalies=_str_list(data.get("anomalies")),
        questions_to_ask=_str_list(data.get("questions_to_ask")),
        confidence=confidence,
    )


def build_user_prompt(*, document_text: str, doc_type_hint: str = "generic",
                      max_chars: int = 16000) -> str:
    truncated = document_text[:max_chars]
    return (
        f"Document type hint: {doc_type_hint}\n\n"
        f"Document text:\n\n{truncated}\n\n"
        "Return the insights JSON."
    )
```

Run: `/home/ubuntu/PycharmProjects/DocWain/.venv/bin/pytest tests/unit/docwain/test_researcher_prompt.py -x -q`. Expected: 3 passed.

- [ ] **Step 2: Add researcher status values**

In `src/api/statuses.py`, add (at module scope, near existing `KG_*` constants):

```python
# Researcher Agent status strand (independent from pipeline_status; same isolation as KG).
RESEARCHER_PENDING = "RESEARCHER_PENDING"
RESEARCHER_IN_PROGRESS = "RESEARCHER_IN_PROGRESS"
RESEARCHER_COMPLETED = "RESEARCHER_COMPLETED"
RESEARCHER_FAILED = "RESEARCHER_FAILED"
```

Per `feedback_mongo_status_stability.md`, these are ADDITIONS — no renames, no removals of existing values.

- [ ] **Step 3: Celery task — researcher**

Create `src/tasks/researcher.py`:

```python
"""Researcher Agent Celery task.

Dispatched from `src.api.pipeline_api.trigger_embedding` alongside embedding + KG.
Reads the canonical extraction JSON from Azure Blob, prompts DocWain for
insights, and writes results to Qdrant (payload mapped by document_id) + Neo4j
(Insight nodes linked to Document).

Isolation: writes ONLY to `researcher.*` field in MongoDB; never touches
`pipeline_status`, `stages.*`, or `knowledge_graph.*`. Spec §5.6.
"""
from __future__ import annotations

import json
import logging
import time as _time
from typing import Any, Dict, List, Optional

from src.celery_app import app
from src.api.statuses import (
    RESEARCHER_COMPLETED,
    RESEARCHER_FAILED,
    RESEARCHER_IN_PROGRESS,
)

logger = logging.getLogger(__name__)


def _extract_doc_text(extraction_json: Dict[str, Any]) -> str:
    """Concatenate all visible text from a canonical extraction JSON."""
    parts: List[str] = []
    for page in (extraction_json.get("pages") or []):
        for b in (page.get("blocks") or []):
            if b.get("text"):
                parts.append(b["text"])
        for t in (page.get("tables") or []):
            for row in (t.get("rows") or []):
                parts.append(" | ".join(str(c) for c in row))
    for sheet in (extraction_json.get("sheets") or []):
        for cell in (sheet.get("cells") or {}).values():
            val = (cell or {}).get("value") if isinstance(cell, dict) else None
            if val is not None:
                parts.append(str(val))
    for slide in (extraction_json.get("slides") or []):
        for e in (slide.get("elements") or []):
            if e.get("text"):
                parts.append(e["text"])
        notes = (slide.get("notes") or "").strip()
        if notes:
            parts.append(notes)
    return "\n\n".join(parts)


def _call_docwain_for_insights(document_text: str, doc_type_hint: str = "generic"):
    from src.docwain.prompts.researcher import (
        RESEARCHER_SYSTEM_PROMPT,
        ResearcherInsights,
        build_user_prompt,
        parse_researcher_response,
    )
    try:
        from src.llm.gateway import LLMGateway
        from src.api.config import Config
        max_tokens = int(getattr(getattr(Config, "Researcher", None), "MAX_TOKENS", 4096))
        gw = LLMGateway()
        user = build_user_prompt(document_text=document_text, doc_type_hint=doc_type_hint)
        result = gw.generate_with_metadata(
            prompt=user,
            system=RESEARCHER_SYSTEM_PROMPT,
            temperature=0.2,
            max_tokens=max_tokens,
        )
        return parse_researcher_response(getattr(result, "text", "") or "")
    except Exception as exc:
        logger.warning("Researcher LLM call failed: %s", exc)
        return ResearcherInsights()


def _load_extraction(document_id: str, subscription_id: str, profile_id: str) -> Optional[Dict[str, Any]]:
    """Load the canonical extraction JSON from Azure Blob.

    Mirrors the pattern in `src.tasks.kg._load_extraction_from_blob` — if a
    helper with that name exists, reuse it; otherwise read directly.
    """
    try:
        from src.tasks.kg import _load_extraction_from_blob  # type: ignore
        return _load_extraction_from_blob(
            document_id=document_id, subscription_id=subscription_id, profile_id=profile_id
        )
    except Exception:
        pass
    # Fallback path — rarely hit
    try:
        from src.api.content_store import BlobStore
        store = BlobStore()
        # Adapt path per existing convention: {sub}/{profile}/{doc}/extraction.json
        path = f"{subscription_id}/{profile_id}/{document_id}/extraction.json"
        raw = store.get(path)
        return json.loads(raw.decode("utf-8")) if raw else None
    except Exception as exc:
        logger.warning("Failed to load extraction for %s: %s", document_id, exc)
        return None


def _write_insights_to_qdrant(document_id: str, subscription_id: str, profile_id: str,
                               insights: Dict[str, Any]) -> None:
    """Best-effort Qdrant payload enrichment. Never raises."""
    try:
        # This depends on the existing Qdrant client pattern. If not trivially
        # available, skip — observability still captures the insights.
        from qdrant_client import QdrantClient
        import os
        client = QdrantClient(url=os.environ["QDRANT_URL"],
                              api_key=os.environ["QDRANT_API_KEY"], timeout=60)
        # Collection name convention: assume subscription_id = collection.
        # Real projects may use a different mapping — adapt if needed.
        collection = subscription_id
        client.set_payload(
            collection_name=collection,
            payload={"researcher_insights": insights},
            points_selector=__import__("qdrant_client").models.Filter(
                must=[__import__("qdrant_client").models.FieldCondition(
                    key="document_id",
                    match=__import__("qdrant_client").models.MatchValue(value=document_id),
                )]
            ),
        )
    except Exception as exc:
        logger.debug("Qdrant insight write skipped for %s: %s", document_id, exc)


def _write_insight_to_neo4j(document_id: str, insights: Dict[str, Any]) -> None:
    """Best-effort Neo4j Insight node creation. Never raises."""
    try:
        from src.kg.neo4j_store import Neo4jStore
        store = Neo4jStore()
        store.create_insight_node(document_id=document_id, insights=insights)
    except AttributeError:
        # create_insight_node doesn't exist yet — inline fallback writes via raw cypher.
        try:
            from src.kg.neo4j_store import Neo4jStore
            store = Neo4jStore()
            cy = (
                "MATCH (d:Document {document_id: $doc_id}) "
                "MERGE (d)-[:HAS_INSIGHT]->(i:Insight {document_id: $doc_id}) "
                "SET i.summary = $summary, i.confidence = $confidence, "
                "    i.key_facts = $key_facts, i.recommendations = $recommendations, "
                "    i.anomalies = $anomalies, i.questions_to_ask = $questions_to_ask, "
                "    i.updated_at = timestamp()"
            )
            with store.driver.session() as session:
                session.run(cy, doc_id=document_id,
                            summary=insights.get("summary", ""),
                            confidence=float(insights.get("confidence", 0.0)),
                            key_facts=insights.get("key_facts", []),
                            recommendations=insights.get("recommendations", []),
                            anomalies=insights.get("anomalies", []),
                            questions_to_ask=insights.get("questions_to_ask", []))
        except Exception as exc:
            logger.debug("Neo4j insight write skipped for %s: %s", document_id, exc)
    except Exception as exc:
        logger.debug("Neo4j insight write skipped for %s: %s", document_id, exc)


def _set_researcher_status(document_id: str, status: str, **extra) -> None:
    """Update MongoDB researcher.* strand. Never touches pipeline_status."""
    try:
        from src.api.document_status import get_document_record
        # Use a best-effort helper — update via the documents collection directly if needed.
        from src.api.dw_newron import get_mongo_collection  # type: ignore
        col = get_mongo_collection("documents")
        update = {"researcher.status": status, "researcher.updated_at": _time.time()}
        for k, v in extra.items():
            update[f"researcher.{k}"] = v
        col.update_one({"document_id": document_id}, {"$set": update})
    except Exception as exc:
        logger.warning("Failed to set researcher status for %s: %s", document_id, exc)


@app.task(bind=True, max_retries=3, soft_time_limit=1200)
def run_researcher_agent(self, document_id: str, subscription_id: str, profile_id: str):
    """Runs the Researcher Agent for a single document. Fully isolated from embedding + KG."""
    started_at = _time.perf_counter()
    _set_researcher_status(document_id, RESEARCHER_IN_PROGRESS, started_at=_time.time())
    try:
        extraction = _load_extraction(document_id, subscription_id, profile_id)
        if not extraction:
            _set_researcher_status(document_id, RESEARCHER_FAILED,
                                   error="extraction not found",
                                   completed_at=_time.time())
            return {"status": RESEARCHER_FAILED, "error": "extraction not found"}

        doc_type_hint = ((extraction.get("metadata") or {}).get("doc_intel") or {}).get("doc_type_hint") or "generic"
        document_text = _extract_doc_text(extraction)
        insights_obj = _call_docwain_for_insights(document_text, doc_type_hint=doc_type_hint)
        from dataclasses import asdict as _asdict
        insights = _asdict(insights_obj)

        _write_insights_to_qdrant(document_id, subscription_id, profile_id, insights)
        _write_insight_to_neo4j(document_id, insights)

        _set_researcher_status(
            document_id, RESEARCHER_COMPLETED,
            summary_preview=(insights.get("summary") or "")[:200],
            confidence=float(insights.get("confidence", 0.0)),
            elapsed_ms=(_time.perf_counter() - started_at) * 1000.0,
            completed_at=_time.time(),
        )
        return {"status": RESEARCHER_COMPLETED, "confidence": insights.get("confidence", 0.0)}
    except Exception as exc:
        logger.warning("Researcher failed for %s: %r", document_id, exc)
        _set_researcher_status(document_id, RESEARCHER_FAILED,
                                error=repr(exc),
                                elapsed_ms=(_time.perf_counter() - started_at) * 1000.0,
                                completed_at=_time.time())
        return {"status": RESEARCHER_FAILED, "error": repr(exc)}
```

- [ ] **Step 4: Register queue + routing**

In `src/celery_app.py`, add `researcher_queue` to the queue dict and route `src.tasks.researcher.run_researcher_agent`:

```python
"researcher_queue": {"exchange": "researcher", "routing_key": "researcher"},
```

And in the routing section:
```python
"src.tasks.researcher.run_researcher_agent": {"queue": "researcher_queue"},
```

Match the existing style in that file.

- [ ] **Step 5: Wire dispatch + isolation tests**

Create `tests/unit/api/test_pipeline_api_researcher_dispatch.py`:

```python
"""trigger_embedding dispatches embed + KG + researcher when Researcher is enabled."""
import asyncio
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def fake_doc_in_screening_completed():
    from src.api.statuses import PIPELINE_SCREENING_COMPLETED
    return {"_id": "doc-r1", "subscription_id": "sub-r", "profile_id": "prof-r",
            "pipeline_status": PIPELINE_SCREENING_COMPLETED}


def test_dispatches_all_three_when_researcher_enabled(fake_doc_in_screening_completed, monkeypatch):
    from src.api import config as cfg_mod
    monkeypatch.setattr(cfg_mod.Config.Researcher, "ENABLED", True, raising=False)

    from src.api import pipeline_api

    counts = {"embed": 0, "kg": 0, "researcher": 0}
    monkeypatch.setattr("src.tasks.embedding.embed_document.delay",
                        lambda *a, **kw: (counts.__setitem__("embed", counts["embed"] + 1) or MagicMock()))
    monkeypatch.setattr("src.tasks.kg.build_knowledge_graph.delay",
                        lambda *a, **kw: (counts.__setitem__("kg", counts["kg"] + 1) or MagicMock()))
    monkeypatch.setattr("src.tasks.researcher.run_researcher_agent.delay",
                        lambda *a, **kw: (counts.__setitem__("researcher", counts["researcher"] + 1) or MagicMock()))

    for attr in ("get_document_record", "get_documents_collection"):
        monkeypatch.setattr(pipeline_api, attr, lambda *a, **kw: fake_doc_in_screening_completed, raising=False)
    monkeypatch.setattr(pipeline_api, "append_audit_log", lambda *a, **kw: None, raising=False)

    try:
        asyncio.run(pipeline_api.trigger_embedding(document_id="doc-r1"))
    except Exception as exc:
        pytest.fail(f"trigger_embedding raised: {exc!r}")

    assert counts == {"embed": 1, "kg": 1, "researcher": 1}


def test_does_not_dispatch_researcher_when_disabled(fake_doc_in_screening_completed, monkeypatch):
    from src.api import config as cfg_mod
    monkeypatch.setattr(cfg_mod.Config.Researcher, "ENABLED", False, raising=False)

    from src.api import pipeline_api

    counts = {"embed": 0, "kg": 0, "researcher": 0}
    monkeypatch.setattr("src.tasks.embedding.embed_document.delay",
                        lambda *a, **kw: (counts.__setitem__("embed", counts["embed"] + 1) or MagicMock()))
    monkeypatch.setattr("src.tasks.kg.build_knowledge_graph.delay",
                        lambda *a, **kw: (counts.__setitem__("kg", counts["kg"] + 1) or MagicMock()))
    monkeypatch.setattr("src.tasks.researcher.run_researcher_agent.delay",
                        lambda *a, **kw: (counts.__setitem__("researcher", counts["researcher"] + 1) or MagicMock()))

    for attr in ("get_document_record", "get_documents_collection"):
        monkeypatch.setattr(pipeline_api, attr, lambda *a, **kw: fake_doc_in_screening_completed, raising=False)
    monkeypatch.setattr(pipeline_api, "append_audit_log", lambda *a, **kw: None, raising=False)

    asyncio.run(pipeline_api.trigger_embedding(document_id="doc-r1"))

    # embed + kg still dispatch; researcher suppressed by flag
    assert counts["embed"] == 1
    assert counts["kg"] == 1
    assert counts["researcher"] == 0
```

Create `tests/integration/test_researcher_isolation.py`:

```python
"""run_researcher_agent never touches pipeline_status; only researcher.* strand."""
from unittest.mock import MagicMock


def test_researcher_task_does_not_touch_pipeline_status(monkeypatch):
    import src.tasks.researcher as r_mod

    seen_updates = []

    class FakeCol:
        def update_one(self, filter, update, **kw):
            seen_updates.append(update)
            return MagicMock(matched_count=1, modified_count=1)

    def fake_get_mongo_collection(name):
        return FakeCol()

    # Patch the Mongo helper the researcher uses.
    try:
        monkeypatch.setattr("src.api.dw_newron.get_mongo_collection",
                             fake_get_mongo_collection, raising=False)
    except Exception:
        pass

    # Stub the heavy lifts so the task runs fully.
    monkeypatch.setattr(r_mod, "_load_extraction",
                         lambda *a, **kw: {"format": "docx", "pages": [{"page_num": 1, "blocks": [{"text": "hi"}]}]},
                         raising=False)
    monkeypatch.setattr(r_mod, "_call_docwain_for_insights",
                         lambda *a, **kw: __import__("src.docwain.prompts.researcher",
                                                     fromlist=["ResearcherInsights"]).ResearcherInsights(
                             summary="s", confidence=0.5
                         ),
                         raising=False)
    monkeypatch.setattr(r_mod, "_write_insights_to_qdrant", lambda *a, **kw: None, raising=False)
    monkeypatch.setattr(r_mod, "_write_insight_to_neo4j", lambda *a, **kw: None, raising=False)

    # Run synchronously via .apply (Celery bind=True pattern).
    try:
        r_mod.run_researcher_agent.apply(args=("doc-iso", "sub-iso", "prof-iso"))
    except Exception:
        pass

    # No captured update should set pipeline_status or stages.* — only researcher.*
    forbidden_prefixes = ("pipeline_status", "stages.", "knowledge_graph")
    for upd in seen_updates:
        if not isinstance(upd, dict):
            continue
        for op, body in upd.items():
            if isinstance(body, dict):
                for k in body.keys():
                    for forbidden in forbidden_prefixes:
                        assert not k.startswith(forbidden), f"researcher wrote forbidden field {k!r}"
```

- [ ] **Step 6: Dispatch in `trigger_embedding`**

In `src/api/pipeline_api.py::trigger_embedding`, immediately AFTER the KG dispatch block (added in Plan 3), add:

```python
    # Plan 4 wave 1: dispatch Researcher Agent as a third parallel training-stage task.
    # Fire-and-forget — failure does not affect embedding or KG.
    try:
        from src.api.config import Config
        researcher_enabled = getattr(getattr(Config, "Researcher", None), "ENABLED", False)
    except Exception:
        researcher_enabled = False
    if researcher_enabled:
        try:
            from src.tasks.researcher import run_researcher_agent
            run_researcher_agent.delay(document_id, subscription_id, profile_id)
            logger.info("Researcher Agent dispatched for %s", document_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Researcher dispatch failed for %s: %s", document_id, exc)
```

Use the actual variable names from that function (the Plan 3 work established that `document_id`, `subscription_id`, `profile_id` are the names).

- [ ] **Step 7: Run tests**

```bash
cd /home/ubuntu/.config/superpowers/worktrees/DocWain/preprod_v02
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/pytest tests/unit/docwain/test_researcher_prompt.py tests/unit/api/test_pipeline_api_researcher_dispatch.py tests/integration/test_researcher_isolation.py -x -q
```

Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add src/docwain/prompts/researcher.py tests/unit/docwain/test_researcher_prompt.py \
        src/tasks/researcher.py src/api/statuses.py src/celery_app.py \
        src/api/pipeline_api.py tests/unit/api/test_pipeline_api_researcher_dispatch.py \
        tests/integration/test_researcher_isolation.py
git commit -m "unified: Researcher Agent — third training-stage parallel task, isolated from embedding + KG"
```

---

### Task 4: Full-suite + bench + smoke validation

Not a code task — validation only.

- [ ] **Step 1: Run all extraction + KG + API + docwain + integration suites**

```bash
cd /home/ubuntu/.config/superpowers/worktrees/DocWain/preprod_v02
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/pytest tests/unit/extraction tests/unit/kg tests/unit/api tests/unit/llm tests/unit/docwain tests/integration -q --timeout=30 2>&1 | tail -10
```

Expected: all pass. Count should be well above the Plan 3 baseline (393 passed) — add ~15-20 new tests.

- [ ] **Step 2: Run bench**

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/python -m tests.extraction_bench.runner 2>&1 | tail -12
```

Expected: 7/7 `[PASS]` at 1.000 (unchanged).

- [ ] **Step 3: Broader sanity**

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/pytest tests/unit -q --timeout=30 2>&1 | tail -8
```

Expected: previously-passing count + Wave 1 additions; 2 pre-existing unrelated failures still expected.

- [ ] **Step 4: Smoke imports**

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/python -c "
from src.docwain.prompts.entity_extraction import parse_entity_response
from src.docwain.prompts.researcher import parse_researcher_response
from src.tasks.researcher import run_researcher_agent
from src.api.statuses import RESEARCHER_PENDING, RESEARCHER_IN_PROGRESS, RESEARCHER_COMPLETED, RESEARCHER_FAILED
from src.api.config import Config
assert Config.Model.PRIMARY_BACKEND in ('vllm', 'cloud', 'azure')
assert isinstance(Config.Model.IDENTITY_SHIM_ENABLED, bool)
assert isinstance(Config.KG.ENTITY_EXTRACTION_ENABLED, bool)
assert isinstance(Config.Researcher.ENABLED, bool)
print('ALL IMPORTS + CONFIG OK')
"
```

Expected: `ALL IMPORTS + CONFIG OK`.

- [ ] **Step 5: Show final branch state**

```bash
git log --oneline preprod_v01..HEAD | head -20
git rev-list --count preprod_v01..HEAD
```

No commit for Task 4. Report all outputs in the final report.

---

## Self-review — spec coverage

- **§5.1 Gateway vLLM primary + identity shim:** Task 1 ✓
- **§5.3 Entity extraction + KG wiring:** Task 2 ✓
- **§5.6 Researcher Agent:** Task 3 ✓
- **§5.2 prompt registry:** deferred (noted in spec as Wave 2; only the `src/docwain/prompts/` package scaffold is added in Wave 1 for the two new prompts)
- **§5.4 chart gen:** deferred (Wave 2)
- **§5.5 domain adapter framework:** deferred (Wave 2)
- **§6 zero-error discipline:** all three tasks feature-flag gated; rollback plan in spec §10
- **§7 success criteria:** Task 4 validation

## Self-review — placeholder scan

- Every code step shows complete code.
- Variable-name hand-offs in Task 3 (`document_id`, `subscription_id`, `profile_id`) are documented as "Plan 3 established these" — not guesses.
- Mongo-helper-name hand-offs in Task 3 use try/except patterns — tolerant of the real helper not matching exactly.

## Self-review — type consistency

- `ExtractedEntities` defined in Task 2, referenced by Task 2's wiring.
- `ResearcherInsights` defined in Task 3, referenced by Task 3's task body.
- `RESEARCHER_*` status constants added in Task 3 Step 2, used by Task 3 Step 3 + integration test.
- Config flags `Config.Model.*`, `Config.KG.ENTITY_EXTRACTION_ENABLED`, `Config.Researcher.ENABLED` all added in Task 1 Step 2, referenced by Tasks 2 and 3.
- Gateway `_apply_identity_shim` defined in Task 1 Step 6, used by Task 1 Step 6 wrapper.

No drift.

Plan complete.
