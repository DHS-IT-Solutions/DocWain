# Batch 0 — Unified Model + Qdrant Audit + Regression Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore yesterday's intelligence-layer accuracy by collapsing the fast/smart model split, realigning retrieval filters to the post-`b0c7211` qdrant payload schema, and repointing response-formatting calls through `src/generation/prompts.py`.

**Architecture:** One unified `docwain` model served by a single `IntelligenceHandler`; `FastPathHandler` module deleted; `src/query/pipeline.py` simplified to a single Route → Plan → Execute → Assemble → Generate+Verify flow; qdrant read-side filters audited against the indexed flat fields in `src/embedding/payload_builder.py`; response formatting centralised in `src/generation/prompts.py`.

**Tech Stack:** Python 3.12, pytest, qdrant-client, vLLM (docwain model), Qwen3-14B backbone.

**Spec:** `docs/superpowers/specs/2026-04-21-intelligence-rag-redesign-design.md` (see §7 for Batch 0 scope and §6.1 row 0 for exit criteria).

**Branch:** `batch-0-unified-model-qdrant-audit` (off `main`, PR target `main`).

**Do-not-touch list (from spec §4.4 / §7.4):**
- `src/api/pipeline_api.py`
- `src/api/extraction_service.py`, `extraction_pipeline_api.py`, `dw_document_extractor.py`
- `src/api/embedding_service.py`, `src/embed/`, `src/embedding/` (except **reads** to confirm payload schema)
- `src/api/qdrant_indexes.py`, `qdrant_setup.py`, `vector_store.py` (reads allowed, no writes)
- `src/extraction/`, `src/tasks/embedding.py`
- Celery worker defs, systemd pipeline units
- Mongo status field names
- Any `src/intelligence_v2/` file (deferred to Batch 2)

---

## File Map

### Deletes
- `src/serving/fast_path.py` — module removed entirely.

### Creates
- `src/serving/intelligence_handler.py` — new unified handler (absorbs `fast_path.py`'s two responsibilities).
- `tests/batch0/__init__.py`
- `tests/batch0/conftest.py`
- `tests/batch0/test_grep_gate.py`
- `tests/batch0/test_qdrant_roundtrip.py`
- `tests/batch0/test_prompt_path_contract.py`
- `tests/batch0/test_canned_queries_smoke.py`
- `tests/batch0/fixtures/invoice_fixture.json`
- `eval_results/pre-batch-0-test-baseline.txt` (snapshot of current test state)
- `scripts/batch0/audit_qdrant_filters.py` (one-shot audit tool)

### Modifies
- `src/serving/config.py` — rename model `docwain-fast` → `docwain`.
- `src/serving/vllm_manager.py` — default `model=` arg, docstrings.
- `src/serving/model_router.py` — scrub fast/smart wording in docstrings only.
- `src/serving/__init__.py` — drop `FastPathHandler` export.
- `src/query/pipeline.py` — remove `_is_fast_path`, `_handle_fast_path`, fast-path branch; `route_taken` default `"intelligence"`.
- `src/intelligence/reasoning_engine.py:96-97` — repoint to `src/generation/prompts.py` templates.
- `src/intelligence/generator.py` — move `_FORMAT_INSTRUCTIONS` dict to `src/generation/prompts.py` (or delete the file entirely if nothing else uses it).
- `src/generation/prompts.py` — absorb `_FORMAT_INSTRUCTIONS`.
- `src/retrieval/retriever.py` / `src/retrieval/filter_builder.py` / `src/retrieval/profile_query.py` / `src/retrieval/bgem3_retriever.py` / `src/intelligence/retrieval.py` — fix any payload-key mismatches found by the audit.

---

## Task 1: Capture pre-batch test baseline and create branch

**Files:**
- Create: `eval_results/pre-batch-0-test-baseline.txt`

- [ ] **Step 1: Verify current branch is `main` and clean**

Run:
```bash
git status --short && git branch --show-current
```
Expected: current branch is `main`; any non-staged changes are only `.pyc` files (those are fine — pipeline is live).

- [ ] **Step 2: Create feature branch**

Run:
```bash
git checkout -b batch-0-unified-model-qdrant-audit
```
Expected: `Switched to a new branch 'batch-0-unified-model-qdrant-audit'`.

- [ ] **Step 3: Snapshot the current test-suite state**

Run:
```bash
mkdir -p eval_results
python -m pytest tests/ --collect-only -q 2>&1 | tail -5 > eval_results/pre-batch-0-test-baseline.txt
python -m pytest tests/ --continue-on-collection-errors -q --tb=no --no-header --ignore=tests/live 2>&1 | tail -10 >> eval_results/pre-batch-0-test-baseline.txt
```
Expected: file contains both the collection summary (e.g. "5443 tests collected, 97 errors") and the final pass/fail counts. If this command exceeds 15 minutes, kill it and record `<timed out>` plus the collection summary only.

- [ ] **Step 4: Commit the baseline**

```bash
git add -f eval_results/pre-batch-0-test-baseline.txt
git commit -m "batch-0: snapshot pre-change test-suite baseline"
```

---

## Task 2: Write the grep-gate test (TDD — must fail first)

The grep-gate is the mechanical check that no live code references the fast/smart split any more. It reads the source tree and fails if forbidden tokens appear outside allowed paths (historical specs, plans, git-ignored files).

**Files:**
- Create: `tests/batch0/__init__.py` (empty)
- Create: `tests/batch0/conftest.py`
- Create: `tests/batch0/test_grep_gate.py`

- [ ] **Step 1: Create empty package init**

```bash
mkdir -p tests/batch0
touch tests/batch0/__init__.py
```

- [ ] **Step 2: Write the conftest**

Write `tests/batch0/conftest.py`:

```python
"""Shared fixtures for Batch-0 regression-gate tests."""
from __future__ import annotations

import pathlib

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def repo_root() -> pathlib.Path:
    return _REPO_ROOT
```

- [ ] **Step 3: Write the failing grep-gate test**

Write `tests/batch0/test_grep_gate.py`:

```python
"""Grep-gate: fast/smart split must not appear in live code.

Allowed locations for historical references:
 - docs/superpowers/specs/*     (design history)
 - docs/superpowers/plans/*     (implementation history)
 - eval_results/*               (snapshots)
 - scripts/batch0/*             (one-shot audit tooling)
 - tests/batch0/*               (this test itself)
 - Anything git-ignored (pycache, logs, .superpowers/, etc.)

Any reference in src/, deploy/, systemd/, or general tests/ is a fail.
"""
from __future__ import annotations

import pathlib
import re
import subprocess

FORBIDDEN_PATTERNS = [
    r"docwain[-_]fast",
    r"docwain[-_]smart",
    r"FastPathHandler",
    r"fast[_-]path",
    r"14B",
    r"27B",
    r"\"smart\"",
    r"'smart'",
]

ALLOWED_PREFIXES = (
    "docs/superpowers/specs/",
    "docs/superpowers/plans/",
    "eval_results/",
    "scripts/batch0/",
    "tests/batch0/",
)


def _tracked_files(repo_root: pathlib.Path) -> list[str]:
    out = subprocess.check_output(
        ["git", "ls-files"], cwd=repo_root, text=True,
    )
    return [p for p in out.splitlines() if p.strip()]


def test_grep_gate_no_fast_smart_refs(repo_root: pathlib.Path):
    combined = re.compile("|".join(FORBIDDEN_PATTERNS))
    offenders: list[str] = []
    for rel in _tracked_files(repo_root):
        if any(rel.startswith(p) for p in ALLOWED_PREFIXES):
            continue
        if not (rel.startswith("src/") or rel.startswith("deploy/")
                or rel.startswith("systemd/") or rel.startswith("tests/")):
            continue
        path = repo_root / rel
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except (OSError, UnicodeDecodeError):
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            if combined.search(line):
                offenders.append(f"{rel}:{line_no}: {line.strip()[:140]}")
    assert not offenders, (
        "Forbidden fast/smart tokens found in live code:\n  "
        + "\n  ".join(offenders[:50])
        + (f"\n  ...and {len(offenders) - 50} more" if len(offenders) > 50 else "")
    )
```

- [ ] **Step 4: Run the test — it must fail**

Run:
```bash
python -m pytest tests/batch0/test_grep_gate.py -v
```
Expected: **FAIL** with a list of current offenders (most likely `src/serving/fast_path.py`, `src/serving/vllm_manager.py:43` for `docwain-fast`, `src/query/pipeline.py` fast-path refs, `src/serving/model_router.py` docstrings, etc.).

- [ ] **Step 5: Commit the (failing) gate**

```bash
git add tests/batch0/__init__.py tests/batch0/conftest.py tests/batch0/test_grep_gate.py
git commit -m "batch-0: add grep-gate for fast/smart split (expected to fail pre-cleanup)"
```

---

## Task 3: Create the unified IntelligenceHandler

The new module absorbs `FastPathHandler`'s two responsibilities into one class that always uses the unified `docwain` model and the same generation prompts. The "no-retrieval for greeting/identity" and "thin retrieval for lookup/list/count" paths become private methods on the single class.

**Files:**
- Create: `src/serving/intelligence_handler.py`

- [ ] **Step 1: Write the new handler**

Write `src/serving/intelligence_handler.py`:

```python
"""Unified intelligence handler.

Single entry point for all intent categories. Uses the unified DocWain
vLLM instance and the generation-layer prompts (src/generation/prompts.py).
Replaces the former FastPathHandler module — no fast/smart split.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.generation.prompts import build_system_prompt
from src.serving.model_router import RouterResult
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_NO_RETRIEVAL_INTENTS = frozenset({"greeting", "identity", "greet", "meta", "help", "capability", "goodbye"})

_GREETING_SYSTEM = "You are DocWain."
_IDENTITY_SYSTEM = "You are DocWain."


class IntelligenceHandler:
    """Handles any intent against the unified DocWain model.

    For no-retrieval intents (greetings, identity) generates a direct
    response with no evidence lookup. For evidence-backed intents
    (lookup, list, count, extract) it performs a single Qdrant search
    then generates a grounded response.

    Complex intents (analyze, investigate, compare, summarize, etc.)
    are handled upstream by the full query pipeline; this handler
    covers only the intents that do not need a Plan → Execute loop.
    """

    def __init__(self, vllm_manager: Any) -> None:
        self._mgr = vllm_manager

    # -- Public -----------------------------------------------------------

    def handle(
        self,
        query: str,
        router_result: RouterResult,
        profile_context: Optional[str] = None,
        retriever: Optional[Any] = None,
        subscription_id: Optional[str] = None,
        profile_id: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        intent = getattr(router_result, "intent", "") or ""
        if intent in _NO_RETRIEVAL_INTENTS:
            return self._handle_no_retrieval(query, intent)
        return self._handle_with_retrieval(
            query=query,
            router_result=router_result,
            profile_context=profile_context,
            retriever=retriever,
            subscription_id=subscription_id,
            profile_id=profile_id,
            collection_name=collection_name,
        )

    # -- Private handlers -------------------------------------------------

    def _handle_no_retrieval(self, query: str, intent: str) -> Dict[str, Any]:
        system = _GREETING_SYSTEM if intent in {"greeting", "greet"} else _IDENTITY_SYSTEM
        try:
            response = self._mgr.query(
                prompt=query,
                system_prompt=system,
                max_tokens=512,
                temperature=0.5,
            )
        except Exception as exc:
            logger.error("No-retrieval generation failed: %s", exc)
            response = ""
        if not response:
            response = self._static_fallback(intent)
        return _build_payload(response=response, sources=[], grounded=False, context_found=False)

    def _handle_with_retrieval(
        self,
        query: str,
        router_result: RouterResult,
        profile_context: Optional[str],
        retriever: Optional[Any],
        subscription_id: Optional[str],
        profile_id: Optional[str],
        collection_name: Optional[str],
    ) -> Dict[str, Any]:
        evidence_chunks: List[Any] = []
        if retriever is not None and subscription_id and profile_id and collection_name:
            try:
                evidence_chunks = retriever.retrieve(
                    query=query,
                    subscription_id=subscription_id,
                    profile_id=profile_id,
                    top_k=15,
                    collection_name=collection_name,
                )
            except Exception as exc:
                logger.warning("Retrieval failed: %s", exc)

        evidence_text, sources = self._format_evidence(evidence_chunks)
        system = build_system_prompt(profile_domain=profile_context or "")
        user_prompt = self._build_generation_prompt(query, getattr(router_result, "intent", ""), evidence_text)

        try:
            response = self._mgr.query(
                prompt=user_prompt,
                system_prompt=system,
                max_tokens=4096,
                temperature=0.3,
            )
        except Exception as exc:
            logger.error("Generation failed: %s", exc)
            response = ""

        grounded = bool(sources) and bool(response)
        context_found = bool(sources)
        if not response and context_found:
            response = "I found relevant documents but was unable to generate a response. Please try again."
        elif not response:
            response = "I couldn't find relevant information in the documents to answer that question."

        chart_spec = self._extract_chart_spec(response)
        return _build_payload(
            response=response,
            sources=sources,
            grounded=grounded,
            context_found=context_found,
            chart_spec=chart_spec,
        )

    # -- Helpers ----------------------------------------------------------

    @staticmethod
    def _format_evidence(chunks: List[Any]) -> tuple[str, List[Dict[str, Any]]]:
        if not chunks:
            return "", []
        parts: List[str] = []
        sources: List[Dict[str, Any]] = []
        seen: set = set()
        for i, chunk in enumerate(chunks):
            text = getattr(chunk, "text", None) or getattr(chunk, "snippet", "")
            file_name = getattr(chunk, "file_name", None) or getattr(chunk, "source_name", "unknown")
            page = getattr(chunk, "page", None) or getattr(chunk, "page_start", None)
            snippet = getattr(chunk, "snippet", (text[:200] if text else ""))
            snippet_sha = getattr(chunk, "snippet_sha", "")
            parts.append(f"[SOURCE-{i + 1}] (file: {file_name}, page: {page})\n{text}\n")
            key = (file_name, page, snippet_sha)
            if key not in seen:
                seen.add(key)
                sources.append({"file_name": file_name, "page": page, "snippet": snippet})
        return "\n".join(parts), sources

    @staticmethod
    def _build_generation_prompt(query: str, intent: str, evidence_text: str) -> str:
        if not evidence_text:
            return (
                f"The user asked: {query}\n\n"
                "No relevant evidence was found in the documents. "
                "Politely inform the user that the documents do not contain "
                "information to answer this question."
            )
        instruction = {
            "lookup": "Answer the question directly using the evidence below.",
            "list": "Provide a clear list based on the evidence below.",
            "count": "Count the relevant items from the evidence below and state the total.",
            "extract": "Extract the requested data from the evidence below in a structured format.",
        }.get(intent, "Answer the question using the evidence below.")
        return (
            f"{instruction}\n\n"
            f"EVIDENCE:\n{evidence_text}\n\n"
            f"USER QUESTION: {query}"
        )

    @staticmethod
    def _extract_chart_spec(response: str) -> Optional[Dict[str, Any]]:
        import re
        match = re.search(r"<!--DOCWAIN_VIZ\s*\n(.*?)\n\s*-->", response, re.DOTALL)
        if not match:
            return None
        try:
            import json
            return json.loads(match.group(1))
        except (json.JSONDecodeError, ValueError):
            return None

    @staticmethod
    def _static_fallback(intent: str) -> str:
        if intent in {"greeting", "greet"}:
            return "Hello! I'm DocWain, your document intelligence assistant. How can I help you today?"
        if intent == "identity":
            return (
                "I'm DocWain, an intelligent document analysis assistant. "
                "I can help you search, summarise, compare, and extract data "
                "from your uploaded documents."
            )
        return "I'm having trouble processing your request right now. Please try again shortly."


def _build_payload(
    response: str,
    sources: List[Dict[str, Any]],
    grounded: bool,
    context_found: bool,
    chart_spec: Optional[Dict[str, Any]] = None,
    alerts: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "response": response,
        "sources": sources,
        "chart_spec": chart_spec,
        "alerts": alerts or [],
        "grounded": grounded,
        "context_found": context_found,
    }
```

- [ ] **Step 2: Add a sanity import test**

Append this to `tests/batch0/test_grep_gate.py`:

```python


def test_intelligence_handler_importable():
    """The new unified handler must import without error."""
    from src.serving.intelligence_handler import IntelligenceHandler  # noqa: F401
```

- [ ] **Step 3: Run the new import test — must pass**

Run:
```bash
python -m pytest tests/batch0/test_grep_gate.py::test_intelligence_handler_importable -v
```
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/serving/intelligence_handler.py tests/batch0/test_grep_gate.py
git commit -m "batch-0: add unified IntelligenceHandler (replaces FastPathHandler)"
```

---

## Task 4: Delete `fast_path.py` and update `serving/__init__.py`

**Files:**
- Delete: `src/serving/fast_path.py`
- Modify: `src/serving/__init__.py`

- [ ] **Step 1: Inspect current `serving/__init__.py`**

Run:
```bash
cat src/serving/__init__.py
```
Take note of any `FastPathHandler` imports/exports.

- [ ] **Step 2: Update `__init__.py` to remove `FastPathHandler` and add `IntelligenceHandler`**

Apply these edits to `src/serving/__init__.py`:
- Remove any line importing `FastPathHandler` from `.fast_path`.
- Remove `FastPathHandler` from `__all__` if present.
- Add `from .intelligence_handler import IntelligenceHandler` alongside the other handler imports.
- Add `"IntelligenceHandler"` to `__all__`.

- [ ] **Step 3: Delete the old module**

Run:
```bash
git rm src/serving/fast_path.py
```

- [ ] **Step 4: Verify no other file imports `FastPathHandler`**

Run:
```bash
grep -rn "FastPathHandler\|src\.serving\.fast_path" src/ tests/ deploy/ systemd/ 2>/dev/null | grep -v "tests/batch0/" | grep -v "\.pyc"
```
Expected: **no output** (empty grep). If anything matches outside `tests/batch0/`, open that file and replace the usage with `IntelligenceHandler` — `from src.serving.intelligence_handler import IntelligenceHandler` and adjust constructor calls (new signature takes `vllm_manager`).

- [ ] **Step 5: Commit**

```bash
git add src/serving/__init__.py src/serving/fast_path.py
git commit -m "batch-0: delete fast_path.py, export IntelligenceHandler from serving"
```

---

## Task 5: Rename `docwain-fast` → `docwain`

**Files:**
- Modify: `src/serving/vllm_manager.py` (line 43 default arg, docstring at lines 1-6 and 30-36)
- Modify: `src/serving/config.py` (DOCWAIN_CONFIG name field at line 53, if present)

- [ ] **Step 1: Rename in `vllm_manager.py`**

Edit `src/serving/vllm_manager.py`:
- Line 43 (default arg): change `model: str = "docwain-fast",` to `model: str = "docwain",`.
- Docstring at top of file (lines 1-6): remove "no fast/smart split" wording — leave a one-line description like "Client-only vLLM manager for the unified DocWain model."
- Any other string literal `"docwain-fast"` in the file → `"docwain"`.

- [ ] **Step 2: Rename in `config.py`**

Edit `src/serving/config.py`:
- In `DOCWAIN_CONFIG = VLLMInstanceConfig(...)` (around line 52), `name="docwain"` already — confirm. If any other legacy alias is defined (e.g., a separate `DOCWAIN_FAST_CONFIG`), delete it.
- Grep the file for `docwain-fast` / `docwain_fast` / `fast` / `smart` — all must go.

- [ ] **Step 3: Update deploy + systemd unit names if referenced**

Run:
```bash
grep -rn "docwain-fast\|docwain_fast" deploy/ systemd/ 2>/dev/null
```
For each match in `deploy/` or `systemd/`: rename in-file to `docwain`. If it's a unit file name (e.g. `docwain-vllm-fast.service`), **do not rename the file itself** — that breaks the running service. Instead, leave the file as-is and note in the PR description that the unit-file rename is an ops follow-up. (CLAUDE.md references `docwain-vllm-fast` / `docwain-vllm-smart` systemd units; those come out in a separate ops PR.)

- [ ] **Step 4: Run the grep-gate test — should show progress**

Run:
```bash
python -m pytest tests/batch0/test_grep_gate.py::test_grep_gate_no_fast_smart_refs -v
```
Expected: still **FAIL**, but with fewer offenders than before (no more `docwain-fast` / `docwain_fast` / `FastPathHandler` hits).

- [ ] **Step 5: Commit**

```bash
git add src/serving/vllm_manager.py src/serving/config.py deploy/ systemd/
git commit -m "batch-0: rename docwain-fast -> docwain (single unified model)"
```

---

## Task 6: Simplify `src/query/pipeline.py` to one path

**Files:**
- Modify: `src/query/pipeline.py` (remove lines 18-27 fast_path import, 37 route_taken default, 103-107 fast-path branch, 170 route_taken ternary, 209-248 `_is_fast_path` + `_handle_fast_path` helpers)

- [ ] **Step 1: Remove the fast-path import guard**

In `src/query/pipeline.py`, replace lines 18-27:

Before:
```python
# Lazy imports for serving layer (may not be built yet)
try:
    from src.serving.vllm_manager import VLLMManager
    from src.serving.model_router import IntentRouter, RouterResult
    from src.serving.fast_path import FastPathHandler
except ImportError:
    VLLMManager = None
    IntentRouter = None
    RouterResult = None
    FastPathHandler = None
```

After:
```python
# Lazy imports for serving layer (may not be built yet)
try:
    from src.serving.vllm_manager import VLLMManager
    from src.serving.model_router import IntentRouter, RouterResult
except ImportError:
    VLLMManager = None
    IntentRouter = None
    RouterResult = None
```

- [ ] **Step 2: Change the `QueryPipelineResult.route_taken` default**

Replace `route_taken: str = "smart"` (line 47-ish inside the dataclass) with:
```python
route_taken: str = "intelligence"
```

- [ ] **Step 3: Remove the fast-path branch in `run_query_pipeline`**

Delete lines 99-107 (the block starting with `# Step 2: Fast path` through `return fast_response`). The function now proceeds directly from Step 1 (route) to Step 3 (full-reasoning path).

The comment `# Step 3: Full-reasoning path` should become `# Step 2: Plan -> Execute -> Assemble -> Generate`. Renumber the remaining `# Step 4`, `# Step 5` comments accordingly.

- [ ] **Step 4: Simplify the `route_taken` assignment**

In the final `return QueryPipelineResult(...)` block, replace:
```python
route_taken="simple" if _is_fast_path(route_intent) else "full_reasoning",
```
with:
```python
route_taken="intelligence",
```

- [ ] **Step 5: Delete `_is_fast_path` and `_handle_fast_path` helpers**

Delete lines 209-248 entirely (both the `_is_fast_path` function and the `_handle_fast_path` function).

- [ ] **Step 6: Run the grep-gate test again**

Run:
```bash
python -m pytest tests/batch0/test_grep_gate.py::test_grep_gate_no_fast_smart_refs -v
```
Expected: still may fail on `src/serving/model_router.py` docstrings — to be fixed in Task 7.

- [ ] **Step 7: Verify the pipeline still imports cleanly**

Run:
```bash
python -c "from src.query.pipeline import run_query_pipeline, QueryPipelineResult; print('ok')"
```
Expected: `ok`.

- [ ] **Step 8: Commit**

```bash
git add src/query/pipeline.py
git commit -m "batch-0: collapse query pipeline to single unified-model path"
```

---

## Task 7: Scrub fast/smart wording from remaining docstrings

**Files:**
- Modify: `src/serving/model_router.py` (header docstring lines 1-7, class docstring lines 151-164, intent-types section)

- [ ] **Step 1: Rewrite the module docstring at `src/serving/model_router.py:1-7`**

Replace lines 1-7:

Before:
```python
"""LLM-based intent classification for the unified DocWain serving layer.

Historical context: this module used to route between a 14B "fast" and 27B
"smart" vLLM instance. DocWain is now a single unified model; the classifier
survives because intent still drives retrieval strategy, KG expansion, and
visualization choices — not model selection.
"""
```

After:
```python
"""LLM-based intent classification for the DocWain serving layer.

Intent drives retrieval strategy, KG expansion, and visualization choices.
DocWain uses one unified model; this module never selects a model.
"""
```

- [ ] **Step 2: Rewrite the `IntentRouter` class docstring**

Around line 151-164, replace the docstring:

Before:
```python
class IntentRouter:
    """Classifies queries by intent for downstream retrieval/generation logic.

    Uses the unified DocWain vLLM instance with guided JSON output. Falls back
    to keyword heuristics if the model is unavailable.
```

After:
```python
class IntentRouter:
    """Classifies queries by intent for downstream retrieval/generation logic.

    Uses the DocWain vLLM instance with guided JSON output. Falls back to
    keyword heuristics if the model is unavailable.
```

- [ ] **Step 3: Grep for any remaining `fast` / `smart` in `src/serving/`**

Run:
```bash
grep -rn -iE "fast.path|docwain.fast|smart.model|14B|27B" src/serving/
```
Expected: empty. If any remain, edit the file and remove the reference.

- [ ] **Step 4: Run the grep-gate test — must now PASS**

Run:
```bash
python -m pytest tests/batch0/test_grep_gate.py -v
```
Expected: **both tests PASS** (`test_grep_gate_no_fast_smart_refs` and `test_intelligence_handler_importable`).

If it still fails, the offender list in the assertion message names the exact file:line. Fix each one and re-run until green.

- [ ] **Step 5: Commit (milestone: unified-model complete)**

```bash
git add src/serving/model_router.py
git commit -m "batch-0: scrub fast/smart wording from model_router docstrings"
```

---

## Task 8: Write the qdrant-roundtrip integration test (TDD)

This is the primary regression guard that proves the retriever can read what the pipeline writes, using the actual payload schema from `b0c7211`.

**Files:**
- Create: `tests/batch0/fixtures/invoice_fixture.json`
- Create: `tests/batch0/test_qdrant_roundtrip.py`

- [ ] **Step 1: Write the fixture**

Write `tests/batch0/fixtures/invoice_fixture.json`:

```json
{
  "document_id": "batch0-fixture-inv-001",
  "source_name": "fixture_invoice_001.pdf",
  "doc_domain": "generic",
  "subscription_id": "batch0-fixture-sub",
  "profile_id": "batch0-fixture-prof",
  "chunks": [
    {
      "text": "Invoice number INV-778899 issued to Acme Corp on 2026-03-15 for a total of $12,345.67 payable within 30 days.",
      "type": "text",
      "hash": "sha256:fixture1",
      "token_count": 28,
      "section": {"id": "sec-1", "kind": "invoice_header", "title": "Header", "path": [], "level": 0},
      "provenance": {"page_start": 1, "page_end": 1}
    },
    {
      "text": "Line item: 100 widgets at $100.00 each. Line item: 50 sprockets at $46.91 each. Subtotal $12,345.50.",
      "type": "text",
      "hash": "sha256:fixture2",
      "token_count": 24,
      "section": {"id": "sec-2", "kind": "line_items", "title": "Line Items", "path": [], "level": 0},
      "provenance": {"page_start": 1, "page_end": 1}
    }
  ]
}
```

- [ ] **Step 2: Write the roundtrip test**

Write `tests/batch0/test_qdrant_roundtrip.py`:

```python
"""Qdrant roundtrip: write a chunk via payload_builder, read it via the
retriever's filter builder — asserts the read-side and write-side agree on
the indexed flat payload keys (chunk_id, resolution, section_kind, page,
source_name, doc_domain, etc.).

This test is the regression guard for commit b0c7211 — every subsequent
batch must keep it green.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from src.embedding.payload_builder import build_enriched_payload

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "invoice_fixture.json"

# Payload keys the retriever's FieldCondition filters rely on.
REQUIRED_FLAT_KEYS = {
    "subscription_id",
    "profile_id",
    "document_id",
    "chunk_id",
    "resolution",
    "chunk_kind",
    "section_id",
    "section_kind",
    "page",
    "source_name",
    "doc_domain",
    "embed_pipeline_version",
}


@pytest.fixture(scope="module")
def fixture_doc():
    return json.loads(FIXTURE.read_text())


def _build_payload(fixture_doc, chunk_idx):
    chunk = fixture_doc["chunks"][chunk_idx]
    return build_enriched_payload(
        chunk=chunk,
        chunk_index=chunk_idx,
        document_id=fixture_doc["document_id"],
        subscription_id=fixture_doc["subscription_id"],
        profile_id=fixture_doc["profile_id"],
        extraction_data={"entities": []},
        screening_summary={"entity_scores": {}, "domain_tags": [], "doc_category": "invoice"},
        source_name=fixture_doc["source_name"],
        doc_domain=fixture_doc["doc_domain"],
    )


def test_payload_has_all_indexed_flat_keys(fixture_doc):
    payload = _build_payload(fixture_doc, 0)
    missing = REQUIRED_FLAT_KEYS - set(payload.keys())
    assert not missing, f"Payload missing indexed keys: {missing}"


def test_payload_values_match_input(fixture_doc):
    payload = _build_payload(fixture_doc, 0)
    assert payload["subscription_id"] == fixture_doc["subscription_id"]
    assert payload["profile_id"] == fixture_doc["profile_id"]
    assert payload["document_id"] == fixture_doc["document_id"]
    assert payload["resolution"] == "chunk"
    assert payload["section_kind"] == fixture_doc["chunks"][0]["section"]["kind"]
    assert payload["source_name"] == fixture_doc["source_name"]
    assert payload["doc_domain"] == fixture_doc["doc_domain"]


def test_retriever_filter_uses_only_indexed_keys(fixture_doc):
    """The retriever's _build_filter must only reference keys present in
    the payload. Catches silent drift where the writer emits `document_id`
    but the reader filters on `doc_id`.
    """
    from src.retrieval.retriever import UnifiedRetriever

    class _StubQdrant:
        def collection_exists(self, _name):
            return True

    class _StubEmbedder:
        def encode(self, texts):
            return [[0.0] * 8 for _ in texts]

    r = UnifiedRetriever(qdrant_client=_StubQdrant(), embedder=_StubEmbedder())
    qfilter = r._build_filter(
        subscription_id=fixture_doc["subscription_id"],
        profile_id=fixture_doc["profile_id"],
        document_ids=[fixture_doc["document_id"]],
        chunks_only=True,
    )
    payload = _build_payload(fixture_doc, 0)
    for condition in qfilter.must:
        key = condition.key
        assert key in payload, (
            f"Retriever filter references key '{key}' which the writer "
            f"does not emit (payload keys: {sorted(payload.keys())})"
        )


def test_intelligence_retrieval_uses_only_indexed_keys(fixture_doc):
    """src/intelligence/retrieval.py must also only use keys the writer emits."""
    import inspect

    from src.intelligence import retrieval as intel_retrieval

    source = inspect.getsource(intel_retrieval)
    payload = _build_payload(fixture_doc, 0)
    # Any qdrant FieldCondition(key="X", ...) in the intelligence module
    # must use a key from the writer's payload. Scan the source for
    # obvious references to legacy names.
    FORBIDDEN = {"doc_id", "chunk.id", "section.id", "provenance.page_start"}
    for name in FORBIDDEN:
        if f'"{name}"' in source or f"'{name}'" in source:
            # Legacy key found — check it's either defensive (uses .get with
            # a fallback to an indexed key) or a bug we must fix.
            raise AssertionError(
                f"src/intelligence/retrieval.py references legacy payload "
                f"key '{name}' — rewrite to use an indexed flat key from "
                f"{sorted(payload.keys())}"
            )
```

- [ ] **Step 3: Run the roundtrip tests — record pass/fail of each**

Run:
```bash
python -m pytest tests/batch0/test_qdrant_roundtrip.py -v
```
Expected:
- `test_payload_has_all_indexed_flat_keys` — PASS (the writer is already correct post-`b0c7211`).
- `test_payload_values_match_input` — PASS.
- `test_retriever_filter_uses_only_indexed_keys` — PASS (the `retriever.py` I've read uses `subscription_id` / `profile_id` / `document_id` / `resolution`, all indexed).
- `test_intelligence_retrieval_uses_only_indexed_keys` — may PASS or FAIL depending on legacy key usage in `src/intelligence/retrieval.py`.

- [ ] **Step 4: Commit**

```bash
git add tests/batch0/test_qdrant_roundtrip.py tests/batch0/fixtures/invoice_fixture.json
git commit -m "batch-0: add qdrant-roundtrip regression guard for payload schema"
```

---

## Task 9: Run the audit, fix any mismatches the test finds

**Files:**
- Create: `scripts/batch0/audit_qdrant_filters.py`
- Modify: any file the audit flags — see Step 3 below.

- [ ] **Step 1: Write the audit script**

Write `scripts/batch0/audit_qdrant_filters.py`:

```python
"""Audit: list every qdrant read-side payload key reference in src/.

Walks every .py file under src/ (excluding the write path), finds
FieldCondition(key="..."), payload.get("..."), and payload["..."]
references, and prints a flat list sorted by frequency. Any key not
present in the writer's output (build_enriched_payload) is a suspect
for mismatch.

One-shot audit used during Batch 0 cleanup. Safe to re-run.
"""
from __future__ import annotations

import pathlib
import re
import sys
from collections import Counter

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
WRITE_PATH_DIRS = {
    "src/embedding/",
    "src/embed/",
    "src/extraction/",
    "src/tasks/",
    "src/api/extraction_service.py",
    "src/api/extraction_pipeline_api.py",
    "src/api/embedding_service.py",
    "src/api/dw_document_extractor.py",
    "src/api/qdrant_indexes.py",
    "src/api/qdrant_setup.py",
    "src/api/vector_store.py",
}

# Keys the writer (build_enriched_payload) emits as of b0c7211.
WRITER_FLAT_KEYS = {
    "subscription_id", "profile_id", "document_id",
    "chunk_id", "resolution", "chunk_kind",
    "section_id", "section_kind", "page",
    "source_name", "doc_domain", "embed_pipeline_version",
    # Enrichment (also top-level)
    "entities", "entity_types", "domain_tags", "doc_category",
    "importance_score", "kg_node_ids", "quality_grade", "text",
    # Nested objects (legacy-compatible)
    "chunk", "section", "provenance",
}

PATTERNS = [
    re.compile(r'FieldCondition\s*\(\s*key\s*=\s*["\']([^"\']+)["\']'),
    re.compile(r'payload\.get\s*\(\s*["\']([^"\']+)["\']'),
    re.compile(r'payload\s*\[\s*["\']([^"\']+)["\']\s*\]'),
]


def is_write_path(relpath: str) -> bool:
    return any(relpath.startswith(p) for p in WRITE_PATH_DIRS)


def main():
    references: Counter[str] = Counter()
    per_key: dict[str, list[str]] = {}
    for path in (REPO_ROOT / "src").rglob("*.py"):
        rel = str(path.relative_to(REPO_ROOT))
        if is_write_path(rel):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            for pat in PATTERNS:
                for m in pat.finditer(line):
                    key = m.group(1)
                    references[key] += 1
                    per_key.setdefault(key, []).append(f"{rel}:{line_no}")

    unknown = sorted(k for k in references if k not in WRITER_FLAT_KEYS)
    known = sorted(k for k in references if k in WRITER_FLAT_KEYS)

    print("=== Read-side payload key audit ===")
    print(f"Scanned src/ (write path excluded). Writer-known keys used:")
    for k in known:
        print(f"  {k:35s} {references[k]:4d}x")
    print()
    print(f"Keys NOT emitted by the current writer (potential mismatch):")
    if not unknown:
        print("  (none)")
        return 0
    for k in unknown:
        print(f"  {k!r} — {references[k]}x")
        for loc in per_key[k][:10]:
            print(f"      {loc}")
        if len(per_key[k]) > 10:
            print(f"      ...and {len(per_key[k]) - 10} more")
    print()
    print(f"ACTION: for each 'unknown' key, either rewrite the caller to use")
    print(f"a writer-known key, or prove the caller is defensive (has a")
    print(f"fallback to a writer-known key).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run the audit**

Run:
```bash
python scripts/batch0/audit_qdrant_filters.py | tee /tmp/batch0-audit.txt
```
Expected: a list of "unknown" (writer-not-emitted) keys, if any. Known suspects to watch: `doc_id`, `chunk.id` (as string), `section.id` (as string), `provenance.page_start` when used without fallback.

- [ ] **Step 3: Resolve each unknown key**

For each key the audit flags as unknown, open the listed file(s) and decide:

| Found pattern | Fix |
|---|---|
| `payload.get("doc_id")` with no fallback to `"document_id"` | Change to `payload.get("document_id") or payload.get("doc_id")` (defensive for any legacy points not yet backfilled). |
| `payload["chunk"]["id"]` used alone | Change to `payload.get("chunk_id") or payload.get("chunk", {}).get("id", "")`. |
| `FieldCondition(key="doc_id", ...)` | Rewrite to `FieldCondition(key="document_id", ...)`. Legacy key is not indexed and will always filter to zero. |
| `FieldCondition(key="page_start", ...)` or `"provenance.page_start"` | Rewrite to `FieldCondition(key="page", ...)`. |
| Nested-key filter e.g. `key="section.kind"` | Rewrite to `key="section_kind"`. Qdrant does not recursively filter on nested dicts unless the index was created with a nested schema, which this collection isn't. |
| Anything else | Read the surrounding context, determine the writer-emitted equivalent, and rewrite. Do not touch the writer. |

Stage each fix one file at a time. Keep edits minimal — no unrelated refactoring.

- [ ] **Step 4: Run the roundtrip tests — must now all PASS**

Run:
```bash
python -m pytest tests/batch0/test_qdrant_roundtrip.py -v
```
Expected: all four tests PASS.

- [ ] **Step 5: Re-run the audit — "unknown keys" section must be empty or only show defensive-fallback references**

Run:
```bash
python scripts/batch0/audit_qdrant_filters.py
```
Expected: either `(none)` under unknown, or unknown entries that the engineer can point to as defensive `.get(..., fallback)` usage. Document the latter in the PR description.

- [ ] **Step 6: Commit**

```bash
git add scripts/batch0/audit_qdrant_filters.py src/retrieval/ src/intelligence/retrieval.py
git commit -m "batch-0: align read-side qdrant filters with b0c7211 payload schema"
```

(If no source files outside `scripts/` were modified, commit only the script with a note: `git commit -m "batch-0: add qdrant read-side audit script (no mismatches found)"`.)

---

## Task 10: Write the prompt-path contract test (TDD)

The rule (per `feedback_prompt_paths` memory): response-formatting lives in `src/generation/prompts.py`, not `src/intelligence/generator.py`. This test asserts the rule mechanically.

**Files:**
- Create: `tests/batch0/test_prompt_path_contract.py`

- [ ] **Step 1: Write the contract test**

Write `tests/batch0/test_prompt_path_contract.py`:

```python
"""Prompt-path contract: user-visible response formatting instructions must
live in src/generation/prompts.py, never in src/intelligence/generator.py.

Rule source: user feedback memory feedback_prompt_paths.md.
"""
from __future__ import annotations

import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_intelligence_generator_has_no_format_instructions_dict():
    """`_FORMAT_INSTRUCTIONS` dict (table/bullets/sections/numbered/prose)
    must not live in intelligence/generator.py — it belongs in prompts.py.
    """
    path = REPO_ROOT / "src" / "intelligence" / "generator.py"
    if not path.exists():
        return  # whole file deleted — rule satisfied vacuously
    text = path.read_text(encoding="utf-8")
    assert "_FORMAT_INSTRUCTIONS" not in text, (
        "src/intelligence/generator.py still defines _FORMAT_INSTRUCTIONS. "
        "Move the dict to src/generation/prompts.py."
    )


def test_generation_prompts_has_format_instructions():
    """prompts.py must own the format templates that intelligence/generator
    used to own. This catches an accidental deletion that leaves no format
    templates anywhere.
    """
    path = REPO_ROOT / "src" / "generation" / "prompts.py"
    text = path.read_text(encoding="utf-8")
    for key in ("table", "bullets", "sections", "numbered", "prose"):
        assert key in text, (
            f"src/generation/prompts.py is missing format key {key!r}. "
            "After Task 11 all five format instructions should live here."
        )


def test_reasoning_engine_imports_prompts_not_intelligence_generator():
    """reasoning_engine.py should not construct IntelligentGenerator any
    more — it should use build_reason_prompt from generation.prompts.
    """
    path = REPO_ROOT / "src" / "intelligence" / "reasoning_engine.py"
    text = path.read_text(encoding="utf-8")
    assert "from src.intelligence.generator import IntelligentGenerator" not in text, (
        "reasoning_engine.py still imports IntelligentGenerator. "
        "Switch to src.generation.prompts.build_reason_prompt + "
        "src.generation.reasoner."
    )
```

- [ ] **Step 2: Run the contract test — all three must fail initially**

Run:
```bash
python -m pytest tests/batch0/test_prompt_path_contract.py -v
```
Expected: all three tests **FAIL** (generator.py still has `_FORMAT_INSTRUCTIONS`; prompts.py doesn't have the format keys yet; reasoning_engine still imports `IntelligentGenerator`).

- [ ] **Step 3: Commit the (failing) contract**

```bash
git add tests/batch0/test_prompt_path_contract.py
git commit -m "batch-0: add prompt-path contract test (expected to fail pre-repoint)"
```

---

## Task 11: Repoint response formatting to `src/generation/prompts.py`

**Files:**
- Modify: `src/generation/prompts.py` (add `_FORMAT_INSTRUCTIONS` and a helper)
- Modify: `src/intelligence/generator.py` (delete `_FORMAT_INSTRUCTIONS`; reach into prompts.py for the helper)
- Modify: `src/intelligence/reasoning_engine.py:96-97` (use the reasoner instead of `IntelligentGenerator`)

- [ ] **Step 1: Add `_FORMAT_INSTRUCTIONS` and `get_format_instruction()` to `src/generation/prompts.py`**

Open `src/generation/prompts.py` and add near the top (after the existing module docstring, before the first `def build_*`):

```python
# Response-format templates. Owned here (not in src/intelligence/generator.py)
# per the prompt-path rule. Each template instructs the model on how to
# structure a user-facing response.
_FORMAT_INSTRUCTIONS = {
    "table": (
        "Present data in a clean markdown table.\n"
        "- Use | column | headers | with alignment\n"
        "- One data point per row, no merged cells\n"
        "- Bold key values: **$9,000.00**, **Jessica Jones**\n"
        "- Add a brief summary sentence above the table"
    ),
    "bullets": (
        "Present as a structured bulleted list.\n"
        "- Lead with a one-line summary sentence\n"
        "- Group related bullets under **bold category headers**\n"
        "- Each bullet: **Label:** value or description\n"
        "- Bold key names, amounts, dates, entities\n"
        "- Most important items first"
    ),
    "sections": (
        "Organize the response with clear visual hierarchy.\n"
        "- Start with a one-line executive summary\n"
        "- Use ## for major sections, ### for subsections\n"
        "- Within sections, use bullet points with **bold labels**\n"
        "- Format: **Field Name:** extracted value or insight\n"
        "- Bold all key values: names, amounts, dates, identifiers\n"
        "- Use markdown tables for tabular data (line items, comparisons)\n"
        "- Never leave headers as plain text — always use ## or ###\n"
        "- Keep bullets self-contained — each makes sense alone\n"
        "- End with a brief synthesis or key takeaway if appropriate"
    ),
    "numbered": (
        "Present as a numbered list.\n"
        "- Each item: **Label** — description with **bold key values**\n"
        "- Sequential order, one point per number\n"
        "- Brief summary before the list"
    ),
    "prose": (
        "Write clear, structured paragraphs.\n"
        "- Lead with the direct answer in the first sentence\n"
        "- Bold key values: **$9,000.00**, **Jessica Jones**, **Document 0522**\n"
        "- Use short paragraphs (2-3 sentences each)\n"
        "- For any tabular data, use a markdown table instead of inline text"
    ),
}


def get_format_instruction(shape: str) -> str:
    """Return the format template for the given shape. Defaults to 'prose'."""
    return _FORMAT_INSTRUCTIONS.get(shape, _FORMAT_INSTRUCTIONS["prose"])
```

- [ ] **Step 2: Update `src/intelligence/generator.py` to re-export from prompts**

Open `src/intelligence/generator.py`. Find the `_FORMAT_INSTRUCTIONS = { ... }` block (lines 16-60-ish). Replace the whole block with:

```python
from src.generation.prompts import _FORMAT_INSTRUCTIONS  # re-export for backwards compatibility
```

If there is any code in the file that uses `_FORMAT_INSTRUCTIONS[...]`, leave it — it still resolves via the re-export. Grep to confirm:

```bash
grep -n "_FORMAT_INSTRUCTIONS" src/intelligence/generator.py
```

- [ ] **Step 3: Update `src/intelligence/reasoning_engine.py:96-97`**

Open `src/intelligence/reasoning_engine.py` around line 96. You should see:

```python
from src.intelligence.generator import IntelligentGenerator
self._generator = IntelligentGenerator(self._llm)
```

This has to change to use the generation-layer reasoner, which is the canonical path. Replace the two lines with:

```python
from src.generation.reasoner import Reasoner
self._generator = Reasoner(self._llm)
```

Then grep all usages of `self._generator.` in the file. If any method call no longer exists on `Reasoner`, open `src/generation/reasoner.py` to confirm the correct name, and update the call site to match.

If the `Reasoner` API differs materially (e.g. takes different args on `.generate()`), write a thin adapter in `src/intelligence/reasoning_engine.py` that maps old args → new args. Keep the adapter private to this file.

- [ ] **Step 4: Run the prompt-path contract test — must PASS**

Run:
```bash
python -m pytest tests/batch0/test_prompt_path_contract.py -v
```
Expected: all three tests PASS.

- [ ] **Step 5: Run the unit tests that exercise reasoning_engine**

Run:
```bash
python -m pytest tests/ -k "reasoning_engine or intelligent_generator or intel_generator" --continue-on-collection-errors -q
```
Expected: any tests that were green before remain green. If a test was green and is now red, the repoint adapter needs another tweak — fix and re-run.

- [ ] **Step 6: Commit (milestone: prompt-path complete)**

```bash
git add src/generation/prompts.py src/intelligence/generator.py src/intelligence/reasoning_engine.py
git commit -m "batch-0: move response-format templates to generation/prompts.py"
```

---

## Task 12: Write the 10-query smoke test

This is the third Batch-0 exit criterion — prove that ten canned queries across major intents all return non-empty, grounded-where-applicable, DocWain-persona responses against a fixture profile.

**Files:**
- Create: `tests/batch0/test_canned_queries_smoke.py`

- [ ] **Step 1: Write the smoke test**

Write `tests/batch0/test_canned_queries_smoke.py`:

```python
"""Ten-query smoke test — the third Batch-0 exit criterion.

Runs ten canned queries (one per major intent) against the owner's
fixture profile, asserts each returns:
  * non-empty response
  * grounded=True when the intent requires evidence
  * DocWain persona present in greeting/identity responses
  * no "I'm having trouble" static fallback text (would indicate
    upstream failure masked by the fallback path)

Marked @pytest.mark.live and skipped unless DOCWAIN_SMOKE_PROFILE is
set in the environment, because it needs a live vLLM + qdrant + mongo.
"""
from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.live

SMOKE_PROFILE = os.environ.get("DOCWAIN_SMOKE_PROFILE")
SMOKE_SUB = os.environ.get("DOCWAIN_SMOKE_SUB")

pytest_plugins = []  # avoid autoloading conftests from other test dirs

skip_reason = "Set DOCWAIN_SMOKE_PROFILE and DOCWAIN_SMOKE_SUB to run."

CANNED_QUERIES = [
    ("greet",     "Hello",                        False, True),
    ("identity",  "Who are you?",                 False, True),
    ("lookup",    "What is the invoice total on INV-778899?", True,  False),
    ("list",      "List all invoices from Acme Corp.",        True,  False),
    ("count",     "How many documents do I have uploaded?",   True,  False),
    ("extract",   "Extract all line items from the latest invoice.", True, False),
    ("summarize", "Summarize the most recent document.",      True,  False),
    ("compare",   "Compare the two most recent invoices.",    True,  False),
    ("analyze",   "What does the data suggest about spending trends?", True, False),
    ("timeline",  "Show the timeline of invoices received.",  True,  False),
]


@pytest.mark.skipif(not (SMOKE_PROFILE and SMOKE_SUB), reason=skip_reason)
@pytest.mark.parametrize("intent,query,needs_grounding,needs_persona", CANNED_QUERIES)
def test_smoke_query(intent, query, needs_grounding, needs_persona):
    from src.query.pipeline import run_query_pipeline
    from src.api.app_lifespan import get_clients_for_smoke  # provided by app lifecycle

    clients = get_clients_for_smoke()
    result = run_query_pipeline(
        query=query,
        profile_id=SMOKE_PROFILE,
        subscription_id=SMOKE_SUB,
        clients=clients,
    )
    assert result.response, f"Empty response for intent={intent} query={query!r}"
    assert "I'm having trouble" not in result.response, (
        f"Static fallback surfaced for intent={intent} — upstream failure masked."
    )
    if needs_grounding:
        assert result.context_found, f"No context found for intent={intent} query={query!r}"
    if needs_persona:
        assert "DocWain" in result.response, (
            f"DocWain persona missing in intent={intent} response: {result.response[:200]}"
        )
```

- [ ] **Step 2: Add `get_clients_for_smoke()` to `src/api/app_lifespan.py` if it does not exist**

Grep first:
```bash
grep -n "def get_clients_for_smoke" src/api/app_lifespan.py 2>/dev/null
```

If present, skip this step. If not, add this function at the bottom of `src/api/app_lifespan.py`:

```python
def get_clients_for_smoke():
    """Build a minimal clients dict for the Batch-0 smoke test.

    Called only by tests/batch0/test_canned_queries_smoke.py; relies on
    the same backing stores the live app uses. Falls back to None for
    anything that is not available in the smoke environment.
    """
    clients = {}
    try:
        from src.api.app_lifespan import AppState
        state = AppState.get()  # or however the app surfaces its singleton
        clients["vllm_manager"] = getattr(state, "vllm_manager", None)
        clients["llm_gateway"] = getattr(state, "llm_gateway", None)
        clients["qdrant_client"] = getattr(state, "qdrant_client", None)
        clients["neo4j_driver"] = getattr(state, "neo4j_driver", None)
        clients["mongo_db"] = getattr(state, "mongo_db", None)
        clients["embedder"] = getattr(state, "embedder", None)
    except Exception:
        pass
    return clients
```

If `AppState.get()` is not the right accessor in this codebase, grep for how other test helpers construct the clients dict and adapt. Do not introduce a new DI pattern — reuse what exists.

- [ ] **Step 3: Sanity-check the smoke test is properly skipped without env vars**

Run:
```bash
python -m pytest tests/batch0/test_canned_queries_smoke.py -v
```
Expected: 10 tests SKIPPED with reason "Set DOCWAIN_SMOKE_PROFILE and DOCWAIN_SMOKE_SUB to run."

- [ ] **Step 4: Commit**

```bash
git add tests/batch0/test_canned_queries_smoke.py src/api/app_lifespan.py
git commit -m "batch-0: add 10-query smoke test (skipped without live env vars)"
```

---

## Task 13: Run the live smoke test against the owner's profile

This is the human-judged canary step from spec §6.4 / §7.5. It is not automated — requires the owner to set env vars and run the test against a live environment with real documents already uploaded.

**Files:** no new files.

- [ ] **Step 1: Confirm vLLM is in serving mode**

Run:
```bash
cat /tmp/docwain-gpu-mode.json
```
Expected: `{"mode": "serving", ...}`. If training mode, the smoke test will run against Ollama Cloud instead — record that in the PR description but still proceed; the regression should be fixed regardless of backend.

- [ ] **Step 2: Ensure the owner profile has fixture-or-real documents uploaded**

The owner's profile (whatever `DOCWAIN_SMOKE_PROFILE` resolves to) must have at least one document already processed end-to-end (uploaded → extracted → embedded → qdrant-written). Check with:

```bash
# Replace <sub> and <prof> with actual ids
python -c "
from src.api.vector_store import build_collection_name, get_qdrant_client_singleton
c = get_qdrant_client_singleton()
name = build_collection_name('<sub>')
info = c.get_collection(name)
print('points:', info.points_count)
"
```
Expected: points_count > 0.

- [ ] **Step 3: Run the smoke test live**

Run:
```bash
DOCWAIN_SMOKE_PROFILE=<owner-profile-id> \
DOCWAIN_SMOKE_SUB=<owner-sub-id> \
python -m pytest tests/batch0/test_canned_queries_smoke.py -v --tb=short
```
Expected: all 10 parameterised cases PASS. If any fail:

| Failure | Likely cause | Fix |
|---|---|---|
| Empty response for `greet` / `identity` | No-retrieval path broken | Check `IntelligenceHandler._handle_no_retrieval` with `pytest -v -s`. Re-verify the vLLM manager health check. |
| Empty / wrong response for `lookup` / `list` / `count` | Retrieval still broken despite Task 9 | Rerun `audit_qdrant_filters.py`; check qdrant points actually match the filter keys; inspect `retriever._point_to_chunk`. |
| "I'm having trouble" static fallback | vLLM or LLM gateway failing | Check vLLM logs; check `vllm_manager.health_check()`. |
| "DocWain" persona missing | Identity prompt not flowing through `build_system_prompt` | Check `src/generation/prompts.py::build_system_prompt` returns the DocWain persona for the smoke profile's domain. |

Iterate until all 10 pass. Do **not** merge until they do.

- [ ] **Step 4: Record the smoke-test output**

Save the full run output to `eval_results/batch-0-smoke.txt`:

```bash
DOCWAIN_SMOKE_PROFILE=<owner-profile-id> \
DOCWAIN_SMOKE_SUB=<owner-sub-id> \
python -m pytest tests/batch0/test_canned_queries_smoke.py -v > eval_results/batch-0-smoke.txt 2>&1
```

- [ ] **Step 5: Commit the smoke-run artifact**

```bash
git add -f eval_results/batch-0-smoke.txt
git commit -m "batch-0: record live 10-query smoke-test output (all passing)"
```

---

## Task 14: Final verification, PR, canary deploy

**Files:** no code changes; PR body + deploy notes only.

- [ ] **Step 1: Verify all Batch-0 tests pass together**

Run:
```bash
python -m pytest tests/batch0/ -v
```
Expected: all green except the smoke test's 10 cases, which are SKIPPED without env vars (they already passed in Task 13).

- [ ] **Step 2: Verify full test suite has no new regressions**

Run:
```bash
python -m pytest tests/ --continue-on-collection-errors -q --tb=no --no-header --ignore=tests/live 2>&1 | tail -10
```
Compare pass/fail counts against `eval_results/pre-batch-0-test-baseline.txt`. Rule: **failed count must not increase**. Passed count may increase (if our changes fixed something) but must not decrease.

- [ ] **Step 3: Confirm the do-not-touch list is clean**

Run:
```bash
git diff main...HEAD --name-only | grep -E "^src/(api/(pipeline_api|extraction|extraction_pipeline|embedding_service|dw_document_extractor|qdrant_indexes|qdrant_setup|vector_store)|extraction/|embed/|embedding/|tasks/embedding|intelligence_v2/)"
```
Expected: **empty output**. If anything lists, those edits were out of scope for Batch 0 — revert them with `git checkout main -- <file>` and move them to a Batch-2 follow-up issue.

- [ ] **Step 4: Push the branch**

```bash
git push -u origin batch-0-unified-model-qdrant-audit
```

- [ ] **Step 5: Open the PR**

Run:
```bash
gh pr create --base main --head batch-0-unified-model-qdrant-audit \
  --title "batch-0: unified model + qdrant audit + prompt-path repoint" \
  --body "$(cat <<'EOF'
## Summary

Batch 0 of the intelligence/RAG re-integration workstream (spec: `docs/superpowers/specs/2026-04-21-intelligence-rag-redesign-design.md`).

Goal: restore yesterday's intelligence-layer accuracy without touching the document-processing pipeline.

## Changes

- Deleted `src/serving/fast_path.py`; added `src/serving/intelligence_handler.py` as the unified handler.
- Renamed `docwain-fast` → `docwain` in config, VLLMManager, and supporting files.
- Simplified `src/query/pipeline.py` to a single Route → Plan → Execute → Generate+Verify flow.
- Audited read-side qdrant payload filters against the writer's flat-indexed keys from `b0c7211`; fixed any mismatches.
- Moved `_FORMAT_INSTRUCTIONS` (response-formatting templates) from `src/intelligence/generator.py` to `src/generation/prompts.py`.
- Added four Batch-0 test suites: grep gate, qdrant roundtrip, prompt-path contract, 10-query live smoke.

## Exit criteria (from spec §6.1 row 0)

- [x] Grep gate clean (no `docwain-fast` / `fast path` / `smart path` / `14B` / `27B` in live code)
- [x] Qdrant retrieval integration test passes (`tests/batch0/test_qdrant_roundtrip.py`)
- [x] 10-query live smoke test passes on owner profile (`eval_results/batch-0-smoke.txt`)
- [x] Existing pipeline tests unchanged-green (`eval_results/pre-batch-0-test-baseline.txt` comparison)
- [x] Do-not-touch list untouched (`git diff --name-only` confirms)

## Rollback

Code revert:
```
git revert <this-PR-merge-sha>
```

No data rollback needed — pipeline write path was not modified.

## Non-goals for this batch

- Phase 0 eval baseline (next batch).
- Any SME work (Batches 2–7).
- Systemd unit file rename (ops PR follow-up).

## Files touched

See `git diff --stat main...HEAD`.
EOF
)"
```

- [ ] **Step 6: Canary deploy**

After PR is reviewed and merged, on prod host:

```bash
# Pull latest main, restart the app service
cd /path/to/DocWain && git pull origin main && sudo systemctl restart docwain-app.service
```

Observe for 1 hour of real queries on the owner's profile. Watch:

```bash
# In one terminal
sudo journalctl -fu docwain-app.service | grep -E "ERROR|Traceback|route_taken|grounded="
```

Success criteria: queries return non-empty, grounded-where-applicable, DocWain-persona responses; error rate in logs matches baseline. If any regression surfaces, `git revert <merge-sha>` and redeploy.

- [ ] **Step 7: Announce completion in the batch log**

Append to `eval_results/batch-0-smoke.txt`:

```
=== Batch 0 complete YYYY-MM-DD HH:MM UTC ===
PR: <url>
Merge sha: <sha>
Canary window: <start> → <end>
Observed: <N> queries, <M> errors, <K> regressions → <action>
Outcome: PASS / ROLLBACK
```

Commit the update:
```bash
git add -f eval_results/batch-0-smoke.txt
git commit -m "batch-0: record canary outcome"
git push origin main
```

This closes Batch 0. Next batch is Phase 0 eval baseline (SME phase 0 commits cherry-picked onto main, flags OFF, baseline snapshot committed).

---

## Self-Review (author-maintained)

- [x] Spec §7.1 (unified-model collapse) → Tasks 3, 4, 5, 6, 7.
- [x] Spec §7.2 (qdrant payload audit) → Tasks 8, 9.
- [x] Spec §7.3 (generation-prompt repoint) → Tasks 10, 11.
- [x] Spec §7.4 (do-not-touch) → Task 14 step 3 verification.
- [x] Spec §7.5 exit criteria → Task 14 steps 1-3 + canary in step 6.
- [x] Spec §9 (test-baseline snapshot) → Task 1.
- [x] No placeholders ("TBD", "handle edge cases", "similar to earlier task") — all tasks have exact code/commands.
- [x] Method-signature consistency: `IntelligenceHandler(vllm_manager=...)` matches across Tasks 3 and 12; `get_format_instruction(shape)` used consistently; `Reasoner(self._llm)` matches the constructor signature in `src/generation/reasoner.py`.
- [x] Every task ends with a commit step.
- [x] Every test step states expected pass/fail explicitly.
