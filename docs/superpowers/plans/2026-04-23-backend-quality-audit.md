# Backend Quality Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a one-shot Python harness that runs 10 DocWain queries against three serving backends (vLLM local bf16, Ollama local Q5_K_M, Ollama Cloud `qwen3.5:397b`) with identical retrieval + prompt, captures side-by-side responses, and produces a written quality verdict.

**Architecture:** Single script `scripts/backend_quality_audit.py` that: (1) bootstraps a Qdrant client + SentenceTransformer embedder using existing repo helpers, (2) uses the existing `HybridRetriever` to pull chunks once per query, (3) builds the prompt once via `src.generation.prompts.build_system_prompt` + `build_reason_prompt`, (4) POSTs the same payload sequentially to three backend endpoints, (5) writes results.md / results.json. No edits to `src/`. Spec: `docs/superpowers/specs/2026-04-23-backend-quality-audit-design.md`.

**Tech Stack:** Python 3.12, `httpx` (HTTP), `qdrant-client` (already in deps), `sentence-transformers` (already in deps via `src.embedding.model_loader`), `src.generation.prompts` (reuse prompt builders), `src.services.retrieval.hybrid_retriever.HybridRetriever` (reuse retrieval).

**Non-goals (do not expand scope):** no latency p95 analysis, no concurrency curve, no `src/` edits, no gateway routing change, no training data extraction.

---

## File structure

**New files:**
- `scripts/backend_quality_audit.py` — the harness (~200 LOC).
- `tests/test_backend_quality_audit.py` — unit tests for pure helpers (prompt shaping, result serialization).
- `docs/audits/2026-04-23-backend-quality/query_bank.md` — 10 queries, version-controlled.
- `docs/audits/2026-04-23-backend-quality/results.md` — generated side-by-side table.
- `docs/audits/2026-04-23-backend-quality/results.json` — generated raw data.
- `docs/audits/2026-04-23-backend-quality/verdict.md` — generated analysis.

**Modified files:** none.

**Git:** branch `audit/backend-quality-2026-04-23` off `preprod_v01`. Commit after each task.

---

### Task 1: Create audit directory and static query bank

**Files:**
- Create: `docs/audits/2026-04-23-backend-quality/query_bank.md`

- [ ] **Step 1: Create branch off preprod_v01**

```bash
git checkout preprod_v01
git checkout -b audit/backend-quality-2026-04-23
```

- [ ] **Step 2: Write the query bank file**

Create `docs/audits/2026-04-23-backend-quality/query_bank.md` with this exact content:

```markdown
# Backend Quality Audit — Query Bank v0

Ten queries spanning four types. The audit harness reads this file to drive its run.
Operator may edit any query before the run; the harness uses whatever is here at execution time.

| # | Type                  | Prompt |
|---|-----------------------|--------|
| 1 | Extraction QA         | What is the total amount on the most recent invoice? |
| 2 | Extraction QA         | List every vendor that appears across the uploaded purchase orders. |
| 3 | Extraction QA         | From the resumes, extract each candidate's most recent job title and company. |
| 4 | Cross-doc synthesis   | Compare the payment terms between the two quotes we uploaded — which is more favorable? |
| 5 | Cross-doc synthesis   | Are there any duplicate line items across the invoices this month? |
| 6 | Short factual         | Who signed the last contract? |
| 7 | Short factual         | When was the earliest document uploaded? |
| 8 | Response intelligence | Based on the invoices and contracts together, what's our likely exposure to vendor X next quarter? |
| 9 | Response intelligence | Walk through what these documents collectively tell us about this project's risk profile. |
| 10| Response intelligence | What's the smartest question I should be asking about this set of documents that I haven't asked yet? |
```

- [ ] **Step 3: Commit**

```bash
git add -f docs/audits/2026-04-23-backend-quality/query_bank.md
git commit -m "audit: add query bank v0 for backend quality audit"
```

---

### Task 2: Scaffold the harness with CLI + dataclasses

**Files:**
- Create: `scripts/backend_quality_audit.py`
- Create: `tests/test_backend_quality_audit.py`

- [ ] **Step 1: Write the failing test for CLI help**

Create `tests/test_backend_quality_audit.py`:

```python
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "backend_quality_audit.py"


def test_script_help_exits_zero():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert "--profile-id" in result.stdout
    assert "--dry-run" in result.stdout
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
cd /home/ubuntu/PycharmProjects/DocWain
.venv/bin/pytest tests/test_backend_quality_audit.py::test_script_help_exits_zero -x -q
```

Expected: FAIL — script does not exist.

- [ ] **Step 3: Write the scaffold**

Create `scripts/backend_quality_audit.py`:

```python
"""Backend quality audit: vLLM local vs Ollama local vs Ollama Cloud.

See docs/superpowers/specs/2026-04-23-backend-quality-audit-design.md for spec.
One-shot harness; not production code.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

AUDIT_DIR = REPO_ROOT / "docs" / "audits" / "2026-04-23-backend-quality"


@dataclass
class BackendConfig:
    label: str
    kind: str  # "vllm" | "ollama_local" | "ollama_cloud"
    base_url: str
    model: str
    api_key: Optional[str] = None


@dataclass
class QueryItem:
    idx: int
    qtype: str
    prompt: str


@dataclass
class BackendResponse:
    label: str
    text: str
    tokens: int
    wall_ms: float
    error: Optional[str] = None


@dataclass
class QueryRun:
    query: QueryItem
    chunk_ids: List[str]
    responses: Dict[str, BackendResponse] = field(default_factory=dict)


DEFAULT_BACKENDS: List[BackendConfig] = [
    BackendConfig(
        label="A_vllm_local",
        kind="vllm",
        base_url="http://localhost:8100/v1",
        model="docwain-fast",
    ),
    BackendConfig(
        label="B_ollama_local",
        kind="ollama_local",
        base_url="http://localhost:11434",
        model="DHS/DocWain:latest",
    ),
    BackendConfig(
        label="C_ollama_cloud",
        kind="ollama_cloud",
        base_url=os.getenv("OLLAMA_HOST", "https://ollama.com"),
        model=os.getenv("OLLAMA_CLOUD_MODEL", "qwen3.5:397b"),
        api_key=os.getenv("OLLAMA_API_KEY"),
    ),
]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DocWain backend quality audit.")
    parser.add_argument("--profile-id", required=False, help="Qdrant profile ID to ground against.")
    parser.add_argument("--collection", required=False, help="Qdrant collection name.")
    parser.add_argument("--top-k", type=int, default=10, help="Chunks retrieved per query.")
    parser.add_argument("--query-bank", default=str(AUDIT_DIR / "query_bank.md"))
    parser.add_argument("--out-dir", default=str(AUDIT_DIR))
    parser.add_argument("--dry-run", action="store_true", help="Use canned responses; no live calls.")
    parser.add_argument("--preflight-only", action="store_true", help="Ping each backend and exit.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    print(f"[audit] profile_id={args.profile_id} collection={args.collection} "
          f"top_k={args.top_k} dry_run={args.dry_run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the test to confirm it passes**

```bash
.venv/bin/pytest tests/test_backend_quality_audit.py::test_script_help_exits_zero -x -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/backend_quality_audit.py tests/test_backend_quality_audit.py
git commit -m "audit: scaffold backend_quality_audit harness with CLI + dataclasses"
```

---

### Task 3: Parse the query bank

**Files:**
- Modify: `scripts/backend_quality_audit.py` (add `load_query_bank`)
- Modify: `tests/test_backend_quality_audit.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_backend_quality_audit.py`:

```python
from scripts.backend_quality_audit import load_query_bank, QueryItem


def test_load_query_bank_returns_ten_items(tmp_path):
    bank_md = tmp_path / "bank.md"
    bank_md.write_text(
        "# Title\n\n"
        "| # | Type | Prompt |\n"
        "|---|------|--------|\n"
        "| 1 | Extraction QA | What is X? |\n"
        "| 2 | Short factual | Who signed? |\n"
    )
    items = load_query_bank(bank_md)
    assert len(items) == 2
    assert items[0] == QueryItem(idx=1, qtype="Extraction QA", prompt="What is X?")
    assert items[1].prompt == "Who signed?"


def test_load_query_bank_ignores_separator_row(tmp_path):
    bank_md = tmp_path / "bank.md"
    bank_md.write_text(
        "| # | Type | Prompt |\n"
        "|---|------|--------|\n"
        "| 1 | X | Y |\n"
    )
    items = load_query_bank(bank_md)
    assert len(items) == 1
```

- [ ] **Step 2: Run test to confirm fail**

```bash
.venv/bin/pytest tests/test_backend_quality_audit.py -x -q
```

Expected: FAIL — `load_query_bank` not defined.

- [ ] **Step 3: Implement `load_query_bank`**

Add to `scripts/backend_quality_audit.py` (place above `parse_args`):

```python
def load_query_bank(path: Path) -> List[QueryItem]:
    """Parse the query bank markdown table into QueryItem list."""
    text = Path(path).read_text(encoding="utf-8")
    items: List[QueryItem] = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cols = [c.strip() for c in line.strip("|").split("|")]
        if len(cols) < 3:
            continue
        if cols[0] in {"#", ""} or set(cols[0]) <= {"-", ":"}:
            continue
        try:
            idx = int(cols[0])
        except ValueError:
            continue
        items.append(QueryItem(idx=idx, qtype=cols[1], prompt=cols[2]))
    return items
```

- [ ] **Step 4: Run test to confirm pass**

```bash
.venv/bin/pytest tests/test_backend_quality_audit.py -x -q
```

Expected: both new tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/backend_quality_audit.py tests/test_backend_quality_audit.py
git commit -m "audit: load query bank from markdown table"
```

---

### Task 4: Retrieval helper using HybridRetriever

**Files:**
- Modify: `scripts/backend_quality_audit.py`

No unit test for this task — it's integration-only against Qdrant Cloud. Smoke test runs in Step 3.

- [ ] **Step 1: Implement retrieval helper**

Append to `scripts/backend_quality_audit.py` (above `main`):

```python
def build_retriever():
    """Construct a HybridRetriever using env-configured Qdrant + embedder."""
    from qdrant_client import QdrantClient
    from src.embedding.model_loader import get_embedding_model
    from src.services.retrieval.hybrid_retriever import HybridRetriever, HybridRetrieverConfig

    qdrant_url = os.environ["QDRANT_URL"]
    qdrant_key = os.environ["QDRANT_API_KEY"]
    client = QdrantClient(url=qdrant_url, api_key=qdrant_key, timeout=60)
    embedder = get_embedding_model()
    retriever = HybridRetriever(client=client, embedder=embedder, config=HybridRetrieverConfig())
    return retriever, client


def retrieve_once(retriever, *, collection: str, profile_id: str, query: str, top_k: int) -> List[Dict[str, Any]]:
    """Run one retrieval and return normalized chunks."""
    candidates = retriever.retrieve(
        collection_name=collection,
        query=query,
        profile_id=profile_id,
        top_k=top_k,
    )
    if not candidates:
        raise RuntimeError(f"Retrieval returned zero chunks for query: {query!r}")
    chunks = []
    for c in candidates:
        chunks.append({
            "id": str(c.id),
            "text": c.text,
            "score": float(c.score),
            "source_name": c.metadata.get("source_name") or c.source or "unknown",
            "section": c.metadata.get("section") or "",
            "page": c.metadata.get("page_number") or c.metadata.get("page") or 0,
            "source_index": len(chunks) + 1,
        })
    return chunks
```

- [ ] **Step 2: Commit scaffolding**

```bash
git add scripts/backend_quality_audit.py
git commit -m "audit: add retrieval helper (HybridRetriever wrapper)"
```

- [ ] **Step 3: Smoke test retrieval (manual)**

List Qdrant collections to find candidate profiles:

```bash
.venv/bin/python -c "
import os
from qdrant_client import QdrantClient
c = QdrantClient(url=os.environ['QDRANT_URL'], api_key=os.environ['QDRANT_API_KEY'], timeout=60)
for col in c.get_collections().collections:
    info = c.get_collection(col.name)
    print(f'{col.name:50s}  points={info.points_count}')
"
```

Expected: print at least one non-empty collection. Record the collection name and a profile_id that has documents (inspect payloads if needed). **Do not continue to Task 5 until a working (collection, profile_id) pair is identified.**

---

### Task 5: Prompt builder helper

**Files:**
- Modify: `scripts/backend_quality_audit.py`
- Modify: `tests/test_backend_quality_audit.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_backend_quality_audit.py`:

```python
from scripts.backend_quality_audit import build_audit_prompt


def test_build_audit_prompt_returns_system_and_user():
    chunks = [
        {
            "id": "chunk1", "text": "The invoice total is $5,000.",
            "score": 0.9, "source_name": "invoice.pdf", "section": "Total",
            "page": 1, "source_index": 1,
        }
    ]
    out = build_audit_prompt(query="What is the invoice total?", chunks=chunks)
    assert "system" in out
    assert "user" in out
    assert "5,000" in out["user"]
    assert out["system"]  # non-empty


def test_build_audit_prompt_deterministic():
    chunks = [
        {"id": "c", "text": "X", "score": 1.0, "source_name": "s",
         "section": "", "page": 0, "source_index": 1}
    ]
    a = build_audit_prompt(query="q", chunks=chunks)
    b = build_audit_prompt(query="q", chunks=chunks)
    assert a == b
```

- [ ] **Step 2: Run test to confirm fail**

```bash
.venv/bin/pytest tests/test_backend_quality_audit.py -x -q
```

Expected: FAIL — `build_audit_prompt` undefined.

- [ ] **Step 3: Implement `build_audit_prompt`**

Add to `scripts/backend_quality_audit.py`:

```python
def build_audit_prompt(*, query: str, chunks: List[Dict[str, Any]], profile_domain: str = "",
                       task_type: str = "extract", output_format: str = "markdown") -> Dict[str, str]:
    """Build the system + user prompt using production Reasoner helpers.

    Returns {"system": str, "user": str} ready for any OpenAI-compatible chat payload.
    """
    from src.generation.prompts import build_system_prompt, build_reason_prompt

    system = build_system_prompt(profile_domain=profile_domain, kg_context="")
    user = build_reason_prompt(
        query=query,
        task_type=task_type,
        output_format=output_format,
        evidence=chunks,
        doc_context=None,
        conversation_history=None,
    )
    return {"system": system, "user": user}
```

- [ ] **Step 4: Run test to confirm pass**

```bash
.venv/bin/pytest tests/test_backend_quality_audit.py -x -q
```

Expected: both new tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/backend_quality_audit.py tests/test_backend_quality_audit.py
git commit -m "audit: prompt builder wrapping Reasoner helpers"
```

---

### Task 6: Backend callers (vLLM, Ollama local, Ollama Cloud)

**Files:**
- Modify: `scripts/backend_quality_audit.py`
- Modify: `tests/test_backend_quality_audit.py`

- [ ] **Step 1: Write failing test (route selection)**

Append to `tests/test_backend_quality_audit.py`:

```python
from scripts.backend_quality_audit import BackendConfig, call_backend


def test_call_backend_dispatches_by_kind(monkeypatch):
    seen = {}

    def fake_call_vllm(cfg, system, user, params):
        seen["vllm"] = True
        return "vllm-text", 42, 10.0

    def fake_call_ollama_local(cfg, system, user, params):
        seen["ol"] = True
        return "ol-text", 21, 20.0

    def fake_call_ollama_cloud(cfg, system, user, params):
        seen["oc"] = True
        return "oc-text", 7, 30.0

    monkeypatch.setattr("scripts.backend_quality_audit._call_vllm", fake_call_vllm)
    monkeypatch.setattr("scripts.backend_quality_audit._call_ollama_local", fake_call_ollama_local)
    monkeypatch.setattr("scripts.backend_quality_audit._call_ollama_cloud", fake_call_ollama_cloud)

    for kind, key in [("vllm", "vllm"), ("ollama_local", "ol"), ("ollama_cloud", "oc")]:
        cfg = BackendConfig(label="x", kind=kind, base_url="u", model="m")
        text, tokens, ms = call_backend(cfg, system="s", user="u", params={})
        assert seen.get(key) is True
    assert seen == {"vllm": True, "ol": True, "oc": True}
```

- [ ] **Step 2: Run to confirm fail**

```bash
.venv/bin/pytest tests/test_backend_quality_audit.py -x -q
```

Expected: FAIL — `call_backend` undefined.

- [ ] **Step 3: Implement backend callers**

Add to `scripts/backend_quality_audit.py`:

```python
import httpx


DEFAULT_PARAMS = {
    "temperature": 0.4,
    "top_p": 0.85,
    "max_tokens": 2048,
}


def _call_vllm(cfg: BackendConfig, system: str, user: str, params: Dict[str, Any]):
    url = f"{cfg.base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": params.get("temperature", DEFAULT_PARAMS["temperature"]),
        "top_p": params.get("top_p", DEFAULT_PARAMS["top_p"]),
        "max_tokens": params.get("max_tokens", DEFAULT_PARAMS["max_tokens"]),
    }
    t0 = time.perf_counter()
    r = httpx.post(url, json=payload, timeout=300.0)
    r.raise_for_status()
    wall_ms = (time.perf_counter() - t0) * 1000.0
    data = r.json()
    text = data["choices"][0]["message"]["content"]
    tokens = data.get("usage", {}).get("completion_tokens", 0)
    return text, int(tokens), float(wall_ms)


def _call_ollama_local(cfg: BackendConfig, system: str, user: str, params: Dict[str, Any]):
    url = f"{cfg.base_url.rstrip('/')}/api/chat"
    payload = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {
            "temperature": params.get("temperature", DEFAULT_PARAMS["temperature"]),
            "top_p": params.get("top_p", DEFAULT_PARAMS["top_p"]),
            "num_predict": params.get("max_tokens", DEFAULT_PARAMS["max_tokens"]),
        },
    }
    t0 = time.perf_counter()
    r = httpx.post(url, json=payload, timeout=300.0)
    r.raise_for_status()
    wall_ms = (time.perf_counter() - t0) * 1000.0
    data = r.json()
    text = data["message"]["content"]
    tokens = int(data.get("eval_count", 0))
    return text, tokens, float(wall_ms)


def _call_ollama_cloud(cfg: BackendConfig, system: str, user: str, params: Dict[str, Any]):
    url = f"{cfg.base_url.rstrip('/')}/api/chat"
    headers = {"Authorization": f"Bearer {cfg.api_key}"} if cfg.api_key else {}
    payload = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {
            "temperature": params.get("temperature", DEFAULT_PARAMS["temperature"]),
            "top_p": params.get("top_p", DEFAULT_PARAMS["top_p"]),
            "num_predict": params.get("max_tokens", DEFAULT_PARAMS["max_tokens"]),
        },
    }
    t0 = time.perf_counter()
    r = httpx.post(url, json=payload, headers=headers, timeout=600.0)
    r.raise_for_status()
    wall_ms = (time.perf_counter() - t0) * 1000.0
    data = r.json()
    text = data["message"]["content"]
    tokens = int(data.get("eval_count", 0))
    return text, tokens, float(wall_ms)


def call_backend(cfg: BackendConfig, *, system: str, user: str, params: Dict[str, Any]):
    if cfg.kind == "vllm":
        return _call_vllm(cfg, system, user, params)
    if cfg.kind == "ollama_local":
        return _call_ollama_local(cfg, system, user, params)
    if cfg.kind == "ollama_cloud":
        return _call_ollama_cloud(cfg, system, user, params)
    raise ValueError(f"Unknown backend kind: {cfg.kind}")
```

- [ ] **Step 4: Run test to confirm pass**

```bash
.venv/bin/pytest tests/test_backend_quality_audit.py -x -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/backend_quality_audit.py tests/test_backend_quality_audit.py
git commit -m "audit: backend callers — vLLM, Ollama local, Ollama Cloud"
```

---

### Task 7: Main loop + output writers

**Files:**
- Modify: `scripts/backend_quality_audit.py`
- Modify: `tests/test_backend_quality_audit.py`

- [ ] **Step 1: Write failing test for markdown writer**

Append to `tests/test_backend_quality_audit.py`:

```python
from scripts.backend_quality_audit import (
    QueryItem, BackendResponse, QueryRun,
    write_results_md, write_results_json,
)


def test_write_results_md_contains_all_columns(tmp_path):
    run = QueryRun(
        query=QueryItem(idx=1, qtype="Extraction QA", prompt="p?"),
        chunk_ids=["c1", "c2"],
        responses={
            "A_vllm_local": BackendResponse("A_vllm_local", "a-text", 10, 100.0),
            "B_ollama_local": BackendResponse("B_ollama_local", "b-text", 20, 200.0),
            "C_ollama_cloud": BackendResponse("C_ollama_cloud", "c-text", 30, 300.0),
        },
    )
    out_md = tmp_path / "results.md"
    write_results_md([run], out_md)
    body = out_md.read_text()
    assert "a-text" in body and "b-text" in body and "c-text" in body
    assert "c1" in body


def test_write_results_json_roundtrips(tmp_path):
    run = QueryRun(
        query=QueryItem(idx=1, qtype="X", prompt="Y"),
        chunk_ids=["c1"],
        responses={"A_vllm_local": BackendResponse("A_vllm_local", "t", 1, 1.0)},
    )
    out_json = tmp_path / "results.json"
    write_results_json([run], out_json)
    payload = json.loads(out_json.read_text())
    assert payload[0]["query"]["idx"] == 1
    assert payload[0]["responses"]["A_vllm_local"]["text"] == "t"
```

- [ ] **Step 2: Run to confirm fail**

```bash
.venv/bin/pytest tests/test_backend_quality_audit.py -x -q
```

Expected: FAIL — writers undefined.

- [ ] **Step 3: Implement writers and the main loop**

Add to `scripts/backend_quality_audit.py`:

```python
def write_results_json(runs: List[QueryRun], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "query": asdict(r.query),
            "chunk_ids": r.chunk_ids,
            "responses": {k: asdict(v) for k, v in r.responses.items()},
        }
        for r in runs
    ]
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_results_md(runs: List[QueryRun], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out: List[str] = []
    out.append("# Backend Quality Audit — Results\n")
    out.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n\n")
    for r in runs:
        out.append(f"## Query {r.query.idx} — {r.query.qtype}\n")
        out.append(f"**Prompt:** {r.query.prompt}\n\n")
        out.append(f"**Retrieved chunks:** `{', '.join(r.chunk_ids) or '(none)'}`\n\n")
        for label in ("A_vllm_local", "B_ollama_local", "C_ollama_cloud"):
            resp = r.responses.get(label)
            out.append(f"### {label}\n")
            if resp is None:
                out.append("_no response captured_\n\n")
                continue
            if resp.error:
                out.append(f"> ERROR: {resp.error}\n\n")
            out.append(f"_tokens={resp.tokens}  wall_ms={resp.wall_ms:.1f}_\n\n")
            out.append("```\n" + (resp.text or "") + "\n```\n\n")
        out.append("---\n\n")
    path.write_text("".join(out), encoding="utf-8")


def run_audit(
    *,
    backends: List[BackendConfig],
    queries: List[QueryItem],
    collection: str,
    profile_id: str,
    top_k: int,
    dry_run: bool,
) -> List[QueryRun]:
    if dry_run:
        return _dry_run(backends, queries)

    retriever, _ = build_retriever()
    runs: List[QueryRun] = []
    for q in queries:
        print(f"[audit] Q{q.idx} [{q.qtype}] retrieving...")
        chunks = retrieve_once(retriever, collection=collection,
                               profile_id=profile_id, query=q.prompt, top_k=top_k)
        chunk_ids = [c["id"] for c in chunks]
        prompt = build_audit_prompt(query=q.prompt, chunks=chunks)
        run = QueryRun(query=q, chunk_ids=chunk_ids)
        for cfg in backends:
            print(f"[audit]   → {cfg.label}")
            try:
                text, tokens, wall_ms = call_backend(
                    cfg, system=prompt["system"], user=prompt["user"], params=DEFAULT_PARAMS
                )
                run.responses[cfg.label] = BackendResponse(cfg.label, text, tokens, wall_ms)
            except Exception as exc:  # noqa: BLE001
                run.responses[cfg.label] = BackendResponse(cfg.label, "", 0, 0.0, error=repr(exc))
        runs.append(run)
    return runs


def _dry_run(backends: List[BackendConfig], queries: List[QueryItem]) -> List[QueryRun]:
    runs = []
    for q in queries:
        run = QueryRun(query=q, chunk_ids=[f"dry-{q.idx}-1", f"dry-{q.idx}-2"])
        for cfg in backends:
            run.responses[cfg.label] = BackendResponse(
                label=cfg.label,
                text=f"[dry-run] {cfg.label} would answer {q.prompt!r}",
                tokens=0, wall_ms=0.0,
            )
        runs.append(run)
    return runs
```

Replace `main()` body with:

```python
def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    queries = load_query_bank(Path(args.query_bank))
    backends = DEFAULT_BACKENDS

    if args.preflight_only:
        return _preflight(backends)

    if not args.dry_run:
        if not args.profile_id or not args.collection:
            print("ERROR: --profile-id and --collection are required for live run.", file=sys.stderr)
            return 2

    runs = run_audit(
        backends=backends, queries=queries,
        collection=args.collection or "", profile_id=args.profile_id or "",
        top_k=args.top_k, dry_run=args.dry_run,
    )
    out_dir = Path(args.out_dir)
    write_results_md(runs, out_dir / "results.md")
    write_results_json(runs, out_dir / "results.json")
    print(f"[audit] wrote {out_dir/'results.md'} and {out_dir/'results.json'}")
    return 0


def _preflight(backends: List[BackendConfig]) -> int:
    probe_payload = {"messages": [{"role": "user", "content": "ping"}], "max_tokens": 4}
    all_ok = True
    for cfg in backends:
        try:
            text, tokens, ms = call_backend(
                cfg, system="You respond with a single word.",
                user="say hello", params={"max_tokens": 8},
            )
            print(f"[preflight] {cfg.label}: OK ({ms:.0f}ms, {tokens} toks) → {text!r}")
        except Exception as exc:  # noqa: BLE001
            all_ok = False
            print(f"[preflight] {cfg.label}: FAIL → {exc!r}")
    return 0 if all_ok else 1
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/pytest tests/test_backend_quality_audit.py -x -q
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/backend_quality_audit.py tests/test_backend_quality_audit.py
git commit -m "audit: main loop, output writers, preflight mode"
```

---

### Task 8: Dry-run end-to-end smoke test

**Files:**
- No code changes. Execute-only.

- [ ] **Step 1: Run in dry-run mode**

```bash
cd /home/ubuntu/PycharmProjects/DocWain
.venv/bin/python scripts/backend_quality_audit.py --dry-run --out-dir /tmp/audit-dryrun
```

Expected: exit 0. Creates `/tmp/audit-dryrun/results.md` and `results.json` with 10 fake rows per backend.

- [ ] **Step 2: Inspect outputs**

```bash
head -60 /tmp/audit-dryrun/results.md
python -c "import json; d=json.load(open('/tmp/audit-dryrun/results.json')); print(len(d), 'queries,', len(d[0]['responses']), 'backends per query')"
```

Expected: `10 queries, 3 backends per query`. Markdown shows correct query/prompt/chunk_ids/backend labels.

- [ ] **Step 3: Run live preflight (just validates backends are reachable)**

```bash
.venv/bin/python scripts/backend_quality_audit.py --preflight-only
```

Expected: three OK lines (vLLM local, Ollama local, Ollama cloud). If any FAIL, fix that backend before continuing (check service status, env vars, quota).

---

### Task 9: Pick profile + collection for live run

**Files:**
- No code changes. Discovery + operator approval.

- [ ] **Step 1: Inspect Qdrant collections**

```bash
.venv/bin/python -c "
import os
from qdrant_client import QdrantClient
c = QdrantClient(url=os.environ['QDRANT_URL'], api_key=os.environ['QDRANT_API_KEY'], timeout=60)
for col in c.get_collections().collections:
    info = c.get_collection(col.name)
    print(f'{col.name:50s}  points={info.points_count}')
"
```

Expected: list of collections with point counts. Pick the collection that holds production document chunks (typically the largest non-test one).

- [ ] **Step 2: Inspect distinct profile_ids in the chosen collection**

```bash
.venv/bin/python -c "
import os, collections
from qdrant_client import QdrantClient
COLLECTION = 'REPLACE_WITH_COLLECTION_FROM_STEP_1'
c = QdrantClient(url=os.environ['QDRANT_URL'], api_key=os.environ['QDRANT_API_KEY'], timeout=60)
pts, _ = c.scroll(collection_name=COLLECTION, limit=1000, with_payload=True, with_vectors=False)
counts = collections.Counter()
for p in pts:
    pid = p.payload.get('profile_id') or p.payload.get('profile') or 'UNKNOWN'
    counts[str(pid)] += 1
for pid, n in counts.most_common(10):
    print(f'{n:6d}  {pid}')
"
```

Expected: list of profile_ids ordered by chunk count. Pick one with a representative spread of documents (invoices, quotes, contracts, resumes if available).

- [ ] **Step 3: Operator confirmation checkpoint**

**STOP HERE.** Present the chosen (`collection`, `profile_id`) pair to the operator with a sample of document types it contains. Do not proceed to the live run without explicit go-ahead.

---

### Task 10: Live run of the full audit

**Files:**
- Generates: `docs/audits/2026-04-23-backend-quality/results.md`
- Generates: `docs/audits/2026-04-23-backend-quality/results.json`

- [ ] **Step 1: Execute the live audit**

```bash
cd /home/ubuntu/PycharmProjects/DocWain
.venv/bin/python scripts/backend_quality_audit.py \
    --profile-id "PROFILE_FROM_TASK_9" \
    --collection "COLLECTION_FROM_TASK_9" \
    --top-k 10
```

Expected: runs for 15–30 minutes. Logs each query + backend. Exits 0. No `ERROR` lines in output.

- [ ] **Step 2: Validate output integrity**

```bash
.venv/bin/python -c "
import json
d = json.load(open('docs/audits/2026-04-23-backend-quality/results.json'))
assert len(d) == 10, f'expected 10 queries, got {len(d)}'
for r in d:
    assert set(r['responses']) == {'A_vllm_local', 'B_ollama_local', 'C_ollama_cloud'}
    for label, resp in r['responses'].items():
        assert resp['error'] is None, f'Q{r[\"query\"][\"idx\"]} {label}: {resp[\"error\"]}'
        assert resp['text'].strip(), f'Q{r[\"query\"][\"idx\"]} {label}: empty response'
print('OK: 10 queries x 3 backends, all non-empty, no errors.')
"
```

Expected: `OK: 10 queries x 3 backends, all non-empty, no errors.`

If any backend returns empty text or an error, note which and fix before proceeding (e.g., vLLM max_tokens too low, Ollama model not pulled, cloud quota hit).

- [ ] **Step 3: Commit generated artifacts**

```bash
git add -f docs/audits/2026-04-23-backend-quality/results.md docs/audits/2026-04-23-backend-quality/results.json
git commit -m "audit: live run results — 10 queries x 3 backends"
```

---

### Task 11: Author verdict.md

**Files:**
- Create: `docs/audits/2026-04-23-backend-quality/verdict.md`

- [ ] **Step 1: Read all 30 responses end-to-end**

```bash
less docs/audits/2026-04-23-backend-quality/results.md
```

Read every query's three responses. Take notes on strengths and weaknesses per backend.

- [ ] **Step 2: Write the verdict document**

Create `docs/audits/2026-04-23-backend-quality/verdict.md` with this structure (fill in with actual per-query analysis):

```markdown
# Backend Quality Audit — Verdict

**Date:** 2026-04-23
**Profile/Collection:** PROFILE_FROM_TASK_9 / COLLECTION_FROM_TASK_9
**Queries:** 10 (3 extraction, 2 cross-doc, 2 short-factual, 3 response-intelligence)

## Per-query verdicts

### Q1 — Extraction QA — "What is the total amount on the most recent invoice?"
- **A_vllm_local:** <one-paragraph analysis>
- **B_ollama_local:** <one-paragraph analysis>
- **C_ollama_cloud:** <one-paragraph analysis>
- **Winner:** <A|B|C>  **Why:** <one sentence>

(Repeat for Q2–Q10.)

## Per-dimension aggregate ranking

| Dimension      | 1st | 2nd | 3rd | Notes |
|----------------|-----|-----|-----|-------|
| Intelligence   |     |     |     |       |
| Grounding      |     |     |     |       |
| Completeness   |     |     |     |       |
| Style          |     |     |     |       |

## Final recommendation

<One of these, with evidence:>
- "Ship on vLLM local — matches or exceeds cloud quality, far less latency."
- "Stay on Ollama Cloud 397B — local 14B cannot match inferential depth on response-intelligence queries."
- "Swap to Ollama local — indistinguishable from cloud on grounded queries, dramatically lower cost."
- "Inconclusive — differences are within noise; escalate to latency audit on all three."
- "Local 14B can't match cloud 397B — need a larger local model or stay remote."

## Caveats & limitations

- 10 queries / 1 profile — directional evidence, not statistically robust.
- Q5_K_M vs bf16 precision asymmetry means B may have carried a quality handicap.
- Wall time recorded but not decisive — this audit was quality-only.

## Follow-on

- If recommendation is local (vLLM or Ollama): run the deferred latency audit on the winning backend.
- If recommendation is cloud: investigate network latency and prompt compression instead of engine swap.
- If inconclusive: repeat on a second profile or expand the bank to 20 queries before escalating.
```

- [ ] **Step 3: Commit verdict**

```bash
git add -f docs/audits/2026-04-23-backend-quality/verdict.md
git commit -m "audit: verdict — per-dimension ranking + final recommendation"
```

- [ ] **Step 4: Hand back to operator**

Print a one-line summary of the final recommendation. The operator decides what happens next (merge branch, open PR, discard, or pursue follow-on).

---

## Self-review — coverage check against spec

**Spec §2 (scope):** Task 10 runs 10 queries × 3 backends ✓. No src/ edits ✓. No gateway changes ✓.

**Spec §3 (held-constant invariants):** Task 7 `run_audit` retrieves once per query and reuses chunks ✓. Task 5 builds prompt once per query ✓. Same sampling params across backends (DEFAULT_PARAMS dict) ✓.

**Spec §4.2 (retrieval fairness — identical chunk IDs):** Task 10 Step 2 asserts all three backends received the same chunk set — the `run_audit` code passes the same `chunks` to every backend for a given query, so chunk_ids are recorded once at the `QueryRun` level (not per-backend), making divergence structurally impossible ✓.

**Spec §5 (query bank):** Task 1 writes the 10 queries ✓.

**Spec §6 (deliverables):** Task 1 (query_bank.md) ✓. Task 7 (results.md / results.json writers) ✓. Task 11 (verdict.md) ✓.

**Spec §7 (exit criteria):** Task 10 Step 2 asserts all 30 cells non-empty with no errors ✓. Task 11 produces per-query verdicts + aggregate ranking + recommendation ✓.

**Spec §8 (operational risks):** Task 8 Step 3 preflight covers cloud auth ✓. Task 8 verifies vLLM + Ollama coexistence ✓. Task 4 Step 3 asserts non-empty retrieval ✓. PII stays on audit branch, not pushed remote without review (covered by branch discipline in header) ✓.

**Placeholder scan:** Query bank is explicit (v0). Collection and profile_id are operator inputs to Task 9/10 — not placeholders, they are inputs the plan elicits before the run. No "TBD" / "TODO" / "add error handling" language.

**Type/name consistency:** `BackendConfig`, `QueryItem`, `BackendResponse`, `QueryRun` are referenced consistently across Tasks 2–7. `build_audit_prompt` defined Task 5, used Task 7. `call_backend` defined Task 6, used Task 7. `load_query_bank` defined Task 3, used Task 7. No drift.

Plan is spec-complete.
