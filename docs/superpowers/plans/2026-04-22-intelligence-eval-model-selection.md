# Intelligence Eval + Unified Model Selection — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up a reproducible intelligence/accuracy/latency scorecard harness; iterate on the engineering layer (prompts/retrieval) against V2 to produce a frozen prompt stack P\*; rank five model contenders under P\* to identify the unified DocWain winner; swap the serving symlink with a live canary and auto-rollback if a non-V2 winner beats V2 by ≥ 0.05 combined score.

**Architecture:** Reuses `scripts/overnight_qa/` (augmented with run-ID + idempotency) as the full-sweep harness; adds a new `scripts/intel_eval/` subsystem (fast-smoke, scorecard aggregator, phase3 runner, judge rubric); reuses `scripts/batch0/canary_smoke.py` for live canary. Model comparison runs a secondary vLLM instance on port 8101 so the live service on port 8000 is untouched until Phase 5.

**Tech Stack:** Python 3.12 (`/home/ubuntu/PycharmProjects/DocWain/.venv/bin/python`), pytest, vLLM, qdrant, Ollama local (for judge), httpx, systemd (for secondary vLLM), bash.

**Spec:** `docs/superpowers/specs/2026-04-22-intelligence-eval-model-selection-design.md`.

**Branch strategy:**
- Phases 1 + scaffolding: `intel-eval-phase1-scaffold` (off main **post-Batch-0**).
- Phase 2 iteration loop: `intel-eval-phase2` (off `intel-eval-phase1-scaffold` after it merges, or off main if we squash).
- Phase 3: `intel-eval-phase3` (off `intel-eval-phase2-P-star` tag).
- Phase 5 (only if swap): `intel-eval-phase5-swap` (off main after Phase 3 ranking PR merges).

**Hard precondition:** Batch 0 (`batch-0-unified-model-qdrant-audit`) must be merged to `main` before Task 1 starts. If Batch 0 is not in `main`, Task 1 blocks with `BLOCKED: Batch 0 not merged`.

---

## File Structure

### New files (scripts/intel_eval/)
| Path | Responsibility |
|---|---|
| `scripts/intel_eval/__init__.py` | Empty package marker |
| `scripts/intel_eval/judge_rubric.md` | Verbatim LLM-judge rubric, 5 sub-dimensions |
| `scripts/intel_eval/judge_config.yaml` | Frozen judge model config (model name, url, seed, temperature) |
| `scripts/intel_eval/score_run.py` | Reads raw overnight_qa results dir, produces scorecard JSON + markdown |
| `scripts/intel_eval/fast_smoke.py` | 45-query live smoke (3 profiles × 15 queries) + scorecard |
| `scripts/intel_eval/spin_contender.sh` | Bash helper to start/stop secondary vLLM on port 8101 |
| `scripts/intel_eval/phase3_runner.py` | Orchestrates Phase 3: cycles 5 contenders through port 8101 |

### New files (tests)
| Path | Responsibility |
|---|---|
| `tests/intel_eval/__init__.py` | Empty |
| `tests/intel_eval/conftest.py` | Shared `fixture_run_dir` fixture |
| `tests/intel_eval/test_score_run.py` | Given fixture results dir, score_run produces expected scorecard |
| `tests/intel_eval/test_fast_smoke.py` | Mocks live API; asserts combined-score formula; asserts retry / error handling |
| `tests/intel_eval/test_latency_gate.py` | Latency fixtures trigger expected gate decisions |
| `tests/intel_eval/fixtures/sample_run/` | Tiny stub results dir for score_run test |

### Modified files
| Path | Change |
|---|---|
| `scripts/overnight_qa/harness.py` | Add `--run-id`, `--target-url` flags; add run-fingerprint idempotency |
| `scripts/overnight_qa/phase5_query.py` | Accept target-url from harness; default `http://localhost:8000` |
| `scripts/overnight_qa/phase6_report.py` | Write results under `results/<run_id>/` instead of flat `results/` |

### New artifacts (gitignored dirs, force-added)
| Path | When |
|---|---|
| `eval_results/intel-baseline-V2.jsonl` | End of Phase 1 |
| `eval_results/phase2-smoke-log.ndjson` | Appended each Phase 2 iteration |
| `eval_deltas/phase2-iter-<N>.md` | One per kept iteration |
| `eval_results/phase3-ranking.md` | End of Phase 3 |
| `deploy/phase5-swap/{winner.json,swap.sh,rollback.sh,canary.txt,watch.md}` | Only if Phase 5 runs |
| `eval_results/phase5-swap-outcome.md` OR `eval_results/phase5-no-swap.md` | Close-out |

---

## Task 1: Preflight — confirm Batch 0 merged + create scaffold branch

**Files:**
- Create: `scripts/intel_eval/__init__.py` (empty)
- Create: `tests/intel_eval/__init__.py` (empty)

- [ ] **Step 1: Verify Batch 0 is in main**

Run:
```bash
cd /home/ubuntu/PycharmProjects/DocWain
git fetch origin
git log origin/main --oneline | grep -E "batch-0|unified model|qdrant audit" | head -5
```

Expected: at least one commit containing `batch-0` in the subject. If empty, **STOP** and report `BLOCKED: Batch 0 not yet merged — this workstream requires Batch 0 to be in main first. See docs/superpowers/specs/2026-04-21-intelligence-rag-redesign-design.md.`

- [ ] **Step 2: Pull latest main**

```bash
git checkout main
git pull --ff-only origin main
```

Expected: fast-forward or already-up-to-date. If non-fast-forward: report BLOCKED (diverged history means something unexpected happened on main).

- [ ] **Step 3: Create the Phase 1 branch**

```bash
git checkout -b intel-eval-phase1-scaffold
```

- [ ] **Step 4: Create empty package markers**

```bash
mkdir -p scripts/intel_eval tests/intel_eval
touch scripts/intel_eval/__init__.py tests/intel_eval/__init__.py
```

- [ ] **Step 5: Commit the scaffold**

```bash
git add scripts/intel_eval/__init__.py tests/intel_eval/__init__.py
git commit -m "intel-eval: scaffold package directories"
```

---

## Task 2: Judge config + rubric

**Files:**
- Create: `scripts/intel_eval/judge_rubric.md`
- Create: `scripts/intel_eval/judge_config.yaml`

- [ ] **Step 1: Write the rubric**

Write `scripts/intel_eval/judge_rubric.md` with EXACTLY this content:

```markdown
# DocWain Response Quality Rubric

You are an evaluator scoring a DocWain response. You will be given:
- The user's query.
- The DocWain response.
- The evidence sources that were retrieved for that response (may be empty).

Rate the response on a 0.0-1.0 scale (0.1 granularity) for EACH of the five sub-dimensions below. Output JSON only. Do not explain.

## Sub-dimensions

1. **Groundedness** — Is every factual claim in the response traceable to at least one of the provided sources? Unsupported claims lower the score. 1.0 = all claims supported; 0.0 = response is largely ungrounded / hallucinated.
2. **Relevance** — Does the response directly answer the user's query? Off-topic content lowers the score. 1.0 = tightly on-topic; 0.0 = answers a different question.
3. **Insight** — Does the response synthesise or connect information beyond literal retrieval? Direct quotes only → ~0.5. Cross-document connections, implications, useful patterns → ~0.8-1.0. Generic platitudes → ~0.3.
4. **Structure** — Does the format match the intent (table for tabular data, list for enumerations, prose for narrative) and is it clean? Messy / wrong shape → low.
5. **Persona** — Is the voice consistent with a professional document intelligence assistant named DocWain? Inconsistent, chatty, or missing persona → low.

## Output format

```
{"groundedness": 0.X, "relevance": 0.X, "insight": 0.X, "structure": 0.X, "persona": 0.X}
```

Nothing else. No prose. No markdown fences around the JSON.
```

- [ ] **Step 2: Write the judge config**

Write `scripts/intel_eval/judge_config.yaml` with EXACTLY this content:

```yaml
# LLM judge configuration — frozen for the intel-eval workstream.
# Any change to these values INVALIDATES prior scorecards.

judge:
  # Default judge. Override with DOCWAIN_INTEL_JUDGE env var.
  # Using local Ollama qwen3:14b to avoid DocWain judging DocWain (bias).
  backend: ollama          # ollama | vllm
  model: qwen3:14b
  url: http://localhost:11434
  temperature: 0.0
  seed: 42
  max_tokens: 512

rubric_path: scripts/intel_eval/judge_rubric.md

# Per-query timeout. Longer than generation because judge sees the whole
# answer + sources.
timeout_seconds: 120
```

- [ ] **Step 3: Sanity-check Ollama has the judge model**

Run:
```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/python -c "
import urllib.request, json
r = urllib.request.urlopen('http://localhost:11434/api/tags', timeout=3)
tags = json.loads(r.read())['models']
names = [m['name'] for m in tags]
print('available ollama models:', names)
assert any(n.startswith('qwen3:14b') for n in names), \
    'qwen3:14b not installed — run: ollama pull qwen3:14b'
print('ok')
"
```

Expected: `ok`. If the model is not installed, STOP and report the exact `ollama pull` command to the user. Do NOT silently fall back.

- [ ] **Step 4: Commit**

```bash
git add scripts/intel_eval/judge_rubric.md scripts/intel_eval/judge_config.yaml
git commit -m "intel-eval: freeze judge model config + rubric"
```

---

## Task 3: score_run.py — TDD (write failing test)

**Files:**
- Create: `tests/intel_eval/conftest.py`
- Create: `tests/intel_eval/fixtures/sample_run/results.jsonl`
- Create: `tests/intel_eval/test_score_run.py`

- [ ] **Step 1: Write the conftest**

Write `tests/intel_eval/conftest.py`:

```python
"""Shared fixtures for intel_eval tests."""
from __future__ import annotations

import pathlib

import pytest

_FIX = pathlib.Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def fixtures_dir() -> pathlib.Path:
    return _FIX


@pytest.fixture()
def sample_run_dir(fixtures_dir: pathlib.Path) -> pathlib.Path:
    return fixtures_dir / "sample_run"
```

- [ ] **Step 2: Write the sample-run fixture**

```bash
mkdir -p tests/intel_eval/fixtures/sample_run
```

Write `tests/intel_eval/fixtures/sample_run/results.jsonl`:

```
{"intent":"lookup","query":"total?","response":"The total is $1,641.75","sources":[{"file_name":"inv1.pdf","page":1,"snippet":"Total: $1,641.75"}],"grounded":true,"context_found":true,"http_status":200,"latency_s":4.2,"expected_fields":{"total":"$1,641.75"},"sanity":"ok","judge":{"groundedness":1.0,"relevance":1.0,"insight":0.6,"structure":0.9,"persona":0.9}}
{"intent":"count","query":"how many?","response":"You have 12 documents","sources":[],"grounded":false,"context_found":true,"http_status":200,"latency_s":3.1,"expected_fields":{"count":"12"},"sanity":"ok","judge":{"groundedness":0.8,"relevance":1.0,"insight":0.4,"structure":0.9,"persona":0.8}}
{"intent":"analyze","query":"trends?","response":"","sources":[],"grounded":false,"context_found":false,"http_status":500,"latency_s":32.5,"expected_fields":{},"sanity":"invalid","judge":null}
```

- [ ] **Step 3: Write the failing test**

Write `tests/intel_eval/test_score_run.py`:

```python
"""Score-run tests — given a results.jsonl fixture, score_run produces
the expected scorecard."""
from __future__ import annotations

import json
import pathlib


def test_scorecard_excludes_invalid_rows(sample_run_dir: pathlib.Path, tmp_path: pathlib.Path):
    from scripts.intel_eval.score_run import score_run

    out = tmp_path / "scorecard.json"
    score_run(sample_run_dir, out)

    data = json.loads(out.read_text())
    assert data["invalid_count"] == 1
    assert data["valid_count"] == 2
    # invalid row must not appear in the per-intent breakdown
    assert "analyze" not in data["per_intent_accuracy"]


def test_scorecard_computes_accuracy_from_expected_fields(sample_run_dir, tmp_path):
    from scripts.intel_eval.score_run import score_run
    out = tmp_path / "scorecard.json"
    score_run(sample_run_dir, out)
    data = json.loads(out.read_text())
    # lookup row: expected total $1,641.75 appears in response → 1.0
    # count row: expected count "12" appears in response → 1.0
    # valid mean should be 1.0
    assert data["accuracy"] == 1.0


def test_scorecard_averages_judge_subdims_into_intelligence(sample_run_dir, tmp_path):
    from scripts.intel_eval.score_run import score_run
    out = tmp_path / "scorecard.json"
    score_run(sample_run_dir, out)
    data = json.loads(out.read_text())
    # lookup: mean(1.0, 1.0, 0.6, 0.9, 0.9) = 0.88
    # count:  mean(0.8, 1.0, 0.4, 0.9, 0.8) = 0.78
    # overall intelligence = mean(0.88, 0.78) = 0.83
    assert abs(data["intelligence"] - 0.83) < 1e-9


def test_scorecard_combined_is_half_each(sample_run_dir, tmp_path):
    from scripts.intel_eval.score_run import score_run
    out = tmp_path / "scorecard.json"
    score_run(sample_run_dir, out)
    data = json.loads(out.read_text())
    assert abs(data["combined"] - (0.5 * data["accuracy"] + 0.5 * data["intelligence"])) < 1e-9


def test_latency_p95_per_intent_band(sample_run_dir, tmp_path):
    from scripts.intel_eval.score_run import score_run
    out = tmp_path / "scorecard.json"
    score_run(sample_run_dir, out)
    data = json.loads(out.read_text())
    # Fixture has one lookup (4.2s) and one count (3.1s) — both in the
    # 'simple' band with a 10s gate. Should PASS the simple band.
    assert data["latency_gates"]["simple"]["p95_s"] <= 10.0
    assert data["latency_gates"]["simple"]["pass"] is True
```

- [ ] **Step 4: Run the tests — must all FAIL**

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/python -m pytest tests/intel_eval/test_score_run.py -v
```

Expected: 5 tests FAIL with `ModuleNotFoundError: No module named 'scripts.intel_eval.score_run'` or similar.

- [ ] **Step 5: Commit failing tests**

```bash
git add tests/intel_eval/conftest.py tests/intel_eval/fixtures tests/intel_eval/test_score_run.py
git commit -m "intel-eval: failing scorecard tests (TDD red)"
```

---

## Task 4: score_run.py — implement

**Files:**
- Create: `scripts/intel_eval/score_run.py`

- [ ] **Step 1: Implement the scorer**

Write `scripts/intel_eval/score_run.py`:

```python
"""Score a raw overnight_qa run — reads results.jsonl, emits a scorecard.

Scorecard structure (JSON):

    {
      "invalid_count": int,
      "valid_count": int,
      "accuracy": float,            # 0-1, micro-avg across valid rows
      "intelligence": float,        # 0-1, micro-avg of per-row judge mean
      "combined": float,            # 0.5 * accuracy + 0.5 * intelligence
      "per_intent_accuracy": {intent: float},
      "per_intent_intelligence": {intent: float},
      "latency_gates": {
        "simple":   {"p95_s": float, "pass": bool, "threshold_s": 10},
        "moderate": {"p95_s": float, "pass": bool, "threshold_s": 20},
        "complex":  {"p95_s": float, "pass": bool, "threshold_s": 30}
      },
      "overall_pass": bool,         # all latency bands pass AND invalid < 10%
      "run_id": str,                # from input dir name
    }

Input `results.jsonl` row shape:

    {
      "intent": "lookup",
      "query": "...",
      "response": "...",
      "sources": [...],
      "grounded": bool,
      "context_found": bool,
      "http_status": int,
      "latency_s": float,
      "expected_fields": {fieldname: expected_value} | {},
      "sanity": "ok" | "invalid",
      "judge": {groundedness, relevance, insight, structure, persona} | null
    }

Field-match accuracy: for each expected field, score 1 if the expected
value appears as a substring in the response text (case-insensitive),
else 0. Per-row accuracy = mean across asked fields. Row with empty
expected_fields gets accuracy=None and is excluded from the accuracy
mean (but counted for intelligence).
"""
from __future__ import annotations

import json
import pathlib
import statistics
from typing import Iterable

_SIMPLE = {"greeting", "identity", "greet", "meta", "help", "capability", "goodbye",
           "lookup", "list", "count"}
_MODERATE = {"extract", "summarize", "timeline"}
_COMPLEX = {"compare", "analyze", "investigate"}

_BANDS = {
    "simple": (_SIMPLE, 10.0),
    "moderate": (_MODERATE, 20.0),
    "complex": (_COMPLEX, 30.0),
}

_JUDGE_KEYS = ("groundedness", "relevance", "insight", "structure", "persona")


def _field_match(expected: dict, response: str) -> float | None:
    if not expected:
        return None
    if not response:
        return 0.0
    response_l = response.lower()
    hits = sum(1 for v in expected.values() if str(v).lower() in response_l)
    return hits / len(expected)


def _band_for(intent: str) -> str | None:
    for band, (members, _) in _BANDS.items():
        if intent in members:
            return band
    return None


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    # closest-rank p95
    sorted_values = sorted(values)
    idx = int(round(0.95 * (len(sorted_values) - 1)))
    return sorted_values[idx]


def score_run(run_dir: pathlib.Path, out_path: pathlib.Path) -> dict:
    run_dir = pathlib.Path(run_dir)
    rows: list[dict] = []
    for line in (run_dir / "results.jsonl").read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))

    invalid = [r for r in rows if r.get("sanity") != "ok"]
    valid = [r for r in rows if r.get("sanity") == "ok"]

    per_intent_acc: dict[str, list[float]] = {}
    per_intent_int: dict[str, list[float]] = {}
    acc_samples: list[float] = []
    int_samples: list[float] = []

    for r in valid:
        intent = r.get("intent", "unknown")
        acc = _field_match(r.get("expected_fields") or {}, r.get("response") or "")
        if acc is not None:
            per_intent_acc.setdefault(intent, []).append(acc)
            acc_samples.append(acc)
        judge = r.get("judge")
        if isinstance(judge, dict):
            sub = [float(judge.get(k, 0.0)) for k in _JUDGE_KEYS]
            row_int = sum(sub) / len(sub)
            per_intent_int.setdefault(intent, []).append(row_int)
            int_samples.append(row_int)

    accuracy = statistics.fmean(acc_samples) if acc_samples else 0.0
    intelligence = statistics.fmean(int_samples) if int_samples else 0.0
    combined = 0.5 * accuracy + 0.5 * intelligence

    # Latency gates (only compute on valid rows)
    band_samples: dict[str, list[float]] = {b: [] for b in _BANDS}
    for r in valid:
        band = _band_for(r.get("intent", ""))
        if band is None:
            continue
        band_samples[band].append(float(r.get("latency_s", 0.0)))
    gates: dict[str, dict] = {}
    for band, (_, thresh) in _BANDS.items():
        p95 = _p95(band_samples[band])
        gates[band] = {"p95_s": p95, "pass": p95 <= thresh, "threshold_s": thresh}

    invalid_ratio = len(invalid) / max(len(rows), 1)
    overall_pass = all(g["pass"] for g in gates.values()) and invalid_ratio < 0.10

    result = {
        "invalid_count": len(invalid),
        "valid_count": len(valid),
        "accuracy": round(accuracy, 4),
        "intelligence": round(intelligence, 4),
        "combined": round(combined, 4),
        "per_intent_accuracy": {k: round(statistics.fmean(v), 4) for k, v in per_intent_acc.items()},
        "per_intent_intelligence": {k: round(statistics.fmean(v), 4) for k, v in per_intent_int.items()},
        "latency_gates": gates,
        "overall_pass": overall_pass,
        "run_id": run_dir.name,
    }
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    result = score_run(args.run_dir, args.out)
    print(json.dumps(result, indent=2, sort_keys=True))
```

- [ ] **Step 2: Run the tests — all should PASS**

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/python -m pytest tests/intel_eval/test_score_run.py -v
```

Expected: 5 passed.

- [ ] **Step 3: Commit**

```bash
git add scripts/intel_eval/score_run.py
git commit -m "intel-eval: score_run.py — scorecard aggregator"
```

---

## Task 5: Latency gate tests + edge cases

**Files:**
- Create: `tests/intel_eval/test_latency_gate.py`

- [ ] **Step 1: Write the test**

```python
"""Hand-crafted latency fixtures to exercise gate decisions."""
from __future__ import annotations

import json
import pathlib


def _write_fixture(tmp_path: pathlib.Path, rows: list[dict]) -> pathlib.Path:
    d = tmp_path / "run-xyz"
    d.mkdir()
    (d / "results.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n"
    )
    return d


def _row(intent: str, latency_s: float, sanity: str = "ok") -> dict:
    return {
        "intent": intent, "query": "q", "response": "r",
        "sources": [], "grounded": True, "context_found": True,
        "http_status": 200, "latency_s": latency_s,
        "expected_fields": {}, "sanity": sanity,
        "judge": {"groundedness": 1.0, "relevance": 1.0, "insight": 1.0,
                  "structure": 1.0, "persona": 1.0},
    }


def test_simple_band_passes_under_10s(tmp_path):
    from scripts.intel_eval.score_run import score_run
    d = _write_fixture(tmp_path, [_row("lookup", 9.5), _row("lookup", 9.9)])
    out = tmp_path / "s.json"
    data = score_run(d, out)
    assert data["latency_gates"]["simple"]["pass"] is True


def test_simple_band_fails_at_11s_p95(tmp_path):
    from scripts.intel_eval.score_run import score_run
    rows = [_row("lookup", 3.0)] * 19 + [_row("lookup", 11.0)]
    d = _write_fixture(tmp_path, rows)
    out = tmp_path / "s.json"
    data = score_run(d, out)
    assert data["latency_gates"]["simple"]["pass"] is False


def test_complex_band_30s_boundary_passes(tmp_path):
    from scripts.intel_eval.score_run import score_run
    d = _write_fixture(tmp_path, [_row("analyze", 29.9)])
    out = tmp_path / "s.json"
    data = score_run(d, out)
    assert data["latency_gates"]["complex"]["pass"] is True


def test_overall_pass_requires_all_bands(tmp_path):
    from scripts.intel_eval.score_run import score_run
    # simple band passes but complex exceeds 30s
    rows = [_row("lookup", 3.0), _row("analyze", 40.0)]
    d = _write_fixture(tmp_path, rows)
    out = tmp_path / "s.json"
    data = score_run(d, out)
    assert data["overall_pass"] is False


def test_overall_pass_requires_invalid_under_10pct(tmp_path):
    from scripts.intel_eval.score_run import score_run
    rows = [_row("lookup", 3.0)] * 9 + [_row("lookup", 3.0, sanity="invalid")] * 2
    d = _write_fixture(tmp_path, rows)
    out = tmp_path / "s.json"
    data = score_run(d, out)
    # 2 / 11 = 18% invalid — exceeds 10% threshold
    assert data["overall_pass"] is False
```

- [ ] **Step 2: Run — must pass**

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/python -m pytest tests/intel_eval/test_latency_gate.py -v
```

Expected: 5 passed.

- [ ] **Step 3: Commit**

```bash
git add tests/intel_eval/test_latency_gate.py
git commit -m "intel-eval: latency-gate edge-case tests"
```

---

## Task 6: Augment overnight_qa with run-ID + target URL

**Files:**
- Modify: `scripts/overnight_qa/harness.py`
- Modify: `scripts/overnight_qa/config.py`
- Modify: `scripts/overnight_qa/phase5_query.py`

- [ ] **Step 1: Read current harness + config**

```bash
cat scripts/overnight_qa/config.py
```

Note the `API_BASE`, `SUBSCRIPTION_ID`, `RESULTS_DIR` constants.

- [ ] **Step 2: Add run-ID support to config.py**

Replace the full contents of `scripts/overnight_qa/config.py` with:

```python
import os

API_BASE = os.getenv("DOCWAIN_QA_API_BASE", "http://localhost:8000")
SUBSCRIPTION_ID = "67fde0754e36c00b14cea7f5"
TEST_USER = "qa-tester@dhsit.co.uk"

PROFILES = {
    "invoices": {"name": "QA - Invoices", "description": "Invoice documents for QA testing"},
    "contracts": {"name": "QA - Contracts", "description": "Contract documents for QA testing"},
    "purchase_orders": {"name": "QA - Purchase Orders", "description": "Purchase order documents for QA testing"},
    "resumes": {"name": "QA - Resumes", "description": "Resume/CV documents for QA testing"},
    "finance_statements": {"name": "QA - Q1 Finance", "description": "Q1 financial statements for QA testing"},
    "expense_reports": {"name": "QA - Monthly Expenses", "description": "Monthly expense reports for QA testing"},
}

DOCS_PER_CATEGORY = 15
QUERIES_PER_PROFILE = 20

# Run ID is stamped by the harness at the start of a run and used for
# the results-dir sub-path. Intel-eval runs set it via --run-id.
RESULTS_DIR_BASE = "scripts/overnight_qa/results"
GENERATED_DOCS_DIR = "scripts/overnight_qa/generated_docs"


def results_dir_for_run(run_id: str) -> str:
    return f"{RESULTS_DIR_BASE}/{run_id}"
```

Also replace the `RESULTS_DIR` import site inside `scripts/overnight_qa/phase5_query.py` and `scripts/overnight_qa/phase6_report.py` with calls to `results_dir_for_run(run_id)`. Search both files for `RESULTS_DIR` and convert to pass `run_id` through.

If the search finds `from scripts.overnight_qa.config import RESULTS_DIR` anywhere, replace it with a function param `run_id` threaded down from `harness.py`.

- [ ] **Step 3: Add --run-id, --target-url CLI flags to harness.py**

Open `scripts/overnight_qa/harness.py`. If it has a `main()` with `argparse`, add:
```python
parser.add_argument("--run-id", default=None,
                    help="Unique run identifier; used for results/<run_id>/ sub-path")
parser.add_argument("--target-url", default=None,
                    help="Override API_BASE. Example: http://localhost:8001 for contender vLLM.")
```

If the file is a library (imported by a separate CLI entry), expose `run_id` and `target_url` as parameters on the top-level `run(...)` function. Thread them through to `api_call(...)` and the results-writer.

Default `run_id`:
```python
from datetime import datetime, timezone
run_id = run_id or datetime.now(timezone.utc).strftime("run-%Y%m%d-%H%M%S")
```

- [ ] **Step 4: Idempotency key — skip re-ingestion if fingerprint matches**

Add a fingerprint file at `scripts/overnight_qa/results/<run_id>/.fingerprint`:

```python
def write_fingerprint(run_id: str, config_sha: str):
    path = pathlib.Path(results_dir_for_run(run_id)) / ".fingerprint"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "run_id": run_id,
        "config_sha": config_sha,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }))

def has_valid_fingerprint(run_id: str, config_sha: str) -> bool:
    path = pathlib.Path(results_dir_for_run(run_id)) / ".fingerprint"
    if not path.exists():
        return False
    data = json.loads(path.read_text())
    return data.get("config_sha") == config_sha
```

`config_sha` is the SHA-256 of the serialized PROFILES + DOCS_PER_CATEGORY + QUERIES_PER_PROFILE + generated_docs dir listing — i.e., anything that would change the upload/query shape. When fingerprint matches, skip `phase1_cleanup`/`phase2_generate_docs`/`phase3_upload`/`phase4_pipeline` and go straight to `phase5_query` with the existing profiles.

- [ ] **Step 5: Smoke-test the augmented harness**

Run:
```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/python -m scripts.overnight_qa.harness --help
```

Expected: argparse help showing `--run-id` and `--target-url`.

- [ ] **Step 6: Commit**

```bash
git add scripts/overnight_qa/
git commit -m "intel-eval: augment overnight_qa with run-id + target-url + idempotency"
```

---

## Task 7: fast_smoke.py — TDD (failing test)

**Files:**
- Create: `tests/intel_eval/test_fast_smoke.py`

- [ ] **Step 1: Write the test**

```python
"""fast_smoke tests — mocks the /api/ask endpoint + judge, asserts
that fast_smoke.run() produces a well-formed scorecard."""
from __future__ import annotations

import json
import pathlib

import pytest


class _FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, answers):
        self.answers = iter(answers)
        self.calls = []

    def post(self, url, json=None, timeout=None):
        self.calls.append({"url": url, "body": json})
        return _FakeResponse(next(self.answers))

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_fast_smoke_all_pass_produces_scorecard(monkeypatch, tmp_path):
    """Three canned queries, all returning grounded non-empty responses.
    Smoke should produce a scorecard with valid_count=3 and pass=True.
    """
    from scripts.intel_eval import fast_smoke

    answers = [
        {"answer": {"response": "INV-001 total is $100. DocWain here.",
                    "sources": [{"file_name": "a.pdf", "page": 1, "snippet": "$100"}],
                    "grounded": True, "context_found": True}},
    ] * 3

    monkeypatch.setattr(fast_smoke, "httpx", _FakeHttpx(answers))
    monkeypatch.setattr(fast_smoke, "score_with_judge",
                        lambda q, r, s: {"groundedness": 1.0, "relevance": 1.0,
                                         "insight": 0.8, "structure": 0.9, "persona": 0.9})

    queries = [("lookup", "total?", {"total": "$100"}) for _ in range(3)]
    out = tmp_path / "smoke.json"
    result = fast_smoke.run(
        base_url="http://localhost:8000", auth=None,
        profile_id="p", subscription_id="s",
        queries=queries, out_path=out,
    )
    scorecard = json.loads(out.read_text())
    assert scorecard["valid_count"] == 3
    assert scorecard["invalid_count"] == 0
    assert scorecard["overall_pass"] is True
    assert scorecard["combined"] > 0


class _FakeHttpx:
    def __init__(self, answers):
        self._answers = list(answers)

    def Client(self, *a, **kw):
        return _FakeClient(self._answers)
```

- [ ] **Step 2: Run — must fail**

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/python -m pytest tests/intel_eval/test_fast_smoke.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Commit**

```bash
git add tests/intel_eval/test_fast_smoke.py
git commit -m "intel-eval: failing fast_smoke test (TDD red)"
```

---

## Task 8: fast_smoke.py — implement

**Files:**
- Create: `scripts/intel_eval/fast_smoke.py`

- [ ] **Step 1: Implement**

Write `scripts/intel_eval/fast_smoke.py`:

```python
"""Fast smoke — 45 live HTTP queries against a running DocWain API.

Used during Phase 2 iteration (after each accepted tweak) to get a
5-minute pass/fail scorecard. Uses scripts.intel_eval.score_run to
aggregate results in the same structure as the full sweep — so iteration
decisions are made on comparable numbers.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import time
from typing import Callable, Sequence

import httpx

from scripts.intel_eval.score_run import score_run


_DEFAULT_QUERIES: list[tuple[str, str, dict]] = [
    # (intent, query, expected_fields)
    # Invoices profile (5 queries)
    ("lookup",    "What is the invoice total on the most recent invoice?", {}),
    ("list",      "List all invoices sorted by date.",                      {}),
    ("count",     "How many invoices are in this profile?",                 {}),
    ("extract",   "Extract line items from the most recent invoice.",       {}),
    ("compare",   "Compare the two highest-value invoices.",                {}),
    # Contracts profile
    ("lookup",    "What is the start date of the most recent contract?",   {}),
    ("list",      "List all contracts by party name.",                      {}),
    ("count",     "How many contracts do I have?",                          {}),
    ("summarize", "Summarize the most recent contract.",                    {}),
    ("analyze",   "What common clauses appear across these contracts?",    {}),
    # Resumes profile
    ("lookup",    "Who is the candidate with the most years of Python experience?", {}),
    ("list",      "List all candidates and their years of experience.",    {}),
    ("count",     "How many resumes are in this profile?",                  {}),
    ("extract",   "Extract the skills section from the top-rated resume.", {}),
    ("compare",   "Compare the two most senior candidates.",                {}),
]


def _sanity(response: str, intent: str, grounded: bool, ctx: bool, status: int) -> str:
    if status != 200:
        return "invalid"
    if not response:
        return "invalid"
    if "I'm having trouble" in response:
        return "invalid"
    if intent in ("greet", "greeting", "identity") and "DocWain" not in response:
        return "invalid"
    return "ok"


def _ask(client, base_url, auth, profile_id, sub_id, query):
    body = {
        "query": query,
        "profile_id": profile_id,
        "subscription_id": sub_id,
        "new_session": True,
    }
    headers = {"Authorization": f"Bearer {auth}"} if auth else {}
    t0 = time.perf_counter()
    try:
        r = client.post(f"{base_url}/api/ask", json=body, headers=headers, timeout=180)
        latency = time.perf_counter() - t0
        payload = r.json()
        ans = payload.get("answer") if isinstance(payload, dict) else {}
        return {
            "status": r.status_code,
            "response": (ans or {}).get("response", ""),
            "sources": (ans or {}).get("sources", []),
            "grounded": (ans or {}).get("grounded", False),
            "context_found": (ans or {}).get("context_found", False),
            "latency_s": latency,
        }
    except Exception as exc:
        return {
            "status": 0, "response": "", "sources": [],
            "grounded": False, "context_found": False,
            "latency_s": time.perf_counter() - t0,
            "error": str(exc),
        }


def score_with_judge(query: str, response: str, sources: list) -> dict:
    """Stub — replaced by the real judge invocation in Task 9."""
    return {"groundedness": 0.0, "relevance": 0.0, "insight": 0.0, "structure": 0.0, "persona": 0.0}


def run(
    base_url: str,
    auth: str | None,
    profile_id: str,
    subscription_id: str,
    queries: Sequence[tuple[str, str, dict]],
    out_path: pathlib.Path,
    judge: Callable | None = None,
) -> dict:
    judge_fn = judge or score_with_judge
    rows: list[dict] = []
    with httpx.Client(timeout=300.0) as client:
        for intent, query, expected in queries:
            r = _ask(client, base_url, auth, profile_id, subscription_id, query)
            sanity = _sanity(r["response"], intent, r["grounded"], r["context_found"], r["status"])
            judge_scores = None
            if sanity == "ok":
                try:
                    judge_scores = judge_fn(query, r["response"], r["sources"])
                except Exception:
                    judge_scores = None
            rows.append({
                "intent": intent, "query": query,
                "response": r["response"], "sources": r["sources"],
                "grounded": r["grounded"], "context_found": r["context_found"],
                "http_status": r["status"], "latency_s": r["latency_s"],
                "expected_fields": expected, "sanity": sanity,
                "judge": judge_scores,
            })
    run_dir = out_path.parent / "__smoke_tmp"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    scorecard = score_run(run_dir, out_path)
    return scorecard


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=os.getenv("DOCWAIN_SMOKE_BASE_URL", "http://localhost:8000"))
    ap.add_argument("--auth", default=os.getenv("DOCWAIN_SMOKE_AUTH"))
    ap.add_argument("--profile", default=os.getenv("DOCWAIN_SMOKE_PROFILE"))
    ap.add_argument("--sub", default=os.getenv("DOCWAIN_SMOKE_SUB"))
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    if not (args.profile and args.sub):
        print("error: --profile and --sub required", file=__import__("sys").stderr)
        raise SystemExit(2)
    run(args.base_url, args.auth, args.profile, args.sub, _DEFAULT_QUERIES, args.out)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run tests**

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/python -m pytest tests/intel_eval/test_fast_smoke.py -v
```

Expected: 1 passed.

- [ ] **Step 3: Commit**

```bash
git add scripts/intel_eval/fast_smoke.py
git commit -m "intel-eval: fast_smoke.py — 15-query live smoke"
```

---

## Task 9: Wire real LLM-judge into fast_smoke

**Files:**
- Modify: `scripts/intel_eval/fast_smoke.py`

- [ ] **Step 1: Replace `score_with_judge` stub with a real Ollama call**

In `scripts/intel_eval/fast_smoke.py`, replace the `score_with_judge` function (currently returns zeros) with:

```python
def score_with_judge(query: str, response: str, sources: list) -> dict:
    """Call the judge model per judge_config.yaml. Returns a 5-sub-dim dict."""
    import yaml, json as _json
    config = yaml.safe_load(
        pathlib.Path(__file__).parent.joinpath("judge_config.yaml").read_text()
    )
    rubric = pathlib.Path(__file__).parent.joinpath(config["rubric_path"]).read_text()
    # Trim sources to first 6 for context budget
    srcs = sources[:6]
    user_msg = (
        f"Query:\n{query}\n\n"
        f"Response:\n{response}\n\n"
        f"Sources:\n{_json.dumps(srcs, indent=2)}\n"
    )
    judge = config["judge"]
    req = {
        "model": judge["model"],
        "prompt": f"SYSTEM: {rubric}\n\nUSER: {user_msg}",
        "stream": False,
        "options": {
            "temperature": judge["temperature"],
            "seed": judge["seed"],
            "num_predict": judge["max_tokens"],
        },
    }
    import urllib.request
    data = _json.dumps(req).encode()
    r = urllib.request.Request(
        f"{judge['url']}/api/generate",
        data=data, method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(r, timeout=config["timeout_seconds"]) as resp:
        out = _json.loads(resp.read().decode())
    text = (out.get("response") or "").strip()
    # Extract the JSON blob — the judge should output just JSON but may
    # wrap it in prose if prompted poorly. Be defensive.
    try:
        scores = _json.loads(text)
    except _json.JSONDecodeError:
        import re
        m = re.search(r"\{[^{}]+\}", text)
        if not m:
            raise RuntimeError(f"Judge returned unparseable response: {text[:200]!r}")
        scores = _json.loads(m.group(0))
    return {
        "groundedness": float(scores.get("groundedness", 0)),
        "relevance": float(scores.get("relevance", 0)),
        "insight": float(scores.get("insight", 0)),
        "structure": float(scores.get("structure", 0)),
        "persona": float(scores.get("persona", 0)),
    }
```

Add `import pathlib` at the top of the file if not present.

- [ ] **Step 2: Re-run tests (must still pass — the test mocks the judge)**

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/python -m pytest tests/intel_eval/ -v
```

Expected: all tests pass (test uses `monkeypatch.setattr(fast_smoke, "score_with_judge", …)`).

- [ ] **Step 3: Manual sanity test against the live Ollama judge**

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/python -c "
from scripts.intel_eval.fast_smoke import score_with_judge
s = score_with_judge(
    query='What is the invoice total?',
    response='The invoice total is \$1,641.75. This is based on SOURCE-1.',
    sources=[{'file_name':'inv.pdf','page':1,'snippet':'Total: \$1,641.75'}],
)
print(s)
assert set(s.keys()) == {'groundedness','relevance','insight','structure','persona'}
assert all(0.0 <= v <= 1.0 for v in s.values())
print('ok')
"
```

Expected: `ok` plus a dict of five floats in [0,1].

- [ ] **Step 4: Commit**

```bash
git add scripts/intel_eval/fast_smoke.py
git commit -m "intel-eval: wire real Ollama judge into fast_smoke"
```

---

## Task 10: Phase 1 baseline — run against V2

**Files:** no code changes. Produces `eval_results/intel-baseline-V2.jsonl`.

- [ ] **Step 1: Verify Batch 0 is in main**

```bash
git log main --oneline | grep "batch-0" | head
```

Expected: at least one `batch-0` commit. If missing, STOP (re-check Task 1).

- [ ] **Step 2: Sanity-check the current serving layer**

```bash
curl -sf http://localhost:8000/api/health 2>&1 | head -3
curl -sf http://localhost:8100/health 2>&1 | head -3
cat /tmp/docwain-gpu-mode.json
```

Expected: both health endpoints return 200, gpu-mode = serving. If not, STOP and ask the owner to bring the service up.

- [ ] **Step 3: Run full overnight_qa sweep**

```bash
mkdir -p eval_results
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/python -m scripts.overnight_qa.harness \
    --run-id baseline-V2-2026-04-22 \
    --target-url http://localhost:8000 \
    2>&1 | tee eval_results/phase1-full-sweep.log
```

This runs ~90 ingestions + 120 queries. Expected duration ~100 min. If it exceeds 3 hours, STOP and investigate — pipeline or vLLM is likely stuck.

- [ ] **Step 4: Score the run**

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/python -m scripts.intel_eval.score_run \
    scripts/overnight_qa/results/baseline-V2-2026-04-22 \
    --out eval_results/intel-baseline-V2-scorecard.json
cat eval_results/intel-baseline-V2-scorecard.json | /home/ubuntu/PycharmProjects/DocWain/.venv/bin/python -m json.tool
```

Expected: a structured scorecard with non-zero `valid_count`, `accuracy`, `intelligence`, `combined`, and three latency gates.

- [ ] **Step 5: Run fast-smoke twice (reproducibility gate)**

```bash
DOCWAIN_SMOKE_PROFILE=69e78d09af9231725f583b3d \
DOCWAIN_SMOKE_SUB=67fde0754e36c00b14cea7f5 \
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/python -m scripts.intel_eval.fast_smoke \
    --out eval_results/phase1-smoke-run1.json

DOCWAIN_SMOKE_PROFILE=69e78d09af9231725f583b3d \
DOCWAIN_SMOKE_SUB=67fde0754e36c00b14cea7f5 \
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/python -m scripts.intel_eval.fast_smoke \
    --out eval_results/phase1-smoke-run2.json
```

Diff the two combined scores:
```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/python -c "
import json
a = json.load(open('eval_results/phase1-smoke-run1.json'))['combined']
b = json.load(open('eval_results/phase1-smoke-run2.json'))['combined']
print(f'run1={a:.4f}  run2={b:.4f}  delta={abs(a-b):.4f}')
assert abs(a - b) < 0.03, f'Smoke non-reproducible: delta={abs(a-b)}'
print('reproducibility gate PASS')
"
```

Expected: `reproducibility gate PASS`. If fail: STOP and investigate judge non-determinism (judge_config seed may not be honoured by Ollama backend).

- [ ] **Step 6: Run live canary on Procurement profile**

```bash
DOCWAIN_SMOKE_PROFILE=69e78d09af9231725f583b3d \
DOCWAIN_SMOKE_SUB=67fde0754e36c00b14cea7f5 \
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/python scripts/batch0/canary_smoke.py \
    --base-url http://localhost:8000 \
    | tee eval_results/phase1-canary-after-batch0.txt
```

Expected: 10/10 pass. This is the "post-Batch-0, pre-intel-iteration" snapshot.

- [ ] **Step 7: Consolidate baseline JSONL**

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/python -c "
import json, pathlib
lines = []
for p in ['eval_results/intel-baseline-V2-scorecard.json',
         'eval_results/phase1-smoke-run1.json',
         'eval_results/phase1-smoke-run2.json']:
    lines.append(json.dumps({'source': p, 'scorecard': json.load(open(p))}))
pathlib.Path('eval_results/intel-baseline-V2.jsonl').write_text('\n'.join(lines) + '\n')
print('wrote eval_results/intel-baseline-V2.jsonl')
"
```

- [ ] **Step 8: Commit baseline artifacts**

```bash
git add -f eval_results/intel-baseline-V2.jsonl \
    eval_results/intel-baseline-V2-scorecard.json \
    eval_results/phase1-smoke-run1.json \
    eval_results/phase1-smoke-run2.json \
    eval_results/phase1-canary-after-batch0.txt \
    eval_results/phase1-full-sweep.log
git commit -m "intel-eval: Phase 1 baseline on V2 — full sweep + smoke × 2 + canary"
```

---

## Task 11: Phase 2 — one reference iteration

**Files:** `src/generation/prompts.py` (example tweak — actual variable depends on what Phase 1 baseline revealed as weakest per-intent score).

This task runs ONE reference iteration end-to-end so the controller + user have a template. Subsequent iterations are run interactively by the controller at the user's pace, one approval at a time.

- [ ] **Step 1: Create the Phase 2 branch**

```bash
git checkout -b intel-eval-phase2
```

- [ ] **Step 2: Identify weakest sub-score from Phase 1**

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/python -c "
import json
sc = json.load(open('eval_results/intel-baseline-V2-scorecard.json'))
pi_int = sc['per_intent_intelligence']
print('Weakest intents by intelligence:')
for k, v in sorted(pi_int.items(), key=lambda kv: kv[1])[:5]:
    print(f'  {k}: {v}')
print()
pi_acc = sc['per_intent_accuracy']
print('Weakest intents by accuracy:')
for k, v in sorted(pi_acc.items(), key=lambda kv: kv[1])[:5]:
    print(f'  {k}: {v}')
"
```

Use the weakest intent to shape the iteration-1 hypothesis.

- [ ] **Step 3: Propose iteration 1 to the user (via controller)**

Controller (not the subagent) constructs the proposal in this format:
```
=== Phase 2 Iteration 1 proposal ===
Hypothesis:
  <one paragraph: what variable, why, expected delta>

Change:
  <file:line — one variable only>

Expected Δ on fast-smoke combined:
  +0.02 to +0.04 (threshold for KEEP is +0.02)

Rollback:
  git reset --hard HEAD~1

APPROVE / DECLINE / AMEND?
```

User replies APPROVE / DECLINE / AMEND.

- [ ] **Step 4: If APPROVED — apply the single edit**

Subagent makes the exact edit the proposal described, in ONE commit:

```bash
# (edit the file)
git add <edited file>
git commit -m "intel-phase2: iter-1 <variable>"
```

- [ ] **Step 5: Run fast-smoke**

```bash
DOCWAIN_SMOKE_PROFILE=69e78d09af9231725f583b3d \
DOCWAIN_SMOKE_SUB=67fde0754e36c00b14cea7f5 \
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/python -m scripts.intel_eval.fast_smoke \
    --out eval_results/phase2-iter-1-smoke.json
```

- [ ] **Step 6: Compute Δ and decide KEEP/REVERT**

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/python -c "
import json
base = json.load(open('eval_results/phase1-smoke-run2.json'))['combined']
iter1 = json.load(open('eval_results/phase2-iter-1-smoke.json'))['combined']
delta = iter1 - base
print(f'baseline={base:.4f}  iter1={iter1:.4f}  delta={delta:+.4f}')
if delta >= 0.02:
    print('DECISION: KEEP')
else:
    print('DECISION: REVERT')
"
```

- [ ] **Step 7: If REVERT**

```bash
git reset --hard HEAD~1
```

Rewrite the commit amendments so there's no empty/reverted commit on the branch.

- [ ] **Step 8: If KEEP — amend commit subject with delta and write delta doc**

```bash
git commit --amend -m "intel-phase2: iter-1 <variable> +ΔX.XX"
mkdir -p eval_deltas
# (write eval_deltas/phase2-iter-1.md with before/after, hypothesis, outcome)
git add eval_deltas/phase2-iter-1.md eval_results/phase2-iter-1-smoke.json
git commit -m "intel-phase2: iter-1 delta doc + smoke artifact"
```

- [ ] **Step 9: Append to smoke log**

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/python -c "
import json, pathlib
rec = {'iter': 1, 'decision': '<KEEP|REVERT>',
       'combined': json.load(open('eval_results/phase2-iter-1-smoke.json'))['combined']}
with open('eval_results/phase2-smoke-log.ndjson', 'a') as f:
    f.write(json.dumps(rec) + '\n')
"
git add -f eval_results/phase2-smoke-log.ndjson
git commit --amend --no-edit
```

- [ ] **Step 10: Report outcome — controller continues iteration loop until stop condition**

Controller loops:
- Propose iter-2, 3, …
- Same steps 3-9.
- STOP when any of:
  - 2 consecutive REVERTs.
  - 8 total iterations.
  - Combined ≥ 0.90.

On STOP: controller tags the branch:
```bash
git tag intel-phase2-P-star
git push origin intel-phase2-P-star  # if credentialed; otherwise defer
```

---

## Task 12: Contender spin-up script

**Files:**
- Create: `scripts/intel_eval/spin_contender.sh`

- [ ] **Step 1: Write the script**

```bash
#!/usr/bin/env bash
# Start/stop a secondary vLLM instance on port 8101 for Phase 3
# contender evaluation. The primary instance on 8100 is NOT touched —
# live serving continues through Batch-0-fixed V2.

set -euo pipefail

ACTION="${1:-}"
MODEL="${2:-}"

VLLM_BIN="/home/ubuntu/PycharmProjects/DocWain/.venv/bin/python -m vllm.entrypoints.openai.api_server"
CONTENDER_PORT="8101"
PID_FILE="/tmp/docwain-contender.pid"
LOG_FILE="/tmp/docwain-contender.log"

case "$ACTION" in
    up)
        if [ -z "$MODEL" ]; then
            echo "usage: spin_contender.sh up <model_path_or_hf_id>" >&2
            exit 2
        fi
        if [ -f "$PID_FILE" ] && kill -0 "$(cat $PID_FILE)" 2>/dev/null; then
            echo "contender already up (pid $(cat $PID_FILE))" >&2
            exit 0
        fi
        echo "starting vLLM contender on :$CONTENDER_PORT with model=$MODEL..."
        nohup $VLLM_BIN \
            --model "$MODEL" \
            --port "$CONTENDER_PORT" \
            --served-model-name docwain-contender \
            --dtype auto \
            --gpu-memory-utilization 0.45 \
            --max-model-len 8192 \
            > "$LOG_FILE" 2>&1 &
        echo $! > "$PID_FILE"
        # Wait for /health
        for i in $(seq 1 60); do
            if curl -sf "http://localhost:$CONTENDER_PORT/health" >/dev/null 2>&1; then
                echo "contender ready on :$CONTENDER_PORT (pid $(cat $PID_FILE))"
                exit 0
            fi
            sleep 2
        done
        echo "contender failed to become healthy in 120s. Tail of log:" >&2
        tail -40 "$LOG_FILE" >&2
        kill "$(cat $PID_FILE)" 2>/dev/null || true
        rm -f "$PID_FILE"
        exit 1
        ;;
    down)
        if [ ! -f "$PID_FILE" ]; then
            echo "no contender pid file; nothing to stop"
            exit 0
        fi
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            kill "$PID"
            # Wait up to 30s for clean shutdown
            for i in $(seq 1 15); do
                if ! kill -0 "$PID" 2>/dev/null; then break; fi
                sleep 2
            done
            if kill -0 "$PID" 2>/dev/null; then
                kill -9 "$PID"
            fi
        fi
        rm -f "$PID_FILE"
        echo "contender stopped"
        ;;
    status)
        if [ -f "$PID_FILE" ] && kill -0 "$(cat $PID_FILE)" 2>/dev/null; then
            echo "up (pid $(cat $PID_FILE))"
            curl -sf "http://localhost:$CONTENDER_PORT/health" >/dev/null && \
                echo "health: ok" || echo "health: FAIL"
        else
            echo "down"
        fi
        ;;
    *)
        echo "usage: spin_contender.sh {up <model>|down|status}" >&2
        exit 2
        ;;
esac
```

- [ ] **Step 2: Make executable**

```bash
chmod +x scripts/intel_eval/spin_contender.sh
```

- [ ] **Step 3: Dry-run status command**

```bash
scripts/intel_eval/spin_contender.sh status
```

Expected: `down`.

- [ ] **Step 4: Commit**

```bash
git add scripts/intel_eval/spin_contender.sh
git commit -m "intel-eval: spin_contender.sh — secondary vLLM on port 8101"
```

---

## Task 13: Phase 3 runner

**Files:**
- Create: `scripts/intel_eval/phase3_runner.py`

- [ ] **Step 1: Implement**

Write `scripts/intel_eval/phase3_runner.py`:

```python
"""Phase 3 runner — cycles the 5 contenders through a secondary vLLM,
runs the overnight_qa full sweep against each, scores with frozen P*
prompts, emits a ranking markdown.

Usage:

    python -m scripts.intel_eval.phase3_runner --pstack-sha <sha> --out eval_results/phase3-ranking.md

Requires:
  - spin_contender.sh executable at scripts/intel_eval/spin_contender.sh
  - Judge Ollama model (per judge_config.yaml) up on :11434
  - Primary vLLM on :8100 (live) is NOT touched by this runner
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time

_CONTENDERS = [
    ("v2-full",     "/home/ubuntu/PycharmProjects/DocWain/models/DocWain-14B-v2"),
    ("v2-awq",      "/home/ubuntu/PycharmProjects/DocWain/models/DocWain-14B-v2-AWQ"),
    ("v5-full",     "/home/ubuntu/PycharmProjects/DocWain/models/DocWain-14B-v5"),
    ("v5-8b",       "/home/ubuntu/PycharmProjects/DocWain/models/DocWain-8B-v5"),
    ("qwen3-14b",   "Qwen/Qwen3-14B"),
]

_SPIN = pathlib.Path(__file__).parent / "spin_contender.sh"


def _wait_health(port: int, timeout_s: int = 120) -> bool:
    import urllib.request
    for _ in range(timeout_s // 2):
        try:
            with urllib.request.urlopen(f"http://localhost:{port}/health", timeout=3):
                return True
        except Exception:
            time.sleep(2)
    return False


def _run_sweep(contender: str, target_url: str, pstack_sha: str) -> pathlib.Path:
    """Run the overnight_qa harness against the contender. Returns the
    results dir."""
    run_id = f"phase3-{contender}-{pstack_sha[:7]}"
    cmd = [
        "/home/ubuntu/PycharmProjects/DocWain/.venv/bin/python",
        "-m", "scripts.overnight_qa.harness",
        "--run-id", run_id,
        "--target-url", target_url,
    ]
    print(f">>> running sweep for {contender}: {' '.join(cmd)}")
    rc = subprocess.call(cmd)
    if rc != 0:
        print(f"!!! sweep for {contender} exited with rc={rc}", file=sys.stderr)
    return pathlib.Path(f"scripts/overnight_qa/results/{run_id}")


def _score(run_dir: pathlib.Path, contender: str) -> dict:
    out = pathlib.Path(f"eval_results/phase3-{contender}-scorecard.json")
    subprocess.check_call([
        "/home/ubuntu/PycharmProjects/DocWain/.venv/bin/python",
        "-m", "scripts.intel_eval.score_run",
        str(run_dir), "--out", str(out),
    ])
    return json.loads(out.read_text())


def _render_ranking(results: list[dict]) -> str:
    qualifying = [r for r in results if r["scorecard"]["overall_pass"]]
    disqualified = [r for r in results if not r["scorecard"]["overall_pass"]]
    qualifying.sort(key=lambda r: -r["scorecard"]["combined"])

    md = ["# Phase 3 Ranking", ""]
    md.append("## Qualifying (ranked by combined score)")
    md.append("| Rank | Contender | Combined | Accuracy | Intelligence | p95 simple | p95 mod | p95 cmpx | Invalid % |")
    md.append("|------|-----------|----------|----------|--------------|------------|---------|----------|-----------|")
    for i, r in enumerate(qualifying, 1):
        s = r["scorecard"]
        md.append(
            f"| {i} | {r['contender']} | {s['combined']:.4f} | {s['accuracy']:.4f} | {s['intelligence']:.4f} | "
            f"{s['latency_gates']['simple']['p95_s']:.1f}s | {s['latency_gates']['moderate']['p95_s']:.1f}s | "
            f"{s['latency_gates']['complex']['p95_s']:.1f}s | "
            f"{100*s['invalid_count']/max(s['valid_count']+s['invalid_count'],1):.1f}% |"
        )
    md.append("")
    md.append("## Disqualified")
    for r in disqualified:
        s = r["scorecard"]
        reasons = []
        for b, g in s["latency_gates"].items():
            if not g["pass"]:
                reasons.append(f"{b}-latency p95={g['p95_s']:.1f}s > {g['threshold_s']}s")
        total = s["valid_count"] + s["invalid_count"]
        inv_ratio = s["invalid_count"] / max(total, 1)
        if inv_ratio >= 0.10:
            reasons.append(f"invalid {100*inv_ratio:.1f}% >= 10%")
        md.append(f"- **{r['contender']}** — {', '.join(reasons) or 'unknown reason'}")
    md.append("")

    if qualifying:
        winner = qualifying[0]
        v2 = next((r for r in results if r["contender"] == "v2-full"), None)
        if v2:
            margin = winner["scorecard"]["combined"] - v2["scorecard"]["combined"]
            md.append(f"## Winner: **{winner['contender']}** (combined = {winner['scorecard']['combined']:.4f})")
            md.append("")
            md.append(f"Margin vs V2-full: **{margin:+.4f}** — {'swap eligible (>= 0.05)' if margin >= 0.05 else 'below swap threshold'}")
    return "\n".join(md) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pstack-sha", required=True,
                    help="Git SHA of the P* frozen prompt stack (intel-phase2-P-star tag resolves to this)")
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("eval_results/phase3-ranking.md"))
    ap.add_argument("--only", nargs="*", default=None,
                    help="Limit to a subset of contenders (by short name)")
    args = ap.parse_args()

    contenders = _CONTENDERS
    if args.only:
        contenders = [c for c in contenders if c[0] in args.only]

    results = []
    for name, model_path in contenders:
        print(f"\n===== contender: {name} =====")
        subprocess.check_call([str(_SPIN), "up", model_path])
        if not _wait_health(8101, 180):
            print(f"!!! contender {name} never became healthy, skipping")
            subprocess.call([str(_SPIN), "down"])
            continue
        try:
            run_dir = _run_sweep(name, "http://localhost:8101", args.pstack_sha)
            scorecard = _score(run_dir, name)
            results.append({"contender": name, "model_path": model_path, "scorecard": scorecard})
        finally:
            subprocess.call([str(_SPIN), "down"])
            # Safety: let GPU memory reclaim before next contender
            time.sleep(10)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(_render_ranking(results))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test CLI**

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/python -m scripts.intel_eval.phase3_runner --help
```

Expected: argparse help.

- [ ] **Step 3: Commit**

```bash
git add scripts/intel_eval/phase3_runner.py
git commit -m "intel-eval: phase3_runner.py — 5-contender ranker"
```

---

## Task 14: Phase 3 execution

**Files:** no code. Produces ranking markdown.

- [ ] **Step 1: Confirm P* branch is tagged**

```bash
git describe intel-phase2-P-star --always
```

Expected: the commit SHA of the frozen P*. If the tag doesn't exist (Phase 2 didn't run or didn't tag), STOP.

- [ ] **Step 2: Confirm GPU has capacity for secondary vLLM**

```bash
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
```

Expected: `memory.free` ≥ 16 GB. If < 16 GB: STOP and report `BLOCKED: GPU free memory insufficient for contender (need 16GB; have XGB)`. Fallback (OOM path) requires explicit user approval per spec §12.

- [ ] **Step 3: Run the full Phase 3 sweep**

```bash
PSTACK_SHA=$(git rev-parse intel-phase2-P-star)
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/python -m scripts.intel_eval.phase3_runner \
    --pstack-sha "$PSTACK_SHA" \
    --out eval_results/phase3-ranking.md \
    2>&1 | tee eval_results/phase3-run.log
```

Runtime: ~8 hours (5 contenders × ~100 min each). Run overnight. If the runner fails mid-way, it leaves a partial set of contender scorecards — resume by `--only` with the remaining contenders.

- [ ] **Step 4: Inspect the ranking**

```bash
cat eval_results/phase3-ranking.md
```

Check:
- Did all 5 contenders qualify? If any are disqualified, note the reason (latency gate vs invalid%).
- Is V2-full in the qualifying set? (It should be; disqualifying V2 means either the harness or Batch-0-merged retrieval has drifted — STOP and investigate.)
- Who's the winner?

- [ ] **Step 5: Commit artifacts**

```bash
git add -f eval_results/phase3-ranking.md eval_results/phase3-run.log \
    eval_results/phase3-*-scorecard.json
git commit -m "intel-eval: Phase 3 ranking across 5 contenders"
```

---

## Task 15: Phase 4 decision + Phase 5 swap bundle

**Files:**
- Create (only if swap): `deploy/phase5-swap/winner.json`
- Create (only if swap): `deploy/phase5-swap/swap.sh`
- Create (only if swap): `deploy/phase5-swap/rollback.sh`
- Create (only if swap): `deploy/phase5-swap/canary.txt`
- Create (only if swap): `deploy/phase5-swap/watch.md`

- [ ] **Step 1: Read the ranking**

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/python -c "
import json, pathlib
# Parse the ranking markdown for the winner and margin. Simpler: read
# individual scorecard JSONs directly.
import glob
cards = {}
for p in glob.glob('eval_results/phase3-*-scorecard.json'):
    name = p.split('phase3-')[1].split('-scorecard.json')[0]
    cards[name] = json.load(open(p))
qualifying = {k: v for k, v in cards.items() if v['overall_pass']}
winner = max(qualifying.items(), key=lambda kv: kv[1]['combined']) if qualifying else (None, None)
v2 = cards.get('v2-full')
print(f'winner: {winner[0]}')
print(f'winner combined: {winner[1][\"combined\"]:.4f}')
print(f'v2-full combined: {v2[\"combined\"]:.4f}')
print(f'margin: {winner[1][\"combined\"] - v2[\"combined\"]:+.4f}')
print(f'swap eligible: {(winner[1][\"combined\"] - v2[\"combined\"]) >= 0.05 and winner[0] != \"v2-full\"}')
"
```

- [ ] **Step 2a: If winner == v2-full OR margin < 0.05 → NO SWAP**

Write `eval_results/phase5-no-swap.md`:

```bash
cat > eval_results/phase5-no-swap.md <<'EOF'
# Phase 5: No Swap

V2-full retained as the unified DocWain model.

See `eval_results/phase3-ranking.md` for detailed contender comparison.

## Action

- Merge `intel-eval-phase2` branch (Phase 2 improvements) to main as a
  separate flag-free PR.
- Phase 3 ranking artifact committed for historical reference.
- No systemd restart, no symlink change, no canary.
EOF

git add -f eval_results/phase5-no-swap.md
git commit -m "intel-eval: Phase 5 no-swap close-out (V2 retained)"
```

Then open the P* PR:

```bash
git push -u origin intel-eval-phase2
# gh pr create --base main --head intel-eval-phase2 --title "intel-phase2: engineering-layer improvements (P*)" ...
```

STOP here. Workstream complete in no-swap path.

- [ ] **Step 2b: If winner ≠ v2-full AND margin ≥ 0.05 → PREPARE SWAP BUNDLE**

Create the branch:
```bash
git checkout main
git pull --ff-only origin main
git checkout -b intel-eval-phase5-swap
mkdir -p deploy/phase5-swap
```

Write `deploy/phase5-swap/winner.json` (fill in the concrete names from Step 1's output):

```json
{
  "winner_name": "<v5-full | v2-awq | v5-8b | qwen3-14b>",
  "winner_model_path": "<absolute path>",
  "served_name": "docwain",
  "source_phase3_run_id": "<phase3-<winner>-<sha>>",
  "winner_combined": <float>,
  "v2_baseline_combined": <float>,
  "margin": <float>
}
```

Write `deploy/phase5-swap/swap.sh`:

```bash
#!/usr/bin/env bash
set -eux
# Snapshot old target for safe rollback
readlink -f /home/ubuntu/PycharmProjects/DocWain/models/docwain-v2-active > /tmp/docwain-pre-swap-target.txt
ln -sfn <WINNER_ABSOLUTE_PATH> /home/ubuntu/PycharmProjects/DocWain/models/docwain-v2-active
sudo systemctl restart docwain-vllm-fast.service
for i in $(seq 1 30); do
    curl -sf http://localhost:8100/health >/dev/null && break
    sleep 2
done
curl -sf http://localhost:8100/health >/dev/null
echo "swap completed; vLLM healthy"
```

Write `deploy/phase5-swap/rollback.sh`:

```bash
#!/usr/bin/env bash
set -eux
PREV_TARGET="$(cat /tmp/docwain-pre-swap-target.txt 2>/dev/null || echo /home/ubuntu/PycharmProjects/DocWain/models/DocWain-14B-v2)"
ln -sfn "$PREV_TARGET" /home/ubuntu/PycharmProjects/DocWain/models/docwain-v2-active
sudo systemctl restart docwain-vllm-fast.service
for i in $(seq 1 30); do
    curl -sf http://localhost:8100/health >/dev/null && break
    sleep 2
done
curl -sf http://localhost:8100/health >/dev/null
echo "rollback completed; vLLM healthy"
```

Write `deploy/phase5-swap/canary.txt`:

```
DOCWAIN_SMOKE_PROFILE=69e78d09af9231725f583b3d \
DOCWAIN_SMOKE_SUB=67fde0754e36c00b14cea7f5 \
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/python \
scripts/batch0/canary_smoke.py --base-url http://localhost:8000
```

Write `deploy/phase5-swap/watch.md`:

```markdown
# Post-swap watch

## 15-minute observation

Run in one terminal:

    sudo journalctl -fu docwain-app.service \
      | grep --line-buffered -E "ERROR|Traceback|grounded=False|empty"

Auto-rollback trigger: ERROR rate > 2 × pre-swap baseline in any 1-min window.

## Baseline capture (run 1 min BEFORE the swap)

    sudo journalctl -u docwain-app.service --since "1 minute ago" \
      | grep -cE "ERROR|Traceback" \
      > /tmp/docwain-pre-swap-error-baseline.txt
```

Commit the bundle:

```bash
chmod +x deploy/phase5-swap/swap.sh deploy/phase5-swap/rollback.sh
git add -f deploy/phase5-swap/
git commit -m "intel-eval: Phase 5 swap bundle (winner=<name>)"
```

**GATE: present bundle to user, get explicit "go to Phase 5" before proceeding to Task 16.**

---

## Task 16: Phase 5 execution (ONLY if Task 15 produced a swap bundle and user said go)

**Files:** no code. Live-affecting.

- [ ] **Step 1: Pre-swap error baseline**

```bash
sudo journalctl -u docwain-app.service --since "1 minute ago" \
  | grep -cE "ERROR|Traceback" \
  > /tmp/docwain-pre-swap-error-baseline.txt
cat /tmp/docwain-pre-swap-error-baseline.txt
```

Record the baseline count.

- [ ] **Step 2: Dry-run swap.sh and rollback.sh on a staging symlink**

```bash
mkdir -p /tmp/phase5-dryrun
ln -sfn /home/ubuntu/PycharmProjects/DocWain/models/DocWain-14B-v2 /tmp/phase5-dryrun/docwain-v2-active
# Verify commands are shell-parsable
bash -n deploy/phase5-swap/swap.sh
bash -n deploy/phase5-swap/rollback.sh
echo "shell syntax OK"
```

- [ ] **Step 3: User GATE — show winner.json, watch.md, canary.txt. Get "go".**

User replies `GO` or `NO-GO`. If NO-GO, STOP — workstream ends with bundle committed, not executed.

- [ ] **Step 4: Execute swap.sh (live-affecting)**

```bash
sudo bash deploy/phase5-swap/swap.sh 2>&1 | tee /tmp/phase5-swap-run.log
```

If command exits non-zero OR health never comes up: run `sudo bash deploy/phase5-swap/rollback.sh` immediately, then STOP with `BLOCKED: swap failed, auto-rolled back`.

- [ ] **Step 5: Run canary**

```bash
bash deploy/phase5-swap/canary.txt 2>&1 | tee eval_results/phase5-canary-after-swap.txt
```

Count failures:
- 0-2 fail → continue to watch
- 3+ fail → run `sudo bash deploy/phase5-swap/rollback.sh` immediately, STOP

- [ ] **Step 6: 15-minute watch via Monitor tool**

Use the Monitor tool with a command that emits only actionable events:

```bash
journalctl -fu docwain-app.service --since "just now" \
  | grep --line-buffered -E "ERROR|Traceback|grounded=False|empty"
```

Watch for 15 minutes. If ERROR rate in any 1-minute window > 2 × baseline: run rollback.sh immediately.

- [ ] **Step 7: Write close-out**

On clean window:

```bash
cat > eval_results/phase5-swap-outcome.md <<EOF
# Phase 5 Swap Outcome

Winner: <name>
Swap at: $(date -u +%Y-%m-%dT%H:%M:%SZ)
Canary: $(grep -c '^.*YES' eval_results/phase5-canary-after-swap.txt)/10 passed
15-min watch: clean
Status: SWAPPED

Pre-swap error baseline (per min): $(cat /tmp/docwain-pre-swap-error-baseline.txt)
EOF

git add -f eval_results/phase5-swap-outcome.md
git commit -m "intel-eval: Phase 5 swap complete (<winner>)"
```

Push + open PR to merge the swap bundle to main as the audit trail.

---

## Self-Review

**Spec coverage:**
- Phase 1 baseline (§4 row 1, §5.1-5.3, §5.5, §11.2) → Tasks 1-10.
- Phase 2 iteration (§4 row 2, §8) → Tasks 11 + controller-driven loop.
- Phase 3 model comparison (§4 row 3, §5.4, §7) → Tasks 12-14.
- Scoring rubric (§6) → Tasks 3-5 (score_run + latency gate tests), Task 9 (judge wiring).
- Phase 4 winner handling (§4 row 4) → Task 15.
- Phase 5 swap (§4 row 5, §9) → Tasks 15b-16.
- Artifacts (§10) → each task that produces artifacts commits them.
- Testing (§11) → Tasks 3, 5, 7, 10 (reproducibility), 16 (rollback dry-run).

**Placeholder scan:**
- `<name>` / `<winner>` / `<WINNER_ABSOLUTE_PATH>` — intentional template fills that the executor substitutes at Task 15 Step 2b based on actual Phase 3 output. Each is called out with `<`/`>` brackets and explicit instruction.
- `<one paragraph: what variable, why, expected delta>` — intentional: each Phase 2 iteration's proposal is generated at iteration time, not pre-written.
- No `TBD`, no `implement later`.

**Type consistency:**
- `score_run(run_dir: pathlib.Path, out_path: pathlib.Path) -> dict` consistent across Tasks 3, 4, 8, 10.
- Scorecard schema consistent — same keys in test fixtures (Task 3), implementation (Task 4), latency-gate tests (Task 5), fast_smoke output (Task 8), phase3 render (Task 13), phase5 decision (Task 15).
- `run_id` naming pattern `"<phase>-<descriptor>-<id>"` consistent.
