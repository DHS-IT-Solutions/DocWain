# DocWain SME Phase 4 — Rich-Mode Responses + New Intents + Domain Persona Injection

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Flip DocWain from extractive RAG to pre-reasoned SME synthesis in the user-visible surface. Three new intents (`diagnose`, `analyze`, `recommend`) land with rich-default response shape; domain-SME persona injection goes live; every response shape decision is governed by `src/generation/prompts.py` and adapter YAMLs loaded from Azure Blob; `src/intelligence/generator.py` remains formatting-free. Recommendation-intent responses gain a post-generation grounding pass that drops unverifiable claims. Existing 0.0 hallucination rate is preserved as a regression gate.

**Architecture (additive):** `src/serving/model_router.py` intent classifier extended with three labels plus `format_hint`. `src/generation/prompts.py` gains rich templates, shape resolution, compact override, and persona injection. `src/agent/core_agent.py` wires the adapter resolver through to prompt builders and enforces per-intent output caps. `src/generation/recommendation_grounding.py` (NEW) runs the post-pass on recommend-intent outputs only. `src/config/feature_flags.py` (from Phase 1 per ERRATA §4) carries the `enable_rich_mode` flag; Phase 4 is a consumer only. No other `src/` files change for formatting.

**Tech Stack:** Python 3.12, pydantic (existing), asyncio (existing), pytest, `pyyaml` for adapter schema tests. Adapter YAMLs + rich-mode prompt templates live in Azure Blob under `sme_adapters/` (established in Phase 1); Phase 4 loads them via the Phase 1 `AdapterLoader`. No new storage systems.

**Related spec:** `docs/superpowers/specs/2026-04-20-docwain-profile-sme-reasoning-design.md` — Sections 5 (domain adapters), 8 (prompts/shape/grounding), 10 (metrics), 12 Phase 4 (exit gate), 13 (rollback).

**Preceding phases (MUST be complete):**
- Phase 0 — baseline captured (`tests/sme_metrics_baseline_YYYY-MM-DD.json` exists, tagged `sme-baseline-v1`)
- Phase 1 — `AdapterLoader` + Blob storage + `generic.yaml`/finance/legal/hr/medical/it_support adapters shipped
- Phase 2 — SME synthesis produces Dossier / Insight Index / Comparative Register / Recommendation Bank on opt-in subscriptions
- Phase 3 — SME retrieval layers A-C on; faithfulness on eval set has moved from 0.514 towards ≥0.80; response shape still compact

**Memory rules that constrain this plan (strict):**
- Response formatting lives ONLY in `src/generation/prompts.py` (and, for the recommendation post-pass, a tight `src/generation/recommendation_grounding.py` that only touches the final response text). NEVER modify `src/intelligence/generator.py` for formatting. This is THE memory rule for Phase 4 — it will be called out again at every touch point.
- No Claude / Anthropic references anywhere in code, commits, templates, YAML strings, or docs.
- Adapter YAMLs + rich-mode prompt templates live in Azure Blob. No hardcoded domain templates under `src/`. The only exception is the emergency-fallback `deploy/sme_adapters/last_resort/generic.yaml` (already shipped in Phase 1).
- MongoDB is control plane only. Profile records carry `profile_domain` (added Phase 1); no new response content is ever written to Mongo.
- Zero internal timeouts. No wall-clock aborts on intent classification, prompt construction, generation, or the recommendation post-pass. External I/O (Blob fetch inside `AdapterLoader`) has its existing per-operation safety timeouts from Phase 1 — not introduced here.
- `enable_rich_mode` defaults OFF. Rollout is per-subscription via the Phase 1 flag registry.
- Profile isolation: every prompt build path passes `(subscription_id, profile_id)` to the adapter resolver; the resolver's cache is keyed by `(subscription_id, domain)`.
- No customer documents in any fixture, eval, or test.
- Engineering-first: no model retraining, no fine-tuning triggers added by this plan.
- `pipeline_status` strings are immutable. Phase 4 touches the query path only; no ingest status changes.

---

## File structure

```
src/generation/prompts.py                        [MODIFIED — rich templates, shape resolution, persona injection]
src/generation/recommendation_grounding.py       [NEW — post-generation grounding pass for recommend intent]
src/generation/__init__.py                       [MODIFIED — re-export new helpers]
src/serving/model_router.py                      [MODIFIED — extend classifier labels + format_hint + URL detection]
src/agent/core_agent.py                          [MODIFIED — wire adapter resolver, pass persona + output_caps]
src/config/feature_flags.py                      [CONSUMER ONLY — Phase 1 owns this module per ERRATA §4]
src/api/admin_flags.py                           [MODIFIED — expose toggle endpoint]

tests/generation/test_prompts_rich.py            [NEW]
tests/generation/test_shape_resolution.py        [NEW]
tests/generation/test_persona_injection.py       [NEW]
tests/generation/test_recommendation_grounding.py[NEW]
tests/serving/test_model_router_intents.py       [NEW]
tests/agent/test_core_agent_rich_wire.py         [NEW]
tests/config/test_features_rich_flag.py          [NEW]
tests/integration/test_phase4_rich_mode.py       [NEW]

tests/sme_evalset_v1/                            [existing — reused]
tests/sme_metrics_phase4_{YYYY-MM-DD}.json       [NEW — frozen Phase 4 snapshot]
tests/sme_human_rating_phase4_{YYYY-MM-DD}.csv   [NEW — post-rating pass]

deploy/sme_adapters/defaults/                    [MODIFIED — rich-mode prompt templates for each domain]
  global/
    prompts/
      rich_analyze.md                            [NEW — generic fallback analyze template]
      rich_diagnose.md                           [NEW]
      rich_recommend.md                          [NEW]
    finance.yaml                                 [MODIFIED — fill response_persona_prompts]
    legal.yaml                                   [MODIFIED]
    hr.yaml                                      [MODIFIED]
    medical.yaml                                 [MODIFIED]
    it_support.yaml                              [MODIFIED]
    generic.yaml                                 [MODIFIED]
    prompts/finance_analyze.md                   [NEW per-domain overrides]
    prompts/finance_diagnose.md
    prompts/finance_recommend.md
    prompts/legal_analyze.md
    # ... one set per shipped domain
```

Boundary: `src/generation/prompts.py` owns prompt text. `src/generation/recommendation_grounding.py` owns only the post-generation rewrite (not construction). `src/agent/core_agent.py` owns wiring — it never builds prompt strings directly. `src/intelligence/generator.py` is not opened by this phase.

---

## Task 1: Audit Phase 1-3 prerequisites

**Files:**
- Audit only (no modifications): `src/intelligence/sme/adapter_loader.py`, `src/retrieval/sme_retrieval.py`, `src/agent/core_agent.py`, `src/generation/prompts.py`, `src/config/feature_flags.py`, `tests/sme_metrics_baseline_*.json`

No production code touched; this is a pre-flight pass. The implementer confirms the artifacts Phase 4 depends on are in place and records the exact call signatures they must integrate with.

- [ ] **Step 1: Confirm Phase 0 baseline exists and is committed**

```bash
ls -la tests/sme_metrics_baseline_*.json
git tag --list 'sme-baseline-v1*'
```

Expected: exactly one baseline JSON, one tag. If absent, **stop** — Phase 0 must complete first.

- [ ] **Step 2: Confirm the Phase 1 `AdapterLoader` exists and exposes the contract Phase 4 needs**

```bash
python -c "
from src.intelligence.sme.adapter_loader import AdapterLoader
al = AdapterLoader
# These methods must exist (names are the Phase 1 contract)
assert hasattr(al, 'load'), 'AdapterLoader.load missing'
assert hasattr(al, 'invalidate'), 'AdapterLoader.invalidate missing'
print('AdapterLoader contract OK')
"
```

Record the exact signature of `load(subscription_id: str, profile_domain: str) -> Adapter` — Phase 4 calls it unchanged. If the signature differs, **halt and reconcile** with Phase 1 before editing anything.

- [ ] **Step 3: Confirm Phase 1 `Adapter` object carries the fields Phase 4 needs**

Walk `src/intelligence/sme/adapter_loader.py` and verify the loaded `Adapter` exposes:
- `persona.role`, `persona.voice`, `persona.grounding_rules`
- `response_persona_prompts.analyze`, `.diagnose`, `.recommend` (blob-relative paths)
- `retrieval_caps.max_pack_tokens` per intent
- `output_caps` per intent
- `version`, `content_hash`

Record any gap. If fields are missing, note them under "Open questions" in the final commit message — do not silently extend the Adapter here; that belongs to a Phase 1 amendment PR.

- [ ] **Step 4: Confirm Phase 3 retrieval is wired and SME artifacts can reach the pack**

```bash
grep -n "sme_retrieval" src/agent/core_agent.py
grep -n "unified_retriever" src/agent/core_agent.py
```

There must be a code path that already populates the pack from Layer A / B / C. Phase 4 adds ONLY shape + persona + post-pass on top.

- [ ] **Step 5: Confirm `enable_rich_mode` flag is shipped by Phase 1 per ERRATA §4**

```bash
grep -n "ENABLE_RICH_MODE\|enable_rich_mode" src/config/feature_flags.py
```

Expected: Phase 1 exposes the string constant `ENABLE_RICH_MODE = "enable_rich_mode"` and the flag is in `_DEFAULTS` with value `False`. Phase 4 does NOT create a parallel `src/config/features.py` module — the consumer path in Task 8/10 calls `get_flag_resolver().is_enabled(sub, ENABLE_RICH_MODE)`. If the constant is missing, halt and reconcile with Phase 1.

- [ ] **Step 6: Confirm `src/intelligence/generator.py` has no formatting code to remove**

```bash
grep -nE "(TASK_FORMATS|build_.*prompt|system_prompt|rich_template)" src/intelligence/generator.py
```

Expected: no matches. If any match exists, treat it as a Phase 4 blocker: file a separate cleanup PR under commit `phase4(sme-memory-rule): relocate stray formatting from generator.py to prompts.py` BEFORE touching anything else. This preserves the memory rule as an invariant.

- [ ] **Step 7: Record audit outputs**

Write the audit result as a commit-message-ready note (no new file in the repo). Contents:
- Which Phase 1 signatures Phase 4 will bind to
- Which adapter fields exist vs. are missing
- Whether `enable_rich_mode` stub exists
- Whether a `generator.py` cleanup PR is required first

- [ ] **Step 8: Commit the audit outcome (doc-only, if any changes)**

If the audit required no code changes, skip the commit. If it required a `generator.py` cleanup, land that commit first under the scoped message above.

---

## Task 2: Extend intent classifier — 3 new intents, `format_hint`, URL detection

**Files:**
- Modify: `src/serving/model_router.py`
- Create: `tests/serving/test_model_router_intents.py`

The intent classifier is the existing UNDERSTAND-stage LLM call (`src/serving/model_router.py`). Phase 4 extends its label set and output schema. It does NOT add a new LLM call — this is a prompt-and-parse change to the existing call.

- [ ] **Step 1: Write the failing tests**

Create `tests/serving/test_model_router_intents.py`:

```python
"""Tests for the extended intent classifier."""
from unittest.mock import AsyncMock, patch

import pytest

from src.serving.model_router import (
    ClassifiedQuery,
    FormatHint,
    classify_query,
)


@pytest.mark.asyncio
async def test_classifier_recognises_analyze_intent():
    with patch("src.serving.model_router._call_classifier_llm", new=AsyncMock(
        return_value='{"intent": "analyze", "format_hint": "auto", '
                     '"entities": [], "urls": []}'
    )):
        result = await classify_query("Analyze Q3 revenue trends across quarters.")
    assert result.intent == "analyze"
    assert result.format_hint == FormatHint.AUTO


@pytest.mark.asyncio
async def test_classifier_recognises_diagnose_intent():
    with patch("src.serving.model_router._call_classifier_llm", new=AsyncMock(
        return_value='{"intent": "diagnose", "format_hint": "auto", '
                     '"entities": [], "urls": []}'
    )):
        result = await classify_query("Why is the nightly backup failing?")
    assert result.intent == "diagnose"


@pytest.mark.asyncio
async def test_classifier_recognises_recommend_intent():
    with patch("src.serving.model_router._call_classifier_llm", new=AsyncMock(
        return_value='{"intent": "recommend", "format_hint": "auto", '
                     '"entities": [], "urls": []}'
    )):
        result = await classify_query("What should we do to improve margins?")
    assert result.intent == "recommend"


@pytest.mark.asyncio
async def test_classifier_parses_compact_format_hint():
    with patch("src.serving.model_router._call_classifier_llm", new=AsyncMock(
        return_value='{"intent": "analyze", "format_hint": "compact", '
                     '"entities": [], "urls": []}'
    )):
        result = await classify_query("Analyze Q3 trends. Keep it short.")
    assert result.format_hint == FormatHint.COMPACT


@pytest.mark.asyncio
async def test_classifier_detects_compact_override_from_text_heuristic():
    # Even if the LLM returns auto, the deterministic post-parse layer
    # escalates to compact when clear compact-override phrasing is present.
    with patch("src.serving.model_router._call_classifier_llm", new=AsyncMock(
        return_value='{"intent": "analyze", "format_hint": "auto", '
                     '"entities": [], "urls": []}'
    )):
        result = await classify_query(
            "Analyze Q3 trends. tl;dr please, one paragraph."
        )
    assert result.format_hint == FormatHint.COMPACT


@pytest.mark.asyncio
async def test_classifier_detects_urls_deterministically():
    # URL detection is deterministic regex, independent of the LLM output.
    with patch("src.serving.model_router._call_classifier_llm", new=AsyncMock(
        return_value='{"intent": "analyze", "format_hint": "auto", '
                     '"entities": [], "urls": []}'
    )):
        result = await classify_query(
            "Analyze this report: https://example.com/q3-report.pdf"
        )
    assert result.urls == ["https://example.com/q3-report.pdf"]


@pytest.mark.asyncio
async def test_classifier_falls_back_to_overview_on_unparseable_llm_output():
    # When the LLM returns garbage, classifier downgrades to the legacy
    # default "overview" intent with auto format — a conservative fallback.
    with patch("src.serving.model_router._call_classifier_llm", new=AsyncMock(
        return_value="not-json ~~~"
    )):
        result = await classify_query("something opaque")
    assert result.intent == "overview"
    assert result.format_hint == FormatHint.AUTO


@pytest.mark.asyncio
async def test_classifier_rejects_unknown_intent_label():
    with patch("src.serving.model_router._call_classifier_llm", new=AsyncMock(
        return_value='{"intent": "telepathy", "format_hint": "auto", '
                     '"entities": [], "urls": []}'
    )):
        result = await classify_query("anything")
    assert result.intent == "overview"  # rejected label → safe fallback
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/serving/test_model_router_intents.py -v
```

Expected: fail — new labels / `FormatHint` not present yet.

- [ ] **Step 3: Extend the classifier**

Modify `src/serving/model_router.py`. Key additions (shown in skeleton; existing code above remains untouched):

```python
# src/serving/model_router.py — excerpt; pre-existing code unchanged above
from __future__ import annotations

import enum
import json
import re
from dataclasses import dataclass
from typing import Any

# Extend the legacy label set. Order matters for the classifier prompt: the
# new labels are APPENDED so the model sees the stable old labels first.
VALID_INTENTS: tuple[str, ...] = (
    "greeting", "identity", "lookup", "list", "count",
    "summarize", "compare", "overview", "investigate",
    "extract", "aggregate",
    "analyze", "diagnose", "recommend",  # NEW
)

_COMPACT_OVERRIDE_MARKERS: tuple[str, ...] = (
    "tl;dr", "tldr", "one paragraph", "one line", "keep it short",
    "short answer", "in brief", "just the answer",
)

_URL_RE = re.compile(r"https?://[^\s\"')>]+", re.IGNORECASE)


class FormatHint(str, enum.Enum):
    AUTO = "auto"
    COMPACT = "compact"
    RICH = "rich"


@dataclass(frozen=True)
class ClassifiedQuery:
    # `query_text` is the canonical ERRATA §9 field — always populated from the
    # classifier input so downstream builders (Task 8) can read it directly
    # without hasattr/getattr fallbacks.
    query_text: str
    intent: str
    format_hint: FormatHint
    entities: list[str]
    urls: list[str]


async def classify_query(query_text: str) -> ClassifiedQuery:
    raw = await _call_classifier_llm(_build_classifier_prompt(query_text))
    parsed = _safe_parse(raw)
    intent = parsed.get("intent")
    if intent not in VALID_INTENTS:
        intent = "overview"
    hint_raw = parsed.get("format_hint", "auto")
    hint = FormatHint(hint_raw) if hint_raw in FormatHint._value2member_map_ else FormatHint.AUTO
    if hint is FormatHint.AUTO and _looks_like_compact_override(query_text):
        hint = FormatHint.COMPACT
    urls = _URL_RE.findall(query_text)
    ents = [e for e in parsed.get("entities", []) if isinstance(e, str)]
    return ClassifiedQuery(
        query_text=query_text,
        intent=intent,
        format_hint=hint,
        entities=ents,
        urls=urls,
    )


def _safe_parse(raw: str) -> dict[str, Any]:
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return {}


def _looks_like_compact_override(q: str) -> bool:
    lowered = q.lower()
    return any(marker in lowered for marker in _COMPACT_OVERRIDE_MARKERS)
```

The updated classifier prompt (kept inside `_build_classifier_prompt` — untouched signature) must include the new labels and `format_hint: "auto" | "compact" | "rich"` in its output schema. Keep the prompt prose short — no persona, no SME instructions, this LLM call is pure classification.

- [ ] **Step 4: Run tests**

```bash
pytest tests/serving/test_model_router_intents.py -v
```

Expected: 8 passing.

- [ ] **Step 5: Commit**

```bash
git add src/serving/model_router.py tests/serving/test_model_router_intents.py
git commit -m "phase4(sme-classifier): extend intent labels + format_hint + URL detection"
```

---

## Task 3: `prompts.py` rich template — `analyze` intent (full code, load-bearing)

**Files:**
- Modify: `src/generation/prompts.py`
- Create: `tests/generation/test_prompts_rich.py` (initial slice; grows across tasks 3–5)

`analyze` is the most representative of the three new intents — the shape is widest (exec summary + observations + patterns + interpretation + caveats), the persona hook is strictest (domain SME voice), and the token accounting is the tightest. Getting `analyze` right is a template for `diagnose` and `recommend` in tasks 4 and 5.

- [ ] **Step 1: Write the failing tests**

Create `tests/generation/test_prompts_rich.py`:

```python
"""Tests for rich-mode analyze prompt template."""
from src.generation.prompts import (
    AnalyzePromptInputs,
    build_analyze_rich_prompt,
)


def _inputs(**overrides):
    base = dict(
        query_text="Analyze Q3 revenue trends across quarters.",
        persona_role="senior financial analyst advising the C-suite",
        persona_voice="direct, quantitative, hedged",
        grounding_rules=(
            "Cite every claim to [doc_id:chunk_id]. Refuse to extrapolate without evidence.",
        ),
        pack_tokens=3500,
        output_cap_tokens=1200,
        evidence_items=[
            {"doc_id": "q1_report", "chunk_id": "c1", "text": "Q1 revenue was $4.2M."},
            {"doc_id": "q2_report", "chunk_id": "c1", "text": "Q2 revenue was $4.8M."},
            {"doc_id": "q3_report", "chunk_id": "c1", "text": "Q3 revenue was $5.3M."},
        ],
        insight_refs=[
            {"type": "trend", "narrative": "QoQ revenue growth 14.3%, 10.4%."},
        ],
        domain="finance",
    )
    base.update(overrides)
    return AnalyzePromptInputs(**base)


def test_analyze_prompt_contains_persona_block():
    p = build_analyze_rich_prompt(_inputs())
    assert "senior financial analyst" in p
    assert "direct, quantitative, hedged" in p


def test_analyze_prompt_contains_grounding_rules():
    p = build_analyze_rich_prompt(_inputs())
    assert "[doc_id:chunk_id]" in p


def test_analyze_prompt_skeleton_has_five_sections():
    p = build_analyze_rich_prompt(_inputs())
    for section in [
        "## Executive summary",
        "## Analysis",
        "## Patterns",
        "## Assumptions & caveats",
        "## Evidence",
    ]:
        assert section in p


def test_analyze_prompt_binds_output_cap_hint():
    p = build_analyze_rich_prompt(_inputs(output_cap_tokens=900))
    # Cap is surfaced to the LLM as a soft target, not a hard instruction
    assert "900" in p


def test_analyze_prompt_inlines_evidence_with_provenance():
    p = build_analyze_rich_prompt(_inputs())
    assert "q1_report:c1" in p
    assert "q3_report:c1" in p


def test_analyze_prompt_inlines_insight_refs():
    p = build_analyze_rich_prompt(_inputs())
    assert "QoQ revenue growth" in p


def test_analyze_prompt_forbids_silent_extrapolation():
    p = build_analyze_rich_prompt(_inputs())
    # The persona-level grounding rule is echoed into the system prompt.
    assert "Refuse to extrapolate without evidence" in p


def test_analyze_prompt_handles_empty_insight_refs():
    p = build_analyze_rich_prompt(_inputs(insight_refs=[]))
    # Must degrade gracefully — still produce a valid skeleton
    assert "## Analysis" in p
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/generation/test_prompts_rich.py -v
```

Expected: fail — symbols absent.

- [ ] **Step 3: Implement `build_analyze_rich_prompt` (load-bearing full code)**

Add to `src/generation/prompts.py` (placed below existing `TASK_FORMATS` — do not modify existing compact templates):

```python
# src/generation/prompts.py — excerpt; existing helpers unchanged
from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class AnalyzePromptInputs:
    query_text: str
    persona_role: str
    persona_voice: str
    grounding_rules: Sequence[str]
    pack_tokens: int
    output_cap_tokens: int
    evidence_items: Sequence[dict]           # doc_id, chunk_id, text
    insight_refs: Sequence[dict] = field(default_factory=tuple)  # type, narrative
    domain: str = "generic"


def build_analyze_rich_prompt(inp: AnalyzePromptInputs) -> str:
    persona = (
        f"You are acting as a {inp.persona_role}. "
        f"Voice: {inp.persona_voice}. "
        f"Domain context: {inp.domain}."
    )
    grounding = "\n".join(f"- {rule}" for rule in inp.grounding_rules)
    evidence = "\n".join(
        f"[{e['doc_id']}:{e['chunk_id']}] {e['text']}" for e in inp.evidence_items
    )
    insights = (
        "\n".join(f"- ({i['type']}) {i['narrative']}" for i in inp.insight_refs)
        if inp.insight_refs else "(none materialized for this query)"
    )
    return (
        f"{persona}\n\n"
        f"Grounding rules (strict):\n{grounding}\n\n"
        f"User question: {inp.query_text}\n\n"
        f"Pre-reasoned insights from this profile:\n{insights}\n\n"
        f"Evidence (cite inline as [doc_id:chunk_id]):\n{evidence}\n\n"
        "Produce a rich analysis response with EXACTLY these sections, in this order:\n"
        "## Executive summary\n"
        "One to three sentences, headline first, streamed before any other section.\n"
        "## Analysis\n"
        "Evidence-grounded narrative. Every quantitative claim carries an inline citation.\n"
        "## Patterns\n"
        "Cross-document patterns tied to at least two distinct docs where possible.\n"
        "## Assumptions & caveats\n"
        "Explicit assumptions; anything you cannot verify is listed here, not hidden.\n"
        "## Evidence\n"
        "Bullet list of (doc_id:chunk_id) items actually cited above.\n\n"
        f"Soft output target: ~{inp.output_cap_tokens} tokens. Quality trumps the target;\n"
        "never pad. If evidence is thin, say so in Executive summary and shorten Analysis.\n"
    )
```

Notes:
- The template is deterministic Python string assembly. Persona, grounding rules, and insights are injected from the adapter (Task 7) — this function is format-only.
- Executive summary is mandated as the first streamed section. The streaming contract is enforced by the order of sections, not by any new transport code.
- No timeouts. No hidden retries. No Claude / Anthropic references.

- [ ] **Step 4: Run tests**

```bash
pytest tests/generation/test_prompts_rich.py -v
```

Expected: 8 passing.

- [ ] **Step 5: Commit**

```bash
git add src/generation/prompts.py tests/generation/test_prompts_rich.py
git commit -m "phase4(sme-prompts): analyze rich template with persona + grounding"
```

---

## Task 4: `prompts.py` rich template — `diagnose` intent (structure + notes)

**Files:**
- Modify: `src/generation/prompts.py`
- Modify: `tests/generation/test_prompts_rich.py` (append diagnose tests)

`diagnose` follows the same construction pattern as `analyze`. Only the section names and emphasis differ. We intentionally ship a shape-parallel implementation so future maintainers reading one understand all three.

- [ ] **Step 1: Append diagnose tests to `tests/generation/test_prompts_rich.py`**

Add test cases:

```python
from src.generation.prompts import (
    DiagnosePromptInputs,
    build_diagnose_rich_prompt,
)


def _diag_inputs(**overrides):
    base = dict(
        query_text="Why is the nightly backup job failing?",
        persona_role="support engineer triaging incidents",
        persona_voice="methodical, symptom-first, evidence-driven",
        grounding_rules=(
            "Rank causes by evidence strength. Never invent log lines.",
        ),
        pack_tokens=2800,
        output_cap_tokens=1500,
        evidence_items=[
            {"doc_id": "incident_8821", "chunk_id": "c3",
             "text": "07:14 disk full on /var/backups"},
        ],
        diagnostic_hits=[
            {"symptom": "backup job exits 1", "doc_id": "incident_8821",
             "chunk_id": "c3", "rank": 1},
        ],
        domain="it_support",
    )
    base.update(overrides)
    return DiagnosePromptInputs(**base)


def test_diagnose_prompt_contains_persona_block():
    p = build_diagnose_rich_prompt(_diag_inputs())
    assert "support engineer" in p


def test_diagnose_prompt_skeleton_has_five_sections():
    p = build_diagnose_rich_prompt(_diag_inputs())
    for s in [
        "## Executive summary",
        "## Symptom",
        "## Causes",
        "## Assumptions & caveats",
        "## Evidence",
    ]:
        assert s in p


def test_diagnose_prompt_ranks_candidate_causes():
    p = build_diagnose_rich_prompt(_diag_inputs())
    # The template mandates a ranked list under Causes
    assert "rank" in p.lower()
```

- [ ] **Step 2: Implement `DiagnosePromptInputs` + builder**

Do NOT duplicate the entire analyze builder. The diagnose template follows the same assembly shape:

- Inputs class mirrors `AnalyzePromptInputs` but replaces `insight_refs` with `diagnostic_hits` (list of `{symptom, doc_id, chunk_id, rank}`).
- Output sections are `## Executive summary`, `## Symptom`, `## Causes (ranked)`, `## Assumptions & caveats`, `## Evidence`.
- Persona and grounding injection are IDENTICAL in structure to analyze (copy the helper; do not refactor into a shared base in this task — a shared base is Task 5's responsibility once all three are in flight).

Keep the implementation under ~45 lines — the same size and style as the analyze builder. Notes inside the docstring:

```
# Diagnose differs from analyze in three ways:
#   1. Section names (Symptom + Causes instead of Patterns)
#   2. Diagnostic hits replace insight refs in the pre-reasoning block
#   3. Soft output target default is higher (1500 vs 1200) — adapter-tunable
# Everything else — persona, grounding, streaming-first executive summary,
# evidence block with inline [doc_id:chunk_id] — is intentionally parallel.
```

- [ ] **Step 3: Run tests and commit**

```bash
pytest tests/generation/test_prompts_rich.py -v
git add src/generation/prompts.py tests/generation/test_prompts_rich.py
git commit -m "phase4(sme-prompts): diagnose rich template mirroring analyze"
```

---

## Task 5: `prompts.py` rich template — `recommend` intent (structure + notes) + shared shape builder

**Files:**
- Modify: `src/generation/prompts.py`
- Modify: `tests/generation/test_prompts_rich.py` (append recommend tests)

`recommend` is the third template. It also introduces the shared shape assembler used by all three — small enough that the refactor is safe now that we have three concrete callers.

- [ ] **Step 1: Append recommend tests**

```python
from src.generation.prompts import (
    RecommendPromptInputs,
    build_recommend_rich_prompt,
)


def _rec_inputs(**overrides):
    base = dict(
        query_text="What should we do to improve Q4 margins?",
        persona_role="CFO-advisor",
        persona_voice="decisive, quantitative, risk-aware",
        grounding_rules=(
            "Every recommendation must cite a Recommendation Bank entry "
            "or an exposed reasoning chain. No speculation.",
        ),
        pack_tokens=3200,
        output_cap_tokens=1000,
        evidence_items=[
            {"doc_id": "q3_pl", "chunk_id": "c2", "text": "COGS 62% of revenue."},
        ],
        bank_entries=[
            {"recommendation": "Renegotiate top-3 vendor contracts",
             "rationale": "COGS concentration in top 3",
             "evidence": ["q3_pl:c2"],
             "estimated_impact": "margin +1.2–1.8pp",
             "assumptions": ["current volume holds"],
             "confidence": 0.74},
        ],
        domain="finance",
    )
    base.update(overrides)
    return RecommendPromptInputs(**base)


def test_recommend_prompt_lists_bank_entries_with_impact():
    p = build_recommend_rich_prompt(_rec_inputs())
    assert "Renegotiate top-3 vendor contracts" in p
    assert "margin +1.2" in p


def test_recommend_prompt_skeleton_has_five_sections():
    p = build_recommend_rich_prompt(_rec_inputs())
    for s in [
        "## Executive summary",
        "## Recommendations",
        "## Rationale & evidence",
        "## Assumptions & caveats",
        "## Evidence",
    ]:
        assert s in p


def test_recommend_prompt_forbids_speculation():
    p = build_recommend_rich_prompt(_rec_inputs())
    assert "Recommendation Bank" in p
```

- [ ] **Step 2: Introduce the shared shape assembler + the recommend builder**

Refactor: extract the common persona/grounding/evidence rendering into a private helper `_render_common_blocks(persona_role, persona_voice, grounding_rules, query_text, evidence_items, domain) -> dict[str, str]`. All three builders consume it. Keep the helper under ~40 lines.

Recommend builder specifics (the parts that are NOT shared):

- Inputs class `RecommendPromptInputs` carries `bank_entries: Sequence[dict]` in place of `insight_refs` / `diagnostic_hits`.
- Renders each bank entry as a compact block: recommendation / rationale / estimated_impact / assumptions / confidence.
- Adds the explicit `Recommendation Bank or exposed reasoning chain` phrase to the header above `## Recommendations` — this exact phrase is load-bearing because Task 9's grounding post-pass searches for it when deciding which claims to validate.
- Output sections: `## Executive summary`, `## Recommendations`, `## Rationale & evidence`, `## Assumptions & caveats`, `## Evidence`.

Notes (in docstring):

```
# Recommend differs from analyze / diagnose:
#   1. Recommendations are pulled from the Recommendation Bank (Layer C
#      artifact) — the pre-reasoning block IS the grounding anchor.
#   2. The post-generation recommendation_grounding pass (Task 9) runs
#      AFTER this template produces a response. Any recommendation the
#      pass cannot tie to a bank entry or cited chunk is dropped and
#      replaced by a candid note. This template does not need to reject
#      ungrounded claims itself — the post-pass is the enforcement point.
#   3. Every bank entry is injected verbatim so the LLM has no reason
#      to re-generate wording — it should quote and cite.
```

- [ ] **Step 3: Run tests and commit**

```bash
pytest tests/generation/test_prompts_rich.py -v
git add src/generation/prompts.py tests/generation/test_prompts_rich.py
git commit -m "phase4(sme-prompts): recommend rich template + shared shape assembler"
```

---

## Task 6: Shape-resolution — compact override / auto-intent / honest-compact fallback (load-bearing)

**Files:**
- Modify: `src/generation/prompts.py`
- Create: `tests/generation/test_shape_resolution.py`

The shape resolver decides whether a given query gets the rich skeleton, the compact skeleton, or the honest-compact fallback (rich skeleton degraded because the pack is too thin). It is the single choke point — every prompt-building call path in the agent goes through `resolve_response_shape` before selecting a template.

- [ ] **Step 1: Write the failing tests**

Create `tests/generation/test_shape_resolution.py`:

```python
"""Tests for shape resolution."""
import pytest

from src.generation.prompts import (
    PackSummary,
    ResponseShape,
    resolve_response_shape,
)
from src.serving.model_router import FormatHint


def _pack(has_sme_artifacts=True, total_chunks=12, distinct_docs=3):
    return PackSummary(
        total_chunks=total_chunks,
        distinct_docs=distinct_docs,
        has_sme_artifacts=has_sme_artifacts,
    )


def _packed_item(text, *, artifact_type=None, provenance=(("d1", "c1"),),
                 sme_backed=False):
    """Builds a PackedItem for from_packed_items factory tests."""
    from src.retrieval.types import PackedItem
    return PackedItem(
        text=text,
        provenance=list(provenance),
        layer="a",
        confidence=0.9,
        rerank_score=0.8,
        sme_backed=sme_backed,
        metadata={"artifact_type": artifact_type} if artifact_type else {},
    )


def test_from_packed_items_splits_by_artifact_type():
    # ERRATA §10 — canonical factory covers bank / evidence / insights.
    items = [
        _packed_item("Q3 revenue $5.3M"),
        _packed_item("QoQ growth 14%", artifact_type="insight", sme_backed=True),
        _packed_item("Renegotiate top-3 vendor contracts",
                     artifact_type="recommendation",
                     provenance=(("q3_pl", "c2"),), sme_backed=True),
    ]
    summary = PackSummary.from_packed_items(items)
    assert summary.total_chunks == 3
    assert summary.distinct_docs == 2
    assert summary.has_sme_artifacts is True
    assert len(summary.bank_entries) == 1
    assert summary.bank_entries[0]["recommendation"] == (
        "Renegotiate top-3 vendor contracts"
    )
    assert summary.bank_entries[0]["evidence"] == ["q3_pl:c2"]
    assert len(summary.insights) == 1
    assert summary.insights[0].text == "QoQ growth 14%"
    # Every item with provenance lands in evidence_items
    assert len(summary.evidence_items) == 3


def test_from_packed_items_empty_pack():
    summary = PackSummary.from_packed_items([])
    assert summary.total_chunks == 0
    assert summary.distinct_docs == 0
    assert summary.has_sme_artifacts is False
    assert summary.bank_entries == ()
    assert summary.evidence_items == ()
    assert summary.insights == ()


def test_from_packed_items_sme_backed_without_artifact_type():
    # A Layer-C-overlap item may have sme_backed=True without an artifact_type.
    items = [_packed_item("generic chunk", sme_backed=True)]
    summary = PackSummary.from_packed_items(items)
    assert summary.has_sme_artifacts is True
    assert summary.bank_entries == ()
    assert summary.insights == ()


def test_compact_override_always_wins():
    shape = resolve_response_shape(
        intent="analyze",
        format_hint=FormatHint.COMPACT,
        pack=_pack(),
        enable_rich_mode=True,
    )
    assert shape is ResponseShape.COMPACT


def test_rich_mode_disabled_forces_compact():
    shape = resolve_response_shape(
        intent="analyze",
        format_hint=FormatHint.AUTO,
        pack=_pack(),
        enable_rich_mode=False,
    )
    assert shape is ResponseShape.COMPACT


def test_trivial_intent_gets_compact_even_with_rich_enabled():
    shape = resolve_response_shape(
        intent="lookup",
        format_hint=FormatHint.AUTO,
        pack=_pack(),
        enable_rich_mode=True,
    )
    assert shape is ResponseShape.COMPACT


def test_analytical_intent_gets_rich_when_artifacts_present():
    shape = resolve_response_shape(
        intent="analyze",
        format_hint=FormatHint.AUTO,
        pack=_pack(has_sme_artifacts=True),
        enable_rich_mode=True,
    )
    assert shape is ResponseShape.RICH


def test_analytical_intent_falls_back_to_honest_compact_when_pack_is_thin():
    shape = resolve_response_shape(
        intent="analyze",
        format_hint=FormatHint.AUTO,
        pack=_pack(has_sme_artifacts=False, total_chunks=2, distinct_docs=1),
        enable_rich_mode=True,
    )
    assert shape is ResponseShape.HONEST_COMPACT


def test_borderline_intent_is_rich_when_artifacts_contributed():
    shape = resolve_response_shape(
        intent="compare",
        format_hint=FormatHint.AUTO,
        pack=_pack(has_sme_artifacts=True),
        enable_rich_mode=True,
    )
    assert shape is ResponseShape.RICH


def test_borderline_intent_is_honest_compact_when_artifacts_missing():
    shape = resolve_response_shape(
        intent="compare",
        format_hint=FormatHint.AUTO,
        pack=_pack(has_sme_artifacts=False, distinct_docs=2),
        enable_rich_mode=True,
    )
    assert shape is ResponseShape.HONEST_COMPACT


def test_explicit_rich_hint_overrides_thin_pack():
    # If the user explicitly said "rich", respect it even with a thin pack —
    # but the honest-compact fallback still applies for borderline intents.
    shape = resolve_response_shape(
        intent="analyze",
        format_hint=FormatHint.RICH,
        pack=_pack(has_sme_artifacts=False, total_chunks=2, distinct_docs=1),
        enable_rich_mode=True,
    )
    assert shape is ResponseShape.RICH
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
pytest tests/generation/test_shape_resolution.py -v
```

Expected: fail — symbols absent.

- [ ] **Step 3: Implement the resolver (full code, load-bearing)**

Append to `src/generation/prompts.py`:

```python
# src/generation/prompts.py — shape resolution (load-bearing)
import enum
from dataclasses import dataclass

from src.retrieval.types import PackedItem
from src.serving.model_router import FormatHint


class ResponseShape(str, enum.Enum):
    COMPACT = "compact"
    RICH = "rich"
    HONEST_COMPACT = "honest_compact"


@dataclass(frozen=True)
class PackSummary:
    """ERRATA §10 — canonical PackSummary shape.

    `bank_entries` / `evidence_items` / `insights` are derived views over the
    underlying `PackedItem` list that Task 8 and Task 9 read directly. The
    factory `from_packed_items` is the single construction path; callers
    never instantiate PackSummary by hand.
    """
    total_chunks: int
    distinct_docs: int
    has_sme_artifacts: bool
    bank_entries: tuple[dict, ...] = ()
    evidence_items: tuple[PackedItem, ...] = ()
    insights: tuple[PackedItem, ...] = ()

    @classmethod
    def from_packed_items(cls, items: list[PackedItem]) -> "PackSummary":
        """Build PackSummary from a post-assembly PackedItem list (Phase 3).

        Filters by `metadata["artifact_type"]`:
          - `"recommendation"` → emitted as plain-dict `bank_entries`
            (Task 9 rewrites operate on dicts, not dataclasses).
          - `"insight"` → kept as `PackedItem` for inline insight refs.
          - everything with `provenance` populated contributes to
            `evidence_items` — what Task 8 passes into rich templates.
        """
        evidence: list[PackedItem] = []
        insights: list[PackedItem] = []
        bank: list[dict] = []
        docs: set[str] = set()
        has_sme = False
        for it in items:
            artifact = it.metadata.get("artifact_type") if it.metadata else None
            if it.sme_backed or artifact in {
                "dossier", "insight", "comparative", "recommendation",
            }:
                has_sme = True
            if it.provenance:
                evidence.append(it)
                for doc_id, _ in it.provenance:
                    docs.add(doc_id)
            if artifact == "insight":
                insights.append(it)
            if artifact == "recommendation":
                bank.append({
                    "recommendation": it.text,
                    "evidence": [f"{d}:{c}" for d, c in it.provenance],
                    "metadata": dict(it.metadata or {}),
                })
        return cls(
            total_chunks=len(items),
            distinct_docs=len(docs),
            has_sme_artifacts=has_sme,
            bank_entries=tuple(bank),
            evidence_items=tuple(evidence),
            insights=tuple(insights),
        )


_TRIVIAL_INTENTS: frozenset[str] = frozenset({
    "greeting", "identity", "lookup", "count",
})

_ANALYTICAL_INTENTS: frozenset[str] = frozenset({
    "analyze", "diagnose", "recommend", "investigate",
})

_BORDERLINE_INTENTS: frozenset[str] = frozenset({
    "compare", "summarize", "aggregate", "list", "overview",
})


def resolve_response_shape(
    *,
    intent: str,
    format_hint: FormatHint,
    pack: PackSummary,
    enable_rich_mode: bool,
) -> ResponseShape:
    """Single choke point for rich vs compact vs honest-compact.

    Precedence (highest first):
      1. Explicit compact override from user    → COMPACT
      2. enable_rich_mode flag OFF              → COMPACT
      3. Explicit rich override from user       → RICH (honors thin pack)
      4. Trivial intent                         → COMPACT
      5. Analytical intent + thin pack          → HONEST_COMPACT
      6. Analytical intent + adequate pack      → RICH
      7. Borderline intent + SME artifacts      → RICH
      8. Borderline intent + no artifacts       → HONEST_COMPACT
      9. Anything else                          → COMPACT (safe default)
    """
    if format_hint is FormatHint.COMPACT:
        return ResponseShape.COMPACT
    if not enable_rich_mode:
        return ResponseShape.COMPACT
    if format_hint is FormatHint.RICH:
        return ResponseShape.RICH
    if intent in _TRIVIAL_INTENTS:
        return ResponseShape.COMPACT
    if intent in _ANALYTICAL_INTENTS:
        if not pack.has_sme_artifacts and pack.total_chunks < 4:
            return ResponseShape.HONEST_COMPACT
        return ResponseShape.RICH
    if intent in _BORDERLINE_INTENTS:
        if pack.has_sme_artifacts:
            return ResponseShape.RICH
        return ResponseShape.HONEST_COMPACT
    return ResponseShape.COMPACT
```

Honest-compact note: a separate rendering helper `build_honest_compact_prompt(...)` wraps the existing compact template with a prefix "The available evidence is limited; the following answer is necessarily compact:" — this preserves the visible intent for the user without silently producing a thin rich answer. Include this helper alongside the resolver, under ~15 lines.

- [ ] **Step 4: Run tests and commit**

```bash
pytest tests/generation/test_shape_resolution.py -v
git add src/generation/prompts.py tests/generation/test_shape_resolution.py
git commit -m "phase4(sme-shape): shape resolution with honest-compact fallback"
```

---

## Task 7: Persona injection from adapter YAML (load-bearing)

**Files:**
- Modify: `src/generation/prompts.py`
- Create: `tests/generation/test_persona_injection.py`
- Modify: `deploy/sme_adapters/defaults/global/finance.yaml` and peers — fill `response_persona_prompts` fields
- Create: `deploy/sme_adapters/defaults/global/prompts/rich_analyze.md`, `rich_diagnose.md`, `rich_recommend.md`

Persona injection is the hook where the adapter's persona, voice, and rich-mode prompt templates land inside the prompt string. It runs at prompt-build time, not at adapter-load time — the `Adapter` object is cached; per-request work is just pulling fields out of it and formatting.

- [ ] **Step 1: Write the failing tests**

Create `tests/generation/test_persona_injection.py`:

```python
"""Tests for adapter-driven persona injection."""
from types import SimpleNamespace

import pytest

from src.generation.prompts import (
    PersonaBundle,
    persona_bundle_from_adapter,
)


def _adapter(
    role="senior financial analyst",
    voice="direct, quantitative",
    rules=("cite every claim",),
    intent_templates=None,
):
    return SimpleNamespace(
        persona=SimpleNamespace(
            role=role, voice=voice, grounding_rules=list(rules),
        ),
        response_persona_prompts=SimpleNamespace(
            analyze=(intent_templates or {}).get("analyze", ""),
            diagnose=(intent_templates or {}).get("diagnose", ""),
            recommend=(intent_templates or {}).get("recommend", ""),
        ),
        version="1.2.0",
        content_hash="deadbeef",
    )


def test_persona_bundle_carries_role_and_voice():
    b = persona_bundle_from_adapter(_adapter(), intent="analyze")
    assert b.role == "senior financial analyst"
    assert b.voice == "direct, quantitative"


def test_persona_bundle_tags_adapter_version():
    b = persona_bundle_from_adapter(_adapter(), intent="analyze")
    assert b.adapter_version == "1.2.0"
    assert b.adapter_content_hash == "deadbeef"


def test_persona_bundle_inherits_empty_template_from_generic():
    # If analyze template is blank, the resolver returns the empty string —
    # the caller (Task 8) falls back to the generic global template.
    b = persona_bundle_from_adapter(_adapter(intent_templates={}), intent="analyze")
    assert b.intent_template_body == ""


def test_persona_bundle_raises_on_unknown_intent():
    with pytest.raises(ValueError):
        persona_bundle_from_adapter(_adapter(), intent="telepathy")


def test_persona_bundle_respects_all_three_new_intents():
    adapter = _adapter(intent_templates={
        "analyze": "ANALYZE BODY",
        "diagnose": "DIAGNOSE BODY",
        "recommend": "RECOMMEND BODY",
    })
    for intent, body in [
        ("analyze", "ANALYZE BODY"),
        ("diagnose", "DIAGNOSE BODY"),
        ("recommend", "RECOMMEND BODY"),
    ]:
        assert persona_bundle_from_adapter(adapter, intent=intent).intent_template_body == body
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
pytest tests/generation/test_persona_injection.py -v
```

Expected: fail — `PersonaBundle` / `persona_bundle_from_adapter` absent.

- [ ] **Step 3: Implement the injection helper (full code, load-bearing)**

Append to `src/generation/prompts.py`:

```python
# src/generation/prompts.py — persona injection (load-bearing)
from dataclasses import dataclass

_RICH_INTENTS: frozenset[str] = frozenset({"analyze", "diagnose", "recommend"})


@dataclass(frozen=True)
class PersonaBundle:
    role: str
    voice: str
    grounding_rules: tuple[str, ...]
    intent_template_body: str
    adapter_version: str
    adapter_content_hash: str


def persona_bundle_from_adapter(adapter, *, intent: str) -> PersonaBundle:
    """Assemble a PersonaBundle for one intent from a loaded Adapter.

    `adapter` is the Phase 1 AdapterLoader output; its structure is taken
    as read — we never mutate it, and we never re-fetch from Blob here.
    Empty intent_template_body signals "no per-domain override, use the
    global rich_{intent}.md template" — the caller handles that fallback
    in Task 8. Per ERRATA §1, `content_hash` + `version` are direct
    attributes on the Adapter instance (populated at load time); we do
    NOT call `last_load_metadata()` to obtain them.
    """
    if intent not in _RICH_INTENTS:
        raise ValueError(
            f"persona_bundle_from_adapter: unsupported intent {intent!r}; "
            f"rich mode applies only to {sorted(_RICH_INTENTS)}"
        )
    body = getattr(adapter.response_persona_prompts, intent, "") or ""
    return PersonaBundle(
        role=adapter.persona.role,
        voice=adapter.persona.voice,
        grounding_rules=tuple(adapter.persona.grounding_rules),
        intent_template_body=body,
        adapter_version=adapter.version,
        adapter_content_hash=adapter.content_hash,
    )
```

- [ ] **Step 4: Fill the default adapter YAMLs**

For each YAML in `deploy/sme_adapters/defaults/global/{finance,legal,hr,medical,it_support,generic}.yaml`, set the `persona` block and `response_persona_prompts` pointers. Sample for `finance.yaml`:

```yaml
persona:
  role: "senior financial analyst advising the C-suite"
  voice: "direct, quantitative, hedged"
  grounding_rules:
    - "Cite every claim to [doc_id:chunk_id]."
    - "Refuse to extrapolate without evidence."
    - "Flag any item whose confidence is below 0.7."
response_persona_prompts:
  analyze:  "prompts/finance_analyze.md"
  diagnose: "prompts/finance_diagnose.md"
  recommend: "prompts/finance_recommend.md"
```

The per-domain prompt markdown files under `prompts/` are human-authored domain flavoring that is appended to the skeleton by Task 8. They are NOT full templates — they are persona-specific adjustments like "Always call out quarter-over-quarter deltas in the Executive summary".

Write the global fallback templates under `prompts/rich_{analyze,diagnose,recommend}.md` — each 30–60 lines of prose domain-neutral guidance.

- [ ] **Step 5: Validate YAML round-trips through the Phase 1 loader**

```bash
python -c "
import yaml
for d in ['finance','legal','hr','medical','it_support','generic']:
    with open(f'deploy/sme_adapters/defaults/global/{d}.yaml') as f:
        y = yaml.safe_load(f)
    assert y['persona']['role'], f'{d} persona.role missing'
    assert y['response_persona_prompts']['analyze'], f'{d} analyze prompt missing'
print('YAML round-trip OK')
"
```

- [ ] **Step 6: Run tests and commit**

```bash
pytest tests/generation/test_persona_injection.py -v
git add src/generation/prompts.py tests/generation/test_persona_injection.py \
        deploy/sme_adapters/defaults/global/
git commit -m "phase4(sme-persona): adapter persona injection + default YAML fills"
```

---

## Task 8: `core_agent.py` — wire adapter resolver, pass persona + output_caps

**Files:**
- Modify: `src/agent/core_agent.py`
- Create: `tests/agent/test_core_agent_rich_wire.py`

This is the single integration seam where classifier output + adapter + pack summary + flag state converge to pick a shape and build a prompt. Kept as thin as possible because `generator.py` stays out of formatting — all construction happens through `prompts.py`.

- [ ] **Step 1: Write the failing tests**

Create `tests/agent/test_core_agent_rich_wire.py`:

```python
"""Tests for core_agent rich-mode wiring."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent.core_agent import CoreAgent
from src.serving.model_router import ClassifiedQuery, FormatHint


def _stub_adapter():
    return SimpleNamespace(
        persona=SimpleNamespace(
            role="senior financial analyst",
            voice="direct, quantitative",
            grounding_rules=["cite every claim"],
        ),
        response_persona_prompts=SimpleNamespace(
            analyze="ANALYZE BODY", diagnose="", recommend="",
        ),
        output_caps=SimpleNamespace(analyze=1200, diagnose=1500, recommend=1000),
        retrieval_caps=SimpleNamespace(max_pack_tokens={"analyze": 6000}),
        version="1.2.0", content_hash="abc",
    )


@pytest.mark.asyncio
async def test_rich_mode_off_uses_compact_path():
    agent = CoreAgent()
    classified = ClassifiedQuery(
        query_text="Analyze Q3 trends.",
        intent="analyze", format_hint=FormatHint.AUTO,
        entities=[], urls=[],
    )
    with patch("src.agent.core_agent._load_adapter",
               new=AsyncMock(return_value=_stub_adapter())), \
         patch("src.agent.core_agent._is_rich_mode_enabled",
               new=AsyncMock(return_value=False)), \
         patch("src.agent.core_agent.build_analyze_rich_prompt") as rich_mock:
        await agent._build_prompt_for_test(
            classified=classified,
            pack_summary=MagicMock(has_sme_artifacts=True,
                                   total_chunks=10, distinct_docs=3,
                                   evidence_items=(), insights=(),
                                   bank_entries=()),
            subscription_id="s", profile_id="p",
        )
    rich_mock.assert_not_called()


@pytest.mark.asyncio
async def test_rich_mode_on_analyze_builds_rich_prompt():
    agent = CoreAgent()
    classified = ClassifiedQuery(
        query_text="Analyze Q3 revenue trends across quarters.",
        intent="analyze", format_hint=FormatHint.AUTO,
        entities=[], urls=[],
    )
    with patch("src.agent.core_agent._load_adapter",
               new=AsyncMock(return_value=_stub_adapter())), \
         patch("src.agent.core_agent._is_rich_mode_enabled",
               new=AsyncMock(return_value=True)), \
         patch("src.agent.core_agent.build_analyze_rich_prompt",
               return_value="RICH_PROMPT") as rich_mock:
        prompt = await agent._build_prompt_for_test(
            classified=classified,
            pack_summary=MagicMock(has_sme_artifacts=True,
                                   total_chunks=10, distinct_docs=3,
                                   evidence_items=(), insights=(),
                                   bank_entries=()),
            subscription_id="s", profile_id="p",
        )
    assert prompt == "RICH_PROMPT"
    rich_mock.assert_called_once()
    inputs = rich_mock.call_args.args[0]
    assert inputs.persona_role == "senior financial analyst"
    assert inputs.output_cap_tokens == 1200
    assert inputs.query_text == (
        "Analyze Q3 revenue trends across quarters."
    )


@pytest.mark.asyncio
async def test_compact_override_bypasses_rich_even_with_flag_on():
    agent = CoreAgent()
    classified = ClassifiedQuery(
        query_text="Keep it short please.",
        intent="analyze", format_hint=FormatHint.COMPACT,
        entities=[], urls=[],
    )
    with patch("src.agent.core_agent._load_adapter",
               new=AsyncMock(return_value=_stub_adapter())), \
         patch("src.agent.core_agent._is_rich_mode_enabled",
               new=AsyncMock(return_value=True)), \
         patch("src.agent.core_agent.build_analyze_rich_prompt") as rich_mock:
        await agent._build_prompt_for_test(
            classified=classified,
            pack_summary=MagicMock(has_sme_artifacts=True,
                                   total_chunks=10, distinct_docs=3,
                                   evidence_items=(), insights=(),
                                   bank_entries=()),
            subscription_id="s", profile_id="p",
        )
    rich_mock.assert_not_called()


@pytest.mark.asyncio
async def test_honest_compact_is_used_when_pack_is_thin():
    agent = CoreAgent()
    classified = ClassifiedQuery(
        query_text="Analyze with thin evidence.",
        intent="analyze", format_hint=FormatHint.AUTO,
        entities=[], urls=[],
    )
    with patch("src.agent.core_agent._load_adapter",
               new=AsyncMock(return_value=_stub_adapter())), \
         patch("src.agent.core_agent._is_rich_mode_enabled",
               new=AsyncMock(return_value=True)), \
         patch("src.agent.core_agent.build_honest_compact_prompt",
               return_value="HONEST_COMPACT_PROMPT") as hc_mock, \
         patch("src.agent.core_agent.build_analyze_rich_prompt") as rich_mock:
        prompt = await agent._build_prompt_for_test(
            classified=classified,
            pack_summary=MagicMock(has_sme_artifacts=False,
                                   total_chunks=2, distinct_docs=1,
                                   evidence_items=(), insights=(),
                                   bank_entries=()),
            subscription_id="s", profile_id="p",
        )
    assert prompt == "HONEST_COMPACT_PROMPT"
    hc_mock.assert_called_once()
    rich_mock.assert_not_called()
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
pytest tests/agent/test_core_agent_rich_wire.py -v
```

Expected: fail — wiring absent.

- [ ] **Step 3: Implement the wiring**

Modify `src/agent/core_agent.py`. The minimal change:

```python
# src/agent/core_agent.py — additions; existing handle() structure preserved
from src.config.feature_flags import ENABLE_RICH_MODE, get_flag_resolver
from src.generation.prompts import (
    AnalyzePromptInputs,
    DiagnosePromptInputs,
    RecommendPromptInputs,
    PackSummary,
    ResponseShape,
    build_analyze_rich_prompt,
    build_diagnose_rich_prompt,
    build_recommend_rich_prompt,
    build_honest_compact_prompt,
    persona_bundle_from_adapter,
    resolve_response_shape,
)
from src.intelligence.sme.adapter_loader import get_adapter_loader


async def _load_adapter(subscription_id: str, profile_domain: str):
    """Thin wrapper — kept as module-level so tests can patch it."""
    return get_adapter_loader().load(subscription_id, profile_domain)


async def _is_rich_mode_enabled(subscription_id: str) -> bool:
    """Thin wrapper — tests patch this to control the flag per call.

    Delegates to Phase 1's canonical SMEFeatureFlags (ERRATA §4). Default
    OFF is encoded in `feature_flags._DEFAULTS`; no duplicate handling here.
    """
    return get_flag_resolver().is_enabled(subscription_id, ENABLE_RICH_MODE)


class CoreAgent:
    # existing fields + handle() preserved

    async def _build_prompt_for_test(
        self, *, classified, pack_summary, subscription_id, profile_id,
    ) -> str:
        """Test seam — exercised by tests/agent/test_core_agent_rich_wire.py.
        Real handle() calls the same resolver inline; this helper exists so
        the shape / persona / template wiring can be covered in isolation.
        """
        enable_rich = await _is_rich_mode_enabled(subscription_id)
        shape = resolve_response_shape(
            intent=classified.intent,
            format_hint=classified.format_hint,
            pack=pack_summary,
            enable_rich_mode=enable_rich,
        )
        if shape is ResponseShape.COMPACT:
            return self._build_compact_prompt(classified, pack_summary)
        domain = await self._resolve_profile_domain(subscription_id, profile_id)
        adapter = await _load_adapter(subscription_id, domain)
        if shape is ResponseShape.HONEST_COMPACT:
            return build_honest_compact_prompt(
                query_text=classified.query_text,
                pack_summary=pack_summary,
            )
        # Investigate is spec §8 analyze-like; reuse the analyze persona lookup
        # so rich-shaped investigate falls into the analyze template.
        template_intent = (
            "analyze" if classified.intent in ("investigate", "overview")
            else classified.intent
        )
        persona = persona_bundle_from_adapter(adapter, intent=template_intent)
        cap = getattr(adapter.output_caps, template_intent, 1200)
        pack_tokens = adapter.retrieval_caps.max_pack_tokens.get(
            template_intent, 6000
        )
        # ERRATA §10: PackSummary already exposes these tuples directly;
        # no _evidence_items_from_pack / _insights_from_pack helpers needed.
        evidence = [
            {"doc_id": p[0][0], "chunk_id": p[0][1], "text": it.text}
            for it in pack_summary.evidence_items
            for p in [it.provenance] if p
        ]
        insights = [
            {"type": it.metadata.get("insight_type", "insight"),
             "narrative": it.text}
            for it in pack_summary.insights
        ]
        if template_intent == "analyze":
            return build_analyze_rich_prompt(AnalyzePromptInputs(
                query_text=classified.query_text,
                persona_role=persona.role, persona_voice=persona.voice,
                grounding_rules=persona.grounding_rules,
                pack_tokens=pack_tokens, output_cap_tokens=cap,
                evidence_items=evidence, insight_refs=insights,
                domain=domain,
            ))
        if template_intent == "diagnose":
            # Diagnose template maps `insights` to `diagnostic_hits`
            # (Task 4). Each hit needs symptom + doc/chunk + rank.
            hits = [
                {"symptom": it.metadata.get("symptom", it.text[:120]),
                 "doc_id": it.provenance[0][0] if it.provenance else "",
                 "chunk_id": it.provenance[0][1] if it.provenance else "",
                 "rank": i + 1}
                for i, it in enumerate(pack_summary.insights)
            ]
            return build_diagnose_rich_prompt(DiagnosePromptInputs(
                query_text=classified.query_text,
                persona_role=persona.role, persona_voice=persona.voice,
                grounding_rules=persona.grounding_rules,
                pack_tokens=pack_tokens, output_cap_tokens=cap,
                evidence_items=evidence, diagnostic_hits=hits,
                domain=domain,
            ))
        if template_intent == "recommend":
            return build_recommend_rich_prompt(RecommendPromptInputs(
                query_text=classified.query_text,
                persona_role=persona.role, persona_voice=persona.voice,
                grounding_rules=persona.grounding_rules,
                pack_tokens=pack_tokens, output_cap_tokens=cap,
                evidence_items=evidence,
                bank_entries=list(pack_summary.bank_entries),
                domain=domain,
            ))
        return self._build_compact_prompt(classified, pack_summary)
```

Notes:
- `investigate` and `overview` are routed through the `analyze` template per
  spec §8 ("overview/investigate are treated as analyze-like"). This closes
  the fall-through where a RICH-shaped `investigate` previously reached the
  compact default.
- Evidence / insight / bank extraction reads `pack_summary.evidence_items`,
  `.insights`, and `.bank_entries` directly per ERRATA §10 — no private
  helper methods on `CoreAgent`.
- The flag resolver wrapper delegates to `SMEFeatureFlags.is_enabled` with
  `ENABLE_RICH_MODE`; default-OFF is Phase 1's responsibility.

`handle()` itself gains four lines: pack_summary assembly (already exists in
Phase 3 via `PackSummary.from_packed_items`), the `enable_rich` lookup, the
shape dispatch, and the adapter load. Nothing else in `handle()` moves.

- [ ] **Step 4: Run tests and commit**

```bash
pytest tests/agent/test_core_agent_rich_wire.py -v
git add src/agent/core_agent.py tests/agent/test_core_agent_rich_wire.py
git commit -m "phase4(sme-agent): wire adapter resolver + shape dispatch in core_agent"
```

---

## Task 9: Recommendation grounding post-pass (load-bearing for 0.0 hallucination)

**Files:**
- Create: `src/generation/recommendation_grounding.py`
- Create: `tests/generation/test_recommendation_grounding.py`
- Modify: `src/agent/core_agent.py` — call the post-pass only when intent == "recommend" and response is rich

Every recommendation must trace to either a Recommendation Bank entry or exposed reasoning in the response. The post-pass scans the LLM output, maps each recommendation sentence to the bank entries the prompt injected, and drops any recommendation it cannot tie to bank OR to an inline `[doc_id:chunk_id]` citation. Dropped claims are replaced by a candid "Note: {N} claim(s) could not be verified" appendix.

- [ ] **Step 1: Write the failing tests**

Create `tests/generation/test_recommendation_grounding.py`:

```python
"""Tests for recommendation grounding post-pass."""
from src.generation.recommendation_grounding import (
    GroundingReport,
    enforce_recommendation_grounding,
)


def _response(body: str) -> str:
    return (
        "## Executive summary\nSome summary.\n\n"
        "## Recommendations\n"
        f"{body}\n\n"
        "## Rationale & evidence\nSome evidence.\n\n"
        "## Assumptions & caveats\nSome caveats.\n\n"
        "## Evidence\n- q3_pl:c2\n"
    )


_BANK = [
    {"recommendation": "Renegotiate top-3 vendor contracts",
     "evidence": ["q3_pl:c2"]},
    {"recommendation": "Freeze hiring in non-revenue roles",
     "evidence": ["q3_hr:c5"]},
]


def test_passes_grounded_recommendations_unchanged():
    body = (
        "1. Renegotiate top-3 vendor contracts [q3_pl:c2].\n"
        "2. Freeze hiring in non-revenue roles [q3_hr:c5].\n"
    )
    resp = _response(body)
    out, report = enforce_recommendation_grounding(resp, bank_entries=_BANK)
    assert "Renegotiate" in out
    assert "Freeze hiring" in out
    assert report.dropped_count == 0


def test_drops_ungrounded_recommendation_and_appends_note():
    body = (
        "1. Renegotiate top-3 vendor contracts [q3_pl:c2].\n"
        "2. Launch a new business unit in EMEA.\n"   # not in bank, no citation
    )
    resp = _response(body)
    out, report = enforce_recommendation_grounding(resp, bank_entries=_BANK)
    assert "Launch a new business unit" not in out
    assert "Renegotiate" in out
    assert report.dropped_count == 1
    assert "Note:" in out
    assert "could not be verified" in out


def test_keeps_recommendation_with_explicit_bank_match_even_without_citation():
    body = (
        "1. Renegotiate top-3 vendor contracts.\n"   # matches bank by text
    )
    resp = _response(body)
    out, report = enforce_recommendation_grounding(resp, bank_entries=_BANK)
    assert "Renegotiate" in out
    assert report.dropped_count == 0


def test_returns_unchanged_when_no_recommendations_section():
    resp = "## Executive summary\nNothing here.\n"
    out, report = enforce_recommendation_grounding(resp, bank_entries=_BANK)
    assert out == resp
    assert report.dropped_count == 0


def test_empty_bank_only_requires_inline_citation():
    body = (
        "1. Do X [doc_a:c1].\n"
        "2. Do Y.\n"  # no citation, no bank → dropped
    )
    resp = _response(body)
    out, report = enforce_recommendation_grounding(resp, bank_entries=[])
    assert "Do X" in out
    assert "Do Y" not in out
    assert report.dropped_count == 1
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
pytest tests/generation/test_recommendation_grounding.py -v
```

Expected: fail — module absent.

- [ ] **Step 3: Implement the post-pass (full code, load-bearing)**

Create `src/generation/recommendation_grounding.py`:

```python
"""Recommendation-intent grounding post-pass.

Runs AFTER an LLM response is produced for a recommend-intent rich prompt.
For every recommendation-section sentence, it requires ONE of:
  (a) a lexical match against a Recommendation Bank entry, OR
  (b) an inline [doc_id:chunk_id] citation.

Failing items are removed. The response gains a candid appendix when any
items are dropped. 0.0 hallucination rate is preserved by refusing to let
an ungrounded recommendation reach the user.

This module is formatting — per the memory rule it lives in the generation
package, NOT in intelligence/. It is imported exactly once, from
src/agent/core_agent.py, only when intent == "recommend" AND shape == RICH.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

_SECTION_HEAD = re.compile(r"^##\s+Recommendations\s*$", re.MULTILINE)
_NEXT_SECTION_HEAD = re.compile(r"^##\s+\S", re.MULTILINE)
_CITATION = re.compile(r"\[[A-Za-z0-9_\-]+:[A-Za-z0-9_\-]+\]")
_ITEM_LINE = re.compile(r"^\s*(?:\d+\.|-|\*)\s+(.+)$", re.MULTILINE)


@dataclass(frozen=True)
class GroundingReport:
    kept_count: int
    dropped_count: int
    dropped_items: tuple[str, ...]


def enforce_recommendation_grounding(
    response: str,
    *,
    bank_entries: Sequence[dict],
) -> tuple[str, GroundingReport]:
    head = _SECTION_HEAD.search(response)
    if head is None:
        return response, GroundingReport(0, 0, ())
    tail_start = head.end()
    nxt = _NEXT_SECTION_HEAD.search(response, pos=tail_start + 1)
    section_end = nxt.start() if nxt else len(response)
    section_body = response[tail_start:section_end]
    bank_signatures = tuple(
        _signature(entry.get("recommendation", "")) for entry in bank_entries
    )
    kept_lines: list[str] = []
    dropped_items: list[str] = []
    for match in _ITEM_LINE.finditer(section_body):
        line = match.group(0)
        claim = match.group(1)
        if _is_grounded(claim, bank_signatures):
            kept_lines.append(line)
        else:
            dropped_items.append(claim.strip())
    new_section = "\n".join(kept_lines).rstrip()
    if new_section:
        new_section += "\n"
    rewritten = response[:tail_start] + "\n" + new_section + response[section_end:]
    if dropped_items:
        rewritten = rewritten.rstrip() + (
            f"\n\nNote: {len(dropped_items)} claim(s) could not be verified "
            f"against profile evidence and were removed.\n"
        )
    return rewritten, GroundingReport(
        kept_count=len(kept_lines),
        dropped_count=len(dropped_items),
        dropped_items=tuple(dropped_items),
    )


def _is_grounded(claim: str, bank_signatures: tuple[str, ...]) -> bool:
    if _CITATION.search(claim):
        return True
    sig = _signature(claim)
    return any(sig_b and sig_b in sig for sig_b in bank_signatures)


def _signature(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).strip()
```

Properties:
- Purely textual — no LLM call.
- No timeouts; the function returns immediately on any input.
- Idempotent when called twice on the same output.
- Preserves all other sections byte-for-byte.

- [ ] **Step 4: Wire the post-pass into `core_agent.py`**

Add, at the point where the rich response text is finalized (after streaming completes, before returning to the caller):

```python
# src/agent/core_agent.py — excerpt
from src.generation.recommendation_grounding import enforce_recommendation_grounding

# ... inside handle() after response text is captured ...
if classified.intent == "recommend" and shape is ResponseShape.RICH:
    response_text, grounding_report = enforce_recommendation_grounding(
        response_text, bank_entries=pack_summary.bank_entries,
    )
    self._trace_grounding_drop(
        subscription_id=subscription_id, profile_id=profile_id,
        query_id=query_id, report=grounding_report,
    )
```

The trace writer (`_trace_grounding_drop`) reuses the Phase 1 trace infrastructure — it only appends to the existing query trace JSONL; it does NOT create a new store. Drops feed pattern mining in Phase 6.

- [ ] **Step 5: Run tests and commit**

```bash
pytest tests/generation/test_recommendation_grounding.py -v
pytest tests/agent/test_core_agent_rich_wire.py -v  # confirm no regression
git add src/generation/recommendation_grounding.py \
        tests/generation/test_recommendation_grounding.py \
        src/agent/core_agent.py
git commit -m "phase4(sme-grounding): recommendation post-pass drops unverifiable claims"
```

---

## Task 10: `enable_rich_mode` flag wiring + admin toggle

**Files:**
- Modify: `src/api/admin_flags.py`
- Create: `tests/config/test_features_rich_flag.py`

Per ERRATA §4 the canonical flag module is Phase 1's `src/config/feature_flags.py`
(class `SMEFeatureFlags`, flag constant `ENABLE_RICH_MODE`, singleton
`get_flag_resolver()`). Phase 4 does NOT create a parallel `src/config/features.py`.
Default-OFF is already encoded in Phase 1's `_DEFAULTS` map; there is no
duplicate default-handling needed here. Phase 4's contribution in Task 10 is:
(a) consumer-side tests showing the master-kill / per-subscription override /
global default semantics via the canonical resolver, and (b) the admin toggle
endpoint that writes through the same `FlagStore`.

- [ ] **Step 1: Write the consumer tests**

Create `tests/config/test_features_rich_flag.py`:

```python
"""Tests for enable_rich_mode flag resolution via the Phase 1 resolver."""
from unittest.mock import MagicMock

import pytest

from src.config.feature_flags import (
    ENABLE_RICH_MODE,
    FlagStore,
    SMEFeatureFlags,
)


def _flags(overrides: dict[str, bool]) -> SMEFeatureFlags:
    store = MagicMock(spec=FlagStore)
    store.get_subscription_overrides.return_value = overrides
    return SMEFeatureFlags(store=store)


def test_defaults_to_false_when_nothing_configured():
    assert _flags({}).is_enabled("sub_any", ENABLE_RICH_MODE) is False


def test_requires_master_flag_on_for_per_sub_override():
    # Per Phase 1 semantics, dependent flags need master ON to surface.
    assert _flags({"enable_rich_mode": True}).is_enabled(
        "sub_a", ENABLE_RICH_MODE,
    ) is False


def test_master_plus_per_sub_returns_true():
    assert _flags({
        "sme_redesign_enabled": True,
        "enable_rich_mode": True,
    }).is_enabled("sub_a", ENABLE_RICH_MODE) is True


def test_master_off_forces_false_even_with_per_sub_on():
    assert _flags({
        "sme_redesign_enabled": False,
        "enable_rich_mode": True,
    }).is_enabled("sub_a", ENABLE_RICH_MODE) is False
```

- [ ] **Step 2: Confirm `src/config/feature_flags.py` already exposes what Phase 4 needs**

Phase 1 already ships:

```python
from src.config.feature_flags import (
    ENABLE_RICH_MODE,        # string constant "enable_rich_mode"
    SMEFeatureFlags,         # resolver class
    get_flag_resolver,       # process-wide singleton accessor
)
```

Phase 4 code paths call `get_flag_resolver().is_enabled(subscription_id, ENABLE_RICH_MODE)`
— see Task 8's `_is_rich_mode_enabled` wrapper. No new resolver code is added
by Phase 4.

- [ ] **Step 3: Expose admin toggle in `src/api/admin_flags.py`**

Add a `PATCH /admin/flags/enable_rich_mode` endpoint that accepts
`{subscription_id, value}` and writes via the Phase 1 `FlagStore` used by
`SMEFeatureFlags`. No new storage — this reuses the Phase 1 flag infrastructure.
Audit-logged via the existing admin-audit middleware.

- [ ] **Step 4: Run tests and commit**

```bash
pytest tests/config/test_features_rich_flag.py -v
git add src/api/admin_flags.py tests/config/test_features_rich_flag.py
git commit -m "phase4(sme-flag): enable_rich_mode admin toggle + consumer tests"
```

---

## Task 11: Integration tests per intent (rich + compact)

**Files:**
- Create: `tests/integration/test_phase4_rich_mode.py`

Integration tests run the full pipeline from a classified query through shape resolution to prompt output, with the adapter loader and flag resolver stubbed. They prove the wiring holds end to end.

- [ ] **Step 1: Write the integration suite**

Create `tests/integration/test_phase4_rich_mode.py`:

```python
"""Phase 4 integration — classifier → shape → adapter → prompt."""
import pytest
from unittest.mock import AsyncMock, patch

from src.agent.core_agent import CoreAgent
from src.serving.model_router import ClassifiedQuery, FormatHint


@pytest.mark.asyncio
@pytest.mark.parametrize("intent", ["analyze", "diagnose", "recommend"])
async def test_rich_mode_on_produces_rich_skeleton(intent):
    agent = CoreAgent()
    classified = ClassifiedQuery(
        query_text=f"Sample {intent} query.",
        intent=intent, format_hint=FormatHint.AUTO,
        entities=[], urls=[],
    )
    with patch("src.agent.core_agent._is_rich_mode_enabled",
               new=AsyncMock(return_value=True)), \
         patch("src.agent.core_agent._load_adapter",
               new=AsyncMock(return_value=_fake_adapter())):
        prompt = await agent._build_prompt_for_test(
            classified=classified,
            pack_summary=_fake_pack(),
            subscription_id="s", profile_id="p",
        )
    assert "## Executive summary" in prompt
    assert "## Assumptions & caveats" in prompt


@pytest.mark.asyncio
async def test_compact_override_bypasses_rich_for_every_new_intent():
    agent = CoreAgent()
    for intent in ("analyze", "diagnose", "recommend"):
        classified = ClassifiedQuery(
            query_text=f"Compact {intent} please.",
            intent=intent, format_hint=FormatHint.COMPACT,
            entities=[], urls=[],
        )
        with patch("src.agent.core_agent._is_rich_mode_enabled",
                   new=AsyncMock(return_value=True)), \
             patch("src.agent.core_agent._load_adapter",
                   new=AsyncMock(return_value=_fake_adapter())):
            prompt = await agent._build_prompt_for_test(
                classified=classified,
                pack_summary=_fake_pack(),
                subscription_id="s", profile_id="p",
            )
        assert "## Executive summary" not in prompt


@pytest.mark.asyncio
async def test_trivial_intent_stays_compact_when_rich_on():
    agent = CoreAgent()
    for intent in ("greeting", "lookup", "count"):
        classified = ClassifiedQuery(
            query_text=f"Trivial {intent} query.",
            intent=intent, format_hint=FormatHint.AUTO,
            entities=[], urls=[],
        )
        with patch("src.agent.core_agent._is_rich_mode_enabled",
                   new=AsyncMock(return_value=True)), \
             patch("src.agent.core_agent._load_adapter",
                   new=AsyncMock(return_value=_fake_adapter())):
            prompt = await agent._build_prompt_for_test(
                classified=classified,
                pack_summary=_fake_pack(),
                subscription_id="s", profile_id="p",
            )
        assert "## Executive summary" not in prompt


@pytest.mark.asyncio
async def test_honest_compact_is_taken_when_pack_is_thin():
    # Analytical intent + thin pack → honest_compact, not rich
    agent = CoreAgent()
    classified = ClassifiedQuery(
        query_text="Analyze trends with scant evidence.",
        intent="analyze", format_hint=FormatHint.AUTO,
        entities=[], urls=[],
    )
    with patch("src.agent.core_agent._is_rich_mode_enabled",
               new=AsyncMock(return_value=True)), \
         patch("src.agent.core_agent._load_adapter",
               new=AsyncMock(return_value=_fake_adapter())):
        prompt = await agent._build_prompt_for_test(
            classified=classified,
            pack_summary=_fake_pack(has_sme_artifacts=False,
                                    total_chunks=1, distinct_docs=1),
            subscription_id="s", profile_id="p",
        )
    assert "necessarily compact" in prompt
```

(Helpers `_fake_adapter` and `_fake_pack` mirror the ones used in Task 7 / 8 — kept in the test file to avoid cross-file drift.)

- [ ] **Step 2: Run the full test suite**

```bash
pytest tests/integration/test_phase4_rich_mode.py -v
pytest tests/generation tests/serving tests/agent tests/config -v  # no regressions
```

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_phase4_rich_mode.py
git commit -m "phase4(sme-integration): classifier→shape→adapter→prompt end-to-end tests"
```

---

## Task 12: Run Phase 0 eval harness → measure all 6 new metrics

**Files:**
- Produce: `tests/sme_metrics_phase4_{YYYY-MM-DD}.json`

This is an operator-run verification, not a code task. The Phase 0 harness (Tasks 1–17 of `2026-04-20-docwain-sme-phase0-baseline.md`) re-runs with the `enable_rich_mode` flag flipped ON for a sandbox subscription; the resulting snapshot is compared to the frozen baseline.

- [ ] **Step 1: Flip the flag ON for the eval subscription**

Use the admin API from Task 10:

```bash
curl -X PATCH "$DOCWAIN_API_URL/admin/flags/enable_rich_mode" \
  -H "Content-Type: application/json" \
  -d '{"subscription_id": "'"$EVAL_SUBSCRIPTION_ID"'", "value": true}'
```

- [ ] **Step 2: Run the baseline harness**

```bash
python -m scripts.sme_eval.run_baseline \
    --eval-dir tests/sme_evalset_v1/queries \
    --fixtures tests/sme_evalset_v1/fixtures/test_profiles.yaml \
    --out tests/sme_metrics_phase4_$(date +%Y-%m-%d).json \
    --results-jsonl tests/sme_results_phase4_$(date +%Y-%m-%d).jsonl
```

- [ ] **Step 3: Compare against baseline**

```bash
python - <<'PY'
import json, glob, os
baseline = sorted(glob.glob("tests/sme_metrics_baseline_*.json"))[-1]
phase4 = sorted(glob.glob("tests/sme_metrics_phase4_*.json"))[-1]
b = json.load(open(baseline)); p = json.load(open(phase4))
print(f"baseline: {baseline}")
print(f"phase 4:  {phase4}")
print(f"faithfulness: {b['ragas']['answer_faithfulness']:.3f} → "
      f"{p['ragas']['answer_faithfulness']:.3f}  (gate ≥ 0.80)")
print(f"hallucination: {b['ragas']['hallucination_rate']:.3f} → "
      f"{p['ragas']['hallucination_rate']:.3f}  (gate = 0.0)")
for k in ("recommendation_groundedness","cross_doc_integration_rate",
          "insight_novelty","sme_persona_consistency",
          "verified_removal_rate","sme_artifact_hit_rate"):
    bv = b['sme_metrics'].get(k,{}).get('value',0.0)
    pv = p['sme_metrics'].get(k,{}).get('value',0.0)
    print(f"{k}: {bv:.3f} → {pv:.3f}")
PY
```

- [ ] **Step 4: Pass thresholds (Section 10)**

Each must hold:
- `answer_faithfulness ≥ 0.80`
- `hallucination_rate = 0.0`
- `context_recall ≥ 0.80`
- `grounding_bypass_rate = 0.0`
- `recommendation_groundedness ≥ 0.95`
- `cross_doc_integration_rate ≥ 0.70`
- `insight_novelty ≥ 0.40`
- `sme_persona_consistency ≥ 4.0 avg (out of 5)`
- `verified_removal_rate ≥ 0.85`
- `sme_artifact_hit_rate ≥ 0.90`

If ANY fails, do NOT toggle the flag wider. Record which metric failed; return to the adapter-tuning loop (Section 13.4 — two tuning iterations before full rollback).

- [ ] **Step 5: Commit the snapshot (pass or fail; the baseline is the record)**

```bash
git add -f tests/sme_metrics_phase4_*.json tests/sme_results_phase4_*.jsonl
git commit -m "phase4(sme-eval): phase 4 metric snapshot — $(date +%Y-%m-%d)"
```

---

## Task 13: Latency regression check (non-blocking flag)

**Files:**
- Extend: `tests/sme_metrics_phase4_{YYYY-MM-DD}.json` (already includes latency)

Rich mode produces longer responses by design. The launch-gate check is: p95 TTFT on complex queries ≤ 1.5 s; p95 total ≤ 15 s. Regression is logged and investigated, but does NOT auto-block per spec Section 10.

- [ ] **Step 1: Extract latency percentiles from the Phase 4 snapshot**

```bash
python - <<'PY'
import json, glob
snap = json.load(open(sorted(glob.glob("tests/sme_metrics_phase4_*.json"))[-1]))
for intent in ("analyze","diagnose","recommend","compare","summarize","lookup"):
    p95 = snap.get("latency_p95_per_intent",{}).get(intent)
    if p95 is None: continue
    status = "OK" if p95 <= 15000 else "INVESTIGATE"
    print(f"p95 total {intent:12s}: {p95:8.0f} ms   {status}")
PY
```

- [ ] **Step 2: Capture TTFT distribution**

TTFT requires streaming captured in `LatencyBreakdown.ttft_ms`. If the Phase 0 harness does not capture TTFT (review `scripts/sme_eval/query_runner.py`), record that as an open question for a harness follow-up; do NOT block Phase 4 on it.

- [ ] **Step 3: Record findings**

Append a `latency_regression_note` field to the Phase 4 snapshot JSON summarising findings. If a regression is present, file a follow-up in `analytics/` under the pattern-mining playbook from Phase 6, NOT as a Phase 4 blocker.

- [ ] **Step 4: Commit any added notes**

```bash
git add -f tests/sme_metrics_phase4_*.json
git commit -m "phase4(sme-eval): latency regression note appended to snapshot"
```

---

## Task 14: Human-rated SME score collection

**Files:**
- Produce: `tests/sme_human_rating_phase4_{YYYY-MM-DD}.csv`

Launch gate: human-rated SME average ≥ 4.0/5.0 improvement over baseline on matched query set.

- [ ] **Step 1: Export rich-mode responses for rating**

```bash
python -c "
from scripts.sme_eval.result_store import JsonlResultStore
from scripts.sme_eval.human_rating import export_for_rating
import datetime, glob
today = datetime.date.today().isoformat()
latest = sorted(glob.glob('tests/sme_results_phase4_*.jsonl'))[-1]
export_for_rating(list(JsonlResultStore(latest).iter_all()),
                  f'tests/sme_human_rating_phase4_{today}.csv')
"
```

- [ ] **Step 2: Distribute to domain SMEs**

One rater per domain. CSV columns as set up in Phase 0: `query_id, query_text, response_text, sme_score_1_to_5, notes`. Raters see the query and the response but NOT whether it was baseline or Phase 4 — this matters for integrity of the comparison.

- [ ] **Step 3: Merge ratings and compute the delta**

```bash
python - <<'PY'
import json, glob, csv
rows = list(csv.DictReader(open(sorted(glob.glob(
    'tests/sme_human_rating_phase4_*.csv'))[-1])))
scores = [float(r['sme_score_1_to_5']) for r in rows if r['sme_score_1_to_5']]
avg = sum(scores)/len(scores) if scores else None
snap_path = sorted(glob.glob('tests/sme_metrics_phase4_*.json'))[-1]
snap = json.load(open(snap_path))
snap['human_rated_sme_score_avg'] = avg
snap['human_rated_count'] = len(scores)
json.dump(snap, open(snap_path,'w'), indent=2)
base = json.load(open(sorted(glob.glob(
    'tests/sme_metrics_baseline_*.json'))[-1]))
delta = (avg or 0) - (base.get('human_rated_sme_score_avg') or 0)
print(f"human-rated avg: {avg:.2f} (n={len(scores)})")
print(f"delta vs baseline: {delta:+.2f}")
PY
```

- [ ] **Step 4: Check the gate**

- Average ≥ 4.0/5.0? If no, do not proceed — go back to adapter / prompt tuning.
- Improvement over baseline? Positive delta required for Phase 4 to pass.

- [ ] **Step 5: Commit**

```bash
git add -f tests/sme_human_rating_phase4_*.csv tests/sme_metrics_phase4_*.json
git commit -m "phase4(sme-eval): human-rated SME scores + baseline delta"
```

---

## Task 15: Phase 4 exit checklist

Run this before declaring Phase 4 done. Every checkbox is a real check with a command or artifact behind it.

- [ ] All 14 preceding tasks committed with passing tests (`git log --oneline | grep phase4` shows the scoped commits)
- [ ] `pytest tests/generation tests/serving tests/agent tests/config tests/integration -v` — all green, zero regressions on the existing suite
- [ ] `src/intelligence/generator.py` diff since Phase 4 start is empty:
  ```bash
  git diff sme-baseline-v1..HEAD -- src/intelligence/generator.py | wc -l
  ```
  Expected: `0`. This is the Phase 4 memory rule — zero exceptions.
- [ ] No `pipeline_status` string added or renamed:
  ```bash
  git diff sme-baseline-v1..HEAD -- src/ | grep -E "PIPELINE_[A-Z_]+"
  ```
  Expected: only existing constants referenced, none declared.
- [ ] No Claude / Anthropic references in any changed file:
  ```bash
  git diff sme-baseline-v1..HEAD | grep -iE "claude|anthropic" | \
      grep -v "Spec has no references"
  ```
  Expected: empty.
- [ ] No wall-clock timeouts introduced on internal steps:
  ```bash
  git diff sme-baseline-v1..HEAD -- src/generation src/agent src/serving \
      | grep -E "timeout|wait_for|TimedOut" | grep -v "external|httpx"
  ```
  Expected: empty (only Blob / LLM external I/O timeouts, which are Phase 1's, not new here).
- [ ] Adapter YAMLs live only under `deploy/sme_adapters/` — no YAMLs under `src/`:
  ```bash
  find src -name '*.yaml'
  ```
  Expected: empty.
- [ ] Phase 4 metric snapshot exists and all launch-gate thresholds (Section 10) hold
- [ ] Human-rated SME avg ≥ 4.0, with positive delta vs baseline
- [ ] Latency note recorded (pass or flagged; non-blocking per spec)
- [ ] `enable_rich_mode` flag confirmed OFF globally; Phase 4 rollout is per-subscription only
- [ ] Rollback rehearsed once — flip flag OFF on the eval subscription and re-run 20 queries; output matches pre-Phase-4 compact shape
- [ ] Spec cross-check: every item in Section 8 (Prompts / Shape / Grounding) has a corresponding task above; every bullet in Section 12 Phase 4 is ticked

Commit the exit checklist itself only if it introduces doc changes — the plan already contains it; normally no commit is needed here.

---

## Self-review appendix

**Spec coverage check:** every item in Section 8 (Prompts / shape / grounding) and Section 12 Phase 4 (exit gate) maps to at least one task:
- New intents `diagnose / analyze / recommend` — Task 2 (classifier), Tasks 3–5 (templates)
- `format_hint` detection — Task 2
- URL detection — Task 2 (delivered ahead of Phase 5's fetcher)
- Rich template skeleton (Section 8) — Tasks 3–5
- Shape resolution (compact override / auto / honest-compact) — Task 6
- Domain persona injection — Task 7
- `core_agent.py` wiring — Task 8
- Post-generation recommendation grounding — Task 9
- `enable_rich_mode` flag — Task 10
- Metrics + human rating + latency — Tasks 12–14
- Launch gate — Task 15

**Memory-rule scan:**
- `src/intelligence/generator.py` — not opened by any task. Audit (Task 1 Step 6) enforces this.
- Formatting lives in `src/generation/prompts.py` and `src/generation/recommendation_grounding.py` — both in the generation package.
- Adapter YAMLs — only under `deploy/sme_adapters/`.
- Zero internal timeouts — all new code is synchronous or awaits existing I/O with Phase 1's safety limits.
- No Claude / Anthropic — template prose, YAMLs, and commit messages reviewed.
- `pipeline_status` immutable — Phase 4 is query-path only.
- Profile isolation — adapter loader already keys on `(subscription_id, domain)`; Task 8 always passes `subscription_id` and `profile_id` through.
- `enable_rich_mode` default OFF — Task 10 test enforces.

**Placeholder scan:** every task has concrete code, commands, or artifacts. No "TBD" / "TODO".

**Type consistency:** `AnalyzePromptInputs`, `DiagnosePromptInputs`, `RecommendPromptInputs`, `PersonaBundle`, `PackSummary`, `ResponseShape`, `FormatHint`, `ClassifiedQuery`, `GroundingReport` each defined once and referenced consistently.

**Load-bearing code is shown in full:**
- Shape resolver (Task 6 Step 3) — full code.
- Persona injection (Task 7 Step 3) — full code.
- Recommendation grounding post-pass (Task 9 Step 3) — full code.
- Analyze prompt template (Task 3 Step 3) — full code.

**Spec gaps / open questions (surfaced, not resolved in Phase 4):**
- Section 8 describes executive summary streaming first. The template orders sections but the underlying LLM streaming contract is assumed (Phase 1 / 3 have set it up). If the serving layer does not respect section-order streaming, a small additional change to the serving prompt handoff is needed — flagged, not fixed here.
- Section 10 requires `sme_persona_consistency` as LLM-judge. Open-question 6 in the spec is whether the judge is local or gateway-based; Phase 0 should have resolved this. If unresolved, Task 12 uses whatever Phase 0 shipped.
- Task 2's classifier prompt change is reused across environments; if any environment pins the classifier prompt separately (e.g., a fine-tuned head), it needs the same update — flagged for the operator.

**Line budget:** target 2000–2400 lines; this plan lands within it.

**ERRATA reconciliation (2026-04-21):** Applied ERRATA §§1, 4, 9, 10 on 2026-04-21.
- §1 (AdapterLoader): Task 7's `persona_bundle_from_adapter` reads
  `adapter.content_hash` / `.version` as direct attributes; Task 8 uses
  `get_adapter_loader()` singleton and `.load()` (no `.get()`).
- §4 (Feature flags): Task 10 no longer creates a parallel
  `src/config/features.py`; all consumers call
  `get_flag_resolver().is_enabled(sub, ENABLE_RICH_MODE)` from Phase 1's
  `src/config/feature_flags.py`.
- §9 (ClassifiedQuery): `query_text` added as first dataclass field in
  Task 2; populated by `classify_query`; Task 8 / Task 11 tests read it
  directly (no hasattr/getattr fallback).
- §10 (PackSummary): extended with `bank_entries`, `evidence_items`,
  `insights`, and `from_packed_items(...)` classmethod factory, with
  unit tests. Task 8 wires directly to these fields; the transient
  `_evidence_items_from_pack` / `_insights_from_pack` helpers were
  removed. Task 9's `pack_summary.bank_entries` access is now supported.
- Collateral fix: Task 8 routes `investigate` (and the `overview`
  fallback) through the `analyze` template per spec §8; no longer
  falls through to compact.

---

*End of Phase 4 plan.*
