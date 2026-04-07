# Latency Optimization & Adaptive Expert Intelligence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce query latency from 30-50s to 3-5s and add an adaptive expert intelligence layer that makes DocWain embody the expertise its documents demand.

**Architecture:** Six independent work streams: (1) vLLM systemd config for fp8 + speculative decoding, (2) token budget reduction in the Reasoner, (3) conditional visualization gating, (4) fast path enrichment, (5) prompt quality improvements, (6) new expert intelligence background pipeline. Streams 1-5 are config/code changes to existing files. Stream 6 is a new module plus integration hooks.

**Tech Stack:** Python 3.12, vLLM, PyMongo, FastAPI, Qwen3-14B (fast), Qwen3.5-27B (smart)

**Live testing:** All verification uses live `curl` calls against the running API — no pytest.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `systemd/docwain-vllm-fast.service` | Modify | fp8, chunked prefill, speculative decoding args |
| `src/generation/reasoner.py` | Modify | Halved token budgets, reduced thinking multiplier |
| `src/generation/prompts.py` | Modify | Depth instructions, conditional viz, dynamic system prompt, expert context section, complexity guide |
| `src/execution/fast_path.py` | Modify | Enriched context (5 chunks, 8K), rich system prompt, task formats, doc intelligence |
| `src/execution/query_classifier.py` | Modify | Tighten SIMPLE classification |
| `src/main.py` | Modify | Conditional visualization guard, insights endpoint |
| `src/intelligence_v2/expert_intelligence.py` | **Create** | Phase 1 + Phase 2 expert analysis pipeline |
| `src/api/app_lifespan.py` | Modify | Cache profile_expertise in app_state |
| `src/api/rag_state.py` | Modify | Add profile_expertise_cache field to AppState |
| `src/api/document_understanding_service.py` | Modify | Trigger background expert analysis after embedding |

---

### Task 1: vLLM Fast Instance — fp8 + Speculative Decoding

**Files:**
- Modify: `systemd/docwain-vllm-fast.service`
- Modify: `/etc/systemd/system/docwain-vllm-fast.service` (deployed copy)

- [ ] **Step 1: Update the systemd service file in the repo**

Edit `systemd/docwain-vllm-fast.service` — replace the ExecStart block:

```ini
ExecStart=/home/ubuntu/PycharmProjects/DocWain/.venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model models/docwain-v2-active \
    --served-model-name docwain-fast \
    --port 8100 \
    --host 0.0.0.0 \
    --dtype fp8 \
    --kv-cache-dtype fp8 \
    --max-model-len 40960 \
    --gpu-memory-utilization 0.85 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --speculative-model yuhuili/EAGLE3-Qwen3-14B \
    --num-speculative-tokens 5 \
    --tensor-parallel-size 1
```

Changes from current:
- `--dtype bfloat16` → `--dtype fp8`
- Added `--kv-cache-dtype fp8`
- `--gpu-memory-utilization 0.90` → `0.85`
- Added `--enable-chunked-prefill`
- Added `--speculative-model yuhuili/EAGLE3-Qwen3-14B`
- Added `--num-speculative-tokens 5`

- [ ] **Step 2: Deploy and restart the service**

```bash
sudo cp systemd/docwain-vllm-fast.service /etc/systemd/system/docwain-vllm-fast.service
sudo systemctl daemon-reload
sudo systemctl restart docwain-vllm-fast
```

Wait for startup (up to 5 minutes for model load with speculative model).

- [ ] **Step 3: Verify the service is healthy**

```bash
systemctl status docwain-vllm-fast
curl -s http://localhost:8100/health
curl -s http://localhost:8100/v1/models | python3 -m json.tool
```

Expected: service active, health returns OK, models endpoint lists `docwain-fast`.

- [ ] **Step 4: Benchmark token throughput**

```bash
curl -s -w "\nTotal time: %{time_total}s\n" http://localhost:8100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "docwain-fast",
    "messages": [{"role": "user", "content": "Explain what a contract termination clause is and why it matters. Be detailed."}],
    "max_tokens": 500,
    "temperature": 0.3
  }'
```

Expected: Total time should be ~4-6s for 500 tokens (was ~10s before). Check Prometheus metrics:

```bash
curl -s http://localhost:8100/metrics | grep -E "inter_token_latency_seconds_sum|inter_token_latency_seconds_count" | tail -4
```

Target: inter-token latency ~8-12ms (was 21ms).

- [ ] **Step 5: Commit**

```bash
git add systemd/docwain-vllm-fast.service
git commit -m "perf(vllm): switch fast instance to fp8 with speculative decoding

Enable fp8 dtype + kv-cache, chunked prefill, EAGLE3 speculative
decoding (5 tokens). Reduce gpu_memory_utilization from 0.90 to 0.85."
```

---

### Task 2: Reduce Token Budgets in Reasoner

**Files:**
- Modify: `src/generation/reasoner.py:37-46` (token budgets)
- Modify: `src/generation/reasoner.py:353` (thinking multiplier)

- [ ] **Step 1: Halve the base token budgets**

In `src/generation/reasoner.py`, replace the `_BASE_TOKENS` dict (lines 37-46):

```python
_BASE_TOKENS: Dict[str, int] = {
    "lookup": 1536,
    "extract": 3072,
    "list": 3072,
    "summarize": 2048,
    "overview": 2048,
    "compare": 3072,
    "investigate": 3072,
    "aggregate": 2048,
}
```

- [ ] **Step 2: Reduce thinking multiplier**

In `src/generation/reasoner.py`, line 353, change:

```python
        base = int(base * 2.5)
```

to:

```python
        base = int(base * 1.5)
```

- [ ] **Step 3: Verify with a live query**

```bash
curl -s -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the total amount in the invoices?",
    "user_id": "test@test.com",
    "profile_id": "<ACTIVE_PROFILE_ID>",
    "subscription_id": "<ACTIVE_SUB_ID>",
    "stream": false
  }' | python3 -c "import sys,json; r=json.load(sys.stdin); print('Response length:', len(r['answer']['response']), 'chars')"
```

Expected: Response should be substantive but not excessively long (under 3000 chars for a simple aggregation).

- [ ] **Step 4: Commit**

```bash
git add src/generation/reasoner.py
git commit -m "perf(reasoner): halve token budgets, reduce thinking multiplier to 1.5x

Lookup: 3072→1536, extract/compare/investigate: 6144→3072,
summarize/overview: 6144→2048, aggregate: 4096→2048.
Thinking multiplier: 2.5x→1.5x (think blocks suppressed via /no_think)."
```

---

### Task 3: Conditional Visualization

**Files:**
- Modify: `src/main.py:1008-1014` (streaming viz block)
- Modify: `src/main.py:1043-1053` (non-streaming viz block)
- Modify: `src/generation/prompts.py` (remove per-task VIZ directives, add conditional directive)

- [ ] **Step 1: Add visualization keyword detection helper to main.py**

Near the top of the `/ask` handler area in `src/main.py`, add a helper function (before the `ask_question_api` function):

```python
import re as _re

_VIZ_KEYWORDS = _re.compile(
    r"\b(chart|graph|plot|visuali[sz]e|diagram|show\s+me\s+a\s+(chart|graph|plot))\b",
    _re.IGNORECASE,
)

def _wants_visualization(query: str, task_type: str = "") -> bool:
    """Return True if the user explicitly asked for a visualization or the
    task type naturally produces chart-worthy data."""
    if _VIZ_KEYWORDS.search(query):
        return True
    if task_type in ("compare", "aggregate"):
        return True
    return False
```

- [ ] **Step 2: Guard the non-streaming visualization call**

In `src/main.py`, replace the non-streaming visualization block (around lines 1043-1053):

```python
    # Post-generation visualization enhancement
    if _wants_visualization(request.query, result.answer.get("metadata", {}).get("task_type", "")):
        try:
            from src.visualization.enhancer import enhance_with_visualization
            normalized = enhance_with_visualization(normalized, request.query, channel="web")
            _media = normalized.get("media")
            if _media:
                logger.info("Visualization attached: %d media items", len(_media))
        except Exception as _viz_exc:
            logger.warning("Visualization enhancement failed: %s", _viz_exc, exc_info=True)
```

- [ ] **Step 3: Guard the streaming visualization call**

In `src/main.py`, replace the streaming visualization block (around lines 1008-1014):

```python
            if _wants_visualization(request.query):
                try:
                    from src.visualization.enhancer import enhance_with_visualization
                    answer_payload = enhance_with_visualization(
                        answer_payload, request.query, channel="web",
                    )
                except Exception as _viz_exc:
                    logger.warning("Stream visualization failed: %s", _viz_exc)
```

- [ ] **Step 4: Remove per-task VIZ directives from TASK_FORMATS**

In `src/generation/prompts.py`, remove these lines from the `TASK_FORMATS` dict:
- Line 120 in "extract": `"- If the response contains a table with 3+ numeric rows, append a <!--DOCWAIN_VIZ--> directive with the appropriate chart_type and data.\n"`
- Line 134 in "compare": same line
- Line 145 in "summarize": same line
- Line 178 in "aggregate": same line

- [ ] **Step 5: Replace blanket VIZ system prompt directive with conditional one**

In `src/generation/prompts.py`, replace the VISUALIZATION DIRECTIVES section at the end of `_SYSTEM_PROMPT` (lines 62-74):

```python
    "VISUALIZATION DIRECTIVES:\n"
    "- Only append a <!--DOCWAIN_VIZ--> directive when the user explicitly "
    "requests a chart, graph, or visualization, OR when your response contains "
    "a comparison/aggregation table with 3+ rows of numeric data.\n"
    "- Do NOT generate visualizations for text-heavy, procedural, simple factual, "
    "or conversational responses.\n"
    "- When a visualization IS warranted, format: <!--DOCWAIN_VIZ\\n"
    "{\"chart_type\": \"...\", \"title\": \"...\", "
    "\"labels\": [...], \"values\": [...], \"unit\": \"...\"}\\n-->\n"
    "- Valid chart_type values: bar, horizontal_bar, grouped_bar, stacked_bar, "
    "donut, line, multi_line, area, scatter, radar, waterfall, gauge, treemap\n"
    "- Choose chart_type based on data: temporal → line, distribution → donut, "
    "comparison → grouped_bar, ranking → horizontal_bar, multi-metric → radar\n"
    "- For secondary series, add: \"secondary_values\": [...], \"secondary_name\": \"...\"\n"
```

- [ ] **Step 6: Verify — query without viz keywords should skip visualization**

```bash
curl -s -w "\nTime: %{time_total}s\n" -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the key terms in the contract?",
    "user_id": "test@test.com",
    "profile_id": "<ACTIVE_PROFILE_ID>",
    "subscription_id": "<ACTIVE_SUB_ID>",
    "stream": false
  }' | python3 -c "import sys,json; r=json.load(sys.stdin); print('Has media:', 'media' in str(r))"
```

Expected: `Has media: False` and faster response time (no 50-500ms viz overhead).

- [ ] **Step 7: Commit**

```bash
git add src/main.py src/generation/prompts.py
git commit -m "perf(viz): only generate visualizations when user requests or data warrants it

Skip enhance_with_visualization for non-viz queries. Remove blanket
VIZ directives from extract/compare/summarize/aggregate task formats.
Add _wants_visualization() gating based on keywords and task type."
```

---

### Task 4: Enrich the Fast Path

**Files:**
- Modify: `src/execution/fast_path.py:20-34` (constants and system prompt)
- Modify: `src/execution/fast_path.py:185-259` (execute_fast_path)
- Modify: `src/execution/fast_path.py:262-308` (execute_fast_path_stream)
- Modify: `src/execution/query_classifier.py:161-167` (tighten SIMPLE classification)

- [ ] **Step 1: Update fast path constants**

In `src/execution/fast_path.py`, replace lines 20-34:

```python
_FAST_PATH_TOP_K = 10  # Qdrant retrieval limit
_FAST_PATH_RERANK_K = 5  # Rerank to top-5
_MAX_CONTEXT_CHARS = 8000  # Richer context for substantive answers
```

Remove the `_SYSTEM_PROMPT` constant entirely (lines 29-34) — we will use the shared prompt from `prompts.py`.

- [ ] **Step 2: Add imports for shared prompts and token budgets**

At the top of `src/execution/fast_path.py`, add:

```python
from src.generation.prompts import build_system_prompt, TASK_FORMATS, _OUTPUT_FORMATS
from src.generation.reasoner import _BASE_TOKENS
```

- [ ] **Step 3: Update execute_fast_path to use enriched context**

Replace the `execute_fast_path` function (lines 185-259) with:

```python
def execute_fast_path(
    query: str,
    profile_id: str,
    subscription_id: str,
    app_state: Any,
) -> Dict[str, Any]:
    """Fast path for SIMPLE queries — enriched with doc intelligence and shared prompts."""
    t0 = time.monotonic()

    # 1. Embed
    vector = _embed_query(app_state.embedding_model, query)

    # 2. Search Qdrant
    points = _qdrant_search(
        app_state.qdrant_client, vector,
        subscription_id, profile_id,
        top_k=_FAST_PATH_TOP_K,
    )
    evidence = _points_to_evidence(points)

    # 3. Rerank to top-5
    reranked = rerank_chunks(
        query, evidence,
        top_k=_FAST_PATH_RERANK_K,
        cross_encoder=getattr(app_state, "reranker", None),
    )

    # 4. Build enriched prompt
    chunk_dicts = [{"text": c.text} for c in reranked]
    context = _build_fast_context(chunk_dicts, max_chunks=_FAST_PATH_RERANK_K)
    prompt = _build_prompt(query, context)

    # 5. Load profile expertise for dynamic system prompt
    profile_expertise = None
    expertise_cache = getattr(app_state, "profile_expertise_cache", None)
    if expertise_cache:
        profile_expertise = expertise_cache.get(profile_id)

    system = build_system_prompt(profile_expertise=profile_expertise)

    # Add task format instruction (default to lookup for fast path)
    task_format = TASK_FORMATS.get("lookup", "")
    system += "\n" + task_format

    # 6. Compute token budget from shared budgets
    max_tokens = _BASE_TOKENS.get("lookup", 1536)

    # 7. Generate response
    llm = app_state.llm_gateway
    answer_text = llm.generate(
        prompt,
        system=system,
        temperature=0.3,
        max_tokens=max_tokens,
    )

    elapsed = time.monotonic() - t0
    sources = _chunks_to_sources(reranked)
    context_found = len(reranked) > 0

    logger.info(
        "[FAST_PATH] query=%r chunks=%d elapsed=%.2fs",
        query[:80], len(reranked), elapsed,
    )

    return {
        "response": answer_text,
        "answer": answer_text,
        "sources": sources,
        "query_type": "SIMPLE",
        "fast_path": True,
        "grounded": context_found,
        "context_found": context_found,
        "metadata": {
            "fast_path": True,
            "query_type": "SIMPLE",
            "elapsed_s": round(elapsed, 3),
            "chunks_used": len(reranked),
        },
    }
```

- [ ] **Step 4: Update execute_fast_path_stream similarly**

Replace the `execute_fast_path_stream` function (lines 262-308) with:

```python
def execute_fast_path_stream(
    query: str,
    profile_id: str,
    subscription_id: str,
    app_state: Any,
) -> Generator[str, None, None]:
    """Streaming version of fast path — enriched with doc intelligence."""
    t0 = time.monotonic()

    # 1. Embed
    vector = _embed_query(app_state.embedding_model, query)

    # 2. Search Qdrant
    points = _qdrant_search(
        app_state.qdrant_client, vector,
        subscription_id, profile_id,
        top_k=_FAST_PATH_TOP_K,
    )
    evidence = _points_to_evidence(points)

    # 3. Rerank to top-5
    reranked = rerank_chunks(
        query, evidence,
        top_k=_FAST_PATH_RERANK_K,
        cross_encoder=getattr(app_state, "reranker", None),
    )

    # 4. Build enriched prompt
    chunk_dicts = [{"text": c.text} for c in reranked]
    context = _build_fast_context(chunk_dicts, max_chunks=_FAST_PATH_RERANK_K)
    prompt = _build_prompt(query, context)

    # 5. Load profile expertise for dynamic system prompt
    profile_expertise = None
    expertise_cache = getattr(app_state, "profile_expertise_cache", None)
    if expertise_cache:
        profile_expertise = expertise_cache.get(profile_id)

    system = build_system_prompt(profile_expertise=profile_expertise)
    task_format = TASK_FORMATS.get("lookup", "")
    system += "\n" + task_format

    max_tokens = _BASE_TOKENS.get("lookup", 1536)

    elapsed_prep = time.monotonic() - t0
    logger.info(
        "[FAST_PATH_STREAM] query=%r chunks=%d prep=%.2fs",
        query[:80], len(reranked), elapsed_prep,
    )

    # 6. Stream response
    llm = app_state.llm_gateway
    yield from llm.generate_stream(
        prompt,
        system=system,
        temperature=0.3,
        max_tokens=max_tokens,
    )
```

- [ ] **Step 5: Tighten SIMPLE classification**

In `src/execution/query_classifier.py`, replace lines 158-167 (the SIMPLE section):

```python
    # ------------------------------------------------------------------
    # 4. SIMPLE — ONLY single-fact lookups (name, date, amount, yes/no)
    # ------------------------------------------------------------------
    if _SINGLE_FACTOID.search(text) and words < 10 and not _MULTI_DOC.search(text):
        signals.append("short_factoid")
        return QueryClassification(query_type="SIMPLE", confidence=0.85, signals=signals)
```

Key changes: removed the second SIMPLE path (`single_entity_lookup`), tightened `words < 15` to `words < 10`. Only pure "what is X" factoid queries under 10 words go to SIMPLE now.

- [ ] **Step 6: Verify fast path produces richer responses**

```bash
curl -s -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the invoice amount?",
    "user_id": "test@test.com",
    "profile_id": "<ACTIVE_PROFILE_ID>",
    "subscription_id": "<ACTIVE_SUB_ID>",
    "stream": false
  }' | python3 -c "
import sys, json
r = json.load(sys.stdin)
ans = r['answer']['response']
print('Fast path:', r['answer'].get('fast_path', 'N/A'))
print('Length:', len(ans), 'chars')
print('Preview:', ans[:300])
"
```

Expected: Response should be more than 1-2 sentences, with bold values and proper formatting.

- [ ] **Step 7: Commit**

```bash
git add src/execution/fast_path.py src/execution/query_classifier.py
git commit -m "quality(fast-path): enrich with shared prompts, 5 chunks, 8K context

Use build_system_prompt() with profile expertise, TASK_FORMATS, and
shared token budgets. Increase rerank from top-3 to top-5, context
from 4K to 8K chars. Tighten SIMPLE classification to pure factoids."
```

---

### Task 5: Prompt Quality — Depth + Complexity Guide

**Files:**
- Modify: `src/generation/prompts.py:13-75` (system prompt rules)
- Modify: `src/generation/prompts.py:78-98` (build_system_prompt signature)
- Modify: `src/generation/prompts.py:325-334` (understand prompt complexity guide)

- [ ] **Step 1: Add depth rules to _SYSTEM_PROMPT**

In `src/generation/prompts.py`, before the `"FORMATTING:\n"` line (line 43), insert these two new rules:

```python
    "11. DEPTH OVER BREVITY: When answering analytical questions, provide "
    "substantive analysis — not just what the document says, but what it "
    "means for the user. Connect dots across evidence. Highlight implications, "
    "risks, or opportunities that a domain expert would notice. A helpful "
    "answer leaves the user more informed than the raw document would.\n"
    "12. MINIMUM SUBSTANCE: Every response must provide actionable insight. "
    "If the evidence supports a detailed answer, give one. Never reduce a "
    "rich evidence base to a single sentence unless the question is purely factual.\n"
    "13. Be direct and concise. Avoid repeating information from the context "
    "verbatim. Synthesize rather than quote.\n\n"
```

- [ ] **Step 2: Update build_system_prompt to accept profile_expertise**

Replace the `build_system_prompt` function (lines 78-98):

```python
def build_system_prompt(
    profile_domain: str = "",
    kg_context: str = "",
    profile_expertise: Optional[Dict] = None,
) -> str:
    """Return the core system prompt, optionally enriched with domain,
    KG context, and profile expertise identity.

    Args:
        profile_domain: The dominant domain of the profile.
        kg_context: Pre-formatted knowledge graph facts.
        profile_expertise: Dict with 'expertise_identity' containing
            'role', 'mindset', 'tone' keys from expert analysis.
    """
    if profile_expertise and profile_expertise.get("expertise_identity"):
        identity = profile_expertise["expertise_identity"]
        prompt = (
            f"You are a {identity.get('role', 'senior subject matter expert')}.\n"
            f"Your approach: {identity.get('mindset', 'Thorough, precise, evidence-based.')}\n"
            f"Communication style: {identity.get('tone', 'Professional, direct, insightful.')}\n\n"
        )
        # Append the rules (everything after the first line of _SYSTEM_PROMPT)
        rules_start = _SYSTEM_PROMPT.index("RULES:\n")
        prompt += _SYSTEM_PROMPT[rules_start:]
    else:
        prompt = _SYSTEM_PROMPT

    if profile_domain and profile_domain != "general":
        prompt += (
            f"\nYou have deep knowledge of documents in this collection, which "
            f"primarily covers the {profile_domain.replace('_', ' ')} domain.\n"
        )

    if kg_context:
        prompt += (
            f"\nYour knowledge from the documents:\n{kg_context}\n"
        )

    return prompt
```

- [ ] **Step 3: Add complexity guide to UNDERSTAND prompt**

In `src/generation/prompts.py`, after the existing TASK TYPE GUIDE section (line 334), add:

```python
        "- COMPLEXITY GUIDE: Only classify as 'simple' if the query asks for a "
        "single, specific fact (a name, date, amount, yes/no). Questions about "
        "risks, implications, processes, recommendations, or 'what should I know "
        "about' are ALWAYS 'complex'.\n"
```

- [ ] **Step 4: Add expert analysis section to build_reason_prompt**

In `src/generation/prompts.py`, in the `build_reason_prompt` function, after the DOCUMENT INTELLIGENCE block (after line 448) and before the EVIDENCE block (line 451), add:

```python
    # Expert analysis context (pre-computed insights)
    expert_insights = None
    if doc_context:
        expert_insights = doc_context.get("expert_insights")
    if expert_insights:
        parts.append("--- EXPERT ANALYSIS ---")
        parts.append("Pre-computed expert observations relevant to this query:")
        for insight in expert_insights[:5]:
            if isinstance(insight, dict):
                cat = insight.get("category", "")
                text = insight.get("insight", "")
                rec = insight.get("recommendation", "")
                parts.append(f"- [{cat.upper()}] {text}")
                if rec:
                    parts.append(f"  Recommendation: {rec}")
            else:
                parts.append(f"- {insight}")
        parts.append("--- END EXPERT ANALYSIS ---")
        parts.append("")
```

- [ ] **Step 5: Add Optional import at the top of prompts.py if not present**

Verify `Optional` and `Dict` are imported. Current imports (line 7): `from typing import Any, Dict, List, Optional` — already present, no change needed.

- [ ] **Step 6: Verify improved response quality**

```bash
curl -s -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the key risks in these documents?",
    "user_id": "test@test.com",
    "profile_id": "<ACTIVE_PROFILE_ID>",
    "subscription_id": "<ACTIVE_SUB_ID>",
    "stream": false
  }' | python3 -c "
import sys, json
r = json.load(sys.stdin)
ans = r['answer']['response']
print('Length:', len(ans), 'chars')
print('---')
print(ans[:800])
"
```

Expected: Should NOT go to fast path (analytical query). Response should be multi-paragraph with structured analysis, not 1-2 sentences.

- [ ] **Step 7: Commit**

```bash
git add src/generation/prompts.py
git commit -m "quality(prompts): add depth rules, dynamic expertise identity, expert context section

Rules 11-13: depth over brevity, minimum substance, synthesize not quote.
build_system_prompt now accepts profile_expertise for adaptive persona.
build_reason_prompt injects expert analysis between doc intelligence and evidence.
UNDERSTAND prompt gets complexity guide to prevent analytical queries going SIMPLE."
```

---

### Task 6: Expert Intelligence Pipeline — Module Creation

**Files:**
- Create: `src/intelligence_v2/expert_intelligence.py`

- [ ] **Step 1: Create the expert intelligence module**

Create `src/intelligence_v2/expert_intelligence.py`:

```python
"""Adaptive Expert Intelligence — background analysis pipeline.

Runs after document embedding completes. Produces a profile_expertise
document that makes DocWain embody the expert its documents demand.

Phase 1: Profile Understanding — single LLM call to identify expertise identity,
         knowledge map, proactive insights, advisory capabilities, knowledge gaps.
Phase 2: Deep Expert Analysis — per-document-cluster LLM calls for connections,
         implications, recommendations.

All calls use the smart path (27B) and run at background priority.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Phase 1 prompt
# ---------------------------------------------------------------------------

_PHASE1_SYSTEM = (
    "You are analyzing a collection of documents to build an expert profile. "
    "Your output must be valid JSON — no markdown fences, no commentary."
)

_PHASE1_PROMPT_TEMPLATE = """\
Below are summaries, entities, and key facts from a document collection.

{doc_summaries}

Based on this collection, produce JSON with these fields:

{{
  "expertise_identity": {{
    "role": "<What kind of expert would deeply understand these documents? Be specific, e.g. 'Senior Contract Negotiation Specialist for IT outsourcing agreements'>",
    "mindset": "<How does this expert approach problems? What do they prioritize?>",
    "tone": "<Communication style: e.g. 'Practical, direct, solution-oriented'>"
  }},
  "knowledge_map": [
    {{
      "area": "<knowledge area covered>",
      "depth": "comprehensive|detailed|partial|minimal",
      "document_ids": ["<doc_ids covering this area>"]
    }}
  ],
  "proactive_insights": [
    {{
      "category": "critical|important|informational",
      "insight": "<What would an expert immediately notice or flag?>",
      "recommendation": "<What action should be taken?>",
      "evidence_refs": ["<document_ids>"]
    }}
  ],
  "advisory_capabilities": ["<What kinds of questions can this expert answer authoritatively?>"],
  "knowledge_gaps": ["<What is NOT covered that a user might expect?>"]
}}
"""

# ---------------------------------------------------------------------------
# Phase 2 prompt
# ---------------------------------------------------------------------------

_PHASE2_SYSTEM = (
    "You are a {role}. Analyze the following document cluster deeply. "
    "Your output must be valid JSON — no markdown fences, no commentary."
)

_PHASE2_PROMPT_TEMPLATE = """\
As a {role} with the mindset "{mindset}", analyze this document cluster:

Topic: {cluster_topic}

Documents:
{cluster_docs}

Produce JSON:
{{
  "cluster_topic": "{cluster_topic}",
  "connections": ["<Connections between documents that aren't explicitly stated>"],
  "implications": ["<Implications a user might miss>"],
  "recommendations": ["<Actionable recommendations based on expert analysis>"]
}}
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_profile_expertise(
    profile_id: str,
    subscription_id: str,
    mongo_client: Any,
    vllm_manager: Any,
) -> Optional[Dict[str, Any]]:
    """Run Phase 1 + Phase 2 expert analysis for a profile.

    Parameters
    ----------
    profile_id : str
        The profile to analyze.
    subscription_id : str
        Tenant scope.
    mongo_client : Any
        MongoDB client or database handle.
    vllm_manager : Any
        VLLMManager instance for smart path queries.

    Returns
    -------
    dict or None
        The profile_expertise document, or None if analysis failed.
    """
    t0 = time.monotonic()
    db = _get_db(mongo_client)

    # Gather existing intelligence
    doc_intel = list(db["documents"].find(
        {"profile_id": profile_id, "subscription_id": subscription_id, "intelligence_ready": True},
        {"document_id": 1, "filename": 1, "intelligence": 1, "_id": 0},
    ))

    if not doc_intel:
        logger.info("[EXPERT] No documents with intelligence for profile=%s", profile_id)
        return None

    doc_ids = [d["document_id"] for d in doc_intel]
    logger.info("[EXPERT] Phase 1: analyzing %d documents for profile=%s", len(doc_intel), profile_id)

    # Build document summaries block for Phase 1
    doc_summaries = _format_doc_summaries(doc_intel)

    # Phase 1: Profile Understanding
    phase1_prompt = _PHASE1_PROMPT_TEMPLATE.format(doc_summaries=doc_summaries)
    try:
        phase1_raw = vllm_manager.query_smart(
            prompt=phase1_prompt,
            system_prompt=_PHASE1_SYSTEM,
            max_tokens=4096,
            temperature=0.3,
        )
        phase1 = _parse_json(phase1_raw)
        if not phase1:
            logger.error("[EXPERT] Phase 1 returned invalid JSON for profile=%s", profile_id)
            return None
    except Exception:
        logger.error("[EXPERT] Phase 1 LLM call failed for profile=%s", profile_id, exc_info=True)
        return None

    logger.info("[EXPERT] Phase 1 complete: role=%s", phase1.get("expertise_identity", {}).get("role", "unknown"))

    # Phase 2: Deep Expert Analysis (per-document cluster)
    clusters = _build_clusters(doc_intel)
    deep_analysis = []
    identity = phase1.get("expertise_identity", {})

    for cluster in clusters:
        try:
            phase2_prompt = _PHASE2_PROMPT_TEMPLATE.format(
                role=identity.get("role", "document analyst"),
                mindset=identity.get("mindset", "thorough and precise"),
                cluster_topic=cluster["topic"],
                cluster_docs=cluster["docs_text"],
            )
            phase2_system = _PHASE2_SYSTEM.format(role=identity.get("role", "document analyst"))
            phase2_raw = vllm_manager.query_smart(
                prompt=phase2_prompt,
                system_prompt=phase2_system,
                max_tokens=2048,
                temperature=0.3,
            )
            phase2 = _parse_json(phase2_raw)
            if phase2:
                deep_analysis.append(phase2)
        except Exception:
            logger.warning("[EXPERT] Phase 2 failed for cluster=%s", cluster["topic"], exc_info=True)
            continue

        # Backpressure: 2s delay between cluster analysis calls
        time.sleep(2)

    # Build final expertise document
    expertise = {
        "profile_id": profile_id,
        "subscription_id": subscription_id,
        **phase1,
        "deep_analysis": deep_analysis,
        "document_ids_analyzed": doc_ids,
        "version": 1,
        "created_at": time.time(),
        "updated_at": time.time(),
    }

    # Upsert to MongoDB
    db["profile_expertise"].replace_one(
        {"profile_id": profile_id},
        expertise,
        upsert=True,
    )

    elapsed = time.monotonic() - t0
    logger.info(
        "[EXPERT] Complete: profile=%s docs=%d clusters=%d elapsed=%.1fs",
        profile_id, len(doc_ids), len(deep_analysis), elapsed,
    )

    return expertise


def get_cached_expertise(
    profile_id: str,
    mongo_client: Any,
) -> Optional[Dict[str, Any]]:
    """Load cached profile expertise from MongoDB."""
    db = _get_db(mongo_client)
    return db["profile_expertise"].find_one(
        {"profile_id": profile_id},
        {"_id": 0},
    )


def is_stale(
    profile_id: str,
    subscription_id: str,
    mongo_client: Any,
) -> bool:
    """Check if expertise needs rebuilding (new docs added/removed)."""
    db = _get_db(mongo_client)
    expertise = db["profile_expertise"].find_one(
        {"profile_id": profile_id},
        {"document_ids_analyzed": 1, "_id": 0},
    )
    if not expertise:
        return True

    current_doc_ids = set(
        d["document_id"] for d in db["documents"].find(
            {"profile_id": profile_id, "subscription_id": subscription_id, "intelligence_ready": True},
            {"document_id": 1, "_id": 0},
        )
    )
    analyzed_ids = set(expertise.get("document_ids_analyzed", []))

    if current_doc_ids != analyzed_ids:
        change_ratio = len(current_doc_ids.symmetric_difference(analyzed_ids)) / max(len(current_doc_ids), 1)
        return change_ratio > 0  # Any change triggers rebuild
    return False


def filter_insights_for_query(
    expertise: Dict[str, Any],
    query: str,
) -> List[Dict[str, Any]]:
    """Return expert insights relevant to the query (simple keyword matching)."""
    if not expertise:
        return []

    query_lower = query.lower()
    query_words = set(query_lower.split())
    scored = []

    for insight in expertise.get("proactive_insights", []):
        insight_text = (insight.get("insight", "") + " " + insight.get("recommendation", "")).lower()
        insight_words = set(insight_text.split())
        overlap = len(query_words & insight_words)
        if overlap > 0:
            scored.append((overlap, insight))

    # Also include deep analysis recommendations
    for analysis in expertise.get("deep_analysis", []):
        topic = analysis.get("cluster_topic", "").lower()
        if any(w in topic for w in query_words):
            for rec in analysis.get("recommendations", []):
                scored.append((2, {"category": "important", "insight": rec, "recommendation": ""}))
            for imp in analysis.get("implications", []):
                scored.append((1, {"category": "informational", "insight": imp, "recommendation": ""}))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [s[1] for s in scored[:5]]


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _get_db(mongo_client: Any) -> Any:
    """Return the database handle from a client or pass through."""
    if hasattr(mongo_client, "get_database"):
        return mongo_client.get_database()
    if hasattr(mongo_client, "documents"):
        return mongo_client
    return mongo_client


def _format_doc_summaries(doc_intel: List[Dict]) -> str:
    """Format document intelligence into a text block for the LLM prompt."""
    parts = []
    for doc in doc_intel:
        doc_id = doc.get("document_id", "unknown")
        filename = doc.get("filename", "unknown")
        intel = doc.get("intelligence", {})
        summary = intel.get("summary", intel.get("document_summary", "No summary"))
        entities = intel.get("entities", intel.get("key_entities", []))
        facts = intel.get("facts", intel.get("key_facts", []))

        parts.append(f"Document: {filename} (ID: {doc_id})")
        parts.append(f"  Summary: {summary}")

        if entities:
            entity_strs = []
            for e in entities[:8]:
                if isinstance(e, dict):
                    entity_strs.append(e.get("value", e.get("name", str(e))))
                else:
                    entity_strs.append(str(e))
            parts.append(f"  Entities: {', '.join(entity_strs)}")

        if facts:
            fact_strs = []
            for f in facts[:5]:
                if isinstance(f, dict):
                    fact_strs.append(f.get("claim", str(f)))
                else:
                    fact_strs.append(str(f))
            parts.append(f"  Key facts: {'; '.join(fact_strs)}")

        parts.append("")

    return "\n".join(parts)


def _build_clusters(doc_intel: List[Dict]) -> List[Dict]:
    """Group documents into clusters by document type for Phase 2 analysis.

    Simple grouping by doc_type from intelligence metadata. Each cluster
    gets its own Phase 2 analysis call.
    """
    by_type: Dict[str, List[Dict]] = {}
    for doc in doc_intel:
        intel = doc.get("intelligence", {})
        doc_type = intel.get("document_type", "general")
        by_type.setdefault(doc_type, []).append(doc)

    clusters = []
    for doc_type, docs in by_type.items():
        docs_text = _format_doc_summaries(docs)
        clusters.append({
            "topic": doc_type,
            "docs_text": docs_text,
            "doc_ids": [d["document_id"] for d in docs],
        })

    return clusters


def _parse_json(text: str) -> Optional[Dict]:
    """Parse JSON from LLM output, stripping markdown fences if present."""
    import json
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON in the text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    return None
```

- [ ] **Step 2: Commit**

```bash
git add src/intelligence_v2/expert_intelligence.py
git commit -m "feat(expert): add adaptive expert intelligence pipeline

Phase 1: Profile understanding via smart path — identifies expertise
identity, knowledge map, proactive insights, advisory capabilities.
Phase 2: Deep analysis per document cluster — connections, implications,
recommendations. Stores in MongoDB profile_expertise collection."
```

---

### Task 7: Integration — AppState, Lifespan, Post-Embedding Trigger

**Files:**
- Modify: `src/api/rag_state.py:13-26` (add cache field)
- Modify: `src/api/app_lifespan.py` (load expertise cache on startup)
- Modify: `src/api/document_understanding_service.py:161-178` (trigger after embedding)
- Modify: `src/main.py` (add insights endpoint)

- [ ] **Step 1: Add profile_expertise_cache to AppState**

In `src/api/rag_state.py`, add a new field to the `AppState` dataclass (after line 25):

```python
    profile_expertise_cache: Dict[str, Dict[str, Any]] = field(default_factory=dict)
```

- [ ] **Step 2: Load expertise cache on startup**

In `src/api/app_lifespan.py`, after the AppState initialization (after the `state = AppState(...)` block), add:

```python
    # Pre-load profile expertise cache
    profile_expertise_cache = {}
    try:
        from pymongo import MongoClient as _MongoClient
        _mc = _MongoClient(Config.MongoDB.URI)
        _db = _mc[Config.MongoDB.DB]
        for exp in _db["profile_expertise"].find({}, {"_id": 0}):
            pid = exp.get("profile_id")
            if pid:
                profile_expertise_cache[pid] = exp
        logger.info("Loaded %d profile expertise entries", len(profile_expertise_cache))
        _mc.close()
    except Exception:
        logger.warning("Could not load profile expertise cache", exc_info=True)
    state.profile_expertise_cache = profile_expertise_cache
```

- [ ] **Step 3: Trigger background expert analysis after embedding**

In `src/api/document_understanding_service.py`, after the embedding block (after line 169, inside the `if embed_after:` block), add:

```python
        # Trigger background expert intelligence analysis
        import threading
        def _run_expert_analysis():
            try:
                from src.intelligence_v2.expert_intelligence import build_profile_expertise, is_stale
                from src.api.rag_state import get_app_state
                from pymongo import MongoClient as _ExpertMongoClient

                app_state = get_app_state()
                if not app_state or not app_state.vllm_manager:
                    logger.info("[EXPERT] vllm_manager not available, skipping expert analysis")
                    return

                _mc = _ExpertMongoClient(Config.MongoDB.URI)
                _db = _mc[Config.MongoDB.DB]

                if not is_stale(profile_id, subscription_id, _db):
                    logger.info("[EXPERT] Expertise still fresh for profile=%s", profile_id)
                    _mc.close()
                    return

                expertise = build_profile_expertise(
                    profile_id=profile_id,
                    subscription_id=subscription_id,
                    mongo_client=_db,
                    vllm_manager=app_state.vllm_manager,
                )

                # Update the in-memory cache
                if expertise and app_state.profile_expertise_cache is not None:
                    app_state.profile_expertise_cache[profile_id] = expertise
                    logger.info("[EXPERT] Updated in-memory cache for profile=%s", profile_id)

                _mc.close()
            except Exception:
                logger.error("[EXPERT] Background expert analysis failed", exc_info=True)

        threading.Thread(target=_run_expert_analysis, daemon=True, name="expert-analysis").start()
        logger.info("[EXPERT] Background expert analysis triggered for profile=%s", profile_id)
```

- [ ] **Step 4: Add insights endpoint to main.py**

In `src/main.py`, add a new endpoint (near the other profile-related endpoints):

```python
@api_router.get("/profile/{profile_id}/insights", tags=["Profile"])
def get_profile_insights(profile_id: str):
    """Return proactive expert insights for a profile."""
    from src.api.rag_state import get_app_state
    app_state = get_app_state()
    if not app_state:
        return {"insights": [], "expertise_identity": None}

    expertise = app_state.profile_expertise_cache.get(profile_id)
    if not expertise:
        # Try loading from MongoDB
        try:
            from src.intelligence_v2.expert_intelligence import get_cached_expertise
            from pymongo import MongoClient as _InsightMongoClient
            from src.api.config import Config
            _mc = _InsightMongoClient(Config.MongoDB.URI)
            _db = _mc[Config.MongoDB.DB]
            expertise = get_cached_expertise(profile_id, _db)
            _mc.close()
            if expertise:
                app_state.profile_expertise_cache[profile_id] = expertise
        except Exception:
            pass

    if not expertise:
        return {"insights": [], "expertise_identity": None}

    insights = expertise.get("proactive_insights", [])
    # Sort: critical first, then important, then informational
    priority = {"critical": 0, "important": 1, "informational": 2}
    insights.sort(key=lambda x: priority.get(x.get("category", ""), 99))

    return {
        "insights": insights[:5],
        "expertise_identity": expertise.get("expertise_identity"),
        "knowledge_gaps": expertise.get("knowledge_gaps", []),
        "advisory_capabilities": expertise.get("advisory_capabilities", []),
    }
```

- [ ] **Step 5: Wire expert insights into the query pipeline**

The Reasoner's `build_reason_prompt` now accepts expert insights via `doc_context["expert_insights"]` (added in Task 5). We need to inject them when building the context.

In `src/agent/core_agent.py`, find where `build_reason_prompt` is called and where `doc_context` is assembled. Add expert insights:

```python
# Before the build_reason_prompt call, add:
from src.intelligence_v2.expert_intelligence import filter_insights_for_query
from src.api.rag_state import get_app_state

app_state = get_app_state()
if app_state and app_state.profile_expertise_cache:
    expertise = app_state.profile_expertise_cache.get(profile_id)
    if expertise:
        relevant_insights = filter_insights_for_query(expertise, query)
        if relevant_insights and doc_context is not None:
            doc_context["expert_insights"] = relevant_insights
```

The exact location depends on the core_agent's handle method. Find the `build_reason_prompt` call and inject before it.

- [ ] **Step 6: Commit**

```bash
git add src/api/rag_state.py src/api/app_lifespan.py src/api/document_understanding_service.py src/main.py src/agent/core_agent.py
git commit -m "feat(expert): integrate expertise into AppState, lifespan, query pipeline

Add profile_expertise_cache to AppState, pre-load on startup.
Trigger background expert analysis after embedding via daemon thread.
Add GET /profile/{id}/insights endpoint for proactive UI surfacing.
Inject filtered expert insights into Reasoner doc_context."
```

---

### Task 8: End-to-End Live Testing

**Files:** None (testing only)

**Note:** Replace `<ACTIVE_PROFILE_ID>` and `<ACTIVE_SUB_ID>` with actual values from the running system. To find them:

```bash
# Get active profile and subscription IDs
python3 -c "
from pymongo import MongoClient
from src.api.config import Config
mc = MongoClient(Config.MongoDB.URI)
db = mc[Config.MongoDB.DB]
profile = db['profiles'].find_one({'status': 'READY'}, {'_id': 0, 'profile_id': 1, 'subscription_id': 1, 'name': 1})
print('Profile:', profile)
# Get a doc with intelligence
doc = db['documents'].find_one({'profile_id': profile['profile_id'], 'intelligence_ready': True}, {'document_id': 1, 'filename': 1, '_id': 0})
print('Doc:', doc)
"
```

- [ ] **Step 1: Verify vLLM fast instance is running with fp8**

```bash
systemctl status docwain-vllm-fast --no-pager | head -15
curl -s http://localhost:8100/health
curl -s http://localhost:8100/metrics | grep "inter_token_latency_seconds_sum\b" | tail -1
```

Expected: active, healthy, metrics available.

- [ ] **Step 2: Restart the DocWain API to pick up all code changes**

```bash
# Find and restart the main API process
sudo systemctl restart docwain-api  # or however the API is started
# OR if running manually:
# kill the existing process and restart
```

Wait for startup, then verify:

```bash
curl -s http://localhost:8000/health || curl -s http://localhost:8000/api/health
```

- [ ] **Step 3: Test latency — simple factoid query (fast path)**

```bash
time curl -s -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the total invoice amount?",
    "user_id": "test@test.com",
    "profile_id": "<ACTIVE_PROFILE_ID>",
    "subscription_id": "<ACTIVE_SUB_ID>",
    "stream": false
  }' | python3 -c "
import sys, json
r = json.load(sys.stdin)
ans = r.get('answer', {})
resp = ans.get('response', '') if isinstance(ans, dict) else str(ans)
print('=== FAST PATH TEST ===')
print('Response length:', len(resp), 'chars')
print('Fast path:', ans.get('fast_path', 'N/A') if isinstance(ans, dict) else 'N/A')
print('---')
print(resp[:500])
"
```

Expected: Response in 2-5 seconds, substantive (not 1-2 sentences), fast_path=True.

- [ ] **Step 4: Test latency — analytical query (full path)**

```bash
time curl -s -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the key risks and important things I should know about these documents?",
    "user_id": "test@test.com",
    "profile_id": "<ACTIVE_PROFILE_ID>",
    "subscription_id": "<ACTIVE_SUB_ID>",
    "stream": false
  }' | python3 -c "
import sys, json
r = json.load(sys.stdin)
ans = r.get('answer', {})
resp = ans.get('response', '') if isinstance(ans, dict) else str(ans)
print('=== ANALYTICAL QUERY TEST ===')
print('Response length:', len(resp), 'chars')
print('---')
print(resp[:800])
"
```

Expected: Response in 5-10 seconds, multi-paragraph structured analysis with headers and bold values.

- [ ] **Step 5: Test streaming**

```bash
time curl -s -N -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Summarize the key points of the documents",
    "user_id": "test@test.com",
    "profile_id": "<ACTIVE_PROFILE_ID>",
    "subscription_id": "<ACTIVE_SUB_ID>",
    "stream": true
  }' | head -c 500
```

Expected: Tokens should start appearing almost immediately (~100ms TTFT), streaming as text/plain.

- [ ] **Step 6: Test visualization gating — no-viz query should have no media**

```bash
curl -s -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "List the entities mentioned in the documents",
    "user_id": "test@test.com",
    "profile_id": "<ACTIVE_PROFILE_ID>",
    "subscription_id": "<ACTIVE_SUB_ID>",
    "stream": false
  }' | python3 -c "
import sys, json
r = json.load(sys.stdin)
ans = r.get('answer', {})
has_media = 'media' in json.dumps(ans)
print('=== VIZ GATING TEST ===')
print('Has media:', has_media)
print('Expected: False')
"
```

- [ ] **Step 7: Test expert intelligence — trigger and verify**

```bash
# Manually trigger expert analysis for the active profile
python3 -c "
from pymongo import MongoClient
from src.api.config import Config
from src.intelligence_v2.expert_intelligence import build_profile_expertise
from src.serving.vllm_manager import VLLMManager

mc = MongoClient(Config.MongoDB.URI)
db = mc[Config.MongoDB.DB]
vm = VLLMManager(
    fast_url=Config.VLLM.FAST_URL,
    smart_url=Config.VLLM.SMART_URL,
    fast_model=Config.VLLM.FAST_MODEL,
    smart_model=Config.VLLM.SMART_MODEL,
    gpu_mode_file=Config.VLLM.GPU_MODE_FILE,
)

profile = db['profiles'].find_one({'status': 'READY'})
print('Analyzing profile:', profile['name'], profile['profile_id'])

result = build_profile_expertise(
    profile_id=profile['profile_id'],
    subscription_id=profile['subscription_id'],
    mongo_client=db,
    vllm_manager=vm,
)

if result:
    print('SUCCESS')
    print('Role:', result['expertise_identity']['role'])
    print('Insights:', len(result.get('proactive_insights', [])))
    print('Knowledge areas:', len(result.get('knowledge_map', [])))
    print('Deep analysis clusters:', len(result.get('deep_analysis', [])))
    print()
    for i, insight in enumerate(result.get('proactive_insights', [])[:3]):
        print(f'  [{insight[\"category\"]}] {insight[\"insight\"][:120]}')
else:
    print('FAILED — check logs')

mc.close()
"
```

Expected: Expert analysis completes, prints the discovered role, insights, and knowledge areas.

- [ ] **Step 8: Test insights endpoint**

```bash
curl -s http://localhost:8000/api/profile/<ACTIVE_PROFILE_ID>/insights | python3 -m json.tool
```

Expected: Returns JSON with `expertise_identity`, `insights` array (up to 5), `knowledge_gaps`, and `advisory_capabilities`.

- [ ] **Step 9: Test expert intelligence in query response**

After expert analysis is cached, query again and verify insights are woven into the response:

```bash
curl -s -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What should I be aware of regarding these documents?",
    "user_id": "test@test.com",
    "profile_id": "<ACTIVE_PROFILE_ID>",
    "subscription_id": "<ACTIVE_SUB_ID>",
    "stream": false
  }' | python3 -c "
import sys, json
r = json.load(sys.stdin)
ans = r.get('answer', {})
resp = ans.get('response', '') if isinstance(ans, dict) else str(ans)
print('=== EXPERT INTELLIGENCE TEST ===')
print('Response length:', len(resp), 'chars')
print('---')
print(resp[:1000])
"
```

Expected: Response should be substantive, expert-toned, with proactive insights woven in — not just document parroting.

- [ ] **Step 10: Final latency comparison**

```bash
echo "=== LATENCY COMPARISON ==="
echo ""
echo "Simple query (fast path):"
time curl -s -o /dev/null -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"What is the contract value?","user_id":"test@test.com","profile_id":"<ACTIVE_PROFILE_ID>","subscription_id":"<ACTIVE_SUB_ID>","stream":false}'

echo ""
echo "Analytical query (full path):"
time curl -s -o /dev/null -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"What are the risks and recommendations for these documents?","user_id":"test@test.com","profile_id":"<ACTIVE_PROFILE_ID>","subscription_id":"<ACTIVE_SUB_ID>","stream":false}'

echo ""
echo "Streaming TTFT:"
time curl -s -N -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"Summarize the documents","user_id":"test@test.com","profile_id":"<ACTIVE_PROFILE_ID>","subscription_id":"<ACTIVE_SUB_ID>","stream":true}' | head -c 50
echo ""
```

Target:
- Simple query: < 5s
- Analytical query: < 10s
- Streaming TTFT: < 1s

---
