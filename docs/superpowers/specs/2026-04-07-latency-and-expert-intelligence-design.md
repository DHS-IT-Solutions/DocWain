# DocWain Latency Optimization & Adaptive Expert Intelligence

**Date:** 2026-04-07  
**Status:** Approved  
**Goal:** Reduce end-to-end query latency from 30-50s to 3-5s while transforming DocWain from a reactive document search tool into a proactive expert intelligence advisor.

---

## Part 1: Latency Optimization

### 1.1 Current State (Measured via Prometheus)

| Metric | Value |
|--------|-------|
| Time-to-first-token (TTFT) | 112ms avg |
| Inter-token latency | 20.88ms (~48 tok/s) |
| Avg prompt tokens | 715 |
| Avg response tokens | 713 |
| Avg end-to-end (non-streaming) | 14.9s |
| User-perceived (with pipeline overhead) | 30-50s |

Bottleneck breakdown: vLLM generation = 99%, RAG pipeline = 1%.

### 1.2 UI Streaming (Considered Done)

The backend already supports streaming via `StreamingResponse` at `src/main.py:982-1037`. The UI (`docwain-ui` repo, develop branch) needs updates to consume the stream:

- `src/services/api/api.ts` — add `apiServiceStream()` using `ReadableStream` from fetch
- `src/services/selfassist/selfassist.ts` — add `selfAssistChatStream()` wrapping the stream API
- `src/pages/Home/Home.tsx` — update `handleSubmit()` to incrementally render tokens via `setMessages` updates

The backend streams `text/plain` tokens with a trailing `<!--DOCWAIN_MEDIA_JSON:...-->` metadata block. No SSE library needed — browser-native `ReadableStream` suffices.

**Impact:** Perceived latency drops from 30-50s to ~112ms TTFT.

### 1.3 vLLM Fast Instance Configuration

**File:** `/etc/systemd/system/docwain-vllm-fast.service`

| Setting | Current | New | Rationale |
|---------|---------|-----|-----------|
| `--dtype` | bfloat16 | fp8 | ~1.8x decode speedup, negligible quality loss at 14B |
| `--kv-cache-dtype` | (default/bf16) | fp8 | Halves KV cache memory, more concurrent request headroom |
| `--gpu-memory-utilization` | 0.90 | 0.85 | Reduce memory pressure, avoid OOM on spikes |
| `--enable-chunked-prefill` | (not set) | true | Better scheduling, reduces TTFT under concurrent load |
| `--speculative-model` | (not set) | EAGLE3 model path | Present in Python config (`src/serving/config.py:62`) but missing from systemd — wiring bug |
| `--num-speculative-tokens` | (not set) | 5 | Standard for EAGLE3, ~1.5-2x decode speedup |
| `--max-model-len` | 40960 | 40960 (unchanged) | Keep full context capacity |

**Expected result:** Inter-token latency drops from ~21ms to ~8-10ms (100-120 tok/s).

### 1.4 Response Length Discipline

**File:** `src/generation/reasoner.py` — `_BASE_TOKENS` dict

| Intent | Current | New |
|--------|---------|-----|
| lookup | 3072 | 1536 |
| extract | 6144 | 3072 |
| list | 6144 | 3072 |
| summarize | 6144 | 2048 |
| overview | 6144 | 2048 |
| compare | 6144 | 3072 |
| investigate | 6144 | 3072 |
| aggregate | 4096 | 2048 |

Additional changes:
- Reduce thinking multiplier from 2.5x to 1.5x (with `/no_think` suppressing think blocks, 2.5x is wasteful)
- Max cap remains 16384
- Scaling multipliers (1.3x for >10 evidence, 1.15x for >5) remain unchanged

**Conciseness instruction** — add to system prompt in `src/generation/prompts.py`:
> "Be direct and concise. Avoid repeating information from the context verbatim. Synthesize rather than quote."

### 1.5 Conditional Visualization

**Current:** `enhance_with_visualization()` runs on every response (`src/main.py:1008-1014`), adding 50-500ms.

**New behavior:** Only trigger visualization when:
1. User explicitly requests it (keywords: chart, graph, plot, visualize, "show me a chart")
2. Intent is `compare` or `aggregate` AND response contains 3+ rows of numeric data

**Implementation:**
- Add `wants_visualization` flag resolved during intent classification (UNDERSTAND step)
- Pass through to post-generation step
- Guard the `enhance_with_visualization` call with this flag
- Remove blanket `<!--DOCWAIN_VIZ-->` instructions from individual `TASK_FORMATS` entries
- Add single conditional directive in `_SYSTEM_PROMPT`:
  > "VISUALIZATION: Only append `<!--DOCWAIN_VIZ-->` directives when the user explicitly requests a chart/graph, OR when your response contains a comparison/aggregation table with 3+ rows of numeric data."

### 1.6 Expected Combined Latency

| Scenario | Before | After |
|----------|--------|-------|
| Time-to-first-token (streaming) | 30-50s (blocked) | ~100ms |
| Typical query (500 tokens) | 15s | 4-5s |
| Short lookup (200 tokens) | 8s | 1.5-2s |
| Complex analysis (1500 tokens) | 35s | 12-15s |

---

## Part 2: Response Quality — Enriching the Fast Path

### 2.1 Problem

The fast path (`src/execution/fast_path.py`) handles queries classified as SIMPLE but produces shallow 1-2 sentence answers because it:
- Uses only top-3 chunks capped at 4,000 chars
- Has a bare system prompt ("be concise and accurate")
- Ignores document intelligence, entities, key facts, KG context
- Ignores conversation history
- Uses a fixed `max_tokens=1024` and `temperature=0.1`

### 2.2 Changes

| Setting | Current | New |
|---------|---------|-----|
| `_FAST_PATH_RERANK_K` | 3 | 5 |
| `_MAX_CONTEXT_CHARS` | 4,000 | 8,000 |
| System prompt | Bare "be concise" | Use `build_system_prompt()` from `prompts.py` with profile expertise |
| Task format | None | Apply `TASK_FORMATS[task_type]` based on classified intent |
| Doc intelligence | None | Include doc summaries + entities if available |
| `max_tokens` | 1024 | Use Reasoner's `_BASE_TOKENS` budget based on task type |
| `temperature` | 0.1 | 0.3 |

### 2.3 SIMPLE/COMPLEX Classification

**Current:** Too many analytical queries classified as SIMPLE, hitting the thin fast path.

**Add to UNDERSTAND prompt** (`src/generation/prompts.py`):
> "COMPLEXITY GUIDE: Only classify as 'simple' if the query asks for a single, specific fact (a name, date, amount, yes/no). Questions about risks, implications, processes, recommendations, or 'what should I know about' are ALWAYS 'complex'."

### 2.4 Depth Instructions

**Add to `_SYSTEM_PROMPT`** in `src/generation/prompts.py`:
> "DEPTH OVER BREVITY: When answering analytical questions, provide substantive analysis — not just what the document says, but what it means for the user. Connect dots across evidence. Highlight implications, risks, or opportunities that a domain expert would notice."
>
> "MINIMUM SUBSTANCE: Every response must provide actionable insight. If the evidence supports a detailed answer, give one. Never reduce a rich evidence base to a single sentence unless the question is purely factual."

---

## Part 3: Adaptive Expert Intelligence

### 3.1 Vision

DocWain reads the documents in a profile and *becomes* the expert those documents demand — not through hardcoded domain templates, but through emergent understanding. Troubleshooting manuals make it a support engineer. Legal contracts make it a legal advisor. Research papers make it a research analyst. One pipeline, no domain-specific code paths.

### 3.2 Architecture

```
Document Upload → Extraction → Understanding → Embedding
                                                    ↓
                                         [Background, non-blocking]
                                                    ↓
                                    Phase 1: Profile Understanding
                                     (single LLM call, smart path)
                                                    ↓
                                    Phase 2: Deep Expert Analysis
                                     (per-doc-cluster LLM calls, smart path)
                                                    ↓
                                    Store: MongoDB profile_expertise collection
                                                    ↓
                              Inject into system prompt + query context at runtime
```

### 3.3 Phase 1: Profile Understanding

**Trigger:** After embedding completes for a profile (new docs added or docs removed).

**Input:** All document summaries + entities + key facts from existing `intelligence_v2` pipeline (already computed, no re-extraction needed).

**LLM call** (smart path, 27B, background priority):

```
You are analyzing a collection of documents to determine:

1. EXPERTISE IDENTITY: What kind of expert would someone need to be to 
   deeply understand and advise on these documents? Describe the expert's 
   role, mindset, and what they'd prioritize.

2. KNOWLEDGE MAP: What are the key knowledge areas covered? What can 
   someone be definitively advised on from these documents?

3. PROACTIVE INSIGHTS: As this expert, what would you immediately tell 
   someone who handed you these documents? What stands out? What needs 
   attention? What opportunities or risks exist?

4. ADVISORY CAPABILITIES: What kinds of questions can you answer with 
   authority? What kinds of guidance can you proactively offer?

5. KNOWLEDGE GAPS: What's NOT covered that a user might expect? Where 
   should the user be cautioned that you can't advise?
```

**Output:** `profile_expertise` document (see Section 3.5 for schema).

### 3.4 Phase 2: Deep Expert Analysis

**Trigger:** After Phase 1 completes.

**Process:** Using the expertise identity as framing, analyze document clusters (grouped by topic via existing cross-document links) to produce:

- **Connections** documents don't explicitly make ("The maintenance guide contradicts the warranty terms in Document B")
- **Implications** a user might miss ("The SLA guarantees 99.9% uptime but repair cycles average 4 hours — only 8.7 hours downtime budget per year")
- **Actionable recommendations** ("Before calling support for E45-E52 errors, check relay connections at panel C3 — resolves 70% of cases based on the fault tree")
- **Anticipatory knowledge** ("Users who encounter X typically also need to know Y")

Each analysis call is scoped to a document cluster (not entire profile), keeping prompt size manageable.

### 3.5 Storage Schema

**Collection:** `profile_expertise` in MongoDB

```python
{
    "profile_id": str,
    "subscription_id": str,
    "expertise_identity": {
        "role": str,        # "Senior Technical Support Engineer for industrial HVAC"
        "mindset": str,     # "Diagnostic-first. Safety > uptime > efficiency."
        "tone": str         # "Practical, direct, solution-oriented"
    },
    "knowledge_map": [
        {
            "area": str,          # "Error code diagnostics"
            "depth": str,         # "comprehensive" | "detailed" | "partial" | "minimal"
            "document_ids": [str]
        }
    ],
    "proactive_insights": [
        {
            "category": str,      # "critical" | "important" | "informational"
            "insight": str,       # The expert observation
            "recommendation": str, # What to do about it (optional)
            "evidence_refs": [str] # document_id references
        }
    ],
    "advisory_capabilities": [str],  # What this expert can help with
    "knowledge_gaps": [str],          # What's NOT covered
    "deep_analysis": [
        {
            "cluster_topic": str,
            "connections": [str],
            "implications": [str],
            "recommendations": [str]
        }
    ],
    "document_ids_analyzed": [str],   # Track what was included
    "version": int,                    # Increment on re-analysis
    "created_at": datetime,
    "updated_at": datetime
}
```

### 3.6 Runtime Integration

**System prompt injection** — modify `build_system_prompt()` in `src/generation/prompts.py`:

```python
def build_system_prompt(profile_domain="", kg_context="", profile_expertise=None):
    if profile_expertise:
        identity = profile_expertise["expertise_identity"]
        prompt = (
            f"You are a {identity['role']}.\n"
            f"Your approach: {identity['mindset']}\n"
            f"Communication style: {identity['tone']}\n\n"
        )
    else:
        prompt = (
            "You are a senior subject matter expert analyzing documents "
            "for a professional.\n\n"
        )
    prompt += _SYSTEM_RULES  # existing rules 1-12
    ...
```

**Query context injection** — in `build_reason_prompt()`, add expert insights as a new section between DOCUMENT INTELLIGENCE and EVIDENCE:

```
--- EXPERT ANALYSIS ---
[Filtered proactive_insights + deep_analysis relevant to the query]
--- END EXPERT ANALYSIS ---
```

Filtering: match expert insights to the query using entity overlap and knowledge_map area matching. Only include relevant insights — not the entire expert analysis.

**Fast path integration** — the fast path loads `profile_expertise` from MongoDB (cached in app state) and uses the same enriched system prompt. No separate code path needed.

### 3.7 Latency Protection

Background analysis must NEVER affect the response pipeline:

1. **GPU priority:** Background analysis uses smart path (27B, port 8200) with lower priority. User queries always go through fast path (14B, port 8100). No resource contention.
2. **Async execution:** Triggered via background task after embedding completes. Uses existing `asyncio` task spawning — no blocking of API endpoints.
3. **Queue with backpressure:** During bulk uploads (10+ docs), queue expert analysis calls and process sequentially with a configurable delay between calls (default: 2s). Never flood the smart path.
4. **Cached at runtime:** `profile_expertise` is loaded from MongoDB once per session and cached in `app_state`. No per-query DB call — just a dict lookup.
5. **Graceful degradation:** If profile_expertise is not yet computed (new profile, analysis still running), fall back to the current generic system prompt. No error, no delay.
6. **Query-time cost:** Zero additional LLM calls. Expert intelligence is pre-computed and injected as prompt context alongside existing document intelligence.

### 3.8 Staleness & Re-trigger

- When documents are added/removed from a profile, compare `document_ids_analyzed` with current document list
- If different, mark expertise as stale and re-trigger background analysis
- **Incremental:** When 1-2 new docs are added, run Phase 2 only on new docs, then update Phase 1 synthesis with the delta
- **Full rebuild:** When >30% of docs change, re-run both phases from scratch

### 3.9 Proactive Greeting (Optional, UI-driven)

When a user starts a new session on a profile with computed expertise, the UI can display the top 3-5 proactive insights as a greeting card before the user types anything.

**API:** New endpoint `GET /api/profile/{profile_id}/insights` returning the top insights from `profile_expertise.proactive_insights` sorted by category (critical first).

**UI integration:** `Home.tsx` fetches insights on profile selection and displays them in the empty state (replacing "Ask anything about documents...").

---

## File Impact Summary

| File | Changes |
|------|---------|
| `/etc/systemd/system/docwain-vllm-fast.service` | fp8, chunked prefill, speculative decoding |
| `src/generation/prompts.py` | Depth instructions, conditional viz directive, dynamic system prompt with expertise identity, expert analysis context section |
| `src/generation/reasoner.py` | Halved token budgets, reduced thinking multiplier |
| `src/execution/fast_path.py` | Enriched context (5 chunks, 8K chars), rich system prompt, task formats, doc intelligence |
| `src/main.py` | Conditional visualization guard |
| `src/intelligence_v2/expert_intelligence.py` | **NEW** — Phase 1 + Phase 2 analysis pipeline |
| `src/intelligence_v2/profile_builder.py` | Trigger expert analysis after profile build |
| `src/api/app_lifespan.py` | Cache profile_expertise in app_state |
| `src/api/query_intelligence.py` or execution pipeline | Load + filter expert insights for query context |
| `src/api/routes` (or main.py) | New `/api/profile/{profile_id}/insights` endpoint |

---

## Non-Goals

- No hardcoded domain templates — expertise emerges from documents
- No separate code paths per domain
- No changes to the existing `computed_profiles` or `collection_insights` pipelines — expert layer builds on top of them
- No real-time expert analysis at query time — everything is pre-computed
- No changes to the Teams standalone service
