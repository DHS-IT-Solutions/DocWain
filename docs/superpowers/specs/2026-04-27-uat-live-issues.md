# UAT Live Issues — captured 2026-04-27

**Mode:** read-only monitoring. **No service restarts** per direction.
Issues are captured here with proposed fixes; nothing is patched live.

---

## Issue #1 — `DELETE /api/document/{id}/embeddings` returns 500 for any doc that was never embedded

**First seen:** 2026-04-27 05:18:59 UTC
**Frequency:** ~22 attempts in the past hour, all 500s, ~270–2600 ms each
**Source IP:** 20.31.70.131 (UAT tester)
**User-visible symptom:** UI batch-delete returning 500 for some/all docs

**Root cause:** `src/api/dataHandler.py:1298`. The `delete_embeddings` exception handler swallows Qdrant collection-missing errors *only* when the message matches:
```python
if "doesn't exist" in error_msg or "not found" in error_msg.lower() or "Not Found" in error_msg:
```
The actual Qdrant error string is `"collection does not exist"` (no apostrophe). The substring check misses, the exception falls through to `return {"status": "error", ...}`, the handler at `src/main.py:1665` raises `HTTPException(500)`.

**Reproduction (live):**
```
$ curl -s -X DELETE http://localhost:8000/api/document/69eb1797af9231725f584a82/embeddings
{"error":{"code":"500","message":"collection does not exist","details":{}}}
```

**Proposed fix (1 line, deferred until next restart window):**
```python
# src/api/dataHandler.py:1298
if "doesn't exist" in error_msg or "does not exist" in error_msg or "not found" in error_msg.lower() or "Not Found" in error_msg:
```

**Severity:** Medium. The user intent (delete embeddings) is *already satisfied* when the collection doesn't exist — there's nothing to delete. UAT testers see error toasts but no actual data corruption. After the fix, these calls return `{"status": "ok", "message": "no embeddings to delete"}`.

**Workaround for UAT testers right now:** the operation has already succeeded conceptually; ignore the toast and proceed.

---

## Issue #2 — Screening category names not normalised before reaching engine (consolidates #2 & #3)

**First seen:** 2026-04-27 05:56:35 UTC; recurring (3+ instances seen in 60 s window)
**User-visible symptom:** UAT tester hits `POST /api/gateway/screen` with any of these category strings — all fail with `ValueError("No screening tools found for category '<X>'")`:
- `category="AI Authorship"` → expected canonical name (per `src/intelligence/usage_help.py:388`); fails because tool's `category` attribute is `"AI Authorship Likelihood"`
- `category="All"` (capital A) → fails because the `_resolve_tools_for_category` path checks for lowercase `"all"`; the executor's `if category == "all"` short-circuit at `unified_executor.py:266` is also case-sensitive

**Root cause (single):** `src/gateway/api.py:126` passes `request.category` directly to the executor without calling `normalize_categories` from `src/screening/helpers.py:48`. That normaliser already does:
- `lower().strip().replace(" ", "_").replace("-", "_")`
- maps `"AI Authorship"` → `"ai_authorship"`, recognises `"All"` → `"all"`
- raises `ValueError("Unsupported category 'X'")` for unknowns

It is implemented but **never called** by the gateway. Without it, raw user input like `"AI Authorship"` and `"All"` passes through to a case-sensitive lookup.

**Severity:** **HIGH for UAT.** All non-lowercase / non-snake_case category requests fail. Affects every UAT tester running the standard screening UI which sends human-readable category names.

**Proposed fix (deferred — needs restart):**
Two complementary changes:
1. Wire `normalize_categories` in `src/gateway/api.py`:
```python
# Before passing to executor:
from src.screening.helpers import normalize_categories
normalized_cats = normalize_categories(request.category)
result = await _executor.execute_screening(
    categories=normalized_cats,
    ...
)
```
2. Align the AI Authorship tool's category attribute with the canonical name:
```python
# src/screening/tools/ai_authorship.py:13
category = "AI Authorship"   # was: "AI Authorship Likelihood"
```
And/or add `"ai_authorship": "AI Authorship Likelihood"` as an alias in the tool resolver.

**Workaround for UAT testers right now:**
- For "All" → send `category="all"` (lowercase) or `category="run"`
- For "AI Authorship" → send `category="AI Authorship Likelihood"` verbatim, or temporarily skip and continue with PII / Readability / Plagiarism / Bias / Compliance / Integrity.

---

## Issue #3 — vLLM context overflow: prompts can exceed `max_model_len` (CRITICAL for 16 GB swap)

**First seen:** 2026-04-27 07:03:06 UTC; recurring (3+ instances in 3 s window from correlation `58cfc0b`)
**User-visible symptom:** `/api/ask` (or any LLM-bearing path) fails after several retries with `HTTP Error 400: Bad Request` propagated from vLLM.

**Concrete error from vLLM:**
```
vllm.exceptions.VLLMValidationError:
  This model's maximum context length is 32768 tokens.
  However, you requested 15360 output tokens and your prompt contains
  at least 17409 input tokens, for a total of at least 32769 tokens.
```

Off by exactly 1 token. App-side log: `src.llm.clients - All local LLM retry attempts failed`.

**Root cause:** Upstream prompt construction does not budget input + output against `max_model_len`. Specifically:
- Reasoner/researcher build prompts that grow with document size, RAG context, KG context, and conversation history.
- `max_tokens` defaults are static (apparently 15,360 in this case — very generous).
- No dynamic clamping like `max_tokens = min(requested, max_model_len - input_tokens - safety_margin)`.

**Severity:** **HIGH today, CRITICAL after 16 GB GPU swap.**
- Today: occurs on a small fraction of requests with very large input contexts.
- Post-swap: `max-model-len=8192` (per AWQ runbook). ANY request with >7K total tokens would fail. That's *most* multi-doc queries.

**Proposed fix (deferred — needs restart):**
Add an `_clamp_max_tokens` helper in `src/llm/clients.py` that computes the safe ceiling per call:
```python
def _clamp_max_tokens(prompt_tokens: int, requested: int, *, ctx_window: int, safety: int = 64) -> int:
    available = max(0, ctx_window - prompt_tokens - safety)
    return min(requested, available)
```
Apply at every LLM call site. Pre-validate; if clamped value <= 0, truncate the prompt at the upstream retrieval / RAG budget before sending.

**Workaround for UAT testers right now:**
- Avoid asking questions about extremely long single-document profiles. Multi-doc + long retrieval contexts are the trigger.
- Re-asking the same question often succeeds because the conversation history / RAG hits a different size on retry.
- The Insights Portal v2 dashboard / endpoints are unaffected — they don't invoke the LLM at query time.

**Pre-swap action item (must address before 16 GB go-live):** before flipping the systemd unit to the 16 GB profile, ship Issue #3 fix. Otherwise the swap will degrade quality severely.

---

## Issue tracker (live)

| # | First seen | Severity | Component | Status | Title |
|---|---|---|---|---|---|
| 1 | 05:18 | Medium | dataHandler.delete_embeddings:1298 | OPEN — fix queued | DELETE /embeddings 500 on collection-missing |
| 2 | 05:56 | **High** | gateway/api.py:126 (no normalize call) | OPEN — fix queued | Screening category not normalised; "AI Authorship", "All" fail |
| 3 | 07:03 | **Critical for 16 GB swap** | llm/clients.py (no max_tokens clamp) | OPEN — must-fix-before-swap | vLLM context overflow on long prompts |
| 4 | 07:24 | High | embedding_service (concurrent attempts + lease) | OPEN — fix queued | Doc marked EMBEDDING_FAILED while fallback actually succeeded |
| 5 | 08:37 | High | dataHandler.create_mongo_client (5s ssTimeout) | OPEN — fix queued | CosmosDB transient drops bubble as 500s on legacy endpoints (v1 intelligence) |
| _ | _ | _ | _ | _ | (more issues appended below as the monitor surfaces them) |

---

## Issue #5 — CosmosDB transient timeouts surface as 500s on user-facing endpoints

**First seen:** 2026-04-27 08:37:48 UTC; recurring **11 times in last 10 min**
**User-visible symptom:** UAT tester hits `GET /api/profiles/{id}/intelligence` (the existing v1 endpoint), gets 500 with `ServerSelectionTimeoutError`. Refresh succeeds because the connection is back by then. Confusing UX.

**Root cause:** `dataHandler.py:285` creates `MongoClient(primary_uri, serverSelectionTimeoutMS=5000)`. The 5 s budget is too aggressive for a globally-distributed CosmosDB cluster that occasionally takes 7–10 s to recover from a topology change. PyMongo gives up, the legacy endpoint bubbles the exception.

**Why my recent fix doesn't help here:** I added a lazy collection resolver to `MongoIndexBackend` (Insights Portal, Issue #5 from yesterday) so the v2 endpoints survive these drops. The v1 `/api/profiles/{id}/intelligence` endpoint still uses the eager module-level `db = mongoClient[Config.MongoDB.DB]` from `dataHandler.py:314`, with no lazy re-resolve and no retry.

**Severity:** **High.** UAT testers see random 500s on the intelligence endpoint. The pattern is "every ~minute, a few requests fail; later requests succeed." Mongo health endpoint stays green because it uses a different short-lived ping.

**Proposed fixes (deferred — needs restart):**
1. **Quick mitigation:** raise `serverSelectionTimeoutMS` from 5000 → 20000 to absorb topology-change windows. One-line change in `dataHandler.py:285`.
2. **Per-endpoint retry:** wrap the legacy intelligence endpoint in a 1-retry-with-backoff decorator (200 ms/500 ms), so a transient drop doesn't surface to the user.
3. **Long-term:** apply the lazy-resolver pattern (used in Insights Portal v2) to all dataHandler accessors so module-level captured clients can be refreshed transparently.

**Workaround for UAT testers right now:**
- If a profile-intelligence call returns 500, **refresh the page once** — the next call almost always succeeds (the cluster recovers in <10 s).
- The Insights Portal v2 dashboard at `/api/profiles/v2/{id}/insights` is more resilient (lazy resolver was added yesterday) and tends to absorb these blips invisibly.

---

## Issue #4 — Race condition: embedding marked FAILED when fallback succeeded

**First seen:** 2026-04-27 07:24:55 UTC (doc `69ef0e63af9231725f586a58`, "complete-health-insurance-brochure.pdf")
**User-visible symptom:** UAT tester sees status = `EMBEDDING_FAILED` on a doc that actually has clean extraction (45,080 chars, 199 expected chunks); they retry or assume bad doc.

**Timeline (reconstructed from logs):**
```
07:24:23  embedding attempt #1 starts (request_id=75f7d150)
07:24:24  attempt #1 downloads pickle (230,931 bytes)
07:24:25  status → EMBEDDING_IN_PROGRESS
07:24:27  attempt #1 detects incomplete pickle (coverage 0.39); kicks source-file fallback
07:24:47  attempt #2 starts (request_id=7f347169) — RACE
07:24:47  attempt #2 hits Lease conflict, retries
07:24:54  attempt #1 finishes fallback, saves versioned blob (lease unavailable; uses versioned write)
07:24:55  attempt #1 fallback assessment: coverage=1.0, total_chars=45080  ← SUCCESS
07:24:55  status → EMBEDDING_FAILED                                       ← but marked failed
07:24:55  attempt #1 logs "embedding end status=SKIPPED"
07:25:02  pre-check shows expected_chunks=199 (extraction is fine)
```

**Root cause:** Two concurrent embedding attempts trigger on the same doc (likely an over-eager retry or duplicate Celery dispatch). Attempt #2 contends for the blob lease while attempt #1 is mid-fallback. The lease-conflict path in attempt #2 ends up writing the FAILED status even though attempt #1's fallback completed successfully. Read the chain:
- `embedding_service.py` retries on lease conflict (3 attempts × backoff 1s/2s/4s)
- After retries it gives up and marks the doc EMBEDDING_FAILED
- But the *first* concurrent invocation already produced a clean fallback artifact and the doc is actually embeddable

**Severity:** **High** for UAT — false-failure UX, will cause testers to discard valid documents.

**Proposed fix (deferred — needs restart):**
Two options:
1. **Idempotency guard:** at task entry, check if status is already `EMBEDDING_IN_PROGRESS` and bail (don't re-dispatch). Eliminates the race entirely.
2. **Lease-loser path:** when attempt #2 gives up on the lease, treat it as "another worker is handling it" → log + return without setting any failure status. The other worker will set the final status.

Option 2 is safer for UAT-day; Option 1 should also be applied to prevent the race in the first place.

```python
# Sketch in src/api/embedding_service.py near line where Lease conflict gives up:
except LeaseConflict:
    logger.info("doc=%s embedding handled by another worker; this attempt yields", doc_id)
    return {"status": "yielded", "reason": "lease_owned_by_other_worker"}
    # do NOT call _set_status(doc_id, "EMBEDDING_FAILED")
```

**Workaround for UAT testers right now:**
- Doc is recoverable. Check status — if `EMBEDDING_FAILED` AND extraction looks clean (coverage=1.0 in logs), retry the embedding via `POST /api/documents/embed`. Or wait — the second attempt may have set the bad status even though the first succeeded; a fresh embed call should pick up cleanly.
- Tester-facing message could be: "If a doc shows EMBEDDING_FAILED but you can see chunks/extraction is fine, click retry once. The pipeline is recovering."

---

## Notes on observed UAT activity

- One UAT tester (20.31.70.131) actively testing batch operations (~22 docs in a tight window).
- Recent successful calls also visible in journal: `POST /api/gateway/screen → 200 (27.7s)` — screening is working.
- Tester's batch of doc IDs (69eb1797…, 69eb1798…) suggests fresh upload + bulk processing flow.
