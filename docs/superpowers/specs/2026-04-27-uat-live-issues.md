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

**Scope update 13:36 UTC:** This same Mongo-timeout class is also firing inside the Celery `profile_intelligence` task (not just the API endpoint). Example: `Profile intelligence: failed for doc=69ef6402... profile=69ef1d72...`. Background-task failures are **silent** to the UAT user — the profile they uploaded will sit without an intelligence report until somebody re-dispatches the task. **Recovery action for ops:** monitor `/api/profiles/{profile_id}/intelligence` for `status=pending` longer than 5 minutes and re-trigger via `POST /api/profiles/{profile_id}/intelligence/regenerate` (the existing endpoint at `src/api/profile_intelligence_api.py:49`). The fix is the same as #5's: raise `serverSelectionTimeoutMS` and add task-level retry.

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

---

# Findings reported by testers (consolidated 2026-04-27)

Source files reviewed:
- `Contracts- Observations and Defects(Testcase).csv` — 21 contract extraction tests (10/8/3 pass/fail/blank)
- `Docwain_test(Sheet1).csv` — 4 cross-cutting defects (consolidated)
- `Docwain_test_1.xlsx` — 10 health-insurance tests (3 pass, 7 fail) + 21 contract tests
- `Docwain Testing Feedback.docx` — observations from 6 testers (Anmol, Pavithra, Rajasekar, Rajesh, Avyaktha, Sreekanth)

---

## Issue #6 — Multi-document tagging / screening sync gap (4 testers, highest-frequency)

**Reported by:** Anmol, Avyaktha, Rajasekar, Rajesh — independently
**User-visible symptom:** docs upload + tag, but only some appear in downstream surfaces:
- Right-side document list shows fewer docs than uploaded (Anmol: 3 uploaded, 1 visible)
- Screening module misses recently-added docs (Avyaktha: Offboarding worked, then Flexible Working + Resignation Letter "do not appear in screening interface")
- Tag-and-Train shows only 1 doc out of 11 (Rajasekar)
- Screening Reports tab empty even though screening_status=COMPLETED (Rajesh)

**Likely root cause:** correlates with **Issue #4** (race-condition marking docs `EMBEDDING_FAILED` while fallback succeeded) — affected docs are completed in some collections but missing from screening/tag-train indexes. May also involve Mongo write-during-timeout (#5).
**Severity:** **Highest of tester-reported set** — blocks core multi-doc workflows.

## Issue #7 — Premium table / chart extraction failure on health-insurance docs

**Reported by:** Health-insurance UAT (`Docwain_test_1.xlsx`)
**User-visible symptom:** 7 of 10 tests fail because DocWain claims premium amounts are not in the document, when they are. Examples:
- "give me details of the premium in health shield plan" → "premium details are absent"
- "highest and lowest premium for plan health shield" → "Premium is not quantified" (expected: 7393 & 274287)
- "premium range for >80 years if sum insured 500000" → "no premium calculations" (expected: 76523, 143880)
- "analyse the premium chart" → "documents do not contain a premium chart" (it does)
- "compare plans … less premium for age <40" → "Wrong result with dollars" (currency confusion)

**Likely root cause:** premium data is in a **table image / multi-column scanned grid** that the current extraction pipeline misses (page-level OCR doesn't structure the grid). Once extraction misses it, retrieval can't find it; the LLM correctly says "not in evidence."
**Severity:** **High** — entire premium-related Q&A surface unusable for insurance docs.

## Issue #8 — Confident hallucination on contract field values (CRITICAL — wrong with citation)

**Reported by:** Contracts UAT
**User-visible symptom:** DocWain returns well-formatted, citation-backed answers that are factually wrong:

| Doc | Field | DocWain returned | Actual |
|---|---|---|---|
| VEND004 | Material | "MAT 003 — Dairy Supply" with "extracted directly from source" | MAT001 Frozen Foods |
| VEND004 | Payment terms | "Net 30 days, Code 0001 … confirmed in SOURCE-1" | Net 45 days, Code 0002 |
| VEND004 | Signature date | "Supplier signature date: 01/12/2025 (from SOURCE-1)" | Field literally contains "Sales Director" |

**Root cause:** model picks up similar-looking nearby spans when the actual field has unexpected content. The body-grounding validator catches *fabricated* claims but not *misquoted* spans — model IS quoting, just the wrong source segment.
**Severity:** **Critical** — wrong values delivered with confidence and citation prose; users won't catch it.

## Issue #9 — Cross-document / cross-source retrieval drops sources

**Reported by:** Anmol ("Multi-Document Retrieval"), Contracts UAT (Tests 16, 17, 18, 20)
**User-visible symptom:** two sub-cases:
- **Multi-doc same-profile:** "When asked about employee wellbeing and flexibility, [the bot] only referenced the Flexible Work Policy and ignored the Sick Leave Policy."
- **Cross-source:** comparing a contract doc with `sap_store_contracts` (structured source) — DocWain can't join.

**Likely root cause:** retrieval `top_k` is too small or biased toward one doc source; AND when DocWain *does* retrieve from multiple, prompt size hits **Issue #3** (32K overflow) and gets truncated, dropping later docs.
**Severity:** **High** — breaks the "research portal" multi-doc reasoning promise.

## Issue #10 — Conflict identified but not resolved

**Reported by:** `Docwain_test(Sheet1).csv`, Contracts Test 8 (VEND003 material consistency)
**User-visible symptom:** DocWain finds two conflicting statements, presents both side-by-side, and stops there — no judgment, no resolution:
- Sick Leave vs Offboarding: "Sick Leave Policy says unused sick leave is not paid out, while Offboarding Checklist suggests it may be included in final pay. The bot listed both without clarification."
- VEND003: returned "Goods means … Fresh Vegetables — MAT003" *and* "Material: MAT003 — Dairy Supply" verbatim, without flagging the contradiction.

**Likely root cause:** the chat reasoner's prompt has no "find contradictions" step. The Insights Portal v2 *has* a `conflict` insight type that does this — but `/api/ask` doesn't surface those persisted conflict insights.
**Severity:** **High for trust** — users may act on the wrong half of a conflict.

## Issue #11 — Consistency-check / "validate" prompts don't engage defect-detection mode

**Reported by:** Contracts UAT (Tests 7, 12)
**User-visible symptom:** prompts beginning "Validate …" or "Check if … is correct" elicit a quote-and-confirm response, not a defect-search:
- Test 7 (VEND002 signature block): expected to flag that signature says "VEND001 Ltd" in a VEND002 contract. DocWain returned "all fields match the source" (the printed value matches itself; the model didn't compare against the contract's own header).
- Test 12 (VEND004 signature date): expected to flag that the date field contains "Sales Director" (wrong type). DocWain made up "01/12/2025 from SOURCE-1."
**Severity:** Medium (sibling to #8; prompt-mode root cause).

## Issue #12 — Chat history not filtered by profile

**Reported by:** Anmol (Problem 5)
**User-visible symptom:** "After choosing the profile in chat history it is showing all the chats of my id irrespective of the profile."
**Severity:** Medium — usability, not data integrity.

## Issue #13 — Visualizations / charts disappear in chat history

**Reported by:** `Docwain_test(Sheet1).csv` row 4, Anmol's table comment, health-insurance Test 3 ("visualisation is not relevant")
**User-visible symptom:** charts render in-session but vanish on revisit. When they do render, sometimes irrelevant or wrong-currency.
**Severity:** Medium — demo polish; trust hit when wrong currency.

## Issue #14 — "View Report" button broken / Screening Reports tab empty

**Reported by:** Anmol (Problem 4), Rajesh
**User-visible symptom:** two adjacent bugs:
- Anmol: clicking "view report" doesn't work
- Rajesh: tab itself is empty for a doc whose status reads "Screening Completed"

**Likely root cause:** could be the same as #6 (sync gap) hitting the report query path — the screening result row exists but is keyed differently from how the UI queries it.
**Severity:** Medium.

## Issue #15 — No loading / progress indicator during long operations

**Reported by:** Anmol (Problem 2)
**User-visible symptom:** "There should be some animation like loading/processing so that the end user is aware that the process is running in background not getting stuck and also if possible it should display estimated time based on the file size or number of files to upload."
**Severity:** Medium — UX trust.

## Issue #16 — Document overview is too generic

**Reported by:** Health-insurance Test 10
**User-visible symptom:** "It gives just a generic idea of the document with very less information." Expected plan names / specific numerics.
**Severity:** Medium — first-contact experience for a new doc set is shallow.

---

## Consolidated tracker — all 16 issues

| # | Severity | Source | Status | Component | Title |
|---|---|---|---|---|---|
| 1 | Medium | log | OPEN | dataHandler.delete_embeddings:1298 | DELETE /embeddings 500 on collection-missing |
| 2 | High | log | OPEN | gateway/api.py:126 | Screening category not normalised |
| 3 | **Critical for swap** | log | OPEN | llm/clients.py | vLLM context overflow on long prompts |
| 4 | High | log | OPEN | embedding_service (race) | Doc EMBEDDING_FAILED while fallback succeeded |
| 5 | High | log | OPEN | dataHandler (5s ssTimeout) | CosmosDB transient drops surface as 500 |
| 6 | **Highest of tester reports** | tester ×4 | OPEN | embedding/screening sync | Multi-doc tag/sync gap |
| 7 | High | tester | OPEN | extraction (table layer) | Premium tables not extracted (insurance) |
| 8 | **Critical** | tester | OPEN | extraction precision + reasoner | Confident hallucination on contract values |
| 9 | High | tester | OPEN | retrieval top_k + ctx clamp | Multi-doc retrieval drops sources |
| 10 | High | tester | OPEN | reasoner prompt | Conflict identified but not resolved |
| 11 | Medium | tester | OPEN | reasoner prompt | "Validate" prompts don't engage defect mode |
| 12 | Medium | tester | OPEN | UI / chat history | Chat history not filtered by profile |
| 13 | Medium | tester | OPEN | UI / visualizations | Charts vanish in history; wrong currency |
| 14 | Medium | tester | OPEN | UI / screening reports | View Report button doesn't open |
| 15 | Medium | tester | OPEN | UI / progress indicator | No loading animation / ETA |
| 16 | Medium | tester | OPEN | reasoner prompt — overview | Document overview too generic |
