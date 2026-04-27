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

## Issue tracker (live)

| # | First seen | Severity | Component | Status | Title |
|---|---|---|---|---|---|
| 1 | 05:18 | Medium | dataHandler.delete_embeddings:1298 | OPEN — fix queued | DELETE /embeddings 500 on collection-missing |
| 2 | 05:56 | **High** | gateway/api.py:126 (no normalize call) | OPEN — fix queued | Screening category not normalised; "AI Authorship", "All" fail |
| _ | _ | _ | _ | _ | (more issues appended below as the monitor surfaces them) |

---

## Notes on observed UAT activity

- One UAT tester (20.31.70.131) actively testing batch operations (~22 docs in a tight window).
- Recent successful calls also visible in journal: `POST /api/gateway/screen → 200 (27.7s)` — screening is working.
- Tester's batch of doc IDs (69eb1797…, 69eb1798…) suggests fresh upload + bulk processing flow.
