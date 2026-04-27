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

## Issue tracker (live)

| # | First seen | Severity | Component | Status | Title |
|---|---|---|---|---|---|
| 1 | 05:18 | Medium | dataHandler.delete_embeddings | OPEN — fix queued | DELETE /embeddings 500 on collection-missing |
| _ | _ | _ | _ | _ | (more issues will be appended below as the monitor surfaces them) |

---

## Notes on observed UAT activity

- One UAT tester (20.31.70.131) actively testing batch operations (~22 docs in a tight window).
- Recent successful calls also visible in journal: `POST /api/gateway/screen → 200 (27.7s)` — screening is working.
- Tester's batch of doc IDs (69eb1797…, 69eb1798…) suggests fresh upload + bulk processing flow.
