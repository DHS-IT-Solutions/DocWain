# UAT Round-2 Readiness Plan

**Date prepared:** 2026-04-27
**Round-2 UAT:** 2026-04-28
**Branch:** `preprod_v03`
**Sister doc (issue list):** `2026-04-27-uat-live-issues.md`

This plan does three things:
1. **Fix plan** — sequence the 16 known issues into restart-window waves with concrete code changes, eval gates, and revert paths.
2. **Gap analysis** — what UAT round 1 *didn't* exercise that could break round 2.
3. **Round-2 readiness checklist** — what must be green before testers start tomorrow.

---

## 1. Fix plan — six waves

Constraints honoured throughout:
- `feedback_v5_failure_lessons.md` — single-flag revertible, validate scorers first
- `feedback_intelligence_rag_zero_error.md` — every batch mechanically verifiable
- `feedback_measure_before_change.md` — baseline + harness in place before quality work
- `feedback_no_customer_data_training.md` — synthetic-only for any prompt evals

Each wave is a single coordinated commit + restart. Waves are ordered for **smallest blast radius first**: the cheapest, most-mechanical fixes ship first; the prompt-engineering and extraction-quality work ships later under explicit eval gates.

### Wave A — Naming + handler fixes (lowest risk, ~30 min, fixes 2 issues)

| Issue | File:line | Change | Eval |
|---|---|---|---|
| #1 | `src/api/dataHandler.py:1298` | Add `or "does not exist" in error_msg` to the substring check | Hit `DELETE /api/document/{nonexistent_id}/embeddings` → expect 200 with `{"status":"ok","message":"no embeddings to delete"}` (was 500) |
| #2a | `src/gateway/api.py:126` | Wrap `request.category` in `normalize_categories(...)` before passing to executor | UAT-style request `category=["AI Authorship","All"]` → no `ValueError`; tools resolve correctly |
| #2b | `src/screening/tools/ai_authorship.py:13` | Change `category = "AI Authorship Likelihood"` → `category = "AI Authorship"` (or add an alias-resolver) | The "AI Authorship" category resolves to the AIAuthorship tool |

**Restart impact:** ~30 s of /api/ask + Celery downtime; existing Insights Portal flags preserved.
**Revert:** `git revert <wave-A-commit>`; flags + Mongo state untouched.

### Wave B — LLM client safety (Issue #3, must-fix-before-swap)

| Issue | File | Change | Eval |
|---|---|---|---|
| #3 | `src/llm/clients.py` | Add `_clamp_max_tokens(prompt_tokens, requested, ctx_window, safety=64)` that returns `min(requested, ctx_window - prompt_tokens - 64)`; if ≤0, raise a structured `PromptTooLargeError` *before* sending to vLLM. Apply at every call site (`generate`, `generate_with_metadata`, `chat_with_metadata`). | Synthetic test sends 30K-token prompt with `max_tokens=15360` → returns 503 with `prompt_too_large` (not 400 from vLLM); same call with 5K prompt unchanged. |
| #3.b | `src/intelligence/ask_pipeline.py` | Catch `PromptTooLargeError` and respond gracefully: truncate retrieval to fit, retry once, else respond with "Your question would require more context than DocWain can fit. Try narrowing to a single document or shorter time window." | Multi-doc query that previously 400'd → now responds with truncated context and a hint. |

**Why now (before round 2):** if a tester switches the GPU to 16 GB AWQ overnight (per the prepared swap path), today's 32K context window drops to 8K. Without #3, the bottom 30% of multi-doc queries break.
**Eval gate:** add `tests/perf/test_prompt_clamp.py` with three fixtures — under-budget, at-budget, over-budget. CI must pass before merge.
**Revert:** flag-gated under `LLM_PROMPT_CLAMP_ENABLED` (default true after gate).

### Wave C — Mongo resilience (Issue #5, broadens beyond v1 endpoint)

| Issue | File:line | Change | Eval |
|---|---|---|---|
| #5.a | `src/api/dataHandler.py:285` | `serverSelectionTimeoutMS=5000` → `20000` (CosmosDB topology-change windows are 7–10 s) | Inject 8 s blip via `iptables` rule on the test host; legacy intelligence endpoint must succeed instead of 500 |
| #5.b | `src/api/profile_intelligence_api.py` (the v1 GET handler) | Wrap the handler body in a 1-retry-with-200ms-backoff decorator on `ServerSelectionTimeoutError` | Same iptables blip → first call retries internally and succeeds; user sees no 500 |
| #5.c | `src/tasks/profile_intelligence.py` (Celery task) | Same retry decorator on the task entry point | Background failures stop being silent — task retries instead of marking pending forever |

**Why this matters for round 2:** today saw 11 instances in 10 min; some tester profiles silently never got their intelligence report. This must close before round 2 to avoid Issue #6's downstream symptoms.
**Revert:** revert single commit; the timeout bump is a config value, the retry decorator is an additive wrapper.

### Wave D — Embedding state-machine + sync gap (Issues #4 + #6)

This is the highest-leverage tester-reported wave: 4 testers complain about the same multi-doc sync gap, and Issue #4 is the leading suspect.

| Issue | File | Change | Eval |
|---|---|---|---|
| #4.a | `src/api/embedding_service.py` (lease-conflict path) | When the lease retries exhaust, **do not** mark `EMBEDDING_FAILED`. Log "yielded to other worker" and return `{"status":"yielded"}`. The first worker's success path remains the only setter of final status. | Two-worker race test fixture: dispatch two embed tasks for same doc concurrently; final status must be `EMBEDDING_COMPLETED`, never `EMBEDDING_FAILED` |
| #4.b | `src/api/embedding_service.py` (entry guard) | At task entry, read current status. If already `EMBEDDING_IN_PROGRESS`, return `{"status":"in_progress","skip_reason":"already_running"}` immediately. | Same fixture as 4.a; only one worker actually computes |
| #6.a | `src/api/screening_service.py` and the screening-reports query | Audit the screening-reports + tag-and-train queries: confirm they read all docs whose `screening_status` is COMPLETED (not just docs whose `embedding_status` is COMPLETED). Anmol+Rajasekar+Rajesh bugs all match the pattern of "screening done, but UI filters by embedding status which the race left wrong." | Manually create a profile with one doc in `EMBEDDING_FAILED` (per #4 race) and one in `EMBEDDING_COMPLETED`. Both have `screening_status=COMPLETED`. UI must show both in screening-reports + tag-and-train. |
| #6.b | `src/api/profile_intelligence_api.py` (regenerate endpoint) | Surface a tester-runnable "Recover stuck docs" admin endpoint that re-dispatches embedding for any doc in `EMBEDDING_FAILED` whose extraction artifact is clean (coverage ≥0.95). | Run on a profile pre-populated with a stuck doc; doc returns to `EMBEDDING_COMPLETED` |

**Round-1 cleanup before round 2:** *before* round-2 testers start, run the recovery endpoint over every UAT profile created today so leftover EMBEDDING_FAILED docs don't carry over.
**Revert:** all changes single-flag-revertible; old behaviour was "lease loser writes failed status" which we just stop doing.

### Wave E — Reasoner prompt upgrades (Issues #8, #10, #11, #16)

This wave touches `src/generation/prompts.py` only (per `feedback_prompt_paths.md` — response formatting belongs there, not in `intelligence/generator.py`).

| Issue | Change |
|---|---|
| #8 (anti-hallucination) | Add system-rule: *"If the document text in the cited span does not match the expected field type (e.g. a date field literally contains 'Sales Director' or other non-date text), report `field is unparseable: <verbatim quote>` instead of constructing a plausible value. Never invent a value to fill in a field whose actual content is unexpected."* |
| #10 (conflict resolution) | Add system-rule: *"When you find two or more spans of the document set that contradict each other, do NOT just list both. Mark them as `CONFLICT:` in the answer, propose the most likely correct interpretation, and explicitly say which document each conflicting span comes from."* |
| #11 (consistency-check mode) | Detect "validate / check / verify / consistent" verbs in the user query → switch to defect-search mode where the prompt instructs the model to *actively look for* contradictions, type mismatches, and rule violations rather than confirm what's printed. |
| #16 (richer overview) | When the query is "give me an overview / summarise this profile / explain what's in the doc set", the prompt now includes the profile's domain (from adapter detection) and instructs: *"List the named entities (plan names, vendor names, employee names), key numerics (premiums, prices, dates, terms), and the document type for each doc. No more than 6 sentences total but with concrete content."* |

**Eval gate (mandatory per `feedback_v5_failure_lessons.md`):**
- Before merge: re-run the contracts CSV (Tests 7, 8, 12) and the Sheet1 health-insurance tests against the new prompts. Pass criterion: Test 7 + Test 12 must flip from current-fail to "flagged correctly"; Test 16 (overview) must include the plan names "Health Shield" verbatim.
- After merge but before round-2: run a synthetic regression set (10 known-conflict, 10 known-anomaly, 10 normal) and confirm none of the *normal* cases newly hallucinate conflicts.

**Revert:** flag-gated under `REASONER_DEFECT_MODE_ENABLED`; off → previous prompts.

### Wave F — Extraction precision (Issues #7, #8 partial)

This wave is the deepest. Per `feedback_engineering_first_model_last.md`, prove at the prompt/retrieval layer first; only then change extraction.

| Issue | Change | Eval |
|---|---|---|
| #7 | The premium grid in the health-insurance brochure is a structured table that the current extraction loses. Two-step fix: (a) verify whether the existing table-extraction path runs on the source PDF; (b) if it does run but loses the grid, check the grid-detection threshold and the multi-column reading-order heuristic. If the table is in an image, route through the image-table path (already exists per spec Section 4 / DocIntel). | After fix: re-upload the brochure; `extracted_text` for that doc must contain the cell values 7393 and 274287 verbatim. Then re-run UAT Test 6 — must produce those numbers. |
| #8 (extraction precision component) | Add a "field-type post-validator" pass after extraction: for known fields (date, amount, code), regex-check that the extracted value matches expected type. If mismatch, mark field as `UNRESOLVED` with the verbatim doc text. The reasoner (post-Wave E) then reports this honestly instead of inventing. | Run VEND004 contract — Test 12 (signature date field has "Sales Director") must produce `signature_date: UNRESOLVED ("Sales Director")` not "01/12/2025". |

**Risk:** extraction changes can regress unrelated docs. **Eval gate:** must pass `tests/extraction_bench/` baseline before merge.
**This is the only wave that may not finish before round 2.** If it doesn't, mark Wave F as round-3 and ship A–E.

### Wave G — UI fixes (Issues #12, #13, #14, #15)

Front-end (and front-end-touching API) fixes. Mostly small.

| Issue | Likely file(s) | Change |
|---|---|---|
| #12 | Chat-history endpoint (likely `src/main.py` or `src/api/profiles_api.py`) | Filter chat history rows by `profile_id` query param when present; return only matching rows. |
| #13 | Visualization persistence — store `media[]` content in the chat-turn record (Mongo `chat_turns`) so it can be re-rendered on history load. Also fix the wrong-currency case by passing the doc's detected currency to the viz prompt. |
| #14 | "View Report" button → audit the screening-report fetch URL it hits; cross-reference with #6 fix (probably the same root cause). |
| #15 | Add a polling endpoint `/api/profiles/{id}/upload-status` that returns `{step: "extracting"\|"screening"\|"embedding", elapsed_ms, eta_ms}`. UI polls during upload and shows a progress bar. ETA can be a simple linear estimate from doc size. |

**Eval:** manual UAT walkthrough on each.
**Revert:** UI-only; back-out is a redeploy of the prior bundle.

---

## 2. Gap analysis — what UAT round 1 didn't capture

These are concerns I'd raise *before* round 2 because round-1 testing didn't exercise them and they could surprise tomorrow.

### 2.1 The Insights Portal v2 is invisible to UAT

None of the test files mention the `/api/profiles/v2/{id}/insights` endpoints, the proactive-injection "Related findings" section in `/api/ask`, the per-domain adapters (insurance, medical, hr, procurement, contract, resume), or the 25 seeded UAT demo profiles. **Yesterday's biggest engineering ship is not under test.**
**Mitigation:** add to round-2 script a dedicated section for the v2 dashboard (see §4 below).

### 2.2 Concurrent / load testing absent

Round 1 was single-tester sequential. We have *zero* evidence that:
- Two testers running screening on the same profile simultaneously work
- 10 concurrent `/api/ask` calls don't queue past the LLM batch capacity
- Researcher v2 burst (5 profiles × 9 insight types in parallel) doesn't OOM the GPU at 90% memory utilization

**Mitigation:** smoke a 5-concurrent /api/ask test before round 2 starts; capture p95.

### 2.3 Long-document edge cases

Round 1 docs were small contracts (8–10 pages) and one health insurance brochure. Untested:
- 100+ page docs
- Multi-language docs
- Docs with malformed tables or rotated pages
- Scans with low OCR quality

**Mitigation:** include 1 long doc and 1 OCR-heavy scan in round 2 corpus.

### 2.4 Auth / session / multi-tenant boundaries

No test verifies that user A cannot read user B's profile. No test verifies that an expired session re-auths cleanly mid-upload.
**Mitigation:** add a 2-account test pair to round 2; run one cross-account read attempt; expect 403.

### 2.5 GPU swap path

The 16 GB GPU swap procedure (AWQ artifact ready, runbook prepared) has been *simulated* on the A100 with reduced memory but never *actually deployed* on the 16 GB target. Round 2 should not be the swap-day; it should run on the A100 with bf16 to keep one variable controlled.
**Mitigation:** explicitly defer the swap to after round 2.

### 2.6 KG (Knowledge Graph) layer

Neo4j Insight nodes are written by researcher v2. Nothing in UAT tests whether KG queries (entity-walk, relationship traversal) actually serve answers in `/api/ask`. The proactive injection uses Mongo + Qdrant; KG sits unused at user-query time.
**Mitigation:** for round 2, log a single explicit KG-traversal probe per session; capture whether KG actually contributes to any answer.

### 2.7 Permission / size limits on uploads

No test of:
- Upload >50 MB doc — does it block, queue, or crash?
- Upload of forbidden file type (.exe, .zip)
- Upload of password-protected PDF
- Upload of corrupt PDF

**Mitigation:** add 4 negative-path tests to round 2.

### 2.8 Sustained operation

The longest live-monitor window today was ~12 hours. We don't know:
- Memory drift in vLLM after 24+ hours
- Disk fill from logs / artifacts
- Mongo collection growth and query degradation

**Mitigation:** check `df -h` and `nvidia-smi` baseline before round 2 starts; check delta after.

### 2.9 PII detection regression

Issue #2 disabled "All" / "AI Authorship" categories from screening today. PII detection was untouched, but no test confirmed it still works *after* the gateway shipped. PII is a customer-data-protection-critical category.
**Mitigation:** add a single PII-screening test as a smoke check to round 2 ground zero.

### 2.10 Feedback-loop closure

No test verified that submitting feedback on `/api/ask` (the existing thumbs-up/down or `/api/feedback` endpoint) actually persists and influences future runs.
**Mitigation:** add 1 explicit feedback-submit test.

---

## 3. Sequenced fix-plan timeline (target: green before round 2)

Round 2 starts tomorrow (2026-04-28). Working backwards from there, with realistic durations:

| Slot (UTC) | Activity | Owner |
|---|---|---|
| Today 18:00–20:00 | Implement Waves A, B, C, D as PRs on `preprod_v03` | engineering |
| Today 20:00–21:00 | Run full unit suite + new prompt-clamp + race-condition tests; confirm all green | engineering |
| Today 21:00–21:30 | **Coordinated restart window** (vLLM-fast + docwain-app + celery-worker). Tell testers in advance. Total downtime ~2 min including readiness probes. | ops |
| Today 21:30–22:00 | Post-restart smoke: hit each fix's eval criterion live; close issues on the tracker | engineering |
| Today 22:00–23:00 | Implement Waves E (reasoner prompts) and G (UI) | engineering |
| Today 23:00–23:30 | Second restart window for Wave E + G | ops |
| 00:00–02:00 | Wave F (extraction precision) IF time permits; otherwise mark for round-3 | engineering |
| 02:00–05:00 | Recovery sweep: run the "Recover stuck docs" admin endpoint over every UAT profile from today; pre-warm vLLM with a dummy prompt so first round-2 query isn't a cold start | ops |
| 05:00–07:00 | Final smoke walkthrough using the round-2 UAT script (§4) | engineering |
| 07:00 | Hand off to round-2 testers | — |

**If anything in Wave A–D fails its gate:** abort that wave; ship the others. Single-flag revertibility means we can land 3 of 4 waves without taking the 4th.

---

## 4. Round-2 UAT script — what testers should explicitly do

I'd hand testers this short checklist on top of their normal scenarios. It surfaces the new-engineering wins and the gaps from §2.

### Smoke ground-zero (do these FIRST, in this order)

1. `GET /api/health` — every component must report `healthy`. If not, escalate before testing.
2. Upload a 1-page synthetic PDF to a fresh profile. Confirm:
   - Doc appears in the right-side document list within 60 s
   - Status transitions UPLOADED → EXTRACTION_COMPLETED → SCREENING_COMPLETED → EMBEDDING_COMPLETED
   - PII category screening runs and reports a result
3. Open `GET /api/profiles/v2/{profile_id}/insights` for the same profile. Expect 0 insights initially.
4. Trigger profile_intelligence regenerate. Wait. Expect insights populated within 3 minutes.

### v2 dashboard (the new product surface, not tested in round 1)

5. Open one of the 25 seeded UAT demo profiles (`uat-demo-insurance-01-*`, `uat-demo-medical-01-*`, etc.).
6. Verify the dashboard shows:
   - Insights list with multiple types (anomaly, gap, recommendation, etc.)
   - At least one severity = `warn` or `critical`
   - Each insight card shows the headline + severity marker
7. Click into a single insight detail. Verify it shows: body text, evidence span quote, document_id reference.
8. Visualizations endpoint returns at least a timeline.

### `/api/ask` proactive injection (new feature)

9. Pick a seeded profile that has `warn`-severity insights.
10. Ask: *"Tell me about my situation."* Verify the response contains a `Related findings:` section with at least one `!` (warn) marker.
11. Ask: *"What is the policy renewal date?"* Verify the answer cites the doc directly, not a `Related findings` rescue.

### Multi-doc + cross-source (Issues #9, #10)

12. Upload two contracts that contradict on the same field (e.g. one says price = $10, another says $12).
13. Ask: *"What's the price?"* — expect the answer to flag the contradiction explicitly with `CONFLICT:` prefix, not just list both.
14. Ask: *"Compare these two contracts."* — expect both docs cited in the answer.

### Defect-detection mode (Issues #8, #11)

15. Upload a contract where the signature date field literally contains a non-date (e.g. "Director").
16. Ask: *"Validate the signature date."* — expect `field is unparseable: 'Director'`, not an invented date.

### Long-document + OCR (Gap #2.3)

17. Upload one 100+ page document. Verify extraction completes; expect to see chunk count in logs.
18. Upload one OCR-heavy scan (low-res photo of a page). Expect extraction completes with reasonable text recovery.

### Negative paths (Gap #2.7)

19. Upload a 60 MB doc — expect either successful processing or a clean error message (not a 5xx).
20. Try uploading a `.exe` file — expect 415 (unsupported media type).
21. Try uploading a corrupt PDF — expect a friendly error, not a stack trace.

### Concurrent (Gap #2.2)

22. Have two testers each open the same profile and ask different questions simultaneously. Both must receive answers within 30 s.

### Auth / cross-account (Gap #2.4)

23. Tester A creates a profile P; Tester B (different account, same subscription) attempts `GET /api/profiles/{P}/intelligence`. Expect 403 or empty (per access-control design).

---

## 5. Round-2 success criteria (gate for marking tomorrow "green")

- ≥ 18 of 23 round-2 script items pass on first attempt
- 0 occurrences of `VLLMValidationError` in the journal during the test window (Issue #3 closed)
- 0 occurrences of `ServerSelectionTimeoutError` propagated to a 5xx (Issue #5 closed)
- 0 docs marked `EMBEDDING_FAILED` whose source extraction has coverage ≥0.95 (Issue #4/#6 closed)
- AI Authorship + All screening categories accept human-readable input (Issue #2 closed)
- DELETE /embeddings on a never-embedded doc returns 200 (Issue #1 closed)
- v2 dashboard endpoints serve real data on at least 5 demo profiles (verifies yesterday's ship)
- One conflict-detection test produces a `CONFLICT:` marker (Issue #10 closed)
- One unparseable-field test produces `UNRESOLVED` (Issue #8/#11 partial, Wave E)

If any of these are red, we hold round 2 until that specific issue closes — no "ship and pray" given the V5 lessons.

---

## 6. Open risks I am NOT asking to fix this round

These are deliberate non-goals; flagging so nobody is surprised:

- **Wave F (extraction precision)** may slip to round 3. Premium-table extraction on health-insurance docs is real engineering, not a 1-hour fix.
- **GPU swap to 16 GB** is deferred until *after* round 2. Round 2 runs on the A100 with bf16. Swap goes in a separate planned window.
- **Cloud-397B failover path** is wired but not tested under simulated outage. Add to round 3.
- **Speculative decoding** (perf upgrade) is not pursued because it competes for GPU memory with the 16 GB swap target.
- **Insights Portal feedback loop** (user thumbs-down on an insight → adapter learns) is not built. Round 4+ if the product direction warrants.
