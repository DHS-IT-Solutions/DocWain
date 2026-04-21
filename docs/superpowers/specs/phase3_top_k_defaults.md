# Phase 3 top-K defaults — 2026-04-21

Canonical per-intent / per-layer top-K values live in
`src/retrieval/top_k.py` (`_BASE_TABLE`). Task 13 of the Phase 3 plan
reserves this document for data-driven tuning notes after Task 12's
live eval run.

## Current defaults (unchanged from plan Task 6)

| Intent family | Layer A | Layer B | Layer C |
|---|---|---|---|
| `greeting`, `identity`, `meta`, `farewell`, `thanks` | 0 | 0 | 0 |
| `lookup`, `count` | 5 | 0 | 0 |
| `extract`, `list`, `aggregate` | 10 | 0 | 2 |
| `compare`, `summarize`, `overview` | 12 | 5 | 5 |
| `analyze`, `diagnose`, `recommend`, `investigate` | 15 | 10 | 10 |

Complexity bumps (additive, clamped by adapter caps):
- `sub_queries >= 2` → Layer A +3
- `sub_queries >= 4` → Layer A +3 (total +6)
- `entities >= 3` → Layer B +2
- `entities >= 6` → Layer B +2 (total +4)
- `temporal_span_months >= 12` → Layer C +2
- `temporal_span_months >= 36` → Layer C +3 (total +5)

## Task 13 status — skipped pending live eval

Task 12's sandbox snapshot was recorded in dry-run mode (no live DocWain
API available in the dev env), so the per-intent top-K tuning signal
required to justify any change to `_BASE_TABLE` is not yet available.

Per plan Task 13 Step 5: *"If eval passes without tuning, skip this task
with a note."* — this document is that note. The defaults above ship
unchanged into Phase 4.

Re-opening Task 13 requires:

1. A real Phase 3 eval snapshot with per-intent faithfulness +
   `sme_artifact_hit_rate` breakdown (script at
   `scripts/sme_eval/run_sandbox.py` — replace `--dry-run` with a
   `--api-base-url` pointing at a sandbox subscription that has
   `ENABLE_SME_RETRIEVAL=true` and completed Phase 2 synthesis).
2. At least one intent with faithfulness in the 0.80–0.82 band OR
   `sme_artifact_hit_rate < 0.90`.
3. A delta within the guardrails (Layer C top-K ±5, adapter
   `retrieval_caps.max_pack_tokens` ±1000 per intent).

Any change beyond those guardrails is out of scope for Phase 3 and must
go through a Phase 4 or spec amendment.
