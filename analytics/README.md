# DocWain SME Analytics and Monthly Review Runbook

This directory holds the outputs of the Phase 6 pattern-mining loop:

- `sme_patterns_{YYYY-MM}.md` — the human-reviewable monthly findings
- `sme_patterns_{YYYY-MM}.json` — the machine-readable snapshot (same data)
- `training_candidates_{YYYY-MM}.json` — stabilized failure clusters flagged
  as candidates for sub-project F (separate, human-gated training project)
- `sme_rollback_{YYYY-MM-DD}.md` — post-mortems for any SME full rollback
  (spec Section 13.3); the next monthly report auto-links them under the
  "Rollback post-mortems" section
- `templates/` — Jinja2 source for the monthly Markdown

## Pipeline and schedule

A systemd timer (`systemd/docwain-sme-pattern-mining.timer`) invokes
`deploy/sme-pattern-mining.sh` at 02:00 UTC on the 1st of every month. The
wrapper runs:

1. `python -m scripts.mine_sme_patterns --analytics-dir analytics` — writes
   the monthly Markdown + JSON snapshot. The orchestrator also runs the
   training-trigger evaluator against all prior monthly JSON snapshots and
   writes `training_candidates_{YYYY-MM}.json` in the same directory.
2. `python scripts/evaluate_training_trigger.py --reports-dir analytics
   --out analytics/training_candidates_{YYYY-MM}.json` — cross-month
   stabilization check; re-emits the candidate list so operators can run it
   off-cycle with tuned thresholds without re-running the full miner.

Neither step triggers retraining. The list is evidence.

**Enable the timer:**

```bash
sudo cp systemd/docwain-sme-pattern-mining.service /etc/systemd/system/
sudo cp systemd/docwain-sme-pattern-mining.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now docwain-sme-pattern-mining.timer
```

Check timer status: `systemctl list-timers docwain-sme-pattern-mining.timer`.

## Reviewing the monthly patterns file

Open `sme_patterns_{YYYY-MM}.md`. Walk the six sections in order:

### 1. Executive summary
Quick gauges. If failure clusters spike 3x or more month-over-month, pause
and diagnose before continuing the review.

### 2. Success patterns
What the SME layer is winning on. These inform:

- Where to keep investment (adapter tuning, persona refinement).
- The "gold-standard" template answers for Phase 0 regression eval.

### 3. Failure patterns
What is failing. Severity score weights:

- Explicit thumbs-down rate (40%)
- Citation-verifier drops (30%)
- Honest-compact fallback rate (30%)

For each cluster, read the `Evidence` block. Ask: does this look like a
tuning problem (persona, adapter thresholds) or a design problem (grounding
semantics, intent router, SMEVerifier logic)? Open an issue tagged
`sme-failure-cluster:<cluster_id>` in either case.

### 4. Artifact utility
The four retrieval layers' retrieval rate and positive-outcome rate. A
"dead weight" flag means a layer is pulled often but correlates with bad
outcomes. Dead-weight layers are candidates for:

- Adapter threshold bumps (confidence floor, max-hops for KG edges).
- Layer gating change (skip this layer for these intents).
- Turning the layer off for a subscription via its feature flag.

### 5. Persona performance
Per-persona SME-score proxy vs the domain baseline. Regression-flagged
personas should be compared against the adapter YAML changes in the last
30 days. Rollback the adapter via the standard per-phase rollback path if
the regression holds.

### 6. Training candidates
Stabilized failure clusters with at least 2 months' presence, at least 20
total volume, and stabilization score at or above 0.55. **This is the bridge
to sub-project F.**

Decision framework per candidate:

- If the cluster reads as an **engineering** problem (wrong persona,
  missing intent handling, grounding too loose): fix in engineering first;
  sub-F stays closed.
- If the cluster persists across 2+ months after engineering fixes and
  severity + volume remain high: convene a sub-F kickoff decision. The
  candidate record is evidence only — a human owner decides.

## Rollback post-mortems

When a full SME rollback happens (spec Section 13.3), write the post-mortem
file `sme_rollback_YYYY-MM-DD.md` into this directory. The next monthly
pattern-mining run auto-discovers and links it under the "Rollback
post-mortems" section of the monthly report.

Minimum post-mortem contents:

1. Trigger condition (which Section 13.4 item fired).
2. Scope (which flag: per-subscription or `sme_redesign_enabled=false`).
3. What tuning was attempted before rollback.
4. What traces show (cite cluster_ids from the prior monthly report).
5. Next steps — keep engineering-first; only escalate to sub-F if traces
   say an engineering ceiling was hit.

## Memory rules applied

- **Engineering-first** — this loop produces evidence; it never triggers
  retraining.
- **Profile isolation** — clusters carry `subscription_ids`; cross-sub
  rollup only at the persona / artifact-utility level.
- **No customer data in analytics outputs** — the monthly Markdown contains
  only fingerprints and cluster-level aggregates; raw query text stays in
  the Blob trace store.
- **Traces in Azure Blob, not Mongo** — the loader reads JSONL blobs via
  `src/storage/azure_blob_client.py`.
- **Redis path** — the orchestrator uses `src.utils.redis_cache` via
  `src.utils.redis_startup`; there is no `src.utils.redis_client`.

## Operating notes

- If the timer fires while sub-F is already in flight for a given cluster,
  the monthly report re-lists the same candidate — that is intentional.
  Candidates are idempotent.
- If a month has zero traces (e.g. master flag off), the monthly file still
  renders with "no ... clusters this month" placeholders; that is a signal,
  not a failure.
- Re-run the month manually:

  ```bash
  python -m scripts.mine_sme_patterns \
      --window-start 2026-04-01 \
      --window-end 2026-04-30T23:59:59 \
      --out-dir analytics/
  ```

- Run with a rolling window instead of a calendar month:

  ```bash
  python -m scripts.mine_sme_patterns --window-days 30 --out-dir analytics/
  ```

- Threshold tuning for the training-trigger evaluator — pass
  `--min-months`, `--min-volume`, `--stabilization-threshold`, or
  `--total-months-window` to `scripts/evaluate_training_trigger.py` or edit
  the wrapper once a reviewer gains enough historical data to recalibrate.

## Disable the timer

```bash
sudo systemctl disable --now docwain-sme-pattern-mining.timer
```

The accrued analytics files remain in place; only future scheduled runs are
suspended.
