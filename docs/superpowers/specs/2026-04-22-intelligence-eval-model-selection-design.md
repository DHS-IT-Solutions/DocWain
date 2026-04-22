# Intelligence Eval + Unified Model Selection — Design Spec

**Date:** 2026-04-22
**Owner:** Muthu
**Status:** Approved in brainstorming; pending spec review before implementation plan
**Prerequisite:** `docs/superpowers/specs/2026-04-21-intelligence-rag-redesign-design.md` Batch 0 **must be merged to `main`** before Phase 1 of this workstream starts. Otherwise the baseline measures a still-broken retrieval layer.
**Related memory:** `feedback_measure_before_change`, `feedback_engineering_first_model_last`, `feedback_unified_model`, `feedback_no_timeouts`, `feedback_no_customer_data_training`, `feedback_accuracy_over_latency` (via `user_muthu`), `feedback_intelligence_rag_zero_error`

## 1. Context

The intelligence layer has been producing responses that are factually wrong / generic / lose persona. Batch 0 of the RAG re-integration workstream fixes the retrieval layer (five read-side qdrant filter mismatches around `source_name`, `page`, etc.) — it is code-reviewed, gate-green, and ready to merge.

Separately, the user has asked for:

1. Iterative analysis to improve the intelligence of responses.
2. A determination of which DocWain model on disk (out of V2, V5, their variants, and a base-Qwen3 control) has the highest intelligence + accuracy + speed.
3. A swap so the winning model becomes the unified serving target.
4. Intensive live tests that validate response accuracy, intelligence, and insight quality from actual documents.

This is a **new workstream** that runs after Batch 0 merges.

## 2. Goals

1. Establish a reproducible intelligence/accuracy/latency scorecard harness.
2. Iterate on engineering-layer variables (prompts, retrieval, context assembly, generation params, verification thresholds) on the current V2 model to find a better **prompt stack P\***.
3. Rank five model contenders under P\*; identify the winner by combined score with a hard latency gate.
4. If the winner materially beats V2, swap the unified serving model to the winner with a live canary and auto-rollback safety rail.
5. Leave a repeatable evaluation pipeline behind so future model or prompt changes can be gated the same way.

## 3. Non-goals

- Model retraining, LoRA tuning, or any change to weights. Engineering-first; weights are selected from what already exists on disk.
- Changes to the document-processing write path (extraction, embedding, qdrant indexing, mongo status transitions).
- Adding new metrics beyond the five already listed for the intelligence rubric.
- Replacing `scripts/overnight_qa/` with a greenfield harness.
- Waiting for Batch 1's SME phase-0 eval harness (tracked separately). This workstream reuses existing tooling and doesn't couple to SME re-integration.
- Cloud-model use for the evaluation or swap (user explicitly excluded Ollama Cloud 397b from the workstream).

## 4. Phase Plan & Gates

Five phases, each with a hard gate before proceeding.

| Phase | Duration | What happens | Gate to proceed |
|---|---|---|---|
| **1** Baseline | ~2h | Augment `scripts/overnight_qa/` + `scripts/evaluate_docwain.py`; run full sweep on V2-current; run fast smoke on Procurement profile. | Baseline snapshot committed to `eval_results/intel-baseline-V2.jsonl`. Fast-smoke reproducible within tolerance on rerun. |
| **2** Engineering iteration | ~4-6h over 5-8 cycles | Loop on V2: propose tweak → user approve → fast smoke → keep-or-revert. Stop at 8 cycles or 2 consecutive no-improvements (plateau). | Frozen **P\*** prompt stack + per-iteration delta docs. Fast-smoke score monotone-improving (or flat at top) vs baseline. |
| **3** Model comparison | ~8h (overnight OK) | Full sweep on 5 contenders under frozen P\*. Rank by combined score among latency-gate passers. | Ranked scorecard committed to `eval_results/phase3-ranking.md`. Winner has ≥ 0.05 margin over V2, or user explicitly accepts smaller margin. |
| **4** Winner handling | ~30 min | If winner = V2: commit engineering improvements, done. If winner ≠ V2 (and margin met): prepare swap bundle (symlink target, restart cmd, rollback cmd, watch plan). | If swap needed: bundle reviewed, user says "go to Phase 5". If no swap: close-out report written. |
| **5** Live canary (only if swap) | 1h + watch | Agent executes swap with user go/no-go before each live-affecting step. 10-query canary + 15-min log watch. Auto-rollback on defined triggers. | 10-query canary PASS on new model + 15 min log-watch error rate ≤ pre-swap baseline × 2. |

## 5. Harness Architecture

### 5.1 Full-sweep harness — augment `scripts/overnight_qa/`

**Current behaviour**: ingests 15 synthetic docs × 6 categories (invoices, contracts, purchase_orders, resumes, finance_statements, expense_reports) into brand-new profiles; polls pipeline until each doc is embedded; runs 20 queries per profile; writes raw result JSON.

**Augmentations** (all additive):
- Deterministic run ID: `scripts/overnight_qa/results/<run_id>/…` where `run_id = "{date}-{model}-{promptstack_sha}"`. Enables diff between runs.
- Idempotency key on ingestion: skip re-ingesting profiles/docs if a matching run fingerprint exists; only re-query.
- New module `scripts/intel_eval/score_run.py` — reads a run's raw results and emits a structured scorecard (JSON + markdown).
- Model target: per §5.4 below, harness runs against a specific vLLM port (primary 8100 = current prod; secondary 8101 = contender under test).

### 5.2 Fast-smoke harness — new `scripts/intel_eval/fast_smoke.py`

- 15 queries × 3 pre-seeded profiles (invoices, contracts, resumes) = ~45 queries total. Uses the same substrate overnight_qa creates, so smoke and full sweep share profiles and docs.
- Runtime target: ≤ 5 minutes.
- Output: pass/fail to stdout + one JSON line appended to `eval_results/phase2-smoke-log.ndjson` for trend tracking.

### 5.3 LLM-judge — reuse `scripts/evaluate_docwain.py`

- `scripts/intel_eval/score_run.py` imports `evaluate_docwain.py`'s scorer as a library.
- Judge model **frozen for the entire workstream** (judge model + temperature 0 + fixed seed) so scores stay comparable across runs. Judge model choice is configurable but committed to `scripts/intel_eval/judge_config.yaml` on Phase 1 entry and not changed thereafter.
- Rubric committed verbatim to `scripts/intel_eval/judge_rubric.md`.

### 5.4 Multi-vLLM-instance strategy (Option X)

- Primary instance: port **8100**, served-model-name `docwain`, serves `models/docwain-v2-active` (current prod). Untouched during Phases 1 and 2.
- Secondary instance: port **8101**, launched per contender during Phase 3 only. Systemd unit `docwain-vllm-eval@.service` (templated, takes model path as instance parameter), or ad-hoc `systemd-run --unit` invocation.
- Phase 3 loop: for each contender, spin up secondary on 8101, run harness against 8101, tear down secondary, next contender.
- Memory pressure: 14B FP8 weights ~16GB; running two 14B instances concurrently = ~32GB. If GPU memory insufficient, the Phase 3 loop alternates: primary stays up on 8100, secondary cycles through contenders. Live service on port 8000 continues to hit 8100 throughout.

### 5.5 Live-canary harness — reuse `scripts/batch0/canary_smoke.py`

Existing script from Batch 0 Task 13. Used at:
- End of Phase 1 — document post-Batch-0 live behavior as the "after Batch 0, before intel work" baseline.
- Phase 5 post-swap — validate the new model doesn't regress live queries.

## 6. Scoring Rubric

### 6.1 Gate 1 — Sanity (binary, query-level)

A query row is marked `invalid` and excluded from scoring if any:
- HTTP status ≠ 200.
- Response body missing `response`, `sources`, `grounded`, or `context_found` keys.
- Response text empty.
- Static fallback `"I'm having trouble"` surfaced.
- On `greeting` / `identity` intents: "DocWain" substring absent.

Model disqualified from ranking if >10% of full-sweep rows are `invalid`.

### 6.2 Gate 2 — Latency (model-level)

p95 latency per intent category must be under:
- Simple (greeting, identity, lookup, list, count): **10 s**
- Moderate (extract, summarize, timeline): **20 s**
- Complex (compare, analyze, investigate): **30 s**

Failure in any band → model disqualified from ranking regardless of other scores.

### 6.3 Scored dimension 1 — Accuracy (0-1)

- **Synthetic docs**: exact-match / field-match on deterministic ground-truth (invoice_number, total, vendor, date, line-item counts). Per-query accuracy = fraction of asked fields present and correct.
- **Real docs (canary only)**: RAGAS-style faithfulness — LLM-judge asks whether every factual claim in the answer is supported by the cited sources, 0-1 at 0.1 granularity.
- Model accuracy = micro-average across queries.

### 6.4 Scored dimension 2 — Intelligence (0-1)

LLM-judge rates each answer on the fixed rubric:

```
Rate this DocWain response on a 0-1 scale (0.1 granularity) for each:
1. Groundedness:   Every claim traceable to a cited source?
2. Relevance:      Directly answers the user's question?
3. Insight:        Synthesises or connects information beyond literal retrieval?
4. Structure:      Format matches the intent (table/list/prose) and is clean?
5. Persona:        Consistent DocWain voice; professional, concise, useful?

Intelligence = mean of the 5 sub-scores for this response.
```

Committed verbatim to `scripts/intel_eval/judge_rubric.md`. Passed as system prompt for every scoring call.

### 6.5 Combined score

```
combined = 0.5 · accuracy + 0.5 · intelligence
```

Reported per-model. Tie-break (within 0.01): prefer lower median latency.

### 6.6 Explicitly not scored

- ROUGE/BLEU/embedding-similarity (too soft).
- Cost ($, GPU-hours) (pool sizes comparable).
- Cross-run consistency (tracked as a separate reproducibility sanity check, not in ranking).

## 7. Model Contender Pool (Phase 3)

Five contenders enter Phase 3, all served through vLLM on port 8101. Cloud models (Ollama Cloud 397b) excluded per user direction.

| # | Model | Path | Why |
|---|---|---|---|
| 1 | DocWain-14B-v2 (full) | `models/DocWain-14B-v2/` | Current prod active — baseline to beat. |
| 2 | DocWain-14B-v2-AWQ | `models/DocWain-14B-v2-AWQ/` | Quantization keeps quality while saving latency? |
| 3 | DocWain-14B-v5 (full) | `models/DocWain-14B-v5/` | Was V5 revert correct? If it wins here, argues for V5.1. |
| 4 | DocWain-8B-v5 (full) | `models/DocWain-8B-v5/` | Smaller-faster tradeoff point. |
| 5 | Qwen3-14B (base) | via vLLM with `--model Qwen/Qwen3-14B` | Control — did DocWain training add net value over base? |

Excluded: `DocWain-14B-v5-sft`, `DocWain-7B-v5`, `DocWain-14B-v5-q5km.gguf`, `DocWain-8B-v5-q5km.gguf` (intermediate or quantization-of-variant; revisit only if their parent wins).

## 8. Engineering Iteration Protocol (Phase 2)

### 8.1 Branch and git discipline

- Branch: `intel-eval-phase2` off `main` (post-Batch-0).
- One iteration = one commit. Commit subject: `intel-phase2: iter-N <variable> +ΔX.XX`.
- Reverts: `git reset --hard HEAD~1`. No half-commits.
- Plateau: branch tagged `intel-phase2-P-star`. Diff vs main is the **P\*** artifact.

### 8.2 Allowed tweak variables

1. Retrieval knobs (`src/retrieval/`, `src/query/`): top-k, reranker weights, hybrid α, filter inclusivity, diversity-vs-score priority.
2. Generation prompts (`src/generation/prompts.py`): system prompt, format instructions, reasoning instructions.
3. Context assembly: source ordering/labeling, chunk-window size.
4. Generation params: temperature, thinking-mode toggle.
5. Verification thresholds: `_MIN_CONFIDENCE`, re-retrieval loop caps.

Disallowed: retraining, write-path edits, schema changes, new dependencies, multi-variable single-iteration tweaks.

### 8.3 One iteration's lifecycle

```
1. Agent writes proposal with:
   - Hypothesis (what and why)
   - Variable (exact file:line)
   - Expected Δ on fast-smoke combined score
   - Rollback command
2. User: APPROVE / DECLINE / AMEND.
3. If APPROVED:
   a. Subagent applies edit in one commit.
   b. Runs fast-smoke (~5 min).
   c. Computes Δ vs last accepted state.
4. Automatic decision:
   - Δ ≥ +0.02  → KEEP (log to eval_deltas/phase2-iter-N.md)
   - otherwise   → REVERT (`git reset --hard HEAD~1`)
```

### 8.4 Stop conditions

- 2 consecutive REVERTs → plateau → freeze P\*.
- 8 total iterations (keep + revert) → stop regardless.
- Combined fast-smoke score ≥ 0.90 → ceiling reached; stop.

## 9. Model-Swap Mechanics (Phase 5)

### 9.1 Preconditions

- Phase 3 scorecard committed; winner identified in writing.
- Winner combined score ≥ V2 combined score + **0.05** (material margin).
- Winner's full-sweep sanity-gate invalid rate <5%.
- User explicit "go to Phase 5".

### 9.2 Pre-swap bundle (`deploy/phase5-swap/`)

- `winner.json` — `{model_path, served_name, source_phase3_run_id, combined_score, v2_baseline_score, margin}`.
- `swap.sh` — ln -sfn + systemctl restart + health wait.
- `rollback.sh` — reverse: ln -sfn back to DocWain-14B-v2 + restart + health wait.
- `canary.txt` — canary_smoke.py invocation with env vars filled.
- `watch.md` — 15-min log-watch instructions.

### 9.3 Execution (agent-executed with user go/no-go)

```
Step 1  Show bundle.                                         GATE: go/no-go.
Step 2  ln -sfn <winner> docwain-v2-active                   (no live effect).
                                                             GATE: restart/abort.
Step 3  sudo systemctl restart docwain-vllm-fast.
        Wait /health (60s max).
        /health not green → auto-rollback, STOP.
Step 4  Run canary_smoke.py.
        All 10 PASS + grounded flags correct → proceed.
        1-2 fail → show delta vs pre-swap, user go/no-go.
        3+ fail → auto-rollback, STOP.
Step 5  15-min log watch via Monitor:
          sudo journalctl -fu docwain-app.service \
            | grep --line-buffered -E "ERROR|Traceback|grounded=False|empty"
        ERROR rate > 2× pre-swap baseline in any 1-min window → auto-rollback.
Step 6  On clean window: write eval_results/phase5-swap-outcome.md, commit.
```

### 9.4 Auto-rollback triggers (no user confirmation)

- `/health` not green 60s post-restart.
- Canary: 3+ of 10 queries fail the sanity gate.
- Log watch: ERROR rate > 2× pre-swap baseline in any 1-min window.

Auto-rollback: subagent runs `rollback.sh`, confirms `/health`, re-runs canary, confirms PASS, writes `phase5-auto-rollback.md` post-mortem, stops. No second attempt.

### 9.5 No-swap path

If V2 retains the crown or no contender clears the +0.05 margin:

- Phase 2's engineering improvements merged to main as a separate small PR (flag-free, `intel-phase2-P-star` branch).
- Phase 3 scorecard committed as `eval_results/phase3-ranking.md`.
- Short `eval_results/phase5-no-swap.md` notes "V2 retained; margin to nearest contender was ΔX.XX".

## 10. Artifacts and Reproducibility

All artifacts committed to the repo under predictable paths:

- `scripts/intel_eval/` — new subdirectory: `fast_smoke.py`, `score_run.py`, `judge_rubric.md`, `judge_config.yaml`.
- `scripts/overnight_qa/` — augmented with run-ID + idempotency; existing structure preserved.
- `scripts/batch0/canary_smoke.py` — reused as-is.
- `eval_results/intel-baseline-V2.jsonl` — Phase 1 baseline.
- `eval_results/phase2-smoke-log.ndjson` — per-iteration fast-smoke history.
- `eval_deltas/phase2-iter-<N>.md` — per-iteration before/after doc.
- `eval_results/phase3-ranking.md` — Phase 3 final ranking.
- `deploy/phase5-swap/` — swap bundle (only if Phase 5 runs).
- `eval_results/phase5-swap-outcome.md` OR `eval_results/phase5-no-swap.md` — close-out.

Reproducibility: every committed scorecard includes the judge model name, judge seed, dataset run-ID, promptstack SHA, and git commit SHA so any run can be re-executed identically.

## 11. Testing Strategy

### 11.1 Harness tests

- `tests/intel_eval/test_score_run.py` — given a fixture results dir, `score_run.py` produces expected scorecard.
- `tests/intel_eval/test_fast_smoke.py` — mocks the live API, asserts the smoke runner computes combined score correctly.
- `tests/intel_eval/test_latency_gate.py` — hand-crafted latency fixtures trigger the expected gate decisions.

### 11.2 Harness reproducibility

At the end of Phase 1, rerun the fast smoke twice and assert combined score matches within the §6 tolerance (deterministic metrics byte-identical; LLM-judge within 0.02). Failure to reproduce blocks Phase 2 entry.

### 11.3 Rollback drill (before Phase 5)

Before the first real swap, subagent runs `swap.sh` + `rollback.sh` in a dry-run loop (on a staging symlink that does not affect live service) to validate the commands execute cleanly end-to-end.

### 11.4 What's out of scope

- Full pytest on `tests/` at each phase gate (pre-existing failures are noisy; pipeline-isolation was covered in Batch 0).
- Cross-profile consistency tests beyond the sanity gate.

## 12. Open Questions / Risks

- **Judge model choice.** The judge must be stable and neutral. Candidates: current prod DocWain-V2 via vLLM on 8100, or local Ollama `qwen3:14b`. Using DocWain to judge DocWain has a bias risk; using a generic Qwen3 is cleaner. Decision deferred to Phase 1 entry, committed to `judge_config.yaml`.
- **GPU memory for Phase 3.** Normal mode: live V2 stays on port 8100, each contender cycles through port 8101 one at a time (concurrent ~32 GB of weights). If even that exceeds GPU capacity, OOM fallback is **live-service swap per contender**: for each contender's run, briefly flip primary 8100 away from V2 to the contender (systemctl restart), run the eval, flip back. This introduces deliberate live-traffic disruption during Phase 3 windows — acceptable because it's bounded, announced, and rolled back immediately after each contender. If the user wants zero live disruption in Phase 3, the OOM fallback is skipped and contenders that don't fit alongside V2 are simply excluded from the ranking with a note in `phase3-ranking.md`.
- **Synthetic vs real doc distribution.** The overnight_qa generator produces clean synthetic docs. Real user docs have OCR noise, layout quirks, multi-page tables. Canary catches this, but Phase 3 rankings are based primarily on synthetic. Mitigation: canary scorecard is included alongside the phase-3 ranking table.
- **Live service disruption.** Option X (secondary instance on 8101) means live on 8000 is untouched in Phases 1–3. Only Phase 5's `systemctl restart docwain-vllm-fast` interrupts live traffic (~15-30s). Canary + rollback mitigate this.
- **"list" template-render bug** noted in Batch 0 canary (literal `"**N items found:**"`) — unrelated to this workstream; tracked as a follow-up issue, does not block.

## 13. Out-of-scope follow-ups

- Batch 1 (SME phase-0 eval harness cherry-pick) — tracked separately; when it lands, its 10 metrics can be layered into `score_run.py` as additional scored dimensions.
- V5.1 training if V5 wins Phase 3 — separate training spec.
- Post-swap systemd unit-file renames — ops PR.
- Cost/GPU-hour tracking — follow-up once rankings stabilize.
