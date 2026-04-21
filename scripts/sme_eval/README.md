# DocWain SME Evaluation Tooling

Measurement harness for sub-project A (Profile-SME reasoning layer).

## Modules
- `schema.py` — Pydantic models (EvalQuery, EvalResult, MetricResult, BaselineSnapshot)
- `query_runner.py` — HTTP client against DocWain /api/ask
- `result_store.py` — Append-only JSONL store for results
- `metrics/` — One metric per file (RAGAS wrapper + 6 new reasoning metrics)
- `aggregate.py` — p50/p95/p99 per intent
- `human_rating.py` — CSV export/import for expert rating
- `run_baseline.py` — CLI orchestrator

## Running locally
See `tests/sme_evalset_v1/README.md` for the operator runbook.

## Adding a new metric
1. Create `scripts/sme_eval/metrics/<name>.py` subclassing `Metric` from `_base.py`.
2. Create `tests/scripts/sme_eval/metrics/test_<name>.py` with at least 3 test cases.
3. Add to `DEFAULT_METRICS` in `run_baseline.py`.
4. Document the metric's value range, pass threshold, and interpretation in the spec.

## Memory rules applied
- No customer data — eval queries are synthetic, see `tests/sme_evalset_v1/README.md`.
- Measurement tool only — no production code changes under `src/` touched by Phase 0.
- The per-request httpx timeout is a per-operation safety limit, NOT a response-latency cutoff (consistent with "No Timeouts; Use Efficiency" rule).
