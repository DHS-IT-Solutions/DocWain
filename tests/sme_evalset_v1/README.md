# DocWain SME Evaluation Set v1

Versioned eval set for DocWain's Profile-SME reasoning sub-project (A).
100 queries per major domain × 6 domains = 600 queries total.

## Domains
- finance — financial SME queries (cost, revenue, trends, recommendations)
- legal — contract/obligation/party queries
- hr — employee/policy/benefits queries
- medical — diagnosis/treatment/record queries
- it_support — symptom/troubleshooting/fix queries
- generic — domain-agnostic document queries

## Query file schema
See `scripts/sme_eval/schema.py` for the authoritative schema.

## Fixture profiles
See `fixtures/test_profiles.yaml` for test subscription/profile IDs.
All test profiles MUST contain synthetic data only — no customer documents.

## Running the baseline
See `scripts/sme_eval/run_baseline.py --help`.

## When to regenerate
The eval set is frozen at v1 for Phase 0 baseline. Subsequent phases may
add queries as `tests/sme_evalset_v2/` — never modify v1 in place.

## Running the baseline — operator runbook

### Prerequisites
1. DocWain API is running (defaults to `http://localhost:8000`; override via `DOCWAIN_API_URL` env).
2. DocWain LLM gateway is reachable (defaults `http://localhost:8100/v1/chat/completions`; override via `DOCWAIN_LLM_URL`).
3. Test subscription + 6 test profiles exist in production DocWain with synthetic documents ingested; their IDs are filled into `tests/sme_evalset_v1/fixtures/test_profiles.yaml`.
4. All test-profile documents have `PIPELINE_TRAINING_COMPLETED`.
5. Seed query YAMLs exist under `tests/sme_evalset_v1/queries/` (Task 16 — content curation with domain experts).

### Run the baseline
```bash
python -m scripts.sme_eval.run_baseline \
    --eval-dir tests/sme_evalset_v1/queries \
    --fixtures tests/sme_evalset_v1/fixtures/test_profiles.yaml \
    --out tests/sme_metrics_baseline_$(date +%Y-%m-%d).json \
    --results-jsonl tests/sme_results_$(date +%Y-%m-%d).jsonl
```

- Runtime: ~10-40 minutes depending on DocWain latency. Sequential by design.
- Output: `tests/sme_metrics_baseline_YYYY-MM-DD.json` (committed, frozen)
- Raw results: `tests/sme_results_YYYY-MM-DD.jsonl` (committed; optional — can be gitignored if size grows)

### Human rating pass (after automated run completes)
```bash
python -c "
from scripts.sme_eval.result_store import JsonlResultStore
from scripts.sme_eval.human_rating import export_for_rating
results = list(JsonlResultStore('tests/sme_results_YYYY-MM-DD.jsonl').iter_all())
export_for_rating(results, 'tests/sme_human_rating_YYYY-MM-DD.csv')
"
# Distribute the CSV to domain experts. Each rates sme_score_1_to_5.
# Collect their rated CSVs, merge, import:
python -c "
from scripts.sme_eval.human_rating import import_ratings
ratings = import_ratings('tests/sme_human_rating_YYYY-MM-DD.csv')
import json
snap = json.load(open('tests/sme_metrics_baseline_YYYY-MM-DD.json'))
vals = list(ratings.values())
snap['human_rated_sme_score_avg'] = sum(vals) / len(vals) if vals else None
snap['human_rated_count'] = len(vals)
json.dump(snap, open('tests/sme_metrics_baseline_YYYY-MM-DD.json', 'w'), indent=2)
"
```

### Interpreting the baseline
The baseline snapshot's `ragas` block should roughly match the pre-existing
`tests/ragas_metrics.json` within noise. If it diverges materially, that's
a signal that either the eval set has drifted or the RAGAS wrapper's heuristics
aren't aligned with `scripts/ragas_evaluator.py`. Investigate before trusting
any later phase's gate.

### Subsequent-phase regression run
Phase 2+ re-runs this baseline against the phase's build and compares to
`tests/sme_metrics_baseline_YYYY-MM-DD.json`. Launch-gate conditions (Section 10
of the design spec) must hold.
