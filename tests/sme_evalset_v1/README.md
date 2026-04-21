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
