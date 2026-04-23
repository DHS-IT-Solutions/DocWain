# Extraction Accuracy Bench

This directory holds the version-controlled accuracy bench for DocWain's extraction pipeline.
See `docs/superpowers/specs/2026-04-23-extraction-accuracy-design.md` §8 for the scoring
definition and gate thresholds.

## Layout

````
tests/extraction_bench/
├── README.md                     # this file
├── __init__.py
├── scoring.py                    # coverage / fidelity / structure / hallucination metrics
├── runner.py                     # iterate bench, run adapters, score, emit report
├── fixtures/
│   └── generate_fixtures.py      # programmatic fixture generation
└── cases/
    └── <doc_id>/
        ├── source.<ext>          # the document under test
        ├── expected.json         # ground-truth canonical JSON
        └── notes.md              # any human context about the doc
````

## Running the bench

```bash
/home/ubuntu/PycharmProjects/DocWain/.venv/bin/python -m tests.extraction_bench.runner
```

Exits non-zero if any bench case fails its gate threshold.

## Adding a real-document case

1. Drop the source file under `cases/<doc_id>/source.<ext>`.
2. Hand-author `cases/<doc_id>/expected.json` following the canonical schema
   defined in `src/extraction/canonical_schema.py`.
3. Add `cases/<doc_id>/notes.md` with any operator context (what this doc tests,
   edge cases, known quirks).
4. Re-run the bench; the new case is picked up automatically.

## Gate thresholds

Per spec §8.4:
- Native path: coverage 100%, fidelity ≥ 0.98, structure 100%, hallucination 0%
- Vision path (Plan 2): coverage ≥ 0.95, fidelity ≥ 0.92, structure ≥ 0.95, hallucination < 0.01
- Handwriting: coverage ≥ 0.90, fidelity ≥ 0.85
