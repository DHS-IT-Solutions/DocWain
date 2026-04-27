---
name: Measure Before You Change
description: Do not propose or ship any accuracy/quality improvement without a measurement harness in place first. Baseline before change, diff after.
type: feedback
originSessionId: 93168fda-607e-4c51-b06c-5b5e0f18a6b1
---
Any work that claims to improve accuracy, quality, latency, or recall MUST start by establishing the measurement. This is non-negotiable.

The rule:
1. **First**, build or identify the eval that will score the change (what docs, what metric, what threshold).
2. **Then**, capture a baseline with current code.
3. **Then**, make the change in small reviewable steps.
4. **Then**, re-run the eval and diff against the baseline.
5. **Only then** decide if the change is worth keeping.

This applies to: extraction accuracy, retrieval accuracy, response faithfulness, grounding correctness, latency budgets, or any other "we made it better" claim.

**Why:** On 2026-04-17 the RAG accuracy workstream was scoped against an 11-day-old RAGAS metric (0.561 recall) that had already been superseded by a model repoint (fresh recall was 0.840). An entire plan was written against a problem that no longer existed. When the fresh baseline came in mid-execution, the plan continued by inertia rather than re-scoping. Result: infrastructure landed, recall REGRESSED (-0.033), faithfulness didn't move, spec gate failed. User rightly frustrated.

**How to apply:**
- When asked to improve anything, the first concrete deliverable is the eval harness and the fresh baseline. If that doesn't exist, build it before writing any spec or plan.
- Never scope improvements against memories more than a day or two old — metrics drift as models, data, and code change.
- If the baseline comes in differently than expected, STOP. Re-scope. Do not push through with the original plan.
- Specs should quote the fresh baseline number, never a historical one from memory.
- For extraction work specifically: a diverse test bench of ~10-15 docs (scanned, native PDF, forms, tables, Excel, Word, multi-column) with ground-truth expected entities/fields is the minimum measurement.
- For retrieval work: a query bank with expected chunk IDs or expected answer fragments per query.
- For generation work: an LLM-judge or rubric-based eval on a held-out set of queries.
