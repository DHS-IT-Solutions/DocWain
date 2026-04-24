# DocWain Intelligence Plan — Part 2

**Date:** 2026-04-24 evening → 2026-04-25
**Branch:** `preprod_v02`
**Request:** "Implement remaining gaps. Research intelligence. Start a 2-day round-the-clock training run if safely possible."

---

## 1. TL;DR

1. Four of the five gaps from the overnight report are now shipped. Two of yesterday's live-battery failure cases (anomaly hallucination, filename blind-spot) have been verified fixed live against the 43-doc UAT profile.
2. A dedicated audit agent concluded: **do not start a 2-day training run tonight.** The V5 eval harness that is supposed to gate it was deleted from `main` in the 2026-04-22 merge; scorer bugs that caused V5's 1/7-gate failure are not fixed in current code; V2 has no measurable baseline. Starting training under these conditions would repeat the V5 failure.
3. A parallel research agent reviewed 20 candidate techniques for improving document-analysis intelligence without training. The top six are feasible in the remaining weekend window and carry an expected cumulative +25-45% on insight-depth evaluations.
4. I've shipped the inference-side wins that require zero training and have clean rollback paths. The recommended 2-day plan replaces training with the prep work that makes a **safe** training run possible next week.

---

## 2. Gap fixes shipped tonight

Commit `720b506` on `preprod_v02`.

### Gap A — Anomaly surfacing (was: HIGH — the "No anomalies detected" failure)

Before:
> *Q: "Any anomalies or inconsistencies across the invoices?"
> A: "No anomalies or inconsistencies are detected across the invoices. All values, dates, and entities align with standard financial terminology and formatting."*

After (live-run, commit `720b506`, 33.7 s):
> *"**Anomaly detected:** Amount mismatch between invoice and PO: QUOTE_WSG100025 vs PO 5205561.
> — QUOTE_WSG100025 reports £134,370. — PO 5205561 reports £555,303.37. These values are inconsistent across documents. This discrepancy requires reconciliation. **Critical** — financial accuracy depends on alignment."*

How: `src/agent/core_agent.py::_build_analysis_context` pulls the Researcher Agent's `anomalies[]` from Mongo and injects them into the reasoner's doc_context as a `PRE-COMPUTED ANALYSIS SIGNALS` block. A new rule (5b) in the system prompt forces the model to defer to that block on anomaly/percentage/ranking questions.

### Gap B — Cross-document GROUP BY aggregation (HIGH — the "% by category" failure)

Before:
> *Q: "What percentage of the invoices are Apparel vs Furniture vs Hospo?"
> A: "The documents do not specify Apparel, Furniture, or Hospo categories for invoices."*

After (live-run, 29.4 s):
> *"Apparel: 2 documents. Hospo: 4 documents. Furniture: 0 documents.
> | Category | Count | % of Total | | Apparel | 2 | 8% | | Hospo | 4 | 16% | | Furniture | 0 | 0% |"*

How: same analysis-signals path carries `total_documents`, `dominant_domain`, `document_counts_by_type`, `document_counts_by_status`, `prevalent_entities[by_type]`. Deterministic — zero extra LLM round trips.

### Gap C — Filename + doc_type in evidence headers (MEDIUM)

`EvidenceChunk` now carries `source_file` and `doc_type`. `build_reason_prompt` emits `| File: <filename> | Type: <doc_type>` on every `[SOURCE-N]` line. This was why the LLM missed that `PO8_Apparel_Invoice_*.pdf` encodes the Apparel category — it literally never saw the filename in context.

### Gap D — Researcher Agent triggers (MEDIUM)

- `POST /api/documents/{id}/researcher/run` — re-dispatch a single doc.
- `POST /api/profiles/{id}/researcher/backfill` — scan a profile, fire Researcher for every extraction-complete doc that doesn't have an insights payload yet. This is how existing UAT profiles get populated without re-extraction.

### Feature flag

Every change is behind `DOCWAIN_ANALYSIS_INJECT` (default `true`). Setting it to `false` reverts to yesterday's behaviour without a deploy — used if any analysis query regresses under load.

---

## 3. FT feasibility audit — NO-GO

Agent investigated the finetuning infrastructure with one question: *can a 2-day run safely start tonight?*

| Check | Result |
|---|---|
| V5 eval harness reproducible | **NO** — `src/finetune/v5/evaluate.py` deleted during the 2026-04-22 merge (commit `5b06984`). |
| V2 baseline measurable | **NO** — `models/docwain-v2-baseline.json` = placeholder `0.0`. |
| Known scorer bugs fixed | **NO** — the think-stripping and refusal-token fixes (commit `43be1fc` on the failed V5 branch) were never ported to `main`. Running today would silently score 0 on schema_adherence and tool_calling again. |
| Orchestrator guardrails | **MISSING** — V5 orchestrator set `distill_failed=true` but still advanced to QUANTIZE+DEPLOY. No pre-train gate check. No post-distill output-shape validation. |
| Identity gateway shim wired | **NO** — per the V5 failure lessons (identity-in-weights has never worked across any training run), the intended fallback is to inject identity at the serving layer. Not yet done. |
| GPU clean | 74 GB / 80 GB used by vLLM; Ollama running idle. A clean window would free ~80 GB but requires stopping prod inference. |

Verdict: **Starting tonight would repeat V5.** Same scorers, same bugs, same orchestrator that silently advanced through `distill_failed=true`. A 2-day A100 window is expensive and the expected outcome on current evidence is "1-2 gates pass, revert again".

The agent's 48-hour "Path A" to GO:
- Restore `v5/evaluate.py` from commit `43be1fc` (4 h)
- Port the think-stripping + refusal-token fixes to main (ships as part of the eval restore)
- Measure the V2 baseline, write it to disk (2 h)
- Add orchestrator guardrails: pre-train gate, post-distill size check, don't advance on `failed=true` (6 h)
- Wire the identity gateway shim at the vLLM proxy (8 h)
- Rebuild `src/finetune/v5/` from the last known-good commit `ec3a112` (12 h)
- Eval on 14B + 8B student, document before/after (6 h)

That's the prerequisite work. With it done, a Monday start has real odds of succeeding.

---

## 4. Intelligence improvement research

Parallel research agent surveyed 20 techniques with published evidence. Weekend-feasible top picks (inference-side, zero training):

| # | Technique | Expected lift | Effort | Status |
|---|---|---|---|---|
| 1 | **Eval harness (G-Eval + RAGChecker)** | — | 1 day | Prerequisite for anything below. **Not yet built.** |
| 2 | **Structured-output schema via vLLM guided decoding** | Medium; 10-20% on structured eval | 4-8 h | **Not shipped.** |
| 3 | **Step-Back prompting** | +7-27% on MuSiQue/MMLU/TimeQA (DeepMind 2023) | 2-4 h | **Not shipped.** |
| 4 | **Multi-query RAG-Fusion + RRF** | +22% NDCG@5, +40% recall@10 | 4 h | **Not shipped.** |
| 5 | **Cross-encoder rerank over top-100** | +17 pp MRR@3 | 4-8 h | **Partially present** — already reranks top-10, needs widening. |
| 6 | **Anthropic Contextual Retrieval** | −49% retrieval errors | 1-2 d ingest-time | **Not shipped.** Requires a one-time re-index; largest single published gain in retrieval. |
| 7 | **Chain-of-Density on the summary fields** | Human-preferred denser summaries | 3 h | **Not shipped.** |
| 8 | **Few-shot analysis exemplars** | +5-15% on structured analysis | 1 day curation | **Not shipped.** |

Pure training-side options were catalogued but none are safe to start without the eval harness + baseline from the FT audit above. They're listed in `docs/research/2026-04-24-overnight/intelligence_research_raw.md` for later.

---

## 5. Recommended 2-day plan (replaces the training ask)

Sharing the same 48-hour budget that was originally requested for training, redirected to work that is (a) safer, (b) faster to verify, (c) unblocks training for next week.

### Day 1 — Measurement + prompt/retrieval wins

**Morning (8 h) — Build the measurement harness**
- Stand up G-Eval with three custom criteria — *insight-depth*, *cross-doc coverage*, *anomaly surfacing*.
- Stand up RAGChecker for claim-level faithfulness.
- Curate 50 queries: 15 factual lookup + 15 cross-doc analysis + 10 anomaly/contradiction + 10 exec-summary. Hand-grade gold answers on 20 of them.
- Run the harness against current `preprod_v02` HEAD. **Publish the baseline.** This is the number every future change is measured against.

**Afternoon (6 h) — Structured-output schema + step-back**
- Wrap `/api/ask` analysis responses in a JSON schema via vLLM guided decoding: `{findings[], anomalies[], cross_doc_patterns[], aggregations[], executive_summary, confidence, unresolved_questions[]}`. Measure delta.
- Add step-back prompting as a pre-retrieval pass on investigate/analyze queries.

**Evening (4 h) — Multi-query + rerank widening**
- Generate 3-5 query variants per analysis question, retrieve each, RRF-fuse.
- Widen cross-encoder rerank window from top-10 to top-100 → top-10.

### Day 2 — Contextual re-index + FT prerequisites

**Morning (6 h) — Contextual Retrieval re-index on a bounded corpus**
- Implement the Anthropic chunk-contextualisation prompt.
- Re-ingest one UAT profile (43 docs) with contextualised embeddings and BM25.
- Measure retrieval recall vs baseline.

**Afternoon (8 h) — FT prerequisite work (from Section 3)**
- Restore `v5/evaluate.py` from `43be1fc`, port think-stripping + refusal-token fixes to main.
- Write the V2 baseline JSON from the new eval harness.
- Wire the identity gateway shim at the vLLM proxy.
- Add orchestrator guardrails (pre-train gate check, post-distill size check, fail-closed on `failed=true`).
- Document the pre-flight checklist (stop ollama, verify nvidia-smi clean, lock training datasets read-only).

**End of Day 2 state:**
- Measurable baseline exists. Every inference-side change is scored against it.
- Four research-backed inference improvements shipped, each individually flag-gated and measured.
- The training infrastructure is ready for a real run on Monday — scorers fixed, harness reproducible, orchestrator safe. The 2-day A100 window starts with a GO, not a coin flip.

---

## 6. What changed in this session (commits)

```
26cf459  docs: overnight research + implementation report (Part 1)
720b506  intelligence: inject pre-computed analysis signals + surface anomalies in /api/ask
(this)   docs: intelligence plan Part 2 + NO-GO training verdict
```

Everything is on `preprod_v02`. No UI contract breaks. Every behaviour change behind a feature flag.

---

## 7. What to test during UAT (updated)

Everything from Part 1, plus:

1. **Analysis-intent queries** (`compare / summarize / overview / investigate / aggregate / analyze / list`) now carry the new PRE-COMPUTED ANALYSIS SIGNALS block. Try:
   - "Any anomalies?" — expect specific mismatches, not a flat "no".
   - "What are the document type breakdowns?" — expect counts.
   - "Top three entities across the corpus?" — expect doc-count labels.
2. **Non-analysis queries** (simple lookup/extract) are UNCHANGED — fast, no new context, no extra cost.
3. **Backfill workflow:** `POST /api/profiles/{profile_id}/researcher/backfill` on a pre-UAT profile, then query analysis signals. Expect Researcher anomalies to populate over the following minutes.

---

## 8. Honesty clauses

1. I did not start a 2-day training run. The evidence said it would fail identically to V5; the expected outcome didn't justify the A100 hours.
2. The analysis-signals injection is an additive prompt-engineering win. It does NOT make the model itself smarter — it removes the model's ignorance of data the pipeline already computed. Pure plumbing.
3. The +25-45% lift estimate from the technique table is a *ceiling* from external benchmarks; actual lift on your UAT corpus depends on the eval harness (Day 1 morning). Treat every number as provisional until measured.
4. Gap E (session-level re-ranking from feedback) and Gap F (thinking traces to UI) from Part 1 are still deferred — neither is on the critical path for the two UAT failure modes.

---

*End of Part 2.*
