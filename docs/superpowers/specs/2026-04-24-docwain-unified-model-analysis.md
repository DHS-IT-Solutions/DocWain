# DocWain Unified Model — Research & Implementation Analysis

**Date:** 2026-04-24
**Branch:** `preprod_v02`
**Status:** Research / analysis; not yet a training plan
**Owner:** Muthu (commissioned this analysis)

## 1. What "unified DocWain model" means

**One model, served from one endpoint (vLLM local), that handles every task DocWain needs:**

1. **Vision + OCR + handwriting** — read printed and handwritten text from page images, receipts, scanned PDFs.
2. **Layout understanding** — identify regions (tables, forms, figures, headers), reading order, document type (invoice, resume, contract).
3. **DocIntel capabilities** — classifier (routing), coverage verifier (did we miss anything on the page?).
4. **Entity and relationship extraction** — for Knowledge Graph ingestion.
5. **Response generation with grounding** — RAG-style answers over retrieved chunks with citations.
6. **Query understanding and decomposition** — parse user intent, choose task type, pick relevant docs.
7. **Structured output generation** — tables, comparisons, timelines.
8. **Chart / graph generation** — produce the `DOCWAIN_VIZ` payloads the frontend renders.
9. **Domain-aware content generation** — emails, summaries, reports adapted to the document domain.
10. **Researcher-Agent behavior** — deep domain-specific analysis (medical, expenses, contracts) via tool use.
11. **Native tool calling** — Qwen3-style `<tool_call>` format so the model can delegate sub-tasks.

**Not a fast/smart split** — one served model. No routing between a small and a large model. Reaffirmed user directive per `feedback_unified_model.md`.

## 2. What this document is (and isn't)

**Is:** a rigorous technical analysis of whether and how a single 14B-class unified DocWain model can realistically cover all 11 capabilities, what the architecture should look like, what training curriculum is needed, what the honest capability ceiling is, and how to evaluate it.

**Isn't:** a training plan to start executing. That's a follow-on (multi-month) workstream after this analysis is approved. Per `feedback_engineering_first_model_last.md`, training only starts after engineering layer captures patterns and identifies the specific gaps training should close.

## 3. Current state — honest baseline

- **V2 model:** Qwen3-14B-bnb-4bit base + SigLIP-SO400M vision encoder + trainable projection MLP. Native tool-calling format. Served via vLLM on port 8100 with local bf16 weights (~28 GB).
- **Training history:**
  - V2 iter_3 merged_16bit scored 4.71/5.0 on its production gate — passed.
  - V3 (weekend, 2026-04-10/12): SFT 0.127, DPO 0.096, 28K SFT / 3.8K DPO pairs, 86% extraction bench pass.
  - V4 (prepared): 31K / 4.3K, not yet trained.
  - **V5 (2026-04-18/20): FAILED.** Passed 1/7 hard gates (teacher) and 2/7 (distilled 8B). Reverted to V2. Post-mortem at `project_v5_post_mortem.md`.
- **Observed limits** (per backend-quality audit 2026-04-23):
  - Local DocWain-14B-v2 in vLLM (bf16) trails Ollama Cloud `qwen3.5:397b` materially on response-intelligence queries (Q9/Q10 of that audit — candidate inferential reasoning, interview-question generation).
  - Local DocWain-14B-v2 in Ollama (Q5_K_M GGUF) produced hallucinations (`$100,000` fabricated on a "name/email/phone" question) and degenerate token loops.
  - Vision capability on scanned / handwritten content: unknown baseline; Plan 2's vision orchestrator uses DocWain for classification + coverage + extraction with Tesseract/EasyOCR fallback catching misses. Fallback invocation rate today is expected to be high.

**Bottom line:** the architecture is already unified. The gap is **capability coverage** — V2 doesn't yet do all 11 tasks at production quality. Training is the closer of that gap.

## 4. State-of-the-art reference points (April 2026)

Models at or near 14B that demonstrably do subsets of the 11 capabilities:

| Model | Vision | OCR | Handwriting | Doc layout | Reasoning | Tool call | Size |
|---|---|---|---|---|---|---|---|
| **Qwen3-VL-14B** (hypothetical)* | ✓ | ✓ | partial | ✓ | strong | native | 14B |
| **Qwen3-VL-32B** | ✓ | ✓ | partial | ✓ | strong | native | 32B |
| **InternVL-3-14B** | ✓ | ✓ | partial | ✓ | solid | via prompt | 14B |
| **Phi-4-multimodal** | ✓ | ✓ | limited | limited | solid | limited | ~5B |
| **DocOwl 2** | ✓ | strong | limited | ✓ | fair | no | 7B |
| **Nougat** (HF) | ✓ | specialized | no | specialized | no | no | 350M |
| **Tesseract + EasyOCR** (non-ML) | image only | strong printed | weak | none | no | no | — |

*Qwen3-VL-14B as a checkpoint may not exist; the Qwen3-VL family publicly ships 2B/7B/32B/72B as of April 2026. If a 14B VL variant isn't available, the realistic bases are (a) Qwen3-VL-7B or (b) Qwen3-VL-32B, or (c) keep V2's architecture (Qwen3-14B-text + grafted SigLIP).

**No open 14B-class model covers all 11 capabilities at production quality today.** Every candidate trails closed-source frontier models (GPT-4V, Claude Opus) on response-intelligence. Gap is narrower on vision/OCR/extraction where open models have caught up.

## 5. Can one 14B-class model do all 11 tasks?

Honest answer: **yes for tasks 1–8; no for task 9 (domain-aware content) and task 10 (Researcher-Agent reasoning depth) at GPT-4/Claude-Opus quality level** — unless we are comfortable with a quality ceiling lower than frontier cloud models.

### Capability ceiling estimate (based on public SOTA for 14B-class)

| Capability | Achievable on 14B unified | Gap to frontier cloud | Mitigation |
|---|---|---|---|
| OCR on printed docs | 95–98% char acc | small | OCR fallback closes the last 2% |
| Handwriting OCR | 80–90% char acc | moderate | fallback + human review for low-conf |
| Layout understanding | ≥ Azure DocAI parity achievable with training | small | bench measures it |
| DocIntel classifier | high (simple routing) | tiny | prompting suffices today |
| DocIntel coverage verifier | high | small | ditto |
| Entity/relation extraction | solid (matches specialized 7B extractors) | small | plus KG consolidation |
| Table structure | achievable with targeted SFT | small | table-transformer benchmarks |
| Response generation (grounded) | solid but trails 397B cloud | **material** — see §3 audit | retrieval quality + Researcher precompute |
| Structured output (tables, timelines) | solid | small | schema-enforced decoding |
| Chart/graph generation (DOCWAIN_VIZ) | achievable with 500–2000 SFT examples | small | format is structured, not creative |
| Domain-aware content generation | **weak without targeted SFT** | **material** | domain adapters (YAML) + few-shot |
| Researcher-Agent deep analysis | **depth-limited at 14B** | **material** | engineering layer (iterative prompting, tool use) |
| Tool calling | already native in Qwen3 | none | works today |

Two material gaps: **response-intelligence depth** and **domain-aware content generation depth**. Both compensatable at the engineering layer (better prompts, retrieval, Researcher precomputation, adapter-based domain specialization), but the 14B model alone won't match GPT-4 on a cold prompt.

**User's pre-registered acceptance of this gap:** the 2026-04-23 audit surfaced the local-vs-cloud gap; user explicitly chose vLLM local primary despite it, with the trade-off being latency (4.7s vs 45s) and the plan being "mitigate at engineering layer." This analysis matches that directive.

## 6. Architectural options

### Option A — Stay on V2's base, upgrade training (RECOMMENDED)
- Base: Qwen3-14B-bnb-4bit + SigLIP-SO400M + trainable projection MLP (current V2)
- Pros: continuity with V2's production weights; vLLM serving already works; 32k context; native tool calls baked in from prior phases.
- Cons: SigLIP was trained for general vision, not OCR-focused; handwriting is a weak point.

### Option B — Switch to Qwen3-VL as base
- Base: Qwen3-VL-7B (if 14B unavailable) or Qwen3-VL-32B
- Pros: purpose-built for vision+doc; better OCR ceiling out of the box.
- Cons: requires re-doing all prior DocWain training from scratch on new base; loses V2's DocWain behavior that was trained in; 32B needs AWQ to fit comfortably on A100 80GB alongside a second instance; 7B may trail V2 on text reasoning.

### Option C — Dual-encoder (reject)
- Two vision encoders (one OCR-specialized, one general)
- Rejected: violates "no secondary ML models" directive from extraction spec — user wants the unified DocWain to be the only ML model.

**Recommendation: A.** Continue evolving V2 via targeted training phases. The architecture is sound; training is what's under-delivered. V5's failure was a training-pipeline failure, not an architectural failure.

## 7. Training curriculum (phased)

Per `feedback_v5_failure_lessons.md`, the eight hard rules apply. Each phase ships only after its gate is green.

### Phase 0: Measurement harness (non-negotiable, first)
- Build a capability-by-capability eval bench with ground truth.
- Task-specific rubrics (see §9).
- Baseline V2 against the bench. Report the numbers. These are the regression gates.

### Phase 1: OCR + layout ground layer
- Goal: raise printed-text OCR to 95%+ char accuracy; table-structure accuracy ≥ 0.9 F1.
- Data: DocLayNet, DocVQA (doc-VQA tasks where answer is a bbox), FUNSD, CORD, SROIE, ICDAR-XML. Mixed real + synthetic.
- Training: SFT on the vision-encoder path (projection MLP + top 4 LLM layers). ~50K examples, 2 epochs.
- Gate: OCR + layout bench improves over V2 baseline by ≥ 10% absolute AND no regression on response-generation bench.

### Phase 2: Handwriting + low-resource scripts
- Goal: handwriting 80%+ char accuracy; receipt + form subset covered.
- Data: IAM, RIMES, ICDAR handwriting sets; synthetic handwritten augmentations of existing docs.
- Training: SFT deeper into the LLM (top 10 layers). 30K examples.
- Gate: handwriting bench ≥ 0.80; printed OCR regression ≤ 1% absolute.

### Phase 3: DocIntel + extraction schema
- Goal: classifier, coverage verifier, structured extraction as trained skills (not just prompted).
- Data: distill from the engineering layer's production logs. Each real extraction → (input, DocWain output, operator correction) triple. The "coverage verifier said yes but human said no" examples are especially valuable.
- Training: mixed SFT + DPO on corrections.
- Gate: extraction bench (Plan 1's bench) ≥ 0.92 weighted on native path, ≥ 0.95 weighted on vision path. Fallback invocation rate in production ≤ 10%.

### Phase 4: Response generation + tool calling
- Goal: close the Q9/Q10 response-intelligence gap to Cloud 397B where possible. Tool calls for long-chain tasks.
- Data: distill from Cloud 397B on DocWain-specific query bank; DPO pairs from user feedback. Tool-call traces from engineering-layer agent use.
- Training: SFT + iterative DPO.
- Gate: backend-quality audit (re-run of 2026-04-23 audit) — local V_next beats or ties V2 on every query category AND closes ≥ 50% of the gap to Cloud 397B on response-intelligence questions.

### Phase 5: Chart/graph + structured output
- Goal: produce DOCWAIN_VIZ-formatted payloads that frontend can render; structured tables respect user-facing schemas.
- Data: 500–2000 hand-crafted examples of (question, retrieved chunks, expected chart JSON). Plus distillation from cloud on structured-output queries.
- Training: small targeted SFT on tops. Schema-aware decoding if supported by vLLM.
- Gate: structured-output bench ≥ 0.95 schema-valid rate; visualization rendered correctly in a UI smoke test on 20 queries.

### Phase 6: Domain adapters (plugin-shaped, per `feedback_domain_extensibility.md`)
- Goal: finance, legal, hr, medical, it_support domains behave with domain-aware prompting. Generic stays default.
- Data: per-domain adapter YAMLs in Azure Blob (per `feedback_adapter_yaml_blob.md`); SME-style distillation on domain examples.
- Training: adapter-specific LoRA, hot-swappable at inference.
- Gate: per-domain rubric ≥ 4.0/5.0 on SME-evaluated samples. Generic adapter regression ≤ 2% absolute.

### Phase 7: Identity + behavior (gateway shim as pragmatic floor)
- Goal: DocWain responds as DocWain without a 200-line system prompt.
- Data: 3000+ diverse identity framings (`feedback_v5_failure_lessons.md` lesson 2) plus behavioral rules. Verified on held-out probe.
- Training: identity baked via SFT on empty-system-prompt examples. If the held-out probe fails, enable the gateway prompt shim (lesson 7) as pragmatic fix rather than re-training indefinitely.
- Gate: identity probe ≥ 0.95 on held-out set. Without the gate shim. If not achievable in reasonable time, ship the shim.

## 8. Data-mix discipline

Per `feedback_no_customer_data_training.md`: **no customer documents in training data, ever**. Synthetic + public + metadata-only patterns only.

Per phase:

- Phase 1: DocLayNet, DocVQA, FUNSD, CORD, SROIE, IAM, RIMES — all public. Synthetic augmentation of formatting variants.
- Phase 2: IAM, RIMES, ICDAR — public. Synthetic handwriting overlays generated from known ground truth.
- Phase 3: production engineering-layer logs — **patterns only** (doc types, layouts, extraction correction deltas), not raw customer content. Per `feedback_intelligence_precompute.md`, extraction already pattern-captures at ingest.
- Phase 4: distillation from Cloud 397B on curated question banks synthesized from public domain documents (e.g., SEC filings, ArXiv papers, legal contracts from public-domain corpora). Never customer queries or responses as training data.
- Phase 5: 500–2000 hand-crafted (question, chart-JSON) examples using public data.
- Phase 6: domain-specific public data per adapter (e.g., public medical abstracts for medical adapter). No customer domain data.
- Phase 7: fully synthetic identity framings.

## 9. Evaluation harness (built before any phase starts)

Per `feedback_measure_before_change.md`, non-negotiable.

**Nine per-capability benches, ground-truth-backed:**

1. **OCR bench** — 30 printed-doc images + ground-truth text. Metric: character accuracy, word accuracy.
2. **Handwriting bench** — 20 handwritten images + ground truth. Char accuracy, word accuracy.
3. **Layout bench** — 30 docs with ground-truth region bboxes and types. IoU on region detection, F1 on region classification.
4. **Table-extraction bench** — 30 docs with ground-truth table structures. Row/column F1; cell-level exact-match.
5. **Entity/relation bench** — 30 docs with ground-truth entities and relationships. Entity F1; relation F1.
6. **Response-generation bench** — 50 question+corpus tuples with reference answers (hand-curated). LLM-judge + human spot-check. Match the 2026-04-23 audit structure so results are comparable.
7. **Structured-output bench** — 30 queries with expected table/chart JSON schemas. Schema-validity rate; field-level exact match.
8. **Tool-calling bench** — 20 tasks that require tool invocation. Correct tool chosen + correct args + correct final answer.
9. **Identity probe** — held-out probe set of 200 identity-questioning prompts. "I am DocWain" or equivalent in ≥ 95%.

Each bench has a "V2 baseline" (run before training starts) and a "promotion threshold" (the gate). Any phase that regresses any bench > 2% absolute blocks promotion until root-caused.

## 10. What the engineering layer still does (model doesn't have to)

Honest accounting: even a well-trained unified model should NOT try to absorb these responsibilities.

| Responsibility | Stays in engineering |
|---|---|
| Retrieval (vector + graph) | Qdrant + Neo4j + HybridRetriever — precomputed at ingestion, read at query time |
| Researcher-Agent deep analysis | Training stage parallel track; outputs to Qdrant + Neo4j (Plan 4). Model generates insights but the *orchestration* and *persistence* is engineering |
| Domain adapter resolution | YAML adapters in Azure Blob, loaded at runtime with subscription override |
| Pipeline orchestration (Celery, HITL gates) | Stays in `src/tasks/`, `src/api/pipeline_api.py` |
| Extraction fallback ensemble (Tesseract, EasyOCR) | Retained until model OCR exceeds their accuracy — then removed per Plan 2 phase 3 |
| Query-time latency budget / streaming | vLLM serving config, not model behavior |
| Observability logs | Redis `kg:log`, `extraction:log` from Plans 2 + 3 |
| KG consolidation + ingestion | Plan 3 already handles this — model just emits structured extraction |

The model's job is **producing the right tokens** for each task. Everything else is engineering.

## 11. Risk + failure-mode mapping to V5 lessons

Per `feedback_v5_failure_lessons.md`. For each hard rule, this analysis's compliance:

1. **Gate distillation on teacher passing identity.** Phase 7 gate is identity probe ≥ 0.95. Distillation (any teacher→student) in Phase 4 ONLY after teacher passes. ✓
2. **Identity-in-weights has historically failed.** Phase 7 budgets 3000+ framings + held-out probe. Gateway shim is a pre-planned fallback if weights-only doesn't converge. Honest contingency. ✓
3. **Validate scorers before trusting results.** Every bench in §9 gets a validation round: run scorer on 10 known-good + 10 known-bad samples before phase gates rely on it. ✓
4. **Stop the line on failed teacher gates.** Phase 3 distillation triggers only after phase 2 gates are green. No partial-pass bypass. ✓
5. **Orchestrator must not proceed on failure.** Training orchestrator code (future) has a hard halt on any gate fail. Not "flag and continue". ✓
6. **TIES merging is incompatible with vision-grafted base.** Noted. Phases 1-7 use standard SFT/DPO only; no TIES. ✓
7. **Gateway shim is an acceptable fallback.** Phase 7 explicitly names it. No religious insistence on weights-only. ✓
8. **Session isolation during training.** Pre-flight: `systemctl disable --now ollama`, `systemctl stop docwain-vllm-fast` during long training runs. GPU scheduler re-claims GPU for serving when training pauses. ✓

Additional risks specific to this unified-model ambition:

- **Catastrophic forgetting across phases.** Multi-task training on vision may degrade pure-text reasoning. Mitigation: rehearsal mix (5–10% of each earlier phase's data carried into subsequent phases); per-phase regression gates on every earlier capability bench.
- **Phase 4 distillation bottleneck.** If Cloud 397B quota is rate-limited, Phase 4 data collection slows. Mitigation: batch distillation into 10K-query runs with caching; distill once, train many.
- **Vision encoder bottleneck.** SigLIP may ceiling at ~95% OCR char accuracy regardless of LLM training. If Phase 1 + 2 plateau below targets, evaluate a vision-encoder swap as Phase 1.5 (Qwen3-VL encoder replacement).
- **Training-compute ceiling.** A100 80GB can handle 14B SFT at reasonable speed (~40-60 hours per phase's 30-50K example set). 32B would require distributed or AWQ-quantized training. Budgeting assumes 14B stays.

## 12. Realistic timeline + cost

Conservative per-phase budget:

| Phase | Wall-clock | Includes |
|---|---|---|
| Phase 0 (bench build) | 1–2 weeks | Dataset curation, ground-truth labeling, scorer validation |
| Phase 1 (OCR + layout) | 2–3 weeks | Data prep, SFT run, DPO on corrections, eval, promotion |
| Phase 2 (handwriting) | 2–3 weeks | Same shape, narrower data |
| Phase 3 (DocIntel + extraction) | 3–4 weeks | Production-log distillation needs cleaning pipeline first |
| Phase 4 (response gen) | 3–4 weeks | Cloud distillation bottleneck |
| Phase 5 (structured output + charts) | 1–2 weeks | Small, targeted |
| Phase 6 (domain adapters) | 2–3 weeks per domain | Domain SMEs required for evaluation |
| Phase 7 (identity) | 1–2 weeks | Plus fallback shim ready |

**Total: 4–6 months wall clock for phases 1–5 + identity. Phase 6 is per-domain and ongoing.**

Plan for 2x schedule slippage budget based on V5's history: **6–12 months** realistic.

Rough compute cost: 8 A100 days per phase × 7 phases = 56 A100-days. At current A100 utilization, this is compatible with existing GPU scheduling if training runs on off-peak and serving has priority.

**Phase 6 (domain adapters) scales by how many domains we ship.** Ongoing work, not a 6-month commitment.

## 13. What this analysis recommends

**Yes — build a unified DocWain model.** It's technically feasible at the 14B scale. The architecture (Qwen3-14B + SigLIP + tool calling) is sound. V5 failed on training pipeline, not on architecture.

**But:**

1. **Don't start training tomorrow.** Build Phase 0's evaluation harness first. The nine per-capability benches ARE the contract. Without them, we repeat V5.
2. **Engineering-layer work comes first or in parallel** (extraction adapter, Researcher Agent, Plan 4 serving swap). Those give the training data (production logs, correction deltas) and prove out what the model actually needs to learn. Per `feedback_engineering_first_model_last.md`.
3. **Accept a quality ceiling** on response-intelligence and domain-aware generation that's below Cloud 397B. Close the gap at the engineering layer (RAG, Researcher Agent, domain adapters), not only at the model layer.
4. **Plan for the gateway prompt shim** as a live backup for identity-in-weights if Phase 7 doesn't converge in 2 weeks.
5. **Gate each phase rigorously.** 2% absolute regression on any prior bench blocks promotion. No "partial pass + move on" fallbacks — that's what burned V5.
6. **Measure before every phase.** V2 re-baseline as the data-prep step, not as a retrospective.

**Explicit non-recommendation:** do NOT skip Phase 0. Every phase after it depends on benches that don't exist yet. If benches are skipped, we're building in the dark again — the condition V5 failed under.

## 14. What Plan 4 could look like (if user wants to start)

If this analysis is approved, **Plan 4 (next spec) is Phase 0: the evaluation harness.**

That spec would cover:
- Dataset curation (which corpora to pull, what to label)
- Ground-truth labeling workflow (hand, tooled, or distilled-then-reviewed)
- Per-capability scorer implementations with unit tests
- V2 baseline numbers recorded per capability
- The monitoring dashboard for per-bench scores over time

Estimated: 2–3 weeks to deliver Phase 0 fully. Then Phase 1 (OCR + layout training) starts with real data and real gates.

**That is the right next step if the roadmap pivots toward the unified model now.**

If the roadmap instead continues with the original post-audit sequence (Plan 4 = vLLM primary serving, Plan 5 = teams_app + standalone rebuild), this analysis stays on the shelf as the reference document for whenever the training workstream begins.

## 15. Summary

- A unified DocWain model handling all 11 capabilities at 14B scale is **feasible** but **non-trivial**.
- **Architecture is already unified** (V2 = Qwen3-14B + SigLIP + tool calls). No architecture change needed to start.
- **The gap is training**, and training has failed before (V5). This analysis maps the 8 hard lessons from V5 to concrete gates.
- **6–12 months realistic** to train through phases 1–5 + identity, plus per-domain adapter work ongoing.
- **Engineering layer must come first** or parallel. Model doesn't absorb retrieval, orchestration, adapter resolution, fallback, observability.
- **Accept the ceiling.** Cloud 397B stays ahead on response-intelligence and deep domain reasoning. Close gap at engineering layer.
- **Don't start without Phase 0 (eval harness).** That was the V5 failure in one sentence.
- **Next step** if this analysis is approved: Plan 4 spec = Phase 0 (eval harness + V2 re-baseline).
