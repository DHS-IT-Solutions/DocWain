# DocWain Model Intelligence Sprint — Design Spec

**Date:** 2026-04-15
**Status:** Approved

## Overview

An intensive training sprint to transform DocWain from a fine-tuned Qwen3-14B into a true document intelligence base model (`DocWain-14B-v2`). The sprint uses Claude as an aggressive teacher to generate ~50,000 eval-gated training examples across extraction, reasoning, hallucination prevention, domain awareness, and content generation. By sprint end, DocWain handles any document type with high intelligence, understands intent and context efficiently, and operates with its own identity baked into the weights.

## Goals

1. **Extraction completeness:** ≥90% information capture across all document types (from ~70%)
2. **Hallucination rate:** ≤5% (from ~15%) — fabrication, wrong attribution, and overconfident gaps all addressed
3. **Intent & context understanding:** ≥90% accuracy — Claude-level reasoning on document tasks
4. **Excel/CSV intelligence:** ≥4.0 judge score — multi-sheet reasoning, formulas, cross-reference
5. **OCR & vision mastery:** ≥95% accuracy — degraded scans, handwriting, diagrams, stamps
6. **Deep reasoning & content generation:** ≥4.0 judge score — multi-hop chains, report/email generation grounded in docs
7. **Cross-document intelligence:** ≥4.0 judge score — entity linking, trends, contradictions, aggregation
8. **Domain awareness:** ≥4.0 judge score — domain-specific reasoning across 8 enterprise domains
9. **Base model identity:** DocWain knows who it is without system prompts, never relies on general knowledge padding

## Non-Goals

- General knowledge or coding ability (document intelligence only)
- Web UI or product features (model quality only)
- Multi-language support beyond English (future sprint)
- DocWain-27B (future — this sprint targets 14B)

---

## Architecture

### Training Sprint Structure

Two phases, each flowing into the next immediately upon gate pass. No artificial delays — phases are effort estimates, not calendar boundaries.

**Phase 1: Reliability Foundation (~4 days effort)**

Three parallel data generation tracks:

| Track | Target | Examples | Focus |
|---|---|---|---|
| Anti-Hallucination | ≤5% hallucination | 5,000 DPO pairs | Fabrication rejection, attribution correction, "not found" honesty |
| Completeness & Extraction | ≥90% info capture | 8,000 SFT | All doc types (PDF, DOCX, Excel, CSV, images, scanned, handwritten), tables, footnotes, multi-page continuations, embedded content |
| Intent & Context Understanding | Claude-level reasoning | 5,000 SFT | Intent detection, context-aware responses, multi-turn reasoning, content generation |

Phase 1 training order:
1. Generate all 18,000 examples via Claude distillation (eval-gated per 1,000 batch)
2. SFT on 13,000 examples (completeness + intent) — LoRA rank 64, lr 2e-5, 3 epochs
3. DPO on 5,000 pairs (anti-hallucination) — beta 0.1, lr 5e-6, 1 epoch
4. Phase 1 gate eval

Phase 1 gate criteria (interim):
- Hallucination ≤8%
- Completeness ≥82%
- Intent accuracy ≥85%

Checkpoint: `docwain-v2-phase1`

**Phase 2: Depth & Mastery (~6 days effort)**

Four curriculum tracks plus domain knowledge injection:

| Track | Target | Examples | Focus |
|---|---|---|---|
| OCR & Vision Mastery | ≥95% accuracy | 4,000 SFT + 1,000 DPO | Degraded scans, handwriting, stamps, watermarks, diagrams, charts |
| Excel/CSV Intelligence | ≥4.0 judge score | 4,000 SFT + 1,000 DPO | Multi-sheet reasoning, formulas, pivots, data types, cross-reference |
| Deep Reasoning & Generation | ≥4.0 judge score | 4,000 SFT + 1,000 DPO | Multi-hop chains, comparative analysis, report generation, emails |
| Cross-Document Intelligence | ≥4.0 judge score | 3,000 SFT + 1,000 DPO | Entity linking, temporal trends, contradiction detection, aggregation |
| Domain Knowledge Injection | ≥4.0 judge score | 12,000 SFT | 8 domains x 3 categories (detection, reasoning, cross-domain) |

Phase 2 training order:
1. Generate all 30,000 examples via Claude distillation (eval-gated)
2. SFT on 27,000 examples in curriculum order: OCR → Excel → Reasoning → Cross-doc → Domain
3. DPO on 4,000 pairs
4. Final gate eval

Checkpoints: `docwain-v2-phase2-sft`, `docwain-v2-final`

---

## Claude Distillation Engine

### Generation Pattern

For each training example, Claude receives a real or synthetic document and a task. Claude produces:
1. **Reasoning chain** — step-by-step thinking (becomes DocWain's `<think>` training data)
2. **Response** — grounded answer with citations (becomes DocWain's `<response>` training data)
3. **Self-audit** — what could be missed, confidence level (teaches completeness awareness)

### DPO Pair Generation

Claude generates two responses for the same document + query:
- **Chosen:** Deeply grounded, cites exact sources, says "not found" when appropriate, flags uncertainty
- **Rejected:** Fabricates plausible-sounding facts, attributes to wrong sections, answers confidently without evidence

### Document Type Matrix

| Category | Types | Special Challenges |
|---|---|---|
| Corporate | Invoices, POs, contracts, policies, reports, memos | Tables, legal language, cross-references |
| Financial | Statements, tax forms, audit reports, budgets | Numbers, formulas, multi-period comparison |
| Medical/Clinical | Records, lab reports, prescriptions, discharge summaries | Abbreviations, structured forms, handwriting |
| Legal | Contracts, filings, patents, NDAs | Clause structure, defined terms, amendments |
| HR | Resumes, offer letters, performance reviews, org charts | Mixed formats, personal data, rankings |
| Technical | Specs, manuals, datasheets, SOPs | Diagrams, tables, version tracking |
| Government | Permits, licenses, regulatory filings, compliance docs | Stamps, signatures, form fields |
| Tabular | Excel workbooks, CSV datasets, pivot tables | Multi-sheet, formulas, data types |
| Scanned/Degraded | Faxes, old photocopies, handwritten notes | OCR noise, rotation, partial content |

### Eval Gate

After every batch of 1,000 generated examples:
1. Run through Claude judge (700-example test bank + batch-specific probes)
2. Score across 5 dimensions: accuracy, completeness, reasoning, honesty, format
3. Only merge batches where all dimensions improve or hold steady
4. If a dimension regresses, analyze failures, regenerate targeted examples, re-evaluate

---

## Domain Knowledge Injection

Domain knowledge expressed through **reasoning patterns**, not memorized facts. DocWain learns how to think about a domain, not what domain rules say. The document provides facts; domain awareness provides the lens.

### Domains (8)

| Domain | Reasoning Patterns | Example Intelligence |
|---|---|---|
| Financial | Period comparisons, variance analysis, ratio interpretation, audit trail logic | "Revenue dropped 12% QoQ — flag against expense increase on page 4" |
| Legal | Clause interdependency, obligation tracking, risk escalation, defined term resolution | "Indemnity clause at 7.2 conflicts with liability cap at 9.1" |
| Medical/Clinical | Diagnosis-treatment chains, drug interaction awareness, timeline reconstruction | "Prescribed dosage exceeds standard range given patient's reported weight" |
| HR/Recruitment | Qualification matching, experience normalization, compliance requirements | "Candidate meets 4/6 required qualifications, lacks mandatory certification" |
| Insurance | Policy coverage mapping, claim-to-policy matching, exclusion identification | "Claim falls under flood exclusion in Section 3B of policy" |
| Government/Regulatory | Compliance checklist matching, deadline tracking, form field validation | "Filing deadline 30 days from issue date — 8 days remaining" |
| Technical/Engineering | Spec compliance, version tracking, measurement validation, tolerance checking | "Measured value 4.7mm exceeds specified tolerance of +/-0.2mm from 4.0mm target" |
| Education | Curriculum alignment, grading criteria, accreditation requirements | "3 of 5 learning outcomes not evidenced in submitted portfolio" |

### Training Categories Per Domain

1. **Domain Detection** (500 per domain, 4,000 total) — Identify domain, explain what domain-specific reasoning applies
2. **Domain Reasoning** (800 per domain, 6,400 total) — Demonstrate domain-aware analysis ("what does it mean" not just "what does it say")
3. **Cross-Domain** (1,600 total) — Documents spanning multiple domains, blend reasoning

---

## Eval Framework

### Expanded Test Bank (700 examples)

| Category | Count | Coverage |
|---|---|---|
| Extraction accuracy | 150 | All doc types from type matrix |
| Table/Excel reasoning | 100 | Nested tables, multi-sheet, formulas, pivots |
| OCR/Vision | 80 | Degraded scans, handwriting, diagrams, stamps |
| Hallucination probes | 150 | Trick questions, unanswerable queries, partial docs |
| Intent understanding | 80 | Ambiguous queries, multi-turn, implicit intent |
| Cross-document | 60 | Comparison, aggregation, contradiction detection |
| Content generation | 80 | Summaries, emails, reports grounded in docs |

### Judge Scoring Dimensions (1-5 each)

1. **Accuracy** — Are all stated facts correct and traceable to source?
2. **Completeness** — Did it capture all relevant information?
3. **Reasoning** — Is the thinking chain logical and grounded?
4. **Honesty** — Does it flag uncertainty and say "not found" when appropriate?
5. **Format** — Is the output well-structured and appropriate for the task?

### Regression Detection

Every checkpoint evaluated against the full 700-example bank. Scores tracked per dimension per document type. If any cell regresses by >0.3 points, training pauses, generates targeted recovery data, retrains before proceeding.

### Automated Pipeline

```
Generate data → Train → Eval (700 examples) →
  All dimensions >= threshold? → Proceed to next phase
  Any regression? → Diagnose → Generate targeted fix data → Retrain → Re-eval
```

Runs autonomously. Manual intervention only if a gate fails twice consecutively.

---

## Model Identity & Base Model Conversion

### Identity Training

Woven into every training example (not a separate track):
- System prompt dependency eliminated — DocWain knows who it is without being told
- Responses always grounded in document evidence, never general knowledge padding
- Consistent voice: precise, enterprise-grade, cites sources naturally
- Self-aware of capabilities: "I can analyze this document but cannot access external data"

### Base Model Conversion (After Final Gate)

1. Merge all LoRA adapters into full weights (16-bit)
2. Strip Qwen identity tokens/behaviors from tokenizer config
3. Rebrand model metadata: `DocWain-14B-v2`
4. Convert to GGUF for Ollama compatibility (fallback)
5. Convert to vLLM-optimized format (FP16 + speculative decoding config)
6. Upload to HuggingFace as `MuthuSubramanian/DocWain-14B-v2`

### Post-Sprint Serving

- vLLM fast instance serves `DocWain-14B-v2` (replaces `docwain-fast`)
- Smart instance stays on Qwen3.5-27B (until DocWain-27B future sprint)
- Standalone service picks up new model via `VLLM_MODEL_NAME` config

---

## Training Data Summary

| Category | SFT | DPO | Total |
|---|---|---|---|
| Phase 1: Completeness & Extraction | 8,000 | — | 8,000 |
| Phase 1: Intent & Context | 5,000 | — | 5,000 |
| Phase 1: Anti-Hallucination | — | 5,000 | 5,000 |
| Phase 2: OCR & Vision | 4,000 | 1,000 | 5,000 |
| Phase 2: Excel/CSV | 4,000 | 1,000 | 5,000 |
| Phase 2: Deep Reasoning & Generation | 4,000 | 1,000 | 5,000 |
| Phase 2: Cross-Document | 3,000 | 1,000 | 4,000 |
| Phase 2: Domain Knowledge | 12,000 | — | 12,000 |
| Existing data | 1,000 | 40 | 1,040 |
| **Total** | **41,000** | **9,040** | **~50,000** |

---

## Final Targets

| Metric | Current | Target |
|---|---|---|
| Hallucination rate | ~15% | ≤5% |
| Extraction completeness | ~70% | ≥90% |
| Intent understanding | unknown | ≥90% |
| Excel/CSV judge score | N/A | ≥4.0 |
| OCR accuracy | ~85% | ≥95% |
| Reasoning depth | ~2.5 | ≥4.0 |
| Cross-doc intelligence | basic | ≥4.0 |
| Content generation | unknown | ≥4.0 |
| Domain awareness | none | ≥4.0 |

## Deliverable

`DocWain-14B-v2` base model — own identity, deployed to vLLM, uploaded to HuggingFace. A document intelligence model that matches Claude on document tasks and surpasses it with persistent memory, cross-document evolution, and domain-aware reasoning.
