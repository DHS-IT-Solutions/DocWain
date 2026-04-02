# DocWain V2+ Finetuning Pipeline — Design Specification

**Date:** 2026-04-02
**Status:** Approved
**Hardware:** NVIDIA A100-SXM4-80GB (80GB VRAM, CUDA 12.4)
**Base Model:** Vision-grafted Qwen3-14B (SigLIP-SO400M + ProjectionMLP + Qwen3-14B)
**Approach:** Sequential Enhancement (V2 pipeline) + Post-Training Refinement Stack
**Judge:** Claude Code (in-process, no external API)
**Data Generator:** Claude Code (synthetic, metadata/patterns/QA only)

---

## Overview

Three-stage upgrade to transform DocWain from a document retrieval assistant into a GPT-class document intelligence model with holistic contextual analysis:

1. **Stage 1 — Data Generation:** Claude Code generates all training datasets (~52K examples total) using its own knowledge and intelligence. No raw document content — only metadata, patterns, Q&A pairs, and reasoning traces.
2. **Stage 2 — Enhanced V2 Pipeline:** Run the existing 6-phase pipeline with upgraded data plus a new Phase 3.7 (Holistic Reasoning SFT).
3. **Stage 3 — Post-Training Refinement:** Three refinement rounds on the merged model for conversational quality, confidence calibration, and inference speed.

**Core constraint:** Fine-tuning trains on metadata, patterns, and Q&A pairs ONLY. Never on raw document content.

---

## Intelligence Architecture — The Four Layers

```
Layer 1: PERCEPTION (Phase 1)
├── Vision encoder sees document images
├── OCR + layout recognition
└── Structural understanding (headers, sections, hierarchy)

Layer 2: COMPREHENSION (Phase 2 + 2.5)
├── What does this document say? (extraction)
├── What does it mean? (semantic understanding)
├── What's missing or unusual? (gap/anomaly detection)
└── How does it relate to other documents? (cross-ref)

Layer 3: ANALYSIS (Phase 3 + 3.5 + 3.7)
├── Tool-calling for structured operations
├── Patterns across documents (trend/pattern recognition)
├── Risk assessment (compliance, anomaly severity)
├── Comparative reasoning (version diffs, benchmarking)
├── Holistic synthesis (multi-source narrative construction)
└── Intent-aware depth calibration

Layer 4: COMMUNICATION (Post-Training Rounds 1-3)
├── Natural conversational flow (multi-turn, disambiguation)
├── Calibrated confidence ("I'm 85% sure because...")
├── Visualization decisions (when to chart vs narrate)
└── Concise reasoning (distilled <think> blocks for speed)
```

---

## Stage 1: Data Generation (Claude Code)

Claude Code generates all datasets in JSONL format. Every training example includes a `<think>` reasoning block.

### JSONL Schema Per Phase

**SFT format (Phase 2, 3, 3.5, 3.7, Round 2, Round 3):**
```json
{"text": "<|im_start|>system\nYou are DocWain...<|im_end|>\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n<think>{reasoning}</think>\n{response}<|im_end|>"}
```

**DPO format (Phase 2.5, Round 1):**
```json
{"prompt": "{system + user message}", "chosen": "<think>{good_reasoning}</think>\n{good_response}", "rejected": "<think>{bad_reasoning}</think>\n{bad_response}"}
```

**Eval format:**
```json
{"benchmark": "TableBench", "query": "...", "context": [...], "reference_answer": "...", "rubric": "...", "expected_tools": [...]}
```

### Phase 2 Data — Document Intelligence (20K examples)

**Output:** `finetune_data/v2/doc_intelligence/{table,layout,ocr,cross_ref}.jsonl`

#### Category 1: Table Understanding (8K, 40%)

| Tier | Count | Description |
|---|---|---|
| Simple | 3K | Single table, clear headers, direct value lookup |
| Medium | 3K | Multi-table, merged cells, computed values, cross-table comparison |
| Hard | 2K | Nested tables, cross-table references, implicit relationships, mathematical verification |

Every `<think>` block demonstrates:
1. Table boundary and header identification
2. Row/column intersection tracing
3. Step-by-step computation
4. Cross-table reference resolution
5. Confidence reasoning based on data clarity

#### Category 2: Layout Comprehension (5K, 25%)

Training targets:
- Header hierarchy detection (H1 > H2 > H3 nesting)
- Multi-column reading order (newspaper-style layouts)
- Sidebar vs main content discrimination
- Form field-label association (spatial proximity mapping)
- Page header/footer vs body content separation
- Section boundary detection and nesting

#### Category 3: OCR Correction (4K, 20%)

Realistic degradation patterns:
- Character confusion matrices (l/1/I, O/0, rn/m, cl/d)
- Word-level recovery from context ("arnount" -> "amount")
- Table structure recovery from misaligned OCR
- Handwriting interpretation with uncertainty flagging
- Mixed-quality regions (clean digital + degraded scan on same page)

#### Category 4: Cross-Document Reasoning (3K, 15%)

Multi-document examples (2-4 document metadata excerpts per example):
- Contract vs amendment: term change identification
- Invoice vs PO: matching and discrepancy detection
- Resume vs job description: qualification matching
- Policy vs compliance report: coverage analysis
- Multi-version comparison: evolution tracking

### Phase 2.5 Data — DPO Pairs (5K pairs)

**Output:** `finetune_data/v2/doc_intelligence/dpo_pairs.jsonl`

Each pair: `{prompt, chosen, rejected}` where chosen has correct extraction + sound `<think>` reasoning.

Corruption dimensions (applied via enhanced `corrupt_extraction()` + Claude Code generation):

1. **Reasoning corruption:** Good `<think>` vs sloppy/wrong reasoning chains
2. **Hallucination injection:** Plausible but fabricated values (close-but-wrong numbers, invented entities)
3. **Over-confidence corruption:** States "High confidence" on ambiguous/degraded data
4. **Omission corruption:** Misses clearly present information
5. **Structure corruption:** Breaks table/entity/relationship structure

### Phase 3 Data — Tool-Calling (8K examples)

**Output:** `finetune_data/v2/tool_calling/tool_calling_sft.jsonl`

Existing `tool_data_generator.py` categories (4K from existing generator) plus Claude Code enhancements (4K new):

**New data types:**
- **Self-verification chains:** Model invokes tool, checks result, re-invokes if inconsistent
- **Multi-step tool orchestration:** Complex tasks requiring 2-4 sequential tool calls
- **Auto-invocation triggers:** Model recognizes when a tool should be used unprompted
- **Tool refusal:** Scenarios where no tool is appropriate (model should reason directly)

9 core tools covered: `ocr_extract`, `layout_extract`, `extract_table`, `extract_entities`, `context_understand`, `cross_reference`, `search_documents`, `summarize_section`, `visualize_data`

### Phase 3.5 Data — Insight Generation (6K examples)

**Output:** `finetune_data/v2/insights/insight_training.jsonl`

| Category | Count | Description |
|---|---|---|
| Pattern Recognition | 1K | Recurring themes, repeated structures across documents |
| Anomaly Detection | 1K | Statistical outliers, logical inconsistencies, unexpected values |
| Trend Analysis | 1K | Temporal patterns with projections, period-over-period changes |
| Comparative Analysis | 1K | Side-by-side document evaluation, version diffs |
| Gap Analysis | 800 | Missing information, incomplete coverage, absent required sections |
| Holistic Synthesis (new) | 700 | Multi-source narrative construction, executive summaries |
| Risk Assessment (new) | 500 | Compliance/financial/operational risk scoring with severity |

Each insight follows the **DocWain Analysis Frame:**

```
<think>
[Step 1: What am I looking at? — Document type, domain, scope]
[Step 2: What are the key facts? — Critical data points]
[Step 3: What patterns emerge? — Trends, repetitions, anomalies]
[Step 4: What's missing? — Expected but absent information]
[Step 5: What does this mean? — Implications, risks, recommendations]
[Step 6: How confident am I? — Evidence strength, gaps, caveats]
</think>

**Summary:** [1-2 sentence executive summary]
**Key Findings:** [Evidence-backed findings]
**Analysis:** [Reasoning connecting findings to conclusions]
**Risk/Opportunity Flags:** [Actionable alerts]
**Confidence:** [Calibrated score with reasoning]
```

### Phase 3.7 Data — Holistic Reasoning (8K examples)

**Output:** `finetune_data/v2/holistic/holistic_training.jsonl`

#### Mode 1: Intent Decomposition (2K)

Model learns to unpack vague queries into structured analytical plans via SFT examples (not DPO — this is SFT training):
- "Tell me about this contract" -> `<think>` decomposes into key parties, terms, risks, deadlines
- "Is this employee a good hire?" -> `<think>` maps to skills match, experience gaps, red flags
- "What should I know about this vendor?" -> `<think>` plans cross-doc synthesis: contracts, invoices, performance

Each example shows the model's `<think>` block explicitly decomposing user intent, then producing a structured response that covers both stated and unstated needs.

#### Mode 2: Evidence Synthesis (2.5K)

Given 5-12 evidence chunks (typical RAG retrieval), model learns:
1. **Triage:** Critical vs supporting vs noise chunks
2. **Connect:** Inter-chunk relationships and chronology
3. **Resolve:** Contradiction resolution with authority reasoning
4. **Narrate:** Coherent story, not bullet dump

#### Mode 3: Contextual Depth Calibration (1.5K)

| Query Type | Expected Depth | Response Pattern |
|---|---|---|
| Lookup | Direct answer + source | 1-2 sentences, citation |
| Extract | Structured output + completeness check | Table/list with coverage note |
| Analyze | Multi-paragraph with reasoning | Assessment with criteria and evidence |
| Synthesize | Full analytical frame | Cross-doc narrative with timeline, trends, risks |

Model auto-detects query depth. Short questions don't get essays. Complex questions don't get one-liners.

#### Mode 4: Domain-Aware Reasoning (2K)

Domain-specific analytical frameworks:

- **Legal:** Obligation mapping, risk clause identification, version comparison
- **Financial:** Numerical consistency, period-over-period, variance analysis
- **HR:** Compliance gaps, lifecycle patterns, qualification matching
- **Medical:** Temporal treatment sequencing, contraindication detection, protocol adherence
- **Policy/Compliance:** Section coverage, regulatory alignment, gap identification

### Post-Training Round 1 Data — Conversational DPO (3K pairs)

**Output:** `finetune_data/v2/post_training/conversational_dpo.jsonl`

| Dimension | Chosen | Rejected |
|---|---|---|
| Opening | Directly addresses question | "Based on my analysis of the provided documents..." |
| Flow | Natural transitions | Bullet dump, no connective tissue |
| Follow-ups | Builds on prior context naturally | Treats each turn as independent |
| Disambiguation | Asks clarifying question when ambiguous | Guesses wrong silently |
| Uncertainty | Acknowledges conflicting sources with reasoning | Picks one arbitrarily |
| Tone | Professional, approachable, matches user register | Overly formal or robotic |

### Post-Training Round 2 Data — Confidence Calibration (2K examples)

**Output:** `finetune_data/v2/post_training/confidence_sft.jsonl`

| Confidence Level | Count | Pattern |
|---|---|---|
| High | 800 | Multiple corroborating sources, clear data |
| Medium | 600 | Single source or minor ambiguity |
| Low | 400 | Conflicting sources, poor data quality |
| Refusal | 200 | Insufficient evidence to answer responsibly |

Each example includes evidence assessment in `<think>`, per-source relevance scoring, contradiction detection, and calibrated confidence statement with reasoning.

### Eval Suite — Held-Out Benchmark (500 examples, frozen)

**Output:** `finetune_data/v2/eval/benchmark.jsonl`

Generated once, never used in training. Same set across all iterations for fair comparison.

| Benchmark | Count | Tests |
|---|---|---|
| DocVQA-mini | 60 | Document question answering accuracy |
| TableBench | 50 | Table extraction & reasoning F1 |
| LayoutEval | 40 | Layout structure understanding |
| HalluBench | 50 | Hallucination detection rate |
| ToolEval | 50 | Tool selection & argument correctness |
| InsightEval | 50 | Insight precision and recall |
| SynthesisEval | 50 | Multi-source narrative coherence |
| ConversationEval | 50 | Multi-turn dialogue quality |
| ConfidenceEval | 50 | Calibration (Expected Calibration Error) |
| RegressionSuite | 50 | V1 baseline preservation |

---

## Stage 2: Enhanced V2 Pipeline

### Phase 1: Vision-Language Alignment (Existing — No Changes)

- **File:** `src/finetune/v2/train_phase1.py`
- Trains ProjectionMLP only (vision & text frozen)
- QA gates: `cosine_sim >= 0.60`, `caption_bleu >= 0.15`

### Phase 2: Document Intelligence SFT (Enhanced Data)

- **File:** `src/finetune/v2/train_phase2.py` (no code changes — uses new data)
- **Data:** 20K examples from Claude Code generation
- 4-stage curriculum: clean -> noisy -> complex -> adversarial
- LoRA: r=64, alpha=128, targets: q/k/v/o/gate/up/down_proj
- 8 epochs, LR 2e-5 cosine, warmup 10%, batch 4x8 (effective 32), bf16
- Projection MLP unfrozen (continues to adapt)
- **QA gates:** `docvqa_accuracy >= 0.75`, `table_f1 >= 0.80`, `layout_map >= 0.70`

### Phase 2.5: DPO Contrastive Preference (Enhanced Corruption)

- **File:** `src/finetune/v2/train_phase2_5_dpo.py` (no code changes — uses new data)
- **Data:** 5K preference pairs from Claude Code + `corrupt_extraction()`
- 3 epochs, LR 5e-6, beta 0.1, batch 2x16 (effective 32), bf16
- **QA gates:** `hallucination_rate <= 0.05`, `extraction_f1_improvement >= 5%`

### Phase 3: Tool-Calling SFT (Enhanced + Self-Verification)

- **File:** `src/finetune/v2/train_phase3.py` (no code changes — uses enriched data)
- **Data:** 8K examples (4K existing generator + 4K Claude Code enhanced)
- Projection frozen, LoRA only
- 5 epochs, LR 1e-5, batch 4x8, bf16
- **QA gates:** `tool_accuracy >= 0.85`, `arg_correctness >= 0.90`, `false_positive_rate <= 0.10`

### Phase 3.5: Insight Generation SFT (Expanded Categories)

- **File:** `src/finetune/v2/train_phase3_5_insights.py` (no code changes — uses expanded data)
- **Data:** 6K examples across 7 categories (5 existing + holistic synthesis + risk assessment)
- Projection frozen, LoRA only
- 4 epochs, LR 1e-5, batch 4x8, bf16
- **QA gates:** `insight_precision >= 0.80`, `insight_recall >= 0.60`

### Phase 3.7: Holistic Reasoning SFT (NEW)

- **File:** `src/finetune/v2/train_phase3_7_holistic.py` (NEW)
- **Data:** 8K examples across 4 reasoning modes
- Projection frozen, LoRA loaded from Phase 3.5 checkpoint
- LoRA: r=64, alpha=128 (same targets)
- 3 epochs, LR 8e-6 cosine, warmup 10%, batch 4x8 (effective 32), bf16
- **Max sequence length: 8192** (longer context for synthesis tasks)
- **QA gates:**
  - `synthesis_coherence >= 0.80` (Claude Code judge)
  - `intent_alignment >= 0.85` (Claude Code judge)
  - `depth_calibration >= 0.75` (Claude Code judge)
  - `domain_accuracy >= 0.80` (Claude Code judge)

### Phase 4: Merge & Promote (Updated)

- **File:** `src/finetune/v2/merge_promote.py` (updated merge sequence)
- Merge order: Phase 2 LoRA + Phase 2.5 DPO + Phase 3 LoRA + Phase 3.5 LoRA + Phase 3.7 LoRA -> base
- Quantize to GGUF Q4_K_M
- Full regression suite: >= 90% pass
- Output: `docwain:v2-base` (intermediate, pre-refinement)

---

## Stage 3: Post-Training Refinement Stack

Operates on **merged full weights** (not LoRA). A100-80GB handles Qwen3-14B full-precision fine-tuning with gradient checkpointing + bf16 (model ~28GB + AdamW optimizer states ~56GB, fits in 80GB with gradient checkpointing reducing activation memory).

### Round 1: Conversational Refinement DPO

- **File:** `src/finetune/v2/post_training/round1_conversational_dpo.py`
- **Data:** 3K preference pairs
- Full model fine-tuning (no LoRA)
- LR: 1e-6 (very conservative), beta: 0.05 (soft preference shaping)
- 2 epochs, batch 2x16 (effective 32), bf16
- **QA gate:** `conversation_quality >= 0.80` (Claude Code judge)

### Round 2: Confidence Calibration SFT

- **File:** `src/finetune/v2/post_training/round2_confidence_sft.py`
- **Data:** 2K calibration examples (high/medium/low/refusal)
- Full model fine-tuning
- LR: 1e-6, 2 epochs, batch 4x8, bf16
- **QA gate:** `ECE <= 0.10` (Claude Code judge)

### Round 3: Reasoning Distillation for Speed

- **File:** `src/finetune/v2/post_training/round3_distillation.py`
- **Process:**
  1. Run fully-trained model on 4K diverse queries with full `<think>` blocks
  2. Filter for high-quality outputs (grounding score >= 0.80)
  3. Claude Code compresses each `<think>` block (40-60% token reduction)
  4. SFT on compressed reasoning + original answer pairs
- LR: 5e-7 (extremely conservative), 1 epoch, batch 4x8, bf16
- **QA gates:**
  - No quality metric drops > 3% vs pre-distillation
  - Inference speed >= 25 tok/s on A100

### Final Promotion

- Re-quantize to GGUF Q4_K_M
- Full regression + all benchmark suite
- Register as `docwain:v2` on Ollama
- After 24h soak: promote to `docwain:latest`

---

## Evaluation Suite

### Judge: Claude Code (In-Process)

Claude Code reads model output + reference answer + rubric, scores 1-5 per dimension.
- No external API calls
- Deterministic scoring per rubric
- Results: `finetune_artifacts/v2/eval/{phase}_{timestamp}.jsonl`

### Scoring Rubrics

**Synthesis Coherence (1-5):**
- 5: Response tells a complete, logical story connecting all relevant evidence
- 4: Mostly coherent, minor gaps in narrative flow
- 3: Key points covered but reads as disconnected bullet points
- 2: Significant gaps, contradicts itself, or misses important connections
- 1: Incoherent or largely irrelevant

**Intent Alignment (1-5):**
- 5: Perfectly addresses what the user actually needs, including unstated needs
- 4: Addresses the explicit question well, minor missed implications
- 3: Answers the literal question but misses the underlying intent
- 2: Partially addresses the question, significant gaps
- 1: Misunderstands the question entirely

**Depth Calibration (1-5):**
- 5: Response length and detail perfectly match query complexity
- 4: Slightly over/under detailed but appropriate
- 3: Noticeably too verbose for simple questions or too brief for complex ones
- 2: Significantly miscalibrated depth
- 1: Completely wrong depth (essay for yes/no, one-liner for analysis request)

**Conversation Quality (1-5):**
- 5: Natural, professional, builds on context, handles ambiguity gracefully
- 4: Good flow, minor stiffness or unnecessary repetition
- 3: Functional but robotic, doesn't leverage conversation history well
- 2: Awkward, repetitive, or ignores prior context
- 1: Incoherent or completely disconnected from conversation

**Confidence Calibration (scored numerically):**
- ECE (Expected Calibration Error): Binned difference between stated confidence and actual accuracy
- Target: ECE <= 0.10

### Gate Summary

| Phase | Gate Metrics | Thresholds |
|---|---|---|
| Phase 1 | cosine_sim, caption_bleu | >= 0.60, >= 0.15 |
| Phase 2 | docvqa, table_f1, layout_map | >= 0.75, >= 0.80, >= 0.70 |
| Phase 2.5 | hallucination_rate, f1_improvement | <= 0.05, >= 5% |
| Phase 3 | tool_acc, arg_correct, fpr | >= 0.85, >= 0.90, <= 0.10 |
| Phase 3.5 | insight_precision, insight_recall | >= 0.80, >= 0.60 |
| Phase 3.7 | synthesis, intent, depth, domain | >= 0.80, >= 0.85, >= 0.75, >= 0.80 |
| Phase 4 | regression_pass_rate | >= 90% |
| Round 1 | conversation_quality | >= 0.80 |
| Round 2 | ECE | <= 0.10 |
| Round 3 | quality_drop, inference_speed | <= 3%, >= 25 tok/s |

---

## Iterative Improvement Loop (Post-Deployment)

After `docwain:v2` is deployed, the daily fine-tune loop targets continuous improvement:

```
Daily Loop:
1. Harvest: Collect learning signals from production
   - Low-confidence queries (model uncertainty)
   - Grounding failures (hallucination corrections)
   - User corrections (explicit feedback)
   - Tool-call failures (wrong tool/arguments)
2. Evaluate: Run benchmark suite, identify weakest dimension
3. Generate: Claude Code creates targeted data for weak spots
4. Train: LoRA on docwain:v2 for the weak dimension
5. Eval: Must pass full regression + improve weak metric
6. Promote: new model -> docwain:latest
7. Repeat daily
```

Data policy enforcement: `enforce_data_policy()` rejects any pair with `source` in `{document_content, raw_text, embedding_vector}` or `answer` > 2000 chars.

---

## File Structure

### New Files

| File | Purpose |
|---|---|
| `src/finetune/v2/train_phase3_7_holistic.py` | Holistic reasoning SFT training loop |
| `src/finetune/v2/post_training/__init__.py` | Post-training package |
| `src/finetune/v2/post_training/round1_conversational_dpo.py` | Conversational refinement DPO |
| `src/finetune/v2/post_training/round2_confidence_sft.py` | Confidence calibration SFT |
| `src/finetune/v2/post_training/round3_distillation.py` | Reasoning distillation for speed |
| `src/finetune/v2/data_generator/__init__.py` | Data generator package |
| `src/finetune/v2/data_generator/phase2_doc_intelligence.py` | Phase 2 dataset generation |
| `src/finetune/v2/data_generator/phase2_5_dpo_pairs.py` | DPO pair generation |
| `src/finetune/v2/data_generator/phase3_tool_traces.py` | Enhanced tool-call data |
| `src/finetune/v2/data_generator/phase3_5_insights.py` | 7-category insight data |
| `src/finetune/v2/data_generator/phase3_7_holistic.py` | Holistic reasoning data |
| `src/finetune/v2/data_generator/post_conversational.py` | Conversation DPO pairs |
| `src/finetune/v2/data_generator/post_confidence.py` | Calibration examples |
| `src/finetune/v2/data_generator/eval_suite.py` | Held-out eval generation |
| `src/finetune/v2/eval/__init__.py` | Eval package |
| `src/finetune/v2/eval/runner.py` | Benchmark runner |
| `src/finetune/v2/eval/rubrics.py` | Claude Code judge rubrics |
| `src/finetune/v2/eval/gate_checker.py` | Pass/fail gate logic |

### Modified Files

| File | Change |
|---|---|
| `src/finetune/v2/pipeline.py` | Add Phase 3.7, post-training orchestration, eval integration |
| `src/finetune/v2/merge_promote.py` | Merge 5 LoRA stages, re-quantize after post-training |

### Data Flow: Generation -> Training

Data generators write to `finetune_data/v2/`. Training phases read from these paths. The training code's `data_dir` config parameters must be updated to point to these locations (currently some phases expect data under `finetune_artifacts/` or `runs/` — the pipeline orchestrator will symlink or copy as needed):

| Phase | Generator Output | Training Input |
|---|---|---|
| Phase 2 | `finetune_data/v2/doc_intelligence/{table,layout,ocr,cross_ref}.jsonl` | `Phase2Config.data_dir` |
| Phase 2.5 | `finetune_data/v2/doc_intelligence/dpo_pairs.jsonl` | Copied to `{phase2_output}/dpo_data/dpo_pairs.jsonl` |
| Phase 3 | `finetune_data/v2/tool_calling/tool_calling_sft.jsonl` | `Phase3Config.data_dir` |
| Phase 3.5 | `finetune_data/v2/insights/insight_training.jsonl` | Copied to `{phase3_output}/insight_data/insight_training.jsonl` |
| Phase 3.7 | `finetune_data/v2/holistic/holistic_training.jsonl` | `HolisticConfig.data_dir` |
| Round 1 | `finetune_data/v2/post_training/conversational_dpo.jsonl` | Direct path |
| Round 2 | `finetune_data/v2/post_training/confidence_sft.jsonl` | Direct path |

### Data Directories

```
finetune_data/v2/
├── doc_intelligence/
│   ├── table.jsonl          (8K examples)
│   ├── layout.jsonl         (5K examples)
│   ├── ocr.jsonl            (4K examples)
│   ├── cross_ref.jsonl      (3K examples)
│   └── dpo_pairs.jsonl      (5K pairs)
├── tool_calling/
│   └── tool_calling_sft.jsonl  (8K examples)
├── insights/
│   └── insight_training.jsonl  (6K examples)
├── holistic/
│   └── holistic_training.jsonl (8K examples)
├── post_training/
│   ├── conversational_dpo.jsonl (3K pairs)
│   └── confidence_sft.jsonl    (2K examples)
└── eval/
    └── benchmark.jsonl         (500 examples, frozen)
```

---

## Estimated Timeline

| Stage | Phase | Est. Hours |
|---|---|---|
| Data Gen | Claude Code generates all datasets | ~2-3 |
| V2 | Phase 1 (if needed) | ~2 |
| V2 | Phase 2 (20K, 8 epochs) | ~8-10 |
| V2 | Phase 2.5 DPO (5K, 3 epochs) | ~3-4 |
| V2 | Phase 3 (8K, 5 epochs) | ~4-5 |
| V2 | Phase 3.5 (6K, 4 epochs) | ~4-5 |
| V2 | Phase 3.7 (8K, 3 epochs, 8192 seq) | ~5-6 |
| V2 | Phase 4 (merge + quantize + regression) | ~1 |
| Post | Round 1 DPO (3K, 2 epochs) | ~2-3 |
| Post | Round 2 SFT (2K, 2 epochs) | ~1-2 |
| Post | Round 3 Distill (4K, 1 epoch) | ~1 |
| **Total** | | **~33-42 hours** |

---

## Success Criteria

| Metric | Target | Baseline |
|---|---|---|
| Document extraction F1 | >= 0.90 | ~0.70 |
| Table extraction F1 | >= 0.85 | ~0.65 |
| Hallucination rate | <= 0.05 | ~0.15 |
| Tool-call accuracy | >= 0.85 | N/A |
| Insight precision | >= 0.80 | N/A |
| Synthesis coherence | >= 0.80 | N/A |
| Intent alignment | >= 0.85 | N/A |
| Conversation quality | >= 0.80 | N/A |
| Confidence ECE | <= 0.10 | N/A |
| Inference speed | >= 25 tok/s | ~15 tok/s |
| V1 regression pass rate | >= 90% | N/A |
